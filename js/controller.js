// The deploy controller in the browser: three ONNX graphs exported from the
// training code (export_web.py).
//   sense.onnx  training-simulator sensing: 32x18 head depth image + 8x8 box cloud
//   init.onnx   task start (believed root pose, placement target) -> task vector
//   step.onnx   one 20 ms control step; the controller state goes in and out
// Sensor noise (the training model) is added here so the graphs stay deterministic.

export class Controller {
  constructor(ort) {
    this.ort = ort;
    this.sense = null;
    this.delay = 1;          // depth / cloud delay (control steps)
    this.noise = true;
    this.pace = null;
    this.buffer = [];
  }

  async loadSense(url) {
    this.sense = await this.ort.InferenceSession.create(url, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
  }

  async loadPolicy(dir, onProgress) {
    const meta = await (await fetch(`${dir}/meta.json`)).json();
    const fetchBytes = async (url) => {
      const r = await fetch(url);
      if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
      const total = Number(r.headers.get('content-length')) || 0;
      const reader = r.body.getReader();
      const chunks = [];
      let got = 0;
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        got += value.length;
        if (onProgress && total) onProgress(Math.min(1, got / total));
      }
      const out = new Uint8Array(got);
      let o = 0;
      for (const c of chunks) { out.set(c, o); o += c.length; }
      return out;
    };
    const opts = { executionProviders: ['wasm'], graphOptimizationLevel: 'all' };
    this.init = await this.ort.InferenceSession.create(await fetchBytes(`${dir}/init.onnx`), opts);
    this.step = await this.ort.InferenceSession.create(await fetchBytes(`${dir}/step.onnx`), opts);
    this.meta = meta;
    this.T = meta.task;
    this.pace = meta.task.pace_default ?? null;
  }

  tensor(arr, dims) { return new this.ort.Tensor('float32', arr instanceof Float32Array ? arr : Float32Array.from(arr), dims); }

  async reset(s, target) {
    const out = await this.init.run({
      root_pos: this.tensor(s.root_pos, [1, 3]),
      root_quat: this.tensor(s.root_quat, [1, 4]),
      target_xy: this.tensor(target, [1, 2]),
    });
    const m = this.meta, [h, o] = m.obs_history.slice(1);
    const task = Float32Array.from(out.task.data);
    if (this.T.PACE != null && this.pace != null) task[this.T.PACE] = this.pace;
    this.state = {
      obs_history: new Float32Array(h * o),
      prev_action: new Float32Array(m.num_actions),
      latch_target: new Float32Array(27),
      latch_valid: new Float32Array(9),
      latch_age: new Float32Array(9),
      task,
      fresh: Float32Array.of(1),
    };
    this.buffer = [];
    this.last = null;
  }

  setTarget(xy) {
    if (!this.state) return;
    this.state.task[this.T.TARGET] = xy[0];
    this.state.task[this.T.TARGET + 1] = xy[1];
  }

  setPace(v) {
    this.pace = v;
    if (this.state && this.T?.PACE != null) this.state.task[this.T.PACE] = v;
  }

  // ---- noise (mirrors camera.depth_noise and the ObjectCloud noise) ----

  depthNoise(z) {
    const n = this.meta.noise, [R, C] = this.meta.image_shape, maxD = this.meta.camera.max_depth;
    const edge = new Uint8Array(R * C);
    for (let r = 0; r < R; r++) for (let c = 0; c < C; c++) {
      const i = r * C + c;
      const dx = c > 0 ? Math.abs(z[i] - z[i - 1]) : 0;
      const dy = r > 0 ? Math.abs(z[i] - z[i - C]) : 0;
      edge[i] = Math.max(dx, dy) > n.edge_threshold ? 1 : 0;
    }
    const out = new Float32Array(R * C);
    for (let r = 0; r < R; r++) for (let c = 0; c < C; c++) {
      const i = r * C + c;
      // roll by -1 along columns and rows (wrapping), as torch.roll(edge, -1, dim)
      const e = edge[i] || edge[r * C + ((c + 1) % C)] || edge[((r + 1) % R) * C + c];
      let v = z[i] + gauss() * (n.std + n.rel * z[i]);
      const hole = Math.random() < n.dropout || (e && Math.random() < n.edge_dropout);
      if (hole) v = maxD;
      if (n.quantization > 0) v = Math.round(v / n.quantization) * n.quantization;
      out[i] = Math.min(Math.max(v, 0), maxD);
    }
    return out;
  }

  cloudNoise(pts, valid, cam) {
    const n = this.meta.cloud_noise, K = valid.length, out = new Float32Array(4 * K);
    const frameDrop = Math.random() < n.frame_dropout;
    for (let k = 0; k < K; k++) {
      let ok = valid[k] > 0.5 && Math.random() >= n.dropout && !frameDrop;
      const rx = pts[3 * k] - cam[0], ry = pts[3 * k + 1] - cam[1], rz = pts[3 * k + 2] - cam[2];
      const t = Math.max(Math.hypot(rx, ry, rz), 1e-6);
      const t2 = t + gauss() * (n.std + n.rel * t);
      if (ok) {
        out[4 * k] = cam[0] + rx / t * t2;
        out[4 * k + 1] = cam[1] + ry / t * t2;
        out[4 * k + 2] = cam[2] + rz / t * t2;
        out[4 * k + 3] = 1;
      }
    }
    return out;
  }

  // ---- one control step ----

  async control(s, box) {
    const t = (a, d) => this.tensor(a, d);
    const sense = await this.sense.run({
      root_pos: t(s.root_pos, [1, 3]), root_quat: t(s.root_quat, [1, 4]),
      joint_pos: t(s.joint_pos, [1, s.joint_pos.length]), box: t(box, [1, 11]),
    });
    const depth = sense.depth.data, pts = sense.cloud_points.data, valid = sense.cloud_valid.data, cam = sense.camera_local.data;
    let img, cloud;
    if (this.noise) {
      img = this.depthNoise(depth);
      cloud = this.cloudNoise(pts, valid, cam);
    } else {
      img = Float32Array.from(depth);
      cloud = new Float32Array(4 * valid.length);
      for (let k = 0; k < valid.length; k++) {
        if (valid[k] > 0.5) cloud.set([pts[3 * k], pts[3 * k + 1], pts[3 * k + 2], 1], 4 * k);
      }
    }
    this.buffer.push({ img, cloud });
    while (this.buffer.length > 8) this.buffer.shift();
    const f = this.buffer[Math.max(0, this.buffer.length - 1 - this.delay)];
    const m = this.meta, st = this.state, [h, o] = m.obs_history.slice(1), J = m.num_actions;
    const out = await this.step.run({
      joint_pos: t(s.joint_pos, [1, J]), joint_vel: t(s.joint_vel, [1, J]), gyro: t(s.gyro, [1, 3]),
      root_quat: t(s.root_quat, [1, 4]), root_pos: t(s.root_pos, [1, 3]),
      depth: t(f.img, [1, f.img.length]), cloud: t(f.cloud, [1, f.cloud.length]),
      obs_history: t(st.obs_history, [1, h, o]), prev_action: t(st.prev_action, [1, J]),
      latch_target: t(st.latch_target, [1, 9, 3]), latch_valid: t(st.latch_valid, [1, 9]),
      latch_age: t(st.latch_age, [1, 9]), task: t(st.task, [1, m.task.dim]), fresh: t(st.fresh, [1]),
    });
    this.state = {
      obs_history: Float32Array.from(out.obs_history_out.data),
      prev_action: Float32Array.from(out.prev_action_out.data),
      latch_target: Float32Array.from(out.latch_target_out.data),
      latch_valid: Float32Array.from(out.latch_valid_out.data),
      latch_age: Float32Array.from(out.latch_age_out.data),
      task: Float32Array.from(out.task_out.data),
      fresh: Float32Array.of(0),
    };
    this.last = {
      img: f.img, cloud: f.cloud,
      targets: Float32Array.from(out.shown_target.data),
      valid: Float32Array.from(out.shown_valid.data),
    };
    return Float64Array.from(out.q_des.data);
  }

  phase(fallen) {
    const t = this.state?.task, T = this.T;
    if (!t) return '';
    if (fallen) return 'fallen';
    if (t[T.RELEASE_ON] > 0.5) return 'release';
    if (t[T.CARRY_ON] > 0.5) return 'carry';
    if (t[T.EST_SEEN] < 0.5) return 'search';
    return 'approach / grip';
  }
}

function gauss() {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
