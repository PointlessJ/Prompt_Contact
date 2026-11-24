import loadMujoco from 'https://cdn.jsdelivr.net/npm/@mujoco/mujoco@3.16.0/mujoco.js';
import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.wasm.min.mjs';
import { Sim } from './sim.js?v=10';
import { Controller } from './controller.js?v=10';

const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
const TEST = params.has('test');
const POLICIES = {
  pace_fix: { dir: 'models/pace_fix', label: 'Pace-controlled (30k, EMA 0.998)' },
  softprompt_v2: { dir: 'models/softprompt_v2', label: 'Soft-Prompt v2 (36k, EMA 0.9995)' },
  v16: { dir: 'models/v16', label: 'v16 release (38k, EMA 0.9995)' },
  release: { dir: 'models/release', label: 'v3 release' },
};
const DEFAULT_POLICY = 'pace_fix';

// Random boxes: all policies were trained on cubes scaled 0.5-1.8 (17-61 cm,
// 1-5 kg); Random draws cubes of 17-45 cm and 0.5-5 kg. The parcel preset draws
// each side separately (25-45 x 18-32 x 15-30 cm, long side along x or y), a
// zero-shot shape outside the cube training mix.
const PARCEL = { l: [25, 45], w: [18, 32], h: [15, 30] };
const CUBE_SCALE = [0.5, 1.32];
const MASS = [0.5, 5.0];
const uni = (a, b) => a + (b - a) * Math.random();
function drawParcel() {
  let l = uni(...PARCEL.l), w = uni(...PARCEL.w);
  if (Math.random() < 0.5) [l, w] = [w, l];
  return [l, w, uni(...PARCEL.h)].map(Math.round);
}
function drawBox() {
  const dims = Array(3).fill(Math.round(34 * uni(...CUBE_SCALE)));
  return { dims, mass: Math.round(20 * uni(...MASS)) / 20 };
}

function setProgress(text, frac) {
  $('load-text').textContent = text;
  if (frac != null) $('load-bar').style.width = `${Math.round(100 * frac)}%`;
}

function log(text) {
  console.log(text);
  if (TEST) fetch('__log', { method: 'POST', body: text }).catch(() => {});
}

async function fetchBytes(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  return new Uint8Array(await r.arrayBuffer());
}

function colormap(z, lo = 0.1, hi = 4.0) {
  const t = Math.min(Math.max((z - lo) / (hi - lo), 0), 1);
  const f = (k) => Math.min(Math.max(1.5 - Math.abs(4 * t - k), 0), 1) * 255;
  return [f(3), f(2), f(1)];
}

class App {
  async start() {
    setProgress('Loading the physics engine and the network runtime', 0.03);
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/';
    const mj = await loadMujoco();
    this.mj = mj;
    const files = await (await fetch('assets/scene/files.json')).json();
    mj.FS.mkdir('/scene');
    mj.FS.mkdir('/scene/assets');
    let done = 0;
    const queue = files.slice();
    const worker = async () => {
      while (queue.length) {
        const f = queue.shift();
        mj.FS.writeFile(`/scene/assets/${f}`, await fetchBytes(`assets/scene/assets/${f}`));
        done++;
        setProgress(`Loading robot meshes (${done}/${files.length})`, 0.05 + 0.35 * done / files.length);
      }
    };
    await Promise.all(Array.from({ length: 6 }, worker));
    mj.FS.writeFile('/scene/g1_box.xml', await fetchBytes('assets/scene/g1_box.xml'));

    this.ctrl = new Controller(ort);
    setProgress('Loading the sensing model', 0.42);
    await this.ctrl.loadSense('models/sense.onnx');
    this.policy = params.get('policy') in POLICIES ? params.get('policy') : DEFAULT_POLICY;
    await this.loadPolicy(this.policy);
    this.sim = new Sim(mj);
    this.sim.load('/scene/g1_box.xml', this.ctrl.meta.joint_names);
    this.defaultQ = this.ctrl.meta.default_joint_pos;

    if (!TEST) {
      const { Renderer } = await import('./render.js?v=10');
      this.view = new Renderer($('view'), this.sim, this.ctrl.meta.contact_bodies);
      this.bindUi();
    }
    this.boxStart = [1.6, 0.0, 0.0];
    this.target = [1.6, 2.0];
    this.speed = 1;
    this.paused = false;
    this.single = 0;
    this.actDelay = 0;
    this.qBuf = [];
    this.box = { dims: [34, 34, 34], mass: 1.25 };
    await this.reset();
    $('loading').hidden = true;
    if (TEST) return this.test();
    this.last = performance.now();
    this.acc = 0;
    requestAnimationFrame(() => this.frame());
  }

  async loadPolicy(name) {
    const p = POLICIES[name];
    setProgress(`Loading the policy: ${p.label}`, 0.45);
    await this.ctrl.loadPolicy(p.dir, (f) => setProgress(`Loading the policy: ${p.label} (${Math.round(100 * f)}%)`, 0.45 + 0.5 * f));
    this.policy = name;
  }

  // ---- episode ----

  async reset() {
    const s = this.sim;
    s.setBox(this.box.dims.map((c) => c / 200), this.box.mass);
    s.placeBox(...this.boxStart);
    s.resetRobot(this.defaultQ);
    this.qBuf = [];
    await this.ctrl.reset(s.read(), this.target);
    this.t = 0;
    this.overSince = null;
    this.fallen = false;
    this.view?.setTarget(this.target);
  }

  randomLayout() {
    const T = this.ctrl.T, u = (a, b) => a + (b - a) * Math.random();
    const r = u(...T.box_distance), a = u(-T.box_bearing, T.box_bearing);
    const bx = r * Math.cos(a), by = r * Math.sin(a);
    const td = u(...T.target_distance), ta = u(-Math.PI, Math.PI);
    this.boxStart = [bx, by, u(-Math.PI, Math.PI)];
    this.target = [bx + td * Math.cos(ta), by + td * Math.sin(ta)];
  }

  // The box sliders follow this.box. A box resting on the floor is put back on
  // the floor at its new height; a carried one keeps its pose.
  setBox(dims, mass, sync = true) {
    this.box = { dims: dims.slice(), mass };
    const half = dims.map((c) => c / 200), p = this.sim.boxPose();
    const resting = p.pos[2] < this.sim.half[2] + 0.03;
    this.sim.setBox(half, mass);
    if (resting) {
      const yaw = 2 * Math.atan2(p.quat[3], p.quat[0]);
      this.sim.placeBox(p.pos[0], p.pos[1], yaw);
    }
    if (sync && !TEST) {
      [['box-l', dims[0]], ['box-w', dims[1]], ['box-h', dims[2]], ['box-mass', mass]].forEach(([id, v]) => {
        $(id).value = v;
        $(`${id}-v`).textContent = v;
      });
    }
  }

  setTarget(xy) {
    this.target = xy;
    this.ctrl.setTarget(xy);
    this.view?.setTarget(xy);
  }

  async controlStep() {
    const s = this.sim;
    const pose = s.boxPose();
    const box = [...pose.pos, ...pose.quat, ...s.half, 1];
    const q = await this.ctrl.control(s.read(), box);
    this.qBuf.push(q);
    while (this.qBuf.length > this.actDelay + 1) this.qBuf.shift();
    s.step(this.qBuf[0]);
    this.t += 0.02;
    if (s.rootPos()[2] < 0.35) this.fallen = true;
  }

  status() {
    const s = this.sim, p = s.boxPose().pos;
    const dist = Math.hypot(p[0] - this.target[0], p[1] - this.target[1]);
    const atTarget = dist < 0.2;
    const placed = atTarget && p[2] < s.half[2] + 0.03;
    return { phase: this.ctrl.phase(this.fallen), dist, placed, atTarget, boxZ: p[2] };
  }

  // ---- loop ----

  async frame() {
    const now = performance.now();
    const dt = Math.min((now - this.last) / 1000, 0.1);
    this.last = now;
    if (this.pending) await this.pending();
    this.pending = null;
    let steps = 0;
    if (!this.paused) {
      this.acc += dt * this.speed;
      while (this.acc >= 0.02 && steps < 4) {
        await this.controlStep();
        this.acc -= 0.02;
        steps++;
      }
      if (steps === 4) this.acc = 0;
    } else if (this.single > 0) {
      this.single--;
      await this.controlStep();
      steps = 1;
    }
    const t1 = performance.now();
    if (steps) this.stepMs = 0.9 * (this.stepMs ?? 0) + 0.1 * (t1 - now) / steps;
    this.view.sync(this.ctrl);
    this.drawHud();
    requestAnimationFrame(() => this.frame());
  }

  drawHud() {
    const st = this.status();
    // Carried above the target for 2.5 s without being put down (large boxes).
    const over = st.atTarget && !st.placed && st.phase === 'carry';
    if (!over) this.overSince = null;
    else if (this.overSince == null) this.overSince = this.t;
    const held = over && this.t - this.overSince > 2.5;
    $('phase').textContent = st.phase + (st.placed ? ' · placed' : held ? ' · at target, not released' : '');
    $('phase').className = st.placed ? 'placed' : st.phase === 'fallen' ? 'fallen' : '';
    const cb = this.ctrl.meta.contact_bodies, last = this.ctrl.last;
    const hands = last ? ['left_hand', 'right_hand'].filter((b) => last.valid[cb.indexOf(b)] > 0.5).length : 0;
    const rt = this.stepMs ? 20 / this.stepMs : 0;
    $('stats').textContent = `t ${this.t.toFixed(this.paused ? 2 : 1)} s · box→target ${st.dist.toFixed(2)} m · box height ${st.boxZ.toFixed(2)} m · hand targets ${hands}/2 · ${this.stepMs ? this.stepMs.toFixed(1) : '–'} ms/step (${rt.toFixed(1)}× real time max)`;
    if (last) {
      const [R, C] = this.ctrl.meta.image_shape, cv = $('depth'), g = cv.getContext('2d');
      const im = g.createImageData(C, R);
      for (let i = 0; i < R * C; i++) {
        const [r, gg, b] = colormap(last.img[i]);
        im.data.set([r, gg, b, 255], 4 * i);
      }
      g.putImageData(im, 0, 0);
    }
  }

  // ---- ui ----

  bindUi() {
    const v = this.view, cv = $('view');
    const side = $('side'), toggle = $('toggle');
    const setOpen = (open) => { side.classList.toggle('closed', !open); toggle.textContent = open ? '✕ Hide controls' : '☰ Controls'; };
    toggle.onclick = () => setOpen(side.classList.contains('closed'));
    setOpen(!window.matchMedia('(max-width: 760px)').matches);
    const act = (fn) => { this.pending = fn; };
    $('btn-reset').onclick = () => act(() => this.reset());
    $('btn-random').onclick = () => act(() => {
      this.randomLayout();
      const b = drawBox();
      this.box = b;
      this.setBox(b.dims, b.mass);
      return this.reset();
    });
    $('btn-pause').onclick = () => { this.paused = !this.paused; $('btn-pause').textContent = this.paused ? 'Resume' : 'Pause'; };
    $('btn-step').onclick = () => { this.paused = true; this.single++; $('btn-pause').textContent = 'Resume'; };
    $('btn-push').onclick = () => this.sim.push(1.0);
    const sel = $('policy');
    for (const [k, p] of Object.entries(POLICIES)) sel.add(new Option(p.label, k, k === this.policy, k === this.policy));
    const syncPaceUi = () => {
      const T = this.ctrl.T, row = $('pace-row'), el = $('pace'), out = $('pace-v');
      const hasPace = T.PACE != null;
      row.hidden = !hasPace;
      if (!hasPace) return;
      const [lo, hi] = T.pace_range ?? [0.3, 0.6];
      el.min = lo; el.max = hi;
      el.value = this.ctrl.pace ?? T.pace_default ?? 0.45;
      out.textContent = Number(el.value).toFixed(2);
    };
    sel.onchange = () => act(async () => {
      $('loading').hidden = false;
      await this.loadPolicy(sel.value);
      syncPaceUi();
      $('loading').hidden = true;
      await this.reset();
    });
    const slider = (id, fn) => { const el = $(id), out = $(`${id}-v`); const up = () => { out.textContent = el.value; fn(Number(el.value)); }; el.oninput = up; up(); };
    slider('speed', (x) => { this.speed = x; });
    $('pace').oninput = () => {
      const x = Number($('pace').value);
      $('pace-v').textContent = x.toFixed(2);
      this.ctrl.setPace(x);
    };
    syncPaceUi();
    const dimsNow = () => ['box-l', 'box-w', 'box-h'].map((id) => Number($(id).value));
    for (const id of ['box-l', 'box-w', 'box-h']) slider(id, () => act(() => this.setBox(dimsNow(), this.box.mass, false)));
    slider('box-mass', (x) => act(() => this.setBox(this.box.dims, x, false)));
    $('box-cube').onclick = () => act(() => this.setBox([34, 34, 34], this.box.mass));
    $('box-parcel').onclick = () => act(() => this.setBox(drawParcel(), this.box.mass));
    $('box-flat').onclick = () => act(() => this.setBox([44, 44, 22], this.box.mass));
    slider('pd-scale', (x) => act(() => this.sim.setPdScale(x)));
    slider('act-delay', (x) => { this.actDelay = x; });
    slider('sense-delay', (x) => { this.ctrl.delay = x; });
    $('noise').onchange = (e) => { this.ctrl.noise = e.target.checked; };
    $('show-targets').onchange = (e) => { v.show.targets = e.target.checked; };
    $('show-cloud').onchange = (e) => { v.show.cloud = e.target.checked; };
    $('show-estimate').onchange = (e) => { v.show.estimate = e.target.checked; };
    $('follow').onchange = (e) => { v.follow = e.target.checked; };

    let drag = false, setMode = false;
    $('btn-target').onclick = () => { setMode = !setMode; $('btn-target').classList.toggle('on', setMode); };
    cv.addEventListener('pointerdown', (e) => {
      const hit = v.floorPoint(e);
      if (setMode && hit.point) {
        this.setTarget(hit.point);
        setMode = false;
        $('btn-target').classList.remove('on');
        return;
      }
      if (hit.hitBox) { drag = true; v.controls.enabled = false; cv.setPointerCapture(e.pointerId); }
    });
    cv.addEventListener('pointermove', (e) => {
      if (!drag) return;
      const hit = v.floorPoint(e);
      if (!hit.point) return;
      const p = this.sim.boxPose(), yaw = 2 * Math.atan2(p.quat[3], p.quat[0]);
      this.boxStart = [hit.point[0], hit.point[1], yaw];
      this.sim.placeBox(hit.point[0], hit.point[1], yaw);
    });
    const end = () => { drag = false; v.controls.enabled = true; };
    cv.addEventListener('pointerup', end);
    cv.addEventListener('pointercancel', end);
    cv.addEventListener('dblclick', (e) => { const hit = v.floorPoint(e); if (hit.point) this.setTarget(hit.point); });
    window.addEventListener('keydown', (e) => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
      const k = e.key.toLowerCase();
      if (k === 'r') $('btn-reset').click();
      else if (k === 'n') $('btn-random').click();
      else if (k === ' ') { e.preventDefault(); $('btn-pause').click(); }
      else if (k === '.') $('btn-step').click();
      else if (k === 'p') $('btn-push').click();
      else if (k === 't') $('btn-target').click();
    });
  }

  // ---- headless check (?test&episodes=N&box=random|cube) ----

  async test() {
    const n = Number(params.get('episodes') ?? 3), mix = params.get('box') !== 'cube';
    let ok = 0;
    log(`TEST start ${this.policy} ${n} episodes, boxes ${mix ? 'random cubes' : '34 cm cube'}`);
    for (let ep = 0; ep < n; ep++) {
      this.randomLayout();
      this.box = mix ? drawBox() : { dims: [34, 34, 34], mass: 1.25 };
      await this.reset();
      let lifted = false;
      const t0 = performance.now();
      const maxSteps = this.ctrl.T.PACE != null ? 1200 : 1000;
      for (let k = 0; k < maxSteps && !this.fallen; k++) {
        await this.controlStep();
        lifted ||= this.sim.boxPose().pos[2] > this.sim.half[2] + 0.1;
      }
      const ms = (performance.now() - t0) / maxSteps;
      const st = this.status();
      const succ = st.dist < 0.2 && !this.fallen;
      ok += succ;
      log(`TEST ep ${ep}: ${succ ? 'OK' : 'FAIL'} box ${this.box.dims.join('x')} cm ${this.box.mass} kg box-target ${st.dist.toFixed(2)} lifted ${lifted} fallen ${this.fallen} phase ${st.phase} | ${ms.toFixed(1)} ms/step`);
    }
    log(`TEST done ${this.policy}: success ${ok}/${n}`);
  }
}

new App().start().catch((e) => {
  console.error(e);
  $('error').hidden = false;
  $('error').textContent = `Could not start: ${e.message || e}`;
  log(`TEST ERROR ${e.stack || e}`);
});
