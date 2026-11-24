// MuJoCo world (WASM): the G1 with the training PD (position actuators), the floor
// and one free box. The same compiled scene the policy was trained in.

const ROOT_Z = 0.759;          // standing pelvis height at the default pose
const PARK = [50, 50, -5];     // the clips' climbing box is not part of the task

export class Sim {
  constructor(mj) {
    this.mj = mj;
    this.model = null;
    this.data = null;
  }

  load(path, jointNames) {
    const mj = this.mj;
    const m = mj.MjModel.from_xml_path(path);
    const d = new mj.MjData(m);
    this.model = m;
    this.data = d;
    this.decimation = 4;
    const jid = (n) => {
      const id = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT.value, n);
      if (id < 0) throw new Error(`joint ${n} missing`);
      return id;
    };
    this.jq = jointNames.map((n) => m.jnt_qposadr[jid(`robot/${n}`)]);
    this.jd = jointNames.map((n) => m.jnt_dofadr[jid(`robot/${n}`)]);
    const root = jid('robot/floating_base_joint');
    this.rootQ = m.jnt_qposadr[root];
    this.rootV = m.jnt_dofadr[root];
    const obj = jid('object0/floating_base_joint');
    this.objQ = m.jnt_qposadr[obj];
    this.objV = m.jnt_dofadr[obj];
    this.objBody = m.jnt_bodyid[obj];
    this.objGeom = -1;
    for (let g = 0; g < m.ngeom; g++) if (m.geom_bodyid[g] === this.objBody) this.objGeom = g;
    // actuator a drives joint actJoint[a] (an index into jointNames)
    const byName = new Map(jointNames.map((n, i) => [`robot/${n}`, i]));
    this.actJoint = [];
    for (let a = 0; a < m.nu; a++) {
      const j = m.actuator_trnid[2 * a];
      this.actJoint.push(byName.get(mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT.value, j)));
    }
    this.kp0 = Float64Array.from({ length: m.nu }, (_, a) => m.actuator_gainprm[10 * a]);
    this.b1 = Float64Array.from({ length: m.nu }, (_, a) => m.actuator_biasprm[10 * a + 1]);
    this.b2 = Float64Array.from({ length: m.nu }, (_, a) => m.actuator_biasprm[10 * a + 2]);
    this.half = [0.17, 0.17, 0.17];
    this.mass = m.body_mass[this.objBody];
  }

  park() {
    const d = this.data;
    for (let i = 0; i < d.mocap_pos.length / 3; i++) d.mocap_pos.set(PARK, 3 * i);
  }

  setPdScale(k) {
    const m = this.model;
    for (let a = 0; a < m.nu; a++) {
      m.actuator_gainprm[10 * a] = this.kp0[a] * k;
      m.actuator_biasprm[10 * a + 1] = this.b1[a] * k;
      m.actuator_biasprm[10 * a + 2] = this.b2[a] * k;
    }
  }

  setBox(half, mass) {
    const m = this.model, g = this.objGeom, b = this.objBody;
    this.half = half.slice();
    this.mass = mass;
    m.geom_size.set(half, 3 * g);
    m.geom_rbound[g] = Math.hypot(...half);
    m.geom_aabb.set([0, 0, 0, ...half], 6 * g);
    const [hx, hy, hz] = half;
    m.body_mass[b] = mass;
    m.body_inertia.set([mass / 3 * (hy * hy + hz * hz), mass / 3 * (hx * hx + hz * hz), mass / 3 * (hx * hx + hy * hy)], 3 * b);
    // Recompute the constants derived from masses (body_subtreemass, which the
    // subtree centre of mass divides by, invweight0, actuator_acc0). Without it a
    // new mass makes the box's dynamics inconsistent and the simulation blows up.
    // mj_setConst overwrites the MjData it gets (qpos0), so give it a scratch one.
    const scratch = new this.mj.MjData(m);
    this.mj.mj_setConst(m, scratch);
    scratch.delete();
  }

  placeBox(x, y, yaw, z = null) {
    const d = this.data;
    d.qpos.set([x, y, z ?? this.half[2] + 0.002, Math.cos(yaw / 2), 0, 0, Math.sin(yaw / 2)], this.objQ);
    d.qvel.fill(0, this.objV, this.objV + 6);
    this.mj.mj_forward(this.model, d);
  }

  boxPose() {
    const q = this.data.qpos, o = this.objQ;
    return { pos: [q[o], q[o + 1], q[o + 2]], quat: [q[o + 3], q[o + 4], q[o + 5], q[o + 6]] };
  }

  resetRobot(defaultQ) {
    const mj = this.mj, m = this.model, d = this.data;
    const box = Array.from(d.qpos.subarray(this.objQ, this.objQ + 7));
    mj.mj_resetData(m, d);
    this.park();
    d.qpos.set(box, this.objQ);
    d.qpos.set([0, 0, ROOT_Z, 1, 0, 0, 0], this.rootQ);
    this.jq.forEach((adr, i) => { d.qpos[adr] = defaultQ[i]; });
    for (let a = 0; a < m.nu; a++) d.ctrl[a] = defaultQ[this.actJoint[a]];
    mj.mj_forward(m, d);
  }

  // Robot-side measurements (odometry = truth).
  read() {
    const d = this.data, q = d.qpos, v = d.qvel;
    return {
      joint_pos: Float32Array.from(this.jq, (a) => q[a]),
      joint_vel: Float32Array.from(this.jd, (a) => v[a]),
      gyro: Float32Array.from(v.subarray(this.rootV + 3, this.rootV + 6)),
      root_quat: Float32Array.from(q.subarray(this.rootQ + 3, this.rootQ + 7)),
      root_pos: Float32Array.from(q.subarray(this.rootQ, this.rootQ + 3)),
    };
  }

  step(qDes) {
    const m = this.model, d = this.data;
    for (let a = 0; a < m.nu; a++) d.ctrl[a] = qDes[this.actJoint[a]];
    for (let i = 0; i < this.decimation; i++) this.mj.mj_step(m, d);
  }

  push(speed) {
    const phi = Math.random() * 2 * Math.PI, v = this.data.qvel;
    v[this.rootV] += speed * Math.cos(phi);
    v[this.rootV + 1] += speed * Math.sin(phi);
  }

  rootPos() {
    const q = this.data.qpos, o = this.rootQ;
    return [q[o], q[o + 1], q[o + 2]];
  }
}
