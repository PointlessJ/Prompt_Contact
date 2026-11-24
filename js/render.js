import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const BODY_COLORS = {
  left_foot: 0x2878ff, right_foot: 0x28c8ff, left_knee: 0x8250ff, right_knee: 0xc850ff,
  left_hand: 0xff8c00, right_hand: 0xffc800, left_elbow: 0xff5a5a, right_elbow: 0xff9696,
  pelvis: 0x3cc85a, torso: 0x78dc78,
};

// MuJoCo is z-up; three.js renders y-up. Everything lives under a z-up root group.
export class Renderer {
  constructor(canvas, sim, bodies) {
    this.sim = sim;
    const r = new THREE.WebGLRenderer({ canvas, antialias: true });
    r.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    r.shadowMap.enabled = true;
    this.renderer = r;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xe8ecf1);
    this.scene = scene;
    this.camera = new THREE.PerspectiveCamera(45, 1, 0.05, 100);
    this.camera.up.set(0, 0, 1);
    this.camera.position.set(0.6, -3.4, 1.9);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.target.set(0.8, 0.0, 0.55);
    this.controls.update();
    this.root = new THREE.Group();
    scene.add(this.root);
    const hemi = new THREE.HemisphereLight(0xffffff, 0x8899aa, 1.6);
    hemi.position.set(0, 0, 1);
    scene.add(hemi);
    const sun = new THREE.DirectionalLight(0xffffff, 1.6);
    sun.position.set(3, -4, 8);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    Object.assign(sun.shadow.camera, { left: -6, right: 6, top: 6, bottom: -6, near: 0.5, far: 25 });
    scene.add(sun, sun.target);
    this.sun = sun;
    this.buildFloor();
    this.buildGeoms();
    this.buildOverlays(bodies);
    this.follow = true;
  }

  buildFloor() {
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(60, 60), new THREE.MeshStandardMaterial({ color: 0xd9dde3, roughness: 0.95 }));
    floor.receiveShadow = true;
    this.root.add(floor);
    const grid = new THREE.GridHelper(60, 120, 0xb4bac4, 0xc8cdd5);
    grid.rotation.x = Math.PI / 2;
    grid.position.z = 0.001;
    this.root.add(grid);
  }

  buildGeoms() {
    const m = this.sim.model;
    this.geoms = [];
    const box = this.sim.objGeom;
    for (let g = 0; g < m.ngeom; g++) {
      const visual = m.geom_group[g] === 2;
      if (!visual && g !== box) continue;
      let mesh;
      if (g === box) {
        mesh = new THREE.Mesh(new THREE.BoxGeometry(2, 2, 2), new THREE.MeshStandardMaterial({ color: 0xc8874f, roughness: 0.8 }));
        mesh.userData.unitBox = true;
        this.boxMesh = mesh;
      } else if (m.geom_type[g] === 7) {
        const id = m.geom_dataid[g];
        const va = m.mesh_vertadr[id], vn = m.mesh_vertnum[id], fa = m.mesh_faceadr[id], fn = m.mesh_facenum[id];
        const geo = new THREE.BufferGeometry();
        geo.setAttribute('position', new THREE.BufferAttribute(Float32Array.from(m.mesh_vert.subarray(3 * va, 3 * (va + vn))), 3));
        geo.setIndex(Array.from(m.mesh_face.subarray(3 * fa, 3 * (fa + fn))));
        geo.computeVertexNormals();
        const mat = m.geom_matid[g];
        const rgba = mat >= 0 ? m.mat_rgba.subarray(4 * mat, 4 * mat + 4) : m.geom_rgba.subarray(4 * g, 4 * g + 4);
        const col = new THREE.Color(rgba[0], rgba[1], rgba[2]);
        mesh = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ color: col, roughness: 0.55, metalness: 0.1 }));
      } else {
        continue;
      }
      mesh.castShadow = true;
      mesh.matrixAutoUpdate = false;
      this.root.add(mesh);
      this.geoms.push([g, mesh]);
    }
  }

  buildOverlays(bodies) {
    this.bodies = bodies;
    this.targets = bodies.map((b) => {
      const s = new THREE.Mesh(new THREE.SphereGeometry(0.035, 16, 12), new THREE.MeshBasicMaterial({ color: BODY_COLORS[b] ?? 0xffffff }));
      s.visible = false;
      this.root.add(s);
      return s;
    });
    this.cloud = new THREE.Points(new THREE.BufferGeometry(), new THREE.PointsMaterial({ color: 0xff3cc8, size: 0.035 }));
    this.cloud.geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(64 * 3), 3));
    this.root.add(this.cloud);
    this.estimate = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(1, 1, 1)), new THREE.LineBasicMaterial({ color: 0xff3cc8 }));
    this.estimate.visible = false;
    this.root.add(this.estimate);
    this.target = new THREE.Mesh(new THREE.CylinderGeometry(0.2, 0.2, 0.01, 40), new THREE.MeshBasicMaterial({ color: 0x28c850, transparent: true, opacity: 0.5 }));
    this.target.rotation.x = Math.PI / 2;
    this.root.add(this.target);
    this.show = { targets: true, cloud: true, estimate: true };
  }

  resize() {
    const c = this.renderer.domElement;
    const w = c.clientWidth, h = c.clientHeight;
    if (c.width !== Math.round(w * this.renderer.getPixelRatio()) || c.height !== Math.round(h * this.renderer.getPixelRatio())) {
      this.renderer.setSize(w, h, false);
      this.camera.aspect = w / h;
      this.camera.updateProjectionMatrix();
    }
  }

  setTarget(xy) { this.target.position.set(xy[0], xy[1], 0.006); }

  sync(ctrl) {
    const m = this.sim.model, d = this.sim.data, xp = d.geom_xpos, xm = d.geom_xmat;
    const M = new THREE.Matrix4();
    for (const [g, mesh] of this.geoms) {
      const p = 3 * g, r = 9 * g;
      M.set(xm[r], xm[r + 1], xm[r + 2], xp[p], xm[r + 3], xm[r + 4], xm[r + 5], xp[p + 1], xm[r + 6], xm[r + 7], xm[r + 8], xp[p + 2], 0, 0, 0, 1);
      if (mesh.userData.unitBox) M.multiply(new THREE.Matrix4().makeScale(m.geom_size[p], m.geom_size[p + 1], m.geom_size[p + 2]));
      mesh.matrix.copy(M);
    }
    const root = this.sim.rootPos();
    const last = ctrl?.last;
    if (last) {
      this.targets.forEach((s, i) => {
        s.visible = this.show.targets && last.valid[i] > 0.5;
        if (s.visible) s.position.set(last.targets[3 * i], last.targets[3 * i + 1], last.targets[3 * i + 2]);
      });
      // cloud: body-yaw frame -> world
      const q = d.qpos, o = this.sim.rootQ;
      const w = q[o + 3], x = q[o + 4], y = q[o + 5], z = q[o + 6];
      const yaw = Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)), c = Math.cos(yaw), s = Math.sin(yaw);
      const pos = this.cloud.geometry.attributes.position.array;
      let n = 0;
      for (let k = 0; k < last.cloud.length / 4; k++) {
        if (last.cloud[4 * k + 3] < 0.5) continue;
        const lx = last.cloud[4 * k], ly = last.cloud[4 * k + 1], lz = last.cloud[4 * k + 2];
        pos[3 * n] = root[0] + c * lx - s * ly;
        pos[3 * n + 1] = root[1] + s * lx + c * ly;
        pos[3 * n + 2] = root[2] + lz;
        n++;
      }
      this.cloud.geometry.setDrawRange(0, n);
      this.cloud.geometry.attributes.position.needsUpdate = true;
      this.cloud.visible = this.show.cloud;
      const t = ctrl.state.task, T = ctrl.T;
      if (this.show.estimate && t[T.EST_SEEN] > 0.5) {
        const top = Math.max(t[T.EST_TOP], 0.05), half = t[T.EST_HALF] > 0 ? t[T.EST_HALF] : 0.17;
        this.estimate.scale.set(2 * half, 2 * half, top);
        this.estimate.position.set(t[T.EST_XY], t[T.EST_XY + 1], top / 2);
        this.estimate.visible = true;
      } else {
        this.estimate.visible = false;
      }
    }
    if (this.follow) {
      const tgt = this.controls.target, dx = root[0] - tgt.x, dy = root[1] - tgt.y;
      tgt.x += dx; tgt.y += dy;
      this.camera.position.x += dx; this.camera.position.y += dy;
    }
    this.sun.position.set(root[0] + 3, root[1] - 4, 8);
    this.sun.target.position.set(root[0], root[1], 0);
    this.controls.update();
    this.resize();
    this.renderer.render(this.scene, this.camera);
  }

  // Floor point under a screen position (for dragging and setting the target).
  floorPoint(ev) {
    const rect = this.renderer.domElement.getBoundingClientRect();
    const ndc = new THREE.Vector2(((ev.clientX - rect.left) / rect.width) * 2 - 1, -((ev.clientY - rect.top) / rect.height) * 2 + 1);
    const ray = new THREE.Raycaster();
    ray.setFromCamera(ndc, this.camera);
    const hitBox = ray.intersectObject(this.boxMesh, false).length > 0;
    const o = ray.ray.origin, dir = ray.ray.direction;
    if (dir.z >= -1e-6) return { hitBox, point: null };
    const t = -o.z / dir.z;
    return { hitBox, point: [o.x + t * dir.x, o.y + t * dir.y] };
  }
}
