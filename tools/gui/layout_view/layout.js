// synthrig layout viewer: runs entirely in the browser, so every slider move updates the
// 3D view, the camera views and the profile coverage at once (no server round trip).
// Python (tools/gui/layout_data.py) sends the shapes and the rig once; edits go back to
// Python only when a slider is released, to be written into the YAML files.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import * as G from "./geom.js";

const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
const COLOURS = { outside: ["#1f77b4", "#6baed6"], inside: ["#e6550d", "#fdae6b"], single: ["#a78bfa", "#c4b5fd", "#94a3b8"] };
const MM = 1000;

// ------------------------------------------------------------------ Streamlit bridge
const inStreamlit = window.parent !== window;
const bridge = {
  send(type, data = {}) { if (inStreamlit) window.parent.postMessage({ isStreamlitMessage: true, type, ...data }, "*"); },
  ready() { this.send("streamlit:componentReady", { apiVersion: 1 }); },
  value(v) { this.send("streamlit:setComponentValue", { value: v, dataType: "json" }); },
  height(h) { this.send("streamlit:setFrameHeight", { height: h }); },
};
function fitHeight() {
  let h = 900;
  try {
    const top = window.frameElement.getBoundingClientRect().top;
    h = Math.max(560, Math.floor(window.parent.innerHeight - top - 6));
  } catch (e) { /* not same-origin: fixed height */ }
  bridge.height(h);
}

// ------------------------------------------------------------------ state
let world = null, S = null, rev = null, geomKey = null, seq = 0;
const SID = Math.random().toString(36).slice(2, 10); // this page load: Python skips values it has handled
const view = { frustums: true, fans: true, labels: true, strip: true, stripH: 190, inset: true };
let results = null;

function buildState(st) {
  const cams = new Map(), lasers = new Map();
  for (const c of st.cameras) {
    const cam = {
      name: c.name, W: c.width, H: c.height, fx: c.fx, fy: c.fy, cx: c.cx, cy: c.cy, D: c.distortion,
      pxum: c.pixel_size_um, pos: c.position.slice(), aim: c.look_at ? c.look_at.slice() : null,
      Rfixed: c.look_at ? null : G.rowsToCols(c.R), upInFile: c.up_in_file, dirty: new Set(), colour: "#94a3b8",
    };
    cam.roll = cam.aim ? G.rollOf(cam.pos, cam.aim, c.up) : 0;
    updateR(cam);
    cams.set(c.name, cam);
  }
  for (const l of st.lasers) {
    const las = {
      name: l.name, pos: l.position.slice(), aim: l.look_at ? l.look_at.slice() : null, Rfixed: l.look_at ? null : G.rowsToCols(l.R),
      upInFile: l.up_in_file, pattern: structuredClone(l.pattern), tangents: l.tangents, rgb: l.rgb, power: l.power_w,
      enabled: l.enabled, sigma: l.dot_sigma_mrad, dirty: new Set(),
    };
    las.roll = las.aim ? G.rollOf(las.pos, las.aim, l.up) : 0;
    las.colour = "#" + new THREE.Color(...l.rgb.map((v) => Math.min(1, v))).getHexString();
    updateR(las);
    lasers.set(l.name, las);
  }
  const used = new Set();
  const pairs = st.pairs.filter((p) => cams.has(p.left) && cams.has(p.right)).map((p) => {
    used.add(p.left); used.add(p.right);
    return { ...p, cams: [cams.get(p.left), cams.get(p.right)] };
  });
  const singles = [...cams.values()].filter((c) => !used.has(c.name));
  const scene = { wheel_profile: st.scene.wheel_profile, dirty: new Set(), profiles: st.wheel_profiles };
  for (const k of ["wheel_x_m", "wheel_rot_deg", "lateral_shift_m"]) scene[k] = structuredClone(st.scene[k]);
  return { cams, lasers, pairs, singles, scene, lights: st.lights, rigName: st.rig_name, selCam: [...cams.keys()][0] };
}

function updateR(dev) {
  dev.R = dev.aim ? G.lookAtR(dev.pos, dev.aim, G.upForRoll(dev.pos, dev.aim, dev.roll)) : dev.Rfixed;
}
const hfov = (cam) => G.deg(2 * Math.atan(cam.W / 2 / cam.fx));
function setHfov(cam, h) { cam.fx = cam.fy = cam.W / 2 / Math.tan(G.rad(h) / 2); cam.dirty.add("hfov"); }

function pairSide(pair) {
  const y = (pair.cams[0].pos[1] + pair.cams[1].pos[1]) / 2;
  return y < world.g.rail_centre_y ? "outside" : "inside";
}
function assignColours() {
  for (const p of S.pairs) { p.side = pairSide(p); p.cams.forEach((c, i) => (c.colour = COLOURS[p.side][i])); }
  S.singles.forEach((c, i) => (c.colour = COLOURS.single[i % 3]));
}
function wheelValue(k) { const f = S.scene[k]; return f.range ? (f.range[0] + f.range[1]) / 2 : f.value; }

// ------------------------------------------------------------------ compute (pure geometry)
function recompute() {
  world.setWheel(wheelValue("wheel_x_m"), wheelValue("lateral_shift_m"));
  assignColours();
  const cams = [...S.cams.values()];
  const hits = {};
  for (const las of S.lasers.values()) {
    if (!las.enabled) continue;
    const n = las.tangents.length, pts = new Float32Array(n * 3), cls = new Uint8Array(n), seen = new Uint8Array(n);
    let m = 0;
    for (const [tx, ty] of las.tangents) {
      const d = G.applyR(las.R, G.unit([tx, ty, 1]));
      const h = world.cast(las.pos, d, 8);
      if (!h) continue;
      const X = G.add(las.pos, G.scale(d, h.t - 0.0008)); // a hair in front of the surface
      pts.set(X, 3 * m); cls[m] = h.cls;
      cams.forEach((c, ci) => { if (G.sees(world, c, X)) seen[m] |= 1 << ci; });
      m++;
    }
    hits[las.name] = { pts, cls, seen, n: m };
  }
  const stats = {};
  cams.forEach((c, ci) => {
    stats[c.name] = {};
    for (const [ln, h] of Object.entries(hits)) {
      let tot = 0, ok = 0;
      for (let i = 0; i < h.n; i++) if (h.cls[i] === G.WHEEL) { tot++; if (h.seen[i] & (1 << ci)) ok++; }
      stats[c.name][ln] = tot ? ok / tot : null;
    }
  });
  const ref = S.pairs.find((p) => p.side === "outside") || S.pairs[0];
  const aim = ref ? pairAim(ref) : (cams.find((c) => c.aim) || {}).aim;
  const phi = aim ? G.sectionAngle(world, aim) : 30;
  const cov = G.profileCoverage(world, cams, phi);
  results = { hits, stats, cov, refPair: ref };
}

// ------------------------------------------------------------------ three.js scene
const canvas = $("gl"), main = $("main");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.05;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.setScissorTest(true);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0e1117);
scene.environment = new THREE.PMREMGenerator(renderer).fromScene(new RoomEnvironment(), 0.04).texture;
scene.add(new THREE.HemisphereLight(0xdfe8ff, 0x3a3020, 0.55));
const sun = new THREE.DirectionalLight(0xffffff, 1.6);
sun.position.set(1.2, -2.0, 3.0);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
Object.assign(sun.shadow.camera, { left: -1.4, right: 1.4, top: 1.4, bottom: -1.4, near: 0.1, far: 8 });
sun.shadow.bias = -0.0004;
scene.add(sun, sun.target);

const cam3 = new THREE.PerspectiveCamera(38, 1, 0.005, 60);
cam3.up.set(0, 0, 1);
cam3.layers.enable(1);
const orbit = new OrbitControls(cam3, canvas);
orbit.enableDamping = false;
orbit.addEventListener("change", () => schedule(false));

const GIZMO = 1; // layer of markers, frustums, fans: shown in the 3D view, not in camera views
const onGizmo = (o) => { o.layers.set(GIZMO); o.traverse((c) => c.layers.set(GIZMO)); return o; };
const mat = {
  tread: new THREE.MeshStandardMaterial({ color: 0xc4c8cc, metalness: 0.9, roughness: 0.28, side: THREE.DoubleSide }),
  wheel: new THREE.MeshStandardMaterial({ color: 0x8c9096, metalness: 0.8, roughness: 0.42, side: THREE.DoubleSide }),
  plate: new THREE.MeshStandardMaterial({ color: 0x5c6067, metalness: 0.6, roughness: 0.6, side: THREE.DoubleSide }),
  rail: new THREE.MeshStandardMaterial({ color: 0x8a8b8c, metalness: 0.75, roughness: 0.38 }),
  tie: new THREE.MeshStandardMaterial({ color: 0x5b4636, roughness: 0.92 }),
  tplate: new THREE.MeshStandardMaterial({ color: 0x3e4044, metalness: 0.5, roughness: 0.6 }),
  rod: new THREE.MeshStandardMaterial({ color: 0x4b5563, metalness: 0.6, roughness: 0.4 }),
};
const WHEEL_MAT = { wheel_flange: mat.tread, wheel_tread: mat.tread, wheel_rim_face: mat.wheel, wheel_plate: mat.plate, wheel_hub: mat.plate };

const wheelGroup = new THREE.Group(), trackGroup = new THREE.Group(), gizmos = new THREE.Group(), hitGroup = new THREE.Group();
const strobes = new THREE.Group(); // the rig's lights (layer 0, so they light the camera views too)
scene.add(wheelGroup, trackGroup, gizmos, hitGroup, strobes);
for (const m of [mat.tread, mat.wheel, mat.plate, mat.rail]) m.envMapIntensity = 0.55;

function ballastTexture() {
  const c = document.createElement("canvas"); c.width = c.height = 256;
  const g = c.getContext("2d");
  g.fillStyle = "#5d5850"; g.fillRect(0, 0, 256, 256);
  let seed = 7; const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
  for (let i = 0; i < 2600; i++) {
    const v = 60 + rnd() * 90 | 0;
    g.fillStyle = `rgb(${v + 8},${v + 4},${v})`;
    g.beginPath(); g.ellipse(rnd() * 256, rnd() * 256, 1.5 + rnd() * 4, 1 + rnd() * 3, rnd() * 3, 0, 7); g.fill();
  }
  const t = new THREE.CanvasTexture(c); t.wrapS = t.wrapT = THREE.RepeatWrapping; t.repeat.set(24, 18);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

function buildStatic(geom) {
  for (const g of [wheelGroup, trackGroup]) { g.traverse((o) => o.geometry && o.geometry.dispose()); g.clear(); }
  for (const [cls, poly] of Object.entries(geom.wheel.sections)) {
    const pts = poly.map(([a, r]) => new THREE.Vector2(r, -a));
    const m = new THREE.Mesh(new THREE.LatheGeometry(pts, 180), WHEEL_MAT[cls] || mat.wheel);
    m.castShadow = m.receiveShadow = true;
    wheelGroup.add(m);
  }
  // a few spokes-like marks on the plate so the wheel rotation is visible
  for (let k = 0; k < 6; k++) {
    const b = new THREE.Mesh(new THREE.BoxGeometry(0.012, 0.004, 0.18), mat.wheel);
    const th = (k * Math.PI) / 3;
    b.position.set(0.27 * Math.sin(th), -0.085, 0.27 * Math.cos(th)); b.rotation.y = th;
    wheelGroup.add(b);
  }
  const H = geom.half_length;
  const M = new THREE.Matrix4().set(0, 0, 1, -H, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1);
  for (const r of geom.rails) {
    const sh = new THREE.Shape(r.poly.map(([y, z]) => new THREE.Vector2(y, z)));
    const g = new THREE.ExtrudeGeometry(sh, { depth: 2 * H, bevelEnabled: false, steps: 1, curveSegments: 1 });
    g.applyMatrix4(M);
    const m = new THREE.Mesh(g, mat.rail); m.castShadow = m.receiveShadow = true;
    trackGroup.add(m);
  }
  for (const b of world.boxes) {
    const size = G.sub(b.max, b.min);
    const m = new THREE.Mesh(new THREE.BoxGeometry(...size), b.cls === G.TIE ? mat.tie : mat.tplate);
    m.position.set(...G.scale(G.add(b.min, b.max), 0.5)); m.castShadow = m.receiveShadow = true;
    trackGroup.add(m);
  }
  const bal = new THREE.Mesh(new THREE.PlaneGeometry(2 * H + 2, 8),
    new THREE.MeshStandardMaterial({ map: ballastTexture(), roughness: 1 }));
  bal.position.z = geom.ballast_z; bal.receiveShadow = true;
  trackGroup.add(bal);
  trackGroup.add(onGizmo(new THREE.AxesHelper(0.15)));
}

// ---- gizmos (rebuilt when the rig is (re)loaded, updated in place afterwards)
const G3 = { cams: new Map(), lasers: new Map(), rods: [], aims: [] };
function lineMat(colour, opacity = 0.9) { return new THREE.LineBasicMaterial({ color: colour, transparent: true, opacity }); }
function faceMat(colour, opacity) {
  return new THREE.MeshBasicMaterial({ color: colour, transparent: true, opacity, side: THREE.DoubleSide, depthWrite: false });
}
function dynLines(n, colour, opacity) {
  const g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(n * 3), 3));
  return new THREE.LineSegments(g, lineMat(colour, opacity));
}
function dynMesh(n, colour, opacity) {
  const g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(n * 3), 3));
  return new THREE.Mesh(g, faceMat(colour, opacity));
}
function setPts(obj, pts) {
  const a = obj.geometry.attributes.position;
  pts.forEach((p, i) => a.setXYZ(i, p[0], p[1], p[2]));
  a.needsUpdate = true; obj.geometry.computeBoundingSphere();
}
function deviceBody(colour, laser) {
  const g = new THREE.Group();
  const m = new THREE.MeshStandardMaterial({ color: colour, metalness: 0.3, roughness: 0.5, emissive: colour, emissiveIntensity: laser ? 0.5 : 0.15 });
  const box = new THREE.Mesh(new THREE.BoxGeometry(laser ? 0.03 : 0.036, laser ? 0.03 : 0.036, laser ? 0.06 : 0.05), m);
  box.position.z = -0.028;
  const lens = new THREE.Mesh(new THREE.CylinderGeometry(0.011, 0.011, 0.02, 20), new THREE.MeshStandardMaterial({ color: 0x111111, roughness: 0.3 }));
  lens.rotation.x = Math.PI / 2; lens.position.z = 0.006;
  g.add(box, lens); g.matrixAutoUpdate = false;
  return g;
}

function buildGizmos() {
  gizmos.traverse((o) => o.geometry && o.geometry.dispose()); gizmos.clear();
  G3.cams.clear(); G3.lasers.clear(); G3.rods = []; G3.aims = [];
  for (const c of S.cams.values()) {
    const o = { body: deviceBody(c.colour), lines: dynLines(16, c.colour, 0.85), faces: dynMesh(12, c.colour, 0.07) };
    gizmos.add(o.body, o.lines, o.faces); G3.cams.set(c.name, o);
  }
  for (const l of S.lasers.values()) {
    const o = { body: deviceBody(l.colour, true), lines: dynLines(16, l.colour, 0.7), faces: dynMesh(12, l.colour, 0.06) };
    gizmos.add(o.body, o.lines, o.faces); G3.lasers.set(l.name, o);
    const pg = new THREE.BufferGeometry();
    const n = l.tangents.length + 4096;
    pg.setAttribute("position", new THREE.BufferAttribute(new Float32Array(n * 3), 3));
    pg.setAttribute("color", new THREE.BufferAttribute(new Float32Array(n * 3), 3));
    o.points = new THREE.Points(pg, new THREE.PointsMaterial({ size: 4, sizeAttenuation: false, vertexColors: true }));
    o.points.frustumCulled = false;
    hitGroup.add(o.points);
  }
  const rodGeo = new THREE.CylinderGeometry(0.008, 0.008, 1, 14); rodGeo.rotateZ(Math.PI / 2);
  for (const p of S.pairs) {
    const rod = new THREE.Mesh(rodGeo, mat.rod); gizmos.add(rod); G3.rods.push([p, rod]);
    const aim = new THREE.Mesh(new THREE.SphereGeometry(0.004, 12, 8), new THREE.MeshBasicMaterial({ color: 0xffffff }));
    gizmos.add(aim); G3.aims.push([p, aim]);
  }
  strobes.clear();
  for (const li of S.lights) {
    const R = G.rowsToCols(li.R);
    const sp = new THREE.SpotLight(0xffffff, 0.5, 0, 1.0, 0.7, 2);
    sp.position.set(...li.position);
    sp.target.position.set(...G.add(li.position, R[2]));
    strobes.add(sp, sp.target);
    const q = new THREE.Mesh(new THREE.PlaneGeometry(li.size_m, li.size_m), faceMat(0xffe08a, 0.35));
    q.matrixAutoUpdate = false;
    q.matrix.makeBasis(new THREE.Vector3(...R[0]), new THREE.Vector3(...R[1]), new THREE.Vector3(...R[2])).setPosition(...li.position);
    gizmos.add(q);
  }
  onGizmo(gizmos);
}

function frustumPts(dev, corners, depth) {
  const c = dev.pos;
  const P = corners.map(([x, y]) => G.add(c, G.applyR(dev.R, [x * depth, y * depth, depth])));
  const lines = [], faces = [];
  P.forEach((p, i) => { const q = P[(i + 1) % P.length]; lines.push(c, p, p, q); faces.push(c, p, q); });
  return { lines, faces };
}

function updateGizmos() {
  for (const c of S.cams.values()) {
    const o = G3.cams.get(c.name);
    o.body.matrix.makeBasis(new THREE.Vector3(...c.R[0]), new THREE.Vector3(...c.R[1]), new THREE.Vector3(...c.R[2])).setPosition(...c.pos);
    o.body.children[0].material.color.set(c.colour); o.body.children[0].material.emissive.set(c.colour);
    o.lines.material.color.set(c.colour); o.faces.material.color.set(c.colour);
    const n = (u, v) => [(u - c.cx) / c.fx, (v - c.cy) / c.fy];
    const corners = [n(-0.5, -0.5), n(c.W - 0.5, -0.5), n(c.W - 0.5, c.H - 0.5), n(-0.5, c.H - 0.5)];
    const depth = c.aim ? G.norm(G.sub(c.aim, c.pos)) * 1.1 : 0.3;
    const f = frustumPts(c, corners, depth);
    setPts(o.lines, f.lines); setPts(o.faces, f.faces);
    o.lines.visible = o.faces.visible = view.frustums;
  }
  for (const l of S.lasers.values()) {
    const o = G3.lasers.get(l.name);
    o.body.matrix.makeBasis(new THREE.Vector3(...l.R[0]), new THREE.Vector3(...l.R[1]), new THREE.Vector3(...l.R[2])).setPosition(...l.pos);
    o.body.visible = true;
    const t = l.tangents;
    let lo = [Infinity, Infinity], hi = [-Infinity, -Infinity];
    for (const [x, y] of t) { lo = [Math.min(lo[0], x), Math.min(lo[1], y)]; hi = [Math.max(hi[0], x), Math.max(hi[1], y)]; }
    let corners;
    if (hi[0] - lo[0] < 1e-9 || hi[1] - lo[1] < 1e-9 || l.pattern.type === "line") corners = [t[0], t[t.length - 1], t[0], t[t.length - 1]];
    else corners = [[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]];
    const depth = l.aim ? G.norm(G.sub(l.aim, l.pos)) * 1.15 : 0.3;
    const f = frustumPts(l, corners, depth);
    setPts(o.lines, f.lines); setPts(o.faces, f.faces);
    o.lines.visible = o.faces.visible = view.fans && l.enabled;
    // hit points: bright if some camera sees them, dim if none does
    const h = results.hits[l.name], pa = o.points.geometry.attributes.position, ca = o.points.geometry.attributes.color;
    const col = new THREE.Color(l.colour), dim = col.clone().multiplyScalar(0.28);
    if (h) {
      const n = Math.min(h.n, pa.count);
      pa.array.set(h.pts.subarray(0, 3 * n));
      for (let i = 0; i < n; i++) { const cc = h.seen[i] ? col : dim; ca.setXYZ(i, cc.r, cc.g, cc.b); }
      o.points.geometry.setDrawRange(0, n);
      pa.needsUpdate = ca.needsUpdate = true;
    }
    o.points.visible = !!h;
  }
  for (const [p, rod] of G3.rods) {
    const xs = p.cams.map((c) => c.pos[0]);
    const x0 = Math.min(...xs) - 0.05, x1 = Math.max(...xs) + 0.05;
    rod.scale.set(x1 - x0, 1, 1);
    rod.position.set((x0 + x1) / 2, (p.cams[0].pos[1] + p.cams[1].pos[1]) / 2, (p.cams[0].pos[2] + p.cams[1].pos[2]) / 2 - 0.03);
  }
  for (const [p, m] of G3.aims) { const a = pairAim(p); if (a) m.position.set(...a); m.material.color.set(COLOURS[p.side][0]); }
  wheelGroup.position.set(...world.o);
  wheelGroup.rotation.set(0, G.rad(wheelValue("wheel_rot_deg")), 0);
  sun.target.position.set(...world.o); sun.position.set(world.o[0] + 1.2, world.o[1] - 2, 3);
}

// ------------------------------------------------------------------ rendering
let pending = false, needCompute = true;
function schedule(compute = true) {
  needCompute = needCompute || compute;
  if (!pending) { pending = true; requestAnimationFrame(frame); }
}

function previewCamera(c, near = 0.01, far = 30) {
  const pc = new THREE.PerspectiveCamera();
  pc.matrixAutoUpdate = false;
  const x = new THREE.Vector3(...c.R[0]), y = new THREE.Vector3(...c.R[1]).negate(), z = new THREE.Vector3(...c.R[2]).negate();
  pc.matrix.makeBasis(x, y, z).setPosition(...c.pos);
  pc.matrixWorldNeedsUpdate = true;
  const w = c.W, h = c.H;
  pc.projectionMatrix.set(
    (2 * c.fx) / w, 0, (w - 2 * c.cx - 1) / w, 0,
    0, (2 * c.fy) / h, (2 * c.cy + 1 - h) / h, 0,
    0, 0, -(far + near) / (far - near), (-2 * far * near) / (far - near),
    0, 0, -1, 0);
  pc.projectionMatrixInverse.copy(pc.projectionMatrix).invert();
  pc.layers.set(0);
  return pc;
}

function stripLayout(W) {
  const cams = [...S.cams.values()];
  if (!view.strip || !cams.length) return { h: 0, boxes: [] };
  const gap = 6, aspects = cams.map((c) => c.W / c.H), sumA = aspects.reduce((a, b) => a + b, 0);
  const h = Math.max(60, Math.min(view.stripH, (W - gap * (cams.length + 1)) / sumA));
  let x = gap;
  const boxes = cams.map((c, i) => { const b = { cam: c, x, y: gap, w: h * aspects[i], h }; x += b.w + gap; return b; });
  return { h: h + 2 * gap, boxes };
}

function frame() {
  pending = false;
  if (!S || !world) return;
  if (needCompute) {
    const t0 = performance.now();
    recompute();
    results.ms = performance.now() - t0;
    updateGizmos(); refreshReadouts(); drawInset();
    needCompute = false;
  }
  const W = main.clientWidth, H = main.clientHeight;
  if (canvas.width !== Math.floor(W * renderer.getPixelRatio()) || canvas.height !== Math.floor(H * renderer.getPixelRatio())) renderer.setSize(W, H, false);
  const strip = stripLayout(W);
  const mh = Math.max(50, H - strip.h);
  cam3.aspect = W / mh; cam3.updateProjectionMatrix();
  renderer.setViewport(0, strip.h, W, mh); renderer.setScissor(0, strip.h, W, mh);
  renderer.render(scene, cam3);
  // camera views along the bottom
  const sdiv = $("strip"); sdiv.style.height = strip.h + "px";
  while (sdiv.children.length > strip.boxes.length) sdiv.lastChild.remove();
  for (const g3 of G3.lasers.values()) g3.points.material.size = 2.5;
  strip.boxes.forEach((b, i) => {
    renderer.setViewport(b.x, b.y, b.w, b.h); renderer.setScissor(b.x, b.y, b.w, b.h);
    renderer.render(scene, previewCamera(b.cam));
    let d = sdiv.children[i];
    if (!d) { d = el("div", "cv"); d.append(el("span")); sdiv.append(d); }
    Object.assign(d.style, { left: b.x + "px", bottom: b.y + "px", width: b.w + "px", height: b.h + "px", borderColor: b.cam.colour });
    const st = results.stats[b.cam.name] || {};
    d.firstChild.innerHTML = `<b style="color:${b.cam.colour}">${b.cam.name}</b> ` + Object.entries(st)
      .map(([ln, f]) => `<span style="color:${S.lasers.get(ln).colour}">${ln} ${f == null ? "–" : Math.round(100 * f) + "%"}</span>`).join(" ");
    d.firstChild.title = "% of this laser's points on the wheel that the camera sees (in view, not hidden)";
  });
  for (const g3 of G3.lasers.values()) g3.points.material.size = 4;
  drawLabels(W, mh);
}

function drawLabels(W, H) {
  const box = $("labels");
  box.style.bottom = (main.clientHeight - H) + "px";
  const items = [];
  if (view.labels) {
    for (const c of S.cams.values()) items.push([c.name, c.pos, c.colour]);
    for (const l of S.lasers.values()) if (l.enabled) items.push([l.name, l.pos, l.colour]);
  }
  while (box.children.length > items.length) box.lastChild.remove();
  items.forEach(([text, p, colour], i) => {
    const v = new THREE.Vector3(...p).project(cam3);
    let d = box.children[i] || box.appendChild(el("div"));
    d.textContent = text; d.style.color = colour;
    d.style.display = v.z < 1 && Math.abs(v.x) < 1.2 && Math.abs(v.y) < 1.2 ? "block" : "none";
    d.style.left = ((v.x + 1) / 2) * W + "px"; d.style.top = ((1 - v.y) / 2) * H + "px";
  });
}

// ------------------------------------------------------------------ profile inset
function drawInset() {
  const cv = $("prof"), box = $("inset");
  box.style.display = view.inset ? "block" : "none";
  if (!view.inset || box.classList.contains("min")) return;
  const dpr = window.devicePixelRatio || 1, Wc = cv.clientWidth, Hc = cv.clientHeight;
  cv.width = Wc * dpr; cv.height = Hc * dpr;
  const g = cv.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.fillStyle = "#0f1420"; g.fillRect(0, 0, Wc, Hc);
  const cov = results.cov, o = world.o, R = world.R;
  // the section drawn where it meets the rail (as seen looking along -x: field side left)
  const Y = (a) => o[1] - a, Z = (dr) => -dr;
  const ys = cov.a.map(Y), zs = cov.dr.map(Z);
  const y0 = Math.min(...ys) - 0.012, y1 = Math.max(...ys) + 0.012;
  const z0 = -0.045, z1 = Math.max(...zs.filter((z) => z < 0.1)) + 0.012;
  const k = Math.min((Wc - 20) / (y1 - y0), (Hc - 24) / (z1 - z0));
  const ox = (Wc - k * (y1 - y0)) / 2, oy = Hc - 18 - (Hc - 24 - k * (z1 - z0)) / 2;
  const sx = (y) => ox + (y - y0) * k, sy = (z) => oy - (z - z0) * k;
  // grid every 10 mm
  g.strokeStyle = "#1e2636"; g.lineWidth = 1; g.font = "10px system-ui"; g.fillStyle = "#5b677c";
  for (let a = 0; a <= 0.16; a += 0.01) { const x = sx(Y(a)); if (x < 0 || x > Wc) continue; g.beginPath(); g.moveTo(x, 0); g.lineTo(x, Hc - 14); g.stroke(); if (Math.round(a * 1000) % 20 === 0) g.fillText(Math.round(a * 1000), x - 6, Hc - 3); }
  for (let z = -0.04; z <= z1; z += 0.01) { const y = sy(z); g.beginPath(); g.moveTo(0, y); g.lineTo(Wc, y); g.stroke(); }
  g.fillText("mm from flange back face", Wc / 2 - 55, 12);
  // rail
  g.fillStyle = "#3b4352";
  for (const r of world.g.rails) {
    if (r.side !== "right") continue;
    g.beginPath(); r.poly.forEach(([y, z], i) => (i ? g.lineTo(sx(y), sy(z)) : g.moveTo(sx(y), sy(z)))); g.closePath(); g.fill();
  }
  // profile
  g.strokeStyle = "#d6dbe3"; g.lineWidth = 1.6; g.beginPath();
  ys.forEach((y, i) => (i ? g.lineTo(sx(y), sy(zs[i])) : g.moveTo(sx(y), sy(zs[i])))); g.stroke();
  // coverage bands, offset along the outward normal (screen): outside pair 5 px, inside 10 px
  const n = ys.length;
  const nrm = ys.map((_, i) => {
    const i0 = Math.max(0, i - 1), i1 = Math.min(n - 1, i + 1);
    const dx = sx(ys[i1]) - sx(ys[i0]), dy = sy(zs[i1]) - sy(zs[i0]), l = Math.hypot(dx, dy) || 1;
    return [dy / l, -dx / l];
  });
  // bands go on the wheel side of the profile (up on the tread), clear of the rail
  const mid = Math.floor(n * 0.75), sgn = nrm[mid][1] > 0 ? -1 : 1;
  const sets = groups();
  sets.forEach((s, si) => {
    const off = 5 + 5 * (s.side === "inside" ? 1 : 0) + (si > 1 ? 5 : 0);
    for (let i = 0; i < n - 1; i++) {
      const cnt = s.cams.reduce((a, c) => a + (cov.seen[c.name][i] && cov.seen[c.name][i + 1] ? 1 : 0), 0);
      if (!cnt) continue;
      g.strokeStyle = s.colour; g.lineWidth = cnt >= s.cams.length && s.cams.length > 1 ? 3.2 : 1.2;
      g.beginPath();
      g.moveTo(sx(ys[i]) + sgn * off * nrm[i][0], sy(zs[i]) + sgn * off * nrm[i][1]);
      g.lineTo(sx(ys[i + 1]) + sgn * off * nrm[i + 1][0], sy(zs[i + 1]) + sgn * off * nrm[i + 1][1]);
      g.stroke();
    }
  });
  // not seen by any camera: yellow ticks on the profile
  g.strokeStyle = "#eab308"; g.lineWidth = 2.2;
  const all = [...S.cams.values()];
  for (let i = 0; i < n - 1; i++) {
    if (all.some((c) => cov.seen[c.name][i])) continue;
    g.beginPath(); g.moveTo(sx(ys[i]), sy(zs[i])); g.lineTo(sx(ys[i + 1]), sy(zs[i + 1])); g.stroke();
  }
  g.fillStyle = "#8b97ab";
  g.fillText(`field side`, 6, 12); g.fillText(`gauge side`, Wc - 58, 12);
}

function groups() {
  const out = S.pairs.map((p) => ({ name: p.name, side: p.side, cams: p.cams, colour: COLOURS[p.side][0] }));
  for (const c of S.singles) out.push({ name: c.name, side: "single", cams: [c], colour: c.colour });
  return out;
}

function coverageMetrics() {
  const cov = results.cov, tot = cov.total, n = cov.arc.length;
  const frac = (pred) => { let s = 0; for (let i = 0; i < n; i++) if (pred(i)) s += cov.arc[i]; return s; };
  const rows = groups().map((s) => {
    const both = frac((i) => s.cams.every((c) => cov.seen[c.name][i]));
    const any = frac((i) => s.cams.some((c) => cov.seen[c.name][i]));
    const idx = [...Array(n).keys()].filter((i) => s.cams.every((c) => cov.seen[c.name][i]));
    const span = idx.length ? `${Math.round(cov.a[idx[0]] * MM)}–${Math.round(cov.a[idx[idx.length - 1]] * MM)} mm` : "–";
    return { s, both: both / tot, any: any / tot, span };
  });
  const sideAny = (side) => (i) => groups().filter((s) => s.side === side).some((s) => s.cams.some((c) => cov.seen[c.name][i]));
  const all = frac((i) => [...S.cams.values()].some((c) => cov.seen[c.name][i]));
  const overlap = frac((i) => sideAny("outside")(i) && sideAny("inside")(i));
  return { rows, all: all / tot, unmeasured: (tot - all) * MM, overlap: overlap * MM };
}

function refreshReadouts() {
  const m = coverageMetrics(), cov = results.cov;
  const aimH = results.refPair ? pairAim(results.refPair)[2] * MM : null;
  $("inset-title").textContent = `Profile coverage, section ${cov.phiDeg.toFixed(0)}° from the bottom` + (aimH != null ? ` (${aimH.toFixed(0)} mm above rail)` : "");
  let h = "<table>";
  h += `<tr><td></td><td class="n">both cams</td><td class="n">any cam</td><td class="n">both: a range</td></tr>`;
  for (const r of m.rows) {
    const label = r.s.side === "outside" ? "outside (field)" : r.s.side === "inside" ? "inside (gauge)" : "single";
    h += `<tr><td><span class="chip" style="background:${r.s.colour}"></span> ${r.s.name} <span style="color:#8b97ab">${label}</span></td>` +
      `<td class="n">${(100 * r.both).toFixed(0)}%</td><td class="n">${(100 * r.any).toFixed(0)}%</td><td class="n">${r.span}</td></tr>`;
  }
  h += `<tr><td>all cameras</td><td></td><td class="n">${(100 * m.all).toFixed(0)}%</td><td class="n">overlap ${m.overlap.toFixed(0)} mm</td></tr>`;
  h += "</table>";
  h += `<div style="color:#8b97ab;margin-top:3px">thick = both cameras of a pair, thin = one, <span style="color:#eab308">yellow</span> = no camera (${m.unmeasured.toFixed(0)} mm of profile). Geometry computed in ${results.ms.toFixed(0)} ms.</div>`;
  $("metrics").innerHTML = h;
  for (const f of readouts) f();
}

// ------------------------------------------------------------------ controls
const refreshers = [], readouts = [];
let saveTimer = null;
function refreshAll() { for (const f of refreshers) f(); }
function changed() { refreshAll(); schedule(true); }
function commit() { clearTimeout(saveTimer); saveTimer = setTimeout(sendEdits, 350); }

function slider(parent, o) {
  const d = el("div", "sl"), lab = el("div", "lab"), name = el("span", null, o.label);
  const num = el("input"); num.type = "number"; num.step = "any";
  const rng = el("input"); rng.type = "range"; rng.min = o.min; rng.max = o.max; rng.step = o.step;
  if (o.title) d.title = o.title;
  lab.append(name, num); d.append(lab, rng); parent.append(d);
  const digits = o.digits ?? 1;
  const show = () => {
    if (o.visible) d.style.display = o.visible() ? "" : "none";
    const v = o.get();
    if (v == null || !isFinite(v)) { d.classList.add("disabled"); return; }
    d.classList.toggle("disabled", o.enabled ? !o.enabled() : false);
    if (v < +rng.min) rng.min = v; if (v > +rng.max) rng.max = v;
    rng.value = v;
    if (document.activeElement !== num) num.value = (+v).toFixed(digits);
  };
  rng.addEventListener("input", () => { o.set(+rng.value); changed(); });
  rng.addEventListener("change", commit);
  num.addEventListener("change", () => { const v = parseFloat(num.value); if (isFinite(v)) { o.set(v); changed(); commit(); } });
  refreshers.push(show); show();
  return d;
}
function toggle(parent, label, get, set) {
  const l = el("label", "tg"), c = el("input"); c.type = "checkbox";
  l.append(c, document.createTextNode(label)); parent.append(l);
  c.addEventListener("change", () => { set(c.checked); changed(); commit(); });
  refreshers.push(() => (c.checked = !!get()));
  c.checked = !!get();
  return l;
}
function section(title, colour, open = false) {
  const d = el("details"); d.open = open;
  const s = el("summary");
  if (colour) { const ch = el("span", "chip"); ch.style.background = colour; s.append(ch); }
  s.append(document.createTextNode(title));
  const b = el("div", "body"); d.append(s, b); $("controls").append(d);
  return b;
}
function readout(parent, fn) { const d = el("div", "read"); parent.append(d); readouts.push(() => (d.innerHTML = fn())); return d; }

const POS = { x: [-1500, 1500], y: [-1800, 300], z: [-300, 1200] };
function vec3(parent, label, get, set, dirty) {
  ["x", "y", "z"].forEach((ax, k) => slider(parent, {
    label: `${label} ${ax} (mm)`, min: POS[ax][0], max: POS[ax][1], step: 1, digits: 1,
    get: () => { const v = get(); return v ? v[k] * MM : null; },
    set: (v) => { const p = get().slice(); p[k] = v / MM; set(p); dirty(); },
  }));
}

// aim helpers: the point on the wheel `h` mm above the rail and `a` mm from the flange back
function aimOnWheelSliders(parent, getAim, setAim, getX, dirty) {
  const cur = () => { const a = getAim(); return a ? { h: a[2] * MM, a: (world.o[1] - a[1]) * MM } : null; };
  slider(parent, {
    label: "aim height above rail (mm)", min: 0, max: 300, step: 0.5, digits: 1, title: "moves the aim point along the wheel surface",
    get: () => cur()?.h, set: (v) => { setAim(world.aimOnWheel(v / MM, cur().a / MM, getX())); dirty(); },
  });
  slider(parent, {
    label: "aim across profile (mm from flange back)", min: 0, max: 145, step: 0.5, digits: 1,
    get: () => cur()?.a, set: (v) => { setAim(world.aimOnWheel(cur().h / MM, v / MM, getX())); dirty(); },
  });
}

function pairAim(p) {
  const [a, b] = p.cams.map((c) => c.aim);
  return a && b ? G.scale(G.add(a, b), 0.5) : a || b;
}
const touch = (dev, what = "pose") => () => dev.dirty.add(what);

function pairControls(p) {
  const [c0, c1] = p.cams;
  const b = section(`Pair ${p.name}: ${p.cams.map((c) => c.name).join(" + ")}`, null, true);
  const sum = b.parentElement.querySelector("summary");
  const chip = el("span", "chip"); sum.prepend(chip);
  const sideLab = el("span", "", ""); sideLab.style.cssText = "color:#8b97ab;font-weight:400"; sum.append(sideLab);
  refreshers.push(() => { chip.style.background = COLOURS[p.side || "outside"][0]; sideLab.textContent = p.side === "inside" ? " inside the rail (gauge side)" : " outside the rail (field side)"; });
  const dirtyBoth = () => p.cams.forEach((c) => { c.dirty.add("pose"); updateR(c); });
  const centre = () => G.scale(G.add(c0.pos, c1.pos), 0.5);
  b.append(el("div", "sub", "rod (both cameras move together)"));
  vec3(b, "rod centre", centre, (v) => { const d = G.sub(v, centre()); p.cams.forEach((c) => (c.pos = G.add(c.pos, d))); }, dirtyBoth);
  slider(b, {
    label: "rod length = baseline (mm)", min: 0, max: 1200, step: 1, digits: 1, title: "camera spacing along the rod (along the rail, x)",
    get: () => Math.abs(c1.pos[0] - c0.pos[0]) * MM,
    set: (v) => {
      const cx = centre()[0], s = c1.pos[0] >= c0.pos[0] ? 1 : -1;
      c0.pos = [cx - (s * v) / MM / 2, c0.pos[1], c0.pos[2]]; c1.pos = [cx + (s * v) / MM / 2, c1.pos[1], c1.pos[2]];
      dirtyBoth();
    },
  });
  if (!c0.aim || !c1.aim) { b.append(el("div", "read", "Orientation is given as euler/R in the YAML: not editable here.")); return; }
  b.append(el("div", "sub", "aim (both cameras look at the same point)"));
  const setAim = (a) => p.cams.forEach((c) => (c.aim = a.slice()));
  aimOnWheelSliders(b, () => pairAim(p), setAim, () => centre()[0], dirtyBoth);
  vec3(b, "aim", () => pairAim(p), setAim, dirtyBoth);
  b.append(el("div", "sub", "optics"));
  slider(b, { label: "roll about optical axis (deg)", min: -180, max: 180, step: 0.5, get: () => c0.roll,
    set: (v) => p.cams.forEach((c) => { c.roll = v; c.dirty.add("roll"); updateR(c); }) });
  slider(b, { label: "field of view, horizontal (deg)", min: 5, max: 120, step: 0.1, get: () => hfov(c0),
    set: (v) => p.cams.forEach((c) => setHfov(c, v)) });
  readout(b, () => {
    const a = pairAim(p), r = p.cams.map((c) => G.norm(G.sub(a, c.pos)));
    const u = p.cams.map((c) => G.unit(G.sub(a, c.pos)));
    const conv = G.deg(Math.acos(Math.max(-1, Math.min(1, G.dot(u[0], u[1])))));
    const up = p.cams.map((c) => G.deg(Math.asin(c.R[2][2])));
    const fmm = c0.fx * c0.pxum / 1000;
    return `range to aim ${r.map((x) => (x * MM).toFixed(0)).join(" / ")} mm · look-up ${up.map((x) => x.toFixed(1)).join(" / ")}°<br>` +
      `convergence ${conv.toFixed(1)}° · ${(r[0] / c0.fx * MM).toFixed(3)} mm/px at aim · f = ${fmm.toFixed(2)} mm (${c0.W}×${c0.H})` +
      (p.rectify ? "" : "<br>stereo pair, not rectified (raw images + calib/pair_" + p.name + ".yml)");
  });
}

function singleCameraControls() {
  const b = section("Single camera (fine-tune)");
  const sel = el("select"); b.append(sel);
  for (const n of S.cams.keys()) sel.append(new Option(n, n));
  sel.value = S.selCam;
  sel.addEventListener("change", () => { S.selCam = sel.value; refreshAll(); });
  const cam = () => S.cams.get(S.selCam);
  const dirty = () => { cam().dirty.add("pose"); updateR(cam()); };
  vec3(b, "position", () => cam().pos, (v) => (cam().pos = v), dirty);
  aimOnWheelSliders(b, () => cam().aim, (a) => (cam().aim = a), () => cam().pos[0], dirty);
  vec3(b, "aim", () => cam().aim, (v) => (cam().aim = v), dirty);
  slider(b, { label: "roll about optical axis (deg)", min: -180, max: 180, step: 0.5, get: () => (cam().aim ? cam().roll : null),
    set: (v) => { cam().roll = v; cam().dirty.add("roll"); updateR(cam()); } });
  slider(b, { label: "field of view, horizontal (deg)", min: 5, max: 120, step: 0.1, get: () => hfov(cam()), set: (v) => setHfov(cam(), v) });
}

const PATTERN_DEFAULT = { line: { type: "line", fan_deg: 30, angle_deg: 0 }, grid: { type: "grid", rows: 15, cols: 21, fan_deg: [24, 16] } };
function laserControls(l) {
  const b = section(`Laser ${l.name}`, l.colour);
  const dirty = () => { l.dirty.add("pose"); updateR(l); };
  toggle(b, "enabled", () => l.enabled, (v) => { l.enabled = v; l.dirty.add("enabled"); });
  vec3(b, "position", () => l.pos, (v) => (l.pos = v), dirty);
  if (l.aim) {
    aimOnWheelSliders(b, () => l.aim, (a) => (l.aim = a), () => l.pos[0], dirty);
    vec3(b, "aim", () => l.aim, (v) => (l.aim = v), dirty);
  }
  const editable = ["grid", "line"].includes(l.pattern.type) && !(l.pattern.jitter_deg > 0);
  if (!editable) { b.append(el("div", "read", `Pattern ${l.pattern.type}: edit it in the YAML.`)); }
  else {
    const row = el("div", "row"), sel = el("select");
    for (const t of ["grid", "line"]) sel.append(new Option(t === "grid" ? "dot grid" : "line (fan)", t));
    row.append(el("span", null, "pattern"), sel); b.append(row);
    refreshers.push(() => (sel.value = l.pattern.type));
    const regen = () => { l.tangents = G.patternTangents(l.pattern, l.sigma); l.dirty.add("pattern"); };
    sel.addEventListener("change", () => { l.pattern = structuredClone(PATTERN_DEFAULT[sel.value]); regen(); rebuildPoints(l); changed(); commit(); });
    const grid = () => l.pattern.type === "grid", line = () => l.pattern.type === "line";
    const fan = (k) => (l.pattern.fan_deg ? l.pattern.fan_deg[k] : l.pattern.pitch_deg[k] * ((k ? l.pattern.rows : l.pattern.cols) - 1));
    const setFan = (k, v) => { const f = [fan(0), fan(1)]; f[k] = v; delete l.pattern.pitch_deg; l.pattern.fan_deg = f; regen(); };
    slider(b, { label: "columns", min: 1, max: 61, step: 1, digits: 0, visible: grid, get: () => (grid() ? l.pattern.cols : 1), set: (v) => { l.pattern.cols = Math.round(v); regen(); rebuildPoints(l); } });
    slider(b, { label: "rows", min: 1, max: 61, step: 1, digits: 0, visible: grid, get: () => (grid() ? l.pattern.rows : 1), set: (v) => { l.pattern.rows = Math.round(v); regen(); rebuildPoints(l); } });
    slider(b, { label: "fan angle across columns (deg)", min: 0.5, max: 120, step: 0.5, visible: grid, get: () => (grid() ? fan(0) : 1), set: (v) => setFan(0, v) });
    slider(b, { label: "fan angle across rows (deg)", min: 0.5, max: 120, step: 0.5, visible: grid, get: () => (grid() ? fan(1) : 1), set: (v) => setFan(1, v) });
    slider(b, { label: "fan angle (deg)", min: 1, max: 120, step: 0.5, visible: line, get: () => (line() ? l.pattern.fan_deg : 1), set: (v) => { l.pattern.fan_deg = v; regen(); rebuildPoints(l); } });
    slider(b, { label: "line orientation about the beam (deg)", min: -90, max: 90, step: 1, visible: line, get: () => (line() ? l.pattern.angle_deg || 0 : 0), set: (v) => { l.pattern.angle_deg = v; regen(); } });
  }
  slider(b, { label: "power (relative W)", min: 0, max: 1000, step: 5, digits: 0, get: () => l.power, set: (v) => { l.power = v; l.dirty.add("power"); } });
  readout(b, () => {
    const h = results.hits[l.name];
    if (!h) return "disabled";
    const c = { 1: 0, 2: 0, 4: 0, 8: 0 }; for (let i = 0; i < h.n; i++) c[h.cls[i]]++;
    return `${l.tangents.length} rays: ${c[1]} on the wheel, ${c[2]} on the rails, ${c[4] + c[8]} on ties/ballast`;
  });
}
function rebuildPoints(l) {
  const o = G3.lasers.get(l.name), n = l.tangents.length + 4096;
  if (o.points.geometry.attributes.position.count >= l.tangents.length) return;
  o.points.geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(n * 3), 3));
  o.points.geometry.setAttribute("color", new THREE.BufferAttribute(new Float32Array(n * 3), 3));
}

const WHEEL_UI = [
  ["wheel_x_m", "position along the rail x (mm)", -1000, 1000, 1, MM, 1],
  ["wheel_rot_deg", "rotation about the axle (deg)", 0, 360, 1, 1, 0],
  ["lateral_shift_m", "lateral shift (mm)", -20, 20, 0.1, MM, 1],
];
function wheelControls() {
  const b = section("Wheel");
  const row = el("div", "row"), sel = el("select");
  for (const n of S.scene.profiles) sel.append(new Option(n, n));
  sel.value = S.scene.wheel_profile;
  row.append(el("span", null, "profile"), sel); b.append(row);
  sel.addEventListener("change", () => { S.scene.wheel_profile = sel.value; S.scene.dirty.add("wheel_profile"); setStatus("loading profile ...", "warn"); commit(); });
  for (const [k, label, lo, hi, step, f, digits] of WHEEL_UI) {
    const fld = S.scene[k];
    b.append(el("div", "sub", label.replace(/ \(.*/, "")));
    if (fld.spec) { b.append(el("div", "read", `random (${JSON.stringify(fld.spec)}), edit it in scene.yaml`)); continue; }
    slider(b, { label, min: lo, max: hi, step, digits, enabled: () => !fld.range, get: () => wheelValue(k) * f,
      set: (v) => { fld.value = v / f; S.scene.dirty.add(k); } });
    toggle(b, "random every frame, between:", () => !!fld.range, (on) => {
      fld.range = on ? [fld.value - 20 * step / f, fld.value + 20 * step / f] : null;
      if (!on) fld.value = wheelValue(k);
      S.scene.dirty.add(k);
    });
    for (const j of [0, 1]) slider(b, { label: j ? "to" : "from", min: lo, max: hi, step, digits, visible: () => !!fld.range,
      get: () => (fld.range ? fld.range[j] * f : null), set: (v) => { fld.range[j] = v / f; S.scene.dirty.add(k); } });
  }
  b.append(el("div", "read", "The rig is fixed in the track frame: moving the wheel does not move the cameras or their aim points."));
}

function displayControls() {
  const b = section("Display");
  toggle(b, "camera frustums", () => view.frustums, (v) => (view.frustums = v));
  toggle(b, "laser fans", () => view.fans, (v) => (view.fans = v));
  toggle(b, "labels", () => view.labels, (v) => (view.labels = v));
  toggle(b, "camera views (bottom strip)", () => view.strip, (v) => (view.strip = v));
  toggle(b, "profile coverage inset", () => view.inset, (v) => (view.inset = v));
  slider(b, { label: "camera view height (px)", min: 80, max: 420, step: 10, digits: 0, get: () => view.stripH, set: (v) => (view.stripH = v) });
  b.append(el("div", "read", "Camera views: pinhole preview of each camera (lens distortion not drawn). Laser points: bright = seen by at least one camera, dark = by none."));
}

function buildPanel() {
  $("controls").innerHTML = ""; refreshers.length = 0; readouts.length = 0;
  for (const p of S.pairs) pairControls(p);
  for (const l of S.lasers.values()) laserControls(l);
  wheelControls();
  singleCameraControls();
  displayControls();
}

// ------------------------------------------------------------------ toolbar / views
const PRESETS = {
  "Iso": [50, 22, 1.4], "Along the rail": [90, 8, 1.0], "Field side": [0, 6, 1.3], "Gauge side": [180, 6, 1.3],
  "Top": [0, 88, 1.6], "Tread close-up": [72, 4, 0.42],
};
function viewTarget() {
  const p = S.pairs.find((q) => q.side === "outside") || S.pairs[0];
  return (p && pairAim(p)) || world.g.key_points.tread_contact;
}
function setView(az, el_, dist, target = viewTarget()) {
  const a = G.rad(az), e = G.rad(el_);
  const back = [Math.sin(a) * Math.cos(e), -Math.cos(a) * Math.cos(e), Math.sin(e)];
  orbit.target.set(...target);
  cam3.position.set(...G.add(target, G.scale(back, dist)));
  orbit.update(); schedule(false);
}
function currentView() {
  const t = orbit.target.toArray(), off = G.sub(cam3.position.toArray(), t), dist = G.norm(off), b = G.scale(off, 1 / dist);
  const aspect = cam3.aspect, hf = 2 * Math.atan(Math.tan(G.rad(cam3.fov) / 2) * aspect);
  return { target: t, azimuth: G.deg(Math.atan2(b[0], -b[1])), elevation: G.deg(Math.asin(b[2])), distance: dist,
    hfov_deg: G.deg(hf), size: [1600, Math.round(1600 / aspect / 2) * 2] };
}
function buildToolbar() {
  const tb = $("toolbar"); tb.innerHTML = "";
  for (const [name, v] of Object.entries(PRESETS)) {
    const bt = el("button", null, name); bt.onclick = () => setView(...v); tb.append(bt);
  }
  const r = el("button", null, "Render this view in Blender");
  r.title = "Path-traced render of the current viewpoint (shown in the 3D view tab)";
  r.onclick = () => sendEdits({ type: "render_view", view: currentView() });
  tb.append(r);
}
$("inset-min").onclick = () => { $("inset").classList.toggle("min"); schedule(true); };

// ------------------------------------------------------------------ edits -> Python
const r4 = (v) => v.map((x) => Math.round(x * 1e5) / 1e5);
function devicePose(dev) {
  const o = {};
  if (dev.dirty.has("pose")) { o.position = r4(dev.pos); if (dev.aim) o.look_at = r4(dev.aim); }
  if (dev.aim && (dev.dirty.has("roll") || (dev.dirty.has("pose") && (dev.upInFile || Math.abs(dev.roll) > 0.05)))) {
    o.up = G.upForRoll(dev.pos, dev.aim, dev.roll).map((x) => Math.round(x * 1e5) / 1e5);
  }
  return o;
}
function cleanPattern(p) {
  if (p.type === "grid") {
    const o = { type: "grid", rows: p.rows, cols: p.cols };
    if (p.fan_deg) o.fan_deg = p.fan_deg.map((x) => Math.round(x * 1000) / 1000); else o.pitch_deg = p.pitch_deg;
    return o;
  }
  if (p.type === "line") {
    const o = { type: "line", fan_deg: Math.round(p.fan_deg * 1000) / 1000, angle_deg: Math.round((p.angle_deg || 0) * 1000) / 1000 };
    if (p.spacing_mrad) o.spacing_mrad = p.spacing_mrad;
    return o;
  }
  return p;
}
function buildEdits() {
  const e = { cameras: {}, lasers: {}, scene: {} };
  for (const c of S.cams.values()) {
    if (!c.dirty.size) continue;
    const o = devicePose(c);
    if (c.dirty.has("hfov")) o.hfov_deg = Math.round(hfov(c) * 1000) / 1000;
    e.cameras[c.name] = o;
  }
  for (const l of S.lasers.values()) {
    if (!l.dirty.size) continue;
    const o = devicePose(l);
    if (l.dirty.has("enabled")) o.enabled = l.enabled;
    if (l.dirty.has("power")) o.power_w = l.power;
    if (l.dirty.has("pattern")) o.pattern = cleanPattern(l.pattern);
    e.lasers[l.name] = o;
  }
  for (const k of S.scene.dirty) {
    if (k === "wheel_profile") e.scene.wheel_profile = S.scene.wheel_profile;
    else { const f = S.scene[k]; e.scene[k] = f.range ? { uniform: f.range.map((x) => Math.round(x * 1e5) / 1e5) } : Math.round(f.value * 1e5) / 1e5; }
  }
  return e;
}
function sendEdits(action = null) {
  clearTimeout(saveTimer);
  if (!inStreamlit) { setStatus("standalone preview: edits are not saved", "warn"); return; }
  bridge.value({ id: `${SID}:${++seq}`, rev, edits: buildEdits(), action });
}
function setStatus(text, level = "") { const s = $("status"); s.textContent = text; s.className = level; }

// ------------------------------------------------------------------ args from Python
function onArgs(args) {
  if (args.geometry && args.geometry.key !== geomKey) {
    geomKey = args.geometry.key;
    world = new G.World(args.geometry);
    buildStatic(args.geometry);
  }
  if (args.rev !== rev) {
    rev = args.rev;
    S = buildState(args.state);
    assignColours();
    buildPanel(); buildGizmos(); buildToolbar();
    setView(...PRESETS["Iso"]);
  }
  if (args.status) setStatus(args.status.text, args.status.level);
  schedule(true);
  fitHeight();
}

window.addEventListener("message", (ev) => { if (ev.data && ev.data.type === "streamlit:render") onArgs(ev.data.args); });
window.addEventListener("resize", () => schedule(false));
try { window.parent.addEventListener("resize", fitHeight); } catch (e) { /* cross-origin */ }
if (inStreamlit) bridge.ready();
else fetch("args.json").then((r) => r.json()).then(onArgs).catch((e) => setStatus("standalone: no args.json (" + e + ")", "err"));
window.__layout = { get S() { return S; }, get results() { return results; }, get world() { return world; }, currentView, setView,
  bench(n = 20) { const t0 = performance.now(); for (let i = 0; i < n; i++) recompute(); return (performance.now() - t0) / n; } };
