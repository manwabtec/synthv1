// Layout geometry in the track frame (x along track, y left, z up, metres). No three.js.
//
// Exact ray casting against the same shapes the Blender scene builds:
//   wheel  = closed (a, r) section polygons revolved about the axle (a = lateral from the
//            flange back face, + towards the field side = -y; r = radius),
//   rails  = (y, z) polygons extruded along x,  ties / tie plates = boxes,  ballast = plane.
// A ray meets the wheel where its (a(t), r(t)) path crosses a polygon edge: a(t) is linear
// and r(t)^2 quadratic in t, so each edge gives a quadratic. No meshes, no marching.

export const WHEEL = 1, RAIL = 2, TIE = 4, BALLAST = 8, ALL = 15;
export const CLASS_NAME = { 1: "wheel", 2: "rail", 4: "tie", 8: "ballast" };

// ------------------------------------------------------------------ small vector helpers
export const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
export const scale = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
export const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
export const norm = (a) => Math.hypot(a[0], a[1], a[2]);
export const unit = (a) => { const n = norm(a) || 1; return [a[0] / n, a[1] / n, a[2] / n]; };
export const deg = (r) => (r * 180) / Math.PI;
export const rad = (d) => (d * Math.PI) / 180;

// Rotation (columns = device x, y, z in the track frame) with z towards the target and
// y = "down" = -up, like synthrig.opticsrig.transforms.look_at_R.
export function lookAtR(pos, target, up) {
  const z = unit(sub(target, pos));
  let x = cross(scale(up, -1), z);
  if (norm(x) < 1e-9) x = cross([0, -1, 0], z); // up parallel to the view: any roll
  x = unit(x);
  const y = cross(z, x);
  return [x, y, z];
}
export const applyR = (R, v) => [
  R[0][0] * v[0] + R[1][0] * v[1] + R[2][0] * v[2],
  R[0][1] * v[0] + R[1][1] * v[1] + R[2][1] * v[2],
  R[0][2] * v[0] + R[1][2] * v[1] + R[2][2] * v[2],
];
// R from the YAML ([[r00, r01, r02], ...], rows) -> columns
export const rowsToCols = (M) => [[M[0][0], M[1][0], M[2][0]], [M[0][1], M[1][1], M[2][1]], [M[0][2], M[1][2], M[2][2]]];

// Roll about the optical axis (deg) relative to "image up = world up" (app.py roll_of).
export function rollOf(pos, aim, up) {
  const z = unit(sub(aim, pos));
  const u0 = sub([0, 0, 1], scale(z, z[2]));
  let u = up || [0, 0, 1];
  u = sub(u, scale(z, dot(u, z)));
  return deg(Math.atan2(dot(z, cross(u0, u)), dot(u0, u)));
}
export function upForRoll(pos, aim, rollDeg) {
  const z = unit(sub(aim, pos));
  const u0 = unit(sub([0, 0, 1], scale(z, z[2])));
  const t = rad(rollDeg);
  return add(scale(u0, Math.cos(t)), scale(cross(z, u0), Math.sin(t)));
}

// ------------------------------------------------------------------ world
export class World {
  constructor(geom) {
    this.g = geom;
    const w = geom.wheel;
    this.R = w.R;
    this.solids = w.solids.map((poly) => Float64Array.from(poly.flat()));
    let amin = Infinity, amax = -Infinity, rmax = 0;
    for (const s of this.solids) for (let i = 0; i < s.length; i += 2) {
      amin = Math.min(amin, s[i]); amax = Math.max(amax, s[i]); rmax = Math.max(rmax, s[i + 1]);
    }
    Object.assign(this, { amin, amax, rmax2: rmax * rmax });
    const H = geom.half_length;
    this.x0 = -H; this.x1 = H;
    this.rails = geom.rails.map((r) => {
      const p = Float64Array.from(r.poly.flat());
      let y0 = Infinity, y1 = -Infinity, z0 = Infinity, z1 = -Infinity;
      for (let i = 0; i < p.length; i += 2) {
        y0 = Math.min(y0, p[i]); y1 = Math.max(y1, p[i]); z0 = Math.min(z0, p[i + 1]); z1 = Math.max(z1, p[i + 1]);
      }
      return { p, y0, y1, z0, z1, side: r.side, part: r.part };
    });
    this.boxes = [];
    const t = geom.ties, [sx, sy, sz] = t.size, [px, py, pz] = t.plate;
    for (let k = Math.ceil(-H / t.spacing); k * t.spacing <= H; k++) {
      const x = k * t.spacing;
      this.boxes.push({ min: [x - sx / 2, -sy / 2, t.z_top - sz], max: [x + sx / 2, sy / 2, t.z_top], cls: TIE });
      for (const s of [-1, 1]) {
        const yc = s * -geom.rail_centre_y;
        this.boxes.push({ min: [x - px / 2, yc - py / 2, t.z_top], max: [x + px / 2, yc + py / 2, t.z_top + pz], cls: RAIL });
      }
    }
    this.ballastZ = geom.ballast_z;
    this.profile = w.profile; // [[a, r], ...]
    this.setWheel(0, 0);
  }

  // wheel origin: on the axle, in the flange back-face plane
  setWheel(x, lateral) {
    const o = this.g.wheel.origin0;
    this.o = [o[0] + x, o[1] + lateral, o[2]];
  }

  // radius of the measured profile at lateral position a (m), clamped to its ends
  profileR(a) {
    const p = this.profile;
    if (a <= p[0][0]) return p[0][1];
    for (let i = 1; i < p.length; i++) {
      if (p[i][0] >= a && p[i - 1][0] <= a) {
        const t = (a - p[i - 1][0]) / (p[i][0] - p[i - 1][0] || 1);
        return p[i - 1][1] + t * (p[i][1] - p[i - 1][1]);
      }
    }
    return p[p.length - 1][1];
  }

  // A point on the wheel surface `h` m above the rail top, `a` m from the flange back
  // face, on the side of the wheel facing x = towardsX (where the cameras are).
  aimOnWheel(h, a, towardsX) {
    const o = this.o, r = this.profileR(a);
    const c = Math.max(-1, Math.min(1, (o[2] - h) / r));
    const s = Math.sqrt(1 - c * c) * (towardsX >= o[0] ? 1 : -1);
    return [o[0] + r * s, o[1] - a, o[2] - r * c];
  }

  // ---------------------------------------------------------------- ray casting
  // First hit of P + t d (|d| = 1) with t in (1e-6, tmax) against the classes in `mask`.
  cast(P, d, tmax = 10, mask = ALL) {
    let best = tmax, cls = 0;
    if (mask & WHEEL) { const t = this.hitWheel(P, d, best); if (t < best) { best = t; cls = WHEEL; } }
    if (mask & RAIL) { const t = this.hitRails(P, d, best); if (t < best) { best = t; cls = RAIL; } }
    if (mask & (TIE | RAIL)) {
      for (const b of this.boxes) {
        if (!(mask & b.cls)) continue;
        const t = hitBox(P, d, b, best);
        if (t < best) { best = t; cls = b.cls; }
      }
    }
    if (mask & BALLAST && d[2] < -1e-9) {
      const t = (this.ballastZ - P[2]) / d[2];
      if (t > 1e-6 && t < best) { best = t; cls = BALLAST; }
    }
    return cls ? { t: best, cls } : null;
  }

  hitWheel(P, d, tmax) {
    const o = this.o;
    const px = P[0] - o[0], py = P[1] - o[1], pz = P[2] - o[2];
    const A0 = -py, At = -d[1];
    const c0 = px * px + pz * pz, c1 = px * d[0] + pz * d[2], c2 = d[0] * d[0] + d[2] * d[2];
    let lo = 1e-6, hi = tmax;
    if (c2 > 1e-12) {
      const disc = c1 * c1 - c2 * (c0 - this.rmax2);
      if (disc < 0) return Infinity;
      const sq = Math.sqrt(disc);
      lo = Math.max(lo, (-c1 - sq) / c2); hi = Math.min(hi, (-c1 + sq) / c2);
    } else if (c0 > this.rmax2) return Infinity;
    if (Math.abs(At) > 1e-12) {
      const ta = (this.amin - A0) / At, tb = (this.amax - A0) / At;
      lo = Math.max(lo, Math.min(ta, tb)); hi = Math.min(hi, Math.max(ta, tb));
    } else if (A0 < this.amin || A0 > this.amax) return Infinity;
    if (lo > hi) return Infinity;
    const aLo = Math.min(A0 + At * lo, A0 + At * hi) - 1e-9, aHi = Math.max(A0 + At * lo, A0 + At * hi) + 1e-9;
    let best = tmax;
    const tryT = (t, a1, da) => {
      if (t > 1e-6 && t < best) {
        const s = (A0 + At * t - a1) / da;
        if (s >= -1e-9 && s <= 1 + 1e-9) return true;
      }
      return false;
    };
    for (const S of this.solids) {
      const n = S.length / 2;
      for (let i = 0; i < n; i++) {
        const j = (i + 1) % n;
        const a1 = S[2 * i], r1 = S[2 * i + 1], a2 = S[2 * j], r2 = S[2 * j + 1];
        if (Math.max(a1, a2) < aLo || Math.min(a1, a2) > aHi) continue;
        const da = a2 - a1;
        if (Math.abs(da) > 1e-12) {
          const k = (r2 - r1) / da, alpha = r1 + k * (A0 - a1), beta = k * At;
          const qa = c2 - beta * beta, qb = 2 * (c1 - alpha * beta), qc = c0 - alpha * alpha;
          let t1, t2;
          if (Math.abs(qa) < 1e-14) {
            if (Math.abs(qb) < 1e-14) continue;
            t1 = t2 = -qc / qb;
          } else {
            const disc = qb * qb - 4 * qa * qc;
            if (disc < 0) continue;
            const sq = Math.sqrt(disc);
            t1 = (-qb - sq) / (2 * qa); t2 = (-qb + sq) / (2 * qa);
          }
          for (const t of [t1, t2]) if (tryT(t, a1, da) && alpha + beta * t >= 0) best = t;
        } else if (Math.abs(At) > 1e-12) {
          const t = (a1 - A0) / At;
          if (t > 1e-6 && t < best) {
            const r = Math.sqrt(Math.max(0, c0 + 2 * c1 * t + c2 * t * t));
            if (r >= Math.min(r1, r2) - 1e-9 && r <= Math.max(r1, r2) + 1e-9) best = t;
          }
        }
      }
    }
    return best < tmax ? best : Infinity;
  }

  hitRails(P, d, tmax) {
    let best = tmax;
    const d1 = d[1], d2 = d[2];
    if (Math.abs(d1) < 1e-12 && Math.abs(d2) < 1e-12) return Infinity;
    for (const r of this.rails) {
      // 2D slab cull on the polygon's (y, z) box
      let lo = 1e-6, hi = best;
      for (const [p, dd, a, b] of [[P[1], d1, r.y0, r.y1], [P[2], d2, r.z0, r.z1]]) {
        if (Math.abs(dd) < 1e-12) { if (p < a || p > b) { lo = 1; hi = 0; } continue; }
        const ta = (a - p) / dd, tb = (b - p) / dd;
        lo = Math.max(lo, Math.min(ta, tb)); hi = Math.min(hi, Math.max(ta, tb));
      }
      if (lo > hi) continue;
      const q = r.p, n = q.length / 2;
      for (let i = 0; i < n; i++) {
        const j = (i + 1) % n;
        const y1 = q[2 * i], z1 = q[2 * i + 1], ey = q[2 * j] - y1, ez = q[2 * j + 1] - z1;
        const den = d1 * ez - d2 * ey;
        if (Math.abs(den) < 1e-14) continue;
        const wy = y1 - P[1], wz = z1 - P[2];
        const t = (wy * ez - wz * ey) / den;
        if (t <= 1e-6 || t >= best) continue;
        const s = (wy * d2 - wz * d1) / den;
        if (s < 0 || s > 1) continue;
        const x = P[0] + t * d[0];
        if (x < this.x0 || x > this.x1) continue;
        best = t;
      }
    }
    return best < tmax ? best : Infinity;
  }
}

function hitBox(P, d, b, tmax) {
  let lo = 1e-6, hi = tmax;
  for (let k = 0; k < 3; k++) {
    if (Math.abs(d[k]) < 1e-12) { if (P[k] < b.min[k] || P[k] > b.max[k]) return Infinity; continue; }
    const ta = (b.min[k] - P[k]) / d[k], tb = (b.max[k] - P[k]) / d[k];
    lo = Math.max(lo, Math.min(ta, tb)); hi = Math.min(hi, Math.max(ta, tb));
    if (lo > hi) return Infinity;
  }
  return lo;
}

// ------------------------------------------------------------------ cameras
// cam: {pos, R (columns), W, H, fx, fy, cx, cy, D}. OpenCV model with k1 k2 p1 p2 k3.
export function project(cam, X) {
  const v = sub(X, cam.pos), R = cam.R;
  const xc = dot(R[0], v), yc = dot(R[1], v), zc = dot(R[2], v);
  if (zc <= 1e-6) return null;
  const x = xc / zc, y = yc / zc;
  const [k1 = 0, k2 = 0, p1 = 0, p2 = 0, k3 = 0] = cam.D;
  const r2 = x * x + y * y, rad_ = 1 + r2 * (k1 + r2 * (k2 + r2 * k3));
  const xd = x * rad_ + 2 * p1 * x * y + p2 * (r2 + 2 * x * x);
  const yd = y * rad_ + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y;
  return [cam.fx * xd + cam.cx, cam.fy * yd + cam.cy, zc];
}
export const inImage = (cam, uv) => uv && uv[0] >= 0 && uv[0] < cam.W && uv[1] >= 0 && uv[1] < cam.H;

// Is X visible from the camera: in the image and nothing in `mask` closer than X - tol.
export function sees(world, cam, X, mask = ALL, tol = 0.002) {
  if (!inImage(cam, project(cam, X))) return false;
  const v = sub(X, cam.pos), dist = norm(v);
  return !world.cast(cam.pos, scale(v, 1 / dist), dist - tol, mask);
}

// ------------------------------------------------------------------ lasers
export function patternTangents(pat, sigmaMrad) {
  const out = [];
  if (pat.type === "grid") {
    const pitch = pat.fan_deg
      ? [pat.fan_deg[0] / Math.max(pat.cols - 1, 1), pat.fan_deg[1] / Math.max(pat.rows - 1, 1)]
      : pat.pitch_deg;
    for (let r = 0; r < pat.rows; r++) for (let c = 0; c < pat.cols; c++) {
      out.push([Math.tan(rad((c - (pat.cols - 1) / 2) * pitch[0])), Math.tan(rad((r - (pat.rows - 1) / 2) * pitch[1]))]);
    }
  } else if (pat.type === "line") {
    const spacing = pat.spacing_mrad ? pat.spacing_mrad * 1e-3 : (sigmaMrad * 1e-3) / 2;
    const half = rad(pat.fan_deg) / 2, n = Math.ceil((2 * half) / spacing) + 1;
    const c = Math.cos(rad(pat.angle_deg || 0)), s = Math.sin(rad(pat.angle_deg || 0));
    for (let i = 0; i < n; i++) {
      const t = Math.tan(-half + (2 * half * i) / (n - 1));
      out.push([t * c, t * s]);
    }
  }
  return out;
}

// ------------------------------------------------------------------ profile coverage
// Which points of the measured profile each camera sees in the wheel section at
// `phiDeg` about the axle (0 = straight down, + towards +x): in the image, facing the
// camera, not hidden by the wheel or rails (same rule as synthrig.layout.profile_coverage).
export function profileCoverage(world, cams, phiDeg) {
  const P = world.profile, n = P.length, o = world.o, phi = rad(phiDeg);
  const sp = Math.sin(phi), cp = Math.cos(phi);
  const pts = [], nrm = [], arc = new Float64Array(n);
  let s = [0];
  for (let i = 1; i < n; i++) s.push(s[i - 1] + Math.hypot(P[i][0] - P[i - 1][0], P[i][1] - P[i - 1][1]));
  for (let i = 0; i < n; i++) {
    const i0 = Math.max(0, i - 1), i1 = Math.min(n - 1, i + 1);
    const ta = P[i1][0] - P[i0][0], tr = P[i1][1] - P[i0][1], tl = Math.hypot(ta, tr) || 1;
    const na = -tr / tl, nr = ta / tl; // outward for this traversal direction
    const [a, r] = P[i];
    pts.push([o[0] + r * sp, o[1] - a, o[2] - r * cp]);
    nrm.push([nr * sp, -na, -nr * cp]);
    arc[i] = (s[i1] - s[i0]) / (i1 - i0 || 1);
  }
  const seen = {};
  for (const cam of cams) {
    const m = new Uint8Array(n);
    for (let i = 0; i < n; i++) {
      if (dot(nrm[i], sub(cam.pos, pts[i])) <= 0) continue;
      m[i] = sees(world, cam, pts[i], WHEEL | RAIL, 0.004) ? 1 : 0;
    }
    seen[cam.name] = m;
  }
  const total = arc.reduce((x, y) => x + y, 0); // = profile length (fractions use the same weights)
  return { phiDeg, a: P.map((p) => p[0]), dr: P.map((p) => p[1] - world.R), arc, seen, total };
}

export function sectionAngle(world, aim) {
  const o = world.o;
  return deg(Math.atan2(aim[0] - o[0], o[2] - aim[2]));
}
