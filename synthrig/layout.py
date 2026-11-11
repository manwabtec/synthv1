"""Fast layout geometry for the GUI: coarse scene meshes, camera frustums, laser hits and
camera-view projections, all in the track frame. numpy only (no bpy, no plotting).

Approximations (fine for placing devices, not for ground truth): coarse wheel mesh, no
occlusion in camera coverage, lasers hit the coarse meshes. The render is exact.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .assets.wheelrail import (WheelRailParams, ballast_level, rail_sections, revolve, ties,
                               _contour, wheel_origin, wheel_sections, wheel_to_track)
from .opticsrig import CameraModel, LaserModel, RigModel
from .opticsrig.transforms import transform_points


@dataclass
class Mesh:
    name: str
    cls: str
    V: np.ndarray  # (N,3) track frame, m
    F: np.ndarray  # (M,3) triangles


def _tris(quads: np.ndarray) -> np.ndarray:
    return np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])


def _resample(poly: np.ndarray, step: float) -> np.ndarray:
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(seg)]
    t = np.linspace(0, s[-1], max(int(s[-1] / step) + 1, 2))
    return np.stack([np.interp(t, s, poly[:, 0]), np.interp(t, s, poly[:, 1])], 1)


def _box(c, size) -> tuple[np.ndarray, np.ndarray]:
    h = np.asarray(size) / 2
    V = np.asarray(c) + h * np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                                      [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]])
    Q = np.array([(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)])
    return V, _tris(Q)


def scene_meshes(p: WheelRailParams, segments: int = 48, step: float = 0.006) -> list[Mesh]:
    """Coarse triangle meshes of the wheel parts, rails, ties and ballast."""
    out = []
    T = wheel_to_track(p)
    for cls, poly in wheel_sections(p).items():
        V, Q = revolve(_resample(poly, step), segments)
        out.append(Mesh(cls, cls, transform_points(T, V), _tris(Q)))
    x0, x1 = p.wheel_x_m - p.track_half_length_m, p.wheel_x_m + p.track_half_length_m
    for side, part, poly in rail_sections(p):
        n = len(poly)
        V = np.concatenate([np.c_[np.full(n, x0), poly], np.c_[np.full(n, x1), poly]])
        Q = np.array([(i, (i + 1) % n, n + (i + 1) % n, n + i) for i in range(n)])
        out.append(Mesh(f"rail_{side}_{part}", f"rail_{part}", V, _tris(Q)))
    for i, (c, s) in enumerate(ties(p)):
        V, F = _box(c, s)
        out.append(Mesh(f"tie_{i}", "tie", V, F))
    z = ballast_level(p)
    V = np.array([[x0 - 1, -2.5, z], [x1 + 1, -2.5, z], [x1 + 1, 2.5, z], [x0 - 1, 2.5, z]])
    out.append(Mesh("ballast", "ballast", V, np.array([[0, 1, 2], [0, 2, 3]])))
    return out


# --------------------------------------------------------------------------- rays
def ray_hits(origin: np.ndarray, dirs: np.ndarray, meshes: list[Mesh], max_dist: float = 10.0):
    """First hit of each ray (Moller-Trumbore). `origin` is (3,) or one per ray (N,3).

    Only triangles whose bounding box meets the rays' bounding box are tested.
    Returns (points (N,3), NaN where nothing is hit; hit class per ray or None).
    """
    origin = np.broadcast_to(np.asarray(origin, float), dirs.shape)
    tris = np.concatenate([m.V[m.F] for m in meshes])  # (T,3,3)
    owner = np.concatenate([np.full(len(m.F), k) for k, m in enumerate(meshes)])
    ends = origin + dirs * max_dist
    lo, hi = np.minimum(origin, ends).min(0), np.maximum(origin, ends).max(0)
    keep = (tris.max(1) >= lo).all(1) & (tris.min(1) <= hi).all(1)
    tris, owner = tris[keep], owner[keep]
    best_t = np.full(len(dirs), np.inf)
    best_k = np.full(len(dirs), -1)
    if len(tris):
        v0, e1, e2 = tris[:, 0], tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]
        for i0 in range(0, len(dirs), 64):  # chunk rays to bound memory
            d = dirs[i0:i0 + 64, None, :]
            p = np.cross(d, e2[None])
            det = np.einsum("rtk,tk->rt", p, e1)
            ok = np.abs(det) > 1e-12
            inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
            s = origin[i0:i0 + 64, None, :] - v0[None]
            u = np.einsum("rtk,rtk->rt", s, p) * inv
            q = np.cross(s, e1[None])
            v = np.einsum("rk,rtk->rt", d[:, 0], q) * inv
            t = np.einsum("tk,rtk->rt", e2, q) * inv
            hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-6) & (t < max_dist)
            t = np.where(hit, t, np.inf)
            j = np.argmin(t, axis=1)
            best_t[i0:i0 + 64] = t[np.arange(len(j)), j]
            best_k[i0:i0 + 64] = np.where(np.isfinite(best_t[i0:i0 + 64]), owner[j], -1)
    pts = origin + dirs * np.where(np.isfinite(best_t), best_t, np.nan)[:, None]
    return pts, [meshes[k].cls if k >= 0 else None for k in best_k]


def laser_rays(L: LaserModel, max_rays: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Projector origin and (subsampled) unit ray directions in the track frame."""
    dirs = L.directions
    if len(dirs) > max_rays:
        dirs = dirs[np.linspace(0, len(dirs) - 1, max_rays).round().astype(int)]
    return L.T_rig_proj[:3, 3], dirs @ L.T_rig_proj[:3, :3].T


def laser_fan_outline(L: LaserModel, length: float) -> np.ndarray:
    """Outline of the pattern's angular extent at `length` m: (K,3) points, closed loop."""
    t = L.tangents
    if np.ptp(t[:, 0]) < 1e-9 or np.ptp(t[:, 1]) < 1e-9:  # a line: its two end rays
        i0, i1 = np.argmin(t[:, 0] + t[:, 1]), np.argmax(t[:, 0] + t[:, 1])
        corners = t[[i0, i1]]
    else:
        lo, hi = t.min(0), t.max(0)
        corners = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])
    d = np.c_[corners, np.ones(len(corners))]
    d = d / np.linalg.norm(d, axis=1, keepdims=True) * length
    return transform_points(L.T_rig_proj, d)


# --------------------------------------------------------------------------- cameras
def camera_frustum(cam: CameraModel, depth: float) -> tuple[np.ndarray, np.ndarray]:
    """(centre (3,), image-corner points (4,3) at `depth` m along the optical axis)."""
    w, h = cam.size
    px = np.array([[-0.5, -0.5], [w - 0.5, -0.5], [w - 0.5, h - 0.5], [-0.5, h - 0.5]])
    n = cam.undistort_to_normalized(px)
    X = np.c_[n, np.ones(4)] * depth
    return cam.T_rig_cam[:3, 3], transform_points(cam.T_rig_cam, X)


def project(cam: CameraModel, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pixel coords (N,2) and in-image mask for track-frame points (ignores occlusion)."""
    Xc = transform_points(np.linalg.inv(cam.T_rig_cam), X)
    front = Xc[:, 2] > 1e-6
    uv = np.full((len(X), 2), np.nan)
    if front.any():
        uv[front] = cam.project(Xc[front])
    w, h = cam.size
    inside = front & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    return uv, inside


def hfov_deg(cam: CameraModel) -> float:
    return float(np.degrees(2 * np.arctan(cam.size[0] / 2 / cam.K[0, 0])))


@dataclass
class Layout:
    meshes: list[Mesh]
    laser_hits: dict[str, tuple[np.ndarray, list]]  # laser -> (points, hit classes)
    coverage: dict[str, dict[str, float]]  # camera -> laser -> fraction of hits in image


def compute(rig: RigModel, p: WheelRailParams) -> Layout:
    """Everything the layout view draws, for a `frame: track` rig."""
    if rig.spec.frame != "track":
        raise ValueError("the layout view needs a `frame: track` rig")
    meshes = scene_meshes(p)
    hits = {}
    for name, L in rig.lasers.items():
        o, d = laser_rays(L)
        hits[name] = ray_hits(o, d, meshes)
    cov = {}
    for cname, cam in rig.cameras.items():
        cov[cname] = {}
        for lname, (pts, _cls) in hits.items():
            ok = ~np.isnan(pts[:, 0])
            cov[cname][lname] = float(project(cam, pts[ok])[1].mean()) if ok.any() else 0.0
    return Layout(meshes, hits, cov)


# --------------------------------------------------------------------------- profile coverage
@dataclass
class ProfileCoverage:
    """Which points of the wheel's outer profile each camera sees, in one section plane."""

    angle_deg: float          # section angle about the axle, 0 = bottom, + towards +x
    a: np.ndarray             # (N,) lateral from flange back face (m)
    dr: np.ndarray            # (N,) radial offset from the taping line (m)
    arc: np.ndarray           # (N,) arc length weight of each point (m)
    seen: dict[str, np.ndarray]  # camera -> (N,) bool


def section_angle(p: WheelRailParams, aim) -> float:
    """Section angle (deg) of the wheel plane through `aim`: 0 = straight down, + = +x."""
    o = wheel_origin(p)
    return float(np.degrees(np.arctan2(aim[0] - o[0], o[2] - aim[2])))


def profile_coverage(rig: RigModel, p: WheelRailParams, meshes: list[Mesh], angle_deg: float,
                     step: float = 0.002) -> ProfileCoverage:
    """For the measured wheel profile in the section at `angle_deg`:
    seen by a camera = in its image, facing it, and not occluded by the wheel or rails."""
    c = _contour(p.wheel_profile)  # the measured profile, flange back face -> field side
    R = p.wheel_section.tape_diameter_mm / 2000.0
    poly = _resample(np.stack([c.a, R + c.dr], 1), step)
    a, r = poly[:, 0], poly[:, 1]
    tng = np.gradient(poly, axis=0)
    # (-t_r, t_a) points out of the wheel for this traversal direction (back face up, over
    # the flange tip, along the tread, down the rim face)
    nrm = np.stack([-tng[:, 1], tng[:, 0]], 1)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
    phi = np.radians(angle_deg)
    o = wheel_origin(p)
    X = o + np.stack([r * np.sin(phi), -a, -r * np.cos(phi)], 1)
    N3 = np.stack([nrm[:, 1] * np.sin(phi), -nrm[:, 0], -nrm[:, 1] * np.cos(phi)], 1)
    arc = np.gradient(np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(poly, axis=0), axis=1))])
    occluders = [m for m in meshes if m.cls.startswith(("wheel_", "rail_"))]
    seen = {}
    for name, cam in rig.cameras.items():
        c = cam.T_rig_cam[:3, 3]
        _, inside = project(cam, X)
        facing = np.einsum("ij,ij->i", N3, c - X) > 0
        ok = inside & facing
        if ok.any():
            d = X[ok] - c
            dist = np.linalg.norm(d, axis=1)
            hits, _ = ray_hits(c, d / dist[:, None], occluders, max_dist=float(dist.max()) + 0.01)
            first = np.linalg.norm(hits - c, axis=1)
            ok[np.nonzero(ok)[0]] = ~(first < dist - 0.004)  # NaN (no hit) compares False -> visible
        seen[name] = ok
    return ProfileCoverage(angle_deg, a, r - R, arc, seen)


@dataclass
class CameraSet:
    """Cameras on one side of the rail (a stereo pair = one rod)."""

    label: str            # "outside" (field side) or "inside" (gauge side)
    cameras: list[str]


def camera_sets(rig: RigModel, p: WheelRailParams) -> list[CameraSet]:
    """Camera groups by side of the right rail: rods first, then stereo pairs, then any
    remaining camera on its own."""
    from .assets.wheelrail import _rail

    _, dims = _rail(p.rail_profile)
    rail_c = -(p.gauge_mm * 1e-3 / 2 + dims["gauge_face_u"])
    sets, used = [], set()
    groups = [list(r.cameras) for r in rig.spec.rods]
    groups += [[s.left, s.right] for s in rig.spec.stereo_pairs if not any(s.left in g for g in groups)]
    groups += [[c] for c in rig.cameras if not any(c in g for g in groups)]
    for g in groups:
        y = float(np.mean([rig.cameras[c].T_rig_cam[1, 3] for c in g]))
        sets.append(CameraSet("outside" if y < rail_c else "inside", g))
        used.update(g)
    return sets
