"""Wheel-on-rail geometry in the track frame, shared by the Blender scene and the layout
view. No bpy.

Track frame (docs/TASK_scenes.md section 1): x along the track, y lateral (+ left), z up,
metres; origin on the track centreline at top-of-rail height (rail tops at z = 0).
The wheel stands on the RIGHT rail (y < 0); its field side faces -y.

Dimensions not from a supplied profile are PLACEHOLDERS (marked in field descriptions).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .profiles import WheelSection, load_rail_profile, load_wheel_contour, rail_dimensions, wheel_parts


class WheelRailParams(BaseModel):
    """Scene parameters for one wheel on track. Lengths in m unless the name says mm."""

    model_config = ConfigDict(extra="forbid")

    # --- what the GUI exposes first
    wheel_x_m: float = Field(0.0, description="wheel centre position along the track (x)")
    wheel_rot_deg: float = Field(0.0, description="wheel rotation about its axle")
    lateral_shift_m: float = Field(0.0, description="wheelset lateral shift (+ = towards +y)")
    wheel_profile: str = "AAR_2A_ref_wide"
    rail_profile: str = "rail_136re_placeholder"
    wheel_section: WheelSection = WheelSection()
    speed_mps: float = Field(0.0, description="sequence mode: wheel rolls along +x at this speed (no slip)")
    surface_modifiers: list[dict] = Field(default_factory=list, description="hook for tread defects "
                                          "(flats, shelling, ...); not implemented yet, must stay empty")
    # --- track (PLACEHOLDER values typical of North American freight track)
    gauge_mm: float = Field(1435.1, description="between gauge faces, 16 mm below top of rail")
    back_to_back_mm: float = Field(1346.2, description="PLACEHOLDER: flange back face to back face")
    track_half_length_m: float = Field(2.0, gt=0, description="track built from x-L to x+L")
    tie_spacing_mm: float = Field(495.0, gt=0, description="PLACEHOLDER")
    tie_size_mm: list[float] = Field([229.0, 2591.0, 178.0], description="PLACEHOLDER [along x, across y, height]")
    tie_plate_mm: list[float] = Field([197.0, 356.0, 20.0], description="PLACEHOLDER [x, y, thickness]")
    # --- appearance
    steel_rgb: list[float] = [0.35, 0.33, 0.30]
    rust: float = Field(0.2, ge=0, le=1)
    roughness: float = Field(0.45, ge=0, le=1)
    ambient: float = Field(0.08, ge=0)
    noise_seed: float = 0.0


# --------------------------------------------------------------------------- rails
@lru_cache(maxsize=8)
def _rail(name: str):
    parts = load_rail_profile(name)
    return parts, rail_dimensions(parts)


def rail_sections(p: WheelRailParams) -> list[tuple[str, str, np.ndarray]]:
    """[(side, part, polygon (N,2) as (y, z))] for both rails in the track frame."""
    parts, dims = _rail(p.rail_profile)
    G = p.gauge_mm * 1e-3
    out = []
    for side, s in (("right", -1.0), ("left", 1.0)):
        y_c = s * (G / 2 + dims["gauge_face_u"])  # rail centreline; gauge side points to y = 0
        for part, poly in parts.items():
            y = y_c - s * poly[:, 0]
            z = poly[:, 1] - dims["height"]
            out.append((side, part, np.stack([y, z], 1)))
    return out


def rail_height(p: WheelRailParams) -> float:
    return _rail(p.rail_profile)[1]["height"]


def ties(p: WheelRailParams) -> list[tuple[np.ndarray, np.ndarray]]:
    """[(centre xyz, size xyz)] of tie boxes, aligned to the wheel at x = 0 grid."""
    sx, sy, sz = (v * 1e-3 for v in p.tie_size_mm)
    spacing = p.tie_spacing_mm * 1e-3
    z_top = -rail_height(p) - p.tie_plate_mm[2] * 1e-3
    lo, hi = p.wheel_x_m - p.track_half_length_m, p.wheel_x_m + p.track_half_length_m
    xs = np.arange(np.ceil(lo / spacing), np.floor(hi / spacing) + 1) * spacing
    return [(np.array([x, 0.0, z_top - sz / 2]), np.array([sx, sy, sz])) for x in xs]


def tie_plates(p: WheelRailParams) -> list[tuple[np.ndarray, np.ndarray]]:
    _, dims = _rail(p.rail_profile)
    px, py, pz = (v * 1e-3 for v in p.tie_plate_mm)
    G = p.gauge_mm * 1e-3
    z = -dims["height"] - pz / 2
    out = []
    for c, _size in ties(p):
        for s in (-1.0, 1.0):
            out.append((np.array([c[0], s * (G / 2 + dims["gauge_face_u"]), z]), np.array([px, py, pz])))
    return out


def ballast_level(p: WheelRailParams) -> float:
    """Ballast surface height: halfway up the ties."""
    return -rail_height(p) - p.tie_plate_mm[2] * 1e-3 - p.tie_size_mm[2] * 1e-3 / 2


# --------------------------------------------------------------------------- wheel
@lru_cache(maxsize=8)
def _contour(name: str):
    return load_wheel_contour(name)


def wheel_sections(p: WheelRailParams) -> dict[str, np.ndarray]:
    """{class: polyline (a, r)} of the wheel cross-section (see profiles.wheel_parts)."""
    return wheel_parts(_contour(p.wheel_profile), p.wheel_section)


def wheel_origin(p: WheelRailParams) -> np.ndarray:
    """Wheel local origin in the track frame: on the axle, in the flange back-face plane.

    The taping-line circle touches the rail top (z = 0); the flange back face is at
    y = -back_to_back/2 + lateral shift.
    """
    R = p.wheel_section.tape_diameter_mm / 2000.0
    return np.array([p.wheel_x_m, -p.back_to_back_mm / 2000.0 + p.lateral_shift_m, R])


def revolve(poly: np.ndarray, segments: int = 256, closed: bool = False):
    """Revolve a section polyline (a, r) about the wheel axle.

    Local wheel frame: x along the track, y = -a (field side is -y), z up; the axle is
    the local y axis. Returns (vertices (M,3), quads (K,4)) as numpy arrays.
    """
    th = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    a, r = poly[:, 0], poly[:, 1]
    V = np.stack([np.outer(r, np.sin(th)), np.repeat(-a[:, None], segments, 1), np.outer(-r, np.cos(th))], -1)
    n = len(poly)
    rows = range(n if closed else n - 1)
    F = [(i * segments + k, i * segments + (k + 1) % segments,
          ((i + 1) % n) * segments + (k + 1) % segments, ((i + 1) % n) * segments + k)
         for i in rows for k in range(segments)]
    return V.reshape(-1, 3), np.asarray(F, np.int64)


def wheel_to_track(p: WheelRailParams) -> np.ndarray:
    """4x4 T_track_wheel: rotation about the axle (local y) by wheel_rot_deg, then origin."""
    t = np.deg2rad(p.wheel_rot_deg)
    c, s = np.cos(t), np.sin(t)
    T = np.eye(4)
    T[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    T[:3, 3] = wheel_origin(p)
    return T


def rolled(p: WheelRailParams, t_s: float) -> WheelRailParams:
    """The wheel `t_s` seconds later, rolling without slip along +x at p.speed_mps."""
    dx = p.speed_mps * t_s
    R = p.wheel_section.tape_diameter_mm / 2000.0
    return p.model_copy(update={"wheel_x_m": p.wheel_x_m + dx,
                                "wheel_rot_deg": p.wheel_rot_deg + float(np.degrees(dx / R))})


def key_points(p: WheelRailParams) -> dict[str, list[float]]:
    """Named aim points in the track frame (for look_at in rig files and the GUI)."""
    o = wheel_origin(p)
    R = p.wheel_section.tape_diameter_mm / 2000.0
    c = _contour(p.wheel_profile)
    tip = int(np.argmax(c.dr))
    tread = np.nonzero(c.dr[tip:] <= 0)[0]  # taping line: where the tread crosses dr = 0
    tape_a = float(c.a[tip + tread[0]]) if len(tread) else float(c.a[-1])
    _, dims = _rail(p.rail_profile)
    rail_c = -(p.gauge_mm * 1e-3 / 2 + dims["gauge_face_u"])
    pts = {
        "wheel_center": o,
        "tread_contact": [o[0], o[1] - tape_a, 0.0],
        "flange_tip_bottom": [o[0], o[1] - c.a[tip], o[2] - R - c.dr[tip]],
        "rail_head_top": [o[0], rail_c, 0.0],
    }
    return {k: [round(float(v), 4) for v in xyz] for k, xyz in pts.items()}
