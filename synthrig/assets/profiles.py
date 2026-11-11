"""Wheel and rail cross-section profiles from CSV (assets/profiles/). No bpy.

Wheel profiles (assets/profiles/wheels/<name>.csv) are tread + flange contours:
  column 1 = lateral position a (mm), 0 at the flange back face, + towards the field side;
  column 2 = radial offset dr (mm) from the taping-line radius, + outward.
Points may be stored in any order; they are chained into one curve by nearest neighbour.
The rest of the wheel cross-section (rim, plate, hub) is completed from WheelSection
parameters, which are PLACEHOLDERS until real drawings are supplied.

Rail profiles (assets/profiles/rails/<name>.csv): columns part, u_mm, h_mm with one closed
polygon per part (head, web, foot); u = lateral from the rail centreline (+ towards the
gauge side), h = height above the rail base.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from ..paths import CODE_ROOT

PROFILE_DIR = CODE_ROOT / "assets" / "profiles"


def available(kind: str) -> list[str]:
    """Profile names (file stems) in assets/profiles/<kind>/ ('wheels' or 'rails')."""
    return sorted(p.stem for p in (PROFILE_DIR / kind).glob("*.csv"))


def _rows(path: Path) -> list[list[str]]:
    with path.open(newline="") as f:
        return [r for r in csv.reader(f) if r and not r[0].lstrip().startswith("#")]


def chain(points: np.ndarray, start: int) -> np.ndarray:
    """Order points into one polyline by repeatedly stepping to the nearest unused point."""
    left = np.ones(len(points), bool)
    order = [start]
    left[start] = False
    for _ in range(len(points) - 1):
        d = np.linalg.norm(points[left] - points[order[-1]], axis=1)
        nxt = np.nonzero(left)[0][int(np.argmin(d))]
        order.append(nxt)
        left[nxt] = False
    return points[order]


# --------------------------------------------------------------------------- wheel
@dataclass
class WheelContour:
    """Tread + flange contour in metres, ordered from the flange back face to the field side."""

    name: str
    a: np.ndarray   # lateral from flange back face (m), + towards field side
    dr: np.ndarray  # radial offset from taping-line radius (m), + outward

    @property
    def points(self) -> np.ndarray:
        return np.stack([self.a, self.dr], 1)


def load_wheel_contour(name: str, units: str = "mm") -> WheelContour:
    """Load assets/profiles/wheels/<name>.csv and order it flange back -> field side."""
    path = PROFILE_DIR / "wheels" / f"{name}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; available: {available('wheels')}")
    pts = np.array([[float(r[0]), float(r[1])] for r in _rows(path)])
    scale = {"mm": 1e-3, "m": 1.0}[units]
    pts = chain(pts * scale, int(np.argmax(pts[:, 0])))[::-1]  # start: field-side end, then reverse
    steps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    if steps.max() > 20 * np.median(steps):
        raise ValueError(f"{path.name}: contour has a gap of {steps.max() * 1e3:.1f} mm after ordering")
    return WheelContour(name, pts[:, 0], pts[:, 1])


class WheelSection(BaseModel):
    """Wheel dimensions around the measured contour. All PLACEHOLDERS except where noted.

    Lengths in mm; lateral positions measured from the flange back face (+ field side).
    """

    tape_diameter_mm: float = Field(914.4, gt=0, description="taping-line diameter; 914.4 = nominal 36 in")
    flange_split_mm: float = Field(5.0, description="contour above this dr (gauge side of the "
                                   "highest point) is labelled flange, below it tread")
    rim_face_slope: float = Field(1.0, gt=0, description="|d dr / d a| above which the field-side "
                                  "end of the contour is labelled rim face (chamfer)")
    rim_thickness_mm: float = Field(50.0, gt=0, description="taping line to rim inner surface (radial)")
    plate_center_mm: float = Field(70.0, description="lateral position of the plate centre")
    plate_thickness_mm: float = Field(25.0, gt=0)
    hub_outer_radius_mm: float = Field(170.0, gt=0)
    bore_radius_mm: float = Field(90.0, gt=0)
    hub_gauge_side_mm: float = Field(-40.0, description="lateral position of the hub's gauge-side face")
    hub_field_side_mm: float = Field(140.0, description="lateral position of the hub's field-side face")


def wheel_parts(contour: WheelContour, s: WheelSection) -> dict[str, np.ndarray]:
    """Cross-section polylines (a, r) in metres per segmentation class.

    r is the absolute radius. Concatenated in the order returned they form one closed
    outline: flange back face -> flange -> tread -> rim face -> rim/plate/hub inner side.
    """
    R = s.tape_diameter_mm / 2000.0
    a, dr = contour.a, contour.dr
    r = R + dr
    tip = int(np.argmax(dr))
    after = np.nonzero(dr[tip:] <= s.flange_split_mm * 1e-3)[0]
    split = tip + (int(after[0]) if len(after) else 0)
    # per-segment slope |d dr / d a| as an angle (vertical segments have d a = 0)
    seg = np.abs(np.arctan2(np.diff(dr), np.diff(a)))
    seg = np.minimum(seg, np.pi - seg)
    steep = np.nonzero(seg[split:] > np.arctan(s.rim_face_slope))[0]
    field = split + int(steep[0]) if len(steep) else len(a) - 1
    mm = 1e-3
    r_rim = R - s.rim_thickness_mm * mm
    pc, pt = s.plate_center_mm * mm, s.plate_thickness_mm * mm
    hub_r, bore = s.hub_outer_radius_mm * mm, s.bore_radius_mm * mm
    h0, h1 = s.hub_gauge_side_mm * mm, s.hub_field_side_mm * mm
    a0, a1 = a[0], a[-1]
    return {
        "wheel_flange": np.stack([np.r_[a0, a[: split + 1]], np.r_[r_rim, r[: split + 1]]], 1),
        "wheel_tread": np.stack([a[split: field + 1], r[split: field + 1]], 1),
        "wheel_rim_face": np.array([*np.stack([a[field:], r[field:]], 1), [a1, r_rim], [pc + pt / 2, r_rim]]),
        "wheel_plate": np.array([[pc + pt / 2, r_rim], [pc + pt / 2, hub_r], [h1, hub_r],
                                 [h0, hub_r], [pc - pt / 2, hub_r], [pc - pt / 2, r_rim], [a0, r_rim]]),
        "wheel_hub": np.array([[h1, hub_r], [h1, bore], [h0, bore], [h0, hub_r]]),
    }


# --------------------------------------------------------------------------- rail
def load_rail_profile(name: str) -> dict[str, np.ndarray]:
    """{part: closed polygon (N,2) in metres as (u, h)} from assets/profiles/rails/<name>.csv."""
    path = PROFILE_DIR / "rails" / f"{name}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; available: {available('rails')}")
    rows = _rows(path)
    if rows and rows[0][0].strip() == "part":
        rows = rows[1:]
    parts: dict[str, list] = {}
    for part, u, h in rows:
        parts.setdefault(part.strip(), []).append((float(u) * 1e-3, float(h) * 1e-3))
    return {k: np.asarray(v) for k, v in parts.items()}


def rail_dimensions(parts: dict[str, np.ndarray]) -> dict[str, float]:
    """Height, head width and the gauge-face lateral offset 16 mm below the top (m)."""
    allp = np.concatenate(list(parts.values()))
    height = float(allp[:, 1].max())
    head = parts["head"]
    near = head[np.abs(head[:, 1] - (height - 0.016)) < 0.01]
    gauge_u = float(near[:, 0].max()) if len(near) else float(head[:, 0].max())
    return {"height": height, "head_width": float(np.ptp(head[:, 0])), "gauge_face_u": gauge_u}
