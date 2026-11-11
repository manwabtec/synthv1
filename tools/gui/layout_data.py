"""Data exchange between the YAML configs and the browser layout viewer (tools/gui/layout_view).

The viewer does all layout geometry in the browser (WebGL drawing, exact ray casting
against the wheel's surface of revolution and the extruded rails), so slider changes need
no server round trip. Python only:
  * `geometry()`   sends the static shapes once per wheel/rail profile choice,
  * `view_state()` sends the current rig + scene values when a scene is (re)loaded,
  * `apply_edits()` writes the viewer's edits back into the comment-preserving YAML docs.

Everything is in the track frame (x along the track, y left, z up, m). No bpy.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

from synthrig.assets import profiles
from synthrig.assets.wheelrail import (WheelRailParams, _contour, _rail, ballast_level, key_points,
                                       rail_height, rail_sections, wheel_sections)
from synthrig.layout import _resample
from synthrig.opticsrig.laser import pattern_tangents, wavelength_to_rgb
from synthrig.opticsrig.schema import Rig
from synthrig.opticsrig.transforms import pose_to_T
from synthrig.sampling import is_spec
from tools.gui import configio as cio

WHEEL_FIELDS = ("wheel_x_m", "wheel_rot_deg", "lateral_shift_m")
TRACK_HALF_LENGTH_M = 2.5  # the viewer draws the track from x = -2.5 to +2.5 m


def _r(a, nd: int = 6):
    return np.round(np.asarray(a, float), nd).tolist()


@lru_cache(maxsize=16)
def geometry(wheel_profile: str, rail_profile: str) -> dict[str, Any]:
    """Static shapes for the viewer, with the wheel at x = 0 and no lateral shift.

    wheel.sections: {class: [[a, r], ...]} section polylines (a = lateral from the flange
        back face, + field side; r = radius), revolved about the axle for drawing.
    wheel.solids:   closed (a, r) polygons whose edges are the wheel surface (ray casting).
    wheel.profile:  the measured tread/flange contour resampled every 2 mm, flange back
        face -> field side (profile coverage).
    wheel.origin0:  wheel origin (on the axle, in the flange back-face plane) for x = 0,
        lateral shift 0; the viewer adds wheel_x_m to x and lateral_shift_m to y.
    """
    p = WheelRailParams(wheel_profile=wheel_profile, rail_profile=rail_profile)
    # the measured contour is dense (~0.2 mm); 1 mm keeps the shape and the ray casting fast
    secs = {k: (_resample(v, 0.001) if k in ("wheel_flange", "wheel_tread", "wheel_rim_face") else v)
            for k, v in wheel_sections(p).items()}
    R = p.wheel_section.tape_diameter_mm / 2000.0
    outline = np.concatenate([secs[k] for k in ("wheel_flange", "wheel_tread", "wheel_rim_face", "wheel_plate")])
    keep = np.r_[True, np.linalg.norm(np.diff(outline, axis=0), axis=1) > 1e-9]
    c = _contour(wheel_profile)
    prof = _resample(np.stack([c.a, R + c.dr], 1), 0.002)
    _, dims = _rail(rail_profile)
    tie = [v * 1e-3 for v in p.tie_size_mm]
    plate = [v * 1e-3 for v in p.tie_plate_mm]
    kp = key_points(p)
    return {
        "key": f"{wheel_profile}|{rail_profile}",
        "wheel": {
            "R": R,
            "origin0": _r([0.0, -p.back_to_back_mm / 2000.0, R]),
            "sections": {k: _r(v) for k, v in secs.items()},
            "solids": [_r(outline[keep]), _r(secs["wheel_hub"])],
            "profile": _r(prof),
        },
        "rails": [{"side": side, "part": part, "poly": _r(poly)} for side, part, poly in rail_sections(p)],
        "rail_centre_y": -(p.gauge_mm * 1e-3 / 2 + dims["gauge_face_u"]),
        "rail_height": rail_height(p),
        "ties": {"spacing": p.tie_spacing_mm * 1e-3, "size": tie, "plate": plate,
                 "z_top": -rail_height(p) - plate[2]},
        "ballast_z": ballast_level(p),
        "half_length": TRACK_HALF_LENGTH_M,
        # key points relative to the wheel: the viewer shifts x by wheel_x_m and y by the
        # lateral shift (except rail_head_top, which belongs to the rail)
        "key_points": kp,
    }


def _layout_value(v):
    if not is_spec(v):
        return v
    kind, arg = next(iter(v.items()))
    return {"uniform": (arg[0] + arg[1]) / 2, "normal": arg[0], "choice": arg[0]}[kind]


def _scene_field(v) -> dict:
    out = {"value": float(_layout_value(v))}
    if is_spec(v):
        kind, arg = next(iter(v.items()))
        if kind == "uniform":
            out["range"] = [float(arg[0]), float(arg[1])]
        else:
            out["spec"] = cio.plain(v)  # shown, not editable in the viewer
    return out


def view_state(rig_doc, scene_doc) -> dict[str, Any]:
    """Current rig + scene values for the viewer (validated; raises if invalid)."""
    rig = Rig.model_validate(cio.plain(rig_doc))
    scene = cio.plain(scene_doc) if scene_doc is not None else {}
    doc_cams = {c["name"]: c for c in rig_doc["cameras"]}
    doc_lasers = {l["name"]: l for l in rig_doc.get("lasers", [])}
    cams = []
    for c in rig.cameras:
        i = c.intrinsics
        cams.append({
            "name": c.name, "width": i.width, "height": i.height, "fx": i.fx, "fy": i.fy, "cx": i.cx, "cy": i.cy,
            "pixel_size_um": i.pixel_size_um, "distortion": list(c.distortion),
            "position": list(c.pose.position), "look_at": c.pose.look_at, "up": c.pose.up,
            "up_in_file": "up" in doc_cams[c.name]["pose"],
            "R": _r(pose_to_T(c.pose)[:3, :3]),
        })
    lasers = []
    for l in rig.lasers:
        lasers.append({
            "name": l.name, "position": list(l.pose.position), "look_at": l.pose.look_at, "up": l.pose.up,
            "up_in_file": "up" in doc_lasers[l.name]["pose"],
            "R": _r(pose_to_T(l.pose)[:3, :3]),
            "pattern": l.pattern.model_dump(mode="json", exclude_none=True),
            "tangents": _r(pattern_tangents(l.pattern, l.dot_sigma_mrad * 1e-3), 7),
            "rgb": l.color_rgb if l.color_rgb is not None else wavelength_to_rgb(l.wavelength_nm),
            "power_w": l.power_w, "enabled": l.enabled, "dot_sigma_mrad": l.dot_sigma_mrad,
        })
    lights = [{"name": li.name, "position": list(li.pose.position), "R": _r(pose_to_T(li.pose)[:3, :3]),
               "size_m": li.size_m} for li in rig.lights]
    pairs = [{"name": s.name, "left": s.left, "right": s.right, "rectify": s.rectify} for s in rig.stereo_pairs]
    p = WheelRailParams()
    fields = {k: _scene_field(scene.get(k, getattr(p, k))) for k in WHEEL_FIELDS}
    return {
        "rig_name": rig.name, "frame": rig.frame, "cameras": cams, "lasers": lasers, "lights": lights,
        "pairs": pairs,
        "scene": {"wheel_profile": scene.get("wheel_profile", p.wheel_profile),
                  "rail_profile": scene.get("rail_profile", p.rail_profile), **fields},
        "wheel_profiles": profiles.available("wheels"),
    }


# --------------------------------------------------------------------------- edits
def _vec(v, nd: int = 4) -> list[float]:
    return [round(float(x), nd) for x in v]


def _set_pose(pose, e: dict) -> None:
    for key in ("position", "look_at"):
        if key in e:
            cio.set_if_changed(pose, key, _vec(e[key]))
    if "up" in e:
        cio.set_if_changed(pose, "up", _vec(e["up"], 5), default=[0.0, 0.0, 1.0])


def apply_edits(rig_doc, scene_doc, edits: dict) -> None:
    """Apply the viewer's edits to the ruamel documents in place (unchanged values keep
    their formatting). `edits` holds only what the user touched:
      cameras: {name: {position, look_at, up, hfov_deg}}
      lasers:  {name: {position, look_at, up, enabled, power_w, pattern}}
      scene:   {wheel_profile, wheel_x_m | wheel_rot_deg | lateral_shift_m: number or {uniform: [lo, hi]}}
    """
    cams = {c["name"]: c for c in rig_doc["cameras"]}
    for name, e in (edits.get("cameras") or {}).items():
        cam = cams.get(name)
        if cam is None:
            continue
        _set_pose(cam["pose"], e)
        if "hfov_deg" in e:
            intr = cam["intrinsics"]
            cur = Rig.model_validate(cio.plain(rig_doc)).camera(name).intrinsics
            cur_hfov = float(np.degrees(2 * np.arctan(cur.width / 2 / cur.fx)))
            if abs(float(e["hfov_deg"]) - cur_hfov) > 1e-3:
                for k in ("fx", "fy", "focal_length_mm"):
                    intr.pop(k, None)
                intr["hfov_deg"] = round(float(e["hfov_deg"]), 3)
    lasers = {l["name"]: l for l in rig_doc.get("lasers", [])}
    for name, e in (edits.get("lasers") or {}).items():
        las = lasers.get(name)
        if las is None:
            continue
        _set_pose(las["pose"], e)
        if "enabled" in e:
            cio.set_if_changed(las, "enabled", bool(e["enabled"]), default=True)
        if "power_w" in e:
            cio.set_if_changed(las, "power_w", round(float(e["power_w"]), 2), default=20.0)
        if "pattern" in e:
            new = {k: (cio.flow(v) if isinstance(v, list) else v) for k, v in e["pattern"].items()}
            if cio.plain(las["pattern"]) != cio.plain(new):
                las["pattern"] = cio.flow_map(new)
    if scene_doc is None:
        return
    for key, v in (edits.get("scene") or {}).items():
        if key == "wheel_profile":
            cio.set_if_changed(scene_doc, key, str(v))
        elif key in WHEEL_FIELDS:
            if isinstance(v, dict) and "uniform" in v:
                new = {"uniform": _vec(v["uniform"])}
                if cio.plain(scene_doc.get(key)) != new:
                    scene_doc[key] = cio.flow_map({"uniform": cio.flow(new["uniform"])})
            else:
                cio.set_if_changed(scene_doc, key, round(float(v), 4), default=0.0)
