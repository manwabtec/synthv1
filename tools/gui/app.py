"""synthrig GUI: edit the wheel/rail layout, render, and inspect results in a browser.

Start it with `python3 infra/cloud.py gui` (runs in the Docker image with the GPU), then
open http://localhost:8501 (VS Code forwards the port when you work on EC2).

Tabs (each updates on its own, so editing the layout does not reload images):
  Layout          sliders for the wheel, camera rods (stereo pairs), cameras and lasers;
                  the 3D layout, profile-coverage inset and camera views update at once
                  (rods = `rods:` in the rig, or its stereo pairs if it has none);
                  valid edits are saved into the YAML files (comments kept)
  Render & images render frames of the run and browse each camera's images
  3D view         Blender render of the whole scene from a free viewpoint, with markers
"""
from __future__ import annotations

import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from synthrig import layout as lay  # noqa: E402
from synthrig.assets import profiles  # noqa: E402
from synthrig.assets.wheelrail import WheelRailParams, key_points  # noqa: E402
from synthrig.opticsrig import RigModel  # noqa: E402
from synthrig.opticsrig.schema import Rig  # noqa: E402
from synthrig.paths import CODE_ROOT, data_root  # noqa: E402
from synthrig.sampling import is_spec  # noqa: E402
from tools.gui import configio as cio  # noqa: E402
from tools.gui import figures  # noqa: E402

st.set_page_config(page_title="synthrig", layout="wide")
S = st.session_state
CONFIGS = CODE_ROOT / "configs"
LAYOUT_SCENES = {"wheel_rail", "wheel_tread"}  # scenes built from WheelRailParams
PATTERN_DEFAULTS = {
    "line": {"type": "line", "fan_deg": 40.0, "angle_deg": 90.0},
    "grid": {"type": "grid", "rows": 15, "cols": 21, "fan_deg": [24.0, 16.0]},
    "random": {"type": "random", "count": 500, "fov_deg": [30.0, 20.0], "min_separation_deg": 0.5, "seed": 1},
}
# slider ranges (track frame, m); extended automatically if a value lies outside
RANGE = {"x": (-1.5, 1.5), "y": (-1.8, 0.3), "z": (-0.3, 1.2)}
POS_STEP = 0.005
WHEEL_FIELDS = [("wheel_x_m", "Wheel position along rail x (m)", -1.0, 1.0, 0.005),
                ("lateral_shift_m", "Lateral shift (m)", -0.02, 0.02, 0.0005),
                ("wheel_rot_deg", "Wheel rotation (deg)", 0.0, 360.0, 1.0)]


# --------------------------------------------------------------------------- state
def load_run(run: Path) -> None:
    run_doc = yaml.safe_load(run.read_text())
    rig_path = (run.parent / run_doc["rig"]).resolve()
    scene_path = (run.parent / run_doc["scene_config"]).resolve() if run_doc.get("scene_config") else None
    S.run, S.run_doc, S.rig_path, S.scene_path = run, run_doc, rig_path, scene_path
    S.rig_doc = cio.load(rig_path)
    S.scene_doc = cio.load(scene_path) if scene_path else None
    S.saved = {"rig": cio.dumps(S.rig_doc), "scene": cio.dumps(S.scene_doc) if S.scene_doc is not None else ""}
    S.gen = S.get("gen", 0) + 1  # new widget keys -> widgets re-read the files
    S.save_msg = ""


def k(*parts) -> str:
    return ":".join(str(p) for p in (S.gen, *parts))


def layout_value(v):
    """A representative value of a fixed value or sampling spec (for the layout view)."""
    if not is_spec(v):
        return v
    kind, arg = next(iter(v.items()))
    return {"uniform": (arg[0] + arg[1]) / 2, "normal": arg[0], "choice": arg[0]}[kind]


def scene_params() -> WheelRailParams:
    doc = cio.plain(S.scene_doc) if S.scene_doc is not None else {}
    return WheelRailParams(**{key: layout_value(v) for key, v in doc.items()})


@st.cache_data(max_entries=32, show_spinner=False)
def compute_layout(rig_json: str, params_json: str):
    rig = RigModel.build(Rig.model_validate(json.loads(rig_json)))
    p = WheelRailParams(**json.loads(params_json))
    p = p.model_copy(update={"track_half_length_m": min(p.track_half_length_m, 1.5)})
    L = lay.compute(rig, p)
    sets = lay.camera_sets(rig, p)
    aim = next((rig.cameras[s.cameras[0]].spec.pose.look_at for s in sets if s.label == "outside"), None)
    aim = aim or next(iter(rig.cameras.values())).spec.pose.look_at or key_points(p)["tread_contact"]
    pc = lay.profile_coverage(rig, p, L.meshes, lay.section_angle(p, aim))
    return L, sets, pc


# --------------------------------------------------------------------------- widgets
def _init(key: str, value) -> dict:
    """Pass the file's value only on first creation; afterwards session state owns it
    (avoids Streamlit's warning when a callback, e.g. 'Aim at', sets the value)."""
    return {} if key in S else {"value": value}


def slider(label, value, key, lo, hi, step, fmt="%.3f", help=None) -> float:
    kk = k(key)
    v = float(S.get(kk, value))
    lo, hi = min(float(lo), v), max(float(hi), v)
    return st.slider(label, min_value=lo, max_value=hi, step=float(step), format=fmt, key=kk, help=help,
                     **_init(kk, float(value)))


def vec3(label, values, key) -> list[float]:
    return [slider(f"{label} {a}", values[i], f"{key}:{a}", *RANGE[a], POS_STEP) for i, a in enumerate("xyz")]


def roll_of(pos, aim, up) -> float:
    """Camera roll (deg) about its optical axis relative to 'image up = world up'."""
    z = np.subtract(aim, pos, dtype=float)
    z /= np.linalg.norm(z)
    u0 = np.array([0.0, 0.0, 1.0]) - z[2] * z
    u = np.asarray(up if up is not None else [0, 0, 1], float)
    u = u - (u @ z) * z
    return float(np.degrees(np.arctan2(z @ np.cross(u0, u), u0 @ u)))


def up_for_roll(pos, aim, roll_deg) -> list[float]:
    z = np.subtract(aim, pos, dtype=float)
    z /= np.linalg.norm(z)
    u0 = np.array([0.0, 0.0, 1.0]) - z[2] * z
    u0 /= np.linalg.norm(u0)
    t = np.radians(roll_deg)
    u = u0 * np.cos(t) + np.cross(z, u0) * np.sin(t)  # Rodrigues (u0 is perpendicular to z)
    return [round(float(v), 6) for v in u]


def _set_aim(key: str, points: dict) -> None:
    choice = S[k(key, "aimsel")]
    if choice in points:
        for a, v in zip("xyz", points[choice]):
            S[k(key, "aim", a)] = float(v)


def aim_editor(aim0, key: str, points: dict) -> list[float]:
    st.selectbox("Aim at", ["(custom)"] + list(points), key=k(key, "aimsel"), on_change=_set_aim,
                 args=(key, points), help="Snap the aim point to a named point on the wheel/rail")
    return vec3("aim", aim0, f"{key}:aim")


def set_roll(pose, roll: float, current: float) -> None:
    if abs(roll - current) > 0.05:
        pose["up"] = cio.flow(up_for_roll(pose["position"], pose["look_at"], roll))


def hfov_editor(intr, current: float, key: str) -> None:
    hfov = slider("Field of view, horizontal (deg)", current, key, 5.0, 120.0, 0.5, "%.1f")
    if abs(hfov - current) > 1e-3:
        for kk in ("fx", "fy", "focal_length_mm"):
            intr.pop(kk, None)
        intr["hfov_deg"] = round(hfov, 3)


def camera_editor(cam, model, points) -> None:
    pose, key = cam["pose"], f"cam:{cam['name']}"
    cio.set_if_changed(pose, "position", vec3("pos", pose["position"], f"{key}:pos"))
    if "look_at" not in pose:
        st.caption("Orientation is given as euler_xyz_deg / R in the YAML (not editable here).")
    else:
        cio.set_if_changed(pose, "look_at", aim_editor(pose["look_at"], key, points))
        r0 = roll_of(pose["position"], pose["look_at"], pose.get("up"))
        set_roll(pose, slider("Roll about optical axis (deg)", r0, f"{key}:roll", -180, 180, 1, "%.0f"), r0)
    hfov_editor(cam["intrinsics"], lay.hfov_deg(model), f"{key}:hfov")
    f_mm = model.K[0, 0] * model.spec.intrinsics.pixel_size_um / 1000
    st.caption(f"{cam['intrinsics']['width']} x {cam['intrinsics']['height']} px; "
               f"equivalent focal length {f_mm:.2f} mm")


def rod_editor(rod, cams: dict, models: dict, points, colour: str) -> None:
    """The cameras of one rod: evenly spaced along the rod axis (default: along the rail),
    sharing an aim point, field of view and roll. Rod length = first-to-last spacing."""
    members = [cams[n] for n in rod["cameras"]]
    if any("look_at" not in c["pose"] for c in members):
        st.caption("Rod editing needs aim points (look_at) on all its cameras.")
        return
    axis = np.asarray(rod.get("axis", [1.0, 0.0, 0.0]), float)
    axis /= np.linalg.norm(axis)
    P = np.array([c["pose"]["position"] for c in members], float)
    order = np.argsort(P @ axis)
    centre = P.mean(0)
    length = float(np.ptp(P @ axis)) if len(P) > 1 else 0.0
    aim0 = np.mean([c["pose"]["look_at"] for c in members], axis=0)
    key = f"rod:{rod['name']}"
    st.markdown(f"<span style='color:{colour}'>&#9632;</span> cameras {' + '.join(rod['cameras'])}",
                unsafe_allow_html=True)
    c = vec3("rod centre", centre.tolist(), f"{key}:c")
    L = slider("Rod length = camera spacing (m)", length, f"{key}:len", 0.0, 1.2, POS_STEP) if len(P) > 1 else 0.0
    aim = aim_editor(aim0.tolist(), key, points)
    r0 = roll_of(centre, aim0, members[0]["pose"].get("up"))
    roll = slider("Roll about optical axis (deg)", r0, f"{key}:roll", -180, 180, 1, "%.0f")
    cur_hfov = lay.hfov_deg(models[rod["cameras"][0]])
    hfov = slider("Field of view, horizontal (deg)", cur_hfov, f"{key}:hfov", 5.0, 120.0, 0.5, "%.1f")
    changed = (np.abs(np.subtract(c, centre)).max() > 1e-9 or abs(L - length) > 1e-9
               or np.abs(np.subtract(aim, aim0)).max() > 1e-9)
    if changed:  # only touch the files when a rod slider actually moved
        n = len(members)
        for rank, i in enumerate(order):
            off = (rank - (n - 1) / 2) * (L / (n - 1) if n > 1 else 0.0)
            cio.set_if_changed(members[i]["pose"], "position", list(np.asarray(c) + off * axis))
            cio.set_if_changed(members[i]["pose"], "look_at", aim)
    if abs(roll - r0) > 0.05 or (changed and "up" in members[0]["pose"]):
        for cam in members:
            cam["pose"]["up"] = cio.flow(up_for_roll(cam["pose"]["position"], cam["pose"]["look_at"], roll))
    if abs(hfov - cur_hfov) > 1e-3:
        for cam in members:
            for kk in ("fx", "fy", "focal_length_mm"):
                cam["intrinsics"].pop(kk, None)
            cam["intrinsics"]["hfov_deg"] = round(hfov, 3)


def laser_editor(las, points) -> None:
    name, pose = las["name"], las["pose"]
    key = f"las:{name}"
    cio.set_if_changed(las, "enabled", st.toggle("enabled", value=bool(las.get("enabled", True)),
                                                 key=k(key, "on")), default=True)
    cio.set_if_changed(pose, "position", vec3("pos", pose["position"], f"{key}:pos"))
    if "look_at" in pose:
        cio.set_if_changed(pose, "look_at", aim_editor(pose["look_at"], key, points))
    pat = las["pattern"]
    kinds = ["line", "grid", "random"]
    if pat["type"] not in kinds:
        st.caption(f"Pattern type {pat['type']} is edited in the YAML.")
    else:
        kind = st.selectbox("Pattern", kinds, index=kinds.index(pat["type"]), key=k(key, "type"))
        if kind != pat["type"]:
            las["pattern"] = cio.flow_map({kk: (cio.flow(v) if isinstance(v, list) else v)
                                           for kk, v in PATTERN_DEFAULTS[kind].items()})
            pat = las["pattern"]
        if kind == "line":
            cio.set_if_changed(pat, "fan_deg", slider("Fan angle (deg)", pat["fan_deg"], f"{key}:fan", 1, 120, 0.5, "%.1f"))
            cio.set_if_changed(pat, "angle_deg", slider("Line orientation about the beam (deg)", pat.get("angle_deg", 0.0),
                                                       f"{key}:ang", -90, 90, 1, "%.0f"))
        elif kind == "grid":
            fan0 = ([pat["pitch_deg"][0] * (pat["cols"] - 1), pat["pitch_deg"][1] * (pat["rows"] - 1)]
                    if "pitch_deg" in pat else list(pat["fan_deg"]))
            c1, c2 = st.columns(2)
            cio.set_if_changed(pat, "cols", int(c1.number_input("Columns", value=int(pat["cols"]), step=1,
                                                                 min_value=1, key=k(key, "cols"))))
            cio.set_if_changed(pat, "rows", int(c2.number_input("Rows", value=int(pat["rows"]), step=1,
                                                                 min_value=1, key=k(key, "rows"))))
            fx = slider("Fan x (deg)", fan0[0], f"{key}:fx", 0.5, 120, 0.5, "%.1f")
            fy = slider("Fan y (deg)", fan0[1], f"{key}:fy", 0.5, 120, 0.5, "%.1f")
            if abs(fx - fan0[0]) > 1e-9 or abs(fy - fan0[1]) > 1e-9:
                pat.pop("pitch_deg", None)
                pat["fan_deg"] = cio.flow([fx, fy])
        else:
            cio.set_if_changed(pat, "count", int(st.number_input("Dots", value=int(pat["count"]), step=10,
                                                                  min_value=1, key=k(key, "n"))))
            fx = slider("Field x (deg)", pat["fov_deg"][0], f"{key}:fx", 0.5, 120, 0.5, "%.1f")
            fy = slider("Field y (deg)", pat["fov_deg"][1], f"{key}:fy", 0.5, 120, 0.5, "%.1f")
            cio.set_if_changed(pat, "fov_deg", [fx, fy])
    cio.set_if_changed(las, "power_w", slider("Power (relative W)", las.get("power_w", 20.0), f"{key}:pw",
                                              0, 1000, 5, "%.0f"), default=20.0)


def wheel_editor(doc) -> None:
    names = profiles.available("wheels")
    cur = doc.get("wheel_profile", "AAR_2A_ref_wide")
    prof = st.selectbox("Wheel profile (assets/profiles/wheels)", names,
                        index=names.index(cur) if cur in names else 0, key=k("wheel", "profile"))
    cio.set_if_changed(doc, "wheel_profile", prof)
    for field, label, lo, hi, step in WHEEL_FIELDS:
        v = doc.get(field, 0.0)
        spec = is_spec(v)
        mode = st.segmented_control(label, ["fixed", "random range"], default="random range" if spec else "fixed",
                                    key=k("wheel", field, "mode"))
        if mode == "random range":
            lo0, hi0 = (v["uniform"] if spec and "uniform" in v else [max(lo, float(layout_value(v)) - 20 * step),
                                                                       min(hi, float(layout_value(v)) + 20 * step)])
            kk = k("wheel", field, "range")
            lo1, hi1 = st.slider("range", min_value=float(min(lo, lo0)), max_value=float(max(hi, hi0)), step=float(step),
                                 key=kk, format="%.4f" if step < 0.001 else "%.3f", **_init(kk, (float(lo0), float(hi0))))
            if cio.plain(v) != {"uniform": [round(lo1, 4), round(hi1, 4)]}:
                doc[field] = cio.flow_map({"uniform": cio.flow([lo1, hi1])})
        else:
            val = slider("value", layout_value(v), f"wheel:{field}:val", lo, hi, step,
                         "%.4f" if step < 0.001 else "%.3f")
            if spec or field in doc or abs(val) > 0:
                cio.set_if_changed(doc, field, round(val, 4))


def validate_and_save(autosave: bool):
    """Validate the edited documents; save changed ones if valid. Returns (rig, params) or Nones."""
    try:
        rig = Rig.model_validate(cio.plain(S.rig_doc))
        p = scene_params()
    except Exception as exc:  # pydantic ValidationError and friends: show, don't save
        st.error(f"Not saved, invalid: {exc}")
        return None, None
    if autosave:
        for kind, doc, path in (("rig", S.rig_doc, S.rig_path), ("scene", S.scene_doc, S.scene_path)):
            if doc is None:
                continue
            text = cio.dumps(doc)
            if text != S.saved[kind]:
                cio.save(doc, path)
                S.saved[kind] = text
                S.save_msg = f"saved {path.relative_to(CODE_ROOT)} at {dt.datetime.now():%H:%M:%S}"
    return rig, p


# --------------------------------------------------------------------------- jobs
def start_job(kind: str, cmd: list[str], log: Path, **info) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    fh = log.open("w")
    proc = subprocess.Popen(cmd, cwd=CODE_ROOT, stdout=fh, stderr=subprocess.STDOUT, text=True)
    S.job = {"kind": kind, "proc": proc, "log": log, "started": dt.datetime.now(), **info}


def job_running() -> bool:
    return "job" in S and S.job["proc"].poll() is None


@st.fragment(run_every=2)
def job_status(kind: str) -> None:
    if "job" not in S or S.job["kind"] != kind:
        return
    j = S.job
    lines = j["log"].read_text().splitlines() if j["log"].exists() else []
    rc = j["proc"].poll()
    secs = (dt.datetime.now() - j["started"]).total_seconds()
    if kind == "render":
        done = sum(" done in " in ln for ln in lines)
        st.progress(min(done / j["frames"], 1.0), text=f"{done}/{j['frames']} frames, {secs:.0f} s")
    if rc is None:
        st.info(f"Running ({secs:.0f} s) ...")
    elif rc == 0:
        st.success(f"Finished in {secs:.0f} s")
    else:
        st.error(f"Failed (exit {rc}); log below")
    with st.expander("log", expanded=rc not in (None, 0)):
        st.code("\n".join(ln for ln in lines[-25:] if "Fra:" not in ln) or "(no output yet)")
    if rc is not None and not j.get("refreshed"):
        j["refreshed"] = True
        st.rerun(scope="app")


# --------------------------------------------------------------------------- images
def display(img: np.ndarray, bits: int = 12) -> np.ndarray:
    """uint16 DN -> 8-bit RGB for display (gamma 2.2 only for viewing)."""
    a = np.clip(img.astype(np.float32) / (2 ** bits - 1), 0, 1) ** (1 / 2.2)
    a = (a * 255).astype(np.uint8)
    return cv2.cvtColor(a, cv2.COLOR_GRAY2RGB) if a.ndim == 2 else cv2.cvtColor(a, cv2.COLOR_BGR2RGB)


def colourise(a: np.ndarray) -> np.ndarray:
    v = a[np.isfinite(a) & (a > 0)]
    lo, hi = (np.percentile(v, [1, 99]) if v.size else (0, 1))
    n = np.clip((a - lo) / max(hi - lo, 1e-9), 0, 1)
    out = cv2.cvtColor(cv2.applyColorMap((n * 255).astype(np.uint8), cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    out[~(np.isfinite(a) & (a > 0))] = 0
    return out


def seg_colours(seg: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(3)
    lut = rng.integers(40, 255, (int(seg.max()) + 1, 3), dtype=np.uint8)
    lut[0] = 0
    return lut[seg]


def images_view(scene: str) -> None:
    from synthrig.dataset import Dataset

    root = data_root() / "gui" / scene
    sets = sorted((m.parent for m in root.glob("*/manifest.json")), reverse=True)
    if not sets:
        st.info("No renders yet for this scene. Use the Render button above.")
        return
    c1, c2, c3 = st.columns([2, 1, 1])
    ds_dir = c1.selectbox("Rendered dataset", sets, format_func=lambda p: p.name)
    ds = Dataset(ds_dir)
    frames = list(ds.frames())
    if not frames:
        st.warning("This dataset has no finished frames.")
        return
    fi = c2.number_input("Frame", 0, len(frames) - 1, 0, key="img_frame")
    rig = yaml.safe_load((ds_dir / "rig.yaml").read_text())
    bits = {c["name"]: c.get("sensor", {}).get("bit_depth", 12) for c in rig["cameras"]}
    lasers = [l["name"] for l in rig.get("lasers", []) if l.get("enabled", True)]
    view = c3.selectbox("Show", ["camera images", "laser masks", "depth", "segmentation"], key="img_view")
    fr = frames[fi]
    cols = st.columns(2)
    for i, cam in enumerate(bits):
        img = fr.image(cam)
        if view == "camera images":
            out = display(img, bits[cam])
        elif view == "laser masks":
            out = (display(img, bits[cam]) * 0.45).astype(np.uint8)
            for j, lname in enumerate(lasers):
                if fr.has(f"{cam}/laser_{lname}_mask.png"):
                    out[fr.laser_mask(cam, lname) > 0] = [(255, 60, 60), (60, 255, 60), (90, 160, 255)][j % 3]
        elif view == "depth":
            out = colourise(fr.depth(cam))
        else:
            out = seg_colours(fr.seg(cam))
        cols[i % 2].image(out, caption=f"{cam} ({img.shape[1]}x{img.shape[0]})", width="stretch")
    with st.expander("frame parameters (meta.json)"):
        st.json({k_: fr.meta[k_] for k_ in ("frame", "seed", "scene_params", "timings_s") if k_ in fr.meta})


# --------------------------------------------------------------------------- tabs
@st.fragment
def layout_tab(autosave: bool) -> None:
    if not (S.rig_doc.get("frame") == "track" and S.run_doc.get("scene") in LAYOUT_SCENES):
        st.warning("The layout editor supports wheel_rail / wheel_tread runs with a `frame: track` rig.")
        return
    left, right = st.columns([1, 2.2])
    with left:
        with st.expander("Wheel", expanded=False):
            wheel_editor(S.scene_doc)
        try:
            p0 = scene_params()
            points = key_points(p0)
            models = RigModel.build(Rig.model_validate(cio.plain(S.rig_doc))).cameras
            sets = {tuple(s.cameras): s.label for s in lay.camera_sets(
                RigModel.build(Rig.model_validate(cio.plain(S.rig_doc))), p0)}
        except Exception as exc:
            st.error(f"Invalid configuration: {exc}")
            return
        cams = {c["name"]: c for c in S.rig_doc["cameras"]}
        rods = S.rig_doc.get("rods") or [{"name": pr["name"], "cameras": [pr["left"], pr["right"]]}
                                         for pr in S.rig_doc.get("stereo_pairs", [])]
        for rod in rods:
            side = sets.get(tuple(rod["cameras"]), "outside")
            label = "outside the rail (field side)" if side == "outside" else "inside the rail (gauge side)"
            with st.expander(f"Rod {rod['name']}: {label}", expanded=False):
                rod_editor(rod, cams, models, points, figures.SET_COLOURS[side][0])
        with st.expander("Individual cameras", expanded=False):
            st.caption("Fine-tune one camera. For cameras on a rod, prefer the rod sliders.")
            name = st.selectbox("Camera", list(cams), key=k("cam", "pick"))
            camera_editor(cams[name], models[name], points)
        for las in S.rig_doc.get("lasers", []):
            with st.expander(f"Laser {las['name']}", expanded=False):
                laser_editor(las, points)
    rig, p = validate_and_save(autosave)
    with right:
        st.caption(S.save_msg or "no changes saved yet")
        if rig is None:
            return
        L, csets, pc = compute_layout(rig.canonical_json(), p.model_dump_json())
        model = RigModel.build(rig)
        st.plotly_chart(figures.layout_3d(model, L, key_points(p), sets=csets, profile=pc), key="fig3d")
        st.caption("Camera views (approximate: coarse geometry, no occlusion); % = laser points in view")
        vcols = st.columns(2)
        for i, cname in enumerate(model.cameras):
            vcols[i % 2].plotly_chart(figures.camera_view(model, L, cname), key=f"view_{cname}")
        with st.expander("Aim points (track frame, m)"):
            st.dataframe({n: v for n, v in key_points(p).items()})


@st.fragment
def render_tab(run: Path) -> None:
    c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1.4])
    n_frames = c1.number_input("Frames", 1, 1000, 1)
    scale = c2.selectbox("Scale", [0.125, 0.25, 0.5, 1.0], index=2)
    samples = c3.selectbox("Samples", [8, 16, 32, 64, 128], index=2)
    device = c4.selectbox("Device", ["GPU", "CPU"])
    if c5.button("Render", type="primary", disabled=job_running(), width="stretch"):
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        out = Path("gui") / run.parent.name / stamp
        start_job("render", [sys.executable, "-m", "synthrig", str(run), "--frames", str(n_frames),
                             "--scale", str(scale), "--samples", str(samples), "--device", device,
                             "--out", str(out)], data_root() / out.with_suffix(".log"), frames=n_frames)
    job_status("render")
    images_view(run.parent.name)


@st.fragment
def view3d_tab(run: Path) -> None:
    c1, c2 = st.columns([1, 2.6])
    with c1:
        try:
            pts3 = key_points(scene_params())
        except Exception:
            pts3 = {"wheel_center": [0.0, -0.67, 0.46]}
        target = st.selectbox("Look at", list(pts3), index=0)
        az = st.slider("Azimuth (deg, 0 = from the field side)", -180, 180, -35)
        el = st.slider("Elevation (deg)", -10, 85, 20)
        dist = st.slider("Distance (m)", 0.5, 6.0, 2.2, 0.1)
        hf = st.slider("Field of view (deg)", 15, 100, 50)
        size = st.selectbox("Resolution", ["960x600", "1280x800", "1920x1200"], index=1)
        spp = st.selectbox("Samples ", [16, 32, 64, 128, 256], index=2)
        frame3 = st.number_input("Frame (scene sampling)", 0, 10000, 0)
        markers = st.toggle("Show camera/laser markers", value=True)
        go_btn = st.button("Render 3D view", type="primary", disabled=job_running(), width="stretch")
    with c2:
        if go_btn:
            stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
            out = data_root() / "gui" / run.parent.name / "overview" / f"{stamp}.png"
            w, h = size.split("x")
            cmd = [sys.executable, "-m", "synthrig.render.observer", str(run), "--out", str(out),
                   "--frame", str(frame3), "--target", *map(str, pts3[target]), "--azimuth", str(az),
                   "--elevation", str(el), "--distance", str(dist), "--hfov", str(hf), "--size", w, h,
                   "--samples", str(spp)] + ([] if markers else ["--no-markers"])
            start_job("overview", cmd, out.with_suffix(".log"))
        job_status("overview")
        shots = sorted((data_root() / "gui" / run.parent.name / "overview").glob("*.png"), reverse=True)
        if shots:
            pick = st.selectbox("Rendered views", shots, format_func=lambda q: q.stem)
            st.image(str(pick), width="stretch")
        else:
            st.info("No 3D views yet: set the viewpoint and press Render 3D view.")


# --------------------------------------------------------------------------- page
runs = sorted(CONFIGS.glob("*/run.yaml"))
with st.sidebar:
    st.header("synthrig")
    default = next((i for i, r in enumerate(runs) if r.parent.name == "wheel_tread"), 0)
    run = st.selectbox("Scene (configs/<scene>/run.yaml)", runs, index=default, format_func=lambda p: p.parent.name)
    if S.get("run") != run:
        load_run(run)
    if st.button("Reload files from disk"):
        load_run(run)
        st.rerun()
    autosave = st.toggle("Save edits to the YAML files", value=True,
                         help="Valid edits are written immediately (comments are kept). Git has the history.")
    st.caption(f"rig: {S.rig_path.relative_to(CODE_ROOT)}")
    if S.scene_path:
        st.caption(f"scene: {S.scene_path.relative_to(CODE_ROOT)}")

tab_layout, tab_render, tab_3d = st.tabs(["Layout", "Render & images", "3D view"])
with tab_layout:
    layout_tab(autosave)
with tab_render:
    render_tab(run)
with tab_3d:
    view3d_tab(run)
