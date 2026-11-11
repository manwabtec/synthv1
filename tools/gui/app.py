"""synthrig GUI: edit the wheel/rail layout, render, and inspect results in a browser.

Start it with `python3 infra/cloud.py gui` (runs in the Docker image with the GPU), then
open http://localhost:8501 (VS Code forwards the port when you work on EC2).

Tabs:
  Layout          edit cameras, lasers and the wheel; the 3D layout and camera views update
                  immediately; valid edits are saved into the YAML files (comments kept)
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
RUNS = CODE_ROOT / "configs" / "runs"
PATTERN_DEFAULTS = {
    "line": {"type": "line", "fan_deg": 40.0, "angle_deg": 90.0},
    "grid": {"type": "grid", "rows": 15, "cols": 21, "fan_deg": [24.0, 16.0]},
    "random": {"type": "random", "count": 500, "fov_deg": [30.0, 20.0], "min_separation_deg": 0.5, "seed": 1},
}
WHEEL_FIELDS = [("wheel_x_m", "Wheel position along rail x (m)", 0.01),
                ("lateral_shift_m", "Lateral shift (m)", 0.001),
                ("wheel_rot_deg", "Wheel rotation (deg)", 5.0)]


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
    return lay.compute(rig, p.model_copy(update={"track_half_length_m": min(p.track_half_length_m, 1.5)}))


# --------------------------------------------------------------------------- widgets
def _init(key: str, value) -> dict:
    """Pass the file's value only on first creation; afterwards session state owns it
    (avoids Streamlit's warning when a callback, e.g. 'Aim at', sets the value)."""
    return {} if key in S else {"value": value}


def num(label, value, key, step=0.01, fmt="%.3f", **kw) -> float:
    key = k(key)
    return st.number_input(label, step=step, format=fmt, key=key, **_init(key, float(value)), **kw)


def vec3(label, values, key, step=0.01) -> list[float]:
    cols = st.columns(3)
    out = []
    for i, a in enumerate("xyz"):
        kk = k(key, a)
        out.append(cols[i].number_input(f"{label} {a}", step=step, format="%.3f", key=kk,
                                        **_init(kk, float(values[i]))))
    return out


def _set_aim(key: str, points: dict) -> None:
    choice = S[k(key, "aimsel")]
    if choice in points:
        for a, v in zip("xyz", points[choice]):
            S[k(key, "aim", a)] = float(v)


def pose_editor(pose, key: str, points: dict) -> None:
    pos = vec3("pos", pose["position"], f"{key}:pos")
    cio.set_if_changed(pose, "position", pos)
    if "look_at" not in pose:
        st.caption("Orientation is given as euler_xyz_deg / R in the YAML (not editable here).")
        return
    st.selectbox("Aim at", ["(custom)"] + list(points), key=k(key, "aimsel"), on_change=_set_aim,
                 args=(key, points), help="Snap the aim point to a named point on the wheel/rail")
    aim = vec3("aim", pose["look_at"], f"{key}:aim")
    cio.set_if_changed(pose, "look_at", aim)


def camera_editor(cam, model, points) -> None:
    pose_editor(cam["pose"], f"cam:{cam['name']}", points)
    intr = cam["intrinsics"]
    current = lay.hfov_deg(model)
    hfov = num("Horizontal field of view (deg)", current, f"cam:{cam['name']}:hfov", 0.5, "%.2f",
               min_value=1.0, max_value=170.0)
    if abs(hfov - current) > 1e-3:
        for key in ("fx", "fy", "focal_length_mm"):
            intr.pop(key, None)
        intr["hfov_deg"] = round(hfov, 3)
    f_mm = model.K[0, 0] * model.spec.intrinsics.pixel_size_um / 1000
    st.caption(f"{intr['width']} x {intr['height']} px, {intr['pixel_size_um']} um pixels; "
               f"equivalent focal length {f_mm:.2f} mm")


def laser_editor(las, points) -> None:
    name = las["name"]
    cio.set_if_changed(las, "enabled", st.toggle("enabled", value=bool(las.get("enabled", True)),
                                                 key=k("las", name, "on")), default=True)
    pose_editor(las["pose"], f"las:{name}", points)
    pat = las["pattern"]
    kinds = ["line", "grid", "random"]
    if pat["type"] not in kinds:
        st.caption(f"Pattern type {pat['type']} is edited in the YAML.")
    else:
        kind = st.selectbox("Pattern", kinds, index=kinds.index(pat["type"]), key=k("las", name, "type"))
        if kind != pat["type"]:
            las["pattern"] = cio.flow_map({kk: (cio.flow(v) if isinstance(v, list) else v)
                                           for kk, v in PATTERN_DEFAULTS[kind].items()})
            pat = las["pattern"]
        c1, c2 = st.columns(2)
        if kind == "line":
            cio.set_if_changed(pat, "fan_deg", c1.number_input("Fan angle (deg)", value=float(pat["fan_deg"]),
                                                              step=1.0, min_value=0.5, max_value=169.0,
                                                              key=k("las", name, "fan")))
            cio.set_if_changed(pat, "angle_deg", c2.number_input("Line orientation (deg)",
                                                                value=float(pat.get("angle_deg", 0.0)),
                                                                step=5.0, key=k("las", name, "ang")))
        elif kind == "grid":
            if "pitch_deg" in pat:  # shown as a fan angle; converted only if edited
                pitch = pat["pitch_deg"]
                fan0 = [pitch[0] * (pat["cols"] - 1), pitch[1] * (pat["rows"] - 1)]
            else:
                fan0 = list(pat["fan_deg"])
            cio.set_if_changed(pat, "cols", int(c1.number_input("Columns", value=int(pat["cols"]), step=1,
                                                                 min_value=1, key=k("las", name, "cols"))))
            cio.set_if_changed(pat, "rows", int(c2.number_input("Rows", value=int(pat["rows"]), step=1,
                                                                 min_value=1, key=k("las", name, "rows"))))
            fx = c1.number_input("Fan x (deg)", value=float(fan0[0]), step=1.0, key=k("las", name, "fx"))
            fy = c2.number_input("Fan y (deg)", value=float(fan0[1]), step=1.0, key=k("las", name, "fy"))
            if abs(fx - fan0[0]) > 1e-9 or abs(fy - fan0[1]) > 1e-9:
                pat.pop("pitch_deg", None)
                pat["fan_deg"] = cio.flow([fx, fy])
        else:
            cio.set_if_changed(pat, "count", int(c1.number_input("Dots", value=int(pat["count"]), step=10,
                                                                  min_value=1, key=k("las", name, "n"))))
            fx = c1.number_input("Field x (deg)", value=float(pat["fov_deg"][0]), step=1.0, key=k("las", name, "fx"))
            fy = c2.number_input("Field y (deg)", value=float(pat["fov_deg"][1]), step=1.0, key=k("las", name, "fy"))
            cio.set_if_changed(pat, "fov_deg", [fx, fy])
    cio.set_if_changed(las, "power_w", num("Power (relative W)", las.get("power_w", 20.0), f"las:{name}:pw",
                                           10.0, "%.1f", min_value=0.0), default=20.0)


def wheel_editor(doc) -> None:
    names = profiles.available("wheels")
    cur = doc.get("wheel_profile", "AAR_2A_ref_wide")
    prof = st.selectbox("Wheel profile (assets/profiles/wheels)", names,
                        index=names.index(cur) if cur in names else 0, key=k("wheel", "profile"))
    cio.set_if_changed(doc, "wheel_profile", prof)
    for field, label, step in WHEEL_FIELDS:
        v = doc.get(field, 0.0)
        spec = is_spec(v)
        mode = st.segmented_control(label, ["fixed", "random range"], default="random range" if spec else "fixed",
                                    key=k("wheel", field, "mode"))
        if mode == "random range":
            lo0, hi0 = (v["uniform"] if spec and "uniform" in v else [float(layout_value(v)) - 10 * step,
                                                                       float(layout_value(v)) + 10 * step])
            c1, c2 = st.columns(2)
            lo = c1.number_input("from", value=float(lo0), step=step, format="%.3f", key=k("wheel", field, "lo"))
            hi = c2.number_input("to", value=float(hi0), step=step, format="%.3f", key=k("wheel", field, "hi"))
            new = {"uniform": [round(lo, 4), round(hi, 4)]}
            if cio.plain(v) != new:
                doc[field] = cio.flow_map({"uniform": cio.flow([lo, hi])})
        else:
            val = st.number_input("value", value=float(layout_value(v)), step=step, format="%.3f",
                                  key=k("wheel", field, "val"))
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


def colourise(a: np.ndarray, valid=None) -> np.ndarray:
    v = a[np.isfinite(a) & (a > 0)] if valid is None else a[valid]
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


def images_tab(run_stem: str) -> None:
    from synthrig.dataset import Dataset

    root = data_root() / "gui" / run_stem
    sets = sorted((m.parent for m in root.glob("*/manifest.json")), reverse=True)
    if not sets:
        st.info("No renders yet for this run. Use the Render button above.")
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
    cams = list(bits)
    cols = st.columns(2)
    for i, cam in enumerate(cams):
        img = fr.image(cam)
        if view == "camera images":
            out = display(img, bits[cam])
        elif view == "laser masks":
            out = (display(img, bits[cam]) * 0.45).astype(np.uint8)
            for j, lname in enumerate(lasers):
                if fr.has(f"{cam}/laser_{lname}_mask.png"):
                    m = fr.laser_mask(cam, lname) > 0
                    out[m] = [(255, 60, 60), (60, 255, 60), (90, 160, 255)][j % 3]
        elif view == "depth":
            out = colourise(fr.depth(cam))
        else:
            out = seg_colours(fr.seg(cam))
        cols[i % 2].image(out, caption=f"{cam} ({img.shape[1]}x{img.shape[0]})", width="stretch")
    with st.expander("frame parameters (meta.json)"):
        st.json({k_: fr.meta[k_] for k_ in ("frame", "seed", "scene_params", "timings_s") if k_ in fr.meta})


# --------------------------------------------------------------------------- page
runs = sorted(RUNS.glob("*.yaml"))
with st.sidebar:
    st.header("synthrig")
    default = next((i for i, r in enumerate(runs) if r.stem == "wheel_rail"), 0)
    run = st.selectbox("Run config", runs, index=default, format_func=lambda p: p.name)
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
    status = st.empty()

tab_layout, tab_render, tab_3d = st.tabs(["Layout", "Render & images", "3D view"])
is_track = S.rig_doc.get("frame") == "track" and S.run_doc.get("scene") == "wheel_rail"

with tab_layout:
    if not is_track:
        st.warning("The layout editor supports wheel_rail runs with a `frame: track` rig "
                   "(e.g. configs/runs/wheel_rail.yaml).")
    else:
        left, right = st.columns([1, 2.2])
        try:
            points = key_points(scene_params())
        except Exception:
            points = {}
        with left:
            st.subheader("Wheel")
            wheel_editor(S.scene_doc)
            try:
                points = key_points(scene_params())
            except Exception:
                pass
            st.subheader("Cameras")
            models = {}
            try:
                models = RigModel.build(Rig.model_validate(cio.plain(S.rig_doc))).cameras
            except Exception:
                pass
            for cam in S.rig_doc["cameras"]:
                with st.expander(cam["name"], expanded=False):
                    if cam["name"] in models:
                        camera_editor(cam, models[cam["name"]], points)
            st.subheader("Lasers")
            for las in S.rig_doc.get("lasers", []):
                with st.expander(las["name"], expanded=False):
                    laser_editor(las, points)
        rig, p = validate_and_save(autosave)
        with right:
            if rig is not None:
                L = compute_layout(rig.canonical_json(), p.model_dump_json())
                model = RigModel.build(rig)
                st.plotly_chart(figures.layout_3d(model, L, key_points(p)), key="fig3d")
                st.caption("Camera views (approximate: coarse geometry, no occlusion); % = laser points in view")
                vcols = st.columns(2)
                for i, cname in enumerate(model.cameras):
                    vcols[i % 2].plotly_chart(figures.camera_view(model, L, cname), key=f"view_{cname}")
                with st.expander("Aim points (track frame, m)"):
                    st.dataframe({n: v for n, v in key_points(p).items()})

with tab_render:
    c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1.4])
    n_frames = c1.number_input("Frames", 1, 1000, 1)
    scale = c2.selectbox("Scale", [0.125, 0.25, 0.5, 1.0], index=2)
    samples = c3.selectbox("Samples", [8, 16, 32, 64, 128], index=2)
    device = c4.selectbox("Device", ["GPU", "CPU"])
    if c5.button("Render", type="primary", disabled=job_running(), width="stretch"):
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        out = Path("gui") / run.stem / stamp
        start_job("render", [sys.executable, "-m", "synthrig", str(run), "--frames", str(n_frames),
                             "--scale", str(scale), "--samples", str(samples), "--device", device,
                             "--out", str(out)], data_root() / out.with_suffix(".log"), frames=n_frames)
    job_status("render")
    images_tab(run.stem)

with tab_3d:
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
            out = data_root() / "gui" / run.stem / "overview" / f"{stamp}.png"
            w, h = size.split("x")
            cmd = [sys.executable, "-m", "synthrig.render.observer", str(run), "--out", str(out),
                   "--frame", str(frame3), "--target", *map(str, pts3[target]), "--azimuth", str(az),
                   "--elevation", str(el), "--distance", str(dist), "--hfov", str(hf), "--size", w, h,
                   "--samples", str(spp)] + ([] if markers else ["--no-markers"])
            start_job("overview", cmd, out.with_suffix(".log"))
        job_status("overview")
        shots = sorted((data_root() / "gui" / run.stem / "overview").glob("*.png"), reverse=True)
        if shots:
            pick = st.selectbox("Rendered views", shots, format_func=lambda q: q.stem)
            st.image(str(pick), width="stretch")
        else:
            st.info("No 3D views yet: set the viewpoint and press Render 3D view.")

status.caption(S.save_msg or "no changes saved yet")
