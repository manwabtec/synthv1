"""synthrig GUI: edit the wheel/rail layout, render, and inspect results in a browser.

Start it with `python3 infra/cloud.py gui` (runs in the Docker image with the GPU), then
open http://localhost:8501 (VS Code forwards the port when you work on EC2).

Tabs:
  Layout          the layout viewer (tools/gui/layout_view): sliders on the left, a WebGL
                  3D view, live camera views and the profile-coverage inset. It computes
                  everything in the browser; Python only stores its edits in the YAML files
                  (comments kept) when a slider is released.
  Render & images render the layout as it is in the GUI and browse each camera's images
  3D view         Blender render of the whole scene from a free viewpoint, with markers
"""
from __future__ import annotations

import copy
import datetime as dt
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from synthrig.assets.wheelrail import WheelRailParams, key_points  # noqa: E402
from synthrig.opticsrig.schema import Rig  # noqa: E402
from synthrig.paths import CODE_ROOT, data_root  # noqa: E402
from synthrig.sampling import is_spec  # noqa: E402
from tools.gui import configio as cio  # noqa: E402
from tools.gui import layout_data  # noqa: E402

st.set_page_config(page_title="synthrig", layout="wide", initial_sidebar_state="collapsed")
S = st.session_state
CONFIGS = CODE_ROOT / "configs"
LAYOUT_SCENES = {"wheel_rail", "wheel_tread"}  # scenes built from WheelRailParams
LAYOUT_VIEW = components.declare_component("layout_view", path=str(Path(__file__).parent / "layout_view"))

# full-width page: no Streamlit header, sidebar or wide margins
st.markdown("""<style>
header[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"] {display: none !important;}
.block-container, [data-testid="stMainBlockContainer"] {padding: 0.4rem 0.8rem 0 0.8rem !important; max-width: 100% !important;}
[data-testid="stTabs"] [data-baseweb="tab-list"] {gap: 1.2rem;}
</style>""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- state
def load_run(run: Path) -> None:
    run_doc = yaml.safe_load(run.read_text())
    rig_path = (run.parent / run_doc["rig"]).resolve()
    scene_path = (run.parent / run_doc["scene_config"]).resolve() if run_doc.get("scene_config") else None
    S.run, S.run_doc, S.rig_path, S.scene_path = run, run_doc, rig_path, scene_path
    S.rig_doc = cio.load(rig_path)
    S.scene_doc = cio.load(scene_path) if scene_path else None
    S.saved = {"rig": cio.dumps(S.rig_doc), "scene": cio.dumps(S.scene_doc) if S.scene_doc is not None else ""}
    S.gen = S.get("gen", 0) + 1  # the layout viewer re-reads the rig when this changes
    S.layout_status = {"text": f"loaded {rel(rig_path)}" + (f" + {rel(scene_path)}" if scene_path else ""), "level": ""}


def rel(path: Path) -> str:
    return str(path.relative_to(CODE_ROOT))


def layout_value(v):
    """A representative value of a fixed value or sampling spec (for the layout view)."""
    if not is_spec(v):
        return v
    kind, arg = next(iter(v.items()))
    return {"uniform": (arg[0] + arg[1]) / 2, "normal": arg[0], "choice": arg[0]}[kind]


def scene_params(doc=None) -> WheelRailParams:
    doc = cio.plain(S.scene_doc if doc is None else doc) or {}
    return WheelRailParams(**{key: layout_value(v) for key, v in doc.items()})


def apply_layout_edits(edits: dict, autosave: bool) -> dict:
    """Apply the viewer's edits; keep them only if the result validates. Returns a status."""
    rig_doc, scene_doc = copy.deepcopy(S.rig_doc), copy.deepcopy(S.scene_doc)
    layout_data.apply_edits(rig_doc, scene_doc, edits)
    try:
        Rig.model_validate(cio.plain(rig_doc))
        if scene_doc is not None:
            scene_params(scene_doc)
    except Exception as exc:  # pydantic ValidationError and friends: report, keep the old docs
        return {"text": f"not applied, invalid: {exc}", "level": "err"}
    S.rig_doc, S.scene_doc = rig_doc, scene_doc
    if not autosave:
        return {"text": "saving is off: edits live in the GUI only (renders still use them)", "level": "warn"}
    saved = []
    for kind, doc, path in (("rig", S.rig_doc, S.rig_path), ("scene", S.scene_doc, S.scene_path)):
        if doc is None:
            continue
        text = cio.dumps(doc)
        if text != S.saved[kind]:
            cio.save(doc, path)
            S.saved[kind] = text
            saved.append(rel(path))
    if not saved:
        return {"text": "no change to save", "level": ""}
    return {"text": f"saved {', '.join(saved)} at {dt.datetime.now():%H:%M:%S}", "level": "ok"}


def snapshot_run(scene: str, stamp: str) -> Path:
    """The run config with the rig and scene exactly as they are in the GUI (saved or not),
    written to <data root>/gui/<scene>/configs/<stamp>/. Renders use this."""
    d = data_root() / "gui" / scene / "configs" / stamp
    d.mkdir(parents=True, exist_ok=True)
    rig = copy.deepcopy(S.rig_doc)
    for las in rig.get("lasers", []):  # file patterns are relative to the original rig file
        pat = las.get("pattern", {})
        if pat.get("type") == "file" and not Path(pat["path"]).is_absolute():
            pat["path"] = str((S.rig_path.parent / pat["path"]).resolve())
    cio.save(rig, d / "rig.yaml")
    run_doc = dict(S.run_doc, rig="rig.yaml")
    if S.scene_doc is not None:
        cio.save(S.scene_doc, d / "scene.yaml")
        run_doc["scene_config"] = "scene.yaml"
    (d / "run.yaml").write_text(yaml.safe_dump(run_doc, sort_keys=False))
    return d / "run.yaml"


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
def start_overview(run: Path, view: dict, frame: int = 0, samples: int = 64, markers: bool = True) -> None:
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    scene = run.parent.name
    out = data_root() / "gui" / scene / "overview" / f"{stamp}.png"
    w, h = view.get("size", [1280, 800])
    cmd = [sys.executable, "-m", "synthrig.render.observer", str(snapshot_run(scene, stamp)), "--out", str(out),
           "--frame", str(frame), "--target", *map(str, view["target"]), "--azimuth", str(view["azimuth"]),
           "--elevation", str(view["elevation"]), "--distance", str(view["distance"]),
           "--hfov", str(view["hfov_deg"]), "--size", str(w), str(h), "--samples", str(samples)]
    start_job("overview", cmd + ([] if markers else ["--no-markers"]), out.with_suffix(".log"))


@st.fragment
def layout_tab(autosave: bool) -> None:
    if not (S.rig_doc.get("frame") == "track" and S.run_doc.get("scene") in LAYOUT_SCENES):
        st.warning("The layout viewer supports wheel_rail / wheel_tread runs with a `frame: track` rig.")
        return
    try:
        state = layout_data.view_state(S.rig_doc, S.scene_doc)
        geom = layout_data.geometry(state["scene"]["wheel_profile"], state["scene"]["rail_profile"])
    except Exception as exc:
        st.error(f"Invalid configuration: {exc}")
        return
    val = LAYOUT_VIEW(rev=S.gen, geometry=geom, state=state, status=S.layout_status, key="layout_view", default=None)
    if not val or val.get("rev") != S.gen or val.get("id") == S.get("layout_last_id"):
        return
    S.layout_last_id = val["id"]
    S.layout_status = apply_layout_edits(val.get("edits") or {}, autosave)
    action = val.get("action") or {}
    if action.get("type") == "render_view":
        if job_running():
            S.layout_status = {"text": "a render is already running; try again when it has finished", "level": "warn"}
        else:
            start_overview(S.run, action["view"])
            S.layout_status = {"text": "Blender render of this view started: see the 3D view tab", "level": "ok"}
    st.rerun(scope="fragment")  # hand the new status (and geometry, if the profile changed) to the viewer


@st.fragment
def render_tab(run: Path) -> None:
    st.caption("Renders the layout exactly as it is in the GUI (saved to the YAML files or not).")
    c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1.4])
    n_frames = c1.number_input("Frames", 1, 1000, 1)
    scale = c2.selectbox("Scale", [0.125, 0.25, 0.5, 1.0], index=2)
    samples = c3.selectbox("Samples", [8, 16, 32, 64, 128], index=2)
    device = c4.selectbox("Device", ["GPU", "CPU"])
    if c5.button("Render", type="primary", disabled=job_running(), width="stretch"):
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        out = Path("gui") / run.parent.name / stamp
        start_job("render", [sys.executable, "-m", "synthrig", str(snapshot_run(run.parent.name, stamp)),
                             "--frames", str(n_frames), "--scale", str(scale), "--samples", str(samples),
                             "--device", device, "--out", str(out)], data_root() / out.with_suffix(".log"),
                  frames=n_frames)
    job_status("render")
    images_view(run.parent.name)


@st.fragment
def view3d_tab(run: Path) -> None:
    c1, c2 = st.columns([1, 3.2])
    with c1:
        st.caption("Tip: the layout tab's **Render this view in Blender** button renders its current viewpoint.")
        try:
            pts3 = key_points(scene_params())
        except Exception:
            pts3 = {"wheel_center": [0.0, -0.67, 0.46]}
        target = st.selectbox("Look at", list(pts3), index=0)
        az = st.slider("Azimuth (deg, 0 = from the field side)", -180, 180, -35)
        el = st.slider("Elevation (deg)", -10, 85, 20)
        dist = st.slider("Distance (m)", 0.3, 6.0, 2.2, 0.1)
        hf = st.slider("Field of view (deg)", 15, 100, 50)
        size = st.selectbox("Resolution", ["960x600", "1280x800", "1920x1200"], index=1)
        spp = st.selectbox("Samples ", [16, 32, 64, 128, 256], index=2)
        frame3 = st.number_input("Frame (scene sampling)", 0, 10000, 0)
        markers = st.toggle("Show camera/laser markers", value=True)
        go_btn = st.button("Render 3D view", type="primary", disabled=job_running(), width="stretch")
    with c2:
        if go_btn:
            w, h = size.split("x")
            start_overview(run, {"target": pts3[target], "azimuth": az, "elevation": el, "distance": dist,
                                 "hfov_deg": hf, "size": [int(w), int(h)]}, frame3, spp, markers)
        job_status("overview")
        shots = sorted((data_root() / "gui" / run.parent.name / "overview").glob("*.png"), reverse=True)
        if shots:
            pick = st.selectbox("Rendered views", shots, format_func=lambda q: q.stem)
            st.image(str(pick), width="stretch")
        else:
            st.info("No 3D views yet: set the viewpoint and press Render 3D view.")


# --------------------------------------------------------------------------- page
runs = sorted(CONFIGS.glob("*/run.yaml"))
top = st.columns([1.4, 1.2, 0.8, 4], vertical_alignment="center")
default = next((i for i, r in enumerate(runs) if r.parent.name == "wheel_tread"), 0)
run = top[0].selectbox("Scene", runs, index=default, format_func=lambda p: f"scene: {p.parent.name}",
                       label_visibility="collapsed")
if S.get("run") != run:
    load_run(run)
autosave = top[1].toggle("Save edits to the YAML files", value=True,
                         help="Valid edits are written when you release a slider (comments are kept). "
                              "Git has the history: `git diff configs/` shows them, `git checkout configs/<scene>/` undoes them.")
if top[2].button("Reload files", help="Discard GUI state and re-read the YAML files"):
    load_run(run)
    st.rerun()
top[3].caption(f"{rel(S.rig_path)}" + (f" · {rel(S.scene_path)}" if S.scene_path else ""))

tab_layout, tab_render, tab_3d = st.tabs(["Layout", "Render & images", "3D view"])
with tab_layout:
    layout_tab(autosave)
with tab_render:
    render_tab(run)
with tab_3d:
    view3d_tab(run)
