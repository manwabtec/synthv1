"""Per-scene render validation (docs/TASK_scenes.md Phase 4). Slow: renders with Blender.

Each scene renders 1 frame at --scale 0.125 --samples 4 (GPU if the container has one).
"""
import json
import os
from pathlib import Path

import numpy as np
import pytest
import yaml

pytest.importorskip("bpy")

from eval.common import disparity_at_dots, dot_gt_bias, dot_lasers  # noqa: E402
from synthrig.assets.classes import CLASSES  # noqa: E402
from synthrig.assets.wheelrail import WheelRailParams  # noqa: E402
from synthrig.dataset import Dataset  # noqa: E402
from synthrig.paths import CODE_ROOT  # noqa: E402

pytestmark = pytest.mark.slow
SCENES = {  # scene -> (min visible GT dots/line samples in cam0, params model)
    "wheel_rail": (300, WheelRailParams),
    "wheel_tread": (60, WheelRailParams),  # cam0 sees the field-side dots only
}


@pytest.fixture(scope="module")
def rendered(tmp_path_factory):
    from synthrig.pipeline import Pipeline, RunConfig

    out = {}
    for scene in SCENES:
        cfg = RunConfig.load(CODE_ROOT / f"configs/{scene}/run.yaml")
        cfg.frames, cfg.render.render_scale, cfg.render.samples = 1, 0.125, 4
        cfg.render.device = "GPU" if os.path.exists("/dev/nvidiactl") else "CPU"
        cfg.output = str(tmp_path_factory.mktemp(scene))
        Pipeline(cfg, log=lambda _m: None).run()
        out[scene] = Path(cfg.output)
    return out


def _frame(path):
    ds = Dataset(path)
    return ds, next(iter(ds.frames()))


def _classes_by_pixel(frame, cam):
    names = {int(k): v["class"] for k, v in frame.meta["objects"].items()}
    seg = frame.seg(cam)
    return {names.get(int(i), "background"): int((seg == i).sum()) for i in np.unique(seg) if i > 0}, seg.size


@pytest.mark.parametrize("scene", list(SCENES))
def test_segmentation_uses_the_taxonomy(rendered, scene):
    ds, fr = _frame(rendered[scene])
    assert ds.manifest["classes"] == list(CLASSES)
    for cam in ds.manifest["cameras"]:
        counts, _ = _classes_by_pixel(fr, cam)
        assert set(counts) <= set(CLASSES), set(counts) - set(CLASSES)


@pytest.mark.parametrize("scene", list(SCENES))
def test_enough_laser_points_visible(rendered, scene):
    _, fr = _frame(rendered[scene])
    assert int(fr.dots()["visible_cam0"].sum()) >= SCENES[scene][0]


@pytest.mark.parametrize("scene", list(SCENES))
def test_dot_ground_truth_is_unbiased(rendered, scene):
    ds, fr = _frame(rendered[scene])
    rig = yaml.safe_load((rendered[scene] / "rig.yaml").read_text())
    names = list(ds.manifest["lasers"])
    offs = [dot_gt_bias(fr, cam, l, names.index(l)) for cam in ds.manifest["cameras"] for l in dot_lasers(rig)]
    offs = np.concatenate(offs)
    assert len(offs) >= 20, f"only {len(offs)} isolated dots"
    assert np.linalg.norm(offs.mean(0)) < 0.05, offs.mean(0)


@pytest.mark.parametrize("scene", list(SCENES))
def test_gt_disparity_matches_dot_pairs(rendered, scene):
    """GT disparity vs the rectified GT dot pairs.

    The GT map reads depth at the nearest pinhole-canvas pixel (no mixing across edges,
    AGENTS rule 7). A half-pixel lookup offset on the canvas becomes zoom x 0.5 px on
    the rectified grid (zoom = rectified / canvas focal length; alpha=0 crops and zooms),
    so the expected error at a dot is <= 0.5 x zoom x local disparity gradient. Check that
    bound for 95% of dots (the rest lie on occlusion edges)."""
    ds, fr = _frame(rendered[scene])
    if not ds.manifest["stereo"]:
        pytest.skip(f"{scene} has no stereo pairs")
    for pair, lr in ds.manifest["stereo"].items():
        zoom = ds.stereo_calib[pair]["P1"][0, 0] / ds.cameras[lr["left"]]["K"][0, 0]
        d, grad = disparity_at_dots(fr, pair, lr["left"], lr["right"], with_gradient=True)
        assert len(d) >= 20
        within = np.abs(d) <= 0.5 * zoom * grad + 0.02
        assert within.mean() >= 0.95, (within.mean(), zoom)
        assert abs(np.median(d)) < 0.05, np.median(d)  # no systematic offset


@pytest.mark.parametrize("scene", list(SCENES))
def test_meta_params_validate(rendered, scene):
    _, fr = _frame(rendered[scene])
    SCENES[scene][1](**fr.meta["scene_params"])
    json.dumps(fr.meta)


def test_wheel_rail_acceptance(rendered):
    """Wheel field-side face + rail head in the stereo pair; both lasers land on the wheel."""
    _, fr = _frame(rendered["wheel_rail"])
    for cam in ("cam0", "cam1"):
        counts, _ = _classes_by_pixel(fr, cam)
        assert counts.get("wheel_rim_face", 0) + counts.get("wheel_plate", 0) > 0 and counts.get("rail_head", 0) > 0
    d = fr.dots()
    names = {int(k): v["class"] for k, v in fr.meta["objects"].items()}
    for li in range(len(d["laser_names"])):
        hit_cls = {names.get(int(o), "") for o in d["object_index"][(d["laser"] == li) & d["hit"]]}
        assert any(c.startswith("wheel_") for c in hit_cls), (d["laser_names"][li], hit_cls)


def test_wheel_tread_acceptance(rendered):
    """Tread + flange >= 40% of cam0 pixels, and dots land on the tread."""
    _, fr = _frame(rendered["wheel_tread"])
    counts, size = _classes_by_pixel(fr, "cam0")
    assert (counts.get("wheel_tread", 0) + counts.get("wheel_flange", 0)) / size >= 0.40, counts
    d = fr.dots()
    names = {int(k): v["class"] for k, v in fr.meta["objects"].items()}
    assert any(names.get(int(o)) == "wheel_tread" for o in d["object_index"][d["hit"]])
