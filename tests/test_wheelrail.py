"""Rig schema extensions, scene-config sampling, profiles and wheel/rail geometry."""
import json
import math

import numpy as np
import pytest
import yaml

from synthrig.assets import profiles, wheelrail
from synthrig.opticsrig import RigModel, load_rig
from synthrig.opticsrig.laser import pattern_tangents
from synthrig.opticsrig.schema import GridPattern, Intrinsics, LinePattern, Rig
from synthrig.paths import CODE_ROOT
from synthrig.sampling import sample_config

RIG = CODE_ROOT / "configs/wheel_rail/rig.yaml"


def test_hfov_sets_focal_length():
    i = Intrinsics(width=2000, height=1000, pixel_size_um=5, hfov_deg=90)
    assert i.fx == pytest.approx(1000.0)
    with pytest.raises(ValueError, match="disagree"):
        Intrinsics(width=2000, height=1000, pixel_size_um=5, hfov_deg=90, focal_length_mm=8)
    again = Intrinsics.model_validate(i.model_dump())  # a validated dump (fx + hfov) re-validates
    assert again.fx == pytest.approx(i.fx)
    rig = load_rig(RIG)
    assert Rig.model_validate(rig.model_dump()).hash() == rig.hash()


def test_grid_fan_angle_spans_outer_dots():
    t = pattern_tangents(GridPattern(rows=3, cols=5, fan_deg=[20, 10]))
    assert math.degrees(math.atan(t[:, 0].max())) == pytest.approx(10.0)
    assert math.degrees(math.atan(t[:, 1].max())) == pytest.approx(5.0)


def test_line_pattern_is_a_dense_fan():
    t = pattern_tangents(LinePattern(fan_deg=40, angle_deg=90), dot_sigma_rad=6e-4)
    assert np.allclose(t[:, 0], 0, atol=1e-12)  # rotated 90 deg: line along projector y
    assert math.degrees(math.atan(t[:, 1].max())) == pytest.approx(20.0)
    step = np.diff(np.arctan(t[:, 1]))
    assert step.max() <= 3e-4 + 1e-9  # half the dot sigma -> continuous line


def test_track_frame_rig_uses_z_up():
    rig = load_rig(RIG)
    assert rig.frame == "track" and rig.cameras[0].pose.up == [0.0, 0.0, 1.0]
    model = RigModel.build(rig)
    R = model.cameras["cam0"].T_rig_cam[:3, :3]
    assert R[2, 1] < 0  # camera y (image down) points down in the world
    assert np.allclose(model.cameras["cam0"].T_rig_cam[:3, 3], [-0.06, -1.45, 0.30])
    demo = Rig.model_validate(yaml.safe_load((CODE_ROOT / "configs/debug_clutter/rig.yaml").read_text()))
    assert demo.frame == "rig" and demo.cameras[0].pose.up == [0.0, -1.0, 0.0]


def test_sampling_is_deterministic_and_order_independent():
    cfg = {"a": 1.0, "b": {"uniform": [0, 1]}, "c": {"choice": ["x", "y"]}, "d": {"normal": [0, 1]}}
    s1 = sample_config(cfg, np.random.default_rng([7, 3]))
    s2 = sample_config(dict(reversed(list(cfg.items()))), np.random.default_rng([7, 3]))
    assert s1 == s2 and s1["a"] == 1.0 and 0 <= s1["b"] < 1 and s1["c"] in ("x", "y")


def test_wheel_profile_is_ordered_and_split():
    c = profiles.load_wheel_contour("AAR_2A_ref_wide")
    assert c.a[0] < 0.001 and c.a[-1] == pytest.approx(0.1453)
    assert np.linalg.norm(np.diff(c.points, axis=0), axis=1).max() < 0.001
    parts = profiles.wheel_parts(c, profiles.WheelSection())
    assert set(parts) == {"wheel_flange", "wheel_tread", "wheel_rim_face", "wheel_plate", "wheel_hub"}
    assert parts["wheel_flange"][:, 1].max() > parts["wheel_tread"][:, 1].max()  # flange tip is highest


def test_wheel_sits_on_the_right_rail():
    p = wheelrail.WheelRailParams()
    k = wheelrail.key_points(p)
    head = [poly for side, part, poly in wheelrail.rail_sections(p) if side == "right" and part == "head"][0]
    assert head[:, 0].min() < k["tread_contact"][1] < head[:, 0].max()  # tread contact over rail head
    assert k["flange_tip_bottom"][1] > head[:, 0].max()  # flange inside the gauge face
    assert k["flange_tip_bottom"][2] < 0  # flange reaches below the rail top
    shifted = wheelrail.WheelRailParams(wheel_x_m=0.4, lateral_shift_m=0.005)
    assert wheelrail.key_points(shifted)["wheel_center"][0] == 0.4
    assert wheelrail.wheel_origin(shifted)[1] - wheelrail.wheel_origin(p)[1] == pytest.approx(0.005)


def test_registered_scene_params_roundtrip_and_repeat():
    pytest.importorskip("bpy")
    from synthrig.pipeline import RunConfig, load_scene

    cfg = RunConfig.load(CODE_ROOT / "configs/wheel_rail/run.yaml")
    scene = load_scene(cfg.scene, cfg.scene_params, cfg.scene_config)
    p1 = scene.params(np.random.default_rng([cfg.seed, 5]))
    p2 = scene.params(np.random.default_rng([cfg.seed, 5]))
    assert p1 == p2 and json.loads(json.dumps(p1)) == p1
    assert wheelrail.WheelRailParams(**p1).wheel_profile == "AAR_2A_ref_wide"
