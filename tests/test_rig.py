"""Fast tests, no Blender needed."""
import numpy as np
import pytest

from synthrig.rig import RigModel, load_rig
from synthrig.rig.schema import Rig
from synthrig.rig.transforms import CV_TO_BLENDER_OBJ, look_at_R, pose_to_T
from synthrig.sensor import apply_sensor

RIG = "configs/rigs/demo_4cam_2laser.yaml"


def test_rig_loads_and_hash_stable():
    a, b = load_rig(RIG), load_rig(RIG)
    assert a.hash() == b.hash()
    m = RigModel.build(a)
    assert set(m.cameras) == {"cam0", "cam1", "cam2", "cam3"}
    s = m.stereo["front"]
    assert abs(np.linalg.norm(s.T) - 0.12) < 1e-9 and not s.vertical


def test_bad_rig_rejected():
    d = load_rig(RIG).model_dump()
    d["stereo_pairs"][0]["right"] = "nope"
    with pytest.raises(ValueError):
        Rig.model_validate(d)


def test_look_at_frame():
    R = look_at_R([0, 0, 0], [0, 0, 5], [0, -1, 0])
    assert np.allclose(R, np.eye(3))
    R = look_at_R([1, 0, 0], [0, 0, 1], [0, -1, 0])
    assert np.allclose(R[:, 2], np.array([-1, 0, 1]) / np.sqrt(2))
    assert np.isclose(np.linalg.det(R), 1)


def test_blender_conversion_is_proper_rotation():
    assert np.isclose(np.linalg.det(CV_TO_BLENDER_OBJ[:3, :3]), 1)


def test_distortion_render_plan_roundtrip():
    cam = RigModel.build(load_rig(RIG), 0.25).cameras["cam2"]
    plan = cam.render_plan()
    # a pinhole-canvas pixel -> ray -> project with distortion -> remap back must agree
    rng = np.random.default_rng(0)
    uv = rng.uniform([0, 0], np.array(cam.size) - 1, (200, 2))
    n = cam.undistort_to_normalized(uv)
    X = np.concatenate([n, np.ones((len(n), 1))], 1)
    assert np.abs(cam.project(X) - uv).max() < 1e-6
    iu, iv = np.rint(uv).astype(int).T
    ni = cam.undistort_to_normalized(np.stack([iu, iv], 1).astype(float))
    assert np.abs(plan.map_x[iv, iu] - (ni[:, 0] * cam.K[0, 0] + plan.K_render[0, 2])).max() < 1e-3
    assert np.abs(plan.map_y[iv, iu] - (ni[:, 1] * cam.K[1, 1] + plan.K_render[1, 2])).max() < 1e-3
    assert (plan.map_x >= 0).all() and (plan.map_x <= plan.size_render[0] - 1).all()
    assert (plan.map_y >= 0).all() and (plan.map_y <= plan.size_render[1] - 1).all()


def test_laser_texture_centroids_match_pattern():
    L = RigModel.build(load_rig(RIG)).lasers["red"]
    img, (hx, hy) = L.texture()
    H, W = img.shape
    for tx, ty in L.tangents[:20]:
        ex, ey = (tx + hx) * W / (2 * hx), (ty + hy) * H / (2 * hy)
        i, j = int(ex), int(ey)
        P = img[j - 6:j + 7, i - 6:i + 7].astype(float)
        yy, xx = np.mgrid[j - 6:j + 7, i - 6:i + 7] + 0.5
        assert abs((P * xx).sum() / P.sum() - ex) < 0.02 and abs((P * yy).sum() / P.sum() - ey) < 0.02


def test_sensor_saturation_and_bits():
    s = load_rig(RIG).camera("cam0").sensor
    rad = np.full((32, 32, 3), 5.0)
    out = apply_sensor(rad, s, np.random.default_rng(0))
    assert out.image.max() == 2 ** s.bit_depth - 1 and out.saturated.all()
    dark = apply_sensor(np.zeros((32, 32, 3)), s, np.random.default_rng(0))
    assert dark.image.mean() < 5
