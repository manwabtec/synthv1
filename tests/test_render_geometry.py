"""Blender tests: the rendered image must agree with the OpenCV camera model.
Uses a portrait, off-centre, distorted camera to exercise every intrinsics path."""
import numpy as np
import pytest

bpy = pytest.importorskip("bpy")

from synthrig.render.blender_backend import BlenderBackend, RenderSettings  # noqa: E402
from synthrig.rig import RigModel  # noqa: E402
from synthrig.rig.schema import Rig  # noqa: E402

RIG = {
    "name": "t", "cameras": [{
        "name": "c", "intrinsics": {"width": 200, "height": 320, "pixel_size_um": 5, "fx": 400,
                                    "cx": 87.3, "cy": 171.6},
        "distortion": [-0.15, 0.05, 0.001, -0.002, 0.0],
        "pose": {"position": [0.05, 0.02, 0], "euler_xyz_deg": [3, -4, 10]}}],
    "lasers": [{"name": "l", "color_rgb": [1, 1, 1], "pose": {"position": [0, 0, 0]},
                "pattern": {"type": "grid", "rows": 3, "cols": 3, "pitch_deg": [4, 4]}, "power_w": 500}],
}


def _emitter(loc):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.004, location=loc, segments=32, ring_count=16)
    ob = bpy.context.active_object
    m = bpy.data.materials.new("e")
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs[1].default_value = 50
    nt.links.new(em.outputs[0], nt.nodes.new("ShaderNodeOutputMaterial").inputs[0])
    ob.data.materials.append(m)


def test_projection_matches_opencv(tmp_path):
    rig = RigModel.build(Rig.model_validate(RIG))
    cam = rig.cameras["c"]
    be = BlenderBackend(RenderSettings(samples=16, filter_width=1.0), tmp_dir=str(tmp_path))
    be.new_frame(rig)
    be.set_ambient((0, 0, 0), 0)
    for o in list(bpy.data.objects):
        if o.name.startswith("laser_"):
            bpy.data.objects.remove(o)
    X_cam = np.array([[0.03, -0.05, 1.0], [-0.06, 0.08, 1.2], [0.0, 0.0, 0.9]])
    T = cam.T_rig_cam
    for x in X_cam:
        _emitter((T[:3, :3] @ x + T[:3, 3]).tolist())
    plan = cam.render_plan()
    r = be.render_camera("c", plan)
    img = plan.apply(r.combined).sum(-1)
    uv = cam.project(X_cam)
    for u, v in uv:
        i, j = int(round(u)), int(round(v))
        P = img[j - 5:j + 6, i - 5:i + 6]
        yy, xx = np.mgrid[j - 5:j + 6, i - 5:i + 6]
        c = np.array([(P * xx).sum(), (P * yy).sum()]) / P.sum()
        assert np.linalg.norm(c - [u, v]) < 0.15, (c, (u, v))


def test_depth_is_planar_z(tmp_path):
    rig = RigModel.build(Rig.model_validate(RIG))
    be = BlenderBackend(RenderSettings(samples=1), tmp_dir=str(tmp_path))
    be.new_frame(rig)
    bpy.ops.mesh.primitive_plane_add(size=20, location=(0, 0, 2.0))
    # plane z=2 in rig/world frame; camera tilted, so check depth against analytic ray hit
    cam = rig.cameras["c"]
    plan = cam.render_plan()
    r = be.render_camera("c", plan)
    Kr = plan.K_render
    R, t = cam.T_rig_cam[:3, :3], cam.T_rig_cam[:3, 3]
    for (u, v) in [(10, 10), (150, 40), (100, 300)]:
        d = R @ np.array([(u - Kr[0, 2]) / Kr[0, 0], (v - Kr[1, 2]) / Kr[1, 1], 1.0])
        s = (2.0 - t[2]) / d[2]
        assert abs(r.depth[v, u] - s) < 1e-3 * s
