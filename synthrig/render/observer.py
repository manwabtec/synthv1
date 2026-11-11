"""Overview render: the configured scene seen from a free 'observer' viewpoint, with the
rig's cameras and lasers drawn as markers. For inspection only (not a dataset output).

  python -m synthrig.render.observer configs/wheel_rail/run.yaml --frame 0 \
      --target 0 -0.75 0.1 --azimuth -60 --elevation 25 --distance 2.0 --out /data/gui/overview.png

The scene is assembled exactly as for the dataset (same seed/frame). Markers never cast
shadows, so they do not block the lasers. Track frame: x along track, y left, z up.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import bpy
import mathutils
import numpy as np

from ..opticsrig import RigModel, load_rig
from ..opticsrig.transforms import CV_TO_BLENDER_OBJ, look_at_R, make_T, transform_points
from ..paths import output_path
from ..pipeline import GPU_BACKENDS, RunConfig, load_scene
from . import devices
from .assembly import build_frame
from .blender_backend import BlenderBackend, RenderSettings


def _emissive(name: str, rgb, strength: float = 2.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (*rgb[:3], 1.0)
    em.inputs["Strength"].default_value = strength
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(em.outputs[0], out.inputs[0])
    return m


def _marker(name: str, V, F, mat, wire: float = 0.0):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(map(float, v)) for v in V], [], [tuple(f) for f in F])
    me.update()
    obj = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(obj)
    obj.data.materials.append(mat)
    obj.visible_shadow = False  # never blocks lasers or lights
    if wire > 0:
        mod = obj.modifiers.new("wire", "WIREFRAME")
        mod.thickness = wire
    return obj


def _box(T, size):
    h = np.asarray(size) / 2
    V = h * np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                      [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]])
    F = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    return transform_points(T, V), F


def add_device_markers(rig: RigModel, T_world_rig: np.ndarray, frustum_m: float) -> None:
    """Camera bodies + wireframe frustums, laser bodies + wireframe fans."""
    from ..layout import camera_frustum, laser_fan_outline

    cam_mat = _emissive("marker_camera", (0.1, 0.7, 1.0), 3.0)
    for name, cam in rig.cameras.items():
        T = T_world_rig @ cam.T_rig_cam
        V, F = _box(T @ make_T(np.eye(3), [0, 0, -0.04]), (0.05, 0.04, 0.08))
        _marker(f"marker_{name}_body", V, F, cam_mat)
        c, corners = camera_frustum(cam, frustum_m)
        pts = transform_points(T_world_rig, np.vstack([c, corners]))
        _marker(f"marker_{name}_frustum", pts, [(0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 1)], cam_mat, 0.002)
    for name, L in rig.lasers.items():
        T = T_world_rig @ L.T_rig_proj
        mat = _emissive(f"marker_{name}", L.color_rgb, 4.0)
        V, F = _box(T @ make_T(np.eye(3), [0, 0, -0.03]), (0.04, 0.04, 0.06))
        _marker(f"marker_{name}_body", V, F, mat)
        o = T[:3, 3]
        ends = transform_points(T_world_rig, laser_fan_outline(L, frustum_m))
        F = [(0, i + 1, (i + 1) % len(ends) + 1) for i in range(len(ends))] if len(ends) > 2 else [(0, 1, 2)]
        _marker(f"marker_{name}_fan", np.vstack([o, ends]), F, mat, 0.0015)


def observer_pose(target, azimuth_deg: float, elevation_deg: float, distance: float) -> np.ndarray:
    """T_world_observer (OpenCV-style) on a sphere around `target`.

    Azimuth 0 looks along +y (from the field side, y < 0, towards the track); +90 looks
    along -x. Elevation is above the horizontal.
    """
    az, el = np.radians(azimuth_deg), np.radians(elevation_deg)
    back = np.array([np.sin(az) * np.cos(el), -np.cos(az) * np.cos(el), np.sin(el)])
    pos = np.asarray(target, float) + distance * back
    return make_T(look_at_R(pos, target, (0, 0, 1)), pos)


def render_overview(config: Path, out: Path, frame: int = 0, target=(0.0, -0.75, 0.1),
                    azimuth: float = -35.0, elevation: float = 20.0, distance: float = 2.2,
                    hfov_deg: float = 50.0, width: int = 1280, height: int = 800, samples: int = 64,
                    device: str = "GPU", markers: bool = True, fill_light: float = 3.0) -> Path:
    cfg = RunConfig.load(config)
    if device in GPU_BACKENDS:
        devices.DEFAULT_BACKENDS = GPU_BACKENDS[device]
    rig = RigModel.build(load_rig(cfg.rig), 1.0)
    backend = BlenderBackend(RenderSettings(samples=samples, device="CPU" if device == "CPU" else "GPU"))
    scene = load_scene(cfg.scene, cfg.scene_params, cfg.scene_config)
    _params, ctx = build_frame(backend, rig, scene, cfg.seed, frame)
    if markers:
        add_device_markers(rig, ctx.T_world_rig, frustum_m=0.35)
    sc = bpy.context.scene
    if fill_light > 0:  # soft overhead light so the whole scene is readable
        ld = bpy.data.lights.new("observer_fill", "SUN")
        ld.energy = fill_light
        ld.angle = np.radians(30)
        obj = bpy.data.objects.new("observer_fill", ld)
        sc.collection.objects.link(obj)
        obj.rotation_euler = (np.radians(35), 0, np.radians(-30))
    cam = bpy.data.cameras.new("observer")
    cam.sensor_fit = "HORIZONTAL"
    cam.angle = np.radians(hfov_deg)
    cam.clip_start, cam.clip_end = 0.01, 100.0
    obj = bpy.data.objects.new("observer", cam)
    sc.collection.objects.link(obj)
    T = observer_pose(target, azimuth, elevation, distance) @ CV_TO_BLENDER_OBJ
    obj.matrix_world = mathutils.Matrix(T.tolist())
    sc.camera = obj
    sc.render.resolution_x, sc.render.resolution_y = width, height
    sc.render.image_settings.file_format = "PNG"
    sc.view_settings.view_transform = "Standard"
    sc.use_nodes = False
    out.parent.mkdir(parents=True, exist_ok=True)
    sc.render.filepath = str(out)
    t = time.time()
    bpy.ops.render.render(write_still=True)
    print(f"[observer] wrote {out} in {time.time() - t:.1f}s")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", type=Path)
    ap.add_argument("--out", required=True, help="PNG path (absolute or under the data root)")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--target", type=float, nargs=3, default=[0.0, -0.75, 0.1])
    ap.add_argument("--azimuth", type=float, default=-35.0)
    ap.add_argument("--elevation", type=float, default=20.0)
    ap.add_argument("--distance", type=float, default=2.2)
    ap.add_argument("--hfov", type=float, default=50.0)
    ap.add_argument("--size", type=int, nargs=2, default=[1280, 800])
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--device", choices=["CPU", "GPU", "OPTIX", "CUDA"], default="GPU")
    ap.add_argument("--no-markers", action="store_true")
    a = ap.parse_args()
    render_overview(a.config, output_path(a.out), a.frame, a.target, a.azimuth, a.elevation, a.distance,
                    a.hfov, a.size[0], a.size[1], a.samples, a.device, not a.no_markers)


if __name__ == "__main__":
    main()
