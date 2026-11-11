"""Blender/Cycles backend. The ONLY module (with scene_api/materials) that imports bpy.

Responsibilities:
  * turn a RigModel into Blender cameras, projector lights and rig lights
  * render one pinhole canvas per camera (RenderPlan) with passes:
      combined RGB, planar Z depth, object index, one light-group pass per laser
  * ray-cast every laser dot for analytic 3D ground truth and per-camera visibility
Coordinate conversion OpenCV-device -> Blender-object happens only in `_place`.
"""
from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass, field

import bpy
import mathutils
import numpy as np
import OpenEXR

from ..opticsrig import RigModel
from ..opticsrig.camera import RenderPlan
from ..opticsrig.transforms import CV_TO_BLENDER_OBJ, inv_T, make_T, transform_points


@dataclass
class RenderSettings:
    samples: int = 64
    filter_width: float = 1.5  # Cycles pixel filter (px); ~optics/sensor MTF
    max_bounces: int = 4
    device: str = "CPU"  # or "GPU" (configure Cycles compute device beforehand)
    threads: int = 0  # 0 = auto
    clip_start: float = 0.01
    clip_end: float = 1000.0
    use_denoising: bool = False  # denoisers smear laser dots: keep off for GT work


def lightgroup_name(kind: str, name: str = "") -> str:
    s = f"{kind}_{name}" if name else kind
    return re.sub(r"[^A-Za-z0-9_]", "_", s)


LG_AMBIENT = "ambient"
LG_RIG_LIGHTS = "rig_lights"
LG_SCENE_LIGHTS = "scene_lights"


def laser_group(laser_name: str) -> str:
    return lightgroup_name("laser", laser_name)


@dataclass
class PinholeRender:
    """Raw per-camera render on the pinhole canvas described by `plan`."""

    camera: str
    plan: RenderPlan
    combined: np.ndarray  # (H,W,3) linear
    depth: np.ndarray  # (H,W) planar z along optical axis, 0 = no hit
    index: np.ndarray  # (H,W) int32 object pass index, 0 = background
    groups: dict[str, np.ndarray] = field(default_factory=dict)  # lightgroup -> (H,W,3)


@dataclass
class DotsGT:
    """Analytic laser-dot ground truth for one frame."""

    laser: np.ndarray  # (N,) laser index into `laser_names`
    laser_names: list[str]
    dot_id: np.ndarray  # (N,) index within its laser's pattern
    hit: np.ndarray  # (N,) bool: ray hit geometry
    X_world: np.ndarray  # (N,3) nan if no hit
    X_rig: np.ndarray  # (N,3)
    normal_world: np.ndarray  # (N,3)
    object_index: np.ndarray  # (N,) pass index of the hit object (0 = none)
    uv: dict[str, np.ndarray]  # camera -> (N,2) distorted pixel coords (nan if behind)
    visible: dict[str, np.ndarray]  # camera -> (N,) bool: in image and not occluded


def _mat(T: np.ndarray) -> mathutils.Matrix:
    return mathutils.Matrix(np.asarray(T, float).tolist())


class _quiet_stdout:
    """Blender prints per-file 'Saved:' lines from C; silence fd 1 during renders."""

    def __enter__(self):
        import sys
        sys.stdout.flush()
        self._saved = os.dup(1)
        self._null = os.open(os.devnull, os.O_WRONLY)
        os.dup2(self._null, 1)

    def __exit__(self, *exc):
        os.dup2(self._saved, 1)
        os.close(self._null)
        os.close(self._saved)


def _read_exr(path: str) -> np.ndarray:
    ch = OpenEXR.File(path).channels()
    if "RGBA" in ch:
        return ch["RGBA"].pixels
    if "RGB" in ch:
        return ch["RGB"].pixels
    if "V" in ch:
        return ch["V"].pixels
    k = next(iter(ch))
    return ch[k].pixels


class BlenderBackend:
    def __init__(self, settings: RenderSettings | None = None, tmp_dir: str | None = None):
        self.s = settings or RenderSettings()
        if self.s.device == "GPU":
            from .devices import configure_gpu

            configure_gpu()
        self.tmp_dir = tmp_dir or tempfile.mkdtemp(prefix="synthrig_")
        self._tex_cache: dict[str, tuple[np.ndarray, tuple[float, float]]] = {}
        self.rig: RigModel | None = None
        self.T_world_rig = np.eye(4)

    # ------------------------------------------------------------------ scene setup
    def new_frame(self, rig: RigModel, cycles_seed: int = 0):
        """Fresh Blender scene with the rig built in. Scenes then add their content."""
        bpy.ops.wm.read_factory_settings(use_empty=True)
        if self.s.device == "GPU":
            # read_factory_settings also resets user preferences, including the Cycles
            # compute device chosen in __init__; without this every frame renders on CPU.
            from .devices import configure_gpu

            configure_gpu()
        sc = bpy.context.scene
        sc.render.engine = "CYCLES"
        sc.cycles.device = self.s.device
        sc.cycles.samples = self.s.samples
        sc.cycles.use_denoising = self.s.use_denoising
        sc.cycles.use_adaptive_sampling = False
        sc.cycles.filter_width = self.s.filter_width
        sc.cycles.max_bounces = self.s.max_bounces
        sc.cycles.seed = int(cycles_seed) % (2 ** 31)
        if self.s.threads > 0:
            sc.render.threads_mode = "FIXED"
            sc.render.threads = self.s.threads
        sc.render.resolution_percentage = 100
        sc.render.film_transparent = False
        sc.view_settings.view_transform = "Standard"
        sc.frame_current = 1

        world = bpy.data.worlds.new("world")
        world.use_nodes = True
        sc.world = world
        self.set_ambient((1, 1, 1), 0.02)

        vl = sc.view_layers[0]
        vl.use_pass_z = True
        vl.use_pass_object_index = True
        groups = [LG_AMBIENT, LG_RIG_LIGHTS, LG_SCENE_LIGHTS] + [laser_group(n) for n in rig.lasers]
        for g in groups:
            vl.lightgroups.add(name=g)
        world.lightgroup = LG_AMBIENT
        self.groups = groups

        self.rig = rig
        self.rig_empty = bpy.data.objects.new("rig", None)
        sc.collection.objects.link(self.rig_empty)
        self._build_cameras(rig)
        self._build_lasers(rig)
        self._build_lights(rig)
        self.set_rig_pose(np.eye(4))

    def set_ambient(self, color, strength):
        bg = bpy.context.scene.world.node_tree.nodes["Background"]
        bg.inputs[0].default_value = (*color[:3], 1.0)
        bg.inputs[1].default_value = float(strength)

    def set_rig_pose(self, T_world_rig: np.ndarray):
        self.T_world_rig = np.asarray(T_world_rig, float)
        self.rig_empty.matrix_world = _mat(self.T_world_rig)
        bpy.context.view_layer.update()

    def _place(self, obj, T_rig_dev: np.ndarray):
        """Parent a Blender object to the rig with an OpenCV-style device pose."""
        obj.parent = self.rig_empty
        obj.matrix_parent_inverse = mathutils.Matrix.Identity(4)
        obj.matrix_basis = _mat(T_rig_dev @ CV_TO_BLENDER_OBJ)

    def _build_cameras(self, rig: RigModel):
        self.cam_objs = {}
        for name, cam in rig.cameras.items():
            data = bpy.data.cameras.new(name)
            data.clip_start, data.clip_end = self.s.clip_start, self.s.clip_end
            obj = bpy.data.objects.new(name, data)
            bpy.context.scene.collection.objects.link(obj)
            self._place(obj, cam.T_rig_cam)
            self.cam_objs[name] = obj

    def _projector_image(self, name, tex):
        img, _ = tex
        H, W = img.shape
        bimg = bpy.data.images.new(f"pattern_{name}", W, H, float_buffer=True, alpha=False)
        # NB: set colour space BEFORE pixels - changing it on a generated image regenerates
        # it (blank). Float buffers are scene-linear, so the pattern values are used as-is.
        bimg.colorspace_settings.name = "Non-Color"
        rgba = np.ones((H, W, 4), np.float32)
        rgba[..., :3] = img[..., None]
        bimg.pixels.foreach_set(np.flipud(rgba).ravel())  # Blender row 0 = bottom
        bimg.update()
        bimg.pack()  # keep pixel data attached to the render session
        return bimg

    def _build_lasers(self, rig: RigModel):
        for name, L in rig.lasers.items():
            if name not in self._tex_cache:
                self._tex_cache[name] = L.texture()
            tex = self._tex_cache[name]
            hx, hy = tex[1]
            ld = bpy.data.lights.new(f"laser_{name}", "SPOT")
            ld.energy = L.power_w
            ld.color = tuple(L.color_rgb)
            ld.shadow_soft_size = 0.0  # point source => sharp dots, exact geometry
            ld.spot_size = min(2 * np.arctan(np.hypot(hx, hy)) * 1.02, np.pi)
            ld.spot_blend = 0.0
            if hasattr(ld, "use_soft_falloff"):
                ld.use_soft_falloff = False
            ld.use_nodes = True
            nt = ld.node_tree
            nt.nodes.clear()
            # Light-space direction -> OpenCV tangents -> texture UV.
            # Blender light looks down -Z with +Y up: tx = X/(-Z); ty_cv = -Y/(-Z) = Y/Z.
            tc = nt.nodes.new("ShaderNodeTexCoord")
            sep = nt.nodes.new("ShaderNodeSeparateXYZ")
            nt.links.new(tc.outputs["Normal"], sep.inputs[0])

            def m(op, a, b):
                n = nt.nodes.new("ShaderNodeMath")
                n.operation = op
                for i, x in enumerate((a, b)):
                    if isinstance(x, (int, float)):
                        n.inputs[i].default_value = x
                    else:
                        nt.links.new(x, n.inputs[i])
                return n.outputs[0]

            negz = m("MULTIPLY", sep.outputs["Z"], -1.0)
            tx = m("DIVIDE", sep.outputs["X"], negz)
            ty_b = m("DIVIDE", sep.outputs["Y"], negz)  # = -ty_cv
            u = m("ADD", m("MULTIPLY", tx, 1.0 / (2 * hx)), 0.5)
            v = m("ADD", m("MULTIPLY", ty_b, 1.0 / (2 * hy)), 0.5)  # v_tex = 0.5 - ty_cv/(2hy)
            comb = nt.nodes.new("ShaderNodeCombineXYZ")
            nt.links.new(u, comb.inputs[0])
            nt.links.new(v, comb.inputs[1])
            it = nt.nodes.new("ShaderNodeTexImage")
            it.image = self._projector_image(name, tex)
            it.extension = "CLIP"
            it.interpolation = "Linear"
            nt.links.new(comb.outputs[0], it.inputs[0])
            em = nt.nodes.new("ShaderNodeEmission")
            nt.links.new(it.outputs["Color"], em.inputs["Color"])
            out = nt.nodes.new("ShaderNodeOutputLight")
            nt.links.new(em.outputs[0], out.inputs[0])
            obj = bpy.data.objects.new(f"laser_{name}", ld)
            bpy.context.scene.collection.objects.link(obj)
            obj.lightgroup = laser_group(name)
            self._place(obj, L.T_rig_proj)

    def _build_lights(self, rig: RigModel):
        from ..opticsrig.transforms import pose_to_T

        for l in rig.spec.lights:
            kind = {"area": "AREA", "point": "POINT", "spot": "SPOT"}[l.type]
            ld = bpy.data.lights.new(f"light_{l.name}", kind)
            ld.energy = l.power_w
            ld.color = tuple(l.color_rgb)
            if kind == "AREA":
                ld.size = l.size_m
            else:
                ld.shadow_soft_size = l.size_m
            if kind == "SPOT":
                ld.spot_size = np.deg2rad(l.spot_angle_deg)
            obj = bpy.data.objects.new(f"light_{l.name}", ld)
            bpy.context.scene.collection.objects.link(obj)
            obj.lightgroup = LG_RIG_LIGHTS
            self._place(obj, pose_to_T(l.pose))

    # ------------------------------------------------------------------ rendering
    def _setup_compositor(self, out_dir: str):
        sc = bpy.context.scene
        sc.use_nodes = True
        tree = sc.node_tree
        tree.nodes.clear()
        rl = tree.nodes.new("CompositorNodeRLayers")
        fo = tree.nodes.new("CompositorNodeOutputFile")
        fo.base_path = out_dir
        fo.format.file_format = "OPEN_EXR"
        fo.format.color_depth = "32"
        fo.format.exr_codec = "ZIP"
        fo.file_slots.clear()
        names = {"Image": "combined", "Depth": "depth", "IndexOB": "index"}
        for g in self.groups:
            names[f"Combined_{g}"] = f"lg_{g}"
        for sock, fname in names.items():
            fo.file_slots.new(fname)
            tree.links.new(rl.outputs[sock], fo.inputs[fname])
        return names

    def render_camera(self, name: str, plan: RenderPlan) -> PinholeRender:
        sc = bpy.context.scene
        obj = self.cam_objs[name]
        data = obj.data
        W, H = plan.size_render
        K = plan.K_render
        px_mm = self.rig.cameras[name].pixel_size_um / 1000.0
        # OpenCV K -> Blender camera. Blender shift is in units of the larger image side;
        # OpenCV pixel centres are integers => continuous principal point is c + 0.5.
        big = max(W, H)
        data.sensor_fit = "HORIZONTAL" if W >= H else "VERTICAL"
        data.sensor_width = W * px_mm
        data.sensor_height = H * px_mm
        data.lens = K[0, 0] * px_mm
        data.shift_x = (W / 2 - (K[0, 2] + 0.5)) / big
        data.shift_y = ((K[1, 2] + 0.5) - H / 2) / big
        sc.render.resolution_x, sc.render.resolution_y = W, H
        sc.render.pixel_aspect_x = sc.render.pixel_aspect_y = 1.0
        sc.camera = obj

        out_dir = os.path.join(self.tmp_dir, name)
        os.makedirs(out_dir, exist_ok=True)
        for f in os.listdir(out_dir):
            os.remove(os.path.join(out_dir, f))
        names = self._setup_compositor(out_dir)
        with _quiet_stdout():
            bpy.ops.render.render(write_still=False)

        def rd(fname):
            return _read_exr(os.path.join(out_dir, f"{fname}{sc.frame_current:04d}.exr"))

        combined = rd("combined")[..., :3].astype(np.float32)
        depth = rd("depth").astype(np.float32)
        if depth.ndim == 3:
            depth = depth[..., 0]
        depth[~np.isfinite(depth) | (depth >= self.s.clip_end * 0.999)] = 0.0
        index = rd("index")
        if index.ndim == 3:
            index = index[..., 0]
        groups = {g: rd(f"lg_{g}")[..., :3].astype(np.float32) for g in self.groups}
        return PinholeRender(name, plan, combined, depth, np.rint(index).astype(np.int32), groups)

    # ------------------------------------------------------------------ dot ground truth
    def cast_dots(self) -> DotsGT:
        sc = bpy.context.scene
        dg = bpy.context.evaluated_depsgraph_get()
        rig = self.rig
        names = list(rig.lasers)
        lid, did, X, Nrm, hit, oidx = [], [], [], [], [], []
        for li, (lname, L) in enumerate(rig.lasers.items()):
            T_w_p = self.T_world_rig @ L.T_rig_proj
            o = T_w_p[:3, 3]
            dirs = L.directions @ T_w_p[:3, :3].T
            for di, d in enumerate(dirs):
                ok, loc, nrm, _, obj, _ = sc.ray_cast(dg, mathutils.Vector(o), mathutils.Vector(d),
                                                      distance=self.s.clip_end)
                lid.append(li)
                did.append(di)
                hit.append(ok)
                X.append(tuple(loc) if ok else (np.nan,) * 3)
                Nrm.append(tuple(nrm) if ok else (np.nan,) * 3)
                oidx.append(obj.pass_index if ok else 0)
        X = np.asarray(X, float).reshape(-1, 3)
        Nrm = np.asarray(Nrm, float).reshape(-1, 3)
        hit = np.asarray(hit, bool)
        lid = np.asarray(lid, np.int32)
        # side of the surface the laser lit (thin sheets: camera must be on the same side)
        proj_o = np.stack([(self.T_world_rig @ rig.lasers[names[i]].T_rig_proj)[:3, 3] for i in lid]) \
            if len(lid) else np.zeros((0, 3))
        lit_side = np.sign(np.einsum("ij,ij->i", np.nan_to_num(Nrm), proj_o - np.nan_to_num(X)))
        uv, vis = {}, {}
        for cname, cam in rig.cameras.items():
            T_w_c = self.T_world_rig @ cam.T_rig_cam
            Xc = transform_points(inv_T(T_w_c), X)
            front = hit & (Xc[:, 2] > 1e-6)
            u = np.full((len(X), 2), np.nan)
            if front.any():
                u[front] = cam.project(Xc[front])
            w, h = cam.size
            inside = front & (u[:, 0] >= -0.5) & (u[:, 0] < w - 0.5) & (u[:, 1] >= -0.5) & (u[:, 1] < h - 0.5)
            v = np.zeros(len(X), bool)
            c = T_w_c[:3, 3]
            for i in np.nonzero(inside)[0]:
                d = X[i] - c
                if np.sign(np.dot(Nrm[i], c - X[i])) != lit_side[i]:
                    continue  # camera sees the other side of the lit surface
                dist = np.linalg.norm(d)
                blocked, *_ = sc.ray_cast(dg, mathutils.Vector(c), mathutils.Vector(d / dist),
                                          distance=dist * (1 - 1e-4) - 1e-5)
                v[i] = not blocked
            uv[cname], vis[cname] = u, v
        return DotsGT(lid, names, np.asarray(did, np.int32), hit, X,
                      transform_points(inv_T(self.T_world_rig), X), Nrm,
                      np.asarray(oidx, np.int32), uv, vis)
