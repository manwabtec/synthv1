"""Application 2: a railway wheel on a rail, viewed from the field side.

Geometry is a simplified revolved wheel profile (hub, plate, rim, tapered tread, flange);
swap in a measured profile or a CAD mesh by replacing WHEEL_PROFILE / build_wheel.
"""
from __future__ import annotations

import bmesh
import bpy
import numpy as np

from synthrig.render.materials import principled
from synthrig.render.scene_api import Scene, SceneContext

# (axial a [m], radius r [m]); a = 0 at the flange (back/gauge) face, + towards field side.
WHEEL_PROFILE = [
    (-0.04, 0.09), (0.14, 0.09), (0.14, 0.13), (0.06, 0.14), (0.05, 0.38), (0.14, 0.40),
    (0.14, 0.450), (0.10, 0.452), (0.05, 0.455), (0.035, 0.457),
    (0.028, 0.465), (0.024, 0.475), (0.018, 0.483), (0.010, 0.485), (0.003, 0.480), (0.0, 0.470),
    (0.0, 0.40), (0.025, 0.38), (0.02, 0.14), (-0.04, 0.13),
]
RAIL_H = 0.17
TREAD_R = 0.455


def revolve(name, profile, segments=256):
    """Mesh from revolving a closed (a, r) profile about the local X axis."""
    bm = bmesh.new()
    th = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    rings = [[bm.verts.new((a, r * np.cos(t), r * np.sin(t))) for t in th] for a, r in profile]
    n = len(profile)
    for i in range(n):
        A, B = rings[i], rings[(i + 1) % n]
        for k in range(segments):
            k2 = (k + 1) % segments
            bm.faces.new((A[k], A[k2], B[k2], B[k]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    for poly in me.polygons:
        poly.use_smooth = True
    return bpy.data.objects.new(name, me)


def box(name, size, loc):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=loc)
    ob = bpy.context.active_object
    ob.scale = size
    ob.name = name
    return ob


class WheelScene(Scene):
    name = "app2_wheel"

    def sample_params(self, rng):
        return {
            "wheel_rot_deg": float(rng.uniform(0, 360)),
            "lateral_offset": float(rng.normal(0, 0.01)),
            "steel_color": (np.array([0.35, 0.33, 0.30]) * rng.uniform(0.7, 1.3)).tolist(),
            "rust": float(rng.uniform(0.0, 0.6)),
            "roughness": float(rng.uniform(0.3, 0.7)),
            "noise_seed": float(rng.uniform(0, 100)),
            "ambient": float(rng.uniform(0.02, 0.2)),
            "rig_pos": (np.array([1.9, -0.45, 0.75]) + rng.normal(0, 0.03, 3)).tolist(),
        }

    def build(self, ctx: SceneContext, p):
        ctx.set_ambient((0.9, 0.95, 1.0), p["ambient"])
        steel = np.asarray(p["steel_color"])
        rust = np.array([0.45, 0.22, 0.10])
        col = (1 - p["rust"]) * steel + p["rust"] * rust

        wheel = revolve("wheel", WHEEL_PROFILE)
        cz = RAIL_H + TREAD_R
        wheel.location = (p["lateral_offset"], 0.0, cz)
        wheel.rotation_euler = (np.deg2rad(p["wheel_rot_deg"]), 0, 0)
        wheel.data.materials.append(principled("wheel", col, p["roughness"], 0.5 * (1 - p["rust"]),
                                               noise_scale=30, noise_contrast=0.5, noise_seed=p["noise_seed"]))
        ctx.add(wheel, "wheel")

        rail_mat = principled("rail", (0.3, 0.29, 0.28), 0.5, 0.4, noise_scale=40, noise_contrast=0.4,
                              noise_seed=p["noise_seed"] + 1)
        x0 = 0.07 + p["lateral_offset"]  # rail head under the tread contact point
        parts = [("rail_foot", (0.15, 4.0, 0.015), (x0, 0, 0.0075)),
                 ("rail_web", (0.017, 4.0, 0.11), (x0, 0, 0.015 + 0.055)),
                 ("rail_head", (0.07, 4.0, 0.045), (x0, 0, RAIL_H - 0.0225))]
        for name, size, loc in parts:
            ob = box(name, size, loc)
            ob.data.materials.append(rail_mat)
            ctx.add(ob, "rail")

        bpy.ops.mesh.primitive_plane_add(size=8.0, location=(0, 0, 0))
        ground = bpy.context.active_object
        ground.data.materials.append(principled("ballast", (0.35, 0.33, 0.3), 0.9, 0.0, noise_scale=60,
                                                noise_contrast=0.8, noise_seed=p["noise_seed"] + 2))
        ctx.add(ground, "ground")
        ctx.rig_look_at(p["rig_pos"], (0.07, 0.0, cz - 0.05))
