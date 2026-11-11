"""debug_clutter: random clutter on a textured floor (generic stereo / dot-triangulation sanity scene)."""
from __future__ import annotations

import bpy
import numpy as np

from synthrig.assets.blender.materials import principled
from synthrig.render.scene_api import Scene, SceneContext, register_scene

SHAPES = ["cube", "sphere", "cylinder", "cone", "torus"]


@register_scene("debug_clutter")
class ClutterScene(Scene):

    def sample_params(self, rng):
        n = int(rng.integers(5, 13))
        objs = []
        for _ in range(n):
            s = float(rng.uniform(0.05, 0.22))
            objs.append({
                "shape": str(rng.choice(SHAPES)),
                "size": s,
                "xy": rng.uniform(-0.45, 0.45, 2).tolist(),
                "rot_deg": rng.uniform(0, 360, 3).tolist(),
                "color": rng.uniform(0.1, 0.9, 3).tolist(),
                "roughness": float(rng.uniform(0.2, 0.9)),
                "metallic": float(rng.choice([0.0, 0.0, 0.0, 0.6])),
                "noise_scale": float(rng.uniform(5, 40)),
                "noise_seed": float(rng.uniform(0, 100)),
            })
        return {
            "objects": objs,
            "floor_color": rng.uniform(0.3, 0.7, 3).tolist(),
            "floor_noise_seed": float(rng.uniform(0, 100)),
            "ambient": float(rng.uniform(0.02, 0.15)),
            "rig_height": float(rng.uniform(0.9, 1.3)),
            "rig_back": float(rng.uniform(0.9, 1.3)),
            "rig_jitter": rng.normal(0, 0.03, 3).tolist(),
        }

    def build(self, ctx: SceneContext, p):
        ctx.set_ambient((1.0, 1.0, 1.0), p["ambient"])
        bpy.ops.mesh.primitive_plane_add(size=6.0, location=(0, 0, 0))
        floor = bpy.context.active_object
        floor.data.materials.append(principled("floor", p["floor_color"], 0.8, 0.0, noise_scale=25,
                                               noise_contrast=0.6, noise_seed=p["floor_noise_seed"]))
        ctx.add(floor, "floor")
        for i, o in enumerate(p["objects"]):
            s, (x, y) = o["size"], o["xy"]
            loc = (x, y, s / 2)
            if o["shape"] == "cube":
                bpy.ops.mesh.primitive_cube_add(size=s, location=loc)
            elif o["shape"] == "sphere":
                bpy.ops.mesh.primitive_uv_sphere_add(radius=s / 2, location=loc, segments=48, ring_count=24)
            elif o["shape"] == "cylinder":
                bpy.ops.mesh.primitive_cylinder_add(radius=s / 2, depth=s, location=loc, vertices=48)
            elif o["shape"] == "cone":
                bpy.ops.mesh.primitive_cone_add(radius1=s / 2, depth=s, location=loc, vertices=48)
            else:
                bpy.ops.mesh.primitive_torus_add(major_radius=s / 2, minor_radius=s / 6,
                                                 location=(x, y, s / 6), major_segments=48, minor_segments=16)
            ob = bpy.context.active_object
            if o["shape"] in ("sphere", "cylinder", "cone", "torus"):
                bpy.ops.object.shade_smooth()
            if o["shape"] in ("cube", "cylinder", "cone"):
                ob.rotation_euler = (0, 0, np.deg2rad(o["rot_deg"][2]))
            ob.data.materials.append(principled(f"m{i}", o["color"], o["roughness"], o["metallic"],
                                                o["noise_scale"], 0.5, o["noise_seed"]))
            ctx.add(ob, o["shape"])
        j = np.asarray(p["rig_jitter"])
        ctx.rig_look_at(np.array([0.0, -p["rig_back"], p["rig_height"]]) + j, (0.0, 0.0, 0.05))
