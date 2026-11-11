"""Blender builders for the wheel-on-rail scene (bpy). Geometry comes from wheelrail.py,
so the layout view and the render always agree. Builders take params, never an rng."""
from __future__ import annotations

import bpy
import mathutils
import numpy as np

from .materials import principled
from ...render.scene_api import SceneContext
from ..wheelrail import (WheelRailParams, ballast_level, rail_sections, revolve, tie_plates, ties,
                        wheel_sections, wheel_to_track)


def resample(poly: np.ndarray, max_step: float) -> np.ndarray:
    """Polyline resampled by arc length so no segment exceeds max_step (m); keeps ends."""
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(seg)]
    n = max(int(np.ceil(s[-1] / max_step)) + 1, len(poly) if s[-1] < max_step else 2)
    t = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(t, s, poly[:, 0]), np.interp(t, s, poly[:, 1])], 1)


def mesh_object(name: str, V: np.ndarray, F, smooth: bool = True) -> bpy.types.Object:
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(v) for v in V], [], [tuple(int(i) for i in f) for f in F])
    me.validate()
    me.update()
    if smooth:
        me.shade_smooth()
    return bpy.data.objects.new(name, me)


def box_object(name: str, center, size) -> bpy.types.Object:
    c, h = np.asarray(center, float), np.asarray(size, float) / 2
    V = c + h * np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                          [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]])
    F = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    return mesh_object(name, V, F, smooth=False)


def extrude_x(name: str, poly_yz: np.ndarray, x0: float, x1: float) -> bpy.types.Object:
    """Prism: closed (y, z) polygon swept along x from x0 to x1, with end caps."""
    n = len(poly_yz)
    V = np.concatenate([np.c_[np.full(n, x0), poly_yz], np.c_[np.full(n, x1), poly_yz]])
    F = [(i, (i + 1) % n, n + (i + 1) % n, n + i) for i in range(n)]
    F += [tuple(range(n))[::-1], tuple(range(n, 2 * n))]
    return mesh_object(name, V, F, smooth=False)


def steel(p: WheelRailParams, name: str, seed_offset: float = 0.0):
    rust_rgb = np.array([0.45, 0.22, 0.10])
    col = (1 - p.rust) * np.asarray(p.steel_rgb) + p.rust * rust_rgb
    return principled(name, col.tolist(), p.roughness, 0.6 * (1 - p.rust), noise_scale=35,
                      noise_contrast=0.35, noise_seed=p.noise_seed + seed_offset)


def build_wheel(ctx: SceneContext, p: WheelRailParams, segments: int = 360) -> bpy.types.Object:
    """One wheel on the right rail, split into part objects per segmentation class."""
    root = bpy.data.objects.new("wheel", None)
    bpy.context.scene.collection.objects.link(root)
    root.matrix_world = mathutils.Matrix(wheel_to_track(p).tolist())
    mat = steel(p, "wheel_steel")
    for cls, poly in wheel_sections(p).items():
        V, F = revolve(resample(poly, 0.0005), segments)
        obj = mesh_object(cls, V, F, smooth=cls in ("wheel_tread", "wheel_flange", "wheel_rim_face"))
        obj.data.materials.append(mat)
        ctx.add(obj, cls)
        obj.parent = root
    return root


def build_track(ctx: SceneContext, p: WheelRailParams) -> None:
    """Both rails, ties, tie plates and a ballast bed around the wheel position."""
    x0, x1 = p.wheel_x_m - p.track_half_length_m, p.wheel_x_m + p.track_half_length_m
    rail_mat = principled("rail_steel", (0.32, 0.30, 0.28), 0.45, 0.5, noise_scale=40,
                          noise_contrast=0.35, noise_seed=p.noise_seed + 1)
    for side, part, poly in rail_sections(p):
        obj = extrude_x(f"rail_{side}_{part}", poly, x0, x1)
        obj.data.materials.append(rail_mat)
        ctx.add(obj, f"rail_{part}")
    wood = principled("tie_wood", (0.22, 0.17, 0.12), 0.85, 0.0, noise_scale=12, noise_contrast=0.5,
                      noise_seed=p.noise_seed + 2)
    for i, (c, s) in enumerate(ties(p)):
        obj = box_object(f"tie_{i}", c, s)
        obj.data.materials.append(wood)
        ctx.add(obj, "tie")
    plate = principled("tie_plate", (0.25, 0.24, 0.23), 0.6, 0.4, noise_scale=30, noise_contrast=0.3,
                       noise_seed=p.noise_seed + 3)
    for i, (c, s) in enumerate(tie_plates(p)):
        obj = box_object(f"tie_plate_{i}", c, s)
        obj.data.materials.append(plate)
        ctx.add(obj, "tie_plate")
    z = ballast_level(p)
    ballast = mesh_object("ballast", np.array([[x0 - 1.5, -2.5, z], [x1 + 1.5, -2.5, z],
                                               [x1 + 1.5, 2.5, z], [x0 - 1.5, 2.5, z]]),
                          [(0, 1, 2, 3)], smooth=False)
    ballast.data.materials.append(principled("ballast", (0.38, 0.36, 0.33), 0.9, 0.0, noise_scale=90,
                                             noise_contrast=0.7, noise_seed=p.noise_seed + 4))
    ctx.add(ballast, "ballast")
