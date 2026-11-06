"""Shared deterministic scene assembly for dataset rendering and Blender viewing."""
from __future__ import annotations

import numpy as np

from .scene_api import Scene, SceneContext
from .blender_backend import BlenderBackend
from ..opticsrig import RigModel


def build_frame(backend: BlenderBackend, rig: RigModel, scene: Scene, seed: int,
                index: int) -> tuple[dict, SceneContext]:
    """Build frame index from its seed; world coordinates are Z-up metres."""
    rng = np.random.default_rng([seed, index])
    params = scene.params(rng)
    backend.new_frame(rig, cycles_seed=seed * 100003 + index)
    context = SceneContext(backend, rig, rng)
    scene.build(context, params)
    backend.set_rig_pose(context.T_world_rig)
    return params, context