"""Shared deterministic scene assembly for dataset rendering and Blender viewing."""
from __future__ import annotations

import numpy as np

from ..sampling import frame_params
from .scene_api import Scene, SceneContext
from .blender_backend import BlenderBackend
from ..opticsrig import RigModel


def build_frame(backend: BlenderBackend, rig: RigModel, scene: Scene, seed: int,
                index: int, sequence=None) -> tuple[dict, SceneContext]:
    """Build frame `index` from (seed, index); world coordinates are Z-up metres.

    `sequence` (RunConfig.sequence_cfg) switches to sequence mode; ctx.frame_info then
    holds sequence_id / frame_in_sequence / t_s for meta.json.
    """
    params, info = frame_params(scene, seed, index, sequence)
    backend.new_frame(rig, cycles_seed=seed * 100003 + index)
    context = SceneContext(backend, rig, np.random.default_rng([seed, index]))
    context.frame_info = info
    scene.build(context, params)
    backend.set_rig_pose(context.T_world_rig)
    return params, context