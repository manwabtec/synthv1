"""The contract between the pipeline and an application ("scene").

A scene is Blender-bound (it builds geometry with bpy) but knows nothing about cameras,
lasers, sensors or the dataset format. It receives a SceneContext and:
  * adds objects via ctx.add(obj, cls)   -> segmentation index + class map
  * optionally sets the ambient light and where the rig sits: ctx.set_rig_pose(T_world_rig)
The rig is fixed; scenes only vary the world in front of it.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import bpy
import numpy as np

from ..rig import RigModel
from ..rig.transforms import rig_pose_look_at
from .blender_backend import LG_SCENE_LIGHTS, BlenderBackend


class SceneContext:
    def __init__(self, backend: BlenderBackend, rig: RigModel, rng: np.random.Generator):
        self.backend = backend
        self.rig = rig
        self.rng = rng
        self.objects: dict[int, dict[str, str]] = {}  # pass_index -> {name, class}
        self.T_world_rig = np.eye(4)
        self._next = 1

    def add(self, obj: bpy.types.Object, cls: str) -> bpy.types.Object:
        """Register an object: links it (if needed) and gives it a unique segmentation index."""
        if obj.name not in bpy.context.scene.collection.all_objects:
            bpy.context.scene.collection.objects.link(obj)
        obj.pass_index = self._next
        self.objects[self._next] = {"name": obj.name, "class": cls}
        self._next += 1
        return obj

    def add_light(self, obj: bpy.types.Object) -> bpy.types.Object:
        if obj.name not in bpy.context.scene.collection.all_objects:
            bpy.context.scene.collection.objects.link(obj)
        obj.lightgroup = LG_SCENE_LIGHTS
        return obj

    def set_ambient(self, color=(1, 1, 1), strength=0.05):
        self.backend.set_ambient(color, strength)

    def set_rig_pose(self, T_world_rig: np.ndarray):
        self.T_world_rig = np.asarray(T_world_rig, float)

    def rig_look_at(self, position, target, world_up=(0, 0, 1)):
        self.set_rig_pose(rig_pose_look_at(position, target, world_up))


class Scene(ABC):
    """Subclass per application. Keep all randomness in sample_params (logged per frame)."""

    name: str = "scene"

    def __init__(self, **overrides: Any):
        self.overrides = overrides

    @abstractmethod
    def sample_params(self, rng: np.random.Generator) -> dict:
        """Draw this frame's scene parameters. Must be JSON-serialisable."""

    @abstractmethod
    def build(self, ctx: SceneContext, params: dict) -> None:
        """Create the world for this frame from params (no hidden randomness)."""

    def params(self, rng: np.random.Generator) -> dict:
        p = self.sample_params(rng)
        p.update(self.overrides)
        return p
