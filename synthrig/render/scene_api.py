"""The contract between the pipeline and an application ("scene").

A scene is Blender-bound (it builds geometry with bpy) but knows nothing about cameras,
lasers, sensors or the dataset format. It receives a SceneContext and:
  * adds objects via ctx.add(obj, cls)   -> segmentation index + class map
  * optionally sets the ambient light and where the rig sits: ctx.set_rig_pose(T_world_rig)
    (not for `frame: track` rigs, whose devices are already placed in the world)
The rig is fixed; scenes only vary the world in front of it.

Scenes register under a short name with @register_scene("name") in scenes/<name>/;
run configs then say `scene: name`. The `module:Class` form still works.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import bpy
import numpy as np

from ..opticsrig import RigModel
from ..opticsrig.transforms import rig_pose_look_at
from .blender_backend import LG_SCENE_LIGHTS, BlenderBackend


class SceneContext:
    def __init__(self, backend: BlenderBackend, rig: RigModel, rng: np.random.Generator):
        self.backend = backend
        self.rig = rig
        self.rng = rng
        self.objects: dict[int, dict[str, str]] = {}  # pass_index -> {name, class}
        self.frame_info: dict = {}  # sequence_id / frame_in_sequence / t_s in sequence mode
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
        if self.rig.spec.frame == "track":
            raise RuntimeError("this rig places its devices in the track frame (frame: track); "
                               "scenes must not move it")
        self.T_world_rig = np.asarray(T_world_rig, float)

    def rig_look_at(self, position, target, world_up=(0, 0, 1)):
        self.set_rig_pose(rig_pose_look_at(position, target, world_up))


SCENES: dict[str, type["Scene"]] = {}


def register_scene(name: str):
    """Class decorator: make a scene loadable as `scene: <name>` in run configs."""
    def deco(cls):
        if name in SCENES and SCENES[name] is not cls:
            raise ValueError(f"scene name {name!r} registered twice")
        cls.name = name
        SCENES[name] = cls
        return cls
    return deco


class Scene(ABC):
    """Subclass per application. Keep all randomness in sample_params (logged per frame).

    `config` is the scene config file's content (fixed values and sampling specs, see
    synthrig.sampling); `overrides` (run config `scene_params`) replace sampled values.
    """

    name: str = "scene"

    def __init__(self, config: dict | None = None, **overrides: Any):
        self.config = dict(config or {})
        self.overrides = overrides

    def sample_params(self, rng: np.random.Generator) -> dict:
        """Draw this frame's scene parameters. Must be JSON-serialisable.

        Default: sample the scene config file. Scenes may override this.
        """
        from ..sampling import sample_config

        return sample_config(self.config, rng)

    @abstractmethod
    def build(self, ctx: SceneContext, params: dict) -> None:
        """Create the world for this frame from params (no hidden randomness)."""

    def sample_sequence(self, rng: np.random.Generator, n_frames: int, fps: float) -> list[dict]:
        """Parameters for the frames of one sequence. Default: independent draws.

        Scenes with motion override this (e.g. one wheel rolling: x and rotation follow t).
        """
        return [self.sample_params(rng) for _ in range(n_frames)]

    def sequence_params(self, rng: np.random.Generator, n_frames: int, fps: float) -> list[dict]:
        out = self.sample_sequence(rng, n_frames, fps)
        for p in out:
            p.update(self.overrides)
        return out

    def params(self, rng: np.random.Generator) -> dict:
        p = self.sample_params(rng)
        p.update(self.overrides)
        return p
