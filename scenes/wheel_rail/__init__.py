"""wheel_rail: one wheel standing on the right rail of a short track section.

Parameters: synthrig.assets.wheelrail.WheelRailParams (wheel position along the track,
rotation, lateral shift, profiles, track and appearance). Values and sampling ranges come
from the scene config (configs/scenes/wheel_rail.yaml). Use a `frame: track` rig: the
devices are placed in the track frame, so this scene never moves the rig.
"""
from __future__ import annotations

from synthrig.assets.blender_wheelrail import build_track, build_wheel
from synthrig.assets.wheelrail import WheelRailParams
from synthrig.render.scene_api import Scene, SceneContext, register_scene


@register_scene("wheel_rail")
class WheelRailScene(Scene):
    def sample_params(self, rng) -> dict:
        """Sample the config, then fill defaults so meta.json logs every value used."""
        return WheelRailParams(**super().sample_params(rng)).model_dump(mode="json")

    def build(self, ctx: SceneContext, params: dict) -> None:
        p = WheelRailParams(**params)
        ctx.set_ambient((0.9, 0.95, 1.0), p.ambient)
        build_track(ctx, p)
        build_wheel(ctx, p)
