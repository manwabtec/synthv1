"""wheel_tread: close-up of one wheel's tread and flange; the wheel rotates so each
frame shows a different tread patch, or (sequence mode) rolls through the view.

Same geometry and parameters as wheel_rail (synthrig.assets.wheelrail.WheelRailParams);
its own camera/laser layout lives in configs/wheel_tread/rig.yaml (`frame: track`).
"""
from __future__ import annotations

from synthrig.assets.wheelrail import WheelRailParams, rolled
from synthrig.render.scene_api import register_scene
from scenes.wheel_rail import WheelRailScene


@register_scene("wheel_tread")
class WheelTreadScene(WheelRailScene):
    def sample_sequence(self, rng, n_frames: int, fps: float) -> list[dict]:
        """One draw for the whole sequence; the wheel then rolls at speed_mps."""
        p0 = WheelRailParams(**self.sample_params(rng))
        if p0.surface_modifiers:
            raise NotImplementedError("surface_modifiers (tread defects) are a hook only for now")
        return [rolled(p0, k / fps).model_dump(mode="json") for k in range(n_frames)]
