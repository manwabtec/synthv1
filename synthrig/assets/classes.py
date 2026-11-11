"""Segmentation class taxonomy shared by every scene (docs/TASK_scenes.md section 1). No bpy.

Ids are stable across scenes and written into manifest.json; append new classes at the end.
"""
from __future__ import annotations

CLASSES: tuple[str, ...] = (
    "background", "wheel_tread", "wheel_flange", "wheel_rim_face", "wheel_plate", "wheel_hub",
    "axle", "rail_head", "rail_web", "rail_foot", "tie", "fastener", "tie_plate", "ballast",
    "truck_frame", "bolster", "brake_component", "car_body", "coupler", "ground", "other",
)
CLASS_ID: dict[str, int] = {name: i for i, name in enumerate(CLASSES)}
