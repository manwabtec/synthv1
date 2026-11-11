"""Renderer-agnostic rig: schema, camera & laser models, transforms, calibration export."""
from __future__ import annotations

from dataclasses import dataclass

from .camera import CameraModel, RenderPlan
from .calib import StereoCalib, stereo_calib
from .laser import LaserModel
from .schema import Rig, load_rig
from .transforms import inv_T, make_T, pose_to_T, rig_pose_look_at, transform_points

__all__ = ["Rig", "load_rig", "RigModel", "CameraModel", "LaserModel", "RenderPlan", "StereoCalib",
           "stereo_calib", "make_T", "inv_T", "pose_to_T", "rig_pose_look_at", "transform_points"]


@dataclass
class RigModel:
    """Validated rig + derived numeric models (what the pipeline actually uses)."""

    spec: Rig
    cameras: dict[str, CameraModel]
    lasers: dict[str, LaserModel]
    stereo: dict[str, StereoCalib]  # rectified pairs only (rectify: true)
    scale: float = 1.0

    @classmethod
    def build(cls, rig: Rig, scale: float = 1.0) -> "RigModel":
        cams = {c.name: CameraModel.from_spec(c, scale) for c in rig.cameras}
        lasers = {l.name: LaserModel.from_spec(l) for l in rig.lasers if l.enabled}
        stereo = {p.name: stereo_calib(p, cams[p.left], cams[p.right]) for p in rig.stereo_pairs
                  if p.rectify}
        return cls(rig, cams, lasers, stereo, scale)
