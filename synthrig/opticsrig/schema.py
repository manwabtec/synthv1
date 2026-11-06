"""Rig schema: the single source of truth for cameras, lasers and rig-mounted lights.

Conventions (see README "Coordinate frames"):
  * Units are metres, degrees, micrometres (pixel size) and milliradians (dot size) as named.
  * Every device frame (camera, laser projector, light) is OpenCV-style:
      x right, y down, z forward (optical / emission axis).
  * Device poses are given in the RIG frame. The rig frame is arbitrary but by convention
    is also OpenCV-style (z forward, y down) with origin at the stereo midpoint.
  * Nothing in this file knows about Blender.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- poses
class Pose(_Strict):
    """Pose of a device frame in the rig frame (device -> rig).

    Give `position` plus exactly one orientation form:
      look_at: target point in rig frame (optical axis points at it); `up` sets roll
               (default [0,-1,0] = rig "up" because y is down).
      euler_xyz_deg: R = Rz @ Ry @ Rx applied to the device axes (identity = aligned with rig).
      R: explicit 3x3 rotation matrix (columns = device axes expressed in rig frame).
    """

    position: list[float] = Field(..., min_length=3, max_length=3)
    look_at: Optional[list[float]] = Field(None, min_length=3, max_length=3)
    up: list[float] = Field([0.0, -1.0, 0.0], min_length=3, max_length=3)
    euler_xyz_deg: Optional[list[float]] = Field(None, min_length=3, max_length=3)
    R: Optional[list[list[float]]] = None

    @model_validator(mode="after")
    def _one_orientation(self):
        n = sum(x is not None for x in (self.look_at, self.euler_xyz_deg, self.R))
        if n > 1:
            raise ValueError("pose: give only one of look_at / euler_xyz_deg / R")
        return self


# --------------------------------------------------------------------------- cameras
class Intrinsics(_Strict):
    width: int = Field(..., gt=0)
    height: int = Field(..., gt=0)
    pixel_size_um: float = Field(..., gt=0)
    # Either fx/fy in pixels or focal_length_mm (converted with pixel size).
    fx: Optional[float] = None
    fy: Optional[float] = None
    focal_length_mm: Optional[float] = None
    # OpenCV convention: pixel centres at integer coordinates. Default = image centre.
    cx: Optional[float] = None
    cy: Optional[float] = None

    @model_validator(mode="after")
    def _focal(self):
        if self.fx is None and self.focal_length_mm is None:
            raise ValueError("intrinsics: give fx (px) or focal_length_mm")
        if self.fx is None:
            self.fx = self.focal_length_mm * 1000.0 / self.pixel_size_um
        if self.fy is None:
            self.fy = self.fx
        if self.cx is None:
            self.cx = (self.width - 1) / 2.0
        if self.cy is None:
            self.cy = (self.height - 1) / 2.0
        if abs(self.fx - self.fy) / self.fx > 1e-6:
            raise ValueError("intrinsics: fx != fy (non-square pixels) is not supported by the Blender backend")
        return self


class Bloom(_Strict):
    enabled: bool = False
    fraction: float = 0.3  # fraction of over-full-well charge that spills
    sigma_px: float = 2.0


class Sensor(_Strict):
    """Post-render sensor model (applied in synthrig.sensor, never in the renderer)."""

    mono: bool = True
    # Response to the renderer's linear R,G,B channels. For mono: weights -> single channel.
    # For colour: per-channel quantum efficiency multipliers.
    rgb_response: list[float] = Field([0.30, 0.59, 0.11], min_length=3, max_length=3)
    bayer: Optional[Literal["RGGB", "BGGR", "GRBG", "GBRG"]] = None  # colour only
    bit_depth: int = Field(12, ge=8, le=16)
    full_well_e: float = 10000.0
    read_noise_e: float = 3.0
    dark_offset_dn: float = 0.0
    # electrons = radiance * exposure * full_well_e  (radiance is Blender's linear units)
    exposure: float = 1.0
    shot_noise: bool = True
    bloom: Bloom = Bloom()


class Camera(_Strict):
    name: str
    intrinsics: Intrinsics
    # OpenCV distortion vector: k1 k2 p1 p2 [k3 [k4 k5 k6]]
    distortion: list[float] = Field(default_factory=lambda: [0.0] * 5)
    pose: Pose
    sensor: Sensor = Sensor()


class StereoPair(_Strict):
    name: str
    left: str
    right: str
    # cv2.stereoRectify alpha: 0 = only valid pixels, 1 = keep all, -1 = OpenCV default
    alpha: float = 0.0


# --------------------------------------------------------------------------- lasers
class GridPattern(_Strict):
    type: Literal["grid"] = "grid"
    rows: int = Field(..., gt=0)
    cols: int = Field(..., gt=0)
    pitch_deg: list[float] = Field(..., min_length=2, max_length=2)  # [x, y] angular pitch
    jitter_deg: float = 0.0  # random per-dot angular jitter (fixed by `seed`: it's hardware)
    seed: int = 0


class RandomPattern(_Strict):
    type: Literal["random"] = "random"
    count: int = Field(..., gt=0)
    fov_deg: list[float] = Field(..., min_length=2, max_length=2)  # full fov [x, y]
    min_separation_deg: float = 0.0
    seed: int = 0


class FilePattern(_Strict):
    """CSV/NPY of dot directions as angles in degrees: columns [angle_x, angle_y]."""

    type: Literal["file"] = "file"
    path: str


class Laser(_Strict):
    name: str
    pose: Pose
    pattern: Union[GridPattern, RandomPattern, FilePattern] = Field(..., discriminator="type")
    # Colour: give wavelength (converted to approximate RGB) or explicit linear RGB.
    wavelength_nm: Optional[float] = None
    color_rgb: Optional[list[float]] = Field(None, min_length=3, max_length=3)
    # Blender light power (W). Only relative brightness matters: calibrate with sensor.exposure.
    power_w: float = 20.0
    dot_sigma_mrad: float = 0.5  # Gaussian 1-sigma angular dot radius
    enabled: bool = True

    @model_validator(mode="after")
    def _colour(self):
        if self.color_rgb is None and self.wavelength_nm is None:
            raise ValueError(f"laser {self.name}: give wavelength_nm or color_rgb")
        return self


# --------------------------------------------------------------------------- lights
class Light(_Strict):
    """Rig-mounted illumination (strobes, ring lights). Moves with the rig."""

    name: str
    type: Literal["area", "point", "spot"] = "area"
    pose: Pose
    power_w: float = 50.0
    color_rgb: list[float] = Field([1.0, 1.0, 1.0], min_length=3, max_length=3)
    size_m: float = 0.1  # area light edge / point+spot radius (soft shadows)
    spot_angle_deg: float = 60.0


# --------------------------------------------------------------------------- rig
class Rig(_Strict):
    name: str
    version: int = 1
    description: str = ""
    cameras: list[Camera]
    stereo_pairs: list[StereoPair] = []
    lasers: list[Laser] = []
    lights: list[Light] = []

    @model_validator(mode="after")
    def _refs(self):
        names = [c.name for c in self.cameras]
        if len(set(names)) != len(names):
            raise ValueError("camera names must be unique")
        for p in self.stereo_pairs:
            for n in (p.left, p.right):
                if n not in names:
                    raise ValueError(f"stereo pair {p.name}: unknown camera {n}")
            L = self.camera(p.left).intrinsics
            R = self.camera(p.right).intrinsics
            if (L.width, L.height) != (R.width, R.height):
                raise ValueError(f"stereo pair {p.name}: cameras must share resolution")
        lnames = [l.name for l in self.lasers]
        if len(set(lnames)) != len(lnames):
            raise ValueError("laser names must be unique")
        return self

    def camera(self, name: str) -> Camera:
        for c in self.cameras:
            if c.name == name:
                return c
        raise KeyError(name)

    def laser(self, name: str) -> Laser:
        for l in self.lasers:
            if l.name == name:
                return l
        raise KeyError(name)

    def canonical_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))

    def hash(self) -> str:
        """Content hash stored with every dataset: identifies exactly which rig made it."""
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()[:16]


def load_rig(path: str | Path) -> Rig:
    path = Path(path)
    data = yaml.safe_load(path.read_text())
    rig = Rig.model_validate(data)
    # resolve relative pattern file paths against the rig file location
    for l in rig.lasers:
        if isinstance(l.pattern, FilePattern) and not Path(l.pattern.path).is_absolute():
            l.pattern.path = str((path.parent / l.pattern.path).resolve())
    return rig
