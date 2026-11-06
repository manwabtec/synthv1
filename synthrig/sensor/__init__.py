"""Sensor model applied AFTER rendering (renderer-agnostic, numpy + OpenCV).

Input: linear radiance in the distorted camera geometry (H,W,3).
Output: integer digital numbers (DN) as the camera would deliver them.

  radiance --(rgb_response)--> per-pixel signal --(exposure*full_well)--> electrons
  -> shot noise (Poisson) -> blooming of over-full-well charge -> clip at full well
  -> read noise (Gaussian, e-) -> gain to DN -> offset -> quantise/clip to bit depth
  Colour + Bayer: mosaic before noise, demosaic after (so noise/aliasing look like a real CFA).
Because this runs on saved linear images, you can sweep exposure/noise without re-rendering.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..opticsrig.schema import Sensor

_BAYER_CODES = {"RGGB": cv2.COLOR_BayerRG2BGR, "BGGR": cv2.COLOR_BayerBG2BGR,
                "GRBG": cv2.COLOR_BayerGR2BGR, "GBRG": cv2.COLOR_BayerGB2BGR}
_BAYER_LAYOUT = {"RGGB": ((0, 1), (1, 2)), "BGGR": ((2, 1), (1, 0)),
                 "GRBG": ((1, 0), (2, 1)), "GBRG": ((1, 2), (0, 1))}


@dataclass
class SensorOutput:
    image: np.ndarray  # uint16 DN: (H,W) mono or (H,W,3) BGR for colour
    raw: np.ndarray | None  # uint16 Bayer mosaic (colour+bayer only)
    saturated: np.ndarray  # bool (H,W): pixels at/over full well


def mosaic(rgb: np.ndarray, pattern: str) -> np.ndarray:
    H, W, _ = rgb.shape
    lay = _BAYER_LAYOUT[pattern]
    out = np.empty((H, W), rgb.dtype)
    for dy in range(2):
        for dx in range(2):
            out[dy::2, dx::2] = rgb[dy::2, dx::2, lay[dy][dx]]
    return out


def apply_sensor(radiance_rgb: np.ndarray, s: Sensor, rng: np.random.Generator) -> SensorOutput:
    rad = np.clip(radiance_rgb.astype(np.float64), 0, None)
    resp = np.asarray(s.rgb_response, float)
    if s.mono:
        sig = rad @ resp  # (H,W)
    else:
        sig = rad * resp  # (H,W,3)
        if s.bayer:
            sig = mosaic(sig, s.bayer)
    e = sig * s.exposure * s.full_well_e
    if s.shot_noise:
        e = rng.poisson(np.clip(e, 0, 1e9)).astype(np.float64)
    if s.bloom.enabled:
        excess = np.clip(e - s.full_well_e, 0, None)
        if excess.any():
            spread = cv2.GaussianBlur(excess, (0, 0), s.bloom.sigma_px)
            e = e + s.bloom.fraction * spread
    saturated = e >= s.full_well_e
    e = np.minimum(e, s.full_well_e)
    if s.read_noise_e > 0:
        e = e + rng.normal(0, s.read_noise_e, e.shape)
    max_dn = 2 ** s.bit_depth - 1
    dn = np.clip(np.rint(e * (max_dn / s.full_well_e) + s.dark_offset_dn), 0, max_dn).astype(np.uint16)
    if saturated.ndim == 3:
        saturated = saturated.any(-1)
    if not s.mono and s.bayer:
        bgr = cv2.cvtColor(dn, _BAYER_CODES[s.bayer])
        return SensorOutput(bgr, dn, saturated)
    if not s.mono:
        dn = dn[..., ::-1].copy()  # RGB -> BGR for OpenCV-style writing
    return SensorOutput(dn, None, saturated)
