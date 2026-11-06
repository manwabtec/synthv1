"""On-disk dataset format: the ONLY interface between the simulator and evaluation.

Layout (FORMAT_VERSION 1):
  <root>/manifest.json            rig name+hash, cameras, stereo pairs, lasers, run config
  <root>/rig.yaml                 validated rig (exact copy of what rendered this)
  <root>/calib/cam_<cam>.yml      OpenCV FileStorage: K, D, image size, T_rig_cam
  <root>/calib/stereo_<pair>.yml  R, T, R1, R2, P1, P2, Q
  <root>/calib/perturbed/...      same files with injected calibration error (optional)
  <root>/frames/<NNNNNN>/
      meta.json                   seed, scene params, T_world_rig, object index -> class
      dots.npz                    laser-dot GT (see DotsGT); per-camera uv/visible, rectified uv
      <cam>/image.png             uint16 DN (mono) or BGR (colour); what the camera delivers
      <cam>/raw.png               Bayer mosaic (colour cameras with bayer)
      <cam>/radiance.exr          linear radiance, distorted geometry, pre-sensor (optional)
      <cam>/depth.exr             planar Z (m) along optical axis, 0 = invalid
      <cam>/seg.png               uint16 object index (0 = background)
      <cam>/laser_<name>.exr      that laser's linear contribution (optional)
      <cam>/laser_<name>_mask.png uint8 {0,255}
      stereo_<pair>/left.png, right.png   rectified camera images
      stereo_<pair>/disparity.exr         GT disparity on rectified left grid (px), 0 = invalid
      stereo_<pair>/depth_valid.png       uint8 mask of valid GT
Real captured data can be written in the same layout (without GT files) and every eval
script runs on it unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import OpenEXR

from ..opticsrig.calib import read_camera_yaml, read_stereo_yaml

FORMAT_VERSION = 1


# ---------------------------------------------------------------- image io
def write_exr(path, arr: np.ndarray):
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    hdr = {"compression": OpenEXR.ZIP_COMPRESSION, "type": OpenEXR.scanlineimage}
    ch = {"RGB": arr} if arr.ndim == 3 else {"Z": arr}
    OpenEXR.File(hdr, ch).write(str(path))


def read_exr(path) -> np.ndarray:
    ch = OpenEXR.File(str(path)).channels()
    for k in ("RGB", "RGBA", "Z", "V", "Y"):
        if k in ch:
            return ch[k].pixels
    return next(iter(ch.values())).pixels


def write_png(path, arr: np.ndarray):
    if not cv2.imwrite(str(path), arr):
        raise IOError(path)


def read_png(path) -> np.ndarray:
    a = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if a is None:
        raise FileNotFoundError(path)
    return a


# ---------------------------------------------------------------- reader
class Frame:
    def __init__(self, root: Path):
        self.root = root
        self.id = root.name
        self.meta = json.loads((root / "meta.json").read_text()) if (root / "meta.json").exists() else {}

    def has(self, rel: str) -> bool:
        return (self.root / rel).exists()

    def image(self, cam):
        return read_png(self.root / cam / "image.png")

    def depth(self, cam):
        return read_exr(self.root / cam / "depth.exr")

    def seg(self, cam):
        return read_png(self.root / cam / "seg.png")

    def radiance(self, cam):
        return read_exr(self.root / cam / "radiance.exr")

    def laser(self, cam, name):
        return read_exr(self.root / cam / f"laser_{name}.exr")

    def laser_mask(self, cam, name):
        return read_png(self.root / cam / f"laser_{name}_mask.png") > 0

    def stereo(self, pair):
        d = self.root / f"stereo_{pair}"
        out = {"left": read_png(d / "left.png"), "right": read_png(d / "right.png")}
        if (d / "disparity.exr").exists():
            out["disparity"] = read_exr(d / "disparity.exr")
            out["valid"] = read_png(d / "depth_valid.png") > 0
        return out

    def dots(self) -> dict | None:
        p = self.root / "dots.npz"
        if not p.exists():
            return None
        z = np.load(p, allow_pickle=False)
        return {k: z[k] for k in z.files}


class Dataset:
    def __init__(self, root, perturbed: bool = False):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if self.manifest.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"unsupported dataset format {self.manifest.get('format_version')}")
        cdir = self.root / "calib" / ("perturbed" if perturbed else "")
        self.cameras = {c: read_camera_yaml(cdir / f"cam_{c}.yml") for c in self.manifest["cameras"]}
        self.stereo_calib = {p: read_stereo_yaml(cdir / f"stereo_{p}.yml") for p in self.manifest["stereo_pairs"]}

    def frames(self):
        for d in sorted((self.root / "frames").iterdir()):
            if d.is_dir():
                yield Frame(d)

    def __len__(self):
        return sum(1 for d in (self.root / "frames").iterdir() if d.is_dir())
