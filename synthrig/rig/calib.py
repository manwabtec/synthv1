"""Calibration export (OpenCV FileStorage YAML, readable from C++ and Python) and
controlled calibration perturbation for sensitivity studies."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .camera import CameraModel
from .schema import StereoPair
from .transforms import inv_T, make_T


@dataclass
class StereoCalib:
    name: str
    left: str
    right: str
    size: tuple[int, int]
    R: np.ndarray  # x_right = R x_left + T
    T: np.ndarray
    R1: np.ndarray
    R2: np.ndarray
    P1: np.ndarray
    P2: np.ndarray
    Q: np.ndarray
    roi1: tuple
    roi2: tuple

    @property
    def vertical(self) -> bool:
        return abs(self.P2[1, 3]) > abs(self.P2[0, 3])

    def disparity_from_depth(self, z_rect: np.ndarray) -> np.ndarray:
        """Disparity for points at depth z along the rectified optical axis."""
        k = -(self.P2[1, 3] if self.vertical else self.P2[0, 3])
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(z_rect > 0, k / z_rect, 0.0)


def stereo_calib(pair: StereoPair, L: CameraModel, R: CameraModel) -> StereoCalib:
    T_r_l = inv_T(R.T_rig_cam) @ L.T_rig_cam
    Rm, Tv = T_r_l[:3, :3], T_r_l[:3, 3]
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        L.K, L.D, R.K, R.D, L.size, Rm, Tv, flags=cv2.CALIB_ZERO_DISPARITY, alpha=pair.alpha)
    return StereoCalib(pair.name, L.name, R.name, L.size, Rm, Tv, R1, R2, P1, P2, Q, roi1, roi2)


def write_camera_yaml(path: Path, cam: CameraModel):
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    fs.write("name", cam.name)
    fs.write("image_width", cam.size[0])
    fs.write("image_height", cam.size[1])
    fs.write("K", cam.K)
    fs.write("D", cam.D.reshape(1, -1))
    fs.write("T_rig_cam", cam.T_rig_cam)
    fs.write("pixel_size_um", cam.pixel_size_um)
    fs.release()


def write_stereo_yaml(path: Path, s: StereoCalib):
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    fs.write("name", s.name)
    fs.write("left", s.left)
    fs.write("right", s.right)
    fs.write("image_width", s.size[0])
    fs.write("image_height", s.size[1])
    for k in ("R", "T", "R1", "R2", "P1", "P2", "Q"):
        fs.write(k, np.asarray(getattr(s, k), float).reshape(3, -1) if k == "T" else getattr(s, k))
    fs.write("vertical", int(s.vertical))
    fs.release()


def read_camera_yaml(path: Path) -> dict:
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    out = {"name": fs.getNode("name").string(),
           "size": (int(fs.getNode("image_width").real()), int(fs.getNode("image_height").real())),
           "K": fs.getNode("K").mat(), "D": fs.getNode("D").mat().ravel(),
           "T_rig_cam": fs.getNode("T_rig_cam").mat()}
    fs.release()
    return out


def read_stereo_yaml(path: Path) -> dict:
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    out = {k: fs.getNode(k).string() for k in ("name", "left", "right")}
    out["size"] = (int(fs.getNode("image_width").real()), int(fs.getNode("image_height").real()))
    for k in ("R", "T", "R1", "R2", "P1", "P2", "Q"):
        out[k] = fs.getNode(k).mat()
    out["T"] = out["T"].ravel()
    out["vertical"] = bool(int(fs.getNode("vertical").real()))
    fs.release()
    return out


def perturb_camera(cam: CameraModel, rng: np.random.Generator, rotation_deg=0.0, translation_mm=0.0,
                   focal_pct=0.0, principal_px=0.0, distortion_rel=0.0) -> CameraModel:
    """Return a copy with a random calibration error of the given 1-sigma magnitudes.
    The images are NOT changed: only what the downstream algorithm believes."""
    c = copy.deepcopy(cam)
    df = 1 + rng.normal(0, focal_pct / 100)
    c.K[0, 0] *= df
    c.K[1, 1] *= df
    c.K[:2, 2] += rng.normal(0, principal_px, 2)
    c.D = c.D * (1 + rng.normal(0, distortion_rel, c.D.shape))
    dR = Rotation.from_rotvec(rng.normal(0, np.deg2rad(rotation_deg), 3)).as_matrix()
    dT = make_T(dR, rng.normal(0, translation_mm / 1000, 3))
    c.T_rig_cam = c.T_rig_cam @ dT
    return c
