"""Camera model: OpenCV pinhole + Brown-Conrady distortion, render-canvas planning and
distortion remap maps. Renderer-agnostic."""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .schema import Camera
from .transforms import pose_to_T


@dataclass
class CameraModel:
    name: str
    K: np.ndarray  # 3x3
    D: np.ndarray  # (n,)
    size: tuple[int, int]  # (width, height)
    T_rig_cam: np.ndarray  # 4x4, cam -> rig
    pixel_size_um: float
    spec: Camera = field(repr=False, default=None)

    @classmethod
    def from_spec(cls, cam: Camera, scale: float = 1.0) -> "CameraModel":
        i = cam.intrinsics
        K = np.array([[i.fx, 0, i.cx], [0, i.fy, i.cy], [0, 0, 1.0]])
        size = (i.width, i.height)
        px = i.pixel_size_um
        if scale != 1.0:
            # scale about pixel edges: centre convention (c + 0.5) * s - 0.5
            w, h = round(i.width * scale), round(i.height * scale)
            sx, sy = w / i.width, h / i.height
            K = np.array([[i.fx * sx, 0, (i.cx + 0.5) * sx - 0.5],
                          [0, i.fy * sy, (i.cy + 0.5) * sy - 0.5], [0, 0, 1.0]])
            size, px = (w, h), px / sx
        return cls(cam.name, K, np.asarray(cam.distortion, float), size, pose_to_T(cam.pose), px, cam)

    @property
    def has_distortion(self) -> bool:
        return bool(np.any(np.abs(self.D) > 0))

    # ---------------------------------------------------------------- projection
    def project(self, X_cam: np.ndarray) -> np.ndarray:
        """Project Nx3 points in the camera frame to distorted pixel coords (Nx2)."""
        X = np.asarray(X_cam, float).reshape(-1, 1, 3)
        uv, _ = cv2.projectPoints(X, np.zeros(3), np.zeros(3), self.K, self.D)
        return uv.reshape(-1, 2)

    def undistort_to_normalized(self, uv: np.ndarray) -> np.ndarray:
        uv = np.asarray(uv, np.float64).reshape(-1, 1, 2)
        if not self.has_distortion:
            n = (uv.reshape(-1, 2) - self.K[:2, 2]) / np.diag(self.K)[:2]
            return n
        crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 40, 1e-10)
        return cv2.undistortPointsIter(uv, self.K, self.D, None, None, crit).reshape(-1, 2)

    # ---------------------------------------------------------------- render planning
    def render_plan(self, margin_px: int = 4) -> "RenderPlan":
        """Pinhole canvas that covers every ray of the distorted image (plus a margin for
        interpolation), and the remap that turns that pinhole render into this camera's image.
        Without distortion the canvas is the image itself and no remap is needed."""
        w, h = self.size
        if not self.has_distortion:
            return RenderPlan(self.K.copy(), (w, h), None, None)
        fx, fy = self.K[0, 0], self.K[1, 1]
        # dense border + grid sample to find the undistorted extent
        us = np.linspace(-0.5, w - 0.5, 256)
        vs = np.linspace(-0.5, h - 0.5, 256)
        border = np.concatenate([
            np.stack([us, np.full_like(us, -0.5)], 1), np.stack([us, np.full_like(us, h - 0.5)], 1),
            np.stack([np.full_like(vs, -0.5), vs], 1), np.stack([np.full_like(vs, w - 0.5), vs], 1)])
        n = self.undistort_to_normalized(border)
        u_p, v_p = n[:, 0] * fx, n[:, 1] * fy  # pinhole pixel offsets from principal point
        left = int(np.ceil(-u_p.min())) + margin_px
        right = int(np.ceil(u_p.max())) + margin_px
        top = int(np.ceil(-v_p.min())) + margin_px
        bottom = int(np.ceil(v_p.max())) + margin_px
        Kr = np.array([[fx, 0, left - 0.5], [0, fy, top - 0.5], [0, 0, 1.0]])
        # principal point at pixel-edge coordinate `left` => centre coordinate left - 0.5
        size_r = (left + right, top + bottom)
        # remap: for every distorted output pixel, where to sample in the pinhole canvas
        uu, vv = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
        nn = self.undistort_to_normalized(np.stack([uu.ravel(), vv.ravel()], 1))
        map_x = (nn[:, 0] * fx + Kr[0, 2]).reshape(h, w).astype(np.float32)
        map_y = (nn[:, 1] * fy + Kr[1, 2]).reshape(h, w).astype(np.float32)
        return RenderPlan(Kr, size_r, map_x, map_y)


@dataclass
class RenderPlan:
    K_render: np.ndarray
    size_render: tuple[int, int]
    map_x: np.ndarray | None  # None => identity (no distortion)
    map_y: np.ndarray | None

    def apply(self, img: np.ndarray, nearest: bool = False) -> np.ndarray:
        """Pinhole render -> distorted camera image."""
        if self.map_x is None:
            return img
        interp = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
        return cv2.remap(img, self.map_x, self.map_y, interp, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
