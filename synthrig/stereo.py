"""Stereo rectification and exact ground-truth disparity (renderer-agnostic)."""
from __future__ import annotations

import cv2
import numpy as np

from .opticsrig.calib import StereoCalib
from .opticsrig.camera import CameraModel, RenderPlan


def rectify_maps(s: StereoCalib, L: CameraModel, R: CameraModel):
    m1 = cv2.initUndistortRectifyMap(L.K, L.D, s.R1, s.P1, s.size, cv2.CV_32FC1)
    m2 = cv2.initUndistortRectifyMap(R.K, R.D, s.R2, s.P2, s.size, cv2.CV_32FC1)
    return m1, m2


def rectify_pair(maps, left: np.ndarray, right: np.ndarray, nearest=False):
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    (l1, l2), (r1, r2) = maps
    return cv2.remap(left, l1, l2, interp), cv2.remap(right, r1, r2, interp)


def gt_disparity(s: StereoCalib, plan_left: RenderPlan, depth_left_pinhole: np.ndarray):
    """GT disparity on the rectified LEFT grid, sampled straight from the left pinhole
    depth render (no double interpolation through the distorted image).

    For rectified pixel p: ray d_rect = P1[:3,:3]^-1 p, d_left = R1^T d_rect, sample the
    pinhole depth Z_left at K_render d_left; then z_rect = (R1 X_left)_z and
    disparity = -P2[0,3] / z_rect (or P2[1,3] for vertical rigs)."""
    W, H = s.size
    uu, vv = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    p = np.stack([uu.ravel(), vv.ravel(), np.ones(uu.size)], 0)
    d_rect = np.linalg.inv(s.P1[:3, :3]) @ p
    d_left = s.R1.T @ d_rect
    Kr = plan_left.K_render
    with np.errstate(divide="ignore", invalid="ignore"):
        u = Kr[0, 0] * d_left[0] / d_left[2] + Kr[0, 2]
        v = Kr[1, 1] * d_left[1] / d_left[2] + Kr[1, 2]
    Wr, Hr = plan_left.size_render
    ui, vi = np.rint(u).astype(np.int64), np.rint(v).astype(np.int64)
    ok = (d_left[2] > 0) & (ui >= 0) & (ui < Wr) & (vi >= 0) & (vi < Hr)
    Z = np.zeros(u.size)
    Z[ok] = depth_left_pinhole[vi[ok], ui[ok]]
    ok &= Z > 0
    X_left = d_left * np.where(ok, Z / np.where(ok, d_left[2], 1), 0)
    z_rect = (s.R1 @ X_left)[2]
    disp = np.where(ok, s.disparity_from_depth(np.where(ok, z_rect, 1.0)), 0.0)
    return disp.reshape(H, W).astype(np.float32), ok.reshape(H, W)


def rectify_points(uv: np.ndarray, cam: CameraModel, Rr: np.ndarray, P: np.ndarray) -> np.ndarray:
    out = np.full_like(uv, np.nan, dtype=np.float64)
    ok = np.isfinite(uv).all(1)
    if ok.any():
        crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 40, 1e-10)
        out[ok] = cv2.undistortPointsIter(uv[ok].reshape(-1, 1, 2).astype(np.float64), cam.K, cam.D,
                                          Rr, P, crit).reshape(-1, 2)
    return out
