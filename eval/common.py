"""Shared evaluation helpers. Read the dataset only (no simulator / bpy imports).

detect_dots, match: dot detection in camera images and GT matching (eval/dots_eval.py).
dot_gt_bias:        intensity centroid in the clean laser image vs GT uv (isolated dots).
disparity_at_dots:  GT disparity map vs the rectified GT dot pairs.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.spatial import cKDTree


def detect_dots(img: np.ndarray, kernel: int = 9, k_sigma: float = 8.0, min_area: int = 2):
    g = img.astype(np.float32)
    if g.ndim == 3:
        g = g.mean(-1)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    top = g - cv2.morphologyEx(g, cv2.MORPH_OPEN, se)
    noise = 1.4826 * np.median(np.abs(top - np.median(top))) + 1e-3
    mask = (top > k_sigma * noise).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    ys, xs = np.nonzero(lab)
    w = top[ys, xs]
    ids = lab[ys, xs]
    sw = np.bincount(ids, w, n)
    cx = np.bincount(ids, w * xs, n) / np.maximum(sw, 1e-9)
    cy = np.bincount(ids, w * ys, n) / np.maximum(sw, 1e-9)
    keep = np.arange(1, n)[stats[1:, cv2.CC_STAT_AREA] >= min_area]
    return np.stack([cx[keep], cy[keep]], 1)


def match(det, gt, radius):
    if len(det) == 0 or len(gt) == 0:
        return np.full(len(gt), -1)
    d, j = cKDTree(det).query(gt, distance_upper_bound=radius)
    j[~np.isfinite(d)] = -1
    # enforce one-to-one: keep the closest GT per detection
    best = {}
    for gi, (dj, dd) in enumerate(zip(j, d)):
        if dj >= 0 and (dj not in best or dd < best[dj][1]):
            best[dj] = (gi, dd)
    out = np.full(len(gt), -1)
    for dj, (gi, _) in best.items():
        out[gi] = dj
    return out


def dot_lasers(rig: dict) -> list[str]:
    """Names of enabled lasers with dot patterns (line lasers have no isolated dots)."""
    return [l["name"] for l in rig.get("lasers", []) if l.get("enabled", True) and l["pattern"]["type"] != "line"]


def dot_gt_bias(frame, cam: str, laser: str, laser_index: int, isolation_px: float = 6.0,
                window: int = 3) -> np.ndarray:
    """(N,2) offsets: intensity centroid in laser_<laser>.exr minus GT uv, per isolated dot.

    A dot is used if no other visible GT dot or line sample (any laser) lies within
    isolation_px and its centroid window lies inside the image.
    """
    dots = frame.dots()
    vis = dots[f"visible_{cam}"]
    uv_all = dots[f"uv_{cam}"][vis]
    mine = (dots["laser"][vis] == laser_index)
    img = frame.laser(cam, laser).astype(np.float64)
    lum = img.mean(-1) if img.ndim == 3 else img
    H, W = lum.shape
    if len(uv_all) < 2:
        return np.zeros((0, 2))
    d, _ = cKDTree(uv_all).query(uv_all, k=2)
    isolated = d[:, 1] > isolation_px
    out = []
    for u, v in uv_all[mine & isolated]:
        ci, cj = int(round(u)), int(round(v))
        if ci - window < 0 or cj - window < 0 or ci + window >= W or cj + window >= H:
            continue
        patch = lum[cj - window: cj + window + 1, ci - window: ci + window + 1]
        if patch.sum() <= 0:
            continue
        jj, ii = np.mgrid[cj - window: cj + window + 1, ci - window: ci + window + 1]
        out.append([(patch * ii).sum() / patch.sum() - u, (patch * jj).sum() / patch.sum() - v])
    return np.asarray(out).reshape(-1, 2)


def _bilinear(img: np.ndarray, uv: np.ndarray):
    """Bilinear samples at pixel coords (pixel centres at integers); NaN where a neighbour
    is invalid (0 = invalid in GT maps)."""
    H, W = img.shape
    u, v = uv[:, 0], uv[:, 1]
    i0, j0 = np.floor(u).astype(int), np.floor(v).astype(int)
    ok = (i0 >= 0) & (j0 >= 0) & (i0 + 1 < W) & (j0 + 1 < H)
    out = np.full(len(uv), np.nan)
    i0, j0, fu, fv = i0[ok], j0[ok], (u - np.floor(u))[ok], (v - np.floor(v))[ok]
    q = np.stack([img[j0, i0], img[j0, i0 + 1], img[j0 + 1, i0], img[j0 + 1, i0 + 1]], 1)
    val = (q[:, 0] * (1 - fu) * (1 - fv) + q[:, 1] * fu * (1 - fv) + q[:, 2] * (1 - fu) * fv + q[:, 3] * fu * fv)
    val[(q <= 0).any(1)] = np.nan
    out[ok] = val
    return out


def disparity_at_dots(frame, pair: str, left: str, right: str, with_gradient: bool = False):
    """GT disparity at each rectified left GT dot minus (x_left - x_right); finite values only.

    with_gradient=True also returns the local disparity gradient |grad d| (px/px) at each
    dot. The GT map samples depth at the nearest pinhole-canvas pixel (no mixing across
    edges), so on a slanted surface its error is bounded by ~0.5 px x |grad d|.
    """
    dots = frame.dots()
    both = dots[f"visible_{left}"] & dots[f"visible_{right}"]
    if not both.any():
        return (np.zeros(0), np.zeros(0)) if with_gradient else np.zeros(0)
    ul = dots[f"uv_rect_{pair}_left"][both]
    ur = dots[f"uv_rect_{pair}_right"][both]
    disp = frame.stereo(pair)["disparity"].astype(np.float64)
    d = _bilinear(disp, ul) - (ul[:, 0] - ur[:, 0])
    ok = np.isfinite(d)
    if not with_gradient:
        return d[ok]
    gy, gx = np.gradient(disp)
    grad = np.hypot(gx, gy)
    grad[disp <= 0] = 0
    g = _bilinear(np.where(disp > 0, grad + 1e-12, 0.0), ul)
    return d[ok], np.nan_to_num(g[ok], nan=np.inf)
