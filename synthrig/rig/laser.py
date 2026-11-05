"""Dot-pattern laser projector model. Renderer-agnostic.

A pattern is a fixed list of dot directions in the projector frame (OpenCV-style,
z = emission axis), stored as tangents (tx, ty) = (x/z, y/z). That list is the analytic
ground truth; the projector texture handed to the renderer is rasterised FROM it, so every
rendered dot has a known 3D ray.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .schema import FilePattern, GridPattern, Laser, RandomPattern
from .transforms import pose_to_T


def wavelength_to_rgb(nm: float) -> list[float]:
    """Approximate linear RGB for a monochromatic wavelength (Bruton-style, no gamma).
    The renderer is RGB, not spectral: fine for colour separation, not for radiometry."""
    if 380 <= nm < 440:
        r, g, b = -(nm - 440) / 60, 0.0, 1.0
    elif 440 <= nm < 490:
        r, g, b = 0.0, (nm - 440) / 50, 1.0
    elif 490 <= nm < 510:
        r, g, b = 0.0, 1.0, -(nm - 510) / 20
    elif 510 <= nm < 580:
        r, g, b = (nm - 510) / 70, 1.0, 0.0
    elif 580 <= nm < 645:
        r, g, b = 1.0, -(nm - 645) / 65, 0.0
    elif 645 <= nm <= 780:
        r, g, b = 1.0, 0.0, 0.0
    else:  # NIR/UV: treat as grey so a mono sensor still sees it; set color_rgb explicitly
        r, g, b = 1.0, 1.0, 1.0
    return [float(r), float(g), float(b)]


def pattern_tangents(p) -> np.ndarray:
    """(N,2) dot directions as tangents in the projector frame."""
    if isinstance(p, GridPattern):
        ax = (np.arange(p.cols) - (p.cols - 1) / 2) * np.deg2rad(p.pitch_deg[0])
        ay = (np.arange(p.rows) - (p.rows - 1) / 2) * np.deg2rad(p.pitch_deg[1])
        gx, gy = np.meshgrid(ax, ay)
        ang = np.stack([gx.ravel(), gy.ravel()], 1)
        if p.jitter_deg > 0:
            ang += np.random.default_rng(p.seed).normal(0, np.deg2rad(p.jitter_deg), ang.shape)
        return np.tan(ang)
    if isinstance(p, RandomPattern):
        rng = np.random.default_rng(p.seed)
        half = np.tan(np.deg2rad(np.asarray(p.fov_deg)) / 2)
        pts: list[np.ndarray] = []
        min_sep = np.tan(np.deg2rad(p.min_separation_deg)) if p.min_separation_deg > 0 else 0
        tries = 0
        while len(pts) < p.count:
            tries += 1
            if tries > p.count * 200:
                raise ValueError("random pattern: min_separation too large for count/fov")
            c = rng.uniform(-half, half)
            if min_sep and pts and np.min(np.linalg.norm(np.asarray(pts) - c, axis=1)) < min_sep:
                continue
            pts.append(c)
        return np.asarray(pts)
    if isinstance(p, FilePattern):
        a = np.load(p.path) if p.path.endswith(".npy") else np.loadtxt(p.path, delimiter=",", ndmin=2)
        return np.tan(np.deg2rad(np.asarray(a, float)[:, :2]))
    raise TypeError(p)


@dataclass
class LaserModel:
    name: str
    T_rig_proj: np.ndarray  # projector -> rig
    tangents: np.ndarray  # (N,2)
    color_rgb: np.ndarray
    power_w: float
    dot_sigma_rad: float
    spec: Laser = field(repr=False, default=None)

    @classmethod
    def from_spec(cls, l: Laser) -> "LaserModel":
        rgb = l.color_rgb if l.color_rgb is not None else wavelength_to_rgb(l.wavelength_nm)
        return cls(l.name, pose_to_T(l.pose), pattern_tangents(l.pattern), np.asarray(rgb, float),
                   l.power_w, l.dot_sigma_mrad * 1e-3, l)

    @property
    def directions(self) -> np.ndarray:
        """Unit ray directions (N,3) in the projector frame."""
        d = np.concatenate([self.tangents, np.ones((len(self.tangents), 1))], 1)
        return d / np.linalg.norm(d, axis=1, keepdims=True)

    def texture(self, max_res: int = 8192, px_per_sigma: float = 1.5):
        """Rasterise the pattern into a projector texture.

        Returns (img HxW float32 with row 0 = top = most negative ty, half_tan (hx, hy)).
        Texture pixel (i, j) covers tangent range; dots are Gaussians at their exact float
        positions so the intensity centroid matches the analytic direction.
        """
        pad = 4 * self.dot_sigma_rad
        hx = float(np.abs(self.tangents[:, 0]).max() + pad)
        hy = float(np.abs(self.tangents[:, 1]).max() + pad)
        step = self.dot_sigma_rad / px_per_sigma  # tangent units per texture pixel
        W = int(min(max_res, np.ceil(2 * hx / step)))
        H = int(min(max_res, np.ceil(2 * hy / step)))
        img = np.zeros((H, W), np.float32)
        sx, sy = W / (2 * hx), H / (2 * hy)  # px per tangent
        for tx, ty in self.tangents:
            # edge coordinates of the dot centre
            ex, ey = (tx + hx) * sx, (ty + hy) * sy
            sgx, sgy = self.dot_sigma_rad * sx, self.dot_sigma_rad * sy
            r = int(np.ceil(4 * max(sgx, sgy))) + 1
            i0, i1 = max(int(ex) - r, 0), min(int(ex) + r + 1, W)
            j0, j1 = max(int(ey) - r, 0), min(int(ey) + r + 1, H)
            if i0 >= i1 or j0 >= j1:
                continue
            ii = np.arange(i0, i1) + 0.5
            jj = np.arange(j0, j1) + 0.5
            g = np.exp(-((ii[None, :] - ex) ** 2) / (2 * sgx ** 2) - ((jj[:, None] - ey) ** 2) / (2 * sgy ** 2))
            np.maximum(img[j0:j1, i0:i1], g.astype(np.float32), out=img[j0:j1, i0:i1])
        return img, (hx, hy)
