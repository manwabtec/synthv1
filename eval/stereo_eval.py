#!/usr/bin/env python3
"""Dense stereo evaluation: run a matcher on the rectified pairs, compare to GT disparity.
Reads the dataset only. Swap `run_matcher` for your own algorithm.

  python eval/stereo_eval.py out/app1_demo [--json results.json]
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from synthrig.dataset import Dataset  # noqa: E402


def to8(img):
    g = img.astype(np.float32)
    if g.ndim == 3:
        g = g.mean(-1)
    lo, hi = np.percentile(g, [0.5, 99.5])
    return np.clip((g - lo) / max(hi - lo, 1) * 255, 0, 255).astype(np.uint8)


def run_matcher(left, right, max_disp):
    nd = int(np.ceil(max_disp * 1.2 / 16) * 16)
    sgbm = cv2.StereoSGBM_create(minDisparity=0, numDisparities=max(nd, 16), blockSize=5,
                                 P1=8 * 25, P2=32 * 25, uniquenessRatio=5, speckleWindowSize=100,
                                 speckleRange=2, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    d = sgbm.compute(to8(left), to8(right)).astype(np.float32) / 16.0
    d[d <= 0] = np.nan
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--json")
    a = ap.parse_args()
    ds = Dataset(a.dataset)
    stats = {p: {"epe": [], "bad1": [], "bad3": [], "density": []} for p in ds.manifest["stereo_pairs"]}
    for fr in ds.frames():
        for p, sc in ds.stereo_calib.items():
            if sc["vertical"]:
                continue  # transpose images for vertical rigs if you need this
            st = fr.stereo(p)
            gt, valid = st["disparity"], st["valid"]
            est = run_matcher(st["left"], st["right"], gt[valid].max() if valid.any() else 64)
            ok = valid & np.isfinite(est)
            err = np.abs(est[ok] - gt[ok])
            s = stats[p]
            s["density"].append(ok.sum() / max(valid.sum(), 1))
            s["epe"].append(float(err.mean()) if err.size else np.nan)
            s["bad1"].append(float((err > 1).mean()) if err.size else np.nan)
            s["bad3"].append(float((err > 3).mean()) if err.size else np.nan)
            print(f"frame {fr.id} {p}: EPE {s['epe'][-1]:.3f}px  bad>1px {100 * s['bad1'][-1]:.1f}%  "
                  f"bad>3px {100 * s['bad3'][-1]:.1f}%  density {100 * s['density'][-1]:.1f}%")
    res = {p: {k: float(np.nanmean(v)) for k, v in s.items()} for p, s in stats.items() if s["epe"]}
    for p, r in res.items():
        print(f"== {p} mean: EPE {r['epe']:.3f}px, bad>1 {100 * r['bad1']:.1f}%, bad>3 {100 * r['bad3']:.1f}%, "
              f"density {100 * r['density']:.1f}%")
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
