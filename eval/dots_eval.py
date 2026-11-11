#!/usr/bin/env python3
"""Laser-dot evaluation. Reads the dataset only (no simulator imports).

1. Detection: find dots in each camera image (top-hat + threshold + intensity centroid),
   match to GT projections of visible dots -> recall, precision, centroid error (px).
2. Stereo triangulation: for dots visible in both cameras of a pair, triangulate the
   DETECTED centroids with the dataset calibration and compare to GT 3D (mm).
   Run with --perturbed to use calib/perturbed/ and see calibration sensitivity.

  python eval/dots_eval.py datasets/debug_clutter [--perturbed] [--json results.json]
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.common import detect_dots, match  # noqa: E402,F401  (shared with tests)
from synthrig.dataset import Dataset  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--perturbed", action="store_true", help="triangulate with calib/perturbed")
    ap.add_argument("--radius", type=float, default=2.5, help="match radius (px)")
    ap.add_argument("--kernel", type=int, default=9, help="top-hat kernel (px)")
    ap.add_argument("--json")
    a = ap.parse_args()
    ds = Dataset(a.dataset, perturbed=a.perturbed)
    cams = ds.manifest["cameras"]
    acc = {c: {"gt": 0, "tp": 0, "det": 0, "err": []} for c in cams}
    tri = {p: [] for p in ds.manifest["stereo_pairs"]}
    for fr in ds.frames():
        dots = fr.dots()
        det_uv = {}
        for c in cams:
            det = detect_dots(fr.image(c), a.kernel)
            vis = dots[f"visible_{c}"]
            gt = dots[f"uv_{c}"][vis]
            m = match(det, gt, a.radius)
            ok = m >= 0
            A = acc[c]
            A["gt"] += len(gt)
            A["det"] += len(det)
            A["tp"] += int(ok.sum())
            A["err"] += list(np.linalg.norm(det[m[ok]] - gt[ok], axis=1))
            full = np.full((len(vis), 2), np.nan)
            idx = np.nonzero(vis)[0]
            full[idx[ok]] = det[m[ok]]
            det_uv[c] = full
        for p, sc in ds.stereo_calib.items():
            L, R = sc["left"], sc["right"]
            both = np.isfinite(det_uv[L]).all(1) & np.isfinite(det_uv[R]).all(1)
            if not both.any():
                continue
            cl, cr = ds.cameras[L], ds.cameras[R]
            ul = cv2.undistortPoints(det_uv[L][both].reshape(-1, 1, 2), cl["K"], cl["D"], R=sc["R1"], P=sc["P1"])
            ur = cv2.undistortPoints(det_uv[R][both].reshape(-1, 1, 2), cr["K"], cr["D"], R=sc["R2"], P=sc["P2"])
            Xh = cv2.triangulatePoints(sc["P1"], sc["P2"], ul.reshape(-1, 2).T, ur.reshape(-1, 2).T)
            X_rect = (Xh[:3] / Xh[3]).T
            # GT: rig frame -> left camera -> rectified left (always TRUE calibration)
            T_rig_left = Dataset(a.dataset).cameras[L]["T_rig_cam"]
            Xr = dots["X_rig"][both]
            X_left = (Xr - T_rig_left[:3, 3]) @ T_rig_left[:3, :3]
            R1_true = Dataset(a.dataset).stereo_calib[p]["R1"]
            X_gt = X_left @ R1_true.T
            tri[p] += list(np.linalg.norm(X_rect - X_gt, axis=1) * 1000)

    res = {"calibration": "perturbed" if a.perturbed else "true", "detection": {}, "triangulation_mm": {}}
    print(f"calibration: {res['calibration']}")
    print(f"{'camera':8s} {'gt':>7s} {'det':>7s} {'recall':>7s} {'prec':>7s} {'err_med':>8s} {'err_rms':>8s}")
    for c, A in acc.items():
        e = np.asarray(A["err"])
        r = {"gt": A["gt"], "detected": A["det"], "recall": A["tp"] / max(A["gt"], 1),
             "precision": A["tp"] / max(A["det"], 1),
             "err_median_px": float(np.median(e)) if len(e) else None,
             "err_rms_px": float(np.sqrt(np.mean(e ** 2))) if len(e) else None}
        res["detection"][c] = r
        print(f"{c:8s} {r['gt']:7d} {r['detected']:7d} {r['recall']:7.3f} {r['precision']:7.3f} "
              f"{(r['err_median_px'] or 0):8.3f} {(r['err_rms_px'] or 0):8.3f}")
    for p, e in tri.items():
        e = np.asarray(e)
        if len(e):
            r = {"n": int(len(e)), "median": float(np.median(e)), "p95": float(np.percentile(e, 95)),
                 "rms": float(np.sqrt(np.mean(e ** 2)))}
            res["triangulation_mm"][p] = r
            print(f"stereo {p}: {r['n']} dots, 3D error median {r['median']:.3f} mm, "
                  f"p95 {r['p95']:.3f} mm, rms {r['rms']:.3f} mm")
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
