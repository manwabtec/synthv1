#!/usr/bin/env python3
"""Contact sheet for one frame: every camera image with GT dots overlaid, laser masks,
rectified stereo pair and GT disparity.  Reads the dataset only.

  python eval/preview.py out/app1_demo --frame 0 --out preview.png
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from synthrig.dataset import Dataset  # noqa: E402

DOT_COLORS = [(60, 60, 255), (60, 255, 60), (255, 120, 60), (255, 60, 255)]


def to8(img, bits=12):
    a = img.astype(np.float32) / (2 ** bits - 1)
    a = np.clip(a ** (1 / 2.2), 0, 1) * 255  # display gamma only
    a = a.astype(np.uint8)
    return cv2.cvtColor(a, cv2.COLOR_GRAY2BGR) if a.ndim == 2 else a


def label(img, text):
    cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def tile(imgs, cols, w):
    rows = []
    for i in range(0, len(imgs), cols):
        row = [cv2.resize(im, (w, int(im.shape[0] * w / im.shape[1]))) for im in imgs[i:i + cols]]
        h = max(r.shape[0] for r in row)
        row = [cv2.copyMakeBorder(r, 0, h - r.shape[0], 0, 0, cv2.BORDER_CONSTANT) for r in row]
        while len(row) < cols:
            row.append(np.zeros_like(row[0]))
        rows.append(np.hstack(row))
    return np.vstack(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--out", default="preview.png")
    ap.add_argument("--width", type=int, default=640)
    a = ap.parse_args()
    ds = Dataset(a.dataset)
    fr = list(ds.frames())[a.frame]
    rig = ds.manifest
    import yaml
    rigspec = yaml.safe_load((Path(a.dataset) / "rig.yaml").read_text())
    bits = {c["name"]: c["sensor"]["bit_depth"] for c in rigspec["cameras"]}
    dots = fr.dots()
    panels = []
    for c in rig["cameras"]:
        im = to8(fr.image(c), bits[c])
        overlay = im.copy()
        if dots is not None:
            for li in range(len(dots["laser_names"])):
                sel = (dots["laser"] == li) & dots[f"visible_{c}"]
                for u, v in dots[f"uv_{c}"][sel]:
                    cv2.circle(overlay, (int(round(u)), int(round(v))), 3, DOT_COLORS[li % 4], 1, cv2.LINE_AA)
        panels.append(label(im, f"{c} image"))
        panels.append(label(overlay, f"{c} + GT dots (visible)"))
    for c in rig["cameras"][:1]:
        for l in rig["lasers"]:
            if fr.has(f"{c}/laser_{l}_mask.png"):
                m = fr.laser_mask(c, l).astype(np.uint8) * 255
                panels.append(label(cv2.cvtColor(m, cv2.COLOR_GRAY2BGR), f"{c} laser '{l}' mask"))
    for p in rig["stereo_pairs"]:
        st = fr.stereo(p)
        L = to8(st["left"], bits[rig["stereo"][p]["left"]])
        R = to8(st["right"], bits[rig["stereo"][p]["right"]])
        pair = np.hstack([L, R])
        for y in range(0, pair.shape[0], max(pair.shape[0] // 12, 1)):
            cv2.line(pair, (0, y), (pair.shape[1], y), (0, 200, 255), 1)
        panels.append(label(pair[:, : L.shape[1]].copy(), f"{p} rectified left (epipolar lines)"))
        panels.append(label(pair[:, L.shape[1]:].copy(), f"{p} rectified right"))
        if "disparity" in st:
            d = st["disparity"]
            v = st["valid"]
            lo, hi = np.percentile(d[v], [1, 99]) if v.any() else (0, 1)
            dn = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
            col = cv2.applyColorMap((dn * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
            col[~v] = 0
            panels.append(label(col, f"{p} GT disparity {lo:.0f}-{hi:.0f}px"))
    sheet = tile(panels, 2, a.width)
    cv2.imwrite(a.out, sheet)
    print(f"wrote {a.out}  {sheet.shape[1]}x{sheet.shape[0]}")


if __name__ == "__main__":
    main()
