#!/usr/bin/env python3
"""Render a synthetic dataset.

  python run.py configs/runs/app1_demo.yaml
  python run.py configs/runs/app1_demo.yaml --frames 2 --scale 0.25 --samples 16 --out out/preview
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from synthrig.pipeline import Pipeline, RunConfig  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--frames", type=int)
    ap.add_argument("--start", type=int, help="first frame index (shard across machines)")
    ap.add_argument("--scale", type=float, help="render_scale override (e.g. 0.25 preview)")
    ap.add_argument("--samples", type=int)
    ap.add_argument("--out")
    ap.add_argument("--seed", type=int)
    a = ap.parse_args()
    cfg = RunConfig.load(a.config)
    if a.frames is not None: cfg.frames = a.frames
    if a.start is not None: cfg.start = a.start
    if a.scale is not None: cfg.render.render_scale = a.scale
    if a.samples is not None: cfg.render.samples = a.samples
    if a.out is not None: cfg.output = a.out
    if a.seed is not None: cfg.seed = a.seed
    Pipeline(cfg).run()


if __name__ == "__main__":
    main()
