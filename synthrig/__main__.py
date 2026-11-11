"""Render a synthetic dataset (run from the repository root).

  python -m synthrig configs/runs/app1_demo.yaml
  python -m synthrig configs/runs/app1_demo.yaml --frames 2 --scale 0.25 --samples 16 --out datasets/preview
  python -m synthrig configs/runs/app1_demo.yaml --header-only --out datasets/run   # manifest/calib only
"""
import argparse

from synthrig.paths import output_path
from synthrig.pipeline import Pipeline, RunConfig


def main():
    ap = argparse.ArgumentParser(prog="python -m synthrig", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--frames", type=int)
    ap.add_argument("--start", type=int, help="first frame index (shard across machines)")
    ap.add_argument("--scale", type=float, help="render_scale override (e.g. 0.25 preview)")
    ap.add_argument("--samples", type=int)
    ap.add_argument("--device", choices=["CPU", "GPU", "OPTIX", "CUDA"])
    ap.add_argument("--out")
    ap.add_argument("--data-root", help="data root (default: sibling data/ or SYNTHRIG_DATA_ROOT)")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--header-only", action="store_true",
                    help="write manifest.json, rig.yaml and calib/ only (no frames)")
    a = ap.parse_args()
    cfg = RunConfig.load(a.config)
    if a.frames is not None: cfg.frames = a.frames
    if a.start is not None: cfg.start = a.start
    if a.scale is not None: cfg.render.render_scale = a.scale
    if a.samples is not None: cfg.render.samples = a.samples
    if a.device is not None: cfg.render.device = a.device
    if a.out is not None: cfg.output = a.out
    cfg.output = str(output_path(cfg.output, a.data_root))
    if a.seed is not None: cfg.seed = a.seed
    if a.header_only:
        Pipeline(cfg, write_header=False).write_header()
    else:
        Pipeline(cfg).run()


if __name__ == "__main__":
    main()
