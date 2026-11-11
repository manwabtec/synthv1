#!/usr/bin/env python3
"""GPU check (docs/TASK_infra.md step 0.3). Runs inside the container with --gpus all.

1. Prints nvidia-smi and Blender's Cycles device list.
2. Enables OptiX (falls back to CUDA).
3. Renders the demo frame on GPU and then on CPU at the same settings and prints both times.
4. Fails unless a GPU device was enabled and the GPU render was faster.
Writes <out>/result.json; frames go to <out>/gpu and <out>/cpu under the data root.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEMO = Path(__file__).resolve().parents[1] / "configs/runs/app1_demo.yaml"


def cycles_devices() -> list[tuple[str, str]]:
    """(type, name) for every device Cycles can see, across all backends."""
    import bpy

    prefs = bpy.context.preferences.addons["cycles"].preferences
    found = []
    for backend in ("OPTIX", "CUDA", "NONE"):
        try:
            prefs.compute_device_type = backend
            prefs.refresh_devices()
        except (TypeError, RuntimeError):
            continue
        found += [(d.type, d.name) for d in prefs.devices if (d.type, d.name) not in found]
    return found


def render_time(device: str, out: str, scale: float, samples: int) -> float:
    """Seconds to render frame 0 of the demo run (scene setup excluded)."""
    from synthrig.pipeline import Pipeline, RunConfig

    cfg = RunConfig.load(DEMO)
    cfg.frames, cfg.output = 1, out
    cfg.render.device, cfg.render.render_scale, cfg.render.samples = device, scale, samples
    cfg.calib_perturbation.enabled = False
    pipe = Pipeline(cfg)
    t = time.time()
    pipe.render_frame(0)
    return time.time() - t


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="diagnostics/gpucheck", help="under the data root, or absolute")
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--skip-cpu", action="store_true", help="GPU render only (no speed comparison)")
    a = ap.parse_args()

    from synthrig.paths import output_path
    from synthrig.render.devices import configure_gpu

    out = output_path(a.out)
    print("--- nvidia-smi")
    smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    print(smi.stdout or smi.stderr)
    print("--- Cycles devices")
    devices = cycles_devices()
    for kind, name in devices:
        print(f"  {kind:<6} {name}")
    enabled = configure_gpu()
    backend = enabled[0].split(":")[0]

    result = {"devices": devices, "enabled": enabled, "backend": backend,
              "scale": a.scale, "samples": a.samples}
    result["gpu_s"] = round(render_time(backend, str(out / "gpu"), a.scale, a.samples), 2)
    print(f"GPU ({backend}) render: {result['gpu_s']} s")
    if not a.skip_cpu:
        result["cpu_s"] = round(render_time("CPU", str(out / "cpu"), a.scale, a.samples), 2)
        print(f"CPU render: {result['cpu_s']} s  (speed-up x{result['cpu_s'] / result['gpu_s']:.1f})")

    gpu_devices = [d for d in devices if d[0] in ("OPTIX", "CUDA")]
    ok = bool(gpu_devices) and (a.skip_cpu or result["gpu_s"] < result["cpu_s"])
    result["passed"] = ok
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"gpucheck {'PASSED' if ok else 'FAILED'} ({out / 'result.json'})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
