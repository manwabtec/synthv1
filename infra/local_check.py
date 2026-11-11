"""Verify dependencies, writable data storage and optional NVIDIA rendering."""
from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from synthrig.paths import CODE_ROOT, data_root


def check(*, gpu: bool = False, render: bool = False) -> None:
    """Raise on missing prerequisites; GPU and smoke-render checks are opt-in."""
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("Python 3.11 is required for the pinned Blender 4.2 module")
    required = ["numpy", "cv2", "pydantic", "yaml", "scipy", "OpenEXR"]
    if gpu or render:
        required.append("bpy")
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError(f"Missing dependencies: {', '.join(missing)}; run uv sync --extra render for Blender")
    root = data_root()
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryFile(dir=root) as probe:
        probe.write(b"synthrig storage check")
    print(f"Python: {sys.version.split()[0]}; dependencies present; writable data root: {root}")
    if gpu:
        if shutil.which("nvidia-smi") is None:
            raise RuntimeError("nvidia-smi is missing; install/check the NVIDIA driver")
        subprocess.run(["nvidia-smi"], check=True)
        subprocess.run([sys.executable, "-m", "synthrig.render.devices"], cwd=CODE_ROOT, check=True)
    if render:
        command = [sys.executable, "-m", "synthrig",
                   str(CODE_ROOT / "configs/debug_clutter/run.yaml"), "--frames", "1",
                   "--scale", "0.125", "--samples", "4", "--out", "diagnostics/smoke",
                   "--device", "GPU" if gpu else "CPU"]
        subprocess.run(command, cwd=CODE_ROOT, check=True)
        if not list((root / "diagnostics/smoke/frames/000000").glob("*/image.png")):
            raise RuntimeError("Smoke render did not produce camera images")
        print(f"Smoke-render output: {root / 'diagnostics/smoke'}")