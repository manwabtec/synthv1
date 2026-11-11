"""Export the configured simulation scene for inspection in Blender's viewport."""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import bpy

from .assembly import build_frame
from .blender_backend import BlenderBackend, RenderSettings
from ..opticsrig import RigModel, load_rig
from ..paths import output_path
from ..pipeline import RunConfig, load_scene


def export_scene(config: Path, destination: Path, frame: int = 0) -> Path:
    """Write a .blend using the same seed/frame assembly as the dataset renderer."""
    if frame < 0:
        raise ValueError("Frame index must be non-negative")
    if destination.suffix.lower() != ".blend":
        raise ValueError("Scene destination must end in .blend")
    cfg = RunConfig.load(config)
    rig = RigModel.build(load_rig(cfg.rig), cfg.render.render_scale)
    backend = BlenderBackend(RenderSettings(**cfg.render.model_dump(exclude={"render_scale"})))
    scene = load_scene(cfg.scene, cfg.scene_params, cfg.scene_config)
    build_frame(backend, rig, scene, cfg.seed, frame, cfg.sequence_cfg)
    destination.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(destination))
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--out", default="scenes/preview.blend")
    parser.add_argument("--data-root")
    parser.add_argument("--open", action="store_true", help="launch installed Blender with the saved scene")
    args = parser.parse_args()
    executable = shutil.which("blender") if args.open else None
    if args.open and executable is None:
        parser.error("--open requires the Blender executable on PATH")
    destination = export_scene(args.config, output_path(args.out, args.data_root), args.frame)
    print(f"Scene saved: {destination}")
    if executable:
        subprocess.Popen([executable, str(destination)])


if __name__ == "__main__":
    main()