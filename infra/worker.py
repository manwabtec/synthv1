#!/usr/bin/env python3
"""Render one shard of a dataset run into a run directory on persistent storage.

Container entrypoint (see infra/docker/entrypoint.sh); `cloud.py submit` starts one
worker container per shard:

  worker.py --config configs/runs/<x>.yaml --run-dir /data/datasets/<scene>/<rig_hash>/<run_id>
            --shard-index K --num-shards N --frames-total F
            [--scale s] [--samples n] [--device OPTIX|CUDA|GPU|CPU] [--scratch /scratch]

Shard contract (docs/TASK_infra.md step 0.4):
  * shard K renders the contiguous block start = K * ceil(F / N), up to F;
  * a frame is done only when frames/<NNNNNN>/_DONE exists; done frames are skipped,
    so re-running a shard finishes only the missing frames;
  * each frame renders into local scratch, is moved into the run dir under a temporary
    name, renamed into place, and only then gets `_DONE`;
  * the header (manifest.json, rig.yaml, calib/) is written once by `submit`; workers
    refuse to run without it, or if the rig/scale do not match it;
  * before every frame the run dir's filesystem must have --min-free-gb free;
  * SIGTERM/SIGINT abandons the current frame, cleans scratch and exits 143.
Logs go to stdout and to <run-dir>/logs/shard_<K>.log.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import signal
import sys
import time
from pathlib import Path
from typing import Callable, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from infra.storage.disk import check_free  # noqa: E402

DONE = "_DONE"
EXIT_INTERRUPTED = 143


class Interrupted(BaseException):
    """Abandon the shard (SIGTERM/SIGINT). BaseException so generic handlers don't swallow it."""


def shard_range(shard: int, num_shards: int, frames_total: int) -> range:
    """Frame indices (0-based, before the run's start offset) for one shard."""
    if not 0 <= shard < num_shards:
        raise ValueError(f"shard index {shard} outside 0..{num_shards - 1}")
    block = math.ceil(frames_total / num_shards)
    start = shard * block
    return range(min(start, frames_total), min(start + block, frames_total))


def frame_dir(run_dir: Path, idx: int) -> Path:
    return run_dir / "frames" / f"{idx:06d}"


def is_done(run_dir: Path, idx: int) -> bool:
    return (frame_dir(run_dir, idx) / DONE).is_file()


def publish(rendered: Path, run_dir: Path, idx: int, shard: int) -> Path:
    """Move a rendered frame dir into the run dir, then mark it done (last)."""
    final = frame_dir(run_dir, idx)
    tmp = final.with_name(f".{final.name}.tmp-{shard}")
    if tmp.exists():
        shutil.rmtree(tmp)
    shutil.move(str(rendered), str(tmp))  # rename if same filesystem, else copy + delete
    if final.exists():  # a partial dir left by an earlier crash (it has no _DONE)
        shutil.rmtree(final)
    os.replace(tmp, final)
    (final / DONE).write_text(json.dumps({"shard": shard, "time_unix": round(time.time(), 3)}) + "\n")
    return final


def run_shard(render: Callable[[int], Path], run_dir: Path, indices: Iterable[int], shard: int,
              min_free_gb: float, log: Callable[[str], None],
              stop_requested: Callable[[], bool] = lambda: False) -> dict:
    """Render and publish every not-yet-done frame in `indices`. Returns counts.

    `stop_requested` is checked before each frame. It backs up the signal handler, whose
    exception can be swallowed if it fires inside Python code that Blender itself runs.
    """
    counts = {"skipped": 0, "rendered": 0}
    for idx in indices:
        if stop_requested():
            raise Interrupted("stop requested")
        if is_done(run_dir, idx):
            counts["skipped"] += 1
            continue
        check_free(run_dir, min_free_gb)
        t = time.time()
        rendered = render(idx)
        publish(rendered, run_dir, idx, shard)
        counts["rendered"] += 1
        log(f"[worker {shard}] frame {idx:06d} published in {time.time() - t:.1f}s")
    return counts


def _check_header(run_dir: Path, rig_hash: str, scale: float) -> None:
    manifest = run_dir / "manifest.json"
    if not manifest.is_file():
        raise SystemExit(f"{manifest} is missing: write the header first (cloud.py submit or python -m synthrig --header-only)")
    m = json.loads(manifest.read_text())
    if m.get("rig_hash") != rig_hash or abs(float(m.get("render_scale", -1)) - scale) > 1e-9:
        raise SystemExit(f"{manifest} was written for rig {m.get('rig_hash')} at scale {m.get('render_scale')}, "
                         f"but this worker has rig {rig_hash} at scale {scale}; refusing to mix them")


def describe(config: str) -> dict:
    """Scene name and rig hash for a run config (used by `cloud.py submit` for the run path)."""
    from synthrig.pipeline import RunConfig, load_scene
    from synthrig.opticsrig.schema import load_rig

    cfg = RunConfig.load(config)
    scene = load_scene(cfg.scene, cfg.scene_params, cfg.scene_config)
    return {"scene": getattr(scene, "name", cfg.scene.split(":")[-1]), "rig_hash": load_rig(cfg.rig).hash()}


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--describe", action="store_true", help="print scene name + rig hash as JSON and exit")
    ap.add_argument("--run-dir", help="dataset directory holding manifest.json (absolute or under the data root)")
    ap.add_argument("--shard-index", type=int, default=int(os.environ.get("SHARD_INDEX", 0)))
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--frames-total", type=int)
    ap.add_argument("--scale", type=float)
    ap.add_argument("--samples", type=int)
    ap.add_argument("--device", choices=["CPU", "GPU", "OPTIX", "CUDA"])
    ap.add_argument("--scratch", default=os.environ.get("SYNTHRIG_SCRATCH", "/scratch"))
    ap.add_argument("--min-free-gb", type=float, default=float(os.environ.get("SYNTHRIG_MIN_FREE_GB", 20)))
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    a = parser().parse_args(argv)
    if a.describe:
        print(json.dumps(describe(a.config)))
        return 0
    if not a.run_dir or a.frames_total is None:
        raise SystemExit("--run-dir and --frames-total are required")

    from synthrig.paths import output_path
    from synthrig.pipeline import Pipeline, RunConfig

    run_dir = output_path(a.run_dir)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    log_file = (run_dir / "logs" / f"shard_{a.shard_index}.log").open("a", buffering=1)

    def log(msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}"
        print(line, flush=True)
        log_file.write(line + "\n")

    stop = {"signal": None}

    def on_signal(signum, _frame):
        stop["signal"] = signal.Signals(signum).name
        raise Interrupted(stop["signal"])

    cfg = RunConfig.load(a.config)
    if a.scale is not None: cfg.render.render_scale = a.scale
    if a.samples is not None: cfg.render.samples = a.samples
    if a.device is not None: cfg.render.device = a.device
    scratch = Path(a.scratch) / f"{run_dir.name}_shard{a.shard_index}"
    cfg.output = str(scratch)
    indices = [cfg.start + i for i in shard_range(a.shard_index, a.num_shards, a.frames_total)]
    log(f"[worker {a.shard_index}] run {run_dir} shard {a.shard_index}/{a.num_shards}: "
        f"frames {indices[0] if indices else '-'}..{indices[-1] if indices else '-'}, scratch {scratch}")
    try:
        pipe = Pipeline(cfg, log=log, write_header=False)
        # Install after Pipeline(): importing bpy replaces Python's SIGTERM/SIGINT handlers.
        signal.signal(signal.SIGTERM, on_signal)
        signal.signal(signal.SIGINT, on_signal)
        _check_header(run_dir, pipe.rig.spec.hash(), pipe.rig.scale)

        def render(idx: int) -> Path:
            pipe.render_frame(idx)
            return scratch / "frames" / f"{idx:06d}"

        counts = run_shard(render, run_dir, indices, a.shard_index, a.min_free_gb, log,
                           stop_requested=lambda: stop["signal"] is not None)
        log(f"[worker {a.shard_index}] finished: {counts['rendered']} rendered, {counts['skipped']} already done")
        return 0
    except Interrupted as exc:
        log(f"[worker {a.shard_index}] {stop['signal'] or exc} received: stopped before finishing the shard; "
            "unfinished frames have no _DONE. Resume to finish them.")
        return EXIT_INTERRUPTED
    except OSError as exc:
        log(f"[worker {a.shard_index}] stopped: {exc}")
        return 1
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
        log_file.close()


if __name__ == "__main__":
    raise SystemExit(main())
