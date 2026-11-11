#!/usr/bin/env python3
"""Run synthrig in Docker on an NVIDIA machine: image, GPU check, shard runs, status.

Works the same on the EC2 GPU instance and on a local Linux/WSL2 workstation; only
infra/config.yaml differs (EC2: written by infra/ec2/setup_host.sh). Host-side tool:
standard library + PyYAML only, Python >= 3.9, so it runs with the system python3.
Everything that needs Blender runs inside the Docker image.

  python3 infra/cloud.py status                 # GPU, Docker image, disk space, running runs
  python3 infra/cloud.py disk                   # free space + dataset sizes only
  python3 infra/cloud.py image build | test
  python3 infra/cloud.py gpucheck
  python3 infra/cloud.py submit --config configs/runs/app1_demo.yaml --frames 8 --shards 2
  python3 infra/cloud.py status --run datasets/<scene>/<rig_hash>/<run_id>
  python3 infra/cloud.py stop datasets/<scene>/<rig_hash>/<run_id>
  python3 infra/cloud.py resume datasets/<scene>/<rig_hash>/<run_id>
  python3 infra/cloud.py exec -- python run.py ...   # any command inside the container
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CONFIG_PATH = ROOT / "infra" / "config.yaml"

from infra.storage import disk  # noqa: E402
from synthrig.paths import data_root as default_data_root  # noqa: E402  (stdlib only)

# Machine-neutral defaults; infra/config.yaml overrides them (EC2 sets /app/synthrig etc.).
DEFAULTS: Dict[str, Any] = {
    "project": "synthrig",
    "image": "synthrig",
    "data_root": None,              # None: SYNTHRIG_DATA_ROOT or the repo's sibling data/
    "scratch_dir": None,            # None: <data_root>/scratch (EC2: instance-store /scratch)
    "docker_root": None,            # None: asked from `docker info` (only used for the disk report)
    "min_free_gb": 20,              # renders refuse to start / continue below this
    "max_parallel_shards": 2,       # worker containers sharing the one GPU
}
# Used only by infra/access/check.py from a laptop.
OPTIONAL = {"owner", "region", "aws_profile", "instance_id"}
CONTAINER_DATA = "/data"
CONTAINER_SCRATCH = "/scratch"
RUN_LABEL = "synthrig.run"


class CloudError(RuntimeError):
    """A user-correctable configuration or command error."""


# ---------------------------------------------------------------- configuration
def load_config(path: Path = CONFIG_PATH) -> Dict[str, Any]:
    """Defaults merged with infra/config.yaml (optional). Rejects typos and <placeholders>."""
    config = dict(DEFAULTS)
    if not path.is_file():
        return _resolve(config)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise CloudError(f"{path.name} must contain a YAML mapping")
    unknown = sorted(set(data) - set(DEFAULTS) - OPTIONAL)
    placeholders = sorted(k for k, v in data.items()
                          if isinstance(v, str) and (not v.strip() or "<" in v or ">" in v))
    if unknown or placeholders:
        details = ([f"unknown keys: {', '.join(unknown)}"] if unknown else []) + \
                  ([f"unresolved placeholders: {', '.join(placeholders)}"] if placeholders else [])
        raise CloudError(f"Invalid infra/config.yaml ({'; '.join(details)})")
    config.update({k: v for k, v in data.items() if v is not None})
    if not isinstance(config["min_free_gb"], (int, float)) or config["min_free_gb"] <= 0:
        raise CloudError("min_free_gb must be a positive number")
    return _resolve(config)


def _resolve(config: Dict[str, Any]) -> Dict[str, Any]:
    """Fill path defaults that depend on other values; paths become absolute strings."""
    config["data_root"] = str(default_data_root(config["data_root"]))
    config["scratch_dir"] = str(Path(config["scratch_dir"] or Path(config["data_root"]) / "scratch").expanduser())
    return config


def docker_root(config: Dict[str, Any]) -> Optional[str]:
    """Configured Docker data-root, else what the daemon reports (None if unreachable)."""
    if config.get("docker_root"):
        return str(config["docker_root"])
    try:
        out = run([*docker(), "info", "-f", "{{.DockerRootDir}}"], capture=True, check=False).stdout.strip()
        return out or None
    except CloudError:
        return None


# ---------------------------------------------------------------- helpers
def run(command: Sequence[str], *, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    if not capture:
        print("+", subprocess.list2cmdline(list(command)), flush=True)
    return subprocess.run(list(command), cwd=ROOT, check=check, text=True,
                          stdout=subprocess.PIPE if capture else None,
                          stderr=subprocess.PIPE if capture else None)


def docker() -> List[str]:
    """`docker`, or `sudo -n docker` when this user cannot reach the Docker socket."""
    if shutil.which("docker") is None:
        raise CloudError("docker is not installed")
    if os.access("/var/run/docker.sock", os.W_OK):
        return ["docker"]
    return ["sudo", "-n", "docker"]


def git(*args: str) -> str:
    return run(["git", *args], capture=True).stdout.strip()


def git_state() -> Dict[str, Any]:
    try:
        return {"sha": git("rev-parse", "--short", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
                "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}
    except (OSError, subprocess.CalledProcessError):
        return {"sha": "unknown", "branch": "unknown", "dirty": True}


def instance_type() -> str:
    """EC2 instance type from IMDSv2, or 'local' off EC2."""
    try:
        req = urllib.request.Request("http://169.254.169.254/latest/api/token", method="PUT",
                                     headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"})
        token = urllib.request.urlopen(req, timeout=1).read().decode()
        req = urllib.request.Request("http://169.254.169.254/latest/meta-data/instance-type",
                                     headers={"X-aws-ec2-metadata-token": token})
        return urllib.request.urlopen(req, timeout=1).read().decode()
    except OSError:
        return "local"


def image_ref(config: Dict[str, Any]) -> str:
    return f"{config['image']}:latest"


def image_info(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Id, creation time and git labels of the local image, or None if not built."""
    result = run([*docker(), "image", "inspect", image_ref(config)], check=False, capture=True)
    if result.returncode != 0:
        return None
    info = json.loads(result.stdout)[0]
    labels = info["Config"].get("Labels") or {}
    return {"id": info["Id"], "created": info["Created"][:19], "size_gb": info["Size"] / disk.GIB,
            "git_sha": labels.get("org.synthrig.git_sha", "unknown"),
            "git_dirty": labels.get("org.synthrig.git_dirty", "unknown")}


def container(config: Dict[str, Any], args: Sequence[str], *, gpu: bool = True, name: Optional[str] = None,
              labels: Optional[Dict[str, str]] = None, env: Optional[Dict[str, str]] = None,
              mount_code: bool = False, detach: bool = False, interactive: bool = False) -> List[str]:
    """`docker run` command line with the data root and scratch mounted, as this user."""
    cmd = [*docker(), "run"]
    cmd += ["-d", "--restart", "on-failure:3"] if detach else ["--rm"]
    if interactive:
        cmd.append("-it")
    if gpu:
        cmd += ["--gpus", "all"]
    if name:
        cmd += ["--name", name]
    for k, v in (labels or {}).items():
        cmd += ["--label", f"{k}={v}"]
    for path in (config["data_root"], config["scratch_dir"]):
        Path(path).mkdir(parents=True, exist_ok=True)  # else Docker creates it owned by root
    if hasattr(os, "getuid"):  # Linux/WSL2: write files as this user, not root
        cmd += ["--user", f"{os.getuid()}:{os.getgid()}"]
    cmd += ["-e", "HOME=/tmp",
            "-v", f"{config['data_root']}:{CONTAINER_DATA}", "-v", f"{config['scratch_dir']}:{CONTAINER_SCRATCH}"]
    if mount_code:
        cmd += ["-v", f"{ROOT}:/code"]
    full_env = {"SYNTHRIG_INSTANCE_TYPE": instance_type(), "SYNTHRIG_MIN_FREE_GB": str(config["min_free_gb"])}
    info = image_info(config)
    if info:
        full_env["SYNTHRIG_IMAGE_DIGEST"] = info["id"]
    full_env.update(env or {})
    for k, v in full_env.items():
        cmd += ["-e", f"{k}={v}"]
    return cmd + [image_ref(config), *args]


def repo_relative(path: str) -> str:
    """Config path as seen inside the image (/code is the repo root)."""
    p = Path(path)
    p = (p if p.is_absolute() else Path.cwd() / p).resolve()
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError as exc:
        raise CloudError(f"{path} must be inside the repository (it is baked into the image)") from exc


def volume_paths(config: Dict[str, Any]) -> Dict[str, str]:
    paths = {"data_root": config["data_root"], "scratch": config["scratch_dir"]}
    root = docker_root(config)
    if root:
        paths["docker_root"] = root
    paths["root_fs"] = Path.cwd().anchor or "/"
    return paths


# ---------------------------------------------------------------- run status (pure filesystem)
def run_status(run_dir: Path) -> Dict[str, Any]:
    """Frames done/total, render time and ETA for one run directory."""
    manifest = run_dir / "manifest.json"
    if not manifest.is_file():
        raise CloudError(f"{run_dir} has no manifest.json")
    total = int(json.loads(manifest.read_text())["run_config"]["frames"])
    frames = run_dir / "frames"
    done = sorted(d for d in frames.glob("[0-9]*") if (d / "_DONE").is_file()) if frames.is_dir() else []
    times = []
    for d in done:
        try:
            times.append(float(json.loads((d / "meta.json").read_text())["timings_s"]["total"]))
        except (OSError, KeyError, ValueError):
            pass
    mean = sum(times) / len(times) if times else None
    size = sum(f.stat().st_size for f in run_dir.rglob("*") if f.is_file())
    return {"run_dir": str(run_dir), "frames_total": total, "frames_done": len(done),
            "mean_frame_s": round(mean, 1) if mean else None,
            "render_hours": round(sum(times) / 3600, 3), "bytes": size,
            "remaining_s": round(mean * (total - len(done))) if mean else None}


# ---------------------------------------------------------------- commands
def cmd_config(_a: argparse.Namespace) -> None:
    config = load_config()
    source = "infra/config.yaml" if CONFIG_PATH.is_file() else "defaults (no infra/config.yaml)"
    print(f"Configuration from {source}:")
    for k, v in config.items():
        print(f"  {k}: {v}")


def cmd_disk(a: argparse.Namespace) -> int:
    config = load_config()
    datasets = str(Path(config["data_root"]) / "datasets")
    if a.json:
        print(json.dumps(disk.as_dict(volume_paths(config), datasets), indent=2))
        return 0
    text, ok = disk.report(volume_paths(config), config["min_free_gb"], datasets)
    print(text)
    return 0 if ok else 1


def _gpu_summary() -> str:
    if shutil.which("nvidia-smi") is None:
        raise CloudError("nvidia-smi not found")
    return run(["nvidia-smi", "--query-gpu=name,driver_version,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader"], capture=True).stdout.strip()


def cmd_status(a: argparse.Namespace) -> int:
    config = load_config()
    if a.run:
        return _run_report(config, a.run)
    problems = []
    g = git_state()
    print(f"== code\n{g['branch']} @ {g['sha']}{' (uncommitted changes)' if g['dirty'] else ''}")
    print("\n== GPU")
    try:
        print(_gpu_summary())
    except (CloudError, subprocess.CalledProcessError) as exc:
        print(f"FAIL: {exc}")
        problems.append("gpu")
    print("\n== Docker image")
    try:
        info = image_info(config)
        if info is None:
            print(f"FAIL: {image_ref(config)} not built; run: python3 infra/cloud.py image build")
            problems.append("image")
        else:
            stale = info["git_sha"] != g["sha"] or info["git_dirty"] == "true"
            print(f"{image_ref(config)} {info['id'][7:19]} built {info['created']} from {info['git_sha']}"
                  f"{' (dirty)' if info['git_dirty'] == 'true' else ''}, {info['size_gb']:.1f} GiB"
                  + ("  [STALE: rebuild with image build]" if stale else ""))
    except (CloudError, subprocess.CalledProcessError) as exc:
        print(f"FAIL: {exc}")
        problems.append("docker")
    print("\n== Storage")
    if instance_type() != "local" and not os.path.ismount(config["scratch_dir"]):
        print(f"WARN: scratch {config['scratch_dir']} is not a mounted volume; on EC2 run: "
              "bash infra/ec2/setup_host.sh (instance store is wiped on stop/start)")
    text, ok = disk.report(volume_paths(config), config["min_free_gb"],
                           str(Path(config["data_root"]) / "datasets"))
    print(text)
    if not ok:
        problems.append("disk")
    print("\n== Running shard workers")
    try:
        ps = run([*docker(), "ps", "--filter", f"label={RUN_LABEL}",
                  "--format", '{{.Names}}\t{{.Status}}\t{{.Label "' + RUN_LABEL + '"}}'], capture=True)
        print(ps.stdout.strip() or "(none)")
    except (CloudError, subprocess.CalledProcessError) as exc:
        print(f"(unavailable: {exc})")
    print(f"\nOVERALL: {'OK' if not problems else 'PROBLEMS: ' + ', '.join(problems)}")
    return 0 if not problems else 1


def _run_report(config: Dict[str, Any], run_arg: str) -> int:
    run_dir = Path(run_arg) if Path(run_arg).is_absolute() else Path(config["data_root"]) / run_arg
    s = run_status(run_dir)
    eta = f"~{s['remaining_s'] / 60:.0f} min single-shard" if s["remaining_s"] is not None else "unknown"
    print(f"{s['run_dir']}\n  frames {s['frames_done']}/{s['frames_total']} done, "
          f"mean {s['mean_frame_s']} s/frame, {s['render_hours']} render-hours, "
          f"{s['bytes'] / disk.GIB:.2f} GiB, remaining {eta}")
    rel = run_dir.relative_to(config["data_root"]).as_posix() if run_dir.is_relative_to(Path(config["data_root"])) \
        else str(run_dir)
    ps = run([*docker(), "ps", "-a", "--filter", f"label={RUN_LABEL}={rel}",
              "--format", "{{.Names}}\t{{.Status}}"], capture=True, check=False)
    print("  workers:\n    " + ("\n    ".join(ps.stdout.split("\n")).strip() or "(none)"))
    for log in sorted((run_dir / "logs").glob("shard_*.log")):
        lines = log.read_text().strip().splitlines()
        print(f"  {log.name}: {lines[-1] if lines else '(empty)'}")
    return 0 if s["frames_done"] == s["frames_total"] else 3


def cmd_image(a: argparse.Namespace) -> int:
    config = load_config()
    if a.action == "build":
        g = git_state()
        tag = f"{config['image']}:{g['sha']}{'-dirty' if g['dirty'] else ''}"
        run([*docker(), "build", "-f", "infra/docker/Dockerfile",
             "--build-arg", f"GIT_SHA={g['sha']}", "--build-arg", f"GIT_DIRTY={str(g['dirty']).lower()}",
             "--label", f"org.synthrig.git_sha={g['sha']}",
             "--label", f"org.synthrig.git_dirty={str(g['dirty']).lower()}",
             "-t", tag, "-t", image_ref(config), "."])
        info = image_info(config)
        print(f"Built {tag} = {image_ref(config)} ({info['id'] if info else '?'})")
        return 0
    # test: unit tests, then a 1-frame CPU render at scale 0.125 and its preview
    run(container(config, ["pytest", "-q", "-p", "no:cacheprovider"], gpu=False))
    if a.no_render:
        return 0
    out = "diagnostics/image_test"
    run(container(config, ["python", "run.py", "configs/runs/app1_demo.yaml", "--frames", "1", "--scale",
                           "0.125", "--samples", "4", "--device", "CPU", "--out", out], gpu=False))
    run(container(config, ["python", "eval/preview.py", f"{CONTAINER_DATA}/{out}", "--frame", "0",
                           "--out", f"{CONTAINER_DATA}/previews/image_test.png"], gpu=False))
    print(f"image test passed; preview: {config['data_root']}/previews/image_test.png")
    return 0


def cmd_gpucheck(a: argparse.Namespace) -> int:
    config = load_config()
    args = ["python", "infra/gpucheck.py"] + (["--skip-cpu"] if a.skip_cpu else [])
    return run(container(config, args), check=False).returncode


def cmd_exec(a: argparse.Namespace) -> int:
    config = load_config()
    command = [c for c in a.command if c != "--"] or ["bash"]
    return run(container(config, command, gpu=not a.no_gpu, mount_code=a.mount_code,
                         interactive=sys.stdin.isatty()), check=False).returncode


def cmd_submit(a: argparse.Namespace) -> int:
    config = load_config()
    g = git_state()
    info = image_info(config)
    if info is None:
        raise CloudError("no image; run: python3 infra/cloud.py image build")
    reasons = []
    if g["dirty"]:
        reasons.append("the working tree has uncommitted changes")
    if info["git_dirty"] != "false" or info["git_sha"] != g["sha"]:
        reasons.append(f"the image was built from {info['git_sha']} (dirty={info['git_dirty']}), not clean {g['sha']}")
    if reasons and not a.allow_dirty:
        raise CloudError("refusing to submit: " + "; ".join(reasons) +
                         ". Commit and rebuild the image, or pass --allow-dirty (recorded in the manifest).")
    if a.shards < 1 or a.frames < 1:
        raise CloudError("--frames and --shards must be >= 1")
    if a.shards > config["max_parallel_shards"]:
        print(f"warning: {a.shards} shards share one GPU (max_parallel_shards={config['max_parallel_shards']})")
    disk.check_free(config["data_root"], config["min_free_gb"])

    cfg = repo_relative(a.config)
    described = run(container(config, ["python", "infra/worker.py", "--describe", "--config", cfg], gpu=False),
                    capture=True)
    meta = json.loads(described.stdout.strip().splitlines()[-1])
    run_id = f"{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{info['git_sha']}"
    rel = f"datasets/{meta['scene']}/{meta['rig_hash']}/{a.run_id or run_id}"
    if (Path(config["data_root"]) / rel / "manifest.json").exists():
        raise CloudError(f"{rel} already exists; pick another --run-id")
    overrides = []
    for flag, value in (("--scale", a.scale), ("--samples", a.samples), ("--device", a.device)):
        if value is not None:
            overrides += [flag, str(value)]
    env = {"SYNTHRIG_ALLOW_DIRTY": str(bool(reasons)).lower()}
    run(container(config, ["python", "run.py", cfg, "--header-only", "--frames", str(a.frames), "--out", rel,
                           *overrides], gpu=a.device != "CPU", env=env))
    spec = {"config": cfg, "frames": a.frames, "shards": a.shards, "overrides": overrides,
            "gpu": a.device != "CPU", "env": env, "image_id": info["id"]}
    (Path(config["data_root"]) / rel / "submit.json").write_text(json.dumps(spec, indent=2) + "\n")
    names = start_workers(config, rel, spec)
    print(f"\nrun dir: {config['data_root']}/{rel}\nworkers: {', '.join(names)}")
    print(f"progress: python3 infra/cloud.py status --run {rel}\nstop:     python3 infra/cloud.py stop {rel}")
    return 0


def start_workers(config: Dict[str, Any], rel: str, spec: Dict[str, Any]) -> List[str]:
    """Start one detached worker container per shard of a submitted run."""
    names = []
    for k in range(spec["shards"]):
        name = f"synthrig-{Path(rel).name.lower()}-s{k}"
        run(container(config, ["--config", spec["config"], "--run-dir", rel, "--shard-index", str(k),
                               "--num-shards", str(spec["shards"]), "--frames-total", str(spec["frames"]),
                               *spec["overrides"]],
                      gpu=spec["gpu"], name=name, labels={RUN_LABEL: rel, "synthrig.shard": str(k)},
                      env=spec["env"], detach=True))
        names.append(name)
    return names


def _remove_workers(rel: str, stop: bool) -> int:
    ids = run([*docker(), "ps", "-aq", "--filter", f"label={RUN_LABEL}={rel}"], capture=True).stdout.split()
    if ids and stop:
        run([*docker(), "stop", "-t", "60", *ids])
    if ids:
        run([*docker(), "rm", "-f" if not stop else "-v", *ids])
    return len(ids)


def cmd_resume(a: argparse.Namespace) -> int:
    """Restart every shard of a run; workers skip frames that already have _DONE."""
    config = load_config()
    run_dir = Path(config["data_root"]) / a.run
    spec_file = run_dir / "submit.json"
    if not spec_file.is_file():
        raise CloudError(f"{spec_file} not found (only runs started by `submit` can be resumed)")
    spec = json.loads(spec_file.read_text())
    info = image_info(config)
    if info is None or info["id"] != spec["image_id"]:
        print("warning: synthrig:latest is not the image this run started with; new frames will use the "
              "current image (generator_env in manifest.json describes the original)")
    s = run_status(run_dir)
    print(f"{s['frames_done']}/{s['frames_total']} frames done; restarting {spec['shards']} shard(s)")
    _remove_workers(a.run, stop=False)
    start_workers(config, a.run, spec)
    return 0


def cmd_stop(a: argparse.Namespace) -> int:
    if not _remove_workers(a.run, stop=True):
        print("no worker containers for that run")
        return 0
    print(f"stopped; finished frames keep _DONE. Resume with: python3 infra/cloud.py resume {a.run}")
    return 0


def cmd_local_check(a: argparse.Namespace) -> int:
    from infra.diagnostics.local import check

    check(gpu=a.gpu, render=a.render)
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("config", help="show the effective configuration").set_defaults(func=cmd_config)
    d = sub.add_parser("disk", help="free space on data/scratch/docker volumes and dataset sizes")
    d.add_argument("--json", action="store_true")
    d.set_defaults(func=cmd_disk)
    s = sub.add_parser("status", help="GPU, image, disk space and workers; or one run with --run")
    s.add_argument("--run", help="run dir (relative to the data root, or absolute)")
    s.set_defaults(func=cmd_status)
    i = sub.add_parser("image", help="build or test the Docker image")
    i.add_argument("action", choices=["build", "test"])
    i.add_argument("--no-render", action="store_true", help="test: unit tests only")
    i.set_defaults(func=cmd_image)
    gc = sub.add_parser("gpucheck", help="Cycles GPU devices + GPU vs CPU render time, in the container")
    gc.add_argument("--skip-cpu", action="store_true")
    gc.set_defaults(func=cmd_gpucheck)
    e = sub.add_parser("exec", help="run a command in the container (default: bash)")
    e.add_argument("--no-gpu", action="store_true")
    e.add_argument("--mount-code", action="store_true", help="use the working tree instead of the image's code")
    e.add_argument("command", nargs=argparse.REMAINDER)
    e.set_defaults(func=cmd_exec)
    sm = sub.add_parser("submit", help="write a run header and start one worker container per shard")
    sm.add_argument("--config", required=True)
    sm.add_argument("--frames", type=int, required=True)
    sm.add_argument("--shards", type=int, default=1)
    sm.add_argument("--scale", type=float)
    sm.add_argument("--samples", type=int)
    sm.add_argument("--device", choices=["GPU", "OPTIX", "CUDA", "CPU"], default="GPU")
    sm.add_argument("--run-id")
    sm.add_argument("--allow-dirty", action="store_true")
    sm.set_defaults(func=cmd_submit)
    rs = sub.add_parser("resume", help="restart a run's workers; only missing frames are rendered")
    rs.add_argument("run", help="run dir relative to the data root")
    rs.set_defaults(func=cmd_resume)
    st = sub.add_parser("stop", help="stop and remove a run's worker containers")
    st.add_argument("run")
    st.set_defaults(func=cmd_stop)
    lc = sub.add_parser("local-check", help="check a uv workstation install (no Docker)")
    lc.add_argument("--gpu", action="store_true")
    lc.add_argument("--render", action="store_true")
    lc.set_defaults(func=cmd_local_check)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parser().parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except (CloudError, OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
