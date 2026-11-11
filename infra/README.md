# Infrastructure: Docker on an NVIDIA machine (EC2 or local)

Phase 0 (`docs/TASK_infra.md`). The same image and the same `cloud.py` commands run on the
project's EC2 instance (a g5.xlarge with an A10G) and on a local NVIDIA workstation. Only
`infra/config.yaml` differs. There is no S3, Terraform, ECR or Batch, and nothing here
creates AWS resources.

## On the EC2 instance

| Where | What |
|---|---|
| `/app/synthrig` (EBS, 400 GB) | EC2 persistent data root: `datasets/`, `previews/`, `diagnostics/`, `logs/`, `cache/` |
| `/scratch` (instance-store NVMe) | Render scratch. **Erased on every stop/start.** |
| `/app/docker` (EBS) | Docker images. The 30 GB root disk is too small for them. |
| `synthrig:latest` | Runtime image: Python 3.11 + Blender 4.2 (`bpy`) + locked dependencies |

| File | Purpose |
|---|---|
| `cloud.py` | Host CLI (system `python3` + PyYAML): status, disk, image, gpucheck, submit/stop/resume, exec |
| `worker.py` | Container entrypoint that renders one shard into a run dir (resumable via `_DONE`) |
| `gpucheck.py` | Lists Cycles devices, enables OptiX, compares GPU and CPU render times |
| `storage/disk.py` | Free-space report and the low-space guard used by `submit` and workers |
| `docker/` | `Dockerfile` and `entrypoint.sh` |
| `ec2/setup_host.sh` | Host setup: data dir, scratch mount, Docker data-root and NVIDIA runtime |
| `check.sh` | End-to-end check, the replacement for CI: run it before merging a branch (EC2 or local) |
| `access/check.py` | From a laptop: AWS identity, EC2 describe and SSM status for the instance |
| `diagnostics/local.py` | `cloud.py local-check` for a non-Docker `uv` workstation install |

### First-time setup (on the instance)

```bash
bash infra/ec2/setup_host.sh          # sudo; also run after every stop/start (scratch is wiped);
                                      # writes infra/config.yaml with the /app + /scratch paths
python3 infra/cloud.py image build    # ~3 min; rebuild after every code change you want to run
python3 infra/cloud.py image test     # pytest in the image + 1-frame CPU render + preview
python3 infra/cloud.py gpucheck       # OptiX vs CPU render time; must PASS
```

## On a local NVIDIA machine

No separate code is needed. Requirements:
- Linux, or Windows 11 with WSL2 (run everything inside the WSL2 distro).
- An NVIDIA driver (`nvidia-smi` works).
- Docker with the NVIDIA Container Toolkit.
- `python3` (3.9+) with PyYAML.

```bash
# once: let Docker use the GPU (skip if `docker run --rm --gpus all ubuntu nvidia-smi` already works)
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
python3 infra/cloud.py status          # defaults: data in ../data, scratch in ../data/scratch
python3 infra/cloud.py image build && python3 infra/cloud.py image test && python3 infra/cloud.py gpucheck
```

All the day-to-day commands below work unchanged. To keep data on another disk, copy
`config.example.yaml` to `infra/config.yaml` and set `data_root` (and optionally
`scratch_dir`, e.g. a local NVMe). Do not run `infra/ec2/setup_host.sh` locally: it
formats the EC2 instance-store disk and moves Docker's data directory.

The non-Docker path (`uv sync --extra render`, then `cloud.py local-check --gpu --render`)
also still works, for example for debugging in an IDE. The Docker image is the reference
environment.

`infra/config.yaml` is optional off EC2 and gitignored everywhere.

## Day to day

Before merging a branch: `bash infra/check.sh` (there is no CI; see the root README).

```bash
python3 infra/cloud.py status               # code, GPU, image freshness, disk space, running workers
python3 infra/cloud.py disk                 # space left on each volume, dataset sizes, frames that still fit

git commit ...; python3 infra/cloud.py image build
python3 infra/cloud.py submit --config configs/debug_clutter/run.yaml --frames 200 --shards 2 --scale 0.5
python3 infra/cloud.py status --run datasets/<scene>/<rig_hash>/<run_id>
python3 infra/cloud.py stop   datasets/<scene>/<rig_hash>/<run_id>   # graceful; done frames are kept
python3 infra/cloud.py resume datasets/<scene>/<rig_hash>/<run_id>   # renders only the missing frames

python3 infra/cloud.py gui                       # browser GUI on http://localhost:8501
python3 infra/cloud.py exec -- python eval/dots_eval.py /data/datasets/<...>   # any command in the image
python3 infra/cloud.py exec --mount-code -- pytest -q                           # test uncommitted code
```

**Command behaviour:**
- **`submit`:**
  - Refuses a dirty tree, or an image not built from the current commit, unless you pass
    `--allow-dirty` (recorded in `manifest.json`).
  - Refuses to start if the data root has less than `min_free_gb` free.
- **Workers:**
  - Run as your user, so output files are yours.
  - Restart up to 3 times on failure.
  - Stop cleanly if free space drops below `min_free_gb`.
- **Paths in the container:** inside the image the data root is `/data` and scratch is
  `/scratch`. `python -m synthrig --out` paths are relative to `/data`.

## Disk space

`cloud.py disk` (and the storage section of `cloud.py status`) shows size, used, free and
use% for the data root, scratch, Docker root and `/`. Paths on the same filesystem share
one row. It also lists the largest datasets and estimates how many more frames fit at the
largest observed MiB/frame. It exits 1 when the data root is below `min_free_gb`.
`--json` gives machine-readable output.

With the demo rig, a frame with radiance and laser EXRs measured about 12 MiB at
`--scale 0.25`. Full resolution has 16× the pixels, so expect up to about 200 MiB.
Measure it with `cloud.py disk` after a short run. Set `outputs.save_radiance: false`
in a run config to save space.

## Cleanup

- Datasets are never deleted by these tools: `rm -rf /app/synthrig/datasets/<...>` by
  hand.
- Old images: `sudo docker image prune`.
- `setup_host.sh` keeps the pre-move `/var/lib/docker`, which can be deleted once Docker
  runs from `/app/docker`.
