# Task (Phase 0, do this FIRST): single EC2 GPU render environment

Goal: one Docker image, with Blender, that runs the tests and renders datasets on the
project's existing EC2 GPU instance. Datasets are stored on the instance's EBS data
volume. This must work before any scene work starts (`docs/TASK_scenes.md`), so every
later phase can be tested at scale.

**Scope change (2026-11-10).** The original plan used S3, Terraform, ECR and AWS Batch.
No S3 bucket is available, and the organisation's IAM policy restricts creating
infrastructure. Phase 0 is therefore scoped to the one existing instance:

- No S3, no Terraform, no ECR, no Batch, no new AWS resources.
- Scale-out across instances is out of scope. The worker's shard contract is kept, so a
  shared filesystem (e.g. EFS) plus Batch could be added later without changing it.
- The same image and `cloud.py` also run on a local NVIDIA workstation (Linux/WSL2 +
  NVIDIA Container Toolkit). The code defaults are the workstation ones (data in
  `../data`); EC2 paths come from `infra/config.yaml`, which `setup_host.sh` writes.

Follow `AGENTS.md`. Each step ends with its acceptance check passing.

---

## Design decisions

| Topic | Decision |
|---|---|
| Host | The existing instance `i-0850e2a48e8724f0a`: a g5.xlarge with an NVIDIA A10G 24 GB, Amazon Linux 2023 and NVIDIA driver 580, reached through SSM Session Manager or VS Code. |
| Storage | **EBS** volume at `/app`. Datasets, previews, logs and diagnostics go in `/app/synthrig`. Docker images go in `/app/docker`, because the 30 GB root disk is too small. |
| Scratch | The **instance-store NVMe** (233 GB) mounted at `/scratch`. It is fast, but erased on every stop/start: `infra/ec2/setup_host.sh` re-creates it. Workers render there, then move finished frames to EBS. |
| Runtime | **One Docker image** (`infra/docker/Dockerfile`), used for tests, GPU checks and rendering. It is built locally on the instance, with no registry. |
| Blender | Pinned to **4.2 LTS** via the `bpy==4.2.0` wheel (option A). Option B, the official tarball, is not needed: the GPU check passes with OptiX. The Blender version, image ID and git commit are written into every dataset's `manifest.json` (`generator_env`). |
| Host tooling | `infra/cloud.py` runs with the system `python3` (3.9) and PyYAML only. Everything that needs Blender runs in the container. The user is not in the `docker` group, so `cloud.py` calls `sudo -n docker`. |
| Credentials | None in the repo, the image or configs. AWS access from a laptop uses Okta → SSO (`docs/ONGOING_WORK.txt`). |
| Cloud awareness | `synthrig/` stays cloud-agnostic. `infra/worker.py` wraps the `Pipeline`. |

## Configuration: `infra/config.yaml` (optional)

Machine-neutral defaults live in `infra/cloud.py` (`DEFAULTS`): the data root is
`SYNTHRIG_DATA_ROOT` or `../data`, and scratch is `<data_root>/scratch`. On EC2,
`setup_host.sh` writes `infra/config.yaml` with `/app/synthrig`, `/scratch` and
`/app/docker`. Copy `infra/config.example.yaml` to override them: `data_root`, `scratch_dir`, `docker_root`, `image`, `min_free_gb` and
`max_parallel_shards`. Unknown keys and `<placeholders>` are rejected.

## Run directory layout (on EBS)

```
/app/synthrig/datasets/<scene>/<rig_hash>/<run_id>/
    manifest.json  rig.yaml  calib/...      # written once by `submit`, before workers start
    submit.json                              # submit parameters, used by `resume`
    frames/<NNNNNN>/...                      # moved in from /scratch by workers, one frame at a time
    frames/<NNNNNN>/_DONE                    # written LAST for each frame => resumable, idempotent
    logs/shard_<k>.log
```
`run_id` is the UTC timestamp plus the short git sha, unless `--run-id` is given. A frame
counts as done only if it has `_DONE`. Workers skip done frames.

---

## Step 0.0: host setup ✅

`bash infra/ec2/setup_host.sh` (needs sudo; idempotent; run again after every stop/start)
does the following:
- Creates `/app/synthrig`, owned by the user.
- Formats the instance-store NVMe if it is blank and mounts it at `/scratch`. It never
  formats EBS.
- Registers the NVIDIA container runtime with Docker and moves Docker's data-root to
  `/app/docker`. The Docker daemon restarts only if something changed.

**Accept:** `cloud.py status` shows the GPU and both volumes, and a CUDA container can see
the GPU.

## Step 0.1: container ✅

The `infra/docker/Dockerfile` image has:
- A CUDA 12.4 *base* image on Ubuntu 22.04. The host driver provides CUDA and OptiX at
  run time.
- A multi-stage build: uv installs Python 3.11 and the locked dependencies (including
  `bpy==4.2.0` and pytest) into `/opt/venv`.
- No build toolchain in the final layer, only the headless X/GL libraries `bpy` needs.
- Entrypoint `infra/docker/entrypoint.sh`: `docker run <image> pytest` works; any other
  arguments go to `infra/worker.py`.

```
python3 infra/cloud.py image build    # tags synthrig:<sha>[-dirty] and synthrig:latest, git labels
python3 infra/cloud.py image test     # pytest in the image + 1-frame CPU render at scale 0.125 + preview
```
**Accept:** `image test` passes, and the preview (`/app/synthrig/previews/image_test.png`)
shows all cameras, the dots, the rectified pair and the disparity.

## Step 0.2: infrastructure as code (dropped)

There are no new AWS resources, so Terraform is not needed. The instance, its EBS volume
and its IAM role are managed by the cloud team.

## Step 0.3: GPU check ✅

`python3 infra/cloud.py gpucheck` runs `infra/gpucheck.py` in the container with
`--gpus all`. It:
1. Prints `nvidia-smi` and Blender's Cycles device list.
2. Enables OptiX, falling back to CUDA.
3. Renders the demo frame at scale 0.5 with 16 samples, first on GPU and then on CPU,
   and prints both times.
4. Fails unless a GPU device is enabled and the GPU render is faster.

It writes `diagnostics/gpucheck/result.json`. `--device OPTIX|CUDA|GPU|CPU` is accepted
by `python -m synthrig`, the worker and `submit`.

**Result (2026-11-10, g5.xlarge):** OptiX on the A10G took 8.7 s; CPU took 55.2 s (6.3×).
This step found a bug: `BlenderBackend.new_frame` called `read_factory_settings`, which
reset the Cycles device, so every frame had been rendering on CPU. It is fixed by
re-applying the GPU selection after the reset.

Idle auto-stop (a CloudWatch alarm) is dropped from this scope: it needs new AWS
resources.

## Step 0.4: worker + shard contract ✅

```
worker.py --config configs/runs/<x>.yaml --run-dir datasets/<scene>/<rig_hash>/<run_id>
          --shard-index K --num-shards N --frames-total F
          [--scale s] [--samples n] [--device OPTIX|CUDA|GPU|CPU] [--scratch /scratch] [--min-free-gb G]
```
- **Frame range per shard:** contiguous blocks, `start = K * ceil(F/N)`.
- **For each frame:**
  1. Skip it if `_DONE` exists.
  2. Check free space on the run dir's filesystem (`--min-free-gb`).
  3. Render it into scratch with `Pipeline.render_frame(idx)`.
  4. Move it into the run dir under a temporary name and rename it into place.
  5. Write `_DONE`.
- **SIGTERM/SIGINT** (`docker stop`, `cloud.py stop`) abandons the current frame,
  cleans scratch and exits 143. `bpy` replaces Python's signal handlers on import, so
  the handlers are installed after the `Pipeline` exists. A stop flag checked between
  frames backs them up.
- **Logs** go to stdout (`docker logs`) and to `logs/shard_<k>.log`.
- **Changes to `synthrig`** (additive):
  - `Pipeline(write_header=False)` and a public `Pipeline.write_header()`.
  - `python -m synthrig --header-only`.
  - A `generator_env` block in `manifest.json`: Blender and Python versions, git sha and
    dirty flag, `allow_dirty`, image ID, render device, and instance type (from IMDSv2,
    or `local`).
  - `created` stays in the manifest, which keeps the format additive. Only `submit`
    writes the header; workers refuse to run without it, or if the rig hash or render
    scale differ.

**Accept (2026-11-10):**
- A 2-shard × 4-frame `submit` produces the same frame set as a single `python -m synthrig` run.
  `meta.json` (minus timings), `dots.npz`, depth, segmentation, disparity and masks are
  byte-identical. Radiance differs by OptiX float noise (≤ 4e-4, occasionally 1 DN in an
  image).
- A 30-frame worker stopped after 3 frames exits 143 in 0.4 s and leaves no scratch
  behind. `resume` then rendered only the 27 missing frames.

## Step 0.5: runs on the instance ✅ (replaces AWS Batch)

```
python3 infra/cloud.py submit --config configs/runs/app1_demo.yaml --frames 1000 --shards 2 [--scale ..] [--samples ..]
      # checks the git tree + image are clean and disk space is enough, writes the header,
      # starts one detached worker container per shard (restart on-failure up to 3x)
python3 infra/cloud.py status --run <run dir>   # frames with _DONE, s/frame, render-hours, size, ETA, worker states
python3 infra/cloud.py stop <run dir>           # graceful stop; finished frames are kept
python3 infra/cloud.py resume <run dir>         # restart all shards; only missing frames render
```
All shards share the one GPU. `max_parallel_shards` (default 2) only triggers a warning.
Instance-hour cost is not computed; `status --run` reports render-hours from frame
timings.

## Step 0.6: docs + guardrails ✅

- **Docs:** `infra/README.md` covers setup, day-to-day commands, resuming, disk space and
  cleanup.
- **Guardrails:**
  - `submit` refuses a dirty working tree, or an image that was not built from the
    current clean commit. `--allow-dirty` overrides this and is recorded in
    `generator_env`.
  - `submit` and every worker frame refuse to run below `min_free_gb`.
- **Status check:** `cloud.py status` reports the code, GPU, image (and whether it is
  stale), disk space and dataset sizes, and running workers. `cloud.py disk` reports only
  the volumes. `infra/ec2/check_ec2.sh` runs both, together with image build/test,
  gpucheck and a demo render.
- **CI:** `.github/workflows/ci.yml` builds the image and runs the tests and a CPU smoke
  render on every PR. It uses no secrets and pushes nothing.

## Out of scope
- S3, Terraform, ECR, AWS Batch, multi-instance scale-out, and idle auto-stop.
- Kubernetes/EKS, SageMaker, a web UI, multi-region.
- Deleting datasets from `cloud.py`. Deletes are manual on purpose.
