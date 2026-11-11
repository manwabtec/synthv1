# synthrig: synthetic images for a fixed multi-camera + laser-dot rig

New here? Start with [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md). Task specifications
and retained previews are listed in the [documentation index](docs/README.md).

A 4-camera rig (cam0 and cam1 form a stereo pair) with two configurable dot-pattern lasers (red and green), rendered with Blender Cycles. The rig is defined entirely in a YAML file. Each application is a separate scene plugin. Evaluation reads only the dataset on disk.

```
rig.yaml ─┐
          ├─► render (Blender) ─► sensor model ─► rectify + GT ─► dataset/ ─► eval/
scene ────┘   pinhole canvas      noise, bloom,    disparity,       (format      (never imports
              + light groups      Bayer, bits      dots 3D/2D        v1)          the simulator)
```

| Package | Knows about | Imports bpy? |
|---|---|---|
| `synthrig/opticsrig` | schema, camera/laser models, transforms, calibration export | no |
| `synthrig/render` | Blender backend, the scene API, the 3D overview renderer | **yes** |
| `synthrig/assets` | profiles, class taxonomy, track-frame wheel/rail geometry | no |
| `synthrig/assets/blender` | Blender builders and materials for the assets | **yes** |
| `synthrig/layout.py`, `synthrig/sampling.py` | layout geometry reference (numpy), scene-config sampling | no |
| `synthrig/sensor` | post-render camera model | no |
| `synthrig/stereo.py` | rectification, exact GT disparity | no |
| `synthrig/dataset` | on-disk format: writer helpers and reader | no |
| `scenes/<name>` | one scene each (`wheel_rail`, `debug_clutter`, ...): geometry, randomisation | yes |
| `eval/` | preview, dot detection/triangulation, dense stereo | no |

The rig, sensor, dataset and eval packages are renderer-agnostic. If you later move to Mitsuba or Isaac, only `synthrig/render` and the scenes change.

## Quickstart with Docker (EC2 GPU instance or local NVIDIA machine)

The same image runs on the project's EC2 g5.xlarge (A10G) and on a local NVIDIA machine
(Linux or WSL2, with the NVIDIA Container Toolkit). On EC2, datasets live on its EBS
volume at `/app/synthrig`; locally they go to `../data` by default. There is no S3.
Details: [`infra/README.md`](infra/README.md).

```bash
bash infra/ec2/setup_host.sh              # EC2 only: once, and after every stop/start
python3 infra/cloud.py image build        # Python 3.11 + Blender 4.2 image
python3 infra/cloud.py image test         # tests + CPU smoke render + preview
python3 infra/cloud.py gpucheck           # OptiX vs CPU timing
python3 infra/cloud.py status             # GPU, image, disk space left, running workers
python3 infra/cloud.py submit --config configs/debug_clutter/run.yaml --frames 100 --shards 2
```

## Before merging a branch: run the check script

There is no CI. Before merging a branch (or after pulling one), run the check on the
machine you are using, either the EC2 instance or a local NVIDIA machine:

```bash
bash infra/check.sh            # full check, about 2 min after the first image build
bash infra/check.sh --quick    # skip the GPU-vs-CPU and demo renders
```

It runs these steps: system info, **disk space**, GPU driver, GPU inside Docker, image
build, `image test` (all unit tests + a CPU smoke render + preview), `gpucheck` (OptiX vs
CPU timing), a 2-frame GPU demo render with preview, and `cloud.py status`. Each step
prints PASS/FAIL/SKIP and the script exits non-zero if any step fails. Logs go to
`<data_root>/logs/check/<time>_<commit>/`; read `summary.txt` first.

## GUI: layout editor, renders, 3D view

```bash
python3 infra/cloud.py gui      # then open http://localhost:8501 (VS Code forwards it from EC2)
```

A browser app running in the Docker image (GPU included), for `wheel_rail`-type runs:

- **Layout**: a WebGL viewer (`tools/gui/layout_view/`) with sliders on the left for each
  stereo pair (rod centre, rod length = baseline, shared aim point, roll, field of view),
  single cameras, lasers (position, aim, dot grid or line, fan angles, power) and the wheel
  (profile, position along the rail, rotation, lateral shift; fixed or a random range).
  Aim points can be set as "height above the rail + position across the profile". The 3D
  view, a live view from every camera and the profile-coverage inset (which part of the
  measured wheel profile each pair sees, with wheel/rail occlusion) update while you drag:
  all geometry runs in the browser (exact ray casting against the revolved wheel profile and
  the extruded rails, the same shapes Blender builds). When you release a slider, valid
  edits are written into the rig and scene YAML files with comments kept (toggle at the top).
  "Render this view in Blender" path-traces the current viewpoint.
- **Render & images**: render N frames of the layout exactly as it is in the GUI (a snapshot
  of the rig and scene goes to `<data_root>/gui/<scene>/configs/`), watch progress, then
  browse each camera's image, laser masks, depth and segmentation.
- **3D view**: a Blender render of the whole scene from a free viewpoint with markers for
  the cameras and lasers, showing where the laser line and dots actually land.

The viewer loads three.js from cdn.jsdelivr.net, so the browser needs internet access.
Renders and 3D views are stored under `<data_root>/gui/<run>/`.

## Quickstart on a workstation (uv)

Install [uv](https://docs.astral.sh/uv/getting-started/installation/). Run commands
from the `code/` repository directory. A lightweight PC needs only the standard
dependencies; a rendering workstation also needs Blender's Python module and an
NVIDIA driver for GPU rendering.

```bash
uv sync
uv run pytest -q tests                            # Blender rendering test skips without bpy
```

On the rendering workstation:

```bash
uv sync --extra render
uv run python infra/cloud.py local-check --gpu --render
uv run python -m synthrig configs/debug_clutter/run.yaml --device GPU --frames 2 --scale 0.25 --samples 16
```

Generated outputs default to `../data/`, outside Git. Relative `output` and
`--out` paths are relative to this data root, not the shell directory. Override
it with `SYNTHRIG_DATA_ROOT` or `--data-root`; relative data-root overrides are
resolved from `code/`. Absolute output paths remain supported.

After rendering or downloading a dataset, run evaluation locally:

```bash
uv run python eval/preview.py ../data/datasets/debug_clutter --frame 0 --out ../data/previews/debug_clutter.png
uv run python eval/dots_eval.py ../data/datasets/debug_clutter
uv run python eval/stereo_eval.py ../data/datasets/debug_clutter
```

The Docker image installs Blender 4.2 and the `render` dependency group from `uv.lock`
(see [`docs/TASK_infra.md`](docs/TASK_infra.md)). Commit `uv.lock`; do not commit `.venv/`.

## Repository and data layout

```text
project/                  Open this parent directory to see code and data together
    code/                   Git repository
        README.md, LICENSE, AGENTS.md (agent rules), pyproject.toml, uv.lock   <- only files in the root
        synthrig/             The simulator package: opticsrig/, sensor/, render/, dataset/,
                              stereo.py, pipeline.py; `python -m synthrig` is the render CLI
        scenes/               One plugin per scene (bpy); never define hardware here
        configs/              <scene>/{rig.yaml, scene.yaml, run.yaml}; sweeps/
        assets/profiles/      wheels/*.csv, rails/*.csv (source data, not generated outputs)
        eval/                 Dataset evaluation and image previews (never imports bpy)
        tools/                config.py (validated edits, sweeps), gui/ (browser app; layout_view/ = WebGL viewer)
        infra/                Docker image, cloud.py, worker, gpucheck, disk report, ec2/ scripts
        tests/                Automated correctness tests
        docs/                 Task specs, status/, datasheets/, previews/
        .github/              Copilot instructions
        .claude/              Claude Code instructions (imports AGENTS.md)
        .secrets/             Git credentials; ignored by git and Docker (see its README)
    data/                   Local outputs (on EC2: /app/synthrig on EBS); never committed
        datasets/  previews/  diagnostics/  logs/  cache/  scratch/
```

`synthrig/dataset/` is format/read/write code, not generated data. Hardware values
live only in `configs/<scene>/rig.yaml`; datasheet references should distinguish
specified, measured, derived and assumed values. Existing demo values are not
manufacturer specifications. Each scene folder holds its rig (camera/laser layout), its
scene parameters (fixed values or sampling ranges) and its run settings.

## Configuration and viewing

```bash
uv run python -m tools.config validate configs/debug_clutter/run.yaml
uv run python -m tools.config set configs/debug_clutter/run.yaml render.samples 16
uv run python -m tools.config validate configs/debug_clutter/rig.yaml --kind rig
uv run python -m tools.config sweep configs/debug_clutter/run.yaml configs/sweeps/ambient_sweep.yaml --out configs/sweeps/ambient_sweep_cases
uv run python -m synthrig.render.viewer configs/debug_clutter/run.yaml --frame 0 --out scenes/debug_clutter.blend --open
```

The viewer exports a `.blend` using the renderer's exact scene assembly path;
`--open` requires the Blender desktop executable on PATH. Re-run after editing
configuration to rebuild the scene. This is not a live configuration GUI.
Sweeps enumerate explicit `scene_params.*` and `seed` values deterministically;
they do not change the physical rig or add hidden randomness. Existing case
files are never overwritten. Dotted config edits address mappings, not list
indices; edit camera/laser lists directly in YAML and validate afterward.

## Storage and disk space

On EC2, all data stays on the instance's EBS volume (`/app/synthrig`); render scratch is
the instance-store NVMe (`/scratch`, erased on stop/start). Check how much space is left
before large runs:

```bash
python3 infra/cloud.py disk          # per-volume size/used/free, dataset sizes, frames that still fit
python3 infra/cloud.py disk --json
```

`submit` and every shard worker refuse to render when the data root has less than
`min_free_gb` (default 20 GiB) free. Datasets are never deleted by the tools.

## Coordinate frames (read this once)

* **Every device frame is OpenCV-style**: x right, y down, z forward (the optical or emission axis). This applies to cameras, laser projectors and lights.
* Device poses in `rig.yaml` are **device → rig**. By convention the rig frame is also OpenCV-style, with its origin at the stereo midpoint.
* The **world** is Blender's frame: Z up, metres. Each frame, the scene decides where the rig sits (`ctx.rig_look_at(...)` or `ctx.set_rig_pose(T_world_rig)`). Calibration is rig-relative, so moving the rig never changes calibration.
* Naming convention: `T_a_b` maps points from frame b to frame a.
* The OpenCV → Blender conversion (Blender looks down −Z with +Y up) exists in exactly one place: `CV_TO_BLENDER_OBJ` in `opticsrig/transforms.py`, used by `BlenderBackend._place`.
* Intrinsics use OpenCV's convention, where pixel centres are at integer coordinates. The backend handles the half-pixel offset and Blender's lens shift. `tests/test_render_geometry.py` proves it against `cv2.projectPoints`.

## How the key pieces work

**Lens distortion.** Blender can't render Brown-Conrady distortion. Each camera instead renders a slightly larger pinhole canvas (`CameraModel.render_plan`), sized automatically to cover every ray of the distorted image. That canvas is then remapped into the distorted image. Depth and segmentation use nearest-neighbour sampling.

**Dot lasers.** Each laser's pattern is an explicit list of ray directions. It can be a `grid`, a seeded `random` DOE-like pattern, or a `file` of angles. That list *is* the ground truth. The projector texture (a Cycles spot light whose emission samples an image through a perspective divide) is rasterised *from* the same list, using sub-pixel Gaussians. Every dot is then ray-cast to get its exact 3D hit point, surface normal, object, and per-camera pixel coordinates and visibility (occlusion and back-facing are checked).

**Laser separation.** Each laser renders into its own Cycles light group. That gives a clean per-laser contribution image and mask from a single render, with no extra passes. `radiance.exr` is saved before the sensor model runs, so you can sweep exposure, noise or laser-to-ambient ratio without re-rendering.

**Sensor model** (`sensor/`), applied after rendering: RGB response → electrons → Poisson shot noise → blooming → full-well clip → read noise → bit depth. Colour cameras are mosaicked before noise and demosaicked after.

**Stereo.** Calibration is exported as OpenCV FileStorage YAML (K, D, R, T, R1, R2, P1, P2, Q). The rectified images are produced from the *processed, distorted* images, the same way a real pipeline would. GT disparity is computed directly from the left pinhole depth, so there is no double interpolation.

**Calibration perturbation.** `calib/perturbed/` holds the same calibration files with a seeded random error of configurable size. The images are not changed. Only what the algorithm believes about the rig changes. Run `dots_eval.py --perturbed` to measure sensitivity.

## Dataset layout (format v1)

See the docstring in `synthrig/dataset/__init__.py` for the full listing. In short: each `frames/NNNNNN/` holds per-camera `image.png` (uint16 DN), `depth.exr`, `seg.png`, `laser_<name>.exr` and `laser_<name>_mask.png`. For rectified pairs it also holds `stereo_<pair>/{left,right}.png` with `disparity.exr` (unrectified pairs: `calib/pair_<pair>.yml` only), plus `dots.npz` and `meta.json` (seed, scene params, rig pose, object classes). `manifest.json` carries the rig hash, so every dataset records exactly which rig produced it. **Write real captures in the same layout (without the GT files), and the eval scripts run on them unchanged.**

## Adding an application

```python
# scenes/my_scene/__init__.py
from synthrig.render.scene_api import Scene, SceneContext, register_scene

@register_scene("my_scene")
class MyScene(Scene):
    def sample_params(self, rng):          # ALL randomness here -> logged in meta.json
        return {"angle": float(rng.uniform(0, 30))}   # (default: sample configs/my_scene/scene.yaml)
    def build(self, ctx: SceneContext, p):  # deterministic given p
        ...                                  # bpy geometry; ctx.add(obj, "class_name")
```

Then create `configs/my_scene/` with `rig.yaml`, `run.yaml` (`scene: my_scene`) and
optionally `scene.yaml`. Scenes never touch cameras, lasers, sensors or files.

## Changing the rig

Edit `configs/<scene>/rig.yaml`. The schema (`opticsrig/schema.py`) validates everything:
* `frame: track` places every device directly in the track frame (x along track, y left, z up); otherwise poses are relative to the rig and the scene places it.
* Cameras take `hfov_deg`, `focal_length_mm` or `fx`, OpenCV distortion, and a pose via `look_at`, `euler_xyz_deg` or `R`.
* Lasers take `wavelength_nm` or `color_rgb`, a pattern (`line` with `fan_deg`; `grid` with `fan_deg` or `pitch_deg`; `random`; `file`), `power_w`, `dot_sigma_mrad` and `enabled`.
* `stereo_pairs` name two cameras as a pair. `rectify: true` (default) adds rectified images, GT disparity and `calib/stereo_<pair>.yml`; `rectify: false` gives raw images plus `calib/pair_<pair>.yml` (R, T between the cameras) for your own stereo code (used for wheel_tread's strongly converging rod cameras).
* The GUI (`python3 infra/cloud.py gui`) edits the most important of these with a live layout view.
* `render_scale` scales every camera consistently, and the exported calibration matches the scaled images.

## Verified in this build

* Rendered emitter centroids match `cv2.projectPoints` to < 0.15 px (portrait, off-centre principal point, k1/k2/p1/p2 distortion, rotated camera).
* Rendered dot centroids have no bias relative to GT (< 0.01 px mean). The ~0.09 px scatter comes from measuring sub-pixel blobs on slanted surfaces.
* GT disparity agrees with the independently projected dot pairs to 0.015 px, with zero residual epipolar error.

## Known limitations / next steps

* The renderer is RGB, not spectral. `wavelength_nm` maps to an approximate colour. That's fine for red/green separation, but not for radiometric work. Laser speckle is not modelled (it would need Mitsuba or a post-process).
* `power_w` values are relative Blender watts. Calibrate brightness with `sensor.exposure` against a real capture.
* There is no motion blur or rolling shutter. Cycles supports motion blur if you need it for a moving target such as a rolling wheel.
* Square pixels only (fx = fy).
* Performance (demo rig, 4 cameras, half resolution, 16 samples): 8.7 s per frame on an A10G with OptiX and 55 s on the g5.xlarge's 4 vCPUs (`cloud.py gpucheck`). Use `render.device: GPU` (or `OPTIX`/`CUDA`) for production. Before 2026-11-10 the GPU setting was silently ignored after the first scene reset, so earlier "GPU" timings were CPU timings. Laser dots are direct light, so they are noise-free at any sample count; samples only affect ambient and diffuse noise.
