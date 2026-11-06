# synthrig: synthetic images for a fixed multi-camera + laser-dot rig

Task specifications, reference snapshots and retained previews are listed in the
[documentation index](docs/README.md).

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
| `synthrig/render` | Blender backend, the scene API, materials | **yes** |
| `synthrig/sensor` | post-render camera model | no |
| `synthrig/stereo.py` | rectification, exact GT disparity | no |
| `synthrig/dataset` | on-disk format: writer helpers and reader | no |
| `scenes/app1`, `scenes/app2` | one application each: geometry, materials, randomisation | yes |
| `eval/` | preview, dot detection/triangulation, dense stereo | no |

The rig, sensor, dataset and eval packages are renderer-agnostic. If you later move to Mitsuba or Isaac, only `synthrig/render` and the scenes change.

## Quickstart

Install [uv](https://docs.astral.sh/uv/getting-started/installation/). Run commands
from the `code/` repository directory. A lightweight PC needs only the standard
dependencies; a rendering workstation also needs Blender's Python module and an
NVIDIA driver for GPU rendering. AWS execution remains a separate Phase 0 task.

```bash
uv sync
uv run pytest -q tests                            # Blender rendering test skips without bpy
```

On the rendering workstation:

```bash
uv sync --extra render
uv run python infra/cloud.py local-check --gpu --render
uv run python run.py configs/runs/app1_demo.yaml --device GPU --frames 2 --scale 0.25 --samples 16
```

Generated outputs default to `../data/`, outside Git. Relative `output` and
`--out` paths are relative to this data root, not the shell directory. Override
it with `SYNTHRIG_DATA_ROOT` or `--data-root`; relative data-root overrides are
resolved from `code/`. Absolute output paths remain supported.

After rendering or downloading a dataset, run evaluation locally:

```bash
uv run python eval/preview.py ../data/datasets/app1_demo --frame 0 --out ../data/previews/app1.png
uv run python eval/dots_eval.py ../data/datasets/app1_demo
uv run python eval/stereo_eval.py ../data/datasets/app1_demo
```

The EC2 image installs Blender 4.2 and the `render` dependency group. The
interactive EC2 and Batch commands are defined by Phase 0 in
[`docs/TASK_infra.md`](docs/TASK_infra.md); until those steps are implemented,
there is no working remote render command. Commit `uv.lock`; do not commit `.venv/`.

## Repository and data layout

```text
project/                  Open this parent directory to see code and data together
    code/                   Git repository, including .git and this README
        infra/                access/, diagnostics/, storage/, AWS orchestration
        synthrig/             opticsrig/, sensor/, render/, dataset/, stereo.py
        scenes/               Final environment plugins; never define hardware here
        configs/              opticsrigs/, runs/, experiments/, scenes/, materials/
        assets/               profiles/, textures/ (source assets, not generated outputs)
        tools/                Validated config editing and parameter-case generation
        eval/                 Dataset evaluation and image previews
        docs/datasheets/      Hardware sources and provenance, not duplicate runtime values
        tests/                Automated correctness tests
    data/                   Local outputs and S3 downloads; never committed
        datasets/
        previews/
        scenes/
        cache/
        logs/
```

`synthrig/dataset/` is format/read/write code, not generated data. Hardware values
live only in `configs/opticsrigs/`; datasheet references should distinguish
specified, measured, derived and assumed values. Existing demo values are not
manufacturer specifications. Material/scene config directories are reserved for
future scene integration; current scene overrides remain in run `scene_params`.

## Configuration and viewing

```bash
uv run python -m tools.config validate configs/runs/app1_demo.yaml
uv run python -m tools.config set configs/runs/app1_demo.yaml render.samples 16
uv run python -m tools.config validate configs/opticsrigs/demo_4cam_2laser.yaml --kind rig
uv run python -m tools.config sweep configs/runs/app1_demo.yaml configs/experiments/ambient_sweep.yaml --out configs/runs/ambient_sweep
uv run python -m synthrig.render.viewer configs/runs/app1_demo.yaml --frame 0 --out scenes/app1.blend --open
```

The viewer exports a `.blend` using the renderer's exact scene assembly path;
`--open` requires the Blender desktop executable on PATH. Re-run after editing
configuration to rebuild the scene. This is not a live configuration GUI.
Sweeps enumerate explicit `scene_params.*` and `seed` values deterministically;
they do not change the physical rig or add hidden randomness. Existing case
files are never overwritten. Dotted config edits address mappings, not list
indices; edit camera/laser lists directly in YAML and validate afterward.

## S3 dataset transfers

Configure `infra/config.yaml` and authenticate through AWS SSO locally (or use an
IAM role with `aws_profile: null`). No credentials belong in the repository.

```bash
uv run python infra/cloud.py push datasets/app1_demo --to s3://YOUR-BUCKET/datasets/app1/run-001
uv run python infra/cloud.py pull s3://YOUR-BUCKET/datasets/app1/run-001 --out datasets/aws-app1
uv run python infra/cloud.py pull s3://YOUR-BUCKET/datasets/app1/run-001 --frames 0:10 --out datasets/aws-subset
```

Push/pull preserve format v1, verify SHA-256, skip matching files and reject
conflicting files. Frame ranges are half-open. Each snapshot includes a
`_TRANSFER.json` checksum index; `_DONE` markers are published last. Pull expects
this index, so legacy/worker uploads without it are not yet supported. Push only
finished datasets, not a directory being rendered. Use a new prefix for changed
or extended runs; these transfers create immutable snapshots, not mirror/delete
syncs. Integrity checking reads remote content and can incur S3 request/transfer
charges. These commands do not provision resources or submit render jobs.

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

See the docstring in `synthrig/dataset/__init__.py` for the full listing. In short: each `frames/NNNNNN/` holds per-camera `image.png` (uint16 DN), `depth.exr`, `seg.png`, `laser_<name>.exr` and `laser_<name>_mask.png`. It also holds `stereo_<pair>/{left,right}.png` with `disparity.exr`, plus `dots.npz` and `meta.json` (seed, scene params, rig pose, object classes). `manifest.json` carries the rig hash, so every dataset records exactly which rig produced it. **Write real captures in the same layout (without the GT files), and the eval scripts run on them unchanged.**

## Adding an application

```python
# scenes/app3/__init__.py
from synthrig.render.scene_api import Scene, SceneContext

class MyScene(Scene):
    name = "app3"
    def sample_params(self, rng):          # ALL randomness here -> logged in meta.json
        return {"angle": float(rng.uniform(0, 30))}
    def build(self, ctx: SceneContext, p):  # deterministic given p
        ...                                  # bpy geometry; ctx.add(obj, "class_name")
        ctx.rig_look_at((0, -1.5, 1.0), (0, 0, 0.3))
```

Then point a run config at `scene: scenes.app3:MyScene`. Scenes never touch cameras, lasers, sensors or files.

## Changing the rig

Edit `configs/opticsrigs/*.yaml`. The schema (`opticsrig/schema.py`) validates everything:
* Cameras take `focal_length_mm` or `fx`, OpenCV distortion, and a pose via `look_at`, `euler_xyz_deg` or `R`.
* Lasers take `wavelength_nm` or `color_rgb`, a pattern, `power_w`, `dot_sigma_mrad` and `enabled`.
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
* Performance: about 60 s per frame for 4 cameras at half resolution with 16 samples on 2 CPU cores. Use a GPU (`render.device: GPU`) for production. Laser dots are direct light, so they are noise-free at any sample count; samples only affect ambient and diffuse noise.
