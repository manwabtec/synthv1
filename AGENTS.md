# Rules for coding agents working in this repo

This repo renders synthetic images for a FIXED physical rig: 4 cameras (a stereo pair plus 2 side cameras) and 2 dot-pattern lasers. The rig never changes between scenes. Only the world in front of it changes. Read `README.md` before changing anything.

## Architecture invariants (do not break)

1. **Layering.** `synthrig/rig`, `synthrig/sensor`, `synthrig/stereo.py`, `synthrig/dataset` and `eval/` must NEVER import `bpy`. Only `synthrig/render/`, `synthrig/assets/` and `scenes/` may import it. There is a test for this. Keep it passing.
2. **The rig is data.** Cameras, lasers, lights and sensor settings live only in `configs/rigs/*.yaml`, validated by `synthrig/rig/schema.py`. Never hard-code a camera or laser parameter in a scene or asset.
3. **Scenes never touch** cameras, lasers, the sensor model, calibration, or file I/O. A scene only builds geometry and materials, registers objects with `ctx.add(obj, cls)`, and sets the rig pose.
4. **Conventions.**
   - Device frames are OpenCV-style (x right, y down, z forward).
   - The OpenCV → Blender conversion lives ONLY in `CV_TO_BLENDER_OBJ` / `BlenderBackend._place`.
   - The world is Z-up, in metres.
   - The track frame is defined in `docs/TASK_scenes.md` §1. Use it everywhere.
   - Naming: `T_a_b` maps points from frame b to frame a.
5. **Determinism.**
   - All randomness goes in `sample_params` (or `sample_sequence`), drawn from the passed-in `rng`.
   - `build()` is a pure function of its params.
   - Frame `i` must be reproducible from `(seed, i)`.
   - No `random`, no `np.random` globals, no time-based seeds.
6. **Dataset format** (`synthrig/dataset/__init__.py`) is the contract with evaluation. Additive changes (new optional files or meta keys) are fine. Renaming or removing anything requires bumping `FORMAT_VERSION` and updating the reader.
7. **Ground truth must stay exact.**
   - Do not enable denoising.
   - Do not give laser lights a radius.
   - Do not blur or resample GT maps with anything other than nearest-neighbour.
8. **Don't rewrite working core code.** That covers `blender_backend.py`, `camera.py`, `laser.py`, `stereo.py` and `sensor/`. Extend them only when a task needs it, and explain why in the PR.

## Order of work

1. `docs/TASK_infra.md` (Phase 0: AWS environment, meaning Docker + ECR + S3 + EC2 GPU + Batch). Do this first.
2. `docs/TASK_scenes.md` (Phases 1-4: asset library and the four railway scenes).

## Infrastructure rules

- All AWS code lives in `infra/`. `synthrig/` must stay cloud-agnostic: no boto3, no S3 paths.
- Account-specific values live only in `infra/config.yaml`. Never commit credentials. Use IAM roles and SSO only.
- Every AWS resource is created by Terraform, is tagged, and can be destroyed with one command.
- Never run `terraform apply`, `destroy`, `submit`, or anything that starts paid resources without the user explicitly saying so in that request. `plan` and `image test` are always fine.

## Workflow

- Work in the phases of `docs/TASK_infra.md`, then `docs/TASK_scenes.md`, one phase per PR or commit. `python -m pytest -q` must pass after every phase.
- After Phase 0, any render or test can also be run in the container: `python infra/cloud.py image test`.
- For every new or changed scene, render a preview and look at it:
  ```
  python run.py configs/runs/<scene>.yaml --frames 2 --scale 0.25 --samples 16 --out out/<scene>
  python eval/preview.py out/<scene> --out out/<scene>_preview.png
  ```
  Commit the preview to `docs/previews/` only when asked.
- Before inventing real-world dimensions (wheel and rail profiles, car lengths, truck wheelbase), check `assets/profiles/` for a CSV. If none exists, use a clearly labelled placeholder and list it in the PR description. Do not present made-up numbers as standards.
- Keep functions small and typed. Put parameter models in pydantic with defaults. Write docstrings that state units and frames.
