# Getting started: play with the wheel/rail scenes

Short, practical steps. Details are in the main [README](../README.md) and
[infra/README](../infra/README.md).

## 1. Once per machine

**On the EC2 instance** (already done there; after a stop/start, repeat the first line):

```bash
bash infra/ec2/setup_host.sh          # mounts /scratch, sets Docker up (needs sudo)
python3 infra/cloud.py image build    # ~3 min the first time
```

**On a local NVIDIA machine** (Linux or WSL2, NVIDIA driver, Docker with the NVIDIA
Container Toolkit, `python3` with PyYAML):

```bash
python3 infra/cloud.py image build
python3 infra/cloud.py gpucheck       # should say PASSED
```

## 2. Open the GUI

```bash
python3 infra/cloud.py gui
```

Open **http://localhost:8501** in your browser. On EC2 through VS Code, the port is
forwarded automatically (check the "Ports" panel if the page does not open). Ctrl+C in
the terminal stops the GUI.

In the sidebar, pick the scene: **wheel_rail** (whole wheel on the rail) or
**wheel_tread** (close-up of the tread).

## 3. Layout tab: move things and see the effect immediately

- **Wheel**: position along the rail, rotation, lateral shift. "fixed" = one value,
  "random range" = a new value every rendered frame.
- **Cameras** (open each one): position x/y/z, aim point x/y/z, field of view.
  "Aim at" snaps the aim point to the tread contact, flange tip, rail head or wheel centre.
- **Lasers**: position, aim point, pattern (line or dots), fan angle, power.

Coordinates are metres in the track frame: **x** along the track, **y** sideways
(+ = towards the left rail; the wheel is on the right rail at about y = -0.75),
**z** up (0 = top of rail).

The 3D view (drag to rotate, scroll to zoom) and the small camera views below it update
straight away. The camera views show what each camera roughly sees and the percentage of
each laser that lands in its image. Edits are saved into `configs/<scene>/rig.yaml` and
`scene.yaml` (switch "Save edits" off in the sidebar to just experiment).

## 4. Render & images tab

Set frames (start with 1), scale (0.5 is quick; 1.0 = full resolution), samples (32),
then **Render**. When it finishes, the images appear: switch "Show" between camera
images, laser masks, depth and segmentation.

## 5. 3D view tab

Choose a viewpoint (look-at point, azimuth, elevation, distance) and press
**Render 3D view** to see the whole scene rendered by Blender, with the cameras (blue)
and lasers (their colour) drawn as markers and the laser line/dots where they really land.

## 6. Undo and keep results

- Every GUI edit is a plain change to the YAML files: `git diff configs/` shows it,
  `git checkout configs/<scene>/` undoes it, `git commit` keeps it.
- Renders made from the GUI are in `<data root>/gui/<scene>/` (`/app/synthrig/gui/` on EC2).

## 7. Without the GUI

```bash
python3 infra/cloud.py exec -- python -m synthrig configs/wheel_tread/run.yaml --frames 2 --scale 0.25
python3 infra/cloud.py submit --config configs/wheel_tread/run.yaml --frames 200 --shards 2   # big runs
python3 infra/cloud.py status --run datasets/<scene>/<rig_hash>/<run_id>
python3 infra/cloud.py disk           # space left
bash infra/check.sh                   # before merging a branch
```

For a rolling wheel, set `mode: sequence` in `configs/wheel_tread/run.yaml` and choose
`speed_mps` in `configs/wheel_tread/scene.yaml`.

## Things that are placeholders

The wheel's tread and flange come from your profile CSV; the rim, plate, hub, rail shape
(136RE-like), ties and back-to-back distance are placeholders. Add more wheel profiles as
CSV files in `assets/profiles/wheels/` and they appear in the GUI's profile list.
