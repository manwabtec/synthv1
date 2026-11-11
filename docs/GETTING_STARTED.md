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
the terminal stops the GUI. The layout view loads three.js from cdn.jsdelivr.net, so the
browser needs internet access.

At the top, pick the scene: **wheel_tread** (close-up of the tread, the default) or
**wheel_rail** (whole wheel on the rail). The page is dark and full-width; settings are on
the left, the view takes the rest.

## 3. Layout tab: move things and see the effect immediately

Everything updates **while you drag** a slider (it all runs in your browser). When you
release it, the change is saved into `configs/<scene>/rig.yaml` / `scene.yaml` (the status
line at the top of the left panel says so). You can also type a value in the box next to a
slider.

Left panel, top to bottom:

- **Pair field / Pair gauge**: one stereo pair per rod (blue = outside the rail, field side;
  orange = inside, gauge side). Rod centre x/y/z, **rod length = baseline**, the shared aim
  point, roll and field of view move both cameras together. Set the aim as
  **height above the rail** (e.g. 63.5 mm = 2.5 in) and **position across the profile**
  (mm from the flange back face), or as plain x/y/z. The readout shows range to the aim,
  look-up angle, convergence angle and mm per pixel.
- **Laser ...**: on/off, position, aim (same two ways), pattern (dot grid: rows, columns,
  two fan angles; line: fan angle, orientation), power.
- **Wheel**: profile, position along the rail, rotation, lateral shift, each fixed or
  "random every frame, between ... and ...". The cameras stay where they are when the
  wheel moves (the rig is fixed in the track frame).
- **Single camera**: fine-tune one camera.
- **Display**: frustums, laser fans, labels, the camera-view strip and its size.

Positions are mm in the track frame: **x** along the track, **y** sideways (+ = towards the
left rail; the wheel is on the right rail at about y = -750; field side y < -790, gauge side
y > -720), **z** up (0 = top of rail).

In the view: drag to orbit, right-drag to pan, scroll to zoom; the buttons at the top left
jump to preset viewpoints. **Render this view in Blender** path-traces exactly that viewpoint
(see the 3D view tab after a few seconds).

- **Bottom strip**: what each camera sees (lit by the rig's strobes; lens distortion not
  drawn), with the % of each laser's points on the wheel that camera sees.
- **Inset, top right**: the wheel profile in the section the field pair aims at, drawn where
  it meets the rail. Blue/orange lines above it = seen by the outside/inside pair (thick =
  both cameras of the pair, thin = one); yellow = seen by no camera. The table gives the %
  of the profile each pair covers, the covered range in mm from the flange back, and the
  overlap between the two sides.

Laser points: bright = seen by at least one camera, dark = by none.

## 4. Render & images tab

Set frames (start with 1), scale (0.5 is quick; 1.0 = full resolution), samples (32),
then **Render**. It renders the layout exactly as it is in the GUI, saved or not. When it finishes, the images appear: switch "Show" between camera
images, laser masks, depth and segmentation.

## 5. 3D view tab

Use **Render this view in Blender** in the Layout tab, or choose a viewpoint here
(look-at point, azimuth, elevation, distance) and press **Render 3D view** to see the whole scene rendered by Blender, with the cameras (blue)
and lasers (their colour) drawn as markers and the laser line/dots where they really land.

## 6. Stereo pairs

Each rod's two cameras form a stereo pair (`stereo_pairs` in `rig.yaml`, `rectify: false`).
A rendered dataset then has the raw images of both cameras, each camera's calibration
(`calib/cam_<name>.yml`: K, distortion, pose) and `calib/pair_<pair>.yml` (R, T from left
to right camera) for your own stereo processing. No rectified images are made for them.

## 7. Undo and keep results

- Every GUI edit is a plain change to the YAML files: `git diff configs/` shows it,
  `git checkout configs/<scene>/` undoes it, `git commit` keeps it. **Reload files** (top)
  re-reads them into the GUI.
- Switch "Save edits to the YAML files" off to experiment without touching the files.
- Renders made from the GUI are in `<data root>/gui/<scene>/` (`/app/synthrig/gui/` on EC2).

## 8. Without the GUI

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
