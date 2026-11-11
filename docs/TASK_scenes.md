# Task: modular asset library + 4 railway scenes

Goal: replace the two demo scenes with four production scenes built from ONE shared library of railway assets:

| Scene id | What the rig sees | Frame mode |
|---|---|---|
| `wheel_rail` | a wheel (or wheelset) standing on the rail, viewed from the field side | independent |
| `wheel_tread` | close-up of tread + flange of one wheel, rolling through the view | independent (optionally sequence) |
| `train` | a whole consist passing the wayside rig | **sequence** |
| `rail_topdown` | track from above: rail heads, fasteners, ties, ballast | independent (optionally sequence) |

Prerequisite: `docs/TASK_infra.md` (Phase 0) is complete. Follow `AGENTS.md`. Do this in the phases below. Each phase ends with passing tests and preview images.

**Decisions by the project owner (2026-11-10), overriding the text below where they differ:**
- **Devices are placed in the track frame.** Rig files with `frame: track` give every
  camera and laser its own position + aim point in track coordinates; the scene never
  moves the rig. Mount files (`configs/mounts/`) are therefore not needed for track-frame
  rigs (still allowed for `frame: rig` rigs).
- **Lasers:** `line` (fan angle, line orientation) and dot patterns (`grid` with `fan_deg`
  or `pitch_deg`, `random`, `file`), chosen per laser. Cameras accept `hfov_deg`.
- **Wheel profiles** are a library: `assets/profiles/wheels/*.csv`, chosen by name in the
  scene config (`wheel_profile`). First profile: `AAR_2A_ref_wide` (tread + flange
  contour; rim/plate/hub completed with placeholders). Rail: `rail_136re_placeholder`.
- **Scene configs** hold fixed values or sampling specs (`{uniform: [lo, hi]}`,
  `{normal: [mean, sd]}`, `{choice: [...]}`), see `synthrig/sampling.py`.
- **GUI** (new Phase 5): a browser app running in the Docker image with tabs for
  (1) layout editing of the most important camera, laser and wheel parameters with a live
  3D layout view that writes back to the config YAML, (2) starting renders and viewing the
  camera images, and (3) a Blender-rendered 3D overview of the scene.

**Progress:** Phase 1 done except sequence mode. Phase 2: profiles, wheel (split into
tread/flange/rim face/plate/hub), rails, ties, tie plates, ballast. Phase 3: `wheel_rail`
scene with `configs/opticsrigs/wheel_rail_v1.yaml` (example layout).
Phase 5 (GUI): `tools/gui/app.py` (`python3 infra/cloud.py gui`) with the Layout,
Render & images and 3D view tabs; layout geometry in `synthrig/layout.py` (no bpy), 3D
overview renderer in `synthrig/render/observer.py`. Still open: sequence mode, wheelset
(both wheels + axle), the other three scenes, Phase 4 per-scene validation tests.

---

## 1. Conventions to adopt first

**Track frame = world frame for all railway scenes:**
- x: along the track, in the direction of travel. y: lateral, positive to the left. z: up. Metres.
- Origin: on the track centreline, at the top-of-rail height (so rail tops sit at z = 0 and ballast is below 0).
- Track gauge is a parameter. Default: 1.435 m standard gauge, measured between the gauge faces of the rail heads, a fixed distance below the top of rail (make that distance a parameter, with a placeholder value).
- "Field side" means outside the rails (|y| > gauge/2). "Gauge side" means inside.

**Rig mounting is data, not code.** Add `configs/mounts/*.yaml`. Each mount is a named rig pose in the track frame, plus jitter ranges:
```yaml
name: field_side_wheel
position: [0.0, -1.9, 0.30]      # metres, track frame
look_at:  [0.0, -0.75, 0.45]
world_up: [0, 0, 1]
jitter: {position_m: 0.02, rotation_deg: 0.5}
```
A scene takes a mount name as a parameter and calls `ctx.set_rig_pose(...)` from it. The same scene can then be rendered with different mountings, and the rig yaml stays untouched.

**Segmentation taxonomy:** a single table in `synthrig/assets/classes.py`, shared by every scene. Class ids are stable across scenes and written into `manifest.json`. Initial classes:
`background, wheel_tread, wheel_flange, wheel_rim_face, wheel_plate, wheel_hub, axle, rail_head, rail_web, rail_foot, tie, fastener, tie_plate, ballast, truck_frame, bolster, brake_component, car_body, coupler, ground, other`.
Part-level labels require the wheel to be built as separate objects (or material slots mapped to classes). See Phase 2.

---

## Phase 1: framework changes (no new geometry yet)

1. **Scene registry.** Add a decorator `@register_scene("wheel_rail")` in `synthrig/render/scene_api.py`. Run configs should say `scene: wheel_rail` instead of `module:Class`. Keep the `module:Class` form working.
2. **Scene parameter ranges in YAML.** Add `configs/scenes/<scene_id>.yaml`, holding every randomisation range (e.g. `rust: [0.0, 0.6]`, `wheel_rot_deg: [0, 360]`). The scene reads ranges from its config, and `sample_params` samples inside them. A scene's pydantic params model validates the file. The run config references it: `scene_config: configs/scenes/wheel_rail.yaml`.
3. **Sequence mode.**
   - Add an optional `Scene.sample_sequence(rng, n_frames) -> list[dict]`, where the default calls `sample_params` n times.
   - Add `mode: independent | sequence` to the run config. In sequence mode the pipeline draws one sequence per `sequence_id`.
   - Write `sequence_id`, `t_s` (time), and a per-frame params dict into `meta.json`.
   - The dataset change is additive only.
4. **Mount loader.** Add `synthrig/assets/mounts.py` (no bpy) that loads `configs/mounts/*.yaml` and returns `T_world_rig` with seeded jitter.
5. **Tests.**
   - (a) Import-layering test: importing `synthrig.opticsrig`, `synthrig.sensor`, `synthrig.dataset`, `synthrig.stereo` and the `eval/` modules must not load `bpy`. Check `sys.modules` in a subprocess.
   - (b) Every registered scene's params round-trip to JSON.
   - (c) Same `(seed, idx)` gives identical params.

## Phase 2: asset library `synthrig/assets/` (bpy builders)

Every builder follows the same shape:
```python
class WheelParams(BaseModel): ...          # units in field names/docstrings, sensible defaults
def build_wheel(ctx: SceneContext, p: WheelParams, T_world_obj: np.ndarray, name: str) -> AssetHandle
```
Builders return an `AssetHandle` (the root object, its parts, and key frames/points such as the tread contact point and the flange tip). They register every part with `ctx.add(obj, CLASS)`. Builders take no rng: the scene samples, the builder builds.

Modules:
- `profiles.py`: load 2D profiles from CSV (`assets/profiles/*.csv`, columns in metres, closed polygon). Ship clearly-named PLACEHOLDER profiles (`wheel_placeholder.csv`, `rail_placeholder.csv`). The real ones (e.g. the wheel profile standard and rail section we actually use) will be dropped in later by the user, so no code change should be needed.
- `wheel.py`: revolve the profile, **split into part objects** by profile segment ranges (tread, flange, rim face, plate, hub), each with its own material and class.
  - Parameters: profile file, tread wear offset (shift the tread segment inward), flange wear, diameter scaling, and surface materials (steel / rust / polished running band on the tread).
  - Keep the existing `revolve()` logic as the base.
- `wheelset.py`: two wheels plus an axle. Parameters: back-to-back distance, axle diameter. Places wheels on rails given the gauge.
- `track.py`: two rails (extrude the rail profile along x), with cant/inclination as a parameter. Ties are instanced at a spacing parameter with jitter. Fasteners are simple placeholder clips, instanced. Tie plates. A ballast bed (displaced / noise-textured mesh; keep the poly count bounded). Optional debris.
- `truck.py`: a simplified 3-piece truck (two side frames, a bolster, two wheelsets, brake beams as boxes). Wheelbase is a parameter.
- `car.py`: a car body as a parametric box/hopper/tank shape, with length, truck centres and couplers. Low detail is fine. It mostly occludes and sets the background.
- `consist.py`: a list of cars, and the `(x_offset → object transforms)` function used by the train sequence.
- `materials.py`: move `render/materials.py` here and add steel/rust/wear-band, concrete-tie, wood-tie and ballast presets. Presets take numbers, not rng.
- `environment.py`: ground, sky/ambient presets (day, overcast, night with only rig lights).

**Performance rules:** use linked duplicates (instances) for repeated parts (ties, fasteners, wheelsets, trucks). Build the whole-train geometry once per sequence, not per frame, and only move transforms per frame. Anything outside every camera's frustum by more than a margin must not be built at all. Add a helper that culls by the rig frustum.

## Phase 3: the four scenes (`scenes/<id>/__init__.py` + `configs/scenes/<id>.yaml` + `configs/runs/<id>.yaml`)

**`wheel_rail`**: a single wheelset on track (a short track segment is enough).
- Randomise: wheel rotation angle, lateral wheelset shift (flange-to-rail clearance within limits), wheel wear/rust/running band, rail wear, ambient preset, mount jitter.
- Mount: `field_side_wheel`.
- Acceptance: the wheel's full field-side face and the rail head are both visible in cam0/cam1, and both lasers land on the wheel.

**`wheel_tread`**: a wheelset on track.
- The mount looks at the tread/flange region (e.g. from the field side, below axle height, angled up toward the tread near the contact zone). Define it as `tread_view` in `configs/mounts/`. The exact viewing geometry will be tuned by the user.
- Randomise: wheel rotation (so a different tread patch is seen each frame), tread condition parameters, lateral shift.
- Acceptance: `wheel_tread` + `wheel_flange` pixels are ≥ 40% of cam0, and dots land on the tread.
- Leave a hook for later tread-defect insertion: `WheelParams.surface_modifiers: list[...]`, empty for now.

**`train`**: a sequence scene.
- A consist of N cars (N and car types from YAML), moving past a fixed wayside mount (`wayside_side`) at speed v.
- Frame k sees the consist at `x = x0 + v * k / fps`.
- Build geometry once per sequence, then move it.
- Write `t_s`, `x_offset_m`, and the id of the car/truck/wheel nearest to each camera's optical axis into `meta.json`.
- Optional flag: Cycles motion blur using `exposure_s` from the scene config (off by default, since it changes GT: document that dot GT then refers to mid-exposure).
- Acceptance: a 30-frame sequence at a low scale renders without rebuilding geometry. Timings in `meta.json` show build time only on frame 0.

**`rail_topdown`**: track only.
- Mount `topdown` above the track, looking down (-z), at a height parameter. Choose the default so the rig's working distance matches the rig yaml.
- Randomise: tie type/spacing, fastener presence (allow a missing clip as a labelled class for later), ballast level, rail wear, debris, lighting.
- Optional sequence mode: the track scrolls along x.
- Acceptance: both rail heads are visible in the stereo pair, and GT disparity is valid on ties and ballast.

Delete `scenes/app1` and `scenes/app2` only after the new scenes pass. Keep `app1` (clutter) as `scenes/debug_clutter`, because it is a useful non-railway sanity scene.

## Phase 4: tests and validation per scene

For each scene id, add a test (marked `slow`, skipped when `bpy` is missing) that renders 1 frame at `--scale 0.125 --samples 4`. It asserts:
1. Every class in `seg.png` exists in the taxonomy.
2. At least N laser dots are visible in cam0 (N set per scene).
3. The dot GT has no bias: the mean of (intensity centroid in the clean `laser_<name>.exr` minus GT uv) is < 0.05 px over isolated dots. Reuse the check logic from `eval/dots_eval.py`. Move the shared helper into `eval/common.py`.
4. The GT disparity at the rectified GT dot pairs matches `uv_left.x - uv_right.x` to < 0.05 px.
5. `meta.json` params validate against the scene's params model.

Also update `README.md`: the scene list, the track-frame convention, how to add a mount, and how to drop in real profile CSVs.

## Out of scope for this task
- Changing the rig yaml, the sensor model, or the dataset reader beyond additive fields.
- Defect generation (cracks, flats, shelling, spalling). Only leave the hooks described above.
- Spectral rendering and laser speckle.
