"""rig.yaml + scene + N frames -> dataset.   (render -> sensor post -> stereo -> write)"""
from __future__ import annotations

import datetime as _dt
import importlib
import json
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict

from .dataset import FORMAT_VERSION, write_exr, write_png
from .paths import output_path
from .opticsrig import RigModel, load_rig
from .opticsrig.calib import perturb_camera, stereo_calib, write_camera_yaml, write_stereo_yaml
from .sensor import apply_sensor
from .stereo import gt_disparity, rectify_maps, rectify_pair, rectify_points


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RenderCfg(_Strict):
    samples: int = 64
    filter_width: float = 1.5
    max_bounces: int = 4
    device: str = "CPU"  # CPU, GPU (OptiX, else CUDA), OPTIX or CUDA
    threads: int = 0
    render_scale: float = 1.0  # <1 for fast previews: scales every camera's K and resolution


class OutputsCfg(_Strict):
    save_radiance: bool = True  # lets you re-run the sensor model without re-rendering
    save_laser_components: bool = True
    # laser mask = laser contribution >= this fraction of the camera's full well
    laser_mask_threshold: float = 0.02


class PerturbCfg(_Strict):
    enabled: bool = False
    seed: int = 0
    rotation_deg: float = 0.05
    translation_mm: float = 0.2
    focal_pct: float = 0.1
    principal_px: float = 0.5
    distortion_rel: float = 0.02


class RunConfig(_Strict):
    rig: str
    scene: str  # "module.path:ClassName"
    scene_params: dict[str, Any] = {}
    output: str = "datasets/default"
    frames: int = 1
    start: int = 0
    seed: int = 0
    render: RenderCfg = RenderCfg()
    outputs: OutputsCfg = OutputsCfg()
    calib_perturbation: PerturbCfg = PerturbCfg()

    @classmethod
    def load(cls, path) -> "RunConfig":
        path = Path(path)
        cfg = cls.model_validate(yaml.safe_load(path.read_text()))
        if not Path(cfg.rig).is_absolute():
            cfg.rig = str((path.parent / cfg.rig).resolve())
        return cfg


def load_scene(spec: str, params: dict):
    mod, cls = spec.split(":")
    return getattr(importlib.import_module(mod), cls)(**params)


def _lum(rgb: np.ndarray, sensor) -> np.ndarray:
    if sensor.mono:
        return rgb @ np.asarray(sensor.rgb_response)
    return (rgb * np.asarray(sensor.rgb_response)).max(-1)


GPU_BACKENDS = {"GPU": ("OPTIX", "CUDA"), "OPTIX": ("OPTIX",), "CUDA": ("CUDA",)}


def generator_env(device: str) -> dict[str, Any]:
    """What produced a dataset: software versions and host. Cloud-agnostic.

    Callers (e.g. infra/) set SYNTHRIG_IMAGE_DIGEST and SYNTHRIG_INSTANCE_TYPE; without
    them the values are "none" and "local".
    """
    import bpy

    sha = os.environ.get("SYNTHRIG_GIT_SHA")
    dirty = os.environ.get("SYNTHRIG_GIT_DIRTY")
    if not sha or sha == "unknown":
        root = Path(__file__).resolve().parents[1]
        try:
            sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True,
                                 text=True, check=True).stdout.strip()
            dirty = str(bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                                            cwd=root, capture_output=True, text=True).stdout.strip())).lower()
        except (OSError, subprocess.CalledProcessError):
            sha = "unknown"
    return {
        "blender_version": bpy.app.version_string,
        "python_version": platform.python_version(),
        "git_sha": sha,
        "git_dirty": dirty or "unknown",
        "allow_dirty": os.environ.get("SYNTHRIG_ALLOW_DIRTY", "false"),
        "image_digest": os.environ.get("SYNTHRIG_IMAGE_DIGEST", "none"),
        "render_device": device,
        "instance_type": os.environ.get("SYNTHRIG_INSTANCE_TYPE", "local"),
    }


class Pipeline:
    def __init__(self, cfg: RunConfig, log=print, write_header: bool = True):
        """Set up the rig and renderer for `cfg`.

        write_header=False skips writing manifest/rig/calib, for shard workers that render
        frames into a dataset whose header was written once by `write_header()`.
        """
        # bpy imported lazily so rig/sensor/dataset stay importable without Blender
        from .render import devices
        from .render.blender_backend import BlenderBackend, RenderSettings

        if cfg.render.device not in ("CPU", *GPU_BACKENDS):
            raise ValueError(f"render.device must be CPU, GPU, OPTIX or CUDA, not {cfg.render.device!r}")
        if cfg.render.device in GPU_BACKENDS:
            devices.DEFAULT_BACKENDS = GPU_BACKENDS[cfg.render.device]
        self.cfg = cfg
        self.log = log
        self.rig_spec = load_rig(cfg.rig)
        self.rig = RigModel.build(self.rig_spec, cfg.render.render_scale)
        r = cfg.render
        self.backend = BlenderBackend(RenderSettings(samples=r.samples, filter_width=r.filter_width,
                                                     max_bounces=r.max_bounces,
                                                     device="CPU" if r.device == "CPU" else "GPU",
                                                     threads=r.threads))
        self.scene = load_scene(cfg.scene, cfg.scene_params)
        self.out = output_path(cfg.output)
        t = time.time()
        self.plans = {n: c.render_plan() for n, c in self.rig.cameras.items()}
        self.rect_maps = {n: rectify_maps(s, self.rig.cameras[s.left], self.rig.cameras[s.right])
                          for n, s in self.rig.stereo.items()}
        log(f"[synthrig] rig {self.rig_spec.name} ({self.rig_spec.hash()}), scale {self.rig.scale}, "
            f"render plans in {time.time() - t:.1f}s")
        if write_header:
            self.write_header()

    # ---------------------------------------------------------------- dataset header
    def write_header(self, extra_env: dict[str, Any] | None = None):
        """Write manifest.json, rig.yaml and calib/ (once per dataset, before any frame).

        Apart from the wall-clock `created`, the header depends only on the run config and
        the software in `generator_env`, so it is identical whoever writes it.
        """
        out = self.out
        (out / "calib").mkdir(parents=True, exist_ok=True)
        (out / "frames").mkdir(exist_ok=True)
        rig = self.rig
        manifest = {
            "format_version": FORMAT_VERSION,
            "created": _dt.datetime.now().isoformat(timespec="seconds"),
            "generator": "synthrig",
            "rig_name": rig.spec.name, "rig_hash": rig.spec.hash(), "render_scale": rig.scale,
            "cameras": list(rig.cameras), "stereo_pairs": list(rig.stereo), "lasers": list(rig.lasers),
            "stereo": {n: {"left": s.left, "right": s.right} for n, s in rig.stereo.items()},
            "scene": self.cfg.scene, "run_config": self.cfg.model_dump(mode="json"),
            "units": {"length": "m", "disparity": "px", "depth": "planar z along optical axis"},
            "generator_env": {**generator_env(self.cfg.render.device), **(extra_env or {})},
        }
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
        (out / "rig.yaml").write_text(yaml.safe_dump(rig.spec.model_dump(mode="json"), sort_keys=False))
        for n, c in rig.cameras.items():
            write_camera_yaml(out / "calib" / f"cam_{n}.yml", c)
        for n, s in rig.stereo.items():
            write_stereo_yaml(out / "calib" / f"stereo_{n}.yml", s)
        pc = self.cfg.calib_perturbation
        if pc.enabled:
            pdir = out / "calib" / "perturbed"
            pdir.mkdir(exist_ok=True)
            rng = np.random.default_rng(pc.seed)
            pert = {n: perturb_camera(c, rng, pc.rotation_deg, pc.translation_mm, pc.focal_pct,
                                      pc.principal_px, pc.distortion_rel) for n, c in rig.cameras.items()}
            for n, c in pert.items():
                write_camera_yaml(pdir / f"cam_{n}.yml", c)
            for p in rig.spec.stereo_pairs:
                write_stereo_yaml(pdir / f"stereo_{p.name}.yml", stereo_calib(p, pert[p.left], pert[p.right]))
            (pdir / "perturbation.json").write_text(pc.model_dump_json(indent=2))

    # ---------------------------------------------------------------- frames
    def run(self):
        for i in range(self.cfg.start, self.cfg.start + self.cfg.frames):
            self.render_frame(i)

    def render_frame(self, idx: int):
        t0 = time.time()
        cfg, rig, be = self.cfg, self.rig, self.backend
        from .render.blender_backend import laser_group
        from .render.assembly import build_frame

        params, ctx = build_frame(be, rig, self.scene, cfg.seed, idx)

        fdir = self.out / "frames" / f"{idx:06d}"
        if fdir.exists():
            shutil.rmtree(fdir)
        fdir.mkdir(parents=True)
        timings = {}
        images, pinhole_depth = {}, {}
        for ci, (cname, cam) in enumerate(rig.cameras.items()):
            t = time.time()
            r = be.render_camera(cname, self.plans[cname])
            timings[f"render_{cname}"] = round(time.time() - t, 2)
            plan, sensor = self.plans[cname], cam.spec.sensor
            cdir = fdir / cname
            cdir.mkdir()
            radiance = plan.apply(r.combined)
            depth = plan.apply(r.depth, nearest=True)
            seg = plan.apply(r.index.astype(np.float32), nearest=True).astype(np.uint16)
            so = apply_sensor(radiance, sensor, np.random.default_rng([cfg.seed, idx, ci, 0x5E]))
            write_png(cdir / "image.png", so.image)
            if so.raw is not None:
                write_png(cdir / "raw.png", so.raw)
            if cfg.outputs.save_radiance:
                write_exr(cdir / "radiance.exr", radiance)
            write_exr(cdir / "depth.exr", depth)
            write_png(cdir / "seg.png", seg)
            for lname in rig.lasers:
                comp = plan.apply(r.groups[laser_group(lname)])
                mask = _lum(comp, sensor) * sensor.exposure >= cfg.outputs.laser_mask_threshold
                write_png(cdir / f"laser_{lname}_mask.png", mask.astype(np.uint8) * 255)
                if cfg.outputs.save_laser_components:
                    write_exr(cdir / f"laser_{lname}.exr", comp)
            images[cname] = so.image
            pinhole_depth[cname] = r.depth

        t = time.time()
        dots = be.cast_dots()
        timings["dots_gt"] = round(time.time() - t, 2)

        dz = {"laser": dots.laser, "laser_names": np.array(dots.laser_names), "dot_id": dots.dot_id,
              "hit": dots.hit, "X_world": dots.X_world, "X_rig": dots.X_rig, "normal_world": dots.normal_world,
              "object_index": dots.object_index}
        for c in rig.cameras:
            dz[f"uv_{c}"] = dots.uv[c]
            dz[f"visible_{c}"] = dots.visible[c]

        for pname, s in rig.stereo.items():
            sdir = fdir / f"stereo_{pname}"
            sdir.mkdir()
            L, R = rig.cameras[s.left], rig.cameras[s.right]
            left, right = rectify_pair(self.rect_maps[pname], images[s.left], images[s.right])
            write_png(sdir / "left.png", left)
            write_png(sdir / "right.png", right)
            disp, valid = gt_disparity(s, self.plans[s.left], pinhole_depth[s.left])
            write_exr(sdir / "disparity.exr", disp)
            write_png(sdir / "depth_valid.png", valid.astype(np.uint8) * 255)
            dz[f"uv_rect_{pname}_left"] = rectify_points(dots.uv[s.left], L, s.R1, s.P1)
            dz[f"uv_rect_{pname}_right"] = rectify_points(dots.uv[s.right], R, s.R2, s.P2)
        np.savez_compressed(fdir / "dots.npz", **dz)

        timings["total"] = round(time.time() - t0, 2)
        meta = {"frame": idx, "seed": [cfg.seed, idx], "scene": cfg.scene, "scene_params": params,
                "T_world_rig": ctx.T_world_rig.tolist(), "objects": ctx.objects, "rig_hash": rig.spec.hash(),
                "timings_s": timings}
        (fdir / "meta.json").write_text(json.dumps(meta, indent=2, default=_json_default))
        vis = {c: int(dots.visible[c].sum()) for c in rig.cameras}
        self.log(f"[synthrig] frame {idx:06d} done in {timings['total']}s  "
                 f"dots hit {int(dots.hit.sum())}/{len(dots.hit)} visible {vis}")


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))
