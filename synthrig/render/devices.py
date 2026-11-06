"""Explicit NVIDIA Cycles selection; fail instead of silently falling back to CPU."""
from __future__ import annotations

import bpy


def configure_gpu() -> list[str]:
    """Prefer OptiX, then CUDA, returning enabled NVIDIA device descriptions."""
    preferences = bpy.context.preferences.addons["cycles"].preferences
    for backend in ("OPTIX", "CUDA"):
        try:
            preferences.compute_device_type = backend
            preferences.refresh_devices()
        except (TypeError, RuntimeError):
            continue
        enabled = []
        for device in preferences.devices:
            device.use = device.type == backend
            if device.use:
                enabled.append(f"{backend}: {device.name}")
        if enabled:
            print("Cycles GPU devices: " + ", ".join(enabled))
            return enabled
    raise RuntimeError("Cycles found no OptiX/CUDA device; check Blender, NVIDIA driver and container GPU access")


if __name__ == "__main__":
    configure_gpu()