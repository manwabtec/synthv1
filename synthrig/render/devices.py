"""Explicit NVIDIA Cycles selection; fail instead of silently falling back to CPU."""
from __future__ import annotations

import bpy

# Backends tried in order when no explicit list is given. synthrig.pipeline narrows this
# for render.device OPTIX or CUDA.
DEFAULT_BACKENDS: tuple[str, ...] = ("OPTIX", "CUDA")


def configure_gpu(backends: tuple[str, ...] | None = None) -> list[str]:
    """Enable the first backend (default OptiX, then CUDA) that has NVIDIA devices."""
    preferences = bpy.context.preferences.addons["cycles"].preferences
    for backend in backends or DEFAULT_BACKENDS:
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
    raise RuntimeError(f"Cycles found no {'/'.join(backends or DEFAULT_BACKENDS)} device; check Blender, NVIDIA driver and container GPU access")


if __name__ == "__main__":
    configure_gpu()