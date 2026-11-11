"""Scene-config sampling: fixed values and randomisation ranges in YAML. No bpy.

A scene config (configs/scenes/<scene>.yaml) maps parameter names to either a fixed value
or one sampling spec:

    wheel_x_m: 0.0                      # fixed
    wheel_rot_deg: {uniform: [0, 360]}  # uniform in [lo, hi)
    lateral_shift_m: {normal: [0, 0.003]}   # mean, standard deviation
    rail_wear: {choice: [new, worn]}    # one of the listed values

Draws use the frame's rng, in sorted key order, so frame i is reproducible from (seed, i)
and does not depend on the order of keys in the file.
"""
from __future__ import annotations

from typing import Any

import numpy as np

SPEC_KEYS = ("uniform", "normal", "choice")


def is_spec(value: Any) -> bool:
    return isinstance(value, dict) and len(value) == 1 and next(iter(value)) in SPEC_KEYS


def sample_value(value: Any, rng: np.random.Generator) -> Any:
    """Draw one value from a spec, or return a fixed value unchanged."""
    if not is_spec(value):
        return value
    kind, arg = next(iter(value.items()))
    if kind == "uniform":
        lo, hi = arg
        return float(rng.uniform(lo, hi))
    if kind == "normal":
        mu, sigma = arg
        return float(rng.normal(mu, sigma))
    options = list(arg)
    return options[int(rng.integers(len(options)))]


def sample_config(config: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    """Fixed values copied, specs drawn (sorted keys -> order-independent)."""
    return {k: sample_value(config[k], rng) for k in sorted(config)}
