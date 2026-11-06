"""Validate/edit YAML configuration or generate deterministic parameter sweeps."""
from __future__ import annotations

import argparse
import copy
import itertools
import os
from pathlib import Path
from typing import Any

import yaml

from synthrig.opticsrig.schema import Rig
from synthrig.pipeline import RunConfig


def read_mapping(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return data


def validate(data: dict[str, Any], kind: str, path: Path) -> None:
    """Validate configuration and referenced hardware without importing Blender."""
    if kind == "rig":
        Rig.model_validate(data)
    else:
        config = RunConfig.model_validate(data)
        rig = Path(config.rig)
        if not rig.is_absolute():
            rig = path.parent / rig
        Rig.model_validate(read_mapping(rig))


def set_value(data: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    if not all(parts):
        raise ValueError("Parameter path cannot be empty")
    target = data
    for name in parts[:-1]:
        if name not in target or not isinstance(target[name], dict):
            raise ValueError(f"Not a mapping: {name} in {dotted}")
        target = target[name]
    target[parts[-1]] = value


def save(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    temporary.replace(path)


def generate(base: Path, specification: Path, destination: Path) -> list[Path]:
    """Create a Cartesian sweep with explicit values and no hidden randomness."""
    original = read_mapping(base)
    validate(original, "run", base)
    spec = read_mapping(specification)
    if set(spec) != {"parameters"} or not isinstance(spec["parameters"], dict) or not spec["parameters"]:
        raise ValueError("Sweep must contain a non-empty parameters mapping")
    parameters = spec["parameters"]
    if any(not isinstance(values, list) or not values for values in parameters.values()):
        raise ValueError("Each sweep parameter requires a non-empty list")
    keys = sorted(parameters)
    if any(not key.startswith("scene_params.") and key != "seed" for key in keys):
        raise ValueError("Sweep only scene_params.* or seed; the physical rig stays fixed")
    cases = []
    for index, values in enumerate(itertools.product(*(parameters[key] for key in keys))):
        data = copy.deepcopy(original)
        for key, value in zip(keys, values):
            set_value(data, key, value)
        target = destination / f"case_{index:04d}.yaml"
        rig = Path(original["rig"])
        if not rig.is_absolute():
            rig = (base.parent / rig).resolve()
        try:
            data["rig"] = os.path.relpath(rig, target.parent).replace(os.sep, "/")
        except ValueError:
            data["rig"] = rig.as_posix()
        data["output"] = f"datasets/{destination.name}/case_{index:04d}"
        validate(data, "run", target)
        if target.exists():
            raise ValueError(f"Refusing to overwrite existing case: {target}")
        cases.append((target, data))
    for target, data in cases:
        save(data, target)
    return [target for target, _data in cases]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("validate", "set"):
        child = commands.add_parser(command)
        child.add_argument("path", type=Path)
        child.add_argument("--kind", choices=["run", "rig"], default="run")
        if command == "set":
            child.add_argument("parameter", help="dotted mapping path, e.g. render.samples")
            child.add_argument("value", help="YAML value")
    sweep = commands.add_parser("sweep")
    sweep.add_argument("base", type=Path)
    sweep.add_argument("specification", type=Path)
    sweep.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "sweep":
            print(f"Created {len(generate(args.base, args.specification, args.out))} run configurations")
        else:
            data = read_mapping(args.path)
            if args.command == "set":
                set_value(data, args.parameter, yaml.safe_load(args.value))
            validate(data, args.kind, args.path)
            if args.command == "set":
                save(data, args.path)
            print(f"Valid {args.kind} configuration: {args.path}")
    except (ValueError, OSError, yaml.YAMLError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()