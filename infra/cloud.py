#!/usr/bin/env python3
"""Manage synthrig's local image and AWS infrastructure lifecycle."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import yaml
from botocore.exceptions import BotoCoreError, ClientError


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
INFRA = ROOT / "infra"
CONFIG_PATH = INFRA / "config.yaml"
TFVARS_PATH = INFRA / "terraform.tfvars.json"
STORAGE_DIR = INFRA / "terraform" / "storage"
RUNTIME_DIR = INFRA / "terraform" / "runtime"

REQUIRED_CONFIG = {
    "project",
    "owner",
    "cost_center",
    "aws_profile",
    "region",
    "bucket",
    "ecr_repo",
    "dev_instance_type",
    "batch_instance_types",
    "batch_max_vcpus",
    "use_spot",
    "dev_idle_stop_minutes",
    "monthly_budget_usd",
}


class CloudError(RuntimeError):
    """A user-correctable cloud configuration or command error."""


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    if not path.is_file():
        raise CloudError(
            f"Missing {display_path(path)}. Copy infra/config.example.yaml "
            "to infra/config.yaml and fill in the AWS values."
        )
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise CloudError(f"{display_path(path)} must contain a YAML mapping")

    missing = sorted(REQUIRED_CONFIG - data.keys())
    placeholders = sorted(
        key for key, value in data.items()
        if isinstance(value, str) and (not value.strip() or "<" in value or ">" in value)
    )
    if missing or placeholders:
        details = []
        if missing:
            details.append(f"missing keys: {', '.join(missing)}")
        if placeholders:
            details.append(f"unresolved placeholders: {', '.join(placeholders)}")
        raise CloudError(f"Invalid {display_path(path)} ({'; '.join(details)})")
    if not data["batch_instance_types"]:
        raise CloudError("batch_instance_types must contain at least one GPU instance type")
    if data.get("vpc_id") and not data.get("subnet_ids"):
        raise CloudError("subnet_ids is required when vpc_id is set")
    return data


def run(command: Sequence[str], *, cwd: Path | None = None) -> None:
    print("+", subprocess.list2cmdline(list(command)))
    subprocess.run(command, cwd=cwd or ROOT, check=True)


def require_tool(name: str) -> None:
    if shutil.which(name) is None:
        raise CloudError(f"Required executable not found on PATH: {name}")


def write_tfvars(config: dict[str, Any]) -> None:
    TFVARS_PATH.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def terraform(directory: Path, action: str) -> None:
    if not directory.is_dir():
        raise CloudError(
            f"Terraform stack {directory.relative_to(ROOT)} is not implemented yet; "
            "complete Phase 0.2 before starting AWS services."
        )
    run(["terraform", f"-chdir={directory}", "init", "-input=false"])
    args = ["terraform", f"-chdir={directory}", action, "-input=false"]
    if action in {"apply", "destroy"}:
        args.extend(["-auto-approve", f"-var-file={TFVARS_PATH}"])
    run(args)


def verify_identity(config: dict[str, Any]) -> None:
    run([
        "aws", "sts", "get-caller-identity", "--profile", config["aws_profile"],
        "--region", config["region"], "--output", "table",
    ])


def services_start(args: argparse.Namespace) -> None:
    if not args.yes:
        raise CloudError("services start creates billable resources; rerun with --yes")
    require_tool("aws")
    require_tool("terraform")
    config = load_config()
    verify_identity(config)
    write_tfvars(config)
    terraform(STORAGE_DIR, "apply")
    terraform(RUNTIME_DIR, "apply")
    print("AWS storage and runtime services are started.")


def services_stop(args: argparse.Namespace) -> None:
    if not args.yes:
        raise CloudError("services stop destroys disposable infrastructure; rerun with --yes")
    require_tool("terraform")
    config = load_config()
    write_tfvars(config)
    terraform(RUNTIME_DIR, "destroy")
    print("Runtime destroyed. The S3 bucket and its Terraform state remain.")


def services_status(_args: argparse.Namespace) -> None:
    load_config()
    require_tool("terraform")
    terraform(STORAGE_DIR, "show")
    terraform(RUNTIME_DIR, "show")


def config_check(_args: argparse.Namespace) -> None:
    config = load_config()
    print(
        f"Configuration is valid for profile {config['aws_profile']} "
        f"in {config['region']} (bucket {config['bucket']})."
    )


def frame_range(value: str) -> tuple[int, int]:
    try:
        start, stop = (int(part) for part in value.split(":"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected START:STOP, with STOP exclusive") from exc
    if start < 0 or stop <= start:
        raise argparse.ArgumentTypeError("Frame range must satisfy 0 <= START < STOP")
    return start, stop


def storage_client(uri: str):
    import boto3
    from infra.storage.s3 import parse_uri

    bucket, _prefix = parse_uri(uri)
    config = load_config()
    if bucket != config["bucket"]:
        raise CloudError("S3 URI bucket must match infra/config.yaml")
    return boto3.Session(
        profile_name=config.get("aws_profile") or None, region_name=config["region"]
    ).client("s3")


def dataset_push(args: argparse.Namespace) -> None:
    from infra.storage.s3 import push
    from synthrig.paths import output_path

    count = push(storage_client(args.to), output_path(args.source, args.data_root), args.to)
    print(f"Uploaded {count} new files to {args.to}; matching files were skipped.")


def dataset_pull(args: argparse.Namespace) -> None:
    from infra.storage.s3 import parse_uri, pull
    from synthrig.paths import output_path

    _bucket, prefix = parse_uri(args.uri)
    destination = output_path(args.out or f"datasets/{prefix.split('/')[-1]}", args.data_root)
    count = pull(storage_client(args.uri), args.uri, destination, args.frames)
    print(f"Downloaded {count} new files to {destination}; matching files were skipped.")


def local_check(args: argparse.Namespace) -> None:
    from infra.diagnostics.local import check

    check(gpu=args.gpu, render=args.render)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)

    config_cmd = commands.add_parser("config", help="validate local AWS configuration")
    config_cmd.set_defaults(func=config_check)

    push_cmd = commands.add_parser("push", help="upload a local dataset to a new S3 run prefix")
    push_cmd.add_argument("source", help="absolute path or path relative to the data root")
    push_cmd.add_argument("--to", required=True, help="s3://bucket/datasets/run")
    push_cmd.add_argument("--data-root")
    push_cmd.set_defaults(func=dataset_push)
    pull_cmd = commands.add_parser("pull", help="download a verified S3 dataset snapshot")
    pull_cmd.add_argument("uri", help="s3://bucket/datasets/run")
    pull_cmd.add_argument("--out", help="absolute path or path relative to the data root")
    pull_cmd.add_argument("--frames", type=frame_range, help="START:STOP (STOP exclusive)")
    pull_cmd.add_argument("--data-root")
    pull_cmd.set_defaults(func=dataset_pull)
    local = commands.add_parser("local-check", help="check this PC without accessing AWS")
    local.add_argument("--gpu", action="store_true", help="require NVIDIA and a Cycles GPU")
    local.add_argument("--render", action="store_true", help="also render one low-resolution frame")
    local.set_defaults(func=local_check)

    services = commands.add_parser("services", help="manage all AWS resources")
    service_commands = services.add_subparsers(dest="services_command", required=True)
    start = service_commands.add_parser("start", help="create storage and runtime resources")
    start.add_argument("--yes", action="store_true", help="confirm creation of billable resources")
    start.set_defaults(func=services_start)
    stop = service_commands.add_parser("stop", help="destroy runtime but preserve S3")
    stop.add_argument("--yes", action="store_true", help="confirm runtime destruction")
    stop.set_defaults(func=services_stop)
    status = service_commands.add_parser("status", help="show Terraform-managed resources")
    status.set_defaults(func=services_status)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        args.func(args)
    except (RuntimeError, ValueError, OSError, subprocess.CalledProcessError, BotoCoreError, ClientError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())