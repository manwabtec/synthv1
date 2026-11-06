#!/usr/bin/env python3
"""Check AWS SSO, EC2, and SSM access for one existing instance."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

import boto3
import yaml
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "infra" / "config.yaml"


def load_access_config() -> tuple[str, str]:
    if not CONFIG_PATH.is_file():
        raise ValueError("Missing infra/config.yaml; copy infra/config.example.yaml and fill aws_profile and region.")

    config: Any = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("infra/config.yaml must contain a YAML mapping.")

    profile = config.get("aws_profile")
    region = config.get("region")
    if not isinstance(profile, str) or not profile.strip() or "<" in profile or ">" in profile:
        raise ValueError("Set aws_profile to your AWS credential-provider profile in infra/config.yaml.")
    if not isinstance(region, str) or not region.strip() or "<" in region or ">" in region:
        raise ValueError("Set region in infra/config.yaml.")
    return profile, region


def test_access(args: argparse.Namespace) -> None:
    profile, region = load_access_config()
    session = boto3.Session(profile_name=profile, region_name=region)
    identity = session.client("sts").get_caller_identity()
    print(f"AWS identity: {identity['Arn']} (account {identity['Account']})")

    response = session.client("ec2").describe_instances(InstanceIds=[args.instance_id])
    instances = [
        instance
        for reservation in response.get("Reservations", [])
        for instance in reservation.get("Instances", [])
    ]
    if not instances:
        raise ValueError(f"EC2 did not return instance {args.instance_id}.")
    state = instances[0]["State"]["Name"]
    print(f"EC2 instance state: {state}")
    if state != "running":
        raise ValueError("The instance must be running before opening an SSM session.")

    ssm_response = session.client("ssm").describe_instance_information(
        Filters=[{"Key": "InstanceIds", "Values": [args.instance_id]}]
    )
    managed = ssm_response.get("InstanceInformationList", [])
    if not managed:
        raise ValueError(
            "SSM does not list this instance. Check its SSM Agent, instance profile "
            "(AmazonSSMManagedInstanceCore), and network access to Systems Manager."
        )
    ping_status = managed[0].get("PingStatus", "Unknown")
    print(f"SSM managed-instance status: {ping_status}")
    if ping_status != "Online":
        raise ValueError("The instance is not Online in SSM yet; check its agent and network access.")

    print("Access check passed: AWS identity, EC2 read access, and SSM connectivity are working.")
    if args.session:
        raise ValueError(
            "Interactive SSM transport is not provided by boto3 alone. Use the AWS Console, "
            "or install/configure the Session Manager plugin and use the AWS CLI for the shell."
        )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("instance_id", help="running EC2 instance ID, for example i-0123456789abcdef0")
    result.add_argument(
        "--session",
        action="store_true",
        help="request an interactive session (requires Session Manager transport outside boto3)",
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        test_access(args)
    except (ValueError, KeyError, NoCredentialsError, ClientError, BotoCoreError, OSError) as exc:
        print(f"Access check failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())