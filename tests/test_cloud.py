from pathlib import Path

import pytest
import yaml

from infra import cloud


def valid_config() -> dict:
    return {
        "project": "synthrig",
        "owner": "owner@example.com",
        "cost_center": "research",
        "aws_profile": "sandbox",
        "region": "us-east-1",
        "vpc_id": None,
        "subnet_ids": [],
        "bucket": "example-synthrig-bucket",
        "ecr_repo": "synthrig",
        "dev_instance_type": "g6.xlarge",
        "batch_instance_types": ["g6.xlarge"],
        "batch_max_vcpus": 4,
        "use_spot": True,
        "dev_idle_stop_minutes": 60,
        "monthly_budget_usd": 100,
    }


def write_config(path: Path, config: dict) -> None:
    path.write_text(yaml.safe_dump(config), encoding="utf-8")


def test_load_config_rejects_placeholders(tmp_path: Path):
    config = valid_config()
    config["bucket"] = "<globally-unique-name>"
    path = tmp_path / "config.yaml"
    write_config(path, config)

    with pytest.raises(cloud.CloudError, match="unresolved placeholders: bucket"):
        cloud.load_config(path)


def test_load_config_requires_subnets_for_existing_vpc(tmp_path: Path):
    config = valid_config()
    config["vpc_id"] = "vpc-123"
    path = tmp_path / "config.yaml"
    write_config(path, config)

    with pytest.raises(cloud.CloudError, match="subnet_ids is required"):
        cloud.load_config(path)


def test_services_stop_destroys_only_runtime(monkeypatch: pytest.MonkeyPatch):
    calls = []
    monkeypatch.setattr(cloud, "require_tool", lambda _name: None)
    monkeypatch.setattr(cloud, "load_config", valid_config)
    monkeypatch.setattr(cloud, "write_tfvars", lambda _config: None)
    monkeypatch.setattr(cloud, "terraform", lambda directory, action: calls.append((directory, action)))

    cloud.main(["services", "stop", "--yes"])

    assert calls == [(cloud.RUNTIME_DIR, "destroy")]


def test_services_start_applies_storage_before_runtime(monkeypatch: pytest.MonkeyPatch):
    calls = []
    monkeypatch.setattr(cloud, "require_tool", lambda _name: None)
    monkeypatch.setattr(cloud, "load_config", valid_config)
    monkeypatch.setattr(cloud, "verify_identity", lambda _config: None)
    monkeypatch.setattr(cloud, "write_tfvars", lambda _config: None)
    monkeypatch.setattr(cloud, "terraform", lambda directory, action: calls.append((directory, action)))

    cloud.main(["services", "start", "--yes"])

    assert calls == [
        (cloud.STORAGE_DIR, "apply"),
        (cloud.RUNTIME_DIR, "apply"),
    ]


def test_transfer_commands(monkeypatch, tmp_path):
    from infra.storage import s3

    calls = []
    monkeypatch.setattr(cloud, "storage_client", lambda _uri: "client")
    monkeypatch.setattr(s3, "push", lambda *args: calls.append(args) or 1)
    monkeypatch.setattr(s3, "pull", lambda *args: calls.append(args) or 1)
    assert cloud.main(["push", "datasets/demo", "--to", "s3://bucket/run", "--data-root", str(tmp_path)]) == 0
    assert calls[0][1] == tmp_path / "datasets/demo"
    assert cloud.main(["pull", "s3://bucket/run", "--frames", "0:2", "--data-root", str(tmp_path)]) == 0
    assert calls[1][2] == tmp_path / "datasets/run"
    assert calls[1][3] == (0, 2)


@pytest.mark.parametrize("value", ["-1:2", "2:1", "1", "a:b", "0:1:2"])
def test_bad_frame_range(value):
    import argparse

    with pytest.raises(argparse.ArgumentTypeError):
        cloud.frame_range(value)