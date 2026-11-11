import json
from pathlib import Path

import pytest
import yaml

from infra import cloud


def test_defaults_are_machine_neutral(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("SYNTHRIG_DATA_ROOT", str(tmp_path / "data"))
    config = cloud.load_config(tmp_path / "missing.yaml")
    assert config["data_root"] == str(tmp_path / "data")
    assert config["scratch_dir"] == str(tmp_path / "data" / "scratch")
    assert config["docker_root"] is None and config["min_free_gb"] == cloud.DEFAULTS["min_free_gb"]


def test_config_overrides_defaults(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"data_root": "/mnt/ebs", "min_free_gb": 5, "region": "us-east-1"}))
    config = cloud.load_config(path)
    assert config["data_root"] == "/mnt/ebs" and config["min_free_gb"] == 5
    assert config["scratch_dir"] == "/mnt/ebs/scratch"


@pytest.mark.parametrize("data, match", [
    ({"owner": "<email>"}, "unresolved placeholders: owner"),
    ({"bucket": "x"}, "unknown keys: bucket"),
    ({"min_free_gb": 0}, "min_free_gb"),
])
def test_config_rejects_bad_values(tmp_path: Path, data, match):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(cloud.CloudError, match=match):
        cloud.load_config(path)


def test_container_mounts_data_and_runs_as_user(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(cloud, "docker", lambda: ["docker"])
    monkeypatch.setattr(cloud, "image_info", lambda _c: {"id": "sha256:abc"})
    monkeypatch.setattr(cloud, "instance_type", lambda: "g5.xlarge")
    config = {**cloud.DEFAULTS, "data_root": str(tmp_path / "d"), "scratch_dir": str(tmp_path / "s")}
    cmd = cloud.container(config, ["pytest"], gpu=False)
    assert "--gpus" not in cmd and "--rm" in cmd
    assert f"{tmp_path / 'd'}:/data" in cmd and (tmp_path / "s").is_dir()
    assert "SYNTHRIG_IMAGE_DIGEST=sha256:abc" in cmd and "SYNTHRIG_INSTANCE_TYPE=g5.xlarge" in cmd
    assert cmd[-2:] == ["synthrig:latest", "pytest"]
    detached = cloud.container(config, ["--config", "x"], detach=True, name="w0")
    assert detached[2:5] == ["-d", "--restart", "on-failure:3"] and "--gpus" in detached


def test_docker_falls_back_to_sudo(monkeypatch):
    monkeypatch.setattr(cloud.shutil, "which", lambda _n: "/usr/bin/docker")
    monkeypatch.setattr(cloud.os, "access", lambda *_a: False)
    assert cloud.docker() == ["sudo", "-n", "docker"]


def test_submit_refuses_dirty_tree_and_stale_image(monkeypatch, capsys):
    defaults = cloud.load_config(Path("/nonexistent.yaml"))
    monkeypatch.setattr(cloud, "load_config", lambda: dict(defaults))
    monkeypatch.setattr(cloud, "git_state", lambda: {"sha": "abc1234", "branch": "b", "dirty": True})
    monkeypatch.setattr(cloud, "image_info", lambda _c: {"id": "x", "git_sha": "0000000", "git_dirty": "false"})
    rc = cloud.main(["submit", "--config", "configs/runs/app1_demo.yaml", "--frames", "2"])
    err = capsys.readouterr().err
    assert rc == 2 and "uncommitted changes" in err and "built from 0000000" in err


def _frame(run_dir: Path, idx: int, seconds: float, done: bool = True):
    d = run_dir / "frames" / f"{idx:06d}"
    d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps({"timings_s": {"total": seconds}}))
    if done:
        (d / "_DONE").write_text("{}")


def test_run_status_counts_only_done_frames(tmp_path: Path):
    (tmp_path / "manifest.json").write_text(json.dumps({"run_config": {"frames": 4}}))
    _frame(tmp_path, 0, 10.0)
    _frame(tmp_path, 1, 20.0)
    _frame(tmp_path, 2, 99.0, done=False)
    s = cloud.run_status(tmp_path)
    assert (s["frames_done"], s["frames_total"], s["mean_frame_s"], s["remaining_s"]) == (2, 4, 15.0, 30)
