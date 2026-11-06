import pytest
import yaml

from synthrig.paths import CODE_ROOT
from synthrig.pipeline import RunConfig
from tools.config import generate, read_mapping, set_value, validate


def test_validate_and_edit():
    path = CODE_ROOT / "configs/runs/app1_demo.yaml"
    data = read_mapping(path)
    set_value(data, "render.samples", 8)
    validate(data, "run", path)
    set_value(data, "render.unknown_setting", 3)
    with pytest.raises(ValueError):
        validate(data, "run", path)


def test_sweep_is_repeatable_and_fixed_rig(tmp_path):
    spec = tmp_path / "sweep.yaml"
    spec.write_text(yaml.safe_dump({"parameters": {"seed": [1, 2], "scene_params.ambient": [0.1, 0.2]}}))
    base = CODE_ROOT / "configs/runs/app1_demo.yaml"
    paths = generate(base, spec, tmp_path / "cases")
    assert len(paths) == 4
    runs = [RunConfig.load(path) for path in paths]
    assert [run.seed for run in runs] == [1, 2, 1, 2]
    assert len({run.rig for run in runs}) == 1
    assert len({run.output for run in runs}) == 4
    before = [path.read_bytes() for path in paths]
    with pytest.raises(ValueError, match="overwrite"):
        generate(base, spec, tmp_path / "cases")
    assert before == [path.read_bytes() for path in paths]


def test_sweep_cannot_change_hardware(tmp_path):
    spec = tmp_path / "sweep.yaml"
    spec.write_text("parameters:\n  rig: [other.yaml]\n")
    with pytest.raises(ValueError, match="physical rig stays fixed"):
        generate(CODE_ROOT / "configs/runs/app1_demo.yaml", spec, tmp_path / "cases")


def test_sweep_can_reference_rig_on_another_drive(tmp_path, monkeypatch):
    import tools.config as config_tools

    def different_drive(*args):
        raise ValueError("path is on another mount")

    spec = tmp_path / "sweep.yaml"
    spec.write_text("parameters:\n  seed: [1234]\n")
    monkeypatch.setattr(config_tools.os.path, "relpath", different_drive)
    paths = generate(CODE_ROOT / "configs/runs/app1_demo.yaml", spec, tmp_path / "cases")
    assert read_mapping(paths[0])["rig"] == (CODE_ROOT / "configs/opticsrigs/demo_4cam_2laser.yaml").as_posix()