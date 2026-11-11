from pathlib import Path

from synthrig.paths import CODE_ROOT, data_root, output_path


def test_default_is_sibling_data(monkeypatch):
    monkeypatch.delenv("SYNTHRIG_DATA_ROOT", raising=False)
    assert data_root() == CODE_ROOT.parent / "data"


def test_outputs_do_not_depend_on_working_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("SYNTHRIG_DATA_ROOT", "../data")
    expected = output_path("datasets/demo")
    monkeypatch.chdir(tmp_path)
    assert output_path("datasets/demo") == expected


def test_override_and_absolute_output(tmp_path, monkeypatch):
    monkeypatch.setenv("SYNTHRIG_DATA_ROOT", str(tmp_path / "environment"))
    assert output_path("datasets/demo", tmp_path) == tmp_path / "datasets/demo"
    assert output_path(tmp_path / "explicit") == tmp_path / "explicit"
    assert isinstance(data_root(), Path)


def test_run_config_resolves_rig_from_config_location(tmp_path, monkeypatch):
    from synthrig.pipeline import RunConfig

    config = CODE_ROOT / "configs/debug_clutter/run.yaml"
    monkeypatch.chdir(tmp_path)
    loaded = RunConfig.load(config)
    assert Path(loaded.rig) == CODE_ROOT / "configs/debug_clutter/rig.yaml"
    assert output_path(loaded.output).name == "debug_clutter"