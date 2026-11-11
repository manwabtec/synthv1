"""Headless GUI test (Streamlit AppTest): the app runs, and opening it never edits configs."""
import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")

from streamlit.testing.v1 import AppTest  # noqa: E402

from synthrig.paths import CODE_ROOT  # noqa: E402

CONFIGS = [CODE_ROOT / "configs/opticsrigs/wheel_rail_v1.yaml", CODE_ROOT / "configs/scenes/wheel_rail.yaml"]


def test_app_opens_without_errors_or_file_changes():
    before = [p.read_bytes() for p in CONFIGS]
    at = AppTest.from_file(str(CODE_ROOT / "tools/gui/app.py"), default_timeout=120).run()
    assert not at.exception, at.exception
    assert [t.label for t in at.tabs] == ["Layout", "Render & images", "3D view"]
    assert [p.read_bytes() for p in CONFIGS] == before  # just looking must not rewrite files

    # edit with saving switched off: the layout recomputes, files stay untouched
    [t for t in at.toggle if t.label.startswith("Save edits")][0].set_value(False).run()
    x = [n for n in at.number_input if n.label == "value"][0]
    x.set_value(0.25).run()
    assert not at.exception, at.exception
    assert [p.read_bytes() for p in CONFIGS] == before
