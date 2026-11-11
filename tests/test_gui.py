"""GUI tests: the Streamlit app opens without touching configs; the layout viewer's data
exchange (tools/gui/layout_data.py) round-trips edits into the YAML text correctly.

The viewer itself (tools/gui/layout_view, JavaScript) is not run here; its geometry
(geom.js) mirrors synthrig.layout, which test_wheelrail covers.
"""
import copy
import json

import pytest

pytest.importorskip("ruamel.yaml")

from synthrig.opticsrig.schema import Rig  # noqa: E402
from synthrig.paths import CODE_ROOT  # noqa: E402
from tools.gui import configio as cio  # noqa: E402
from tools.gui import layout_data as ld  # noqa: E402

SCENES = ("wheel_rail", "wheel_tread")
CONFIGS = [CODE_ROOT / f"configs/{scene}/{f}.yaml" for scene in SCENES for f in ("rig", "scene")]


def _docs(scene):
    return cio.load(CODE_ROOT / f"configs/{scene}/rig.yaml"), cio.load(CODE_ROOT / f"configs/{scene}/scene.yaml")


@pytest.mark.parametrize("scene", SCENES)
def test_viewer_payload_is_json_and_complete(scene):
    rig_doc, scene_doc = _docs(scene)
    st = ld.view_state(rig_doc, scene_doc)
    g = ld.geometry(st["scene"]["wheel_profile"], st["scene"]["rail_profile"])
    assert len(json.dumps({"state": st, "geometry": g})) < 400_000
    assert {c["name"] for c in st["cameras"]} == {c["name"] for c in rig_doc["cameras"]}
    for las in st["lasers"]:
        assert len(las["tangents"]) > 0
    # the ray-casting outline is closed around the measured contour
    outline = g["wheel"]["solids"][0]
    assert outline[0][1] > 0.3 and len(outline) > 100


def test_unchanged_edits_leave_text_identical():
    rig_doc, scene_doc = _docs("wheel_tread")
    before = cio.dumps(rig_doc), cio.dumps(scene_doc)
    st = ld.view_state(rig_doc, scene_doc)
    edits = {"cameras": {c["name"]: {"position": c["position"], "look_at": c["look_at"]} for c in st["cameras"]},
             "lasers": {l["name"]: {"position": l["position"], "look_at": l["look_at"], "power_w": l["power_w"],
                                    "enabled": l["enabled"]} for l in st["lasers"]},
             "scene": {"wheel_profile": st["scene"]["wheel_profile"]}}
    ld.apply_edits(rig_doc, scene_doc, edits)
    assert (cio.dumps(rig_doc), cio.dumps(scene_doc)) == before


def test_edits_change_only_what_was_edited():
    rig_doc, scene_doc = _docs("wheel_tread")
    orig = cio.dumps(rig_doc)
    cam = rig_doc["cameras"][0]["name"]
    las = rig_doc["lasers"][0]["name"]
    ld.apply_edits(rig_doc, scene_doc, {
        "cameras": {cam: {"position": [0.3, -0.9, 0.02], "hfov_deg": 40.0, "up": [0.0, 0.1, 0.99499]}},
        "lasers": {las: {"pattern": {"type": "line", "fan_deg": 25.0, "angle_deg": 10.0}}},
        "scene": {"wheel_x_m": 0.05, "lateral_shift_m": {"uniform": [-0.003, 0.003]}},
    })
    rig = Rig.model_validate(cio.plain(rig_doc))
    c = rig.camera(cam)
    assert c.pose.position == [0.3, -0.9, 0.02] and c.intrinsics.hfov_deg == 40.0
    assert c.pose.up == pytest.approx([0.0, 0.1, 0.99499], abs=1e-4)  # written to 0.1 mm / 1e-4
    assert rig.laser(las).pattern.type == "line" and rig.laser(las).pattern.fan_deg == 25.0
    assert scene_doc["wheel_x_m"] == 0.05 and cio.plain(scene_doc["lateral_shift_m"]) == {"uniform": [-0.003, 0.003]}
    or_lines, new_lines = orig.splitlines(), cio.dumps(rig_doc).splitlines()
    assert len(or_lines) == len(new_lines)  # inline maps stay inline
    changed = [i for i, (a, b) in enumerate(zip(or_lines, new_lines)) if a != b]
    assert len(changed) == 3, changed  # camera intrinsics, camera pose, laser pattern
    comments = lambda lines: [ln.split("#", 1)[1] for ln in lines if "#" in ln]  # noqa: E731
    assert comments(or_lines) == comments(new_lines)


def test_app_opens_without_errors_or_file_changes():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    before = [p.read_bytes() for p in CONFIGS]
    at = AppTest.from_file(str(CODE_ROOT / "tools/gui/app.py"), default_timeout=120).run()
    assert not at.exception, at.exception
    assert [t.label for t in at.tabs] == ["Layout", "Render & images", "3D view"]
    at.selectbox[0].set_value(CODE_ROOT / "configs/wheel_rail/run.yaml").run()
    assert not at.exception, at.exception
    assert [p.read_bytes() for p in CONFIGS] == before  # just looking must not rewrite files


def test_deepcopy_keeps_comments():
    rig_doc, _ = _docs("wheel_tread")
    assert cio.dumps(copy.deepcopy(rig_doc)) == cio.dumps(rig_doc)
