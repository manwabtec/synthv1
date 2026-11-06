import subprocess
import sys

from synthrig.paths import CODE_ROOT


def test_cloud_agnostic_modules_do_not_load_blender_or_aws():
    result = subprocess.run(
        [sys.executable, "-c", "import sys; import synthrig.opticsrig, synthrig.sensor, "
         "synthrig.dataset, synthrig.stereo, synthrig.pipeline; "
         "assert 'bpy' not in sys.modules; assert 'boto3' not in sys.modules"],
        cwd=CODE_ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def test_access_module_uses_repository_config():
    from infra.access.check import CONFIG_PATH

    assert CONFIG_PATH == CODE_ROOT / "infra/config.yaml"


def test_old_access_launcher_still_has_help():
    result = subprocess.run(
        [sys.executable, "infra/test_access.py", "--help"],
        cwd=CODE_ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "instance_id" in result.stdout