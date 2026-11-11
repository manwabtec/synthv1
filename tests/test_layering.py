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


def test_eval_and_host_tools_do_not_load_blender():
    code = ("import runpy, sys\n"
            "for f in ['eval/preview.py', 'eval/dots_eval.py', 'eval/stereo_eval.py']:\n"
            "    runpy.run_path(f, run_name='not_main')\n"
            "import infra.cloud, infra.worker, infra.disk\n"
            "import synthrig.layout, synthrig.sampling, synthrig.assets.wheelrail, synthrig.assets.classes\n"
            "try:\n    import tools.gui.figures, tools.gui.configio\nexcept ModuleNotFoundError:\n    pass\n"
            "assert 'bpy' not in sys.modules, 'bpy loaded'")
    result = subprocess.run([sys.executable, "-c", code], cwd=CODE_ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_access_module_uses_repository_config():
    from infra.access_check import CONFIG_PATH

    assert CONFIG_PATH == CODE_ROOT / "infra/config.yaml"


def test_access_check_has_help():
    result = subprocess.run(
        [sys.executable, "-m", "infra.access_check", "--help"],
        cwd=CODE_ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "instance_id" in result.stdout