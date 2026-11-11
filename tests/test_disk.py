from pathlib import Path

import pytest

from infra.storage import disk


def _dataset(root: Path, frames: int, bytes_per_frame: int) -> Path:
    (root / "frames").mkdir(parents=True)
    (root / "manifest.json").write_text("{}")
    for i in range(frames):
        d = root / "frames" / f"{i:06d}"
        d.mkdir()
        (d / "image.bin").write_bytes(b"x" * bytes_per_frame)
    return root


def test_missing_path_is_measured_at_existing_parent(tmp_path: Path):
    v = disk.volume("data_root", tmp_path / "not" / "yet")
    assert not v.exists and v.total > 0 and v.free <= v.total


def test_same_filesystem_is_one_row(tmp_path: Path):
    rows = disk.volumes({"data_root": str(tmp_path / "a"), "scratch": str(tmp_path / "b")})
    assert len(rows) == 1 and rows[0].labels == ["data_root", "scratch"]


def test_dataset_sizes_and_frame_estimate(tmp_path: Path):
    _dataset(tmp_path / "datasets/a", frames=2, bytes_per_frame=1000)
    _dataset(tmp_path / "datasets/x/y/b", frames=1, bytes_per_frame=5000)
    sizes = disk.dataset_sizes(tmp_path / "datasets")
    assert [s.frames for s in sizes] == [2, 1]
    assert disk.typical_frame_bytes(sizes) == pytest.approx(5002)  # 5000 + manifest share


def test_check_free_raises_when_low(tmp_path: Path):
    disk.check_free(tmp_path, 0)
    with pytest.raises(OSError, match="GiB free"):
        disk.check_free(tmp_path, 10 ** 9)


def test_report_flags_low_data_root(tmp_path: Path):
    _dataset(tmp_path / "datasets/a", frames=1, bytes_per_frame=10)
    text, ok = disk.report({"data_root": str(tmp_path)}, min_free_gb=10 ** 9,
                           datasets_dir=str(tmp_path / "datasets"))
    assert not ok and "LOW" in text and "datasets: 1" in text
    text, ok = disk.report({"data_root": str(tmp_path)}, min_free_gb=0.001)
    assert ok and "status: OK" in text
