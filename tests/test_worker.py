"""Shard contract of infra/worker.py, with a fake renderer (no Blender needed)."""
from pathlib import Path

import pytest

from infra import worker


@pytest.mark.parametrize("frames, shards", [(8, 2), (10, 3), (3, 5), (1, 1), (1000, 50)])
def test_shards_cover_every_frame_once(frames, shards):
    covered = [i for k in range(shards) for i in worker.shard_range(k, shards, frames)]
    assert covered == list(range(frames))


def test_shard_index_out_of_range():
    with pytest.raises(ValueError):
        worker.shard_range(2, 2, 10)


class FakeRender:
    def __init__(self, scratch: Path, fail_at=None):
        self.scratch, self.fail_at, self.calls = scratch, fail_at, []

    def __call__(self, idx: int) -> Path:
        if idx == self.fail_at:
            raise worker.Interrupted("SIGTERM")
        self.calls.append(idx)
        d = self.scratch / "frames" / f"{idx:06d}"
        d.mkdir(parents=True)
        (d / "meta.json").write_text("{}")
        return d


def test_interrupted_shard_resumes_only_missing_frames(tmp_path: Path):
    run_dir, log = tmp_path / "run", lambda _m: None
    with pytest.raises(worker.Interrupted):
        worker.run_shard(FakeRender(tmp_path / "s1", fail_at=2), run_dir, range(4), 0, 0, log)
    assert [worker.is_done(run_dir, i) for i in range(4)] == [True, True, False, False]

    # a crash between move and _DONE leaves a partial dir: it must be re-rendered, not trusted
    (run_dir / "frames" / "000002").mkdir()
    again = FakeRender(tmp_path / "s2")
    counts = worker.run_shard(again, run_dir, range(4), 0, 0, log)
    assert again.calls == [2, 3] and counts == {"skipped": 2, "rendered": 2}
    assert all(worker.is_done(run_dir, i) for i in range(4))
    assert not list((run_dir / "frames").glob(".*"))  # no temp dirs left behind
    assert not (tmp_path / "s2" / "frames" / "000003").exists()  # moved out of scratch


def test_low_disk_stops_before_rendering(tmp_path: Path):
    render = FakeRender(tmp_path / "s")
    with pytest.raises(OSError, match="GiB free"):
        worker.run_shard(render, tmp_path / "run", range(2), 0, 10 ** 9, lambda _m: None)
    assert render.calls == []


def test_stop_flag_stops_between_frames(tmp_path: Path):
    render = FakeRender(tmp_path / "s")
    with pytest.raises(worker.Interrupted):
        worker.run_shard(render, tmp_path / "run", range(5), 0, 0, lambda _m: None,
                         stop_requested=lambda: len(render.calls) >= 2)
    assert render.calls == [0, 1] and worker.is_done(tmp_path / "run", 1)
