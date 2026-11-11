"""Free-space report for the volumes synthrig writes to (stdlib only, Python >= 3.9).

Volumes are the persistent EBS data root, the instance-store scratch and Docker's
data-root. Sizes are bytes internally and GiB in reports. A path that does not exist yet
is measured at its nearest existing parent, so the report works before first use.
"""
from __future__ import annotations

import os
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

GIB = 1024 ** 3


@dataclass
class Volume:
    """Usage of the filesystem holding `path` (bytes)."""

    label: str
    path: str
    mount: str
    total: int
    used: int
    free: int
    exists: bool
    labels: List[str] = field(default_factory=list)  # every label sharing this filesystem

    @property
    def pct_used(self) -> float:
        return 100.0 * self.used / self.total if self.total else 0.0


@dataclass
class DatasetSize:
    """On-disk size of one dataset directory (bytes)."""

    path: str
    bytes: int
    frames: int

    @property
    def bytes_per_frame(self) -> Optional[float]:
        return self.bytes / self.frames if self.frames else None


def _existing(path: Path) -> Path:
    while not path.exists() and path != path.parent:
        path = path.parent
    return path


def _mount_point(path: Path) -> str:
    path = path.resolve()
    while not os.path.ismount(path) and path != path.parent:
        path = path.parent
    return str(path)


def volume(label: str, path: str | Path) -> Volume:
    """Measure the filesystem that holds `path` (or would hold it once created)."""
    target = Path(path).expanduser()
    probe = _existing(target)
    usage = shutil.disk_usage(probe)
    return Volume(label, str(target), _mount_point(probe), usage.total, usage.used, usage.free,
                  target.exists(), [label])


def volumes(paths: Dict[str, str]) -> List[Volume]:
    """Measure each labelled path; paths on the same filesystem are merged into one row."""
    by_device: Dict[int, Volume] = {}
    for label, path in paths.items():
        v = volume(label, path)
        dev = os.stat(_existing(Path(v.path))).st_dev
        if dev in by_device:
            by_device[dev].labels.append(label)
        else:
            by_device[dev] = v
    return list(by_device.values())


def dataset_sizes(datasets_dir: str | Path) -> List[DatasetSize]:
    """Size and frame count of every dataset (a directory holding manifest.json)."""
    root = Path(datasets_dir)
    if not root.is_dir():
        return []
    result = []
    for manifest in sorted(root.rglob("manifest.json")):
        ds = manifest.parent
        total = sum(f.stat().st_size for f in ds.rglob("*") if f.is_file())
        frames_dir = ds / "frames"
        frames = sum(1 for d in frames_dir.iterdir() if d.is_dir()) if frames_dir.is_dir() else 0
        result.append(DatasetSize(str(ds), total, frames))
    return result


def typical_frame_bytes(sizes: List[DatasetSize]) -> Optional[float]:
    """Largest average frame size across datasets: a conservative per-frame estimate."""
    per_frame = [s.bytes_per_frame for s in sizes if s.bytes_per_frame]
    return max(per_frame) if per_frame else None


def check_free(path: str | Path, min_free_gb: float) -> Volume:
    """Raise OSError if the filesystem holding `path` has less than `min_free_gb` GiB free."""
    v = volume("target", path)
    if v.free < min_free_gb * GIB:
        raise OSError(f"only {v.free / GIB:.1f} GiB free on {v.mount} (holding {path}); "
                      f"need at least {min_free_gb:g} GiB. Free space or move data, then retry.")
    return v


def report(paths: Dict[str, str], min_free_gb: float, datasets_dir: Optional[str] = None) -> tuple[str, bool]:
    """Human-readable table plus ok flag (False if the data root is below `min_free_gb`)."""
    rows = volumes(paths)
    lines = [f"{'VOLUME':<28} {'MOUNT':<12} {'SIZE':>8} {'USED':>8} {'FREE':>8} {'USE%':>5}"]
    ok = True
    for v in rows:
        low = v.free < min_free_gb * GIB and "data_root" in v.labels
        ok &= not low
        name = ",".join(v.labels) + ("" if v.exists else " (missing)")
        lines.append(f"{name:<28} {v.mount:<12} {v.total / GIB:>7.1f}G {v.used / GIB:>7.1f}G "
                     f"{v.free / GIB:>7.1f}G {v.pct_used:>4.0f}%" + ("  LOW" if low else ""))
    if datasets_dir is not None:
        sizes = dataset_sizes(datasets_dir)
        total = sum(s.bytes for s in sizes)
        lines.append("")
        lines.append(f"datasets: {len(sizes)} under {datasets_dir}, {total / GIB:.2f} GiB total")
        for s in sorted(sizes, key=lambda s: s.bytes, reverse=True)[:10]:
            per = f"{s.bytes_per_frame / 1024 ** 2:.0f} MiB/frame" if s.bytes_per_frame else "no frames"
            lines.append(f"  {s.bytes / GIB:8.2f}G  {s.frames:6d} frames  {per:>16}  {s.path}")
        per_frame = typical_frame_bytes(sizes)
        data = next((v for v in rows if "data_root" in v.labels), None)
        if per_frame and data:
            usable = max(0.0, data.free - min_free_gb * GIB)
            lines.append(f"room for ~{int(usable // per_frame)} more frames at the largest "
                         f"{per_frame / 1024 ** 2:.0f} MiB/frame (keeping {min_free_gb:g} GiB free)")
    lines.append("")
    lines.append(f"status: {'OK' if ok else 'LOW SPACE'} (minimum free on data_root: {min_free_gb:g} GiB)")
    return "\n".join(lines), ok


def as_dict(paths: Dict[str, str], datasets_dir: Optional[str] = None) -> dict:
    """Machine-readable form of `report` (bytes)."""
    out = {"volumes": [{**asdict(v), "pct_used": round(v.pct_used, 1)} for v in volumes(paths)]}
    if datasets_dir is not None:
        out["datasets"] = [{**asdict(s), "bytes_per_frame": s.bytes_per_frame}
                           for s in dataset_sizes(datasets_dir)]
    return out
