"""Stable local output paths, independent of the process working directory."""
from __future__ import annotations

import os
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]


def data_root(root: str | Path | None = None) -> Path:
    """Return the data directory; relative overrides are relative to the code root."""
    selected = root if root is not None else os.environ.get("SYNTHRIG_DATA_ROOT")
    if selected is None:
        return CODE_ROOT.parent / "data"
    path = Path(selected).expanduser()
    return (path if path.is_absolute() else CODE_ROOT / path).resolve()


def output_path(output: str | Path, root: str | Path | None = None) -> Path:
    """Resolve a relative output beneath the data root, or preserve an absolute path."""
    path = Path(output).expanduser()
    return (path if path.is_absolute() else data_root(root) / path).resolve()