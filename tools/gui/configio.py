"""Comment-preserving YAML editing for the GUI (ruamel.yaml round trip). No bpy."""
from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq

_yaml = YAML()
_yaml.preserve_quotes = True
_yaml.width = 4096
_yaml.indent(mapping=2, sequence=4, offset=2)


def load(path: Path) -> CommentedMap:
    return _yaml.load(Path(path).read_text())


def dumps(doc: Any) -> str:
    buf = io.StringIO()
    _yaml.dump(doc, buf)
    return buf.getvalue()


def save(doc: Any, path: Path) -> None:
    """Atomic write (temp file + rename) so a crash never leaves half a config."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(dumps(doc))
    os.replace(tmp, path)


def plain(doc: Any) -> Any:
    """ruamel containers -> plain dicts/lists (for pydantic validation)."""
    return json.loads(json.dumps(doc, default=str))


def flow(values) -> CommentedSeq:
    """A list written inline as [a, b, c], rounded to 0.1 mm / 0.001 deg."""
    seq = CommentedSeq([round(float(v), 4) if isinstance(v, float) else v for v in values])
    seq.fa.set_flow_style()
    return seq


def flow_map(d: dict) -> CommentedMap:
    m = CommentedMap(d)
    m.fa.set_flow_style()
    return m


def _same(a: Any, b: Any) -> bool:
    a, b = plain(a), plain(b)
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b))
    return a == b


def set_if_changed(mapping: CommentedMap, key: str, value: Any, default: Any = None) -> bool:
    """Set mapping[key] = value unless equal (a missing key counts as `default`);
    lists become inline sequences."""
    old = mapping.get(key, default)
    new = flow(value) if isinstance(value, (list, tuple)) else value
    if _same(old, new):
        return False
    mapping[key] = new
    return True
