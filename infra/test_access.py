#!/usr/bin/env python3
"""Compatibility launcher; use python -m infra.access.check for new commands."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from infra.access.check import main


if __name__ == "__main__":
    raise SystemExit(main())