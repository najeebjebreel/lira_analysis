"""Compatibility wrapper for the benchmark CLI."""

from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.benchmark_cli import main


if __name__ == "__main__":
    main()
