#!/usr/bin/env python
"""Report which configured data paths resolve on this machine, and which do not.

Run this first after cloning. It is faster to see the whole picture here than to
discover one missing input at a time from tracebacks.

    python scripts/check_paths.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from swot_reservoir_storage.common import paths  # noqa: E402


def main() -> int:
    user = paths.config_dir() / "paths.yaml"
    if not user.exists():
        print(f"No {user}.\n"
              f"  cp config/paths.example.yaml config/paths.yaml\n"
              f"  then edit it and run this again.")
        return 1
    report = paths.describe()
    print(report)
    return 0 if "0 not resolvable" in report.splitlines()[0] else 2


if __name__ == "__main__":
    raise SystemExit(main())
