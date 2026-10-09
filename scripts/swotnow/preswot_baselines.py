"""Score the two pre-SWOT relations (TMS-OS area through the SRTM-derived and the GRS-implied relation)
on the held-out reservoirs.

    python scripts/swotnow/preswot_baselines.py [--out FILE]
"""
import argparse
from pathlib import Path

from swot_reservoir_storage.common import paths
from swot_reservoir_storage.swotnow.evaluation.preswot_baselines import run

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=paths.get("work_dir", must_exist=False) / "swotnow_eval" / "preswot_baselines.csv")
    run(out_csv=ap.parse_args().out)
