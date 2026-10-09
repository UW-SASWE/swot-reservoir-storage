"""Assemble the SWOTNOW evaluation tables from the per-seed scorings.

    python scripts/swotnow/assemble_tables.py [--scored DIR] [--out DIR]

`--scored` is the directory written by `score.py`. Writes `per_reservoir_anomaly.csv`,
`per_reservoir_change.csv` and `summary.csv` to `--out`.
"""
import argparse
from pathlib import Path

from swot_reservoir_storage.common import paths
from swot_reservoir_storage.swotnow.evaluation.assemble import assemble

if __name__ == "__main__":
    work = paths.get("work_dir", must_exist=False) / "swotnow_eval"
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", type=Path, default=work / "gauge_free")
    ap.add_argument("--out", type=Path, default=work / "tables")
    args = ap.parse_args()
    assemble(args.scored, args.out, paths.get("runs_dir"))
