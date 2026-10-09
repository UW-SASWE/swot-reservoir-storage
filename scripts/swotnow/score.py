"""Score every trained seed gauge-free on the held-out gauged reservoirs.

    python scripts/swotnow/score.py [--out DIR]

Reads `<runs_dir>/swot_gru_<run_tag>/` for each seed in `training.seeds` and writes
`DIR/uvp_<run_tag>.csv` (default DIR: `<work_dir>/swotnow_eval/gauge_free`).
"""
import argparse
from pathlib import Path

from swot_reservoir_storage.common import paths
from swot_reservoir_storage.common.config import load
from swot_reservoir_storage.swotnow.evaluation import gauge_free

if __name__ == "__main__":
    cfg = load("swotnow")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=paths.get("work_dir", must_exist=False) / "swotnow_eval" / "gauge_free")
    args = ap.parse_args()
    for seed in cfg.training.seeds:
        tag = cfg.evaluation.run_tag.format(seed=seed)
        gauge_free.run(model_dir=paths.get("runs_dir") / f"swot_gru_{tag}", out_csv=args.out / f"uvp_{tag}.csv")
