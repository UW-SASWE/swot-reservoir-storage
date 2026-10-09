"""Fit the per-reservoir storage error model (random and systematic error, with and without SWOT).

    python scripts/global_uncertainty/fit_error_model.py

Writes `<work_dir>/global_uncertainty/per_reservoir_error_model.csv`. Settings are in
`config/global_uncertainty.yaml`. Run before `aggregate_global_totals.py`.
"""
from swot_reservoir_storage.common import paths
from swot_reservoir_storage.global_uncertainty.calibration.error_model import OUT, main

if __name__ == "__main__":
    main(apply_univ=paths.get("global_universe"), out_path=OUT,
         measured_path=paths.get("smoother_measured_summary"))
