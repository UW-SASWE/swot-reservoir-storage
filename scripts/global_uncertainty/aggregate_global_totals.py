"""Aggregate the per-reservoir error model to global and regional storage uncertainty.

    python scripts/global_uncertainty/aggregate_global_totals.py

Reads `per_reservoir_error_model.csv` from `fit_error_model.py` and writes `global_totals.csv` and
`continental_breakdown.csv` to `<work_dir>/global_uncertainty/`, under both correlation bounds.
"""
from swot_reservoir_storage.global_uncertainty.propagation.global_totals import main

if __name__ == "__main__":
    main()
