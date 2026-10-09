"""Estimate reservoir storage anomaly with the trained SWOTNOW model.

    python scripts/swotnow/predict.py --lake-ids 7830181213 4520053003 --out estimates.csv
    python scripts/swotnow/predict.py --lake-ids-file ids.txt --end 2026-08-31 --out estimates.parquet

Reservoirs are named by their SWOT Prior Lake Database `lake_id`. Inputs are read from `config/paths.yaml`
(SWOT passes, optical area, precipitation, the area-elevation relation, reservoir inventories).
`--model-dir` holds `model.pt`, `attr_norm.npz` and `dyn_norm.npz`; the default is `swotnow_model`.
`--capacity-csv` (columns lake_id, cap_mcm) overrides the catalogued capacity.

The output is a storage anomaly in million m3, on an arbitrary offset: subtract each reservoir's own
mean over the period you analyse. See `swotnow/inference.py` for the columns.
"""
import argparse
from pathlib import Path

import pandas as pd

from swot_reservoir_storage.common import paths
from swot_reservoir_storage.common.config import load
from swot_reservoir_storage.swotnow import inference

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lake-ids", nargs="*", default=[])
    ap.add_argument("--lake-ids-file", type=Path, help="one lake_id per line")
    ap.add_argument("--end", default=str(load("swotnow").period.end))
    ap.add_argument("--model-dir", type=Path, default=None)
    ap.add_argument("--capacity-csv", type=Path)
    ap.add_argument("--out", type=Path, required=True, help=".csv or .parquet")
    a = ap.parse_args()
    ids = list(a.lake_ids)
    if a.lake_ids_file:
        ids += [ln.strip() for ln in a.lake_ids_file.read_text().splitlines() if ln.strip()]
    if not ids:
        ap.error("give --lake-ids or --lake-ids-file")
    cap = None
    if a.capacity_csv:
        c = pd.read_csv(a.capacity_csv, dtype={"lake_id": str})
        cap = dict(zip(c.lake_id, c.cap_mcm))
    est = inference.predict(ids, a.end, a.model_dir, cap)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    est.to_parquet(a.out, index=False) if a.out.suffix == ".parquet" else est.to_csv(a.out, index=False)
    print(f"{est.lake_id.nunique()} reservoirs, {len(est):,} days -> {a.out}")
