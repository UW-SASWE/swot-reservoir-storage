"""Check that the preprocessing rebuilds the feature table and static attributes from raw observations.

    python tests/check_preprocessing.py

Needs the data of `config/paths.yaml`, including an existing `features` table and `static_attributes`
table to compare with. It exits non-zero on any difference.

1. `swotnow.preprocessing.features` against `features`, on a random sample of reservoirs, for every
   column that does not come from a gauge (the table's gauge-derived target columns are not built).
2. `swotnow.preprocessing.static_attributes` against `static_attributes`, for every reservoir.
"""
import sys

import numpy as np
import pandas as pd

from swot_reservoir_storage.common import paths
from swot_reservoir_storage.swotnow.preprocessing import features as F
from swot_reservoir_storage.swotnow.preprocessing import static_attributes as S

N_SAMPLE = 60
failures = []

ship = pd.read_parquet(paths.get("features"))
ship["lake_id"] = ship["lake_id"].astype(str)
rng = np.random.default_rng(0)
sample = sorted(rng.choice(sorted(ship.lake_id.unique()), N_SAMPLE, replace=False))
cap = {l: float(ship.loc[ship.lake_id == l, "cap_mcm_eff"].iloc[0]) for l in sample}
new = F.build(sample, ship["date"].max(), capacity=cap, progress_every=0)
cols = [c for c in new.columns if c in ship.columns]
for lid in sample:
    a = new[new.lake_id == lid].reset_index(drop=True)
    b = ship[ship.lake_id == lid].sort_values("date").reset_index(drop=True)
    if len(a) != len(b):
        failures.append((lid, "row count"))
        continue
    for c in cols:
        same = ((a[c] == b[c]).all() if c in ("date", "lake_id") else
                np.allclose(a[c].astype(float), b[c].astype(float), equal_nan=True, rtol=0, atol=1e-9))
        if not same:
            failures.append((lid, c))
print(f"features: {len(sample)} reservoirs, {len(cols)} columns, {sum(1 for f in failures if f[1] != 'static')} differences")

att = pd.read_parquet(paths.get("static_attributes"))
att["lake_id"] = att["lake_id"].astype(str)
rebuilt = S.build(sorted(att.lake_id)).set_index("lake_id")
ref = att.set_index("lake_id").loc[rebuilt.index]
n0 = len(failures)
for c in ref.columns:
    if c not in rebuilt.columns:
        failures.append(("static", c))
    elif not np.allclose(rebuilt[c].astype(float), ref[c].astype(float), equal_nan=True, rtol=0, atol=1e-9):
        failures.append(("static", c))
print(f"static attributes: {len(rebuilt)} reservoirs, {len(ref.columns)} columns, {len(failures) - n0} differences")

if failures:
    print("FAILED:", failures[:10])
    sys.exit(1)
print("ALL CHECKS PASSED")
