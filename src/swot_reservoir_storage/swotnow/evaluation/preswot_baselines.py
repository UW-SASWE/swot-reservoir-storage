"""The pre-SWOT baseline, scored on the same basis as SWOTNOW, through both available relations.

The comparison SWOTNOW is measured against has to be the strongest pre-SWOT method available,
not the most convenient one. The operational default converts TMS-OS optical area through a
relation derived from SRTM, and that is what the main evaluation uses as arm D. But a second
relation exists, implied by the Global Reservoir Storage product's own paired area and storage
columns, and it scores higher on the comparison set. Reporting SWOTNOW's gain only against the
weaker relation would overstate it.

Two arms, neither using SWOT and neither using a gauge:

    srtm   TMS-OS area -> SRTM-derived relation      the operational default
    grs    TMS-OS area -> GRS-implied relation       the stronger alternative

Both are scored as anomalies against the daily gauge, on the days each arm is defined, and the
headline comparison is restricted to reservoirs where both exist so it is like-for-like.

Correlation and amplitude are reported alongside anomaly R2 because the three are related by
R2 = 2*r*alpha - alpha^2: a single coefficient near -0.7 can come from poor timing, from wrong
magnitude, or from both, and those have different implications for what a pre-SWOT method is
actually failing at.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from ...common import paths
from ...common.config import load as load_config, record as record_config

__all__ = ["run", "srtm_relation", "grs_relation", "tmsos_area", "score_anomaly"]

_CFG = load_config("swotnow")

# This scorer requires 60 days where the gauge-free scorer requires 30. The thresholds are
# deliberately different and neither is a typo: these series are sampled on TMS-OS observation
# dates, which are far sparser than the daily grid the other scorer works on, so a low bar here
# would admit reservoirs whose metrics rest on a handful of observations. Both thresholds decide
# which reservoirs enter the population and therefore move every median in the table.
MIN_DAYS = 60


def tmsos_area(lake_id: str):
    """TMS-OS surface area in km2, from the first configured source that has this reservoir."""
    for key in ("tmsos_area_primary", "tmsos_area"):
        root = paths.get_optional(key)
        if root is None:
            continue
        src = root / f"{lake_id}.csv"
        if not src.exists():
            continue
        d = pd.read_csv(src, parse_dates=["date"])
        if "area" not in d:
            continue
        d = d.dropna(subset=["area"]).sort_values("date")
        if d.empty:
            continue
        # square metres in some files, square kilometres in others
        a = d["area"] / 1e6 if d["area"].max() > 1e4 else d["area"]
        return pd.Series(a.values, index=pd.DatetimeIndex(d["date"].values))
    return None


def srtm_relation(lake_id: str):
    """Area -> storage from the SRTM-derived relation, storage re-integrated as INT A dh.

    The `Storage` column in these files is the integral of h dA, a different quantity; storage is
    recomputed here from area and elevation by the trapezoid rule.

    The result is then sorted by AREA and made non-decreasing. Sorting by elevation is not enough:
    the interpolation this feeds is indexed by area, and a relation that is not monotone in area
    would let a larger surface map to a smaller volume.
    """
    p = paths.get("srtm_aec") / f"{lake_id}.csv"
    if not p.exists():
        return None
    rows = []
    for line in open(p):
        if line.startswith("#") or line.startswith("CumArea"):
            continue
        q = line.strip().split(",")
        if len(q) >= 4 and q[0]:
            try:
                rows.append((float(q[0]), float(q[1])))
            except ValueError:
                pass
    a = pd.DataFrame(rows, columns=["area", "elev"]).dropna().sort_values("elev")
    if len(a) < 5:
        return None
    A = a.area.values
    e = a.elev.values
    S = np.concatenate([[0.0], np.cumsum(0.5 * (A[1:] + A[:-1]) * np.diff(e))])
    o = np.argsort(A)
    return A[o], np.maximum.accumulate(S[o])


def grs_relation(grand_id):
    """Area -> storage implied by the GRS product's own paired area and storage columns.

    GRS publishes a time series, not a relation, so one is derived by binning its area range into
    24 quantile bins and taking the median storage in each. The median rather than the mean
    because the series carries occasional extreme values, and quantile bins rather than equal
    width because reservoir areas are strongly skewed and equal-width bins would leave most of
    them empty.
    """
    root = paths.get_optional("grs")
    if root is None:
        return None
    f = root / f"{int(grand_id)}.csv"
    if not f.exists():
        return None
    try:
        d = pd.read_csv(f)
        a = pd.to_numeric(d["Area(km2)"], errors="coerce")
        st = pd.to_numeric(d["Storage(km3)"], errors="coerce") * 1e3      # km3 -> million m3
    except Exception:
        return None
    ok = a.notna() & st.notna() & (a > 0)
    if ok.sum() < 24:
        return None
    a, st = a[ok].values, st[ok].values
    o = np.argsort(a)
    a, st = a[o], st[o]
    edges = np.percentile(a, np.linspace(0, 100, 25))
    storages, centres = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (a >= lo) & (a <= hi)
        if m.any():
            storages.append(np.median(st[m]))
            centres.append(0.5 * (lo + hi))
    if len(storages) < 5:
        return None
    return np.array(centres), np.maximum.accumulate(np.array(storages))


def score_anomaly(pred, true) -> dict:
    """Anomaly skill: n, R2, correlation, amplitude. Both series centred on their own means."""
    ok = np.isfinite(pred) & np.isfinite(true)
    if ok.sum() < MIN_DAYS:
        return dict(n=int(ok.sum()), r2a=np.nan, cc=np.nan, amp=np.nan)
    p = pred[ok] - pred[ok].mean()
    t = true[ok] - true[ok].mean()
    if np.std(t) == 0 or np.std(p) == 0:
        return dict(n=int(ok.sum()), r2a=np.nan, cc=np.nan, amp=np.nan)
    return dict(n=int(ok.sum()),
                r2a=float(1 - np.sum((p - t) ** 2) / np.sum(t ** 2)),
                cc=float(np.corrcoef(p, t)[0, 1]),
                amp=float(np.std(p) / np.std(t)))


def run(run_tag: str | None = None, out_csv: Path | None = None) -> pd.DataFrame:
    """Score both pre-SWOT relations on the held-out reservoirs. One row per reservoir."""
    run_tag = run_tag or _CFG.evaluation.run_tag.format(seed=_CFG.training.seeds[0])
    run_dir = paths.get("runs_dir") / f"swot_gru_{run_tag}"

    universe = (pd.read_csv(paths.get("global_universe"), dtype={"lake_id": str},
                            low_memory=False)
                .drop_duplicates("lake_id").set_index("lake_id"))
    leaveout = pd.read_parquet(run_dir / "predictions_leaveout.parquet")[
        ["date", "lake_id", "phase", "true_mcm"]]
    leaveout["lake_id"] = leaveout.lake_id.astype(str)
    leaveout["date"] = pd.to_datetime(leaveout.date)

    rows = []
    for lid, g in leaveout.groupby("lake_id"):
        g = g.sort_values("date")
        dates = pd.DatetimeIndex(g.date)
        y = g.true_mcm.values.astype(float)

        area = tmsos_area(lid)
        if area is None or len(area) < 10:
            continue
        area = area[~area.index.duplicated()].sort_index()

        grand_id = universe.GRAND_ID.get(lid)
        row = dict(lake_id=lid)
        for tag, relation in (("srtm", srtm_relation(lid)),
                              ("grs", grs_relation(grand_id) if pd.notna(grand_id) else None)):
            if relation is None:
                continue
            areas, storages = relation
            st = pd.Series(np.interp(area.values, areas, storages), index=area.index)
            # carried forward onto the gauge's dates, as every other sparse arm is
            ser = st.reindex(st.index.union(dates)).ffill().reindex(dates).values
            for k, v in score_anomaly(ser, y).items():
                row[f"{tag}_{k}"] = v
        rows.append(row)

    res = pd.DataFrame(rows)
    if out_csv:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        res.to_csv(out_csv, index=False)
        record_config(out_csv.parent, _CFG, extra={"run_tag": run_tag, "n_reservoirs": len(res)})
    return res


def summarise(res: pd.DataFrame) -> str:
    """The like-for-like comparison, on reservoirs where both relations exist."""
    both = res.dropna(subset=["srtm_r2a", "grs_r2a"])
    lines = ["PRE-SWOT BASELINES, gauge-free anomalies, like-for-like panel",
             f"{'curve':22s} {'anomaly R2':>11s} {'corr':>7s} {'corr^2':>7s} {'amplitude':>10s}"]
    for tag, label in (("srtm", "TMS-OS + SRTM"), ("grs", "TMS-OS + GRS")):
        lines.append(f"{label:22s} {both[f'{tag}_r2a'].median():11.3f} "
                     f"{both[f'{tag}_cc'].median():7.3f} "
                     f"{both[f'{tag}_cc'].median()**2:7.3f} "
                     f"{both[f'{tag}_amp'].median():10.3f}")
    d = (both.grs_r2a - both.srtm_r2a).dropna()
    lines.append(f"\nGRS - SRTM: median {d.median():+.3f}, GRS better on "
                 f"{100*(d>0).mean():.0f}%, p={stats.wilcoxon(d)[1]:.2g}  (n={len(d)})")
    return "\n".join(lines)
