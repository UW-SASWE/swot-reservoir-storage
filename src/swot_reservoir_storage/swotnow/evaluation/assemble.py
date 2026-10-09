"""Assembling the SWOTNOW evaluation tables reported in the paper.

Two measures, because either alone misleads.

**Anomaly skill** asks whether the estimate reproduces the storage *level* as it departs from
the reservoir's own mean. It comes from the gauge-free scoring in `gauge_free.py`, already on a
common basis across arms.

**Change skill** asks whether the estimate reproduces the *movement* of storage over a given
number of days. This is where the two arms separate, and it is why the anchor's strong anomaly
R2 cannot be reported on its own: a series that holds its last value between overpasses tracks a
slow level well while carrying no information about change at all. Most storage variance is slower
than three weeks, so a flat line scores well on the first measure and near-nothing on the second.

Both are pooled over the seeds in `training.seeds`, and the summary reports the spread across
them rather than one run's number.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from ...common import paths
from ...common.config import load as load_config, record as record_config

__all__ = ["kge2", "change_skill", "assemble"]

_CFG = load_config("swotnow")

SEEDS = tuple(_CFG.training.seeds)
HORIZONS = tuple(_CFG.evaluation.horizons_days)
MIN_DAYS_CHANGE = int(_CFG.evaluation.min_days_scored)


def kge2(r: float, alpha: float) -> float:
    """Two-component Kling-Gupta efficiency on a change series.

    NOT the standard three-term KGE, and not taken from the literature. The mean-ratio term is
    dropped because the mean of a change series is near zero, which makes that ratio unstable.
    Define it wherever it is reported rather than citing it.

    There is no natural zero: an estimate that never changes has alpha = 0 and, by the convention
    used here, r = 0, which scores about -0.41 rather than 0.
    """
    return 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2)


def change_skill(tag: str, runs_dir: Path | None = None) -> pd.DataFrame:
    """Skill of the estimated CHANGE over each horizon, per reservoir, for one run."""
    runs_dir = Path(runs_dir) if runs_dir else paths.get("runs_dir")
    R = runs_dir / f"swot_gru_{tag}"

    daily = pd.read_parquet(R / "predictions_daily.parquet")
    truth = pd.read_parquet(R / "predictions_leaveout.parquet")[
        ["date", "lake_id", "phase", "true_mcm"]]
    d = daily.merge(truth, on=["date", "lake_id", "phase"], how="inner")
    d["date"] = pd.to_datetime(d["date"])

    rows = []
    for lid, g in d.sort_values(["lake_id", "date"]).groupby("lake_id"):
        # The *_raw columns, NOT *_mcm. The trainer's daily writer applies clip(f, 0, 1)
        # unconditionally, ignoring the no-clip configuration the product ships with. Under
        # per-reservoir centring the series is negative about half the time, so clipping at zero
        # flattens every negative anomaly and changes the magnitudes (not the directions).
        t = g.true_mcm.values
        a = g.anchor_mcm_raw.values
        p = g.pred_mcm_raw.values

        for h in HORIZONS:
            dt = t[h:] - t[:-h]
            da = a[h:] - a[:-h]
            dp = p[h:] - p[:-h]
            ok = np.isfinite(dt) & np.isfinite(da) & np.isfinite(dp)
            if ok.sum() < MIN_DAYS_CHANGE or np.std(dt[ok]) < 1e-9:
                continue

            def stat(x):
                sd = np.std(x[ok])
                r = float(np.corrcoef(x[ok], dt[ok])[0, 1]) if sd > 1e-12 else 0.0
                alpha = float(sd / np.std(dt[ok]))
                return r, alpha, kge2(r, alpha)

            ar, aa, ak = stat(da)
            nr, na, nk = stat(dp)
            rows.append(dict(lake_id=str(lid), horizon_days=h,
                             anchor_r=ar, anchor_alpha=aa, anchor_kge2=ak,
                             swotnow_r=nr, swotnow_alpha=na, swotnow_kge2=nk,
                             # How often the anchor does not move at all between the two days.
                             # This is the carry-forward showing up directly in the metric.
                             frac_days_no_change=float(np.mean(np.abs(da[ok]) < 1e-9))))

    out = pd.DataFrame(rows)
    out["seed"] = int(tag.split("_s")[1])
    return out


def assemble(gauge_free_dir: Path, out_dir: Path, runs_dir: Path | None = None):
    """Build the three evaluation tables from per-seed gauge-free scorings and run outputs.

    `gauge_free_dir` holds one CSV per seed, named uvp_<run_tag>.csv, as written by
    `gauge_free.run`. Returns (anomaly, change, summary).
    """
    gauge_free_dir = Path(gauge_free_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tags = {s: _CFG.evaluation.run_tag.format(seed=s) for s in SEEDS}

    # ---- anomaly: renamed from the scorer's arm names to the paper's ---------------------
    # The scorer's 'ungauged' arm IS the product. Its 'gauged' arm is NOT carried into these tables:
    # the published models are trained gauge-free, so handing them a gauge-derived reference level
    # feeds them an input they were not fitted to, and scoring them with the gauge-free switch left on
    # makes the arm numerically identical to the product. Either way the row would be meaningless.
    frames = []
    for s in SEEDS:
        f = gauge_free_dir / f"uvp_{tags[s]}.csv"
        d = pd.read_csv(f, dtype={"lake_id": str})
        frames.append(pd.DataFrame(dict(
            lake_id=d.lake_id, seed=s, n_days=d.n, cap_mcm=d.cap_mcm,
            preswot_r2a=d.preswot_r2a,
            anchor_r2a=d.ungauged_anchor_r2a,
            swotnow_r2a=d.ungauged_r2a,
            swotnow_cc=d.ungauged_cc, swotnow_alpha=d.ungauged_amp,
            anchor_cc=d.ungauged_anchor_cc, anchor_alpha=d.ungauged_anchor_amp)))
    A = pd.concat(frames, ignore_index=True)
    A.to_csv(out_dir / "per_reservoir_anomaly.csv", index=False)

    C = pd.concat([change_skill(tags[s], runs_dir) for s in SEEDS], ignore_index=True)
    C.to_csv(out_dir / "per_reservoir_change.csv", index=False)

    rows = []
    for arm, col in (("pre-SWOT (no SWOT)", "preswot_r2a"),
                     ("SWOT anchor alone", "anchor_r2a"),
                     ("SWOTNOW", "swotnow_r2a")):
        per = A.groupby("seed")[col].median()
        rows.append(dict(table="anomaly", arm=arm, metric="anomaly_R2",
                         median=round(float(per.mean()), 3),
                         seed_min=round(float(per.min()), 3),
                         seed_max=round(float(per.max()), 3)))

    # Paired within seed, then summarised across seeds. Pairing matters: reservoirs differ far
    # more from each other than the arms differ on any one reservoir, so an unpaired comparison
    # of medians would be dominated by which reservoirs happened to be scoreable.
    for label, other in (("vs pre-SWOT", "preswot_r2a"), ("vs anchor", "anchor_r2a")):
        deltas, pvals, wins = [], [], []
        for _, g in A.groupby("seed"):
            d = (g.swotnow_r2a - g[other]).dropna()
            deltas.append(d.median())
            pvals.append(stats.wilcoxon(d)[1])
            wins.append(100 * (d > 0).mean())
        rows.append(dict(table="anomaly_paired", arm=f"SWOTNOW {label}",
                         metric="delta_anomaly_R2",
                         median=round(float(np.mean(deltas)), 3),
                         seed_min=round(float(np.min(deltas)), 3),
                         seed_max=round(float(np.max(deltas)), 3),
                         win_pct=round(float(np.mean(wins)), 0),
                         p_max=float(np.max(pvals))))

    for h in HORIZONS:
        sub = C[C.horizon_days == h]
        per_a = sub.groupby("seed").anchor_kge2.median()
        per_n = sub.groupby("seed").swotnow_kge2.median()
        deltas, pvals = [], []
        for _, g in sub.groupby("seed"):
            d = (g.swotnow_kge2 - g.anchor_kge2).dropna()
            deltas.append(d.median())
            pvals.append(stats.wilcoxon(d)[1])
        rows.append(dict(table="change", arm=f"h={h}d", metric="dS_KGE2",
                         median=round(float(per_n.mean()), 3),
                         seed_min=round(float(per_n.min()), 3),
                         seed_max=round(float(per_n.max()), 3),
                         anchor=round(float(per_a.mean()), 3),
                         delta_min=round(float(np.min(deltas)), 3),
                         delta_max=round(float(np.max(deltas)), 3),
                         p_max=float(np.max(pvals))))

    S = pd.DataFrame(rows)
    S.to_csv(out_dir / "summary.csv", index=False)
    record_config(out_dir, _CFG, extra={"seeds": list(SEEDS),
                                        "n_anomaly_rows": len(A), "n_change_rows": len(C)})
    return A, C, S
