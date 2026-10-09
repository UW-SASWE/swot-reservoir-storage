"""The gauge-free evaluation: what survives when a reservoir has no in-situ record.

This is the scorer behind the paper's present-day numbers. A gauged reservoir would score
well partly *because* it is gauged (the anchor's reference level could come from the gauge), which
overstates what the method delivers on the ungauged majority. Here the gauge is withheld from
the estimate and used only as the reference to score against.

Four arms, the same reservoirs, the same dates:

    A  gauged            the anchor's mean fill fraction comes from the gauge
    B  ungauged          that value is replaced by the nominal constant an ungauged reservoir
                         gets, which is what deployment actually supplies
    C  ungauged anchor   arm B without the network's correction: the physics alone
    D  pre-SWOT          TMS-OS optical area through an SRTM-derived relation. No SWOT, and no
                         gauge either -- the relation is a DEM product and the area is optical --
                         which is what makes it the right comparator for B

Everything is scored as an anomaly, each series minus its own mean over the dates compared. That
is the only quantity an ungauged reservoir can claim: without a gauge nothing ties an estimate to
a datum. Nothing is clipped to [0, 1] anywhere, because a clip presumes the absolute fill fraction
that does not exist here.

Arm D is sparse, observed only on TMS-OS dates, so it is carried forward onto the same daily grid
the SWOT anchor is carried onto. Arms A-C are then ALSO re-scored on D's own support, so the
B-against-D comparison is on identical dates rather than merely identical reservoirs. Those are
the `p_*` columns, and they are the ones to use for that comparison.

Arm A is a DIAGNOSTIC and is not reported in the paper. The published models are trained gauge-free,
so arm A hands the network a gauge-derived reference level it was never fitted to. Arms B-D are the
results.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ...common import paths
from ...common.config import load as load_config, record as record_config
from ..model import trainer as T
from ..model.normalisation import build_norm_and_pool, load_routed_precip

__all__ = ["run", "score_anomaly", "rebuild_anchor", "preswot_storage"]

_CFG = load_config("swotnow")

# A reservoir needs at least this many scoreable days for a metric to mean anything. This is the
# scorer's own threshold and it is deliberately NOT common.metrics._MIN_N: these are daily series
# over the SWOT era, not the monthly series the shared metrics were written for. Changing it
# changes which reservoirs enter the population, which changes every median in the table.
MIN_DAYS = 30


# Normalisation and features are built under the TRAINING configuration. The models are trained
# gauge-free, so rebuilding the normalisation with a gauge-derived reference level (typically far from
# the deployment constant 0.01) would mis-scale the network's anchor input without raising an error.
# Arm A is therefore built from the gauge directly (see `run`).

# ---------------------------------------------------------------------------------------------
# Arm D: the pre-SWOT estimate. No SWOT observation and no gauge enters this path.
# ---------------------------------------------------------------------------------------------

def load_relation(lake_id: str):
    """The SRTM-derived area-elevation relation, as area -> cumulative storage.

    Storage is integrated as the integral of A dh by the trapezoid rule. The `Storage` column in
    these files is not that integral (it is the integral of h dA, the complementary prism); it is read
    and discarded here.
    """
    rows = []
    for line in open(paths.get("srtm_aec") / f"{lake_id}.csv"):
        if line.startswith("#") or line.startswith("CumArea"):
            continue
        q = line.strip().split(",")
        if len(q) >= 4 and q[0]:
            try:
                rows.append((float(q[0]), float(q[1]), float(q[3])))
            except ValueError:
                pass
    a = pd.DataFrame(rows, columns=["area_km2", "elev_m", "storage_published"])
    a = a.dropna(subset=["area_km2", "elev_m"]).sort_values("elev_m")
    if len(a) < 3:
        return np.array([]), np.array([])
    A = a["area_km2"].values.astype(float)
    e = a["elev_m"].values.astype(float)
    dV = 0.5 * (A[1:] + A[:-1]) * np.diff(e)
    return A, np.concatenate([[0.0], np.cumsum(dV)])


def load_tmsos_area(lake_id: str):
    """TMS-OS surface area for one reservoir, from the first source that has it."""
    for key in ("tmsos_area_primary", "tmsos_area"):
        root = paths.get_optional(key)
        if root is None:
            continue
        src = root / f"{lake_id}.csv"
        if not src.exists():
            continue
        df = pd.read_csv(src, parse_dates=["date"])[["date", "area"]].dropna()
        df = df.sort_values("date")
        if df.empty:
            continue
        # Units are square metres in some files and square kilometres in others. No reservoir in
        # this study exceeds 10,000 km2, so a maximum above that identifies square metres.
        df["area_km2"] = df["area"] / 1e6 if df["area"].max() > 1e4 else df["area"]
        return df.set_index("date")["area_km2"]
    return None


def preswot_storage(lake_id: str):
    """Storage from optical area through the DEM relation. Returns None where either is absent."""
    if not (paths.get("srtm_aec") / f"{lake_id}.csv").exists():
        return None
    area = load_tmsos_area(lake_id)
    if area is None:
        return None
    areas, storages = load_relation(lake_id)
    if len(areas) == 0:
        return None
    return pd.Series(np.interp(area.values, areas, storages), index=area.index)


# ---------------------------------------------------------------------------------------------
# The anchor, rebuilt with the reference level supplied rather than derived
# ---------------------------------------------------------------------------------------------

def rebuild_anchor(sub: pd.DataFrame, cap: float, mean_ff: float, screen: str | None = None):
    """The trainer's anchor, with its mean fill fraction supplied by the caller.

    This is what lets a gauged reservoir be scored as though it were ungauged: the only thing
    the gauge contributes to the anchor is that constant, so replacing it reproduces exactly the
    information an ungauged reservoir has.

    The pass-level screen must be applied here exactly as the trainer applies it; otherwise the model
    is scored on an anchor it was not trained on. If you change the trainer's screening, change it here too.
    """
    wse_values = sub["wse"].dropna()
    wse_ref = float(np.median(wse_values.values))
    sa_ref = float(sub["sa_smooth"].dropna().median() if sub["sa_smooth"].notna().any() else 1.0)

    anchor = np.full(len(sub), np.nan)
    for i in range(len(sub)):
        if sub.loc[i, "on_swot_pass"] == 1 and np.isfinite(sub.loc[i].get("wse", np.nan)):
            sa = (sub.loc[i, "sa_smooth"]
                  if np.isfinite(sub.loc[i].get("sa_smooth", np.nan)) else sa_ref)
            anchor[i] = (sub.loc[i, "wse"] - wse_ref) * sa / cap + mean_ff

    mode = T.ANCHOR_SCREEN if screen is None else screen
    if mode.startswith("causal_"):
        # build_daily already applied the causal screen to `anchor_ff` under the training configuration,
        # with the nominal offset; shift it to the requested offset. The increment gate is invariant to a
        # constant offset, so this is the screened series for that offset (the only approximation is the
        # trainer's clip to [-0.5, 2], which these anomalies do not reach).
        if not T.GAUGEFREE:
            raise RuntimeError("causal screens are scored through build_daily; that needs "
                               "inputs.use_gauge: false")
        return sub["anchor_ff"].values - T.UNGAUGED_MEAN_FF + mean_ff
    if mode in ("hampel", "hampel_interp"):
        idx = np.flatnonzero(np.isfinite(anchor))
        if len(idx) >= 10:
            if mode == "hampel":
                anchor[idx] = T._hampel(anchor[idx], T.ANCHOR_SCREEN_K, T.ANCHOR_SCREEN_N)
            else:
                anchor[idx] = T._hampel_interp(anchor[idx], idx,
                                               T.ANCHOR_SCREEN_K, T.ANCHOR_SCREEN_N)
    return pd.Series(anchor).ffill().fillna(mean_ff).clip(-0.5, 2.0).values


def score_anomaly(pred, truth) -> dict:
    """Anomaly skill of one series against another: n, R2, correlation, RMSE, amplitude.

    Both series are centred on their own means over the days where both are finite, so a constant
    offset -- which is all a missing datum amounts to -- costs nothing. Returns NaN metrics rather
    than raising when there is too little to score, so a thin reservoir drops out of the population
    instead of stopping the run.

    Note that `r2a` divides by the reference's own variance. It is a ratio, not an error: a
    reservoir that barely moves yields a large negative value for any estimate of ordinary
    magnitude. Filter on variability before taking a population minimum.
    """
    p = np.asarray(pred, float)
    t = np.asarray(truth, float)
    ok = np.isfinite(p) & np.isfinite(t)
    p, t = p[ok], t[ok]
    if len(t) < MIN_DAYS or np.std(t) == 0 or np.std(p) == 0:
        return dict(n=len(t), r2a=np.nan, cc=np.nan, rmse=np.nan, amp=np.nan)
    p = p - p.mean()
    t = t - t.mean()
    return dict(n=len(t),
                r2a=1 - np.sum((p - t) ** 2) / np.sum(t ** 2),
                cc=float(np.corrcoef(p, t)[0, 1]),
                rmse=float(np.sqrt(np.mean((p - t) ** 2))),
                amp=float(np.std(p) / np.std(t)))


# ---------------------------------------------------------------------------------------------

ARMS = ("gauged", "ungauged", "ungauged_anchor", "preswot")


def run(model_dir: Path | None = None, out_csv: Path | None = None,
        progress_every: int = 25) -> pd.DataFrame:
    """Score all four arms on the held-out gauged reservoirs. Returns one row per reservoir."""
    model_dir = Path(model_dir) if model_dir else paths.get("swotnow_model")
    ungauged_mean_ff = T.UNGAUGED_MEAN_FF

    # The offset setting must hold for the NORMALISATION as well as for each reservoir's
    # features. build_norm_and_pool calls build_daily over the whole training pool, and
    # anchor_ff is the first dynamic feature, so the setting moves that feature's pool mean by
    # by roughly the difference between a typical fill fraction and the nominal 0.01. Normalising
    # with one setting and building features with the other feeds the network inputs on a scale
    # it was not fitted to, which raises no error.
    if not T.GAUGEFREE:
        raise RuntimeError("inputs.use_gauge is true but the published models are trained gauge-free; "
                           "scoring them this way mis-normalises the network (see the note above).")
    print("reconstructing training-pool normalisation ...", flush=True)
    df, attr_lookup, dyn_mean, dyn_std = build_norm_and_pool()

    norm = np.load(model_dir / "attr_norm.npz", allow_pickle=True)
    attr_mean = norm["mean"].astype(np.float32)
    attr_std = norm["std"].astype(np.float32)
    model = T.ScalableGRU(T.N_DYN, T.N_ATTR, hidden=_CFG.model.hidden, attr_dim=T.ATTR_DIM)
    model.load_state_dict(torch.load(model_dir / "model.pt", map_location="cpu"))
    model.eval()

    splits = pd.read_csv(T.SPLITS_CSV, dtype={"lake_id": str}).dropna(subset=["lake_id"])
    held = {l for l, s in zip(splits.lake_id, splits.split) if s in ("val", "test")}
    target = sorted(held & set(df.lake_id.unique()) & set(attr_lookup.index))
    print(f"held-out gauged reservoirs to score: {len(target)}  (model {model_dir.name})",
          flush=True)

    rows = []
    for k, lid in enumerate(target, 1):
        sub = df[df.lake_id == lid]
        d = T.build_daily(sub, load_routed_precip(lid))
        if d is None:
            continue
        d = d.reset_index(drop=True)
        cap = float(d["_cap"].iloc[0])
        truth = d["storage_mcm"].values
        # Arm A's gauge-derived reference level, computed as the trainer computes it when it is not
        # gauge-free. It is a diagnostic: the network was never trained on it.
        _stor = pd.Series(truth).dropna()
        mean_ff_gauged = float(np.clip((_stor / cap).mean(), 0.01, 0.99)) if len(_stor) >= 5 else ungauged_mean_ff
        av = np.clip((attr_lookup.loc[lid].values.astype(np.float32) - attr_mean) / attr_std,
                     -5, 5)

        out = {}
        for arm, mff in (("gauged", mean_ff_gauged), ("ungauged", ungauged_mean_ff)):
            dd = d.copy()
            dd["anchor_ff"] = rebuild_anchor(dd, cap, mff)
            dv = np.clip((dd[T.DYN_FEATS].values - dyn_mean) / dyn_std, -5, 5).astype(np.float32)
            long_win = [dv[t - T.LONG_WIN:t] for t in range(T.LONG_WIN, len(dd))]
            short_win = [dv[t - T.SHORT_WIN:t] for t in range(T.LONG_WIN, len(dd))]
            if not long_win:
                break
            with torch.no_grad():
                r = model(torch.tensor(np.stack(long_win), dtype=torch.float32),
                          torch.tensor(np.stack(short_win), dtype=torch.float32),
                          torch.tensor(np.tile(av, (len(long_win), 1)),
                                       dtype=torch.float32)).numpy()
            anch = dd["anchor_ff"].values[T.LONG_WIN:]
            out[arm] = (anch + r) * cap
            if arm == "ungauged":
                out["ungauged_anchor"] = anch * cap
        if len(out) < 3:
            continue

        dates = pd.to_datetime(d["date"].values)[T.LONG_WIN:]
        try:
            pre = preswot_storage(lid)
        except Exception:
            pre = None
        if pre is None or len(pre) < 5:
            out["preswot"] = np.full(len(dates), np.nan)
            pre_support = np.zeros(len(dates), bool)
        else:
            pre = pre[~pre.index.duplicated()].sort_index()
            # carried forward onto the daily grid exactly as the SWOT anchor is carried
            ser = pre.reindex(pre.index.union(dates)).ffill().reindex(dates)
            out["preswot"] = ser.values
            pre_support = np.asarray(dates >= pre.index.min())

        y = truth[T.LONG_WIN:]
        ok = np.isfinite(y)
        if ok.sum() < MIN_DAYS:
            continue

        row = dict(lake_id=lid, n=int(ok.sum()), cap_mcm=cap, mean_ff_gauged=mean_ff_gauged)
        for arm in ARMS:
            for m, v in score_anomaly(out[arm][ok], y[ok]).items():
                row[f"{arm}_{m}"] = v
        # paired re-score on arm D's own support, so B against D is on identical dates
        both = ok & pre_support & np.isfinite(out["preswot"])
        if both.sum() >= MIN_DAYS:
            for arm in ARMS:
                for m, v in score_anomaly(out[arm][both], y[both]).items():
                    row[f"p_{arm}_{m}"] = v
        rows.append(row)
        if progress_every and k % progress_every == 0:
            print(f"  {k}/{len(target)}", flush=True)

    res = pd.DataFrame(rows)
    if out_csv:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        res.to_csv(out_csv, index=False)
        record_config(out_csv.parent, _CFG, extra={"model_dir": str(model_dir),
                                                   "n_reservoirs": len(res)})
        print(f"\nscored {len(res)} reservoirs -> {out_csv}")
    return res
