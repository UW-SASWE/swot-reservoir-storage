#!/usr/bin/env python3
"""SWOTNOW: the SWOT-anchored GRU that estimates present-day reservoir storage.

The model does not predict storage directly. It predicts a *correction* to a physical
estimate, the anchor, which converts a SWOT water-surface elevation into a storage value:

    S_anchor = S_ref + (h - h_ref) * A                          (Methods, Eq. 1)

evaluated on each SWOT pass and held constant until the next one. The network adds a residual
to that, and the residual is bounded to [-1, 1] by a final tanh, so the estimate cannot wander
arbitrarily far from the physics it is built on.

    fill_fraction(t) = anchor_ff(t) + residual(t)
    storage(t)       = fill_fraction(t) * capacity

The fill fraction is not clipped to [0, 1] in the published configuration: every series is
centred on its own mean, which puts about half the record below zero. See OUTPUTS.

ARCHITECTURE
    long window  (past 180 d)  -> GRU          -> h_long
    short window (past 10 d)   -> GRU          -> h_short
    static attributes (23)     -> encoder MLP  -> h_attr
    concat -> MLP head, tanh   -> residual in [-1, 1]

    The attribute encoder maps observable traits (geometry, climate, relation shape) to a context
    vector, so a reservoir absent from training still gets one. This is what makes the model
    usable on ungauged reservoirs.

TRAINING POOL (leave-reservoirs-out)
    Two kinds of reservoir, both drawn from the `insitu_splits` table:

    - gauged        target = gauge storage, weight 1.0
    - ungauged      target = the anchor itself, weight W_PSEUDO (see config: training.
                    pseudo_weight). The purpose is regularisation, not supervision: it
                    penalises inventing dynamics where no measurement could contradict them,
                    while still exposing the attribute encoder to their static traits.

    Reservoirs in the 'val' and 'test' splits are held out entirely -- no training window, no
    target in any form, not even as an anchor target -- so the evaluation measures
    generalisation to reservoirs the model has never seen.

    `insitu_splits` lists candidates; the pool is what survives the data requirements in
    build_daily. Print the counts from a run.

INPUTS   all resolved through common.paths; see config/paths.example.yaml
    features            daily multi-sensor table, one row per reservoir per day
    static_attributes   the 23 traits in ATTR_COLS (use the corrected-integral table; see
                        config/paths.example.yaml)
    precip              ERA5-Land daily precipitation, routed (optional; absent for some reservoirs)
    insitu_splits       train / val / test assignment
    causal_surface_area trailing-smoothed area, when anchor.area_smoother is 'causal'

CONFIGURATION
    config/swotnow.yaml; its committed values are the published configuration. Settings are not
    read from environment variables (see docs/design-notes.md).

OUTPUTS   under work_dir/runs/<tag>/
    model.pt, attr_norm.npz, metrics_leaveout.csv, predictions_leaveout.parquet,
    predictions_daily.parquet, run_config.json

    predictions_daily.parquet carries clipped (`pred_mcm`, `anchor_mcm`) and unclipped
    (`pred_mcm_raw`, `anchor_mcm_raw`) columns. The clipped ones apply clip(f, 0, 1) whatever the
    configuration says. Under the published configuration, score the *_raw columns.
"""

import argparse, warnings
import os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Inputs are located through common.paths (config/paths.yaml), not from this file's location.
from swot_reservoir_storage.common import paths as _paths
from swot_reservoir_storage.common.config import load as _load_config

_CFG = _load_config("swotnow")
from .causal_screens import causal_screen  # noqa: E402

FEAT       = _paths.get("features")
ATTR_FILE  = _paths.get("static_attributes")
SPLITS_CSV = _paths.get("insitu_splits")
PRECIP_DIR = _paths.get("precip")

LONG_WIN   = _CFG.model.long_window
SHORT_WIN  = _CFG.model.short_window
TAU        = _CFG.training.precip_routing_tau_days
SEED       = _CFG.training.seeds[0]
LR         = _CFG.training.learning_rate
EPOCHS     = _CFG.training.max_epochs
PATIENCE   = _CFG.training.patience
BATCH_SIZE = _CFG.training.batch_size
N_THREADS  = _CFG.training.n_threads
ATTR_DIM   = _CFG.model.attr_dim
W_PSEUDO   = _CFG.training.pseudo_weight
MIN_TEST_OBS = _CFG.training.min_test_observations

# Optional extra inputs (off in the published configuration): SWOT's per-pass height uncertainty
# and dark-water fraction. See docs/design-notes.md.
QUALITY_FEATS = _CFG.inputs.get("quality_features", False)
DYN_FEATS = ["anchor_ff", "hls_water_frac", "sar_water_frac", "precip_routed",
             "has_hls", "has_sar", "days_since_swot", "sin_doy", "cos_doy"]
if QUALITY_FEATS:
    DYN_FEATS = DYN_FEATS + ["swot_wse_u", "swot_dark_frac"]
N_DYN = len(DYN_FEATS)

ATTR_COLS = [
    "log_cap_mcm", "log_area_skm", "mean_depth_m", "DAM_HGT_M", "log_catch_skm",
    "sin_lat", "cos_lat", "sin_lon", "cos_lon", "dor_pc",
    "use_irrigation", "use_hydroelectricity", "use_water_supply",
    "use_flood_control", "use_recreation", "use_other",
    "irr_idx",
    "precip_mean_annual", "precip_seasonality", "precip_dry_months",
    "aec_slope_mid", "aec_area_frac_mid",
    "log_wse_std",   # SWOT-era WSE variability -- the dominant signal for ScalableGRU's residual gate
]
N_ATTR = len(ATTR_COLS)


# ── Metrics ─────────────────────────────────────────────────────────────────

def r2_bc(yt, yp):
    m = np.isfinite(yt) & np.isfinite(yp)
    if m.sum() < 3 or np.std(yt[m]) < 1e-9: return np.nan
    return float(stats.pearsonr(yt[m], yp[m])[0] ** 2)

def nse(yt, yp):
    m = np.isfinite(yt) & np.isfinite(yp)
    if m.sum() < 3: return np.nan
    denom = np.sum((yt[m] - yt[m].mean()) ** 2)
    return float(1 - np.sum((yt[m] - yp[m]) ** 2) / (denom + 1e-9))

def rmse(yt, yp):
    m = np.isfinite(yt) & np.isfinite(yp)
    if m.sum() < 3: return np.nan
    return float(np.sqrt(np.mean((yp[m] - yt[m]) ** 2)))


# ── Precip routing ───────────────────────────────────────────────────────────

def route_precip(vals, tau):
    a = 1.0 / tau; out = np.empty(len(vals)); acc = 0.0
    for i, p in enumerate(vals):
        acc = (1 - a) * acc + a * (p if np.isfinite(p) else 0.0); out[i] = acc
    return out


# ── Model ────────────────────────────────────────────────────────────────────

class ScalableGRU(nn.Module):
    """Dual-timescale GRU with a static-attribute encoder.

    A long (180 d) and a short (10 d) window each pass through a GRU; the attribute encoder gives a
    per-reservoir context vector; the head outputs a residual in [-1, 1]. A separate gate reads the
    full attribute vector and scales the residual per reservoir, because the residual helps mainly on
    high-variability reservoirs (docs/design-notes.md).
    """
    def __init__(self, n_dyn, n_attr, hidden=64, attr_dim=16, dropout=0.2):
        super().__init__()
        self.gru_long  = nn.GRU(n_dyn, hidden, batch_first=True)
        self.gru_short = nn.GRU(n_dyn, hidden, batch_first=True)
        # Attribute encoder: maps observable static traits → context vector.
        # Two-layer MLP so the model can learn non-linear combinations of attributes
        # (e.g. cap × seasonality interaction for fill/drain dynamics).
        self.attr_enc = nn.Sequential(
            nn.Linear(n_attr, attr_dim * 2), nn.ReLU(),
            nn.Linear(attr_dim * 2, attr_dim), nn.Tanh(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * 2 + attr_dim, hidden), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1), nn.Tanh(),
        )
        self.gate = nn.Sequential(
            nn.Linear(n_attr, 8), nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, long_seq, short_seq, attrs):
        _, hl = self.gru_long(long_seq)
        _, hs = self.gru_short(short_seq)
        resid = self.head(torch.cat([hl[-1], hs[-1], self.attr_enc(attrs)], dim=1)).squeeze(-1)
        gate = torch.sigmoid(self.gate(attrs)).squeeze(-1)
        return resid * gate


# ── Per-reservoir daily feature builder ──────────────────────────────────────

# Outlier screen on the SWOT pass series. anchor.screen names a screen from causal_screens.CANDIDATES
# (as causal_<name>), which judges pass i from passes <= i only and repairs by carry-forward, or one of
# the two-sided screens 'hampel' / 'hampel_interp', or 'none'. The published setting is
# causal_C2b_incr_n4. The two-sided screens look ahead and are kept for the ablation record only;
# causal_* needs anchor.area_smoother = causal. See docs/design-notes.md.
ANCHOR_SCREEN = "" if _CFG.anchor.screen == "none" else _CFG.anchor.screen
ANCHOR_SCREEN_K = int(_CFG.anchor.screen_window)
ANCHOR_SCREEN_N = float(_CFG.anchor.screen_nsigma)

# Gauge-free anomaly configuration: no gauge-derived offset (inputs.use_gauge), series centred on their
# own means (target.centre_per_reservoir), no clip of the fill fraction (target.clip_fill_fraction), and
# the checkpoint metric (training.select_on). The first two belong together. See docs/design-notes.md.
GAUGEFREE    = not _CFG.inputs.use_gauge
ANOM_TARGET  = _CFG.target.centre_per_reservoir
NO_CLIP      = not _CFG.target.clip_fill_fraction
SELECT       = _CFG.training.select_on               # "mse" | "kge"
UNGAUGED_MEAN_FF = _CFG.anchor.ungauged_mean_fill

# Surface area: anchor.area_smoother = 'causal' replaces the feature table's two-sided `sa_smooth` with a
# one-sided smoother read from the `causal_surface_area` path, so the value at pass k uses only passes
# <= k. 'twosided' is the retrospective variant and is not a defensible present-day estimate.
CAUSAL_SA = _CFG.anchor.area_smoother == "causal"
# anchor.causal_area_strict: raise if any SWOT pass row lacks a causal area (it would silently keep the
# two-sided area). The published table covers every pass row.
CAUSAL_SA_STRICT = bool(_CFG.anchor.causal_area_strict)
CAUSAL_SA_FILE = _paths.get("causal_surface_area", must_exist=False)
_CAUSAL_SA_TABLE = None


def _hampel_flags(x, k, nsig):
    """True where the pass is an outlier against its k neighbours on each side."""
    x = np.asarray(x, dtype=float); bad = np.zeros(len(x), bool)
    for i in range(len(x)):
        lo, hi = max(0, i - k), min(len(x), i + k + 1)
        w = np.delete(x[lo:hi], min(i - lo, hi - lo - 1))
        if len(w) < 3:
            continue
        med = np.median(w)
        mad = 1.4826 * np.median(np.abs(w - med))
        if mad > 0 and abs(x[i] - med) > nsig * mad:
            bad[i] = True
    return bad


def _hampel_interp(x, day_idx, k, nsig):
    """Reject outliers and interpolate across them in time (the plain Hampel filter substitutes the
    local median, which also flattens genuine peaks)."""
    x = np.asarray(x, dtype=float); bad = _hampel_flags(x, k, nsig)
    if (~bad).sum() < 3:
        return x
    return np.interp(day_idx, np.asarray(day_idx)[~bad], x[~bad])


def _hampel(x, k, nsig):
    """Replace pass-level outliers by the median of their k neighbours on each side."""
    x = np.asarray(x, dtype=float); out = x.copy()
    for i in range(len(x)):
        lo, hi = max(0, i - k), min(len(x), i + k + 1)
        w = np.delete(x[lo:hi], min(i - lo, hi - lo - 1))
        if len(w) < 3:
            continue
        med = np.median(w)
        mad = 1.4826 * np.median(np.abs(w - med))
        if mad > 0 and abs(x[i] - med) > nsig * mad:
            out[i] = med
    return out


def set_causal_area_table(table):
    """Use an in-memory causal-area table (lake_id, date, sa_causal, area_raw) instead of the file.

    For reservoirs outside the table in ``causal_surface_area``; see ``swotnow.inference``."""
    global _CAUSAL_SA_TABLE
    t = table.copy()
    t["lake_id"] = t["lake_id"].astype(str)
    t["date"] = pd.to_datetime(t["date"])
    _CAUSAL_SA_TABLE = t.set_index(["lake_id", "date"])[
        [c for c in ("sa_causal", "area_raw") if c in t.columns]]


def build_daily(df_res, precip_routed, wse_ref=None):
    """wse_ref=None keeps the reference elevation as the median of the reservoir's whole WSE record.
    That is a per-reservoir constant computed from the full record and is a declared exception to
    the no-look-ahead rule; the argument lets the no-look-ahead test hold it equal between a full
    record and a truncated one, so that everything else can be tested for look-ahead."""
    sub = df_res.sort_values("date").reset_index(drop=True).copy()
    cap = float(sub["cap_mcm_eff"].iloc[0]) if "cap_mcm_eff" in sub.columns else np.nan
    if not np.isfinite(cap) or cap <= 0:
        return None

    # Reference WSE and surface area for Method-B anchor computation.
    # Use entire series (not just training) so the anchor is stable globally.
    wse_vals = sub["wse"].dropna()
    if len(wse_vals) < 3:
        return None
    if wse_ref is None:
        wse_ref = float(np.median(wse_vals.values))
    sa_ref  = float(sub["sa_smooth"].dropna().median()
                    if sub["sa_smooth"].notna().any() else 1.0)

    # mean_ff: used as anchor fallback and for static stat encoding.
    # For in-situ reservoirs, derive from gauge. For pseudo-label, use storage_mcm
    # (= Method-B fill fraction from SWOT) if available, else carry-forward mean.
    if CAUSAL_SA:
        global _CAUSAL_SA_TABLE
        if _CAUSAL_SA_TABLE is None:
            _t = pd.read_parquet(CAUSAL_SA_FILE)
            _t["lake_id"] = _t["lake_id"].astype(str)
            _t["date"] = pd.to_datetime(_t["date"])
            _CAUSAL_SA_TABLE = _t.set_index(["lake_id", "date"])[
                [c for c in ("sa_causal", "area_raw") if c in _t.columns]]
        _lid = str(sub["lake_id"].iloc[0])
        _key = pd.MultiIndex.from_arrays(
            [np.repeat(_lid, len(sub)), pd.to_datetime(sub["date"]).values])
        _tab = _CAUSAL_SA_TABLE.reindex(_key)
        _cs = _tab["sa_causal"].values
        _area_raw = (_tab["area_raw"].values if "area_raw" in _tab.columns
                     else np.full(len(sub), np.nan))
        if CAUSAL_SA_STRICT:
            _passrow = ((sub["on_swot_pass"] == 1) & sub["wse"].notna()).values
            _miss = int((_passrow & ~np.isfinite(_cs)).sum())
            if _miss:
                raise RuntimeError(
                    f"anchor.causal_area_strict: reservoir {_lid} has {_miss} SWOT pass row(s) with "
                    f"no causal area in {CAUSAL_SA_FILE}; they would silently keep the two-sided "
                    f"LOWESS area. Rebuild the table with scripts/swotnow/build_causal_area.py.")
        # only replace where the causal value exists; elsewhere keep the table's value so the
        # pass count does not change between configurations
        sub = sub.copy()
        sub["sa_smooth"] = np.where(np.isfinite(_cs), _cs, sub["sa_smooth"].values)

    if GAUGEFREE:
        # Deployment anchor: no gauge-derived offset. The anchor the network reads is the one an
        # ungauged reservoir receives, so no storage record is needed to build it.
        mean_ff = UNGAUGED_MEAN_FF
    else:
        stor_vals = sub["storage_mcm"].dropna()
        if len(stor_vals) >= 5:
            mean_ff = float(np.clip((stor_vals / cap).mean(), 0.01, 0.99))
        else:
            return None

    # Compute Method-B carry-forward anchor (forward-filled between SWOT passes).
    anchor = np.full(len(sub), np.nan)
    for i in range(len(sub)):
        if sub.loc[i, "on_swot_pass"] == 1 and np.isfinite(sub.loc[i].get("wse", np.nan)):
            sa = (sub.loc[i, "sa_smooth"]
                  if np.isfinite(sub.loc[i].get("sa_smooth", np.nan)) else sa_ref)
            anchor[i] = (sub.loc[i, "wse"] - wse_ref) * sa / cap + mean_ff
    if ANCHOR_SCREEN in ("hampel", "hampel_interp"):
        _idx = np.flatnonzero(np.isfinite(anchor))
        if len(_idx) >= 10:
            if ANCHOR_SCREEN == "hampel":
                anchor[_idx] = _hampel(anchor[_idx], ANCHOR_SCREEN_K, ANCHOR_SCREEN_N)
            else:
                anchor[_idx] = _hampel_interp(anchor[_idx], _idx,
                                              ANCHOR_SCREEN_K, ANCHOR_SCREEN_N)
    elif ANCHOR_SCREEN.startswith("causal_"):
        if not CAUSAL_SA:
            raise RuntimeError("anchor.screen = causal_* needs anchor.area_smoother = causal "
                               "(it reads the raw per-pass area from the causal-area table)")
        _idx = np.flatnonzero(np.isfinite(anchor))
        if len(_idx):
            # No `len(_idx) >= 10` gate here: that count is the reservoir's WHOLE record, so it
            # would decide on the future. The screens carry their own history requirement instead.
            _dates = pd.to_datetime(sub["date"]).values[_idx]
            _td = ((_dates - _dates[0]) / np.timedelta64(1, "D")).astype(float)
            anchor[_idx], _ = causal_screen(ANCHOR_SCREEN[len("causal_"):], _td,
                                            anchor[_idx], _area_raw[_idx])
    sub["anchor_ff"] = pd.Series(anchor).ffill().fillna(mean_ff).clip(-0.5, 2.0).values

    area = float(sub["area_skm"].iloc[0]) if "area_skm" in sub.columns else 1.0
    sub["hls_water_frac"] = (
        (sub["hls_water_area_km2"] / max(area, 0.1)).clip(upper=2.0).ffill().fillna(0.0))
    sub["sar_water_frac"] = (
        (sub["sar_water_area_km2"] / max(area, 0.1)).clip(upper=2.0).ffill().fillna(0.0)
        if "sar_water_area_km2" in sub.columns else 0.0)
    sub["precip_routed"] = (
        sub["date"].map(precip_routed).fillna(0.0).values
        if precip_routed is not None else 0.0)
    for c in ["has_hls", "has_sar", "days_since_swot", "sin_doy", "cos_doy"]:
        if c not in sub.columns: sub[c] = 0.0
    if QUALITY_FEATS:
        _pass = (sub["on_swot_pass"] == 1) & sub.get("wse", pd.Series(np.nan, index=sub.index)).notna()
        for _src, _dst, _log in (("wse_u", "swot_wse_u", True), ("dark_frac", "swot_dark_frac", False)):
            _v = pd.to_numeric(sub[_src], errors="coerce") if _src in sub.columns else pd.Series(np.nan, index=sub.index)
            _v = _v.where(_pass)                      # only meaningful on a pass; hold it between passes
            if _log:
                _v = np.log1p(_v.clip(lower=0))
            sub[_dst] = _v.ffill().fillna(0.0).values
    sub[DYN_FEATS] = sub[DYN_FEATS].fillna(0.0)
    sub["_cap"]    = cap
    sub["_meanff"] = mean_ff
    return sub


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    global W_PSEUDO
    torch.manual_seed(SEED); np.random.seed(SEED); torch.set_num_threads(N_THREADS)
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--tag",    type=str, default="v15")
    ap.add_argument("--w-pseudo", type=float, default=W_PSEUDO,
                    help="loss weight for pseudo-label (Method-B) reservoirs "
                         "(default matches the shipped v15: %(default)s)")
    ap.add_argument("--seed", type=int, default=SEED,
                    help="torch/numpy seed (default %(default)s); vary it to measure run-to-run spread")
    ap.add_argument("--dry-run", action="store_true",
                    help="build data structures only, skip training")
    args = ap.parse_args()
    W_PSEUDO = args.w_pseudo   # overrides the module constant for this run
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    run_dir = _paths.get("work_dir", must_exist=False) / "runs" / f"swot_gru_{args.tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    # Record the configuration beside the output, so this run can be traced back to the
    # settings that produced it rather than to a shell invocation nobody kept.
    from swot_reservoir_storage.common.config import record as _record
    _record(run_dir, _CFG, extra={"tag": args.tag, "seed": args.seed,
                                  "w_pseudo": W_PSEUDO})

    print("=" * 66)
    print(f"v15 Scalable SWOT-GRU  (long={LONG_WIN}d short={SHORT_WIN}d "
          f"attr_dim={ATTR_DIM})")
    print(f"  leave-reservoirs-out evaluation | pseudo-label weight={W_PSEUDO}")
    print("=" * 66, flush=True)

    # ── Load static attributes ────────────────────────────────────────────────
    attrs_df = pd.read_parquet(ATTR_FILE)
    attrs_df["lake_id"] = attrs_df["lake_id"].astype(str)
    # Mean-impute any ATTR_COLS column with missing values: one NaN attribute is enough to turn a
    # whole val/test loss to NaN (the loss mask is on the target, not the prediction).
    for col in ATTR_COLS:
        n_missing = attrs_df[col].isna().sum()
        if n_missing:
            attrs_df[col] = attrs_df[col].fillna(attrs_df[col].mean())
            print(f"  Imputed {n_missing} missing {col}")
    attr_lookup = attrs_df.set_index("lake_id")[ATTR_COLS]

    # ── Define training pool ──────────────────────────────────────────────────
    # dtype=str is required: with a missing value, pandas would read lake_id as float64 and append
    # ".0" to every id, breaking the split_map lookups below.
    splits = pd.read_csv(SPLITS_CSV, dtype={"lake_id": str}).dropna(subset=["lake_id"])
    split_map = dict(zip(splits["lake_id"], splits["split"]))

    # Reservoirs in the attribute table but NOT in any in-situ split → pseudo-label
    attr_lids  = set(attr_lookup.index)
    insitu_lids = set(split_map.keys())
    pseudo_lids = attr_lids - insitu_lids

    train_lids = (
        {l for l, s in split_map.items() if s == "train" and l in attr_lids} |
        pseudo_lids
    )
    val_lids  = {l for l, s in split_map.items() if s == "val"  and l in attr_lids}
    test_lids = {l for l, s in split_map.items() if s == "test" and l in attr_lids}

    print(f"  Pool — train: {len(train_lids)}  val(hold-out): {len(val_lids)}  "
          f"test(hold-out): {len(test_lids)}", flush=True)

    # ── Load features & build per-reservoir daily series ─────────────────────
    all_lids = train_lids | val_lids | test_lids
    df = pd.read_parquet(FEAT)
    df["date"] = pd.to_datetime(df["date"]); df["lake_id"] = df["lake_id"].astype(str)
    df = df.sort_values(["lake_id", "date"]).reset_index(drop=True)

    res_data, res_is_pseudo = {}, {}
    skipped = 0
    for lid in sorted(all_lids):
        sub = df[df["lake_id"] == lid]
        if len(sub) == 0: skipped += 1; continue
        ppath = PRECIP_DIR / f"{lid}.csv"
        precip = None
        if ppath.exists():
            pr = pd.read_csv(ppath, parse_dates=["date"]).set_index("date")["precip_mm"]
            precip = pd.Series(route_precip(pr.values, TAU), index=pr.index)
        d = build_daily(sub, precip)
        if d is None: skipped += 1; continue
        res_data[lid] = d
        res_is_pseudo[lid] = lid in pseudo_lids

    print(f"  Built daily series: {len(res_data)}  (skipped {skipped})", flush=True)

    # Partition into actual splits (some lids may have been skipped)
    # Sorted, so training order does not depend on Python's per-process string-hash seed and runs are
    # repeatable under torch.manual_seed. Everything downstream reads these three lists.
    tr_lids   = sorted(l for l in train_lids if l in res_data)
    va_lids   = sorted(l for l in val_lids   if l in res_data)
    te_lids   = sorted(l for l in test_lids  if l in res_data)

    # ── Normalisation ─────────────────────────────────────────────────────────
    # Dynamic: use all training-pool data
    dyn_arrs = [res_data[l][DYN_FEATS].values for l in tr_lids]
    dyn_all  = np.vstack(dyn_arrs)
    dyn_mean = np.nan_to_num(np.nanmean(dyn_all, 0)).astype(np.float32)
    dyn_std  = np.nan_to_num(np.nanstd(dyn_all, 0), nan=1.0).astype(np.float32)
    dyn_std[dyn_std < 1e-6] = 1.0

    # Static attributes: z-score using training pool means; imputation already done
    attr_arr  = np.array([attr_lookup.loc[l].values for l in tr_lids
                          if l in attr_lookup.index], dtype=np.float32)
    attr_mean = np.nan_to_num(attr_arr.mean(0)).astype(np.float32)
    attr_std  = np.nan_to_num(attr_arr.std(0),  nan=1.0).astype(np.float32)
    attr_std[attr_std < 1e-6] = 1.0
    # Save normalisation params for inference
    np.savez(run_dir / "attr_norm.npz", mean=attr_mean, std=attr_std,
             cols=np.array(ATTR_COLS))

    def norm_attr(lid):
        if lid not in attr_lookup.index:
            return np.zeros(N_ATTR, dtype=np.float32)
        raw = attr_lookup.loc[lid].values.astype(np.float32)
        return np.clip((raw - attr_mean) / attr_std, -5, 5)

    def norm_dyn(d):
        return np.clip(((d[DYN_FEATS].values - dyn_mean) / dyn_std), -5, 5
                       ).astype(np.float32)

    # ── Sample builder ────────────────────────────────────────────────────────
    def samples_for(lid, phase):
        """
        phase: 'train' | 'val_or_test'
        Returns lists: long_seqs, short_seqs, attr_vecs, targets (ff, anchor_ff),
                       weights, dates.
        Target for in-situ: gauge fill fraction.
        Target for pseudo-label: anchor_ff itself (Method-B carry-forward) →
          trivial zero-residual objective that regularises the model toward
          persistence when no gauge calibration is available.
        """
        d   = res_data[lid].reset_index(drop=True)
        dv  = norm_dyn(d)
        av  = norm_attr(lid)
        cap = d["_cap"].iloc[0]
        is_pseudo = res_is_pseudo[lid]

        dates = d["date"].values
        stor  = d["storage_mcm"].values.astype(float)
        anch  = d["anchor_ff"].values.astype(float)
        w_res = W_PSEUDO if is_pseudo else 1.0

        # (B) Anomaly target: remove each reservoir's own level from BOTH the target and the
        # anchor, so the loss is on the change and carries no level term. The residual then
        # learns "how does this reservoir depart from its own mean", which is the quantity the
        # gauge-free product reports. Offsets are computed over the reservoir's own window only.
        off_t, off_a = 0.0, 0.0
        if ANOM_TARGET:
            _tt = (anch if is_pseudo else np.where(np.isfinite(stor), stor / cap, np.nan))
            _tt = np.asarray(_tt, float)[LONG_WIN:]
            if np.isfinite(_tt).sum() >= MIN_TEST_OBS:
                off_t = float(np.nanmean(_tt))
            _aa = anch[LONG_WIN:]
            if np.isfinite(_aa).sum() >= MIN_TEST_OBS:
                off_a = float(np.nanmean(_aa))

        L, S, AT, Y, W, DT = [], [], [], [], [], []
        for t in range(LONG_WIN, len(d)):
            # Target fill fraction
            if is_pseudo:
                # Use anchor_ff as pseudo-label → pushes residual toward 0
                ff = float(anch[t])
            else:
                ff = stor[t] / cap if np.isfinite(stor[t]) else np.nan
                if not np.isfinite(ff):
                    continue

            L.append(dv[t - LONG_WIN:t])
            S.append(dv[t - SHORT_WIN:t])
            AT.append(av)
            Y.append([np.clip(ff - off_t, -0.5, 2.0), anch[t] - off_a])
            W.append(w_res)
            DT.append(dates[t])
        return L, S, AT, Y, W, DT

    print("Building samples …", flush=True)
    def gather(lids):
        L, S, AT, Y, W = [], [], [], [], []
        for lid in lids:
            l, s, at, y, w, _ = samples_for(lid, "train")
            L += l; S += s; AT += at; Y += y; W += w
        return L, S, AT, Y, W

    Ltr, Str, ATtr, Ytr, Wtr = gather(tr_lids)
    Lva, Sva, ATva, Yva, Wva = gather(va_lids)
    # (D) reservoir index per validation sample, so early stopping can score per reservoir
    # rather than pooling series of different magnitudes into one number.
    va_res_idx = []
    for _i, _lid in enumerate(va_lids):
        _l, _, _, _, _, _ = samples_for(_lid, "train")
        va_res_idx += [_i] * len(_l)
    va_res_idx = np.array(va_res_idx)
    print(f"  Train samples: {len(Ltr):,}  |  Val samples: {len(Lva):,}", flush=True)

    if args.dry_run:
        print("\n[dry-run] Data OK — exiting before training."); return

    def to_t(L, S, AT, Y, W):
        return (torch.tensor(np.stack(L),  dtype=torch.float32),
                torch.tensor(np.stack(S),  dtype=torch.float32),
                torch.tensor(np.stack(AT), dtype=torch.float32),
                torch.tensor(np.array(Y),  dtype=torch.float32),
                torch.tensor(np.array(W),  dtype=torch.float32))

    trL, trS, trAT, trY, trW = to_t(Ltr, Str, ATtr, Ytr, Wtr)
    vaL, vaS, vaAT, vaY, vaW = to_t(Lva, Sva, ATva, Yva, Wva)
    loader = DataLoader(TensorDataset(trL, trS, trAT, trY, trW),
                        batch_size=BATCH_SIZE, shuffle=True)

    # ── Model + optimiser ─────────────────────────────────────────────────────
    model = ScalableGRU(N_DYN, N_ATTR, hidden=args.hidden, attr_dim=ATTR_DIM)
    print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}\n",
          flush=True)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=5, factor=0.5, min_lr=1e-5)
    ckpt  = run_dir / "model.pt"
    best, patience_cnt = float("inf"), 0

    def _val_kge(pred, true, idx):
        """(D) mean per-reservoir anomaly KGE, returned NEGATED so lower is better.

        KGE penalises amplitude != 1, which plain MSE does not. With an anomaly target the
        cheapest way to cut MSE is to shrink toward zero, so selecting on MSE would actively
        reward shrinkage.
        """
        vals = []
        for i in np.unique(idx):
            m = (idx == i)
            pp, tt = pred[m], true[m]
            ok = np.isfinite(pp) & np.isfinite(tt)
            if ok.sum() < MIN_TEST_OBS: continue
            pp, tt = pp[ok] - pp[ok].mean(), tt[ok] - tt[ok].mean()
            if np.std(tt) < 1e-9 or np.std(pp) < 1e-9: continue
            cc = float(np.corrcoef(pp, tt)[0, 1])
            al = float(np.std(pp) / np.std(tt))
            vals.append(1 - np.sqrt((cc - 1) ** 2 + (al - 1) ** 2))
        return -float(np.mean(vals)) if vals else float("inf")

    # ── Training loop ─────────────────────────────────────────────────────────
    for ep in range(1, EPOCHS + 1):
        model.train(); tot, n = 0.0, 0
        for bL, bS, bAT, bY, bW in loader:
            resid = model(bL, bS, bAT)
            pred  = (bY[:, 1] + resid).clamp(-0.5, 2.0)
            # Weighted masked MSE: index before difference to avoid 0×NaN bug
            mask  = torch.isfinite(bY[:, 0])
            pred_m, tgt_m, w_m = pred[mask], bY[:, 0][mask], bW[mask]
            if len(pred_m) == 0: continue
            loss = (w_m * (pred_m - tgt_m) ** 2).mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * mask.sum().item(); n += mask.sum().item()

        model.eval()
        with torch.no_grad():
            vp  = (vaY[:, 1] + model(vaL, vaS, vaAT)).clamp(-0.5, 2.0)
            msk = torch.isfinite(vaY[:, 0])
            vl  = ((vaW[msk] * (vp[msk] - vaY[:, 0][msk]) ** 2).mean()).item()
            if SELECT == "kge":
                vl = _val_kge(vp.numpy(), vaY[:, 0].numpy(), va_res_idx)
        sched.step(vl)
        if ep == 1 or ep % 5 == 0:
            print(f"  Epoch {ep:3d}  train={tot/max(n,1):.4f}  val={vl:.4f}  "
                  f"lr={opt.param_groups[0]['lr']:.1e}", flush=True)
        if vl < best:
            best, patience_cnt = vl, 0
            torch.save(model.state_dict(), ckpt)
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"  Early stop ep {ep} (best val={best:.4f})", flush=True)
                break

    model.load_state_dict(torch.load(ckpt)); model.eval()

    # ── Leave-reservoirs-out evaluation ──────────────────────────────────────
    print("\nEvaluating on HELD-OUT reservoirs (never seen in training) …",
          flush=True)
    rows, pred_rows, daily_rows = [], [], []

    for phase_lids, phase_name in [(va_lids, "val"), (te_lids, "test")]:
        for lid in phase_lids:
            _, _, _, Y_all, _, DT_all = samples_for(lid, "val_or_test")
            if len(Y_all) < MIN_TEST_OBS: continue
            d = res_data[lid].reset_index(drop=True)
            dv = norm_dyn(d); av = norm_attr(lid)
            cap = d["_cap"].iloc[0]

            # Collect all timesteps (require target for metrics)
            L_, S_, AT_, Y_, DT_ = [], [], [], [], []
            # (B) the evaluation must be centred exactly as training was, or a centred
            # prediction would be scored against an uncentred target.
            _off_t, _off_a = 0.0, 0.0
            if ANOM_TARGET:
                _ffs = (d["storage_mcm"].values.astype(float) / cap)[LONG_WIN:]
                _ans = d["anchor_ff"].values.astype(float)[LONG_WIN:]
                if np.isfinite(_ffs).sum() >= MIN_TEST_OBS: _off_t = float(np.nanmean(_ffs))
                if np.isfinite(_ans).sum() >= MIN_TEST_OBS: _off_a = float(np.nanmean(_ans))
            for t in range(LONG_WIN, len(d)):
                ff = d["storage_mcm"].iloc[t] / cap if np.isfinite(d["storage_mcm"].iloc[t]) else np.nan
                if not np.isfinite(ff): continue
                L_.append(dv[t - LONG_WIN:t])
                S_.append(dv[t - SHORT_WIN:t])
                AT_.append(av)
                Y_.append([np.clip(ff - _off_t, -0.5, 2.0),
                           float(d["anchor_ff"].iloc[t]) - _off_a])
                DT_.append(d["date"].iloc[t])

            if len(L_) < MIN_TEST_OBS: continue
            with torch.no_grad():
                r = model(
                    torch.tensor(np.stack(L_),  dtype=torch.float32),
                    torch.tensor(np.stack(S_),  dtype=torch.float32),
                    torch.tensor(np.stack(AT_), dtype=torch.float32),
                ).numpy()
            Y_ = np.array(Y_)
            yt = Y_[:, 0] * cap
            # Training optimises (anchor + residual).clamp(-0.5, 2.0), so metrics are computed on the
            # raw value written beside the clipped one. The clipped column is the product where
            # storage must stay within [0, capacity]; use the raw one for evaluation and anomalies.
            yp_raw = (Y_[:, 1] + r) * cap
            # (C) clip(f,0,1) presumes an absolute fill fraction. With no gauge there is no
            # datum, and with an anomaly target the series is centred on zero, so clipping
            # would truncate half of it. Score raw in those configurations.
            yp = (yp_raw if (NO_CLIP or ANOM_TARGET)
                  else np.clip(Y_[:, 1] + r, 0, 1) * cap)
            rows.append({"lake_id": lid, "phase": phase_name, "n": len(yt),
                         "r2_bc": r2_bc(yt, yp), "nse": nse(yt, yp),
                         "rmse_mcm": rmse(yt, yp), "cap_mcm": cap,
                         "r2_bc_raw": r2_bc(yt, yp_raw), "nse_raw": nse(yt, yp_raw),
                         "rmse_mcm_raw": rmse(yt, yp_raw)})
            for dd, pv, tv, pr in zip(DT_, yp, yt, yp_raw):
                pred_rows.append({"date": dd, "lake_id": lid, "phase": phase_name,
                                  "pred_mcm": float(pv), "true_mcm": float(tv),
                                  "pred_mcm_raw": float(pr)})

            # Daily series (all days, no target required)
            Ld, Sd, ATd, anch_d, dt_d = [], [], [], [], []
            for t in range(LONG_WIN, len(d)):
                Ld.append(dv[t - LONG_WIN:t]); Sd.append(dv[t - SHORT_WIN:t])
                ATd.append(av); anch_d.append(float(d["anchor_ff"].iloc[t]))
                dt_d.append(d["date"].iloc[t])
            with torch.no_grad():
                r2 = model(
                    torch.tensor(np.stack(Ld),  dtype=torch.float32),
                    torch.tensor(np.stack(Sd),  dtype=torch.float32),
                    torch.tensor(np.stack(ATd), dtype=torch.float32),
                ).numpy()
            for dd, pv, ay in zip(dt_d, r2, anch_d):
                daily_rows.append({"date": dd, "lake_id": lid, "phase": phase_name,
                                   "pred_mcm": float(np.clip(ay + pv, 0, 1) * cap),
                                   "anchor_mcm": float(np.clip(ay, 0, 1) * cap),
                                   "pred_mcm_raw": float((ay + pv) * cap),
                                   "anchor_mcm_raw": float(ay * cap)})

    m = pd.DataFrame(rows)
    m.to_csv(run_dir / "metrics_leaveout.csv", index=False)
    pd.DataFrame(pred_rows).to_parquet(
        run_dir / "predictions_leaveout.parquet", index=False)
    pd.DataFrame(daily_rows).to_parquet(
        run_dir / "predictions_daily.parquet", index=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n  Leave-out results (val+test, never-seen reservoirs):")
    for ph in ["val", "test"]:
        s = m[m["phase"] == ph]
        if s.empty: continue
        print(f"  [{ph.upper():4s}] n={len(s):2d}  "
              f"R²_bc={s['r2_bc'].median():.3f}  "
              f"NSE={s['nse'].median():.3f}  "
              f"RMSE={s['rmse_mcm'].median():.1f} MCM")
    if not m.empty:
        print(f"  [ALL ] n={len(m):2d}  "
              f"R²_bc={m['r2_bc'].median():.3f}  "
              f"NSE={m['nse'].median():.3f}  "
              f"RMSE={m['rmse_mcm'].median():.1f} MCM")
    # No expected values are printed: score a run against the evaluation tables in the data deposit.
    print(f"\n  Outputs → {run_dir}", flush=True)


if __name__ == "__main__":
    main()
