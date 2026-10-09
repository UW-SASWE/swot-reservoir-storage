"""Run the trained SWOTNOW model on any reservoir that SWOT observes.

    from swot_reservoir_storage.swotnow import inference
    est = inference.predict(["7830181213"], end="2026-08-31")

Everything is built from raw observations: the daily feature table (``preprocessing.features``), the
one-sided area (``preprocessing.causal_area``), the static attributes
(``preprocessing.static_attributes``) and routed precipitation. No gauge is read.

Two files travel with the model: ``attr_norm.npz`` (static-attribute normalisation) and
``dyn_norm.npz`` (dynamic-feature normalisation and the value that replaces a missing static attribute; ``scripts/swotnow/export_normalisation.py`` writes it
from the training pool). Both are needed, and neither can be rebuilt from a single reservoir.

The estimate for day x uses observations up to x only, apart from the per-reservoir constants listed in
``config/swotnow.yaml`` (for example the reference elevation, the median of the whole water-level
record). The first ``long_window`` days of a record are used as input history and are not estimated.

Output columns, in million m3 (like the ``*_raw`` columns of the published predictions):

    anchor_mcm_raw   the physical anchor alone: (water level - reference level) x area, held between passes
    pred_mcm_raw     the SWOTNOW estimate: the anchor plus the learned correction

Both sit on an arbitrary offset (the nominal reference storage is 1% of capacity), so they are
storage *anomalies*. Subtract a reservoir's own mean over the period you analyse to compare with
anything else.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..common import paths
from .model import trainer as T
from .model.normalisation import load_routed_precip
from .preprocessing import causal_area, features, static_attributes

__all__ = ["export_dyn_norm", "load_model", "predict"]


def export_dyn_norm(out_path) -> Path:
    """Write the dynamic-feature normalisation from the training pool (needs the training data)."""
    from .model.normalisation import build_norm_and_pool
    _, _, dyn_mean, dyn_std = build_norm_and_pool()
    attrs = pd.read_parquet(T.ATTR_FILE)
    attr_fill = np.array([attrs[c].mean() for c in T.ATTR_COLS], dtype=np.float32)
    out_path = Path(out_path)
    np.savez(out_path, mean=dyn_mean, std=dyn_std, cols=np.array(T.DYN_FEATS), attr_fill=attr_fill)
    return out_path


def load_model(model_dir):
    """The network, and the static-attribute fill, mean and std, and the dynamic-feature mean and std."""
    model_dir = Path(model_dir)
    attr = np.load(model_dir / "attr_norm.npz", allow_pickle=True)
    dyn_file = model_dir / "dyn_norm.npz"
    if not dyn_file.exists():
        raise FileNotFoundError(
            f"{dyn_file} not found. It holds the dynamic-feature normalisation, which the model needs and "
            "which is not stored with the weights by the trainer. Download it with the model, or write it "
            "with scripts/swotnow/export_normalisation.py.")
    dyn = np.load(dyn_file, allow_pickle=True)
    if list(dyn["cols"]) != list(T.DYN_FEATS):
        raise ValueError("dyn_norm.npz was written for a different list of dynamic features")
    from ..common.config import load as load_config
    cfg = load_config("swotnow")
    net = T.ScalableGRU(T.N_DYN, T.N_ATTR, hidden=cfg.model.hidden, attr_dim=T.ATTR_DIM)
    net.load_state_dict(torch.load(model_dir / "model.pt", map_location="cpu"))
    net.eval()
    attr_mean = attr["mean"].astype(np.float32)
    # a missing attribute is replaced by the mean over the attribute table, as in training
    fill = dyn["attr_fill"].astype(np.float32) if "attr_fill" in dyn.files else attr_mean
    return (net, fill, attr_mean, attr["std"].astype(np.float32),
            dyn["mean"].astype(np.float32), dyn["std"].astype(np.float32))


def predict(lake_ids, end, model_dir=None, capacity: dict | None = None, progress_every: int = 25) -> pd.DataFrame:
    """SWOTNOW estimates for the given reservoirs: lake_id, date, anchor_mcm_raw, pred_mcm_raw.

    ``capacity`` (lake_id -> million m3) overrides the catalogued capacity. Reservoirs that cannot be
    assembled (fewer than two usable SWOT passes, or no capacity) or that have no more than
    ``long_window`` days of record are left out and listed in the printed summary.
    """
    if not T.GAUGEFREE:
        raise RuntimeError("inputs.use_gauge is true, but the published model is gauge-free.")
    model_dir = Path(model_dir) if model_dir else paths.get("swotnow_model")
    net, attr_fill, attr_mean, attr_std, dyn_mean, dyn_std = load_model(model_dir)
    ids = [str(l) for l in lake_ids]

    feats = features.build(ids, end, capacity, progress_every=0)
    if feats.empty:
        return pd.DataFrame(columns=["lake_id", "date", "anchor_mcm_raw", "pred_mcm_raw"])
    tabs, bad = [], {}
    for lid, g in feats.groupby("lake_id"):
        dates = g.loc[(g.on_swot_pass == 1) & g.wse.notna(), "date"].sort_values().values
        tab, problem = causal_area.for_reservoir(lid, dates)
        if problem:
            bad[lid] = problem
        else:
            tabs.append(tab)
    if bad:
        raise RuntimeError(f"causal area could not be built for {len(bad)} reservoir(s): {dict(list(bad.items())[:5])}")
    T.set_causal_area_table(pd.concat(tabs, ignore_index=True))

    attrs = static_attributes.build(list(feats.lake_id.unique())).set_index("lake_id")[T.ATTR_COLS]
    rows, short = [], []
    for k, (lid, sub) in enumerate(feats.groupby("lake_id"), 1):
        d = T.build_daily(sub, load_routed_precip(lid))
        if d is None or len(d) <= T.LONG_WIN:
            short.append(lid)
            continue
        d = d.reset_index(drop=True)
        cap = float(d["_cap"].iloc[0])
        a = attrs.loc[lid].values.astype(np.float32)
        a = np.where(np.isfinite(a), a, attr_fill)
        av = np.clip((a - attr_mean) / attr_std, -5, 5)
        dv = np.clip((d[T.DYN_FEATS].values - dyn_mean) / dyn_std, -5, 5).astype(np.float32)
        idx = range(T.LONG_WIN, len(d))
        with torch.no_grad():
            r = net(torch.tensor(np.stack([dv[t - T.LONG_WIN:t] for t in idx])),
                    torch.tensor(np.stack([dv[t - T.SHORT_WIN:t] for t in idx])),
                    torch.tensor(np.tile(av, (len(idx), 1)), dtype=torch.float32)).numpy()
        anch = d["anchor_ff"].values[T.LONG_WIN:]
        rows.append(pd.DataFrame({"lake_id": lid, "date": pd.to_datetime(d["date"].values)[T.LONG_WIN:],
                                  "anchor_mcm_raw": anch * cap, "pred_mcm_raw": (anch + r) * cap}))
        if progress_every and k % progress_every == 0:
            print(f"  {k}/{feats.lake_id.nunique()}", flush=True)
    if short:
        print(f"not estimated (no more than {T.LONG_WIN} days of record, or capacity unusable): {len(short)}")
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["lake_id", "date", "anchor_mcm_raw", "pred_mcm_raw"])
