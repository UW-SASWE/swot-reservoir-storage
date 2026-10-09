"""Reconstructing the training-pool normalisation the model was fitted with.

The dynamic-feature means and standard deviations were never saved at training time. Only the
static-attribute normalisation was (``attr_norm.npz``). So any script that runs the trained model
has to rebuild the dynamic normalisation, and it has to rebuild it *the way training built it* --
over the same pool of reservoirs, through the same feature builder, with the same imputation.

This is the most delicate part of running the published model, because getting it wrong produces
no error: the network simply receives inputs on a different scale and returns plausible but wrong
numbers. Two ways to get it wrong:

- **Normalising over the wrong pool.** The statistics come from the training reservoirs only:
  the 'train' split plus every reservoir that has attributes but no split assignment (those are
  the ungauged pool, whose target is the anchor itself). Held-out reservoirs must not contribute,
  or the normalisation carries information from the set being scored.

- **Imputing only some attributes.** Many reservoirs are missing at least one attribute, and a
  single NaN attribute makes the network return NaN for every day of that reservoir. Every column
  in ``ATTR_COLS`` is mean-imputed, exactly as the trainer does it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import trainer as T

__all__ = ["build_norm_and_pool"]


def build_norm_and_pool():
    """Rebuild the daily feature table, the attribute lookup and the dynamic normalisation.

    Returns
    -------
    df : DataFrame
        The full feature table, sorted by reservoir then date.
    attr_lookup : DataFrame
        Static attributes indexed by ``lake_id``, mean-imputed, columns in ``ATTR_COLS`` order.
    dyn_mean, dyn_std : ndarray
        Per-feature statistics over the training pool, in ``DYN_FEATS`` order. A standard
        deviation below 1e-6 is set to 1.0, so a constant feature passes through as zero rather
        than dividing by nothing.
    """
    attrs = pd.read_parquet(T.ATTR_FILE)
    attrs["lake_id"] = attrs["lake_id"].astype(str)
    for c in T.ATTR_COLS:
        if attrs[c].isna().any():
            attrs[c] = attrs[c].fillna(attrs[c].mean())
    attr_lookup = attrs.set_index("lake_id")[T.ATTR_COLS]

    splits = pd.read_csv(T.SPLITS_CSV)
    splits["lake_id"] = splits["lake_id"].astype(str)
    split_map = dict(zip(splits["lake_id"], splits["split"]))

    attr_lids = set(attr_lookup.index)
    # A reservoir with attributes but no split assignment is an ungauged pool member. It trains
    # against its own anchor, so it belongs in the normalisation pool.
    pseudo = attr_lids - set(split_map)
    train_lids = {l for l, s in split_map.items() if s == "train" and l in attr_lids} | pseudo

    df = pd.read_parquet(T.FEAT)
    df["date"] = pd.to_datetime(df["date"])
    df["lake_id"] = df["lake_id"].astype(str)
    df = df.sort_values(["lake_id", "date"]).reset_index(drop=True)

    dyn_stack = []
    for lid in train_lids:
        sub = df[df["lake_id"] == lid]
        if len(sub) == 0:
            continue
        precip = load_routed_precip(lid)
        d = T.build_daily(sub, precip)
        if d is not None:
            dyn_stack.append(d[T.DYN_FEATS].values)

    dyn_all = np.vstack(dyn_stack)
    dyn_mean = np.nan_to_num(np.nanmean(dyn_all, 0)).astype(np.float32)
    dyn_std = np.nan_to_num(np.nanstd(dyn_all, 0), nan=1.0).astype(np.float32)
    dyn_std[dyn_std < 1e-6] = 1.0
    return df, attr_lookup, dyn_mean, dyn_std


def load_routed_precip(lake_id: str):
    """Routed daily precipitation for one reservoir, or None where there is no record.

    Present for about a fifth of reservoirs and identically zero for the rest, so None is an
    ordinary outcome rather than a problem. Factored out of the loop above because every script
    that runs the model needs precipitation on exactly these terms, and re-deriving it ad hoc is
    how the feature path drifts away from the training path.
    """
    p = T.PRECIP_DIR / f"{lake_id}.csv"
    if not p.exists():
        return None
    pr = pd.read_csv(p, parse_dates=["date"]).set_index("date")["precip_mm"]
    return pd.Series(T.route_precip(pr.values, T.TAU), index=pr.index)
