"""Causal SWOT surface area for the anchor, covering EXACTLY the pass rows the model reads.

The pass selection is build_features' rule verbatim (quality flag 0 or 1; one pass per date, best
quality first; no filter on dark_frac), so the pass set matches the feature table by construction.
The builder then verifies that: every pass row in features.parquet must be found in the raw
Hydrocron file, or the run stops.

The smoother itself is unchanged from the shipped one (a trailing tricube-weighted mean, a
local-CONSTANT fit; a trailing LOWESS was tried first and doubled the variance). Two properties are
worth stating because the no-look-ahead test exercises both:

  * the value at pass k uses only passes <= k;
  * the WINDOW WIDTH, round(0.3 * n_passes), is a per-reservoir constant computed from the
    reservoir's whole record. It is declared as a static constant, like the reference elevation
    and the water-level-variability attribute, and not made causal.

Reservoirs with fewer than MIN_FIT passes carry the raw per-pass area, which is what
build_features writes for them too, so no future information enters there.

Output: the file named by the `causal_surface_area` path  (lake_id, date, sa_causal, area_raw)

area_raw is the unsmoothed per-pass area. It is carried for screens that judge a pass by how much of
the reservoir SWOT actually saw on it (a partial observation has a small area).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ...common import paths
from ...common.config import load as load_config

_CFG = load_config("swotnow")
SWOT_START = pd.Timestamp(str(_CFG.period.start))
FRAC = float(_CFG.anchor.area_smoother_frac)
MIN_FIT = 5


def load_passes(lid):
    """The pass selection of ml/build_features.load_swot_passes, keeping the raw area_total."""
    f, tcol = paths.get('swot_granules') / f'{lid}.csv', 'time'
    if not f.exists():
        f, tcol = paths.get('swot_granules_cache') / f'{lid}.csv', 'time_str'
    if not f.exists():
        return None
    try:
        d = pd.read_csv(f, low_memory=False)
    except pd.errors.EmptyDataError:
        return None
    if d.empty or tcol not in d.columns:
        return None
    d['_date'] = pd.to_datetime(d[tcol], utc=True, errors='coerce').dt.tz_localize(None).dt.normalize()
    d['quality_f'] = pd.to_numeric(d['quality_f'], errors='coerce')
    d = d[d['quality_f'].isin([0, 1])].copy()
    for c in ('wse', 'area_total'):
        d[c] = pd.to_numeric(d.get(c, np.nan), errors='coerce')
    d = d[d['wse'] > -1e6].dropna(subset=['_date', 'wse', 'area_total'])
    d = d[d['_date'] >= SWOT_START]
    if d.empty:
        return None
    # one pass per date, best quality first -- exactly as build_features does
    d = d.sort_values(['_date', 'quality_f']).groupby('_date').first().reset_index()
    return d.rename(columns={'_date': 'date'})[['date', 'area_total']].sort_values('date')


def window_width(n, frac=FRAC):
    """Passes in the trailing window: round(frac * n), a per-reservoir constant of the whole record."""
    return max(MIN_FIT, int(round(frac * n)))


def causal_smooth(t, y, frac=FRAC, w=None):
    """Trailing tricube-weighted MEAN over the last w passes.

    w defaults to window_width(n), a static per-reservoir constant. Passing it explicitly lets
    test_no_lookahead.py hold the width equal between a full record and a truncated one, so that
    what is tested is whether the VALUE at pass k uses only passes <= k.
    """
    n = len(y)
    if n < MIN_FIT:
        return np.asarray(y, float).copy()
    if w is None:
        w = window_width(n, frac)
    out = np.empty(n)
    for k in range(n):
        lo = max(0, k - w + 1)
        tt, yy = t[lo:k + 1], y[lo:k + 1]
        d = t[k] - tt
        mx = d.max() if d.max() > 0 else 1.0
        ww = (1 - np.abs(d / mx) ** 3) ** 3
        s = ww.sum()
        out[k] = float(np.sum(ww * yy) / s) if s > 0 else y[k]
    return out


def for_reservoir(lid, pass_dates):
    """Causal area for the pass dates of one reservoir: DataFrame(lake_id, date, sa_causal, area_raw).

    Returns (table, problem): ``problem`` is None, or a short message when the raw file is missing or
    lacks some of the requested pass dates (then ``table`` is None).
    """
    d = load_passes(lid)
    if d is None:
        return None, 'no raw file / no usable passes'
    d = d[d['date'].isin(pass_dates)]
    if len(d) != len(pass_dates):
        return None, f'{len(pass_dates) - len(d)} of {len(pass_dates)} feature passes absent from raw file'
    t = (d['date'] - d['date'].iloc[0]).dt.days.values.astype(float)
    cs = np.clip(causal_smooth(t, d['area_total'].values.astype(float)), 0, None)
    return pd.DataFrame({'lake_id': lid, 'date': d['date'].values, 'sa_causal': cs,
                         'area_raw': d['area_total'].values.astype(float)}), None


def main():
    feat = pd.read_parquet(paths.get('features'), columns=['lake_id', 'date', 'on_swot_pass', 'wse'])
    feat['date'] = pd.to_datetime(feat['date'])
    feat['lake_id'] = feat['lake_id'].astype(str)
    fp = feat[(feat.on_swot_pass == 1) & feat.wse.notna()][['lake_id', 'date']]
    by = {lid: g['date'].sort_values().values for lid, g in fp.groupby('lake_id')}
    print(f'reservoirs with passes in features.parquet: {len(by)}  pass rows: {len(fp):,}', flush=True)

    rows, missing = [], []
    for i, (lid, dates) in enumerate(by.items(), 1):
        tab, problem = for_reservoir(lid, dates)
        if problem:
            missing.append((lid, problem))
            continue
        rows.append(tab)
        if i % 300 == 0:
            print(f'  {i}/{len(by)}', flush=True)

    if missing:
        print(f'\nFAILED: {len(missing)} reservoirs whose feature passes could not all be found:', file=sys.stderr)
        for m in missing[:20]:
            print('  ', m, file=sys.stderr)
        sys.exit(1)

    out = pd.concat(rows, ignore_index=True)
    out['lake_id'] = out['lake_id'].astype(str)
    key = set(zip(out.lake_id, out.date))
    cov = np.fromiter(((a, b) in key for a, b in zip(fp.lake_id, fp.date)), bool, len(fp))
    print(f'\ncoverage of features pass rows: {cov.sum():,} / {len(fp):,} = {100 * cov.mean():.2f}%')
    assert cov.all(), 'coverage is not 100% -- the pass set does not match features.parquet'
    OUT = paths.get('causal_surface_area', must_exist=False)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f'wrote {OUT}  ({len(out):,} rows, {out.lake_id.nunique()} reservoirs)')


if __name__ == '__main__':
    main()
