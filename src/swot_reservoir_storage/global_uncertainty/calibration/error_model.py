#!/usr/bin/env python3
"""Per-reservoir error model (random + systematic), both arms, generalised to the whole SWOT-usable
universe via a sigma_S-anchored scaling.

From the smoother validation (the `smoother_metrics` file) each reservoir has, per
arm (U1 = SWOT-only, U0 = area-only), a held-out absolute RMSE and a bias-corrected RMSE. We split:
    sigma_rand = rmse_bc                         (RANDOM error; cancels in quadrature across reservoirs)
    bias_sys   = sqrt(rmse_abs^2 - rmse_bc^2)    (SYSTEMATIC error; accumulates / correlated bound)
Both terms scale with the reservoir's observed dynamic-storage variability
    sigma_S = wse_std[m] * AREA_SKM[km2]   (== MCM)   (validated: error ~ sigma_S, r~0.82)
so we fit ONE robust coefficient per {arm, component} through the origin (median-ratio = Theil-Sen
through 0) on the validation reservoirs, leave-one-out check it, then apply to all usable rows.

The shared bathymetry/bed term (absolute-total only; SWOT-INDEPENDENT; cancels in the dynamic dU) is
carried as a proxy anchor_sigma_mcm = ANCHOR_REL * S_ref where an absolute anchor exists; the
absolute total is therefore a secondary product on the anchored subset, while the DYNAMIC dU covers
every usable reservoir.

Output: per_reservoir_error_model.csv under work_dir/global_uncertainty/
  (lake_id, region, sigma_S, sigma_rand_U1, bias_sys_U1, sigma_rand_U0, bias_sys_U0,
   S_ref_mcm, anchor_sigma_mcm)
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

# UNIV is the v1 universe table, not v2: they differ on wse_std for many shared lake_ids, and
# sigma_S = wse_std * AREA_SKM is this model's central quantity (docs/design-notes.md).
from swot_reservoir_storage.common import paths as _paths
from swot_reservoir_storage.common.config import load as _load_config

_CFG = _load_config("global_uncertainty")

SM        = _paths.get("smoother_metrics")
UNC_SIGMA = _CFG.error_model.fit_denominator
UNC_MINCOV = float(_CFG.error_model.min_window_coverage)
UNIV      = _paths.get("global_universe_v1")
ANCHOR    = _paths.get("storage_anchor")
OUT       = _paths.get("work_dir", must_exist=False) / "global_uncertainty" / "per_reservoir_error_model.csv"
# ---- absolute-arm bed term -------------------------------------------------------------------
# S_ref = ANCHOR_FILL * capacity, where ANCHOR_FILL is the gauge-measured sum-weighted fill fraction
# (sum of true storage over sum of capacity across the gauged reservoirs) and ANCHOR_REL is the robust dispersion (1.4826 * MAD) of that anchor's relative error.
# A capacity fraction cannot exceed capacity by construction. The bed term is SWOT-independent: it
# cancels in the dynamic dU but inflates both arms, so it lowers only the absolute arm's percentage
# reduction. See docs/design-notes.md.
ANCHOR_FILL = 0.41
ANCHOR_REL = 0.59


def split_errors(df, arm):
    """Return (sigma_rand, bias_sys) Series for an arm, from rmse_abs and rmse_bc."""
    rand = df[f'{arm}_rmse_bc']
    syst = np.sqrt(np.clip(df[f'{arm}_rmse'] ** 2 - df[f'{arm}_rmse_bc'] ** 2, 0, None))
    return rand, syst


def median_ratio(y, x):
    """Robust coefficient k for y ~ k*x through the origin (median of per-point ratios)."""
    m = np.isfinite(y) & np.isfinite(x) & (x > 0)
    return float(np.median((y[m] / x[m]).values))


def loo_check(y, x, label):
    """Leave-one-out: predict each reservoir's error from coefficient fit on the others."""
    m = np.isfinite(y) & np.isfinite(x) & (x > 0)
    yv, xv = y[m].values, x[m].values
    pred = np.array([np.median(np.delete(yv, i) / np.delete(xv, i)) * xv[i] for i in range(len(yv))])
    r = stats.pearsonr(pred, yv)[0] if len(yv) > 3 else np.nan
    rmse = float(np.sqrt(np.mean((pred - yv) ** 2)))
    print(f'   LOO {label:18s} n={len(yv):3d}  r(pred,actual)={r:.2f}  RMSE={rmse:.1f} MCM  '
          f'k={median_ratio(y, x):.4f}')
    return median_ratio(y, x)


def main(apply_univ=UNIV, out_path=OUT, measured_path=None):
    # FIT always on v1 universe (the gauged reservoirs' validated sigma_S; mostly single-lake so stable),
    # then APPLY the coefficients to `apply_univ` (e.g. global_universe_v2.csv with combined wse_std).
    # If measured_path is given (production-smoother summary.csv), sigma_S is the MEASURED
    # storage_std of the smoothed U1 series — fit AND apply use it, with the wse_std*AREA
    # proxy as fallback for apply-rows the production run could not compute.
    m = pd.read_csv(SM, dtype={'lake_id': str})
    u = pd.read_csv(UNIV, dtype={'lake_id': str})
    v = m.merge(u[['lake_id', 'wse_std', 'AREA_SKM', 'region']], on='lake_id', how='left')
    v['sigma_S'] = v['wse_std'] * v['AREA_SKM']

    meas = None
    if measured_path is not None:
        meas = pd.read_csv(measured_path, dtype={'lake_id': str})
        meas = meas[['lake_id', 'storage_std']].dropna().drop_duplicates('lake_id')
        v = v.merge(meas, on='lake_id', how='left')
        both = v[v['sigma_S'].notna() & v['storage_std'].notna() & (v['sigma_S'] >= 5)]
        if len(both) > 5:
            r = stats.pearsonr(np.log(both['storage_std']), np.log(both['sigma_S']))[0]
            print(f'measured storage_std vs wse_std*AREA proxy (validation set, n={len(both)}): '
                  f'log-log r={r:.2f}')
        v['sigma_S'] = v['storage_std'].combine_first(v['sigma_S'])
        # physical bound: storage lives in [0, cap] so its std cannot exceed cap/2.
        # Multi-tile reservoirs with cross-tile WSE datum inconsistency otherwise carry
        # impossible sigma_S (seen up to 136x capacity) and dominate the global sum.
        if 'cap_mcm' in v.columns:
            v['sigma_S'] = np.minimum(v['sigma_S'], 0.5 * v['cap_mcm'].fillna(np.inf))
    # filter to sigma_S >= 5 MCM: removes small-reservoir noise where the linear model breaks down
    # (tiny reservoirs have a floor rmse_bc that inflates k; the global sum is dominated by large sigma_S)
    v = v[v['sigma_S'].notna() & (v['sigma_S'] >= 5)].copy()
    # Paired fit: both arms are fitted on reservoirs where BOTH exist, so the gain divides coefficients
    # estimated on the same population.
    n_all = len(v)
    v = v[v['U1_rmse_bc'].notna() & v['U0_rmse_bc'].notna()].copy()
    print(f'paired fit: {n_all} -> {len(v)} reservoirs with BOTH arms')

    # ---- fit denominator (see UNC_SIGMA above) --------------------------------------
    # Reservoir SELECTION stays on full-record sigma_S in both modes, so the fit population
    # is identical and only the normalisation changes.
    v['sigma_fit'] = v['sigma_S']
    if UNC_SIGMA == 'window':
        pred_path = SM.parent / 'predictions.parquet'
        if not pred_path.exists():
            raise FileNotFoundError(f'UNC_SIGMA=window needs {pred_path}')
        pr = pd.read_parquet(pred_path, columns=['lake_id', 'true_mcm'])
        pr['lake_id'] = pr['lake_id'].astype(str)
        sw = pr.groupby('lake_id')['true_mcm'].std().rename('sigma_win')
        v = v.merge(sw, on='lake_id', how='left')
        n_before = len(v)
        v = v[v['sigma_win'] > 0].copy()
        ratio = v['sigma_win'] / v['sigma_S']
        print(f'  [UNC_SIGMA=window] window/full-record sigma ratio: '
              f'median {ratio.median():.3f}, p5 {ratio.quantile(.05):.3f}, '
              f'p95 {ratio.quantile(.95):.3f}')
        print(f'  dropped {n_before - len(v)} reservoir(s) with zero within-window variability')
        if UNC_MINCOV > 0:
            n2 = len(v)
            v = v[(v['sigma_win'] / v['sigma_S']) >= UNC_MINCOV].copy()
            print(f'  [UNC_MINCOV={UNC_MINCOV}] kept {len(v)} of {n2} reservoirs whose window '
                  f'covers >={UNC_MINCOV:.0%} of full-record variability')
        v['sigma_fit'] = v['sigma_win']

    print('=' * 64)
    print(f'Fitting sigma_S-anchored error model on {len(v)} validation reservoirs'
          + (' [sigma_S = MEASURED storage_std]' if meas is not None else '')
          + f' [fit denominator: {UNC_SIGMA}]')
    print('=' * 64)
    rand1, syst1 = split_errors(v, 'U1')
    rand0, syst0 = split_errors(v, 'U0')
    k = {}
    k['rand_U1'] = loo_check(rand1, v['sigma_fit'], 'random  SWOT (U1)')
    k['syst_U1'] = loo_check(syst1, v['sigma_fit'], 'systematic SWOT')
    k['rand_U0'] = loo_check(rand0, v['sigma_fit'], 'random  area (U0)')
    k['syst_U0'] = loo_check(syst0, v['sigma_fit'], 'systematic area')
    print('\nCoefficients (error per unit sigma_S):')
    for kk, vv in k.items():
        print(f'   k_{kk:8s} = {vv:.4f}')
    # The SWOT gain lives in the RANDOM (dynamic) component; the SYSTEMATIC component is similar
    # between arms because both share the same AEC/datum (it cancels in dynamic dS, degrades only the
    # absolute total). So the headline-critical check is the dynamic (random) gain.
    assert k['rand_U0'] > k['rand_U1'], 'area random error should exceed SWOT random error'
    print(f"   -> dynamic (random) SWOT gain: U0/U1 = {k['rand_U0']/k['rand_U1']:.2f}x; "
          f"systematic comparable ({k['syst_U0']:.2f} area vs {k['syst_U1']:.2f} SWOT, both AEC-limited)")

    # ---- apply to all usable reservoirs of the apply-universe ----
    ua = pd.read_csv(apply_univ, dtype={'lake_id': str})
    g = ua[(ua['swot_tier'] == 'usable') & ua['wse_std'].notna() & ua['AREA_SKM'].notna()
           & (ua['wse_std'] > 0)].copy()
    g = g.drop_duplicates('lake_id')
    g['sigma_S'] = g['wse_std'] * g['AREA_SKM']
    g['sigma_source'] = 'proxy'
    if meas is not None:
        g = g.merge(meas, on='lake_id', how='left')
        has = g['storage_std'].notna()
        g.loc[has, 'sigma_S'] = g.loc[has, 'storage_std']
        g.loc[has, 'sigma_source'] = 'measured'
        # same physical cap/2 bound as the fit set (proxy rows included: wse_std*AREA
        # inflates identically for datum-inconsistent multi-tile reservoirs)
        n_clamp = int((g['sigma_S'] > 0.5 * g['CAP_MCM'].fillna(np.inf)).sum())
        g['sigma_S'] = np.minimum(g['sigma_S'], 0.5 * g['CAP_MCM'].fillna(np.inf))
        print(f'apply universe: {int(has.sum()):,} measured / {int((~has).sum()):,} proxy-fallback; '
              f'{n_clamp:,} clamped at 0.5*CAP')
    g['sigma_rand_U1'] = k['rand_U1'] * g['sigma_S']
    g['bias_sys_U1'] = k['syst_U1'] * g['sigma_S']
    g['sigma_rand_U0'] = k['rand_U0'] * g['sigma_S']
    g['bias_sys_U0'] = k['syst_U0'] * g['sigma_S']

    # absolute-total anchor term (secondary; only where anchored)
    # S_ref comes from the production smoother file where it exists, and from the storage-anchor table
    # otherwise. Every reservoir needs a bed term: the aggregator uses nansum, so a missing one would
    # count as zero uncertainty.
    a = pd.read_csv(ANCHOR, dtype={'lake_id': str})[['lake_id', 'S_ref_mcm']]
    g = g.merge(a, on='lake_id', how='left')
    if meas is not None:
        full = pd.read_csv(measured_path, dtype={'lake_id': str})
        if 'S_ref_mcm' in full.columns:
            full = (full[['lake_id', 'S_ref_mcm']].dropna().drop_duplicates('lake_id')
                    .rename(columns={'S_ref_mcm': '_S_ref_full'}))
            g = g.merge(full, on='lake_id', how='left')
            n_before = int(g['S_ref_mcm'].notna().sum())
            g['S_ref_mcm'] = g['S_ref_mcm'].combine_first(g['_S_ref_full'])
            g = g.drop(columns='_S_ref_full')
            print(f'anchor coverage: {n_before:,} -> {int(g.S_ref_mcm.notna().sum()):,} '
                  f'reservoirs (production S_ref filled the gap)')
    # Keep the production S_ref for reference/coverage reporting, but base the BED TERM on
    # the gauge-calibrated capacity-fraction anchor where a capacity is known.
    g['S_ref_anchor_mcm'] = np.where(g['CAP_MCM'].notna() & (g['CAP_MCM'] > 0),
                                     ANCHOR_FILL * g['CAP_MCM'], g['S_ref_mcm'])
    g['anchor_sigma_mcm'] = ANCHOR_REL * g['S_ref_anchor_mcm']

    keep = ['lake_id', 'region', 'sigma_S', 'sigma_rand_U1', 'bias_sys_U1',
            'sigma_rand_U0', 'bias_sys_U0', 'S_ref_mcm', 'S_ref_anchor_mcm',
            'anchor_sigma_mcm']
    if 'sigma_source' in g.columns:
        keep.append('sigma_source')
    out = g[keep]
    out.to_csv(out_path, index=False)
    print(f'\nApplied to {len(out):,} usable reservoirs; {out.S_ref_mcm.notna().sum():,} have an '
          f'absolute anchor (for the secondary absolute total).')
    print(f'NaN sigma_S rows: {int(out.sigma_S.isna().sum())}')
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    import sys
    if '--legacy' in sys.argv:
        # Legacy: the v1 universe, the proxy sigma_S and the module defaults. Does not reproduce the
        # published numbers.
        main()
    else:
        # Default: the published model. sigma_S is the measured storage SD from the production smoother,
        # applied to the v2 universe, with the window fit denominator and 0.5 minimum window coverage.
        main(apply_univ=_paths.get("global_universe"),
             out_path=OUT,
             measured_path=_paths.get("smoother_measured_summary"))
