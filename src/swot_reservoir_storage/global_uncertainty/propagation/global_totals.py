#!/usr/bin/env python3
"""
analysis/global_storage_mc.py — Step 6: aggregate the per-reservoir error model into the global
headline: SWOT's net impact on reservoir-storage uncertainty, dU = U0 - U1 (km3).

Inputs the per-reservoir error model results/ml/uncertainty_model.csv (Step 5b):
  sigma_rand_{U1,U0}   random error (cancels in quadrature)        -> the DYNAMIC storage error
  bias_sys_{U1,U0}     systematic error (accumulates / correlated) -> degrades the ABSOLUTE total
  anchor_sigma_mcm     shared bathymetry/bed term (SWOT-independent; cancels in dU)

Two headline quantities (each under BOTH correlation bounds, global + per region):
  DYNAMIC  (lead, all 31,551): error = sigma_rand only (a constant per-reservoir offset cancels in
           dS), so this isolates SWOT's clean observational gain (k_rand area/SWOT ~ 2.9x).
  ABSOLUTE (secondary, anchored subset): error = sqrt(sigma_rand^2 + bias_sys^2) + shared bed term;
           SWOT's gain shrinks because the systematic/AEC/datum term is shared and large.

Correlation bracket (robustness): aggregate each error series two ways —
  independent : U = sqrt(sum e_i^2)                                  (errors cancel; narrow)
  correlated  : U = sqrt(sum_region (sum_{i in region} e_i)^2)       (region-common shift; wide)
Lead with the conservative (correlated) number; the gain holds under both.

A 1000-draw Monte-Carlo cross-checks the analytical dynamic aggregate.

Output: results/global_netimpact/{global_totals.csv, continental_breakdown.csv}
"""
import numpy as np
import pandas as pd
from pathlib import Path

from swot_reservoir_storage.common import paths as _paths
from swot_reservoir_storage.common.config import load as _load_config

_CFG = _load_config("global_uncertainty")

OUT = _paths.get("work_dir", must_exist=False) / "global_uncertainty"
OUT.mkdir(parents=True, exist_ok=True)
MC_DRAWS = int(_CFG.aggregation.draws)
RNG = np.random.default_rng(int(_CFG.aggregation.seed))


def agg(err, region, bound):
    """Aggregate a per-reservoir error series (MCM) -> km3, under a correlation bound."""
    err = np.asarray(err, float)
    if bound == 'independent':
        return np.sqrt(np.nansum(err ** 2)) / 1e3
    # correlated: fully correlated within region, independent across regions
    s = pd.Series(err).groupby(np.asarray(region)).sum()
    return np.sqrt(np.nansum(s.values ** 2)) / 1e3


def arm_U(df, arm, kind, bound):
    """Global-or-regional aggregated uncertainty (km3) for one arm/kind/bound."""
    rand = df[f'sigma_rand_{arm}']
    if kind == 'dynamic':
        err = rand
        bed = 0.0
    else:  # absolute: random + systematic per reservoir, plus the shared bed term
        err = np.sqrt(rand ** 2 + df[f'bias_sys_{arm}'] ** 2)
        bed = agg(df['anchor_sigma_mcm'].fillna(0.0), df['region'], bound)
    return np.hypot(agg(err, df['region'], bound), bed)


def report(df, label):
    row = dict(scope=label, n=len(df))
    for kind in ('dynamic', 'absolute'):
        for bound in ('independent', 'correlated'):
            u0 = arm_U(df, 'U0', kind, bound); u1 = arm_U(df, 'U1', kind, bound)
            tag = f'{kind[:3]}_{bound[:4]}'
            row[f'U0_{tag}'] = u0; row[f'U1_{tag}'] = u1; row[f'dU_{tag}'] = u0 - u1
    return row


def mc_dynamic(df):
    """Monte-Carlo cross-check of the dynamic aggregate (km3 std of the global total error)."""
    reg = df['region'].fillna('NA').values
    out = {}
    for arm in ('U0', 'U1'):
        s = df[f'sigma_rand_{arm}'].values
        # independent: each reservoir its own draw
        ind = RNG.normal(0, 1, (MC_DRAWS, len(s))) * s
        # correlated: one common factor per region
        codes, uniq = pd.factorize(reg)
        z = RNG.normal(0, 1, (MC_DRAWS, len(uniq)))
        cor = z[:, codes] * s
        out[arm] = (ind.sum(1).std() / 1e3, cor.sum(1).std() / 1e3)
    return out


def main(model_path=None, out_dir=OUT):
    # The error model writes per_reservoir_error_model.csv into the same directory this stage
    # writes its totals to, so the default input is derived from the output directory rather
    # than named twice.
    global OUT
    OUT = Path(out_dir); OUT.mkdir(parents=True, exist_ok=True)
    if model_path is None:
        model_path = OUT / 'per_reservoir_error_model.csv'
    m = pd.read_csv(model_path, dtype={'lake_id': str})
    m['region'] = m['region'].fillna('Unknown')
    print(f'reservoirs in headline set: {len(m):,}  '
          f'(anchored for absolute: {m.S_ref_mcm.notna().sum():,})')

    glob = report(m, 'GLOBAL')
    regs = [report(g, r) for r, g in m.groupby('region') if len(g) >= 3]
    allrows = pd.DataFrame([glob] + regs).round(2)
    allrows.to_csv(OUT / 'global_totals.csv', index=False)
    allrows[allrows.scope != 'GLOBAL'].to_csv(OUT / 'continental_breakdown.csv', index=False)

    g = glob
    print('\n' + '=' * 72)
    print('HEADLINE — SWOT net impact on global reservoir storage UNCERTAINTY')
    print('=' * 72)
    print(f"\nGlobal, {g['n']:,} SWOT-usable reservoirs")
    print('  DYNAMIC storage (lead; constant offsets cancel) [km3]:')
    print(f"    correlated  : without SWOT U0={g['U0_dyn_corr']:8.1f}  with SWOT U1={g['U1_dyn_corr']:8.1f}"
          f"  -> SWOT cuts uncertainty by {g['dU_dyn_corr']:7.1f} km3 "
          f"({100*g['dU_dyn_corr']/g['U0_dyn_corr']:.0f}%)")
    print(f"    independent : without SWOT U0={g['U0_dyn_inde']:8.2f}  with SWOT U1={g['U1_dyn_inde']:8.2f}"
          f"  -> dU={g['dU_dyn_inde']:7.2f} km3 ({100*g['dU_dyn_inde']/g['U0_dyn_inde']:.0f}%)")
    print(f"  ABSOLUTE total (secondary; +systematic +shared bed) [km3]:")
    print(f"    correlated  : U0={g['U0_abs_corr']:8.1f}  U1={g['U1_abs_corr']:8.1f}  dU={g['dU_abs_corr']:7.1f}")
    print(f"    independent : U0={g['U0_abs_inde']:8.1f}  U1={g['U1_abs_inde']:8.1f}  dU={g['dU_abs_inde']:7.1f}")

    mc = mc_dynamic(m)
    print('\nMonte-Carlo cross-check of DYNAMIC aggregate (km3; should match analytical):')
    print(f"   U0 indep MC={mc['U0'][0]:.2f} vs analytic {g['U0_dyn_inde']:.2f} | "
          f"corr MC={mc['U0'][1]:.1f} vs {g['U0_dyn_corr']:.1f}")
    print(f"   U1 indep MC={mc['U1'][0]:.2f} vs analytic {g['U1_dyn_inde']:.2f} | "
          f"corr MC={mc['U1'][1]:.1f} vs {g['U1_dyn_corr']:.1f}")

    print('\nBy region — DYNAMIC, correlated bound [km3]:')
    for r in sorted(regs, key=lambda x: -x['dU_dyn_corr']):
        print(f"   {r['scope']:14s} n={r['n']:6d}  U0={r['U0_dyn_corr']:7.1f}  U1={r['U1_dyn_corr']:7.1f}"
              f"  dU={r['dU_dyn_corr']:7.1f}")
    print(f"\nSaved: {OUT}/global_totals.csv , continental_breakdown.csv")


if __name__ == '__main__':
    import sys
    # To write elsewhere, configure a different work_dir.
    main()
