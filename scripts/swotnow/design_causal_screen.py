"""Choose a causal anchor screen on the TRAINING pool only.

A screen must be chosen without the held-out set, or its gain is partly a selection
effect. This script scores candidates on the gauged TRAINING reservoirs only; the held-out
reservoirs are scored once, after a screen has been chosen.

The candidate set (causal_screens.CANDIDATES) and this rule were fixed before running:

    choose the candidate with the highest median anomaly R2 on the training pool, PROVIDED
      (a) it beats C0 (no screen) by more than 0.02 in median anomaly R2,
      (b) the paired Wilcoxon signed-rank p against C0 is below 0.05 / 6 (Bonferroni: six
          candidates are compared against C0, and a training pool of ~225 reservoirs will
          throw up a 'significant' candidate by chance otherwise),
      (c) the median amplitude ratio stays within 0.90-1.10 (a screen that wins by damping the
          signal is a shrinkage artefact).
    If no candidate satisfies all three, the answer is C0.

Scoring is the anchor of Eq. (1) carried forward to the daily grid and compared with the gauge as
anomalies over the days both exist. No network is involved: the anchor is a deterministic function
of the SWOT pass series, so this isolates the screen exactly. The strict (causal) area from
build_causal_sa.py is used, so the design reflects the pipeline that will be trained.

A non-causal reference row (the two-sided Hampel screen, on the same area) is printed for scale
and is NOT a candidate.
"""
import sys
import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))      # run from a checkout
from swot_reservoir_storage.common import paths
from swot_reservoir_storage.swotnow.model.causal_screens import CANDIDATES
MEAN_FF = 0.01
MIN_GAUGE_DAYS = 30
K2, NS2 = 3, 3.0                       # the two-sided Hampel, for the reference row
ALPHA = 0.05 / 6
MIN_GAIN = 0.02


def two_sided_reference(t, x):
    """The two-sided Hampel reference (non-causal): k=3 each side, then interpolate across rejects."""
    n = len(x)
    if n < 10:
        return x.copy()
    bad = np.zeros(n, bool)
    for i in range(n):
        lo, hi = max(0, i - K2), min(n, i + K2 + 1)
        w = np.delete(x[lo:hi], min(i - lo, hi - lo - 1))
        if len(w) < 3:
            continue
        m = np.median(w); s = 1.4826 * np.median(np.abs(w - m))
        if s > 0 and abs(x[i] - m) > NS2 * s:
            bad[i] = True
    return np.interp(t, t[~bad], x[~bad]) if (~bad).sum() >= 3 else x.copy()


def anomaly_r2(e, o):
    e = e - e.mean(); o = o - o.mean()
    return 1 - np.sum((e - o) ** 2) / np.sum(o ** 2) if np.sum(o ** 2) > 0 else np.nan


def main():
    CAUSAL_SA = paths.get('causal_surface_area')
    OUT = paths.get('work_dir', must_exist=False) / 'swotnow' / 'causal_screen_design.csv'
    sp = pd.read_csv(paths.get('insitu_splits')).dropna(subset=['lake_id'])
    train = {str(int(v)) for v in sp.loc[sp.split == 'train', 'lake_id']}
    f = pd.read_parquet(paths.get('features'),
                        columns=['lake_id', 'date', 'on_swot_pass', 'wse', 'cap_mcm_eff', 'storage_mcm'])
    f['date'] = pd.to_datetime(f['date']); f['lake_id'] = f['lake_id'].astype(str)
    f = f[f.lake_id.isin(train)]
    sa = pd.read_parquet(CAUSAL_SA); sa['lake_id'] = sa['lake_id'].astype(str); sa['date'] = pd.to_datetime(sa['date'])
    sa = sa.set_index(['lake_id', 'date'])

    rows, rej_rows = [], []
    for lid, g in f.groupby('lake_id'):
        g = g.sort_values('date').reset_index(drop=True)
        cap = g.cap_mcm_eff.dropna()
        if cap.empty or cap.iloc[0] <= 0 or g.wse.notna().sum() < 3:
            continue
        cap = float(cap.iloc[0]); ref = float(np.median(g.wse.dropna()))
        p = g[(g.on_swot_pass == 1) & g.wse.notna()]
        gd = g[g.storage_mcm.notna()]
        if len(p) < 14 or len(gd) < 40:
            continue
        key = pd.MultiIndex.from_arrays([np.repeat(lid, len(p)), p['date']])
        area = sa.reindex(key)
        if area.sa_causal.isna().any():
            raise SystemExit(f'{lid}: causal area missing on {int(area.sa_causal.isna().sum())} passes')
        t = (p['date'] - p['date'].iloc[0]).dt.days.values.astype(float)
        x = (p.wse.values - ref) * area.sa_causal.values / cap + MEAN_FF
        a_raw = area.area_raw.values

        res = {'lake_id': lid, 'n_pass': len(p)}
        series = {name: fn(t, x, a_raw) for name, fn in CANDIDATES.items()}
        series['REF_two_sided'] = (two_sided_reference(t, x), np.zeros(len(x), bool))
        ok = True
        for name, (v, rj) in series.items():
            s = pd.Series(np.nan, index=g.index); s.loc[p.index] = v; s = s.ffill()
            m = gd.index[s.loc[gd.index].notna()]
            if len(m) < MIN_GAUGE_DAYS:
                ok = False; break
            e = s.loc[m].values * cap; o = gd.storage_mcm.loc[m].values
            res[name] = anomaly_r2(e, o)
            res[name + '|amp'] = float(np.std(e) / np.std(o))
            res[name + '|rej'] = float(rj.mean())
        if ok:
            rows.append(res)
    r = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    r.to_csv(OUT, index=False)
    n = len(r)
    print(f'training-pool gauged reservoirs scored: {n}  (held-out set not read)\n')

    base = r['C0_none']
    print(f"{'candidate':<18} {'median R2':>9} {'amp':>6} {'rejected':>9} {'vs C0':>8} {'higher on':>10} {'p':>9}")
    verdict = {}
    for name in list(CANDIDATES) + ['REF_two_sided']:
        med, amp, rj = r[name].median(), r[name + '|amp'].median(), r[name + '|rej'].mean()
        if name == 'C0_none':
            print(f'{name:<18} {med:>+9.3f} {amp:>6.3f} {100*rj:>8.1f}%   (baseline)')
            continue
        d = r[name] - base
        p = wilcoxon(r[name], base).pvalue
        tag = '   [non-causal reference, not a candidate]' if name == 'REF_two_sided' else ''
        print(f'{name:<18} {med:>+9.3f} {amp:>6.3f} {100*rj:>8.1f}% {d.median():>+8.3f} {100*(d>0).mean():>9.0f}% {p:>9.2g}{tag}')
        if name != 'REF_two_sided':
            verdict[name] = (med - base.median() > MIN_GAIN, p < ALPHA, 0.90 <= amp <= 1.10, med)

    print(f'\nrule: gain > {MIN_GAIN} over C0, Wilcoxon p < {ALPHA:.4f}, amplitude in [0.90, 1.10]')
    passing = {k: v[3] for k, v in verdict.items() if all(v[:3])}
    for k, (a, b, c, med) in verdict.items():
        print(f'  {k:<18} gain>{MIN_GAIN}: {a!s:<5}  p<alpha: {b!s:<5}  amp ok: {c!s:<5}')
    chosen = max(passing, key=passing.get) if passing else 'C0_none'
    print(f'\nCHOSEN: {chosen}')


if __name__ == '__main__':
    main()
