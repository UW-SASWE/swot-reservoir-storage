"""Acceptance test: nothing dated after day x may influence SWOTNOW's input for day x.

Run it under the committed configuration (config/swotnow.yaml, which is the strict one):

  python tests/test_no_lookahead.py

It exits non-zero if any check fails. To confirm the test can fail, put `screen: hampel_interp` in
config/swotnow.local.yaml and run it again: with that two-sided screen it must FAIL, or the test is
not testing anything.

What is tested, channel by channel (DYN_FEATS = the network's nine time-varying inputs):

  anchor_ff          T1 coverage of the causal area, T2 the screen, T3 the area smoother, and
                     T4 the assembled channel under deletion of everything after day t
  hls_water_frac     T5 recomputed as-of from the raw HLS files
  days_since_swot    T5 recomputed as-of from the pass dates
  precip_routed      T4 routing the truncated precipitation gives the same values
  sin_doy, cos_doy   T5 equal to the calendar formula
  sar_water_frac     T5 constant (no SAR feature file is loaded in this configuration)
  has_hls, has_sar   per-reservoir existence flags: DECLARED STATIC, checked constant only

DECLARED STATIC CONSTANTS (computed from the full record, held equal by design, and not tested
for look-ahead): the reference elevation wse_ref; the smoother's window width round(0.3 * n_passes);
the static attributes including log_wse_std; the anomaly-centring offsets; the normalisation
statistics; and the by-reservoir train/test split. Methods must say so.

T4 truncates the rows; the parquet-derived channels were built once on the full timeline, so a
two-sided operation inside build_features would not show up under truncation. That is why T5
recomputes those channels independently from the raw files instead.
"""
import sys
import numpy as np, pandas as pd
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'src'))      # run from a checkout without installing
from swot_reservoir_storage.common import paths
from swot_reservoir_storage.swotnow.model import trainer as T
from swot_reservoir_storage.swotnow.model.causal_screens import causal_screen
from swot_reservoir_storage.swotnow.preprocessing.causal_area import causal_smooth, window_width

HLS_DIR = paths.get('hls_area')
SWOT_START = pd.Timestamp('2023-08-01')
HLS_MAX_GAP = 10
LONG_WIN = T.LONG_WIN
RNG = np.random.default_rng(0)
FAILS = []


def report(name, n, bad, extra=''):
    ok = bad == 0
    if not ok:
        FAILS.append(name)
    print(f'  {"PASS" if ok else "FAIL"}  {name:<46} {bad:>6} mismatches of {n:<7} {extra}')


def screened_values(t, x, a):
    """The active screen applied to a pass series; returns the repaired series."""
    if T.ANCHOR_SCREEN.startswith('causal_'):
        return causal_screen(T.ANCHOR_SCREEN[len('causal_'):], t, x, a)[0]
    if T.ANCHOR_SCREEN in ('hampel', 'hampel_interp') and len(x) >= 10:
        if T.ANCHOR_SCREEN == 'hampel':
            return T._hampel(x, T.ANCHOR_SCREEN_K, T.ANCHOR_SCREEN_N)
        return T._hampel_interp(x, t, T.ANCHOR_SCREEN_K, T.ANCHOR_SCREEN_N)
    return np.asarray(x, float).copy()


def main():
    print(f'configuration: ANCHOR_SCREEN={T.ANCHOR_SCREEN!r}  CAUSAL_SA={T.CAUSAL_SA}  '
          f'STRICT={T.CAUSAL_SA_STRICT}  table={Path(T.CAUSAL_SA_FILE).name}')
    strict = T.CAUSAL_SA and T.CAUSAL_SA_STRICT and T.ANCHOR_SCREEN.startswith('causal_')
    print('  ' + ('STRICT configuration: every check must pass' if strict else
                  'NOT the strict configuration: failures below are expected'))

    df = pd.read_parquet(paths.get('features'))
    df['date'] = pd.to_datetime(df['date']); df['lake_id'] = df['lake_id'].astype(str)
    df = df.sort_values(['lake_id', 'date']).reset_index(drop=True)
    tab = pd.read_parquet(T.CAUSAL_SA_FILE)
    tab['lake_id'] = tab['lake_id'].astype(str); tab['date'] = pd.to_datetime(tab['date'])
    tab = tab.set_index(['lake_id', 'date'])
    has_raw = 'area_raw' in tab.columns
    by = {lid: g.reset_index(drop=True) for lid, g in df.groupby('lake_id')}
    lids = sorted(by)

    # ---------------------------------------------------------------- T1 coverage
    print('\nT1  causal-area coverage of SWOT pass rows')
    p = df[(df.on_swot_pass == 1) & df.wse.notna()]
    key = pd.MultiIndex.from_arrays([p.lake_id, p.date])
    covered = tab.sa_causal.reindex(key).notna().values
    report('every pass row has a causal area', len(p), int((~covered).sum()),
           f'coverage {100 * covered.mean():.2f}%')

    # ---------------------------------------------------------------- T2 screen truncation
    print('\nT2  the screen: value at pass j from data <= j equals value from the full record')
    n = bad = 0
    for lid in lids[::3]:
        g = by[lid]
        pp = g[(g.on_swot_pass == 1) & g.wse.notna()]
        if len(pp) < 14:
            continue
        k = pd.MultiIndex.from_arrays([np.repeat(lid, len(pp)), pp.date])
        tb = tab.reindex(k)
        if tb.sa_causal.isna().any():
            continue
        cap = g.cap_mcm_eff.dropna()
        if cap.empty or cap.iloc[0] <= 0:
            continue
        ref = float(np.median(g.wse.dropna()))
        x = (pp.wse.values - ref) * tb.sa_causal.values / float(cap.iloc[0]) + T.UNGAUGED_MEAN_FF
        a = tb.area_raw.values if has_raw else np.full(len(x), np.nan)
        t = (pp.date - pp.date.iloc[0]).dt.days.values.astype(float)
        full = screened_values(t, x, a)
        for j in range(9, len(x) - 3):
            n += 1
            if abs(screened_values(t[:j + 1], x[:j + 1], a[:j + 1])[j] - full[j]) > 1e-12:
                bad += 1
    report('screened anchor value', n, bad, f'({100 * bad / max(n, 1):.1f}%)')

    # ---------------------------------------------------------------- T3 smoother truncation
    print('\nT3  the area smoother: value at pass j uses only passes <= j (window width held fixed)')
    if not has_raw:
        print('  SKIP  table has no area_raw column')
    else:
        n = bad = 0
        for lid in lids[::3]:
            g = by[lid]
            pp = g[(g.on_swot_pass == 1) & g.wse.notna()]
            if len(pp) < 14:
                continue
            k = pd.MultiIndex.from_arrays([np.repeat(lid, len(pp)), pp.date])
            y = tab.reindex(k).area_raw.values
            t = (pp.date - pp.date.iloc[0]).dt.days.values.astype(float)
            w = window_width(len(y))
            full = causal_smooth(t, y, w=w)
            for j in range(9, len(y)):
                n += 1
                if abs(causal_smooth(t[:j + 1], y[:j + 1], w=w)[j] - full[j]) > 1e-9 * max(1.0, abs(full[j])):
                    bad += 1
        report('smoothed area', n, bad)

    # ---------------------------------------------------------------- T4 deletion via build_daily
    print('\nT4  delete every observation after day t, rebuild, compare rows <= t (per channel)')
    chans = T.DYN_FEATS
    counts = {c: [0, 0, 0.0] for c in chans}              # n compared, n mismatched, max abs diff
    used = 0
    picks = RNG.choice(len(lids), size=len(lids), replace=False)
    for ix in picks:
        if used >= 30:
            break
        lid = lids[ix]
        sub = by[lid]
        pp = sub[(sub.on_swot_pass == 1) & sub.wse.notna()]
        if len(pp) < 30 or sub.wse.notna().sum() < 3:
            continue
        ref = float(np.median(sub.wse.dropna()))
        pr_path = T.PRECIP_DIR / f'{lid}.csv'
        pr = None
        if pr_path.exists():
            pr = pd.read_csv(pr_path, parse_dates=['date']).set_index('date')['precip_mm']

        def route(series):
            return None if series is None else pd.Series(T.route_precip(series.values, T.TAU), index=series.index)
        full = T.build_daily(sub, route(pr), wse_ref=ref)
        if full is None:
            continue
        # cut points: after the warm-up window, with at least 3 later passes, so look-ahead could matter
        cand = pp.date.iloc[len(pp) // 3: len(pp) - 3].values
        cand = [pd.Timestamp(c) for c in cand if pd.Timestamp(c) >= sub.date.iloc[0] + pd.Timedelta(days=LONG_WIN)]
        if len(cand) < 3:
            continue
        cuts = [cand[i] for i in np.linspace(0, len(cand) - 1, 5).astype(int)]
        made = False
        for tcut in cuts:
            sub_t = sub[sub.date <= tcut]
            if sub_t.storage_mcm.notna().sum() < 5:
                continue
            pr_t = None if pr is None else pr[pr.index <= tcut]
            tr = T.build_daily(sub_t, route(pr_t), wse_ref=ref)
            if tr is None:
                continue
            m = len(tr)
            for c in chans:
                a, b = full[c].values[:m].astype(float), tr[c].values.astype(float)
                d = np.nanmax(np.abs(a - b)) if np.isfinite(a - b).any() else 0.0
                mism = int((~np.isclose(a, b, rtol=0, atol=1e-12, equal_nan=True)).sum())
                counts[c][0] += m; counts[c][1] += mism; counts[c][2] = max(counts[c][2], float(d))
            made = True
        used += made
    print(f'  reservoirs tested: {used}, five cut dates each')
    for c in chans:
        n, b, mx = counts[c]
        report(f'{c}', n, b, f'max |diff| {mx:.3g}')

    # ---------------------------------------------------------------- T5 as-of recomputation
    print('\nT5  channels built upstream, recomputed independently as-of each day')
    ds_n = ds_b = hl_n = hl_b = tr_n = tr_b = sar_b = sar_n = const_b = const_n = 0
    for lid in lids[::4]:
        g = by[lid]
        dts = g.date.values
        # days_since_swot from the pass dates
        passd = g.date.where((g.on_swot_pass == 1) & g.wse.notna())
        last = passd.ffill()
        exp = (g.date - last).dt.days.astype(float)
        got = g.days_since_swot.astype(float)
        ok = np.isclose(exp.values, got.values, equal_nan=True)
        ds_n += len(g); ds_b += int((~ok).sum())
        # sin/cos of day of year
        doy = g.date.dt.dayofyear.values
        tr_n += 2 * len(g)
        tr_b += int((~np.isclose(np.sin(2 * np.pi * doy / 365.25), g.sin_doy.values, atol=1e-9)).sum()
                    + (~np.isclose(np.cos(2 * np.pi * doy / 365.25), g.cos_doy.values, atol=1e-9)).sum())
        # hls: last raw observation <= t within HLS_MAX_GAP days, else NaN
        hp = HLS_DIR / f'{lid}.csv'
        if hp.exists() and hp.stat().st_size >= 50:
            h = pd.read_csv(hp, parse_dates=['time']).rename(columns={'time': 'date'})
            h['date'] = pd.to_datetime(h.date).dt.normalize()
            h = h.dropna(subset=['water_area_km2']).sort_values('date')
            h = h[h.date >= SWOT_START].drop_duplicates('date').set_index('date').water_area_km2
            if len(h):
                obs = h.reindex(g.date)
                lastd = pd.Series(g.date.where(obs.notna().values).values, index=g.index).ffill()
                age = (g.date - lastd).dt.days.astype(float)
                asof = pd.Series(obs.ffill().values, index=g.index).where(age <= HLS_MAX_GAP)
                ok = np.isclose(asof.values, g.hls_water_area_km2.values.astype(float), equal_nan=True)
                hl_n += len(g); hl_b += int((~ok).sum())
        # static flags and the SAR channel
        d = T.build_daily(g, None, wse_ref=None)
        if d is not None:
            sar_n += len(d); sar_b += int((d.sar_water_frac.values != 0.0).sum())
            for c in ('has_hls', 'has_sar'):
                const_n += 1; const_b += int(d[c].nunique() > 1)
    report('days_since_swot (as-of pass dates)', ds_n, ds_b)
    report('hls_water_area (as-of raw HLS files)', hl_n, hl_b)
    report('sin_doy, cos_doy (calendar)', tr_n, tr_b)
    report('sar_water_frac constant zero', sar_n, sar_b)
    report('has_hls, has_sar constant per reservoir', const_n, const_b)

    print('\n' + ('ALL CHECKS PASSED' if not FAILS else f'FAILED: {", ".join(FAILS)}'))
    sys.exit(1 if FAILS else 0)


if __name__ == '__main__':
    main()
