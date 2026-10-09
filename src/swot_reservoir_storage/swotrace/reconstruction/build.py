#!/usr/bin/env python3
"""
SWOTRACE — SWOT-Trained Reanalysis of Area-derived Capacity Estimates.

Monthly reservoir storage, 1984-2021, for reservoirs outside the GRanD catalogue.

STANDALONE. Everything the build needs is in this one file: if a function is here it runs, and if
it is not here it is not part of the product.

WHAT IT BUILDS
    Monthly water storage for reservoirs that lie OUTSIDE the GRanD catalogue of
    documented dams, from 1984 to 2021 -- the decades before SWOT existed. These reservoirs
    have no published storage product of any kind: GRS covers GRanD only, and few of
    these reservoirs have the recorded dam height that geometric-template methods require.

HOW IT WORKS
    1. SWOT measured water level and surface area together, 2023 onward, giving each
       reservoir's own area-elevation relationship over the band SWOT happened to observe.
    2. That band is narrow. An SRTM-derived curve supplies reach above and below it, aligned
       to SWOT's vertical datum and rescaled to meet SWOT's areas at the join.
    3. The combined curve is integrated to volume, then historical satellite water-extent
       area (1984-2021) is read through it to give monthly storage.

    SWOT never observed the historical period. Its entire contribution is the SHAPE of each
    reservoir's curve; the time variation comes from the optical area record.

SCOPE, AND THE LIMIT THAT MATTERS
    This is an ANOMALY / SHAPE product. Absolute storage is NOT validated -- use the
    variation, not the level. Because storage here is a re-scaling of surface area, no curve
    can beat the quality of the area input.

DATUM SIGN
    swot_datum_shift returns d = median(h_srtm(a) - h_swot(a)), so SRTM reads HIGHER than
    SWOT by d, and the SRTM elevation matching a SWOT-frame elevation g is (g + d).

Output goes under work_dir/swotrace (see config/paths.yaml).
"""
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.isotonic import IsotonicRegression
import warnings
warnings.filterwarnings("ignore")

from swot_reservoir_storage.common import paths as _paths
from swot_reservoir_storage.common.config import load as _load_config

_CFG = _load_config("swotrace")

# ── inputs ──────────────────────────────────────────────────────────────────────

HYPS     = _paths.get("global_hypsometry")
UNIV     = _paths.get("global_universe")
AEC_DIR  = _paths.get("srtm_aec")
SWOT_DIR = _paths.get("swot_hydrocron_combined")
ZG_DIR   = _paths.get("gsw_cloud_corrected")
GSW_DIR  = _paths.get("gsw_monthly")
OUT_DIR  = _paths.get("work_dir", must_exist=False) / "swotrace"
OUT_DIR.mkdir(parents=True, exist_ok=True)
COMBINED = SWOT_DIR                       # SWOT Hydrocron cache (alias kept for clarity)

# ── thresholds ──────────────────────────────────────────────────────────────────
N_GRID     = _CFG.curve.n_grid
MIN_MONTHS = _CFG.area.min_months
VF_FLOOR   = float(_CFG.area.valid_fraction_floor)
PX_KM2     = float(_CFG.area.pixel_area_km2)
MIN_BAND_M = _CFG.curve.min_band_m
NBINS      = _CFG.curve.n_bins
WSE_STD_LOW_MID  = _CFG.confidence.wse_std_low_mid
WSE_STD_MID_HIGH = _CFG.confidence.wse_std_mid_high

# ── curve inputs ────────────────────────────────────────────────────────────────

def srtm_curve(lid):
    """RAT/TMS-OS AEC file -> (h[], S_mcm[], A_km2[]). Absolute in both axes.

    RAT's Elevation=f(CumArea) polynomial fit is constrained to be monotonic
    only on the grid it was FIT on; the shipped CSV's output grid is unioned
    with the raw observed CumArea range (rat's extrapolate_reservoir, via
    np.unique(np.concatenate((x_pred, aec['CumArea'].values)))), which can run
    past where the monotonicity constraint was actually checked. A quadratic
    fit then folds back past its vertex -- confirmed directly against the raw
    file for one reservoir (Stampede, 7730007313): Elevation rises to a peak
    near CumArea=10.5 then FALLS back down through CumArea=18.3, i.e. the same
    elevation maps to two different areas. Sorting that by Elevation (as this
    function must, to build h(A)) interleaves the two branches into a sawtooth.
    Rare, but a real defect of the shipped data, not a smoothing artifact -- truncate at the fold rather than trust
    the curve past it, since nothing constrains the polynomial out there.
    """
    p = AEC_DIR / f"{lid}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, comment="#", on_bad_lines="skip")
    need = ["Elevation", "CumArea", "Storage (mil. m3)"]
    if any(c not in df for c in need):
        return None
    for c in need:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=need).sort_values("CumArea")
    if len(df) < 10:
        return None
    e = df["Elevation"].values
    fold = np.where(np.diff(e) < -1e-6)[0]
    if len(fold):
        df = df.iloc[: fold[0] + 1]
    if len(df) < 10:
        return None
    df = df.sort_values("Elevation")
    return (df["Elevation"].values, df["Storage (mil. m3)"].values,
            df["CumArea"].values)

def swot_datum_shift(h_sw, A_sw, h_sr, A_sr):
    """Vertical shift aligning the SRTM curve to SWOT's own measurements.

    Gauge-free by construction. Both A(h) curves are monotonic in h over the
    range that matters, so they invert to h(A); for a given physical AREA the
    two curves should report the same elevation up to a constant datum shift.
    Matching in the AREA domain (median of h_sr(a) - h_sw(a) over the shared
    area range) is robust to SRTM's area-SCALE bias in a way that matching
    A(h) curve SHAPES directly is not: shape-matching over a narrow, near-
    linear SWOT band is underdetermined and can latch onto noise.
    """
    # restrict to the area range both curves actually cover
    a_lo = max(A_sw.min(), A_sr.min())
    a_hi = min(A_sw.max(), A_sr.max())
    if a_hi <= a_lo:
        return 0.0, 1.0
    ga = np.linspace(a_lo, a_hi, 60)
    # invert via interpolation: A_sw/A_sr are monotonic non-decreasing in h,
    # so (A, h) sorted by A gives a valid function h(A)
    h_sw_at_a = np.interp(ga, np.sort(A_sw), h_sw[np.argsort(A_sw)])
    h_sr_at_a = np.interp(ga, np.sort(A_sr), h_sr[np.argsort(A_sr)])
    d = float(np.median(h_sr_at_a - h_sw_at_a))

    # scale: after aligning datum, compare A(h) magnitudes over SWOT's own band.
    # A ratio-of-medians (not a median-of-ratios) so points where SRTM's area is
    # near zero don't blow the estimate up via division; still guarded and
    # capped, since a single multiplicative constant fit from a narrow overlap
    # is inherently fragile and an extreme value would corrupt the whole curve.
    lo, hi = h_sw.min(), h_sw.max()
    gh = np.linspace(lo, hi, 60)
    a_target = np.interp(gh, h_sw, A_sw)
    a_try = np.interp(gh - d, h_sr, A_sr)
    denom = float(np.median(a_try))
    scale = float(np.median(a_target)) / denom if denom > 1e-3 else 1.0
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    scale = float(np.clip(scale, 0.1, 10.0))
    return d, scale

def confidence_tier(wse_std):
    if wse_std <= WSE_STD_LOW_MID:
        return "low"
    if wse_std <= WSE_STD_MID_HIGH:
        return "mid"
    return "high"

def swot_curve_combined(lid):
    p = COMBINED / f"{lid}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if df.empty or "wse" not in df or "area" not in df:
        return None
    df["wse"] = pd.to_numeric(df["wse"], errors="coerce")
    df["area"] = pd.to_numeric(df["area"], errors="coerce")
    df = df[(df["wse"] > -1e6)].dropna(subset=["wse", "area"])
    if len(df) < 20:
        return None
    a, w = df["area"].values, df["wse"].values
    if np.nanstd(a) <= 0 or np.nanstd(w) <= 0:
        return None
    iso = IsotonicRegression(out_of_bounds="clip").fit(a, w)
    order = np.argsort(a); a_s = a[order]; h_s = iso.predict(a_s)
    d = pd.DataFrame({"h": h_s, "a": a_s}).groupby("h", as_index=False)["a"].mean().sort_values("h")
    if len(d) < 5:
        return None
    h, A = d["h"].values, d["a"].values
    if (h[-1] - h[0]) < 0.5:
        return None
    S = np.concatenate([[0.0], np.cumsum(np.diff(h) * (A[1:] + A[:-1]) / 2.0)])
    return h, S, A

def gsw_area(lid):
    p = GSW_DIR / f"{lid}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, usecols=["month", "gsw_water_area_km2", "valid_frac"])
    df = df[df["valid_frac"] >= 0.5].dropna(subset=["gsw_water_area_km2"])
    if len(df) < 12:
        return None
    return df.set_index("month")["gsw_water_area_km2"]

# ── the hybrid curve (this is the live builder) ─────────────────────────────────

def srtm_curve_fixed(lid):
    """RAT AEC with storage re-derived as the integral of A dh.

    Returns (h[], S_mcm[], A_km2[]) sorted by elevation, or None. Storage is
    referenced to the curve's own lowest elevation, the same convention RAT's
    column used, so only the shape changes.
    """
    sr = srtm_curve(lid)
    if sr is None:
        return None
    h_sr, _S_published, A_sr = sr
    o = np.argsort(h_sr)
    h, A = h_sr[o], A_sr[o]
    # km^2 * m == million m^3, so no unit conversion is needed here
    S = np.concatenate([[0.0], np.cumsum(np.diff(h) * (A[1:] + A[:-1]) / 2.0)])
    return h, S, A

def aligned_area_scale(h_sw, A_sw, h_sr, A_sr, d_shift):
    """Area scale re-estimated on the CORRECTLY aligned curve (fix 2).

    swot_datum_shift's own `scale` is computed with the wrong sign internally,
    so it cannot be reused; it is recomputed here over SWOT's observed band.
    """
    lo, hi = float(h_sw.min()), float(h_sw.max())
    g = np.linspace(lo, hi, 60)
    target = np.interp(g, h_sw, A_sw)
    attempt = np.interp(g + d_shift, h_sr, A_sr)
    m_att = float(np.median(attempt))
    if m_att <= 1e-9:
        return 1.0
    return float(np.clip(float(np.median(target)) / m_att, 0.1, 10.0))

def build_hybrid_curve(lid):
    """Returns (gh, a_h, s_h, d_shift, scale, band_frac, srtm_cover) or None."""
    sr = srtm_curve_fixed(lid)
    sw = swot_curve_combined(lid)
    if sr is None or sw is None:
        return None
    h_sr, S_sr, A_sr = sr
    h_sw, S_sw, A_sw = sw
    d_shift, _scale_bad = swot_datum_shift(h_sw, A_sw, h_sr, A_sr)
    scale = aligned_area_scale(h_sw, A_sw, h_sr, A_sr, d_shift)

    # v2: span the UNION of SRTM's range and SWOT's observed band
    h_lo = min(float(h_sr.min()), float(h_sw.min()))
    h_hi = max(float(h_sr.max()), float(h_sw.max()))
    gh = np.linspace(h_lo, h_hi, N_GRID)

    # v3 FIX 2: SRTM's area at a SWOT-frame elevation g is interp(g + d, ...)
    a_sc = np.interp(gh + d_shift, h_sr, A_sr) * scale
    lo, hi = float(h_sw.min()), float(h_sw.max())
    inside = (gh >= lo) & (gh <= hi)
    if inside.sum() < 3:
        return None
    # diagnostic: is SRTM actually defined across SWOT's band now?
    x_in = gh[inside] + d_shift
    srtm_cover = float(((x_in >= h_sr.min()) & (x_in <= h_sr.max())).mean())

    a_h = a_sc.copy()
    a_h[inside] = np.interp(gh[inside], h_sw, A_sw)
    for side, edge in ((gh < lo, lo), (gh > hi, hi)):
        if side.sum() == 0:
            continue
        a_edge_sw = float(np.interp(edge, h_sw, A_sw))
        a_edge_sr = float(np.interp(edge + d_shift, h_sr, A_sr) * scale)
        if a_edge_sr > 1e-6:
            a_h[side] = a_h[side] * (a_edge_sw / a_edge_sr)

    # v2: physically valid curve -- non-negative, non-decreasing in elevation
    a_h = np.maximum.accumulate(np.clip(a_h, 0.0, None))
    s_h = np.concatenate([[0.0], np.cumsum(np.diff(gh) * (a_h[1:] + a_h[:-1]) / 2.0)])
    # anchor on SRTM's own (now correctly integrated) storage, in SWOT's frame
    h_sr_frame = h_sr - d_shift
    of = np.argsort(h_sr_frame)
    s_h = s_h + float(np.interp(gh[0], h_sr_frame[of], S_sr[of]))

    order = np.argsort(a_h)
    a_hs, s_hs, gh_s = a_h[order], s_h[order], gh[order]
    a_hs, idx = np.unique(a_hs, return_index=True)
    s_hs, gh_s = s_hs[idx], gh_s[idx]
    if len(a_hs) < 5:
        return None
    return gh_s, a_hs, s_hs, float(d_shift), float(scale), \
        float(inside.sum()) / len(gh), srtm_cover

# ── historical area ─────────────────────────────────────────────────────────────

def areas_from_hist(df):
    """raw / uniform / zhaogao monthly area (km2) from the occurrence histograms.

    Bins are ordered by occurrence ASCENDING, so the highest-occurrence bins are
    the LAST ones -- water fills from the top down.
    """
    W = df[[f"water_{b}" for b in range(NBINS)]].values
    L = df[[f"land_{b}" for b in range(NBINS)]].values
    N = df[[f"nodata_{b}" for b in range(NBINS)]].values
    n_w, n_l, n_n = W.sum(1), L.sum(1), N.sum(1)
    visible = n_w + n_l
    total = visible + n_n

    raw = n_w * PX_KM2
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(visible > 0, n_w / visible, np.nan)
    uniform = frac * total * PX_KM2

    # Zhao-Gao: walk bins from highest occurrence down until the cumulative
    # VISIBLE count reaches the visible water count -> that is the threshold bin;
    # then add the hidden pixels above it.
    zg = np.full(len(df), np.nan)
    Wc = W[:, ::-1].cumsum(1)          # descending occurrence
    Vc = (W + L)[:, ::-1].cumsum(1)
    Nc = N[:, ::-1].cumsum(1)
    for i in range(len(df)):
        if visible[i] <= 0 or total[i] <= 0:
            continue
        k = int(np.searchsorted(Vc[i], n_w[i]))
        k = min(k, NBINS - 1)
        hidden_water = Nc[i, k]
        # linear interpolation inside the threshold bin, so the estimate is not
        # quantised to the bin edges
        if k > 0:
            span = Vc[i, k] - Vc[i, k-1]
            if span > 0:
                f = np.clip((n_w[i] - Vc[i, k-1]) / span, 0, 1)
                hidden_water = Nc[i, k-1] + f * (Nc[i, k] - Nc[i, k-1])
        zg[i] = (n_w[i] + hidden_water) * PX_KM2
    return raw, uniform, zg, np.where(total > 0, visible / total, np.nan)

def area_series(lid):
    """(pd.Series indexed by month, source). Zhao-Gao if available, else raw GSW.

    Both branches use the vf>=0.05 FLOOR, not a 0.5 gate (which measures polygon oversize, not data quality). The floor
    exists only to drop months that are essentially all cloud, where the
    reconstruction has no visible pixels to calibrate its threshold against.
    """
    p = ZG_DIR / f"{lid}.csv"
    if p.exists():
        d = pd.read_csv(p)
        need = [f"{k}_{b}" for k in ("water", "land", "nodata") for b in range(NBINS)]
        if all(c in d.columns for c in need):
            _raw, _unif, zg, vf = areas_from_hist(d)
            s = pd.DataFrame({"month": d["month"].astype(str), "a": zg, "vf": vf})
            s = s[(s.vf >= VF_FLOOR) & np.isfinite(s.a) & (s.a > 0)]
            if len(s) >= MIN_MONTHS:
                return s.set_index("month")["a"], "zhaogao"
    q = GSW_DIR / f"{lid}.csv"
    if not q.exists():
        return None, None
    d = pd.read_csv(q, usecols=["month", "gsw_water_area_km2", "valid_frac"])
    d = d.dropna(subset=["gsw_water_area_km2"])
    d = d[(d.valid_frac >= VF_FLOOR) & (d.gsw_water_area_km2 > 0)]
    if len(d) < MIN_MONTHS:
        return None, None
    return d.set_index("month")["gsw_water_area_km2"], "gsw_raw"

# ── build ───────────────────────────────────────────────────────────────────────

def main():
    ht = pd.read_csv(HYPS, dtype={"lake_id": str})
    gu = pd.read_csv(UNIV, dtype={"lake_id": str}, low_memory=False)
    gu = gu.dropna(subset=["lake_id"]).drop_duplicates("lake_id")
    m = ht.merge(gu[["lake_id", "has_grand_link", "swot_tier"]], on="lake_id", how="left")
    ng = m[m["has_grand_link"] == False]                                    # noqa: E712
    have_area = {p.stem for p in GSW_DIR.glob("*.csv")} | {p.stem for p in ZG_DIR.glob("*.csv")}
    target = ng[((ng["selected"] == 1) | (ng["swot_tier"] == "usable"))
                & ng["lake_id"].isin(have_area)].drop_duplicates("lake_id")
    lids = target["lake_id"].tolist()
    gate1 = dict(zip(target["lake_id"], target["selected"] == 1))
    wse_map = dict(zip(target["lake_id"], target["wse_std"]))
    print(f"Target population: {len(lids)} non-GRanD with area "
          f"({sum(gate1.values())} Gate-1, {len(lids)-sum(gate1.values())} added by v4)", flush=True)

    curve_rows, ts_rows, meta_rows, log_rows = [], [], [], []
    for i, lid in enumerate(lids, 1):
        built = build_hybrid_curve(lid)
        if built is None:
            log_rows.append(dict(lake_id=lid, status="curve_build_failed")); continue
        gh, a_h, s_h, d_shift, scale, band_frac, srtm_cover = built
        area_s, src = area_series(lid)
        if area_s is None:
            log_rows.append(dict(lake_id=lid, status="no_area")); continue

        for h_i, a_i, s_i in zip(gh, a_h, s_h):
            curve_rows.append((lid, h_i, a_i, s_i))
        av = area_s.values.astype(float)
        a_lo, a_hi = float(a_h.min()), float(a_h.max())
        below, above = av < a_lo, av > a_hi
        in_range = ~(below | above)
        st = np.full(len(av), np.nan)
        if in_range.any():
            st[in_range] = np.interp(av[in_range], a_h, s_h)
        for month, s_i, a_i, ok in zip(area_s.index, st, av, in_range):
            ts_rows.append((lid, month, float(s_i), float(a_i), bool(ok)))

        s_valid = st[in_range]
        storage_std = float(np.std(s_valid)) if in_range.sum() > 1 else 0.0
        frac_in = float(in_range.mean())
        meta_rows.append(dict(
            lake_id=lid, wse_std=wse_map[lid], confidence_tier=confidence_tier(wse_map[lid]),
            gate1_pass=bool(gate1[lid]), area_source=src,
            swot_datum_shift_m=d_shift, swot_area_scale=scale,
            swot_band_frac_of_curve=band_frac, srtm_cover_of_swot_band=srtm_cover,
            n_months=len(area_s), frac_in_range=frac_in,
            frac_below_floor=float(below.mean()), frac_above_ceiling=float(above.mean()),
            curve_area_min=a_lo, curve_area_max=a_hi, storage_std=storage_std,
            quality_ok=bool(frac_in >= 0.5 and storage_std > 0 and 0.1 < scale < 10.0),
        ))
        log_rows.append(dict(lake_id=lid, status="ok"))
        if i % 1000 == 0:
            print(f"  ...{i}/{len(lids)}, {len(meta_rows)} shipped", flush=True)

    pd.DataFrame(curve_rows, columns=["lake_id", "h_m", "area_km2", "storage_mcm"]
                 ).to_parquet(OUT_DIR / "aev_curves.parquet", index=False)
    pd.DataFrame(ts_rows, columns=["lake_id", "month", "storage_mcm", "area_km2_input", "in_range"]
                 ).to_parquet(OUT_DIR / "storage_timeseries.parquet", index=False)
    meta = pd.DataFrame(meta_rows); meta.to_parquet(OUT_DIR / "reservoir_metadata.parquet", index=False)
    log = pd.DataFrame(log_rows); log.to_csv(OUT_DIR / "build_log.csv", index=False)

    print("\n=== v4 build complete ===")
    print(f"Target {len(lids)} | shipped {len(meta)} ({len(meta)/len(lids)*100:.1f}%)")
    print(log["status"].value_counts().to_string())
    print("\nArea source:"); print(meta["area_source"].value_counts().to_string())
    print(f"\nMonths retained: {meta.n_months.sum():,}")
    print("\nConfidence tiers:"); print(meta["confidence_tier"].value_counts().to_string())


if __name__ == "__main__":
    main()
