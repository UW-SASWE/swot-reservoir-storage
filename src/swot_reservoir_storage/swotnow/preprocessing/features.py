"""The daily feature table SWOTNOW reads, assembled from the raw observations of each reservoir.

One row per reservoir per day from the start of the SWOT science era. Nothing here uses a gauge
record, so the table can be built for any reservoir that SWOT observes.

    SWOT, held between passes   wse, sa_smooth, wse_u, area_tot_u, dark_frac, xtrk_dist
                                on_swot_pass, days_since_swot
    optical, held <= N days     hls_water_area_km2, hls_cloud_frac, hls_sensor, hls_days_since
    calendar                    sin_doy, cos_doy, year
    static                      cap_mcm, cap_mcm_eff, area_skm, lat, lon, has_hls, hls_coverage_frac

``sa_smooth`` here is a two-sided LOWESS of the per-pass area. The strictly causal model does not use
it for the anchor: it reads the one-sided area from ``preprocessing.causal_area`` instead.

Pass selection is: quality flag 0 or 1, one pass per date (best quality first), SWOT era only, with
a reservoir needing at least two passes and a known capacity.

The training table additionally carries gauge-derived target columns. They are not produced here;
``assemble`` accepts ``capacity`` to set the capacity explicitly (a reservoir whose gauge exceeds its
catalogued capacity is given the gauge maximum plus 5%).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess

from ...common import paths
from ...common.config import load as load_config

warnings.filterwarnings("ignore")

_CFG = load_config("swotnow")
SWOT_START = pd.Timestamp(str(_CFG.period.start))
LOWESS_FRAC = float(_CFG.features.lowess_frac)
HLS_MAX_GAP = int(_CFG.features.hls_max_gap_days)

_SWOT_COLS = ["wse", "sa_smooth", "wse_u", "area_tot_u", "dark_frac", "xtrk_dist"]
_static = None


def static_lookup() -> pd.DataFrame:
    """cap_mcm, area_skm, lat, lon by lake_id (str).

    The GRanD crosswalk (``reservoir_lookup``) takes precedence; ``global_universe_v1`` fills the
    reservoirs it does not list. Both are optional individually, but one of them must exist.
    """
    global _static
    if _static is not None:
        return _static
    parts = []
    lk = paths.get_optional("reservoir_lookup")
    if lk is not None:
        a = pd.read_csv(lk).dropna(subset=["lake_id"])
        a["lake_id_str"] = a["lake_id"].astype(int).astype(str)
        parts.append(a[["lake_id_str", "GRAND_ID", "CAP_MCM", "AREA_SKM", "lat", "lon"]])
    u = pd.read_csv(paths.get("global_universe_v1"), dtype={"lake_id": str}, low_memory=False,
                    usecols=["lake_id", "GRAND_ID", "CAP_MCM", "AREA_SKM", "lat_dam", "lon_dam"])
    u = u.dropna(subset=["lake_id"]).drop_duplicates("lake_id")
    u = u.rename(columns={"lake_id": "lake_id_str", "lat_dam": "lat", "lon_dam": "lon"})
    if parts:
        # universe rows are used only to fill reservoirs the crosswalk lacks, and only those with a
        # GRanD id, as before
        u = u.dropna(subset=["GRAND_ID"])
        u = u[~u["lake_id_str"].isin(parts[0]["lake_id_str"])]
    parts.append(u)
    t = pd.concat(parts, ignore_index=True)
    _static = (t.set_index("lake_id_str")[["CAP_MCM", "AREA_SKM", "lat", "lon"]]
               .rename(columns={"CAP_MCM": "cap_mcm", "AREA_SKM": "area_skm"}))
    return _static


def load_swot_passes(lake_id: str):
    """Quality-filtered SWOT passes for one reservoir, indexed by date, or None."""
    fpath, time_col = paths.get("swot_granules") / f"{lake_id}.csv", "time"
    if not fpath.exists():
        cache = paths.get_optional("swot_granules_cache")
        fpath, time_col = (cache / f"{lake_id}.csv" if cache is not None else fpath), "time_str"
    if not fpath.exists():
        return None
    try:
        df = pd.read_csv(fpath, low_memory=False)
    except pd.errors.EmptyDataError:
        return None
    if df.empty or time_col not in df.columns:
        return None
    df["_date"] = pd.to_datetime(df[time_col], utc=True, errors="coerce").dt.tz_localize(None).dt.normalize()
    df["quality_f"] = pd.to_numeric(df["quality_f"], errors="coerce")
    df = df[df["quality_f"].isin([0, 1])].copy()
    for col in ["wse", "area_total", "wse_u", "area_tot_u", "dark_frac", "xtrk_dist"]:
        df[col] = pd.to_numeric(df.get(col, np.nan), errors="coerce")
    df = df[df["wse"] > -1e6].dropna(subset=["_date", "wse", "area_total"])
    df = df[df["_date"] >= SWOT_START]
    if df.empty:
        return None
    df = df.sort_values(["_date", "quality_f"]).groupby("_date").first().reset_index()
    df = df.rename(columns={"_date": "date"})
    if len(df) >= 5:
        t = (df["date"] - df["date"].iloc[0]).dt.days.values.astype(float)
        df["sa_smooth"] = lowess(df["area_total"].values, t, frac=LOWESS_FRAC, it=3, return_sorted=False)
    else:
        df["sa_smooth"] = df["area_total"]
    df["sa_smooth"] = df["sa_smooth"].clip(lower=0)
    return df[["date"] + _SWOT_COLS].set_index("date").sort_index()


def forward_fill_swot(passes: pd.DataFrame, daily_idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Hold each pass until the next; add on_swot_pass and days_since_swot."""
    daily = passes.reindex(daily_idx)
    daily["on_swot_pass"] = (~daily["wse"].isna()).astype(np.int8)
    daily[_SWOT_COLS] = daily[_SWOT_COLS].ffill()
    last_pass = pd.Series(np.nan, index=daily_idx, dtype="float64")
    for d in passes.index[passes.index.isin(daily_idx)]:
        last_pass[d:] = d.value
    days = (pd.Series(daily_idx, index=daily_idx).values.astype("datetime64[ns]").astype(np.int64)
            - last_pass.values.astype(np.int64)) / 86400e9
    daily["days_since_swot"] = np.where(np.isnan(last_pass.values.astype(float)), np.nan, days)
    return daily


def load_hls(lake_id: str):
    """Optical water area for one reservoir, indexed by date, or None."""
    fpath = paths.get("hls_area") / f"{lake_id}.csv"
    if not fpath.exists() or fpath.stat().st_size < 50:
        return None
    df = pd.read_csv(fpath, parse_dates=["time"]).rename(columns={"time": "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.dropna(subset=["water_area_km2"]).sort_values("date")
    df = df[df["date"] >= SWOT_START]
    return df[["date", "water_area_km2", "cloud_frac", "sensor"]].set_index("date")


def forward_fill_hls(hls, daily_idx: pd.DatetimeIndex, max_gap: int = HLS_MAX_GAP) -> pd.DataFrame:
    """Hold optical observations for up to max_gap days, then blank; add hls_days_since."""
    cols = ["water_area_km2", "cloud_frac", "sensor"]
    if hls is None or hls.empty:
        return pd.DataFrame({"hls_water_area_km2": np.nan, "hls_cloud_frac": np.nan,
                             "hls_sensor": np.nan, "hls_days_since": np.nan}, index=daily_idx)
    daily = hls.reindex(daily_idx)
    has_obs = ~daily["water_area_km2"].isna()
    last_obs = pd.Series(np.nan, index=daily_idx)
    for d in daily_idx[has_obs]:
        last_obs[d:] = d.value
    days_since = (pd.Series(daily_idx, index=daily_idx).values.astype("datetime64[ns]").astype(np.int64)
                  - last_obs.values.astype(np.int64)) / 86400e9
    days_since = np.where(np.isnan(last_obs.values.astype(float)), np.nan, days_since)
    daily[cols] = daily[cols].ffill()
    too_old = np.isnan(days_since) | (days_since > max_gap)
    for c in cols:
        daily.loc[too_old, c] = np.nan
    daily["hls_days_since"] = days_since
    daily.loc[too_old, "hls_days_since"] = np.nan
    return daily.rename(columns={"water_area_km2": "hls_water_area_km2", "cloud_frac": "hls_cloud_frac",
                                 "sensor": "hls_sensor"})[
        ["hls_water_area_km2", "hls_cloud_frac", "hls_sensor", "hls_days_since"]]


def assemble(lake_id: str, end, capacity: float | None = None):
    """Daily feature rows for one reservoir from SWOT_START to ``end``, or None.

    None is returned when the reservoir has fewer than two usable passes or no capacity. ``capacity``
    (million m3) overrides the catalogued capacity in ``cap_mcm_eff``; ``cap_mcm`` always keeps the
    catalogue value.
    """
    lake_id = str(lake_id)
    daily_idx = pd.date_range(SWOT_START, pd.Timestamp(end), freq="D")
    passes = load_swot_passes(lake_id)
    if passes is None or len(passes) < 2:
        return None
    swot_daily = forward_fill_swot(passes, daily_idx)
    hls = load_hls(lake_id)
    hls_daily = forward_fill_hls(hls, daily_idx)
    has_hls = hls is not None and not hls.empty
    hls_cov = len(hls[hls.index >= SWOT_START]) / max(len(daily_idx), 1) if has_hls else 0.0

    st = static_lookup()
    static = st.loc[lake_id] if lake_id in st.index else pd.Series(dtype=float)
    doy = daily_idx.day_of_year
    temp = pd.DataFrame({"sin_doy": np.sin(2 * np.pi * doy / 365.25),
                         "cos_doy": np.cos(2 * np.pi * doy / 365.25),
                         "year": daily_idx.year}, index=daily_idx)
    cap = float(static.get("cap_mcm", np.nan))
    cap_eff = float(capacity) if capacity is not None else cap
    if not np.isfinite(cap_eff):
        return None

    df = pd.concat([swot_daily, hls_daily, temp], axis=1)
    df["cap_mcm"] = cap
    df["cap_mcm_eff"] = cap_eff
    df["cap_adjusted"] = int(capacity is not None and (not np.isfinite(cap) or cap_eff > cap))
    df["area_skm"] = static.get("area_skm", np.nan)
    df["lat"] = static.get("lat", np.nan)
    df["lon"] = static.get("lon", np.nan)
    df["has_hls"] = int(has_hls)
    df["hls_coverage_frac"] = hls_cov
    df["hls_sensor"] = df["hls_sensor"].map({"L30": 0, "S30": 1})
    df.index.name = "date"
    df.insert(0, "lake_id", lake_id)
    return df.reset_index()


def build(lake_ids, end, capacity: dict | None = None, progress_every: int = 100) -> pd.DataFrame:
    """Feature rows for many reservoirs. Reservoirs that cannot be assembled are left out."""
    frames, skipped = [], []
    for i, lid in enumerate(lake_ids, 1):
        d = assemble(lid, end, (capacity or {}).get(str(lid)))
        (frames if d is not None else skipped).append(d if d is not None else lid)
        if progress_every and i % progress_every == 0:
            print(f"  {i}/{len(lake_ids)}", flush=True)
    if skipped:
        print(f"not assembled (fewer than two usable passes, or no capacity): {len(skipped)}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
