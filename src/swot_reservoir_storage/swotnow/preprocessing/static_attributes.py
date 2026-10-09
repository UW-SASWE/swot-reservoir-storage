"""The 23 static reservoir attributes the network's attribute encoder reads.

Sources, per reservoir:

    GRanD record        size, mean depth, dam height, catchment area, degree of regulation, main use
    reservoir polygon   irregularity index (perimeter / perimeter of an equal-area circle)
    ERA5 precipitation  mean annual total, seasonality, dry-season length
    area-elevation      slope and area fraction at mid elevation, from the SRTM-derived relation
    SWOT                log of the standard deviation of SWOT-era water level

Reservoirs without a GRanD record take capacity, area and coordinates from ``global_universe_v1`` and
have no dam height, depth, catchment, regulation or use; they get the defaults below, and
attributes that cannot be computed stay NaN. A NaN attribute is replaced by the training-pool mean at
run time (``inference.predict``), so it contributes nothing after normalisation.

Defaults carried over from the model's training: dam height falls back to depth, then to 30 m.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ...common import paths

COLUMNS = [
    "lake_id", "GRAND_ID", "lat", "lon",
    "log_cap_mcm", "log_area_skm", "mean_depth_m", "DAM_HGT_M", "log_catch_skm",
    "sin_lat", "cos_lat", "sin_lon", "cos_lon",
    "dor_pc",
    "use_irrigation", "use_hydroelectricity", "use_water_supply",
    "use_flood_control", "use_recreation", "use_other",
    "irr_idx",
    "precip_mean_annual", "precip_seasonality", "precip_dry_months",
    "aec_slope_mid", "aec_area_frac_mid",
    "log_wse_std",
]
TOP_USES = ["Irrigation", "Hydroelectricity", "Water supply", "Flood control", "Recreation"]
DEFAULT_DAM_HEIGHT_M = 30.0


def era5_descriptors(lake_id: str) -> dict:
    f = paths.get("precip") / f"{lake_id}.csv"
    if not f.exists():
        return {}
    p = pd.read_csv(f, parse_dates=["date"]).dropna(subset=["precip_mm"]).copy()
    if len(p) < 60:
        return {}
    p["month"], p["year"] = p["date"].dt.month, p["date"].dt.year
    mean_annual = p.groupby("year")["precip_mm"].sum().mean()
    monthly = p.groupby("month")["precip_mm"].mean()
    seasonality = monthly.std() / (mean_annual / 12 + 1e-6)
    seq = list(monthly < monthly.mean()) * 2          # longest below-mean run, circular
    max_run = run = 0
    for v in seq:
        run = run + 1 if v else 0
        max_run = max(max_run, run)
    return {"precip_mean_annual": mean_annual, "precip_seasonality": seasonality,
            "precip_dry_months": min(max_run, 12)}


def aec_descriptors(lake_id: str) -> dict:
    """Shape of the SRTM-derived area-elevation relation. Storage is the integral of A dh."""
    f = paths.get("srtm_aec") / f"{lake_id}.csv"
    if not f.exists():
        return {}
    aec = pd.read_csv(f, comment="#")
    if "Storage" not in aec.columns:
        return {}
    aec = aec.dropna(subset=["Elevation", "Storage"]).copy()
    if len(aec) < 5:
        return {}
    elev, cum = aec["Elevation"].values, aec["CumArea"].values
    stor = np.concatenate([[0.0], np.cumsum(np.diff(elev) * (cum[1:] + cum[:-1]) / 2.0)]) * 1e6
    e_range, s_max = elev[-1] - elev[0], stor[-1]
    if e_range < 0.01:
        return {}
    norm_e = (elev - elev[0]) / e_range
    mid = np.searchsorted(norm_e, 0.5)
    lo, hi = max(0, mid - 2), min(len(norm_e) - 1, mid + 2)
    area_frac = cum[mid] / (cum[-1] + 1e-9)
    if s_max < 1 or norm_e[hi] - norm_e[lo] < 1e-6:     # volume or spacing too small for a slope
        return {"aec_area_frac_mid": area_frac}
    norm_s = stor / s_max
    return {"aec_slope_mid": (norm_s[hi] - norm_s[lo]) / (norm_e[hi] - norm_e[lo]),
            "aec_area_frac_mid": area_frac}


def build(lake_ids) -> pd.DataFrame:
    """Static attributes for the given reservoirs, in ``COLUMNS`` order."""
    import geopandas as gpd

    ids = [str(l) for l in lake_ids]
    univ = pd.read_csv(paths.get("global_universe_v1"), dtype={"lake_id": str}, low_memory=False,
                       usecols=["lake_id", "GRAND_ID", "lat_dam", "lon_dam", "CAP_MCM", "AREA_SKM", "wse_std"]
                       ).drop_duplicates("lake_id")
    poly_path = paths.get_optional("reservoir_polygons")
    gdf = gpd.GeoDataFrame()
    if poly_path is not None:
        gdf = gpd.read_file(poly_path)
        gdf["lake_id"] = gdf["lake_id"].astype(str)
        gdf["GRAND_ID"] = gdf["GRAND_ID"].astype(int)
        gdf = gdf[gdf["lake_id"].isin(ids)].copy()

    # crosswalk: polygon table where it covers the reservoir, otherwise the universe table
    rest = univ[univ["lake_id"].isin(set(ids) - set(gdf.get("lake_id", [])))]
    rest = rest.rename(columns={"lat_dam": "lat", "lon_dam": "lon"}).drop(columns=["CAP_MCM", "AREA_SKM", "wse_std"])
    base = [gdf[["lake_id", "GRAND_ID", "lat", "lon"]]] if len(gdf) else []
    attrs = pd.concat(base + [rest[["lake_id", "GRAND_ID", "lat", "lon"]]], ignore_index=True)

    grand_path = paths.get_optional("grand_dams")
    if grand_path is not None:
        grand = gpd.read_file(grand_path, columns=["GRAND_ID", "CAP_MCM", "AREA_SKM", "DAM_HGT_M", "DEPTH_M",
                                                   "CATCH_SKM", "LAT_DD", "LONG_DD", "DOR_PC", "MAIN_USE"])
        grand["GRAND_ID"] = grand["GRAND_ID"].astype(int)
        attrs = attrs.merge(grand.drop(columns="geometry"), on="GRAND_ID", how="left")
    else:
        for c in ("CAP_MCM", "AREA_SKM", "DAM_HGT_M", "DEPTH_M", "CATCH_SKM", "LAT_DD", "LONG_DD",
                  "DOR_PC", "MAIN_USE"):
            attrs[c] = np.nan

    size = univ.set_index("lake_id")
    attrs["CAP_MCM"] = attrs["CAP_MCM"].where(attrs["CAP_MCM"] > 0)
    attrs["AREA_SKM"] = attrs["AREA_SKM"].where(attrs["AREA_SKM"] > 0)
    fb_cap, fb_area = attrs["lake_id"].map(size["CAP_MCM"]), attrs["lake_id"].map(size["AREA_SKM"])
    attrs["CAP_MCM"] = attrs["CAP_MCM"].fillna(fb_cap.where(fb_cap > 0))
    attrs["AREA_SKM"] = attrs["AREA_SKM"].fillna(fb_area.where(fb_area > 0))
    attrs["DAM_HGT_M"] = attrs["DAM_HGT_M"].where(
        attrs["DAM_HGT_M"] > 0, attrs["DEPTH_M"].where(attrs["DEPTH_M"] > 0, DEFAULT_DAM_HEIGHT_M))
    attrs["CATCH_SKM"] = attrs["CATCH_SKM"].where(attrs["CATCH_SKM"] > 0)
    attrs["lat"] = attrs["lat"].fillna(attrs["LAT_DD"])
    attrs["lon"] = attrs["lon"].fillna(attrs["LONG_DD"])

    attrs["mean_depth_m"] = (attrs["CAP_MCM"] * 1e6) / (attrs["AREA_SKM"] * 1e6)
    attrs["log_cap_mcm"] = np.log1p(attrs["CAP_MCM"])
    attrs["log_area_skm"] = np.log1p(attrs["AREA_SKM"])
    attrs["log_catch_skm"] = np.log1p(attrs["CATCH_SKM"])
    for name, col in (("lat", "lat"), ("lon", "lon")):
        attrs[f"sin_{name}"] = np.sin(np.radians(attrs[col]))
        attrs[f"cos_{name}"] = np.cos(np.radians(attrs[col]))
    attrs["dor_pc"] = attrs["DOR_PC"].clip(0, 1000)
    for use in TOP_USES:
        attrs[f"use_{use.lower().replace(' ', '_')}"] = (attrs["MAIN_USE"] == use).astype(float)
    attrs["use_other"] = (~attrs["MAIN_USE"].isin(TOP_USES)).astype(float)

    attrs["irr_idx"] = np.nan
    if len(gdf):
        ea = gdf[["lake_id", "geometry"]].to_crs("ESRI:54009")
        ea["irr_idx"] = ea.geometry.length / (2 * np.sqrt(np.pi * ea.geometry.area.clip(1)))
        attrs = attrs.drop(columns="irr_idx").merge(ea[["lake_id", "irr_idx"]], on="lake_id", how="left")

    for fn in (era5_descriptors, aec_descriptors):
        d = pd.DataFrame([{"lake_id": l, **fn(l)} for l in attrs["lake_id"]])
        attrs = attrs.merge(d, on="lake_id", how="left")
    attrs = attrs.merge(size[["wse_std"]].reset_index(), on="lake_id", how="left")
    attrs["log_wse_std"] = np.log1p(attrs["wse_std"])
    return attrs[[c for c in COLUMNS if c in attrs.columns]].reset_index(drop=True)
