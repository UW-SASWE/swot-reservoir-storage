# Data

Nothing in this directory is committed. None of the inputs is redistributable: the SWOT
granules, the in-situ gauge records and the published comparison products each have their own
access terms, and several come from national agencies that license them individually.

What follows is where each dataset comes from. Point `config/paths.yaml` at wherever you put
them, then run `python scripts/check_paths.py`.

**You do not need all of it.** The three analyses have different inputs. Each script reports the
paths it needs when one is missing, so start with the analysis you care about and add data until
it runs.

Identifiers, papers, licences and how each was confirmed are in `sources.yaml` (in this directory); entries marked
`checked: "no"` are still to be confirmed.

The products this repository generates, with the trained model, are archived at
<https://doi.org/10.5281/zenodo.23248841>.

## Observations

| Key in `paths.yaml` | Dataset | Where from |
|---|---|---|
| `swot_granules` | SWOT Level-2 Lake Single-Pass (L2_HR_LakeSP): water-surface elevation and area per pass | NASA PO.DAAC, via the Hydrocron API. Needs an Earthdata login in `~/.netrc`. |
| `hls_area` | Harmonized Landsat and Sentinel-2 optical water area | Derived via Google Earth Engine; needs `earthengine authenticate` |
| `tmsos_area` | TMS-OS fused optical + Sentinel-1 surface area | Produced by the RAT 3.0 package (Das et al., 2022; Minocha et al., 2023) |
| `gsw_monthly` | JRC Global Surface Water monthly history, 1984– | Pekel et al. (2016), via Google Earth Engine |
| `sar_area` | Sentinel-1 surface area | Google Earth Engine. Optional: tested and not adopted. |
| `precip` | ERA5-Land precipitation, routed per reservoir | Google Earth Engine. Optional. |

## Reference and auxiliary

| Key | Dataset | Where from |
|---|---|---|
| `pld` | SWOT Prior Lake Database | NASA/CNES. Supplies reservoir coordinates and the `lake_id` join key. |
| `global_universe` | Reservoir universe with SWOT coverage tiers | Built by this repository from Global Dam Watch + PLD |
| `srtm_aec` | Area-elevation curves from SRTM | Produced by the RAT package from SRTM |
| `insitu` | Gauge records, unified schema | See below |

### In-situ gauge records

Held by national agencies and **obtained individually**: USACE, USBR, USGS NWIS, USBR RISE,
CDEC and TWDB (United States); CEDEX and MITECO (Spain); CONAGUA (Mexico); CWC (India); the
Bureau of Meteorology (Australia); ANA (Brazil); the Environment Agency (United Kingdom); and
the European Environment Agency.

Two things to know before relying on them. They are the **reference only** — no gauge is ever an
input to an estimate in this work. And their geography is the binding limitation of the study:
six countries, a majority of the calibration panel in the United States, and no records at all
for any African country, China, Russia, or continental Europe outside Iberia.

## Published products, for comparison

Optional. Without them the historical evaluation runs against gauges alone.

| Key | Product | Reference |
|---|---|---|
| `grs` | Global Reservoir Storage | Li et al. (2023) |
| `glolakes` | GloLakes | Hou et al. (2024) |
| `lakes3d` | 3D-LAKES | Huang et al. (2025) |
| `grdl` | GRDL deep-learning bathymetry | — |

## External software

| Key | What | Note |
|---|---|---|
| `rat_src` | RAT 3.0 source tree | Only needed to rebuild the TMS-OS area record yourself |

## Storage

Budget a few tens of GB for a full rebuild of the historical product. The SWOT granule cache and
the monthly surface-water history dominate.
