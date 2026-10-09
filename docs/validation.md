# Validation

What each component was checked against, and how to repeat the check. "Shipped" means the product
tables released with the paper.

## SWOTNOW

| Check | Result |
|---|---|
| `build_daily` under the committed configuration vs. the reference feature builder | Identical on a 120-reservoir sample, every numeric column, zero tolerance |
| `swotnow.evaluation.gauge_free.run` (four seeds), then `assemble` | `per_reservoir_anomaly.csv` and `per_reservoir_change.csv` match the shipped files (worst difference 0.0); `summary.csv` to 1e-23 |
| `tests/test_no_lookahead.py` | Passes: every pass row has a causal area, and the screen, the area smoother and all nine input channels are unchanged when everything after day *t* is deleted |
| Same test under a two-sided screen (`screen: hampel_interp`) | Fails, so the test can detect look-ahead |

### Preprocessing and inference

| Check | Result |
|---|---|
| `preprocessing.features` rebuilt from raw SWOT and optical records vs. the feature table | Identical (tolerance 1e-9) on a random sample of reservoirs, every non-gauge column (`tests/check_preprocessing.py`) |
| `preprocessing.static_attributes` vs. the static-attribute table | Identical for every reservoir and column (same script) |
| `inference.predict`, built entirely from raw observations, vs. the anchor of the published estimates | Anchor identical to the last digit for the held-out reservoirs |
| `inference.predict` vs. the anomaly R² of the evaluation table, per reservoir | Median identical; every reservoir within 4e-6 (the reservoirs with a missing static attribute need `attr_fill` from `dyn_norm.npz`) |

Per-reservoir constants computed from a reservoir's whole record are declared in
`config/swotnow.yaml` and are held equal by the test; the train/test split is by reservoir.

## SWOTRACE

Against the reference implementation, with zero tolerance: `srtm_curve`, `swot_curve_combined`,
`build_hybrid_curve` and `area_series` agree on every reservoir tested.

## Global uncertainty

The per-reservoir error model and `global_totals.csv` / `continental_breakdown.csv` reproduce the
shipped files bit-identically.

## Pre-SWOT baselines

`preswot_baselines` reproduces the shipped comparison table (worst difference at floating-point
precision).

## Practice for new evaluations

- Compare against the shipped table per reservoir, not by median: scorers can agree on the median while
  disagreeing on the days they include.
- Anomaly R² divides by the reference's variance, so a population minimum is only meaningful after
  filtering on variability.
- Score under the training configuration. The input normalisation must be rebuilt under the same
  settings the model was trained with.

## Not included

The Table S3 pipeline, the screen-audit and screen-design explorations, the figure scripts and the RTS
storage smoother are not in this repository. The published `calibration/` product folder lacks the
`predictions.parquet` that the error model reads beside `metrics.csv`, so `smoother_metrics` must point
at a `metrics.csv` that still has it.
