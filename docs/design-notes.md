# Design notes

Why the less obvious settings are what they are. The config files and source say *what* each setting
does; this file gives the reasoning. Evaluation results (counts, scores, test statistics) are reported
in the paper, not here.

## Configuration lives in files

The published settings are the committed defaults in `config/*.yaml`. A model built under different
settings is a different model, so every run writes `run_config.json` beside its output, and a missing
key raises instead of defaulting.

Inputs are located through `config/paths.yaml`, never from the source file's location, so moving a
file cannot repoint them.

## SWOTNOW

### The anchor and the correction

The network predicts a bounded residual (final tanh, so within [-1, 1]) on top of the physical anchor
`S = S_ref + (h - h_ref) * A`, held between passes. The estimate therefore cannot wander far from the
physics. The attribute encoder (23 static traits to a context vector) replaces a per-reservoir
embedding, which cannot generalise beyond the training set; most reservoirs are ungauged.

The residual gate scales the residual per reservoir from the static attributes. The residual helps
mainly on high-variability reservoirs; on flat reservoirs it mostly adds noise. A gate driven by a
static attribute present on every timestep avoids the failure of a per-timestep "SWOT-gated" blend,
which collapses to zero through gradient starvation because SWOT-pass dates are a small share of
timesteps.

### Outlier screen on the pass series

A bad pass is held for the whole interval to the next one, so a screen matters more than the pass
count suggests.

The two-sided screens (`hampel`, `hampel_interp`) compare each pass with k neighbours on *both* sides
and interpolate across rejects, so the anchor on day x depends on later passes. They are selectable
for ablation only. `hampel` replaces a reject with the local median, which also flattens genuine
extremes and damps amplitude; `hampel_interp` interpolates and keeps amplitude.

The causal screens (`causal_screens.py`) decide pass i from passes <= i only, judge it against
*accepted* history, repair by carrying the last accepted value forward, and accept a pass after two
consecutive rejections (persistent disagreement is a real change in the reservoir). The published
screen is the increment gate `causal_C2b_incr_n4`. It was selected on the training reservoirs only, by
a rule fixed before the held-out set was scored: highest median anomaly R2, provided it beats no
screen by more than 0.02 with a Bonferroni-corrected Wilcoxon p < 0.05 and keeps amplitude within
0.90-1.10 (`scripts/swotnow/design_causal_screen.py`). A screen judged only on accepted history can
reject a genuine change; the lock-in rule bounds how long that persists.

### Gauge-free, anomaly target

The product is a storage-*change* estimate with no gauge input. Training therefore must not see a
gauge-derived reference level that deployment does not supply. Four settings make training match
deployment, and they belong together:

- `inputs.use_gauge: false`: the nominal reference level for every reservoir.
- `target.centre_per_reservoir: true`: target and anchor centred on their own means, so the loss is on
  the anomaly and carries no level term. (Alone, a near-zero anchor against an absolute target leaves
  the residual to span the level.)
- `target.clip_fill_fraction: false`: a clip to [0, 1] presumes a datum and would delete the half of a
  centred series that sits below its mean.
- `training.select_on`: `mse` is the published choice; `kge` penalises amplitude != 1 as a guard against
  the shrinkage an anomaly MSE target can reward.

`predictions_daily.parquet` carries both clipped (`pred_mcm`, `anchor_mcm`) and unclipped (`*_raw`)
columns, written unconditionally. Under the published configuration the raw columns are the product;
the clipped ones are wrong in magnitude for centred series.

### Causal area

`sa_smooth` in the feature table is a two-sided LOWESS, inadmissible for a nowcast. The causal area is
a trailing tricube-weighted *mean* (a local-constant fit; a trailing local-linear fit has roughly
twice the variance because it extrapolates at its own boundary). Its builder covers exactly the feature
table's pass rows, using the feature builder's selection rule verbatim (including the `dark_frac`
filter and de-duplication by date), and the strict setting refuses to run on any gap so that no pass
can keep the two-sided area.

### Declared per-reservoir constants

Computed from a reservoir's whole record and not made causal: the reference elevation (median of the
record), the smoother width `round(0.3 * passes)`, the reference area, the static attributes
including `log_wse_std`, the optical/radar availability flags, the centring offsets and the
normalisation statistics. The train/test split is by reservoir, so other reservoirs' later dates are in
the training set. `tests/test_no_lookahead.py` checks the claim for every time-varying input and holds
these constants equal by design.

### Inputs

- Nine dynamic channels; three are constant over a record (`sar_water_frac`, `has_sar`, `has_hls`) and
  are retained so tensor shapes match the published model. Radar was tested as a live input and did not
  change accuracy. Routed precipitation exists for only some reservoirs and is zero for the rest.
- `quality_features` (off): SWOT's per-pass uncertainty `wse_u` and `dark_frac` as network inputs did not
  help; the uncertainty is more useful as a screen than as a channel.
- `static_attributes` must be the table whose `aec_slope_mid` uses the volume integral of A dh, not RAT's
  `Storage` column, which integrates h dA.

### Thresholds that decide the evaluation population

`min_days_scored` (the differenced change series) and `min_days_anomaly` (the anomaly metrics) are
different thresholds and both move every median. `min_test_observations` is the trainer's.

## Global uncertainty

- `fit_denominator: window` divides by the storage standard deviation *within each reservoir's own
  held-out window*, so numerator and denominator describe the same period. The `full` option divides by
  the full-record SD; when the held-out window covers only part of the record's variability that
  deflates the coefficient unevenly across regions.
- `min_window_coverage`: below a minimum share of the record's variability, the two denominators
  disagree; above it they converge.
- `enforce_capacity_bound`: storage lies in [0, capacity], so its SD cannot exceed half the capacity.
- Both correlation bounds are reported because the spatial correlation of systematic error is
  unknown: `independent` (errors cancel in quadrature across reservoirs, the optimistic bound) and
  `correlated` (fully correlated within a region, independent across regions, the pessimistic one). The
  percentage reduction is identical under both because it is the ratio of two fitted coefficients, so
  the agreement is algebra, not corroboration.
- The error model fits on `global_universe_v1` and applies to the v2 universe, deliberately:
  `sigma_S = wse_std * AREA_SKM` is the model's central quantity and `wse_std` differs between the two
  tables for a large share of shared `lake_id`s. Both tables repeat `lake_id`, so deduplicate before
  joining.
- `smoother_metrics` must be a `metrics.csv` that still sits beside its `predictions.parquet`: the error
  model reads that sibling for within-window variability.
- Both arms are fitted on reservoirs where both exist (paired fit). Fitting each arm on its own
  population divides coefficients estimated on different, systematically different reservoirs.
- Every reservoir needs a bed term. The aggregator uses `nansum`, so a missing one counts as zero
  uncertainty. `S_ref` comes from the production smoother file, with the storage-anchor table as a
  fallback.
- The bed term is `S_ref = ANCHOR_FILL * capacity`, where `ANCHOR_FILL` is the gauge-measured
  sum-weighted fill fraction (sum of true storage over sum of capacity) and `ANCHOR_REL` the robust
  dispersion (1.4826 x MAD) of that anchor's relative error. A capacity fraction cannot exceed capacity
  by construction, whereas an area-elevation-curve anchor sits on its clamp floor for many reservoirs,
  and debiasing it with a median fitted on gauged (large) reservoirs does not transfer to the whole
  population.
- The bed term is SWOT-independent: it cancels in the dynamic dU but inflates both arms, so it lowers
  only the absolute arm's percentage reduction.
- `--legacy` reproduces the pre-correction behaviour only so old runs can be re-created; no number in
  the paper comes from it.

## SWOTRACE

SWOT never observed the period reconstructed. It contributes a constraint on part of the
area-elevation relationship; all temporal variation comes from the historical surface-area record. The
product is a SWOT-informed reconstruction, not a SWOT observation of the past, and that governs how far
it can be pushed.

- `n_grid`: number of elevation points across the combined relation.
- `min_band_m`: a reservoir whose water surface moves less than this cannot support a shape metric.
- Never gate on `valid_frac`. It measures polygon oversize, not data quality, so gating removes
  reservoirs with loose polygons and biases an evaluation toward large reservoirs with tight ones. The
  pipeline divides the wet-pixel count by the observable fraction, above a small floor.
- Cloud-corrected area uses recurrence gap-filling of cloudy pixels (Zhao and Gao, 2018). Scaling by
  the observable fraction alone is cheaper but harmful at storage level, so the recurrence method is
  used.
- The confidence tiers are thresholds on the SD of SWOT-era water level: how much a reservoir moves
  determines whether SWOT constrains a useful span of its relationship.
- `swot_datum_shift` returns `d = median(h_srtm(a) - h_swot(a))`: SRTM reads higher than SWOT by `d`, so
  the SRTM elevation matching a SWOT-frame elevation `g` is `g + d`. Datum alignment is done in the area
  domain; matching curve shapes over a narrow, near-linear SWOT band is underdetermined and can latch
  onto noise.
- The build is a single module so that there is exactly one `build_hybrid_curve`.

## Data paths

- `tmsos_area_primary` and `tmsos_area` are read in that order, taking the first with a record for a
  reservoir. Files are `<lake_id>.csv` with a `date` and an `area` column in m2 or km2 (the loader
  detects which).
- `swot_hydrocron_combined` is a different, larger cache than `swot_granules`; check which one an
  analysis wants.
- The dam coordinates in some inventories are 0.0 for most rows; use the SWOT Prior Lake Database
  coordinates (`pld`).
- `runs_dir` holds trained runs (`swot_gru_<tag>`); `swotnow_model` defaults to the published model.

## Running the trained model

- The dynamic-feature normalisation is not saved at training time; every script that runs the model
  rebuilds it over the training pool ('train' split plus every reservoir with attributes but no split,
  whose target is the anchor itself), through the same feature builder and with the same imputation.
  Held-out reservoirs must not contribute. A mismatch raises no error.
- A single NaN attribute makes the network return NaN for every day of that reservoir, so every column in
  `ATTR_COLS` is mean-imputed, as the trainer does.
- Scoring must use the training configuration. Rebuilding the normalisation with the gauge-free switch
  off mis-scales the anchor channel. The scorer's "gauged" arm is a diagnostic only, because the models
  are not trained on a gauge-derived offset.
- `gauge_free.rebuild_anchor` reuses the screened `anchor_ff` that `build_daily` already produced, so a
  causal-screen model is scored on the anchor it was trained on.

## Storage from the SRTM-derived relation

The `Storage` column in the RAT area-elevation files is the integral of h dA, not of A dh. The repository
recomputes storage as the integral of A dh by the trapezoid rule.
