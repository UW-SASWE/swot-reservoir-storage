# SWOT's net impact on reservoir storage monitoring

Code for three analyses of what the Surface Water and Ocean Topography mission adds to reservoir
storage estimation, accompanying the paper *Quantifying the net impact of SWOT on reservoir
monitoring*.

The three are not interchangeable, and they are not equally direct evidence:

| | What it does | How SWOT is used |
|---|---|---|
| **SWOTNOW** | Estimates present-day storage daily, using no in-situ record, and from each reservoir's own record only the observations up to the day being estimated (plus a few declared per-reservoir constants; see `config/swotnow.yaml`) | Directly: each SWOT water level converts to a storage value |
| **SWOTRACE** | Reconstructs monthly storage anomalies for 9,401 reservoirs, 1984–2021 | Indirectly: SWOT never observed the period reconstructed. It constrains part of an area–elevation relationship, and the historical surface-area record supplies all the temporal variation |
| **Global uncertainty** | Propagates per-reservoir error to an uncertainty on globally summed storage anomaly | At one further remove: an error relationship calibrated on the present-day analysis, applied to reservoirs that were never scored |

Both products report a storage **anomaly** — a reservoir's departure from its own mean — not an
absolute volume. Without a gauge nothing ties an estimate to a datum, and no surface-measuring
instrument constrains the elevation of a reservoir bed.

## Status

> SWOTNOW is the strictly causal model of the paper (increment-gate outlier screen, one-sided area,
> four seeds); `tests/test_no_lookahead.py` checks the causality claim by deleting everything after
> day *t* and rebuilding. What was checked against the shipped results, and what is not included
> (the Table S3 pipeline, the screen-audit explorations, the figure scripts), is in
> `docs/validation.md`.

## Data and trained model

The trained SWOTNOW model, the SWOTRACE and global-uncertainty products and the evaluation tables are
archived at <https://doi.org/10.5281/zenodo.23248841> (CC BY 4.0). Inputs from other providers are
listed in `docs/sources.yaml`.

## AI assistance

Parts of this code and its documentation were written, refactored and checked with Claude Sonnet 5.5 and Claude Opus 5.5
(Anthropic) through Claude Code. The author specified the analyses, ran and reviewed the results, and is
responsible for the code and for every claim made with it.

## Getting started

```bash
conda env create -f environment.yml
conda activate swot-reservoir-storage
pip install -e .

cp config/paths.example.yaml config/paths.yaml
$EDITOR config/paths.yaml          # point each entry at your copy of the data
python scripts/check_paths.py      # says what resolves and what does not
```

`check_paths.py` prints every configured input and whether it is present, so you see the whole
picture at once rather than discovering one missing file at a time.

**You do not need all the data.** The three analyses have different inputs; start with the one
you care about and add data until it runs. `docs/data-sources.md` says where each dataset comes from.
None of it is redistributable, so none of it is committed.

## Running the analyses

Each analysis is a short sequence of scripts. Settings come from `config/<analysis>.yaml`, inputs from
`config/paths.yaml`, and every run writes `run_config.json` beside its output.

**SWOTNOW**
```bash
python scripts/swotnow/build_causal_area.py          # one-sided SWOT area table
python scripts/swotnow/train.py --seed 42            # repeat for each seed in training.seeds
python scripts/swotnow/score.py                      # gauge-free scoring of every seed
python scripts/swotnow/assemble_tables.py            # per-reservoir tables and summary
python scripts/swotnow/preswot_baselines.py          # the two pre-SWOT comparison relations
python tests/test_no_lookahead.py                    # causality check
python tests/check_preprocessing.py                  # feature builder reproduces the feature table
```

**Estimate any reservoir SWOT observes** with the trained model (download it from the Zenodo record above;
it needs `model.pt`, `attr_norm.npz` and `dyn_norm.npz`):
```bash
python scripts/swotnow/predict.py --lake-ids 7830181213 --out estimates.csv
```
The feature table, the one-sided area and the static attributes are built from raw observations by
`swotnow.preprocessing.features`, `causal_area` and `static_attributes`; no gauge is read. The output
is a storage anomaly in million m3 on an arbitrary offset: subtract each reservoir's own mean.
A reservoir without a GRanD record has no dam height, depth, catchment, regulation or use, and those
attributes take the training-pool mean. `scripts/swotnow/export_normalisation.py` writes `dyn_norm.npz`
from the training data; it is only needed to rebuild that file.

**SWOTRACE**
```bash
python scripts/swotrace/build_swotrace.py
```

**Global uncertainty**
```bash
python scripts/global_uncertainty/fit_error_model.py
python scripts/global_uncertainty/aggregate_global_totals.py
```

## How the repository is arranged

```
config/          paths.yaml (yours, ignored by git) and one YAML per analysis
src/             the package: common/, swotnow/, swotrace/, global_uncertainty/
scripts/         command-line entry points, one directory per analysis
```

Why the less obvious settings are what they are is in `docs/design-notes.md`.

`config/` holds one YAML per analysis (`swotnow.yaml`, `swotrace.yaml`, `global_uncertainty.yaml`),
`scripts/` the command-line entry points (one directory per analysis; `design_causal_screen.py`
reproduces the choice of outlier screen), and `tests/` the no-look-ahead test. `docs/validation.md` lists what each
component was checked against.

Two rules the code holds to:

**No absolute paths outside `config/`.** Every input resolves through `common.paths`, which
reads your `paths.yaml`. A missing input raises at the point it is requested, naming the key,
the file that should define it and what the data is.

**No behaviour steered by environment variables.** Settings live in `config/*.yaml`, the committed
defaults are the published configuration, and every run writes `run_config.json` beside its
output, so a result can be traced to the settings that produced it.

## Three things worth knowing before reading the results

**Two measures, two answers.** On the storage *anomaly* the trained SWOTNOW sits above its own anchor
(median anomaly R² 0.531 against 0.481 on the 113-reservoir comparison set, higher in all four seeds).
On the estimated *change* the correction improves on the anchor from 3 days upward, but at one day
it is worse than an estimate that never changes. Reporting either alone gives the wrong impression,
which is why both are reported. A causal outlier screen recovers little: against no screen at all its
paired effect on the anomaly is zero.

**Anomaly R² is a ratio, not an error.** It divides by the reference's own variance, so a
reservoir that barely moves produces a large negative value for any estimate of ordinary
magnitude. Those are not failures. Filter on variability before taking a population minimum.

**The gauge records are geographically uneven**, and this limits all three analyses. Six
countries, a majority of the calibration panel in the United States, and no records for any
African country, China, Russia, or continental Europe outside Iberia. This is the binding
limitation of the work.

## Citing

See `CITATION.cff`. The accompanying paper is not yet published; the entry will be completed on
acceptance.

## Licence

MIT, for the code. The datasets it reads carry their own terms — see `docs/data-sources.md`.
