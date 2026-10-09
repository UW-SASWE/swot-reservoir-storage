"""SWOT's net impact on reservoir storage monitoring.

Three analyses, each with its own subpackage:

    swotnow             present-day storage estimation from SWOT elevation
    swotrace            historical storage reconstruction, 1984-2021
    global_uncertainty  propagation of per-reservoir error to a global total

`common` holds what they share: path resolution, run configuration, and the metrics.

Nothing here contains an absolute path. Every input is resolved through
`common.paths`, which reads `config/paths.yaml`. Run `scripts/check_paths.py`
first to see what is wired up.
"""
__version__ = "0.1.0"
