"""Build the SWOTRACE historical storage-anomaly product.

    python scripts/swotrace/build_swotrace.py

Reads the paths listed under SWOTRACE in `config/paths.yaml`, takes its settings from
`config/swotrace.yaml`, and writes the product under `work_dir`. Long: it processes every
reservoir in the target set.
"""
from swot_reservoir_storage.swotrace.reconstruction.build import main

if __name__ == "__main__":
    main()
