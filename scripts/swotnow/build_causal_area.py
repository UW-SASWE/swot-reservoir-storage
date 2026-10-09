"""Build the causal (one-sided) SWOT surface-area table that SWOTNOW reads.

    python scripts/swotnow/build_causal_area.py

Needs the paths `features`, `swot_granules`, `swot_granules_cache` and writes `causal_surface_area`.
The run stops, and writes nothing, unless it covers 100.00% of the feature table's SWOT pass rows.
"""
from swot_reservoir_storage.swotnow.preprocessing.causal_area import main

if __name__ == "__main__":
    main()
