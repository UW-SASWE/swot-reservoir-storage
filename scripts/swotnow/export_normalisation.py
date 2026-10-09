"""Write the dynamic-feature normalisation that the trained model needs at run time.

    python scripts/swotnow/export_normalisation.py [--out FILE]

Rebuilds the statistics over the training pool, as the model was trained, and saves them as
`dyn_norm.npz` (default: next to the model in `swotnow_model`). Needs the training data. Anyone who
downloads the published model gets this file with it and does not need to run this.
"""
import argparse
from pathlib import Path

from swot_reservoir_storage.common import paths
from swot_reservoir_storage.swotnow.inference import export_dyn_norm

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=paths.get("swotnow_model") / "dyn_norm.npz")
    print("wrote", export_dyn_norm(ap.parse_args().out))
