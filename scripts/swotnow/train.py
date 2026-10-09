"""Train one SWOTNOW seed.

    python scripts/swotnow/train.py --seed 42

Writes `<work_dir>/runs/swot_gru_<run_tag>/`, with the run tag taken from `evaluation.run_tag` in
`config/swotnow.yaml` (default `strict_c2b_s{seed}`). Repeat for each seed in `training.seeds`.
Other options (`--hidden`, `--w-pseudo`, `--dry-run`) are those of the trainer.
"""
import sys

from swot_reservoir_storage.common.config import load
from swot_reservoir_storage.swotnow.model.trainer import main

if __name__ == "__main__":
    cfg = load("swotnow")
    seed = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else int(cfg.training.seeds[0])
    if "--tag" not in sys.argv:
        sys.argv += ["--tag", cfg.evaluation.run_tag.format(seed=seed)]
    if "--seed" not in sys.argv:
        sys.argv += ["--seed", str(seed)]
    main()
