"""Run configuration.

Settings live in ``config/*.yaml``. The defaults committed there are the ones that produced the
published results, and every run records the configuration it used beside its output, so a result
can be traced to its settings. Nothing is read from environment variables.

    from swot_reservoir_storage.common.config import load
    cfg = load("swotnow")
    cfg.anchor.screen          # 'causal_C2b_incr_n4'
    cfg.training.seeds         # [42, 43, 44, 45]

A key that is not in the file raises rather than defaulting to None, so that an absent setting
cannot silently change the configuration you report.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from . import paths

__all__ = ["Config", "load", "record"]


class Config:
    """A nested mapping reachable by attribute, so ``cfg.anchor.screen`` works.

    Wraps a plain dict rather than replacing it: ``cfg.to_dict()`` returns exactly what was
    read from the file, which is what gets written into the run record.
    """

    def __init__(self, data: dict[str, Any], _trail: str = ""):
        self._data = data
        self._trail = _trail

    def __getattr__(self, key: str) -> Any:
        if key.startswith("_"):
            raise AttributeError(key)
        if key not in self._data:
            where = f"{self._trail}.{key}" if self._trail else key
            known = ", ".join(sorted(k for k in self._data if not k.startswith("_")))
            raise KeyError(
                f"'{where}' is not set in this configuration.\n"
                f"  Available at this level: {known}\n"
                f"  Add it to the YAML file rather than relying on a default."
            )
        v = self._data[key]
        trail = f"{self._trail}.{key}" if self._trail else key
        return Config(v, trail) if isinstance(v, dict) else v

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def get(self, key: str, default: Any = None) -> Any:
        """Explicit opt-in to a default, for genuinely optional settings."""
        v = self._data.get(key, default)
        return Config(v, f"{self._trail}.{key}") if isinstance(v, dict) else v

    def to_dict(self) -> dict[str, Any]:
        # default=str because YAML parses a bare date like 2023-08-01 into a datetime.date,
        # which json cannot encode. Without it, every script that records its configuration
        # dies at the point of writing the record -- after the work is done, which is the worst
        # moment to fail. Dates become ISO strings, which is what a run record wants anyway.
        return json.loads(json.dumps(self._data, default=str))

    def __repr__(self) -> str:
        return f"Config({self._trail or 'root'}: {', '.join(sorted(self._data))})"


def load(name: str) -> Config:
    """Load ``config/<name>.yaml``.

    A local ``config/<name>.local.yaml``, if present, is merged over it one key at a time, so
    you can change one setting without copying the whole file. Local overrides are ignored by
    git; the committed file always holds the published configuration.
    """
    base = paths.config_dir() / f"{name}.yaml"
    if not base.exists():
        available = ", ".join(sorted(p.stem for p in paths.config_dir().glob("*.yaml")))
        raise FileNotFoundError(f"No configuration named '{name}'. Available: {available}")

    with open(base) as fh:
        data = yaml.safe_load(fh) or {}

    local = paths.config_dir() / f"{name}.local.yaml"
    if local.exists():
        with open(local) as fh:
            data = _deep_merge(data, yaml.safe_load(fh) or {})

    return Config(data)


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def record(out_dir: Path, cfg: Config, extra: dict[str, Any] | None = None) -> Path:
    """Write the configuration that produced a run, beside its output.

    Called by every script that writes results, so that a directory of outputs can be traced back
    to the settings that made it.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": cfg.to_dict(),
    }
    if extra:
        payload.update(extra)
    f = out_dir / "run_config.json"
    f.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return f
