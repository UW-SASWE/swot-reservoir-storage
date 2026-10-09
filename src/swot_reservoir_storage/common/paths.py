"""Where the data lives.

Every input this project reads sits outside the repository: SWOT granules, optical and radar
surface-area records, in-situ gauge series, digital elevation models, published comparison
products. None of it can be committed, and none of it is in the same place on two machines. This
module is the single place that knows where any of it is, so that no other file in the package
contains an absolute path.

How it works
------------
One YAML file, ``config/paths.yaml``, maps a short logical name to a directory or file on this
machine. ``config/paths.example.yaml`` is committed as a template; ``paths.yaml`` itself is
ignored by git, because it is specific to whoever is running the code.

    from swot_reservoir_storage.common import paths
    feat = paths.get("features")            # -> Path, checked to exist
    swot = paths.get("swot_granules")

Resolution order, first hit wins:

1. the environment variable ``SRS_<NAME>`` (upper case), for a one-off override;
2. the entry in ``config/paths.yaml``;
3. the entry in ``config/paths.example.yaml``, which holds only placeholders.

Why it raises instead of guessing
---------------------------------
A missing input is reported at the moment it is asked for, naming the key, the file that should
define it and what the data is. The alternative -- returning a path that does not exist and
letting pandas fail ten frames later on a file it cannot open -- is how an afternoon gets lost.
``get()`` therefore checks existence by default. Pass ``must_exist=False`` for an output
directory that is about to be created.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

__all__ = ["repo_root", "config_dir", "get", "get_optional", "describe", "PathNotConfigured"]

# The package sits at <repo>/src/swot_reservoir_storage/common/paths.py, so the repository root
# is four parents up. This is the only place a relative location is assumed, and it is a fact
# about the package layout rather than about anyone's machine.
_REPO_ROOT = Path(__file__).resolve().parents[3]

_USER_FILE = "paths.yaml"
_EXAMPLE_FILE = "paths.example.yaml"

_cache: dict[str, Any] | None = None


class PathNotConfigured(KeyError):
    """A logical path was requested that no configuration file defines."""


def repo_root() -> Path:
    """The repository root, derived from this file's own location."""
    return _REPO_ROOT


def config_dir() -> Path:
    return _REPO_ROOT / "config"


def _load() -> dict[str, Any]:
    global _cache
    if _cache is not None:
        return _cache

    merged: dict[str, Any] = {}
    example = config_dir() / _EXAMPLE_FILE
    user = config_dir() / _USER_FILE

    for f in (example, user):          # user entries overwrite example placeholders
        if not f.exists():
            continue
        with open(f) as fh:
            loaded = yaml.safe_load(fh) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"{f} must contain a mapping of name -> path, got {type(loaded).__name__}")
        merged.update({k: v for k, v in loaded.items() if v is not None})

    if not merged:
        raise PathNotConfigured(
            f"No path configuration found. Copy {example} to {user} and fill it in."
        )
    _cache = merged
    return merged


def _describe_key(name: str) -> str:
    """The comment the example file carries for this key, used in error messages."""
    example = config_dir() / _EXAMPLE_FILE
    if not example.exists():
        return ""
    note, want = [], False
    for line in example.read_text().splitlines():
        s = line.strip()
        if s.startswith("#") and not want:
            text = s.lstrip("# ").rstrip()
            # skip the rules of dashes that separate sections of the example file
            if text and set(text) != {"-"}:
                note.append(text)
        elif s.startswith(f"{name}:"):
            want = True
            break
        elif s and not s.startswith("#"):
            note = []
    return " ".join(note[-3:]) if want and note else ""


def get(name: str, *, must_exist: bool = True) -> Path:
    """Resolve a logical name to a path on this machine.

    Parameters
    ----------
    name
        A key from ``config/paths.example.yaml``, e.g. ``"features"``.
    must_exist
        Raise if the resolved path is not present. Set False for a directory this
        call is about to create.
    """
    env = os.environ.get(f"SRS_{name.upper()}")
    if env:
        p = Path(env).expanduser()
    else:
        cfg = _load()
        if name not in cfg:
            known = ", ".join(sorted(cfg))
            raise PathNotConfigured(
                f"'{name}' is not defined in {config_dir() / _USER_FILE}.\n"
                f"  Known names: {known}\n"
                f"  Add it there, or set the environment variable SRS_{name.upper()}."
            )
        p = Path(str(cfg[name])).expanduser()

    if must_exist and not p.exists():
        hint = _describe_key(name)
        raise FileNotFoundError(
            f"The path configured for '{name}' does not exist:\n"
            f"    {p}\n"
            f"  Defined in: {config_dir() / _USER_FILE}\n"
            + (f"  This should point to: {hint}\n" if hint else "")
            + "  Fix the path there, or set SRS_"
            + name.upper()
            + " to override it for this run."
        )
    return p


def get_optional(name: str) -> Path | None:
    """As :func:`get`, but returns None instead of raising when unset or absent.

    For inputs the analysis can proceed without -- routed precipitation, for instance, which
    is available for about a fifth of reservoirs.
    """
    try:
        return get(name)
    except (PathNotConfigured, FileNotFoundError):
        return None


def describe() -> str:
    """A report of every configured path and whether it is present.

    Printed by ``scripts/check_paths.py`` so a new user can see in one go what is wired up
    and what is missing, instead of discovering it one traceback at a time.
    """
    try:
        cfg = _load()
    except PathNotConfigured as e:
        return str(e)

    rows, missing = [], 0
    for k in sorted(cfg):
        try:
            p = get(k)
            rows.append(f"  OK       {k:<26} {p}")
        except FileNotFoundError:
            missing += 1
            rows.append(f"  MISSING  {k:<26} {cfg[k]}")
        except PathNotConfigured:
            missing += 1
            rows.append(f"  UNSET    {k:<26} -")
    head = f"{len(cfg)} paths configured, {missing} not resolvable"
    return "\n".join([head, ""] + rows)
