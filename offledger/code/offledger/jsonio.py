"""Deterministic, self-describing JSON output (sorted keys, floats rounded to 10 decimals)."""
from __future__ import annotations

import json
import math
import platform
from pathlib import Path

import numpy as np

from . import RESULT_SPEC_VERSION, RESULT_SPEC_VERSION_DEFAULT

FLOAT_DECIMALS = 10


def canonical(obj):
    """Recursively convert to JSON-native types; round floats; non-finite floats -> None."""
    if isinstance(obj, dict):
        return {str(k): canonical(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [canonical(v) for v in obj]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        x = float(obj)
        if not math.isfinite(x):
            return None
        x = round(x, FLOAT_DECIMALS)
        return 0.0 if x == 0.0 else x  # no "-0.0"
    if isinstance(obj, np.ndarray):
        return canonical(obj.tolist())
    if obj is None or isinstance(obj, str):
        return obj
    raise TypeError(f"cannot serialise {type(obj)!r}")


def dumps(obj) -> str:
    return json.dumps(canonical(obj), sort_keys=True, indent=1, ensure_ascii=False,
                      allow_nan=False) + "\n"


def header(experiment: str, parameters: dict, notes: list, seeds=None) -> dict:
    return {
        "spec_version": RESULT_SPEC_VERSION.get(experiment, RESULT_SPEC_VERSION_DEFAULT),
        "experiment": experiment,
        "parameters": parameters,
        "seeds": seeds,
        "notes": list(notes),
        "software": {"python": platform.python_version(), "numpy": np.__version__},
    }


def write(path, obj) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps(obj), encoding="utf-8")
    return path


def read(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
