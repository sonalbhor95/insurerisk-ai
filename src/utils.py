from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd


def save_json(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    raise TypeError(f"Object of type {type(value)} is not JSON serializable")


def normalized_gini(y_true, y_score) -> float:
    """Compute normalized Gini for ranking quality."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if len(np.unique(y_true)) <= 1:
        return 0.0

    order = np.argsort(-y_score)
    y_true = y_true[order]
    cumulative_true = np.cumsum(y_true)
    total_true = cumulative_true[-1]
    if total_true == 0:
        return 0.0
    lorenz = cumulative_true / total_true
    random = np.arange(1, len(y_true) + 1) / len(y_true)
    gini = np.sum(lorenz - random) / len(y_true)

    perfect_order = np.argsort(-y_true)
    perfect_true = y_true[perfect_order]
    perfect_lorenz = np.cumsum(perfect_true) / total_true
    perfect_gini = np.sum(perfect_lorenz - random) / len(y_true)
    return float(gini / perfect_gini) if perfect_gini != 0 else 0.0


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))
