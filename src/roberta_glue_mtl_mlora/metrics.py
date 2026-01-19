from __future__ import annotations

import math
from typing import Any


def as_float(x: Any, ndigits: int = 6) -> float:
    """Round safely for JSON output."""
    try:
        return float(round(float(x), ndigits))
    except Exception:
        return float(x)


def matthews_corrcoef_from_counts(tp: int, tn: int, fp: int, fn: int) -> float:
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denom == 0:
        return 0.0
    return ((tp * tn) - (fp * fn)) / denom


def f1_from_counts(tp: int, fp: int, fn: int) -> float:
    denom = (2 * tp + fp + fn)
    if denom == 0:
        return 0.0
    return (2.0 * tp) / denom


def pearson_from_sums(
    n: float,
    sum_x: float,
    sum_y: float,
    sum_x2: float,
    sum_y2: float,
    sum_xy: float,
) -> float:
    """Pearson correlation computed from running sums (numerically stable enough here)."""
    num = n * sum_xy - sum_x * sum_y
    den_x = n * sum_x2 - sum_x * sum_x
    den_y = n * sum_y2 - sum_y * sum_y
    denom = math.sqrt(max(den_x, 0.0) * max(den_y, 0.0))
    if denom == 0.0:
        return 0.0
    return num / denom
