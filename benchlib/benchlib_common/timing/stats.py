from __future__ import annotations

from typing import Dict, List
import numpy as np


def summarize_ms(samples: List[float]) -> Dict[str, float]:
    if not samples:
        return {"mean_ms": 0.0, "median_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0, "p95_ms": 0.0}
    arr = np.asarray(samples, dtype=np.float64)
    return {
        "mean_ms": float(arr.mean()),
        "median_ms": float(np.median(arr)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "p95_ms": float(np.percentile(arr, 95)),
    }
