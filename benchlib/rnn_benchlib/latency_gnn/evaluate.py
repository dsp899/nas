from __future__ import annotations

from typing import Dict, List, Mapping, Sequence

import numpy as np
import tensorflow as tf



def compute_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, float]:
    yt = np.asarray(list(y_true), dtype=np.float64)
    yp = np.asarray(list(y_pred), dtype=np.float64)
    if yt.size == 0:
        return {"count": 0.0, "mape": 0.0, "mae": 0.0, "rmse_log": 0.0, "spearman": 0.0}
    return {
        "count": float(yt.size),
        "mape": float(np.mean(np.abs(yp - yt) / np.clip(yt, 1e-3, None))),
        "mae": float(np.mean(np.abs(yp - yt))),
        "rmse_log": float(np.sqrt(np.mean((np.log1p(np.clip(yp, 0.0, None)) - np.log1p(yt)) ** 2))),
        "spearman": _spearman(yt, yp),
    }



def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size <= 1:
        return 0.0
    ra = _rank(a)
    rb = _rank(b)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])



def _rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    return ranks



def collect_predictions(
    model: tf.keras.Model,
    encoded_samples: Sequence[Mapping[str, object]],
    target_names: Sequence[str],
    device: str | None = None,
) -> Dict[str, Dict[str, List[float]]]:
    out = {name: {"y_true": [], "y_pred": []} for name in target_names}
    ctx = tf.device(device) if device else _null_device()
    with ctx:
        for sample in encoded_samples:
            pred = model(sample, training=False)
            targets = np.asarray(sample["targets"], dtype=np.float32)
            for idx, name in enumerate(target_names):
                out[name]["y_true"].append(float(targets[idx]))
                out[name]["y_pred"].append(float(tf.reshape(pred[name], [-1])[0].numpy()))
    return out



def evaluate_model(
    model: tf.keras.Model,
    encoded_samples: Sequence[Mapping[str, object]],
    target_names: Sequence[str],
    device: str | None = None,
) -> Dict[str, Dict[str, float]]:
    preds = collect_predictions(model, encoded_samples, target_names, device=device)
    return {name: compute_metrics(v["y_true"], v["y_pred"]) for name, v in preds.items()}


class _null_device:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False
