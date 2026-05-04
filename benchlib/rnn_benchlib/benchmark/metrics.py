
from __future__ import annotations

import statistics
from typing import Iterable, List, Optional, Sequence

import numpy as np

from rnn_benchlib.config.schemas import NumericCheck, RuntimeSummary


def _safe_mean(values: Sequence[float]) -> float:
    vals = [float(v) for v in values]
    return float(statistics.mean(vals)) if vals else 0.0


def _stats(values: Sequence[float]) -> dict:
    vals = [float(v) for v in values]
    if not vals:
        return {
            'count': 0,
            'mean_ms': 0.0,
            'median_ms': 0.0,
            'std_ms': 0.0,
            'min_ms': 0.0,
            'max_ms': 0.0,
            'p95_ms': 0.0,
            'p99_ms': 0.0,
        }
    arr = np.asarray(vals, dtype=np.float64)
    return {
        'count': int(arr.size),
        'mean_ms': float(np.mean(arr)),
        'median_ms': float(np.median(arr)),
        'std_ms': float(np.std(arr, ddof=0)),
        'min_ms': float(np.min(arr)),
        'max_ms': float(np.max(arr)),
        'p95_ms': float(np.percentile(arr, 95)),
        'p99_ms': float(np.percentile(arr, 99)),
    }


def summarize_component_times(
    clip_encoder_times_ms: Sequence[float],
    clip_bridge_times_ms: Sequence[float],
    clip_head_times_ms: Sequence[float],
    clip_e2e_sum_times_ms: Sequence[float],
    clip_e2e_wall_times_ms: Sequence[float],
    video_encoder_sum_times_ms: Sequence[float],
    video_bridge_sum_times_ms: Sequence[float],
    video_head_clip_sum_times_ms: Sequence[float],
    video_aggregation_times_ms: Sequence[float],
    video_head_times_ms: Sequence[float],
    video_e2e_sum_times_ms: Sequence[float],
    video_e2e_wall_times_ms: Sequence[float],
    init_ms: Optional[float],
) -> RuntimeSummary:
    sum_stats = _stats(clip_e2e_sum_times_ms)
    wall_stats = _stats(clip_e2e_wall_times_ms)
    return RuntimeSummary(
        init_ms=None if init_ms is None else float(init_ms),
        steady_clip_encoder_mean_ms=_safe_mean(clip_encoder_times_ms),
        steady_clip_bridge_mean_ms=_safe_mean(clip_bridge_times_ms),
        steady_clip_head_mean_ms=_safe_mean(clip_head_times_ms),
        steady_clip_e2e_sum_mean_ms=_safe_mean(clip_e2e_sum_times_ms),
        steady_video_encoder_sum_mean_ms=_safe_mean(video_encoder_sum_times_ms),
        steady_video_bridge_sum_mean_ms=_safe_mean(video_bridge_sum_times_ms),
        steady_video_head_clip_sum_mean_ms=_safe_mean(video_head_clip_sum_times_ms),
        steady_video_aggregation_mean_ms=_safe_mean(video_aggregation_times_ms),
        steady_video_head_mean_ms=_safe_mean(video_head_times_ms),
        steady_video_e2e_sum_mean_ms=_safe_mean(video_e2e_sum_times_ms),
        steady_video_e2e_wall_mean_ms=_safe_mean(video_e2e_wall_times_ms),
        steady_clip_e2e_sum_count=sum_stats['count'],
        steady_clip_e2e_sum_mean_ms_stat=sum_stats['mean_ms'],
        steady_clip_e2e_sum_median_ms=sum_stats['median_ms'],
        steady_clip_e2e_sum_std_ms=sum_stats['std_ms'],
        steady_clip_e2e_sum_min_ms=sum_stats['min_ms'],
        steady_clip_e2e_sum_max_ms=sum_stats['max_ms'],
        steady_clip_e2e_sum_p95_ms=sum_stats['p95_ms'],
        steady_clip_e2e_sum_p99_ms=sum_stats['p99_ms'],
        steady_clip_e2e_wall_count=wall_stats['count'],
        steady_clip_e2e_wall_mean_ms=wall_stats['mean_ms'],
        steady_clip_e2e_wall_median_ms=wall_stats['median_ms'],
        steady_clip_e2e_wall_std_ms=wall_stats['std_ms'],
        steady_clip_e2e_wall_min_ms=wall_stats['min_ms'],
        steady_clip_e2e_wall_max_ms=wall_stats['max_ms'],
        steady_clip_e2e_wall_p95_ms=wall_stats['p95_ms'],
        steady_clip_e2e_wall_p99_ms=wall_stats['p99_ms'],
    )


def compute_numeric_check(y_ref: np.ndarray, y_test: np.ndarray, atol: float = 1e-5, rtol: float = 1e-5) -> NumericCheck:
    y_ref = np.asarray(y_ref)
    y_test = np.asarray(y_test)
    if y_ref.shape != y_test.shape:
        raise ValueError(f"Shapes incompatibles para comparación: y_ref={y_ref.shape}, y_test={y_test.shape}")
    abs_diff = np.abs(y_ref - y_test)
    return NumericCheck(
        max_abs_diff=float(np.max(abs_diff)),
        mean_abs_diff=float(np.mean(abs_diff)),
        allclose_atol_1e5_rtol_1e5=bool(np.allclose(y_ref, y_test, atol=atol, rtol=rtol)),
    )


def as_float_list(values: Iterable[float]) -> List[float]:
    return [float(v) for v in values]
