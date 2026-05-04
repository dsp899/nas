
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

from rnn_benchlib.config.schemas import BenchmarkRecord, MEASUREMENT_SCHEMA_VERSION
from rnn_benchlib.storage.jsonl import write_jsonl


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _delegate_backend_hint(runtime_kind: str, uses_flex: bool) -> str:
    if runtime_kind == "float":
        return "tensorflow"
    if uses_flex:
        return "flex"
    return "tflite_or_xnnpack"


def benchmark_record_to_measurement(record: BenchmarkRecord, *, graph_id: str, uses_flex: bool, quantization_mode: str = "none") -> Dict[str, Any]:
    extra = record.extra or {}
    return {
        "schema_version": MEASUREMENT_SCHEMA_VERSION,
        "measurement_id": f"measure_{record.model_id}_{record.runtime_kind}_{record.video_index}",
        "graph_id": graph_id,
        "model_id": record.model_id,
        "profile": {
            "hardware_target": record.experiment.device_name,
            "execution_site": (record.experiment.extra or {}).get("execution_site", "host"),
            "runtime_preset": record.runtime_kind,
            "threads": record.threads,
            "batch_size": record.batch_size,
            "warmup_runs": (record.experiment.extra or {}).get("warmup_runs"),
            "steady_runs": (record.experiment.extra or {}).get("steady_runs"),
            "unit": "clip_window",
        },
        "runtime_context": {
            "encoder_runtime": "tensorflow" if record.runtime_kind == "float" else "tflite",
            "head_runtime": "tensorflow" if record.runtime_kind == "float" else "tflite",
            "encoder_delegate_backend": _delegate_backend_hint(record.runtime_kind, uses_flex),
            "head_delegate_backend": "tensorflow" if record.runtime_kind == "float" else "xnnpack_or_default",
            "quantization_mode": quantization_mode,
            "memory_mode": record.memory_mode,
        },
        "targets": {
            "latency_clip_e2e_wall_ms": record.runtime_summary.steady_clip_e2e_wall_mean_ms,
            "latency_clip_e2e_sum_ms": record.runtime_summary.steady_clip_e2e_sum_mean_ms,
            "latency_clip_encoder_ms": record.runtime_summary.steady_clip_encoder_mean_ms,
            "latency_clip_bridge_ms": record.runtime_summary.steady_clip_bridge_mean_ms,
            "latency_clip_head_ms": record.runtime_summary.steady_clip_head_mean_ms,
            "latency_video_e2e_wall_ms": record.runtime_summary.steady_video_e2e_wall_mean_ms,
            "latency_video_e2e_sum_ms": record.runtime_summary.steady_video_e2e_sum_mean_ms,
            "latency_video_encoder_sum_ms": record.runtime_summary.steady_video_encoder_sum_mean_ms,
            "latency_video_bridge_sum_ms": record.runtime_summary.steady_video_bridge_sum_mean_ms,
            "latency_video_head_clip_sum_ms": record.runtime_summary.steady_video_head_clip_sum_mean_ms,
            "latency_video_aggregation_ms": record.runtime_summary.steady_video_aggregation_mean_ms,
            "latency_video_head_ms": record.runtime_summary.steady_video_head_mean_ms,
        },
        "statistics": {
            "clip_e2e_wall": {
                "count": record.runtime_summary.steady_clip_e2e_wall_count,
                "mean_ms": record.runtime_summary.steady_clip_e2e_wall_mean_ms,
                "median_ms": record.runtime_summary.steady_clip_e2e_wall_median_ms,
                "std_ms": record.runtime_summary.steady_clip_e2e_wall_std_ms,
                "min_ms": record.runtime_summary.steady_clip_e2e_wall_min_ms,
                "max_ms": record.runtime_summary.steady_clip_e2e_wall_max_ms,
                "p95_ms": record.runtime_summary.steady_clip_e2e_wall_p95_ms,
                "p99_ms": record.runtime_summary.steady_clip_e2e_wall_p99_ms,
            },
            "clip_e2e_sum": {
                "count": record.runtime_summary.steady_clip_e2e_sum_count,
                "mean_ms": record.runtime_summary.steady_clip_e2e_sum_mean_ms_stat,
                "median_ms": record.runtime_summary.steady_clip_e2e_sum_median_ms,
                "std_ms": record.runtime_summary.steady_clip_e2e_sum_std_ms,
                "min_ms": record.runtime_summary.steady_clip_e2e_sum_min_ms,
                "max_ms": record.runtime_summary.steady_clip_e2e_sum_max_ms,
                "p95_ms": record.runtime_summary.steady_clip_e2e_sum_p95_ms,
                "p99_ms": record.runtime_summary.steady_clip_e2e_sum_p99_ms,
            },
            "init_ms": record.runtime_summary.init_ms,
        },
        "shape_context": {
            "seq": record.seq,
            "clips_per_video": record.clips_per_video,
            "feature_dim": extra.get("feature_dim"),
            "video_steps": extra.get("video_steps"),
            "direction": extra.get("direction"),
            "rnn": extra.get("rnn"),
            "units": [extra.get("units_0"), extra.get("units_1"), extra.get("units_2")],
            "head_units": extra.get("head_units"),
            "num_classes": extra.get("num_classes"),
            "video_decision": extra.get("video_decision"),
            "video_decision_input": extra.get("video_decision_input"),
        },
        "status": {
            "ok": True,
            "measurement_error": None,
        },
        "created_at_utc": utc_now_iso(),
    }


def write_measurement_jsonl(path: str, records: List[Dict[str, Any]]) -> None:
    write_jsonl(path, records)
