from __future__ import annotations

from typing import Any, Dict

from rnn_benchlib.storage.state import stable_hash


def canonical_benchmark_signature(raw_signature: Dict[str, Any] | None) -> Dict[str, Any]:
    """Normaliza la firma lógica del benchmark.

    Solo incluye los campos que definen la medición lógica reutilizable.
    Quedan fuera los campos de runtime de ejecución y también runtime/runtime_kinds,
    porque esos se materializan como perfiles (float/tflite) dentro del mismo benchmark.
    """
    sig = dict(raw_signature or {})
    payload = {
        "feature_source": str(sig.get("feature_source", "synthetic")),
        "num_videos": int(sig.get("num_videos", 8)),
        "feature_seed": int(sig.get("feature_seed", 0)),
        "distribution": str(sig.get("distribution", "normal")),
        "warmup_runs": int(sig.get("warmup_runs", 5)),
        "steady_runs": int(sig.get("steady_runs", 10)),
        "threads": max(1, int(sig.get("threads", 1))),
        "benchmark_mode": str(sig.get("benchmark_mode", "host_parallel")),
    }
    if sig.get("feature_npy"):
        payload["feature_npy"] = str(sig.get("feature_npy"))
    if sig.get("hardware_target"):
        payload["hardware_target"] = str(sig.get("hardware_target"))
    return payload


def benchmark_id_from_signature(raw_signature: Dict[str, Any] | None) -> str:
    canonical = canonical_benchmark_signature(raw_signature)
    return stable_hash({"benchmark": canonical}, prefix="bench")
