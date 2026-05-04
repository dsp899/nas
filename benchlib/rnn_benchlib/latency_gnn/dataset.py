from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Sequence

from rnn_benchlib.latency_gnn.artifacts import (
    MeasurementBundle,
    make_dataset_id,
    read_dataset_rows,
    resolve_profile_ids,
    iter_measurement_bundles_for_dataset,
    write_dataset_artifacts,
)

PRIMARY_TARGET = "latency_clip_e2e_wall_ms"
AUXILIARY_TARGETS = [
    "latency_clip_encoder_ms",
    "latency_clip_bridge_ms",
    "latency_clip_head_ms",
]
ALL_TARGETS = AUXILIARY_TARGETS + [PRIMARY_TARGET]


@dataclass
class AggregatedLatencySample:
    sample_id: str
    lot_id: str
    benchmark_id: str
    graph_id: str
    model_id: str
    runtime_kind: str
    profile_id: str
    graph_record: Dict[str, Any]
    targets: Dict[str, float]
    target_std: Dict[str, float]
    target_p95: Dict[str, float]
    measurement_count: int
    weight: float
    graph_features: Dict[str, Any]
    runtime_features: Dict[str, Any]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "AggregatedLatencySample":
        return cls(**payload)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _std(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mu = mean(values)
    return (sum((v - mu) ** 2 for v in values) / len(values)) ** 0.5


def _flatten_units(units: Sequence[Any], size: int = 3) -> list[int]:
    vals = [int(u or 0) for u in units[:size]]
    if len(vals) < size:
        vals.extend([0] * (size - len(vals)))
    return vals


def _extract_graph_features(graph_record: Dict[str, Any]) -> Dict[str, Any]:
    model_config = graph_record.get("model_config", {})
    encoder = model_config.get("encoder_spec", {})
    head = model_config.get("head_spec", {})
    feature_spec = model_config.get("feature_spec", {})
    decision = model_config.get("decision_spec", {})
    units = _flatten_units(encoder.get("units", []))
    edges = graph_record.get("edges", {})
    return {
        "rnn": encoder.get("rnn", "unknown"),
        "layers": int(encoder.get("layers", 0) or 0),
        "units": units,
        "total_units": int(sum(units)),
        "max_units": int(max(units) if units else 0),
        "direction": encoder.get("direction", "unknown"),
        "memory_mode": encoder.get("memory_mode", "unknown"),
        "seq": int(encoder.get("seq", 0) or 0),
        "clip_embedding_dim": int(encoder.get("clip_embedding_dim", 0) or 0),
        "head_units": int(head.get("head_units", 0) or 0),
        "num_classes": int(head.get("num_classes", 0) or 0),
        "video_decision": decision.get("video_decision", "unknown"),
        "video_decision_input": decision.get("video_decision_input", "unknown"),
        "feature_dim": int(feature_spec.get("feature_dim", 0) or 0),
        "video_steps": int(feature_spec.get("video_steps", 0) or 0),
        "pooling": feature_spec.get("pooling", "unknown"),
        "num_op_nodes": int(graph_record.get("graph_meta", {}).get("num_op_nodes", len(graph_record.get("op_nodes", []))) or 0),
        "num_tensor_nodes": int(graph_record.get("graph_meta", {}).get("num_tensor_nodes", len(graph_record.get("tensor_nodes", []))) or 0),
        "num_op_to_tensor_edges": len(edges.get("op_to_tensor", []) or []),
        "num_tensor_to_op_edges": len(edges.get("tensor_to_op", []) or []),
        "num_op_to_op_edges": len(edges.get("op_to_op", []) or []),
        "num_bridge_tensors": len(graph_record.get("bridge_tensors", []) or []),
        "num_delegate_partitions": len(graph_record.get("execution_annotations", {}).get("delegate_partitions", []) or []),
    }


def _extract_runtime_features(row: Dict[str, Any], runtime_kind: str, profile_id: str) -> Dict[str, Any]:
    profile = row.get("profile", {})
    runtime_context = row.get("runtime_context", {})
    return {
        "runtime_kind": runtime_kind,
        "profile_id": profile_id,
        "hardware_target": profile.get("hardware_target", "unknown"),
        "execution_site": profile.get("execution_site", "unknown"),
        "runtime_preset": profile.get("runtime_preset", runtime_kind),
        "threads": int(profile.get("threads", 1) or 1),
        "batch_size": int(profile.get("batch_size", 1) or 1),
        "encoder_runtime": runtime_context.get("encoder_runtime", runtime_kind),
        "head_runtime": runtime_context.get("head_runtime", runtime_kind),
        "encoder_delegate_backend": runtime_context.get("encoder_delegate_backend", "unknown"),
        "head_delegate_backend": runtime_context.get("head_delegate_backend", "unknown"),
        "quantization_mode": runtime_context.get("quantization_mode", "none"),
        "memory_mode_runtime": runtime_context.get("memory_mode", "unknown"),
    }


def _aggregate_measurements(bundle: MeasurementBundle, output_root: str | Path) -> AggregatedLatencySample | None:
    rows = [row for row in bundle.rows if (row.get("status") or {}).get("ok", True)]
    if not rows:
        return None
    first = rows[0]
    grouped = {name: [] for name in ALL_TARGETS}
    p95_map = {name: [] for name in ALL_TARGETS}
    for row in rows:
        targets = row.get("targets", {})
        stats = row.get("statistics", {})
        for name in ALL_TARGETS:
            grouped[name].append(_safe_float(targets.get(name)))
            if name == PRIMARY_TARGET:
                p95_map[name].append(_safe_float((stats.get("clip_e2e_wall") or {}).get("p95_ms")))
    mean_targets = {name: (mean(values) if values else 0.0) for name, values in grouped.items()}
    std_targets = {name: _std(values) for name, values in grouped.items()}
    p95_targets = {name: (mean(values) if values else mean_targets[name]) for name, values in p95_map.items()}
    graph_id = first.get("graph_id") or bundle.graph_record.get("graph_id")
    runtime_features = _extract_runtime_features(first, bundle.runtime_kind, bundle.profile_id)
    graph_features = _extract_graph_features(bundle.graph_record)
    graph_features["clips_per_video"] = int((first.get("shape_context") or {}).get("clips_per_video", 0) or 0)
    sample_key = "::".join([
        bundle.lot_id,
        bundle.benchmark_id,
        graph_id,
        bundle.runtime_kind,
        bundle.profile_id,
        runtime_features["hardware_target"],
        str(runtime_features["threads"]),
        runtime_features["quantization_mode"],
    ])
    root = Path(output_root).expanduser().resolve()
    graph_record_abs = bundle.path.parents[2] / "graphs" / "graph_record.json"
    weight = 1.0 / max(1e-3, 1.0 + std_targets[PRIMARY_TARGET])
    return AggregatedLatencySample(
        sample_id=hashlib.sha1(sample_key.encode("utf-8")).hexdigest()[:16],
        lot_id=bundle.lot_id,
        benchmark_id=bundle.benchmark_id,
        graph_id=graph_id,
        model_id=bundle.model_id,
        runtime_kind=bundle.runtime_kind,
        profile_id=bundle.profile_id,
        graph_record=bundle.graph_record,
        targets=mean_targets,
        target_std=std_targets,
        target_p95=p95_targets,
        measurement_count=len(rows),
        weight=weight,
        graph_features=graph_features,
        runtime_features=runtime_features,
        metadata={
            "measurement_path": str(bundle.path.relative_to(root)),
            "graph_record_path": str(graph_record_abs.relative_to(root)),
            "shape_context": first.get("shape_context") or {},
            "measurement_rows": len(bundle.rows),
            "target_wall_median_ms": median(grouped[PRIMARY_TARGET]) if grouped[PRIMARY_TARGET] else 0.0,
            "lot_member": dict(bundle.member_record),
        },
    )


def discover_latency_samples_for_dataset(
    output_root: str | Path,
    lot_id: str,
    benchmark_id: str,
    runtime_filter: Iterable[str] | None = None,
) -> List[AggregatedLatencySample]:
    samples: List[AggregatedLatencySample] = []
    for bundle in iter_measurement_bundles_for_dataset(output_root, lot_id, benchmark_id, runtime_filter=runtime_filter):
        sample = _aggregate_measurements(bundle, output_root)
        if sample is not None:
            samples.append(sample)
    samples.sort(key=lambda s: (s.model_id, s.runtime_kind, s.profile_id, s.sample_id))
    return samples


def summarize_samples(samples: Sequence[AggregatedLatencySample], lot_id: str, benchmark_id: str, dataset_id: str, profile_ids: Dict[str, str]) -> Dict[str, Any]:
    runtimes: Dict[str, int] = {}
    hardware: Dict[str, int] = {}
    for sample in samples:
        runtimes[sample.runtime_kind] = runtimes.get(sample.runtime_kind, 0) + 1
        hw = sample.runtime_features.get("hardware_target", "unknown")
        hardware[hw] = hardware.get(hw, 0) + 1
    return {
        "schema_version": "rnn_latency_gnn_dataset_v2",
        "dataset_id": dataset_id,
        "lot_id": lot_id,
        "benchmark_id": benchmark_id,
        "profile_ids": profile_ids,
        "num_samples": len(samples),
        "num_graphs": len({sample.graph_id for sample in samples}),
        "runtime_breakdown": runtimes,
        "hardware_breakdown": hardware,
        "primary_target": PRIMARY_TARGET,
        "target_names": list(ALL_TARGETS),
    }


def export_benchmark_dataset(
    output_root: str | Path,
    lot_id: str,
    benchmark_id: str,
    runtime_filter: Iterable[str] | None = None,
) -> Dict[str, Any]:
    profile_ids = resolve_profile_ids(output_root, lot_id, benchmark_id, runtime_filter=runtime_filter)
    runtime_scope = "both" if set(profile_ids.keys()) == {"float", "tflite"} else next(iter(profile_ids.keys()))
    dataset_id = make_dataset_id(output_root, lot_id, benchmark_id, runtime_filter=runtime_filter)
    samples = discover_latency_samples_for_dataset(output_root, lot_id, benchmark_id, runtime_filter=runtime_filter)
    export_payload = summarize_samples(samples, lot_id, benchmark_id, dataset_id, profile_ids)
    export_payload["runtime_scope"] = runtime_scope
    write_dataset_artifacts(output_root, lot_id, dataset_id, export_payload, [sample.to_dict() for sample in samples])
    return export_payload


def load_exported_samples(output_root: str | Path, lot_id: str, dataset_id: str) -> List[AggregatedLatencySample]:
    return [AggregatedLatencySample.from_dict(row) for row in read_dataset_rows(output_root, lot_id, dataset_id)]
