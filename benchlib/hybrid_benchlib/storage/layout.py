from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict

from benchlib_common.artifacts.layout import ensure_dir


@dataclass(frozen=True)
class HybridArtifactPaths:
    hybrid_dir: str
    metadata_path: str
    registry_path: str
    benchmark_dir: str

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class HybridExperimentPaths:
    experiment_dir: str
    results_jsonl_path: str
    summary_json_path: str
    event_trace_jsonl_path: str

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


def build_artifact_paths(output_root: str, hybrid_model_id: str) -> HybridArtifactPaths:
    root = Path(output_root)
    hybrid_dir = root / "hybrid_models" / hybrid_model_id
    benchmark_dir = hybrid_dir / "benchmark"
    ensure_dir(benchmark_dir)
    return HybridArtifactPaths(
        hybrid_dir=str(hybrid_dir),
        metadata_path=str(hybrid_dir / "metadata.json"),
        registry_path=str(root / "hybrid_registry.json"),
        benchmark_dir=str(benchmark_dir),
    )


def build_experiment_paths(output_root: str, experiment_id: str) -> HybridExperimentPaths:
    root = Path(output_root)
    experiment_dir = root / "hybrid_benchmark_runs" / experiment_id
    ensure_dir(experiment_dir)
    return HybridExperimentPaths(
        experiment_dir=str(experiment_dir),
        results_jsonl_path=str(experiment_dir / "benchmark_results.jsonl"),
        summary_json_path=str(experiment_dir / "summary.json"),
        event_trace_jsonl_path=str(experiment_dir / "event_trace.jsonl"),
    )
