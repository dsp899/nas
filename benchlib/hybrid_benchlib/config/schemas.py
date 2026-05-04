from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Literal, Optional

CnnBackend = Literal["float", "tflite", "xmodel"]
RnnBackend = Literal["float", "tflite"]
RuntimePreset = Literal["float_all", "tflite_all", "xmodel_tflite"]
PipelineOverlapMode = Literal["cnn_rnn_overlap", "cnn_rnn_serialized"]


@dataclass(frozen=True)
class HybridPipelineConfig:
    cnn_backend: CnnBackend = "xmodel"
    rnn_backend: RnnBackend = "tflite"
    cnn_workers: int = 3
    hop: int = 1
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HybridBenchmarkConfig:
    runtime_preset: RuntimePreset = "xmodel_tflite"
    overlap_mode: PipelineOverlapMode = "cnn_rnn_overlap"
    cnn_workers: int = 3
    hop: int = 1
    sample_stride_frames: int = 1
    video_fps: float = 30.0
    num_videos: int = 8
    frames_per_video: int = 64
    warmup_runs: int = 3
    steady_runs: int = 5
    threads: int = 1
    seed: int = 1234
    prefer_cached_component_benchmarks: bool = True
    xmodel_summary_path: Optional[str] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HybridBundleRecord:
    hybrid_model_id: str
    cnn_model_id: str
    rnn_model_id: str
    created_at_utc: str
    feature_dim: int
    num_classes: int
    pipeline_config: Dict[str, Any]
    compatibility: Dict[str, Any]
    references: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
