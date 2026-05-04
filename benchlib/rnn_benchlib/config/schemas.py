from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional


RNNType = Literal["lstm", "gru"]
DirectionType = Literal["unidirectional", "bidirectional"]
MemoryMode = Literal["none", "carry_forward"]
VideoDecisionStrategy = Literal["average", "majority", "max_prob"]
VideoDecisionInput = Literal["clip_logits", "clip_embeddings"]
DatasetProfile = Literal["ucf50", "ucf101"]
RuntimeRecommendation = Literal["tflite_runtime_or_tensorflow", "tensorflow_full"]
ConversionMode = Literal[
    "builtin_only",
    "builtin_plus_select_tf_ops",
    "failed",
]
ARTIFACT_SCHEMA_VERSION = "rnn_benchlib_artifacts_v4"
GRAPH_SCHEMA_VERSION = "rnn_perfsage_graph_v3"
MEASUREMENT_SCHEMA_VERSION = "rnn_perfsage_measurement_v2"


@dataclass(frozen=True)
class SearchSpace:
    layers: tuple[int, ...] = (1, 2, 3)
    rnn: tuple[str, ...] = ("lstm", "gru")
    units_0: tuple[int, ...] = (8, 16, 32, 64, 128, 256, 300, 512, 700, 900, 1024)
    units_1: tuple[int, ...] = (0, 8, 16, 32, 64, 128, 256, 300, 512, 700, 900, 1024)
    units_2: tuple[int, ...] = (0, 8, 16, 32, 64, 128, 256, 300, 512, 700, 900, 1024)
    direction: tuple[str, ...] = ("unidirectional", "bidirectional")
    memory_mode: tuple[str, ...] = ("none", "carry_forward")
    seq: tuple[int, ...] = (3, 6, 9, 12)
    head_units: tuple[int, ...] = (64, 128, 256, 512, 1024)
    video_decision: tuple[str, ...] = ("average", "majority", "max_prob")
    video_decision_input: tuple[str, ...] = ("clip_logits", "clip_embeddings")

@dataclass(frozen=True)
class ExperimentConfig:
    dataset_profile: DatasetProfile = "ucf101"
    num_classes: int = 101
    feature_dim: int = 512
    video_steps: int = 36
    search_space: SearchSpace = field(default_factory=SearchSpace)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        return payload


@dataclass(frozen=True)
class ModelSpec:
    layers: int
    rnn: RNNType
    units_0: int
    units_1: int
    units_2: int
    direction: DirectionType
    memory_mode: MemoryMode
    seq: int
    head_units: int
    num_classes: int
    video_decision: VideoDecisionStrategy
    video_decision_input: VideoDecisionInput

    def units_list(self) -> List[int]:
        raw = [self.units_0]
        if self.layers >= 2:
            raw.append(self.units_1)
        if self.layers >= 3:
            raw.append(self.units_2)
        return raw

    def normalized_units_list(self) -> List[int]:
        return [u for u in self.units_list() if u > 0]

    def encoder_output_dim(self) -> int:
        last_units = self.normalized_units_list()[-1]
        if self.direction == "bidirectional":
            return last_units * 2
        return last_units

    def clips_per_video(self, video_steps: int = 36) -> int:
        return video_steps // self.seq

    def uses_inter_clip_memory(self) -> bool:
        return self.memory_mode == "carry_forward"

    def as_key_dict(self) -> Dict[str, Any]:
        return {
            "layers": self.layers,
            "rnn": self.rnn,
            "units_0": self.units_0,
            "units_1": self.units_1,
            "units_2": self.units_2,
            "direction": self.direction,
            "memory_mode": self.memory_mode,
            "seq": self.seq,
            "head_units": self.head_units,
            "num_classes": self.num_classes,
            "video_decision": self.video_decision,
            "video_decision_input": self.video_decision_input,
        }


@dataclass(frozen=True)
class FeatureSpec:
    source: Literal["synthetic", "npy", "cnn_precomputed"] = "synthetic"
    backbone: Optional[Literal["vgg16", "inceptionv3", "resnet50"]] = None
    pooling: str = "avg"
    feature_dim: int = 256
    video_steps: int = 36
    frame_size: int = 224
    preprocess_name: Optional[str] = None

    def clips_per_video(self, seq: int) -> int:
        return self.video_steps // seq


@dataclass
class ConversionInfo:
    status: Literal["ok", "partial", "failed"]
    conversion_mode: ConversionMode
    uses_flex: bool
    ops: List[str] = field(default_factory=list)
    target_runtime_recommendation: RuntimeRecommendation = "tensorflow_full"
    error: Optional[str] = None
    warning: Optional[str] = None
    quantization_mode: Literal["none", "dynamic_range", "float16", "int8"] = "none"
    encoder_tflite_path: Optional[str] = None
    head_tflite_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ArtifactPaths:
    model_dir: str
    meta_dir: str
    spec_path: str
    manifest_path: str
    source_dir: str
    encoder_keras_dir: str
    head_keras_dir: str
    compiled_dir: str
    encoder_tflite_path: Optional[str]
    head_tflite_path: Optional[str]
    graphs_dir: str
    encoder_tflite_graph_path: Optional[str]
    head_tflite_graph_path: Optional[str]
    graph_record_path: str
    reports_dir: str
    conversion_report_path: str
    benchmark_dir: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModelRecord:
    model_id: str
    seed: int
    spec: ModelSpec
    feature_spec: FeatureSpec
    conversion: ConversionInfo
    artifacts: ArtifactPaths
    created_at_utc: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "seed": self.seed,
            "spec": asdict(self.spec),
            "feature_spec": asdict(self.feature_spec),
            "conversion": self.conversion.to_dict(),
            "artifacts": self.artifacts.to_dict(),
            "created_at_utc": self.created_at_utc,
        }


@dataclass
class ExperimentMeta:
    experiment_id: str
    experiment_name: str
    created_at_utc: str
    host_name: str
    device_name: str
    runtime: Literal["float", "tflite", "both"]
    notes: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ClipTiming:
    clip_index: int
    carry_state: bool
    clip_encoder_ms: float
    clip_bridge_ms: float
    clip_head_ms: float
    clip_e2e_sum_ms: float
    clip_e2e_wall_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VideoTiming:
    video_encoder_sum_ms: float
    video_bridge_sum_ms: float
    video_head_clip_sum_ms: float
    video_aggregation_ms: float
    video_head_ms: float
    video_e2e_sum_ms: float
    video_e2e_wall_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class NumericCheck:
    max_abs_diff: float
    mean_abs_diff: float
    allclose_atol_1e5_rtol_1e5: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RuntimeSummary:
    init_ms: Optional[float]
    steady_clip_encoder_mean_ms: float
    steady_clip_bridge_mean_ms: float
    steady_clip_head_mean_ms: float
    steady_clip_e2e_sum_mean_ms: float
    steady_video_encoder_sum_mean_ms: float
    steady_video_bridge_sum_mean_ms: float
    steady_video_head_clip_sum_mean_ms: float
    steady_video_aggregation_mean_ms: float
    steady_video_head_mean_ms: float
    steady_video_e2e_sum_mean_ms: float
    steady_video_e2e_wall_mean_ms: float
    steady_clip_e2e_sum_count: int
    steady_clip_e2e_sum_mean_ms_stat: float
    steady_clip_e2e_sum_median_ms: float
    steady_clip_e2e_sum_std_ms: float
    steady_clip_e2e_sum_min_ms: float
    steady_clip_e2e_sum_max_ms: float
    steady_clip_e2e_sum_p95_ms: float
    steady_clip_e2e_sum_p99_ms: float
    steady_clip_e2e_wall_count: int
    steady_clip_e2e_wall_mean_ms: float
    steady_clip_e2e_wall_median_ms: float
    steady_clip_e2e_wall_std_ms: float
    steady_clip_e2e_wall_min_ms: float
    steady_clip_e2e_wall_max_ms: float
    steady_clip_e2e_wall_p95_ms: float
    steady_clip_e2e_wall_p99_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkRecord:
    experiment: ExperimentMeta
    model_id: str
    runtime_kind: Literal["float", "tflite"]
    memory_mode: MemoryMode
    seq: int
    video_index: int
    clips_per_video: int
    threads: int
    batch_size: int
    runtime_summary: RuntimeSummary
    clip_timings: List[ClipTiming]
    video_timing: VideoTiming
    numeric_check: Optional[NumericCheck] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment": self.experiment.to_dict(),
            "model_id": self.model_id,
            "runtime_kind": self.runtime_kind,
            "memory_mode": self.memory_mode,
            "seq": self.seq,
            "video_index": self.video_index,
            "clips_per_video": self.clips_per_video,
            "threads": self.threads,
            "batch_size": self.batch_size,
            "runtime_summary": self.runtime_summary.to_dict(),
            "clip_timings": [c.to_dict() for c in self.clip_timings],
            "video_timing": self.video_timing.to_dict(),
            "numeric_check": None if self.numeric_check is None else self.numeric_check.to_dict(),
            "extra": self.extra,
        }
