from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Literal, Optional

DatasetProfile = Literal["ucf50", "ucf101"]
PoolingMode = Literal["avg", "max"]
CnnRuntime = Literal["float", "tflite", "xmodel"]

DEFAULT_NUM_CLASSES = {"ucf50": 50, "ucf101": 101}


@dataclass(frozen=True)
class CnnExperimentConfig:
    dataset_profile: DatasetProfile = "ucf101"
    num_classes: int = 101
    calibration_samples: int = 64

    @classmethod
    def from_dataset_profile(cls, dataset_profile: DatasetProfile) -> "CnnExperimentConfig":
        return cls(dataset_profile=dataset_profile, num_classes=DEFAULT_NUM_CLASSES[dataset_profile])

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CnnModelSpec:
    backbone_name: str
    input_size: int
    pooling_mode: PoolingMode = "avg"
    projection_dim: int = 256
    num_classes: int = 101

    def to_key_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TfliteExportConfig:
    optimize_default: bool = False
    allow_select_tf_ops: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class XilinxQuantConfig:
    vitis_ai_version: str = "3.0"
    target_board: str = "zcu102"
    calibration_samples: int = 64
    calibration_seed: int = 1234
    arch_json: str = "/opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU102/arch.json"
    quantization_mode: str = "ptq_int8"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class XilinxCompileConfig:
    target_board: str = "zcu102"
    arch_json: str = "/opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU102/arch.json"
    vai_c_bin: str = "vai_c_tensorflow2"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ConversionStatus:
    status: Literal["ready", "missing", "failed"]
    path: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CnnArtifactRecord:
    model_id: str
    spec: CnnModelSpec
    experiment: CnnExperimentConfig
    created_at_utc: str
    model_dir: str
    float_extractor_path: str
    float_classifier_path: str
    feature_dim: int
    tflite_extractor: ConversionStatus = field(default_factory=lambda: ConversionStatus(status="missing"))
    tflite_classifier: ConversionStatus = field(default_factory=lambda: ConversionStatus(status="missing"))
    quantized_extractor: ConversionStatus = field(default_factory=lambda: ConversionStatus(status="missing"))
    quantized_classifier: ConversionStatus = field(default_factory=lambda: ConversionStatus(status="missing"))
    xmodel_extractor: ConversionStatus = field(default_factory=lambda: ConversionStatus(status="missing"))
    xilinx_bundle_dir: Optional[str] = None
    notes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
