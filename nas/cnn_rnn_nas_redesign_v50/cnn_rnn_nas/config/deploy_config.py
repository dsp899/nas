from dataclasses import asdict, dataclass
from typing import Any, Dict

DEFAULT_ZCU102_ARCH_JSON = "/opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU102/arch.json"


@dataclass(frozen=True)
class CnnVitisAiDeployConfig:
    vitis_ai_version: str = "3.0"
    target_board: str = "zcu102"
    output_layer: str = "frame_features"
    calibration_split: str = "train"
    calibration_samples: int = 128
    quantization_mode: str = "ptq_int8"
    arch_json: str = DEFAULT_ZCU102_ARCH_JSON
    save_calibration_numpy: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RnnTfliteExportConfig:
    runtime: str = "tflite"
    tflite_precision: str = "fp32"

    def __post_init__(self) -> None:
        precision = self.tflite_precision.strip().lower()
        runtime = self.runtime.strip().lower()
        if runtime != "tflite":
            raise ValueError(f"Runtime de exportación no soportado: {self.runtime!r}")
        if precision not in {"fp32"}:
            raise ValueError(f"Precisión TFLite no soportada en esta versión: {self.tflite_precision!r}")
        object.__setattr__(self, "runtime", runtime)
        object.__setattr__(self, "tflite_precision", precision)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CnnQuantEvalConfig:
    eval_split: str = "test"
    video_aggregation: str = "mean_probs"
    save_frame_predictions: bool = False
    save_video_predictions: bool = True

    def __post_init__(self) -> None:
        eval_split = self.eval_split.strip().lower()
        aggregation = self.video_aggregation.strip().lower()
        if eval_split not in {"train", "val", "test"}:
            raise ValueError(f"Split de evaluación cuantizada no soportado: {self.eval_split!r}")
        if aggregation not in {"mean_probs"}:
            raise ValueError(f"Agregación de vídeo no soportada: {self.video_aggregation!r}")
        object.__setattr__(self, "eval_split", eval_split)
        object.__setattr__(self, "video_aggregation", aggregation)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CnnExtractorEvalConfig:
    eval_split: str = "test"
    video_aggregation: str = "mean_probs"
    save_frame_predictions: bool = False
    save_video_predictions: bool = True

    def __post_init__(self) -> None:
        eval_split = self.eval_split.strip().lower()
        aggregation = self.video_aggregation.strip().lower()
        if eval_split not in {"train", "val", "test"}:
            raise ValueError(f"Split de evaluación del extractor no soportado: {self.eval_split!r}")
        if aggregation not in {"mean_probs"}:
            raise ValueError(f"Agregación de vídeo no soportada: {self.video_aggregation!r}")
        object.__setattr__(self, "eval_split", eval_split)
        object.__setattr__(self, "video_aggregation", aggregation)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RnnTfliteEvalConfig:
    eval_split: str = "test"
    save_clip_predictions: bool = False
    save_video_predictions: bool = True

    def __post_init__(self) -> None:
        eval_split = self.eval_split.strip().lower()
        if eval_split not in {"train", "val", "test"}:
            raise ValueError(f"Split de evaluación TFLite no soportado: {self.eval_split!r}")
        object.__setattr__(self, "eval_split", eval_split)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RnnDeployEvalConfig:
    eval_split: str = "test"
    save_clip_predictions: bool = False
    save_video_predictions: bool = True
    save_quantized_sequences: bool = True

    def __post_init__(self) -> None:
        eval_split = self.eval_split.strip().lower()
        if eval_split not in {"train", "val", "test"}:
            raise ValueError(f"Split de evaluación deploy RNN no soportado: {self.eval_split!r}")
        object.__setattr__(self, "eval_split", eval_split)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
