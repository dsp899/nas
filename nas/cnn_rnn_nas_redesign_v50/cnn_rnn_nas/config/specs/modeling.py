from dataclasses import dataclass
from typing import Optional, Tuple

from ..supported.cnn_supported import CNN_BACKBONES
from ..supported.rnn_supported import (
    RNN_DIRECTIONS,
    RNN_MEMORY_MODES,
    RNN_TYPES,
    RNN_VIDEO_DECISIONS,
    RNN_VIDEO_DECISION_INPUTS,
)


def normalize_memory_mode(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in RNN_MEMORY_MODES:
        raise ValueError(f"memory_mode no soportado: {value!r}")
    return normalized


@dataclass(frozen=True)
class ClassificationHeadSpec:
    hidden_units: Tuple[int, ...]
    dropouts: Tuple[float, ...]

    def __post_init__(self) -> None:
        hidden_units = tuple(int(unit) for unit in self.hidden_units)
        dropouts = tuple(float(value) for value in self.dropouts)
        if len(hidden_units) != len(dropouts):
            raise ValueError("hidden_units y dropouts deben tener la misma longitud")
        if len(hidden_units) > 3:
            raise ValueError("El head admite como máximo 3 capas ocultas")
        for unit in hidden_units:
            if unit < 64 or unit > 2048:
                raise ValueError("Cada capa del head debe estar en el rango [64, 2048]")
        for value in dropouts:
            if value < 0.0 or value > 0.7:
                raise ValueError("Cada dropout del head debe estar en el rango [0.0, 0.7]")
        object.__setattr__(self, "hidden_units", hidden_units)
        object.__setattr__(self, "dropouts", dropouts)

    @property
    def num_layers(self) -> int:
        return len(self.hidden_units)

    @property
    def first_hidden_units(self) -> int:
        return int(self.hidden_units[0]) if self.hidden_units else 0

    @property
    def tag(self) -> str:
        return f"head_{self.num_layers}_layers"


@dataclass(frozen=True)
class CnnExtractorSpec:
    backbone: str
    weights: Optional[str] = "imagenet"
    trainable: bool = True
    feature_dim: int = 512

    def __post_init__(self) -> None:
        if self.backbone not in CNN_BACKBONES:
            raise ValueError(f"cnn no soportada: {self.backbone!r}")
        if int(self.feature_dim) <= 0:
            raise ValueError("feature_dim debe ser > 0")

    @property
    def tag(self) -> str:
        return f"{self.backbone}_fd{int(self.feature_dim)}"


@dataclass(frozen=True)
class CnnModelSpec:
    extractor: CnnExtractorSpec
    head: ClassificationHeadSpec


@dataclass(frozen=True)
class RnnEncoderSpec:
    rnn: str
    direction: str
    units: Tuple[int, int, int]
    memory_mode: str
    seq_length: int

    def __post_init__(self) -> None:
        rnn = self.rnn.strip().lower()
        direction = self.direction.strip().lower()
        memory_mode = normalize_memory_mode(self.memory_mode)
        if rnn not in RNN_TYPES:
            raise ValueError(f"rnn no soportada: {self.rnn!r}")
        if direction not in RNN_DIRECTIONS:
            raise ValueError(f"direction no soportada: {self.direction!r}")
        if len(self.units) != 3:
            raise ValueError("units debe tener exactamente tres valores")
        if int(self.seq_length) <= 0:
            raise ValueError("seq_length debe ser > 0")
        object.__setattr__(self, "rnn", rnn)
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "memory_mode", memory_mode)

    @property
    def active_units(self) -> Tuple[int, ...]:
        return tuple(int(unit) for unit in self.units if int(unit) != 0)

    @property
    def encoder_output_dim(self) -> int:
        last = self.active_units[-1]
        return int(last) * (2 if self.direction == "bidirectional" else 1)


@dataclass(frozen=True)
class RnnModelSpec:
    encoder: RnnEncoderSpec
    head: ClassificationHeadSpec
    video_decision: str
    video_decision_input: str

    def __post_init__(self) -> None:
        video_decision = self.video_decision.strip().lower()
        video_decision_input = self.video_decision_input.strip().lower()
        if video_decision not in RNN_VIDEO_DECISIONS:
            raise ValueError(f"video_decision no soportado: {self.video_decision!r}")
        if video_decision_input not in RNN_VIDEO_DECISION_INPUTS:
            raise ValueError(f"video_decision_input no soportado: {self.video_decision_input!r}")
        if video_decision_input == "clip_embeddings" and video_decision != "average":
            raise ValueError("clip_embeddings solo soporta video_decision='average'")
        object.__setattr__(self, "video_decision", video_decision)
        object.__setattr__(self, "video_decision_input", video_decision_input)

    @property
    def tag(self) -> str:
        units_text = "_".join(f"{unit}u" for unit in self.encoder.units)
        head_units = self.head.first_hidden_units
        return (
            f"{self.encoder.rnn}_{units_text}_{self.encoder.direction}_"
            f"{self.encoder.memory_mode}_h{int(head_units)}_{self.video_decision}_{self.video_decision_input}"
        )


@dataclass(frozen=True)
class SequenceSpec:
    seq_length: int

    def __post_init__(self) -> None:
        if int(self.seq_length) <= 0:
            raise ValueError("seq_length debe ser > 0")

    @property
    def tag(self) -> str:
        return f"seq{int(self.seq_length):03d}"
