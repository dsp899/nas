from dataclasses import dataclass
from typing import Optional

from ..supported.shared_supported import PARTITION_MODES


def normalize_partition_mode(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in PARTITION_MODES:
        raise ValueError(f"partition_mode no soportado: {value!r}")
    return normalized


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    split: str


@dataclass(frozen=True)
class PartitionSpec:
    mode: str
    val_fraction: float

    def __post_init__(self) -> None:
        mode = normalize_partition_mode(self.mode)
        val_fraction = float(self.val_fraction)
        if not 0.0 <= val_fraction < 0.5:
            raise ValueError("val_fraction debe estar en [0, 0.5)")
        if mode == "train_val_test" and val_fraction <= 0.0:
            raise ValueError("train_val_test requiere val_fraction > 0")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "val_fraction", val_fraction)


@dataclass(frozen=True)
class CnnFeatureSourceRef:
    cnn_training_signature: str = ""
    cnn_feature_export_signature: str = ""

    @property
    def is_resolved(self) -> bool:
        return bool(self.cnn_training_signature and self.cnn_feature_export_signature)
