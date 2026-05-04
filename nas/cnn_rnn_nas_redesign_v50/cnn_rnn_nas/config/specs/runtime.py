from dataclasses import dataclass
from typing import Optional

from ..supported.shared_supported import OPTIMIZER_NAMES


@dataclass(frozen=True)
class OptimizerSpec:
    name: str
    learning_rate: float
    momentum: float = 0.9
    nesterov: bool = True
    weight_decay: float = 0.0
    clipnorm: Optional[float] = None

    def __post_init__(self) -> None:
        name = self.name.strip().lower()
        if name not in OPTIMIZER_NAMES:
            raise ValueError(f"Optimizador no soportado: {self.name!r}")
        object.__setattr__(self, "name", name)


@dataclass(frozen=True)
class SchedulerSpec:
    reduce_lr_on_plateau: bool = False
    reduce_lr_factor: float = 0.5
    reduce_lr_patience: int = 2
    min_learning_rate: float = 1e-6


@dataclass(frozen=True)
class RuntimeSpec:
    gpu: str
    mixed_precision: bool
    random_seed: int
    project_root: str
