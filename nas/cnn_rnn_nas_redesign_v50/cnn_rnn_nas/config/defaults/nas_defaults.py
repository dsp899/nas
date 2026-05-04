from dataclasses import dataclass
from typing import Optional, Tuple

from ..specs.runtime import OptimizerSpec, SchedulerSpec
from .rnn_defaults import RNN_COMPONENT_DEFAULTS
from ..supported.cnn_supported import CNN_BACKBONES


DEFAULT_NAS_CONTROLLER_OPTIMIZER_SPEC = OptimizerSpec(
    name="adam",
    learning_rate=0.001,
    momentum=0.9,
    nesterov=True,
    weight_decay=0.0,
)
DEFAULT_NAS_CONTROLLER_SCHEDULER_SPEC = SchedulerSpec(
    reduce_lr_on_plateau=False,
    reduce_lr_factor=0.5,
    reduce_lr_patience=2,
    min_learning_rate=1e-6,
)


@dataclass(frozen=True)
class NasControllerModelDefaults:
    lstm_dim: int = 64


@dataclass(frozen=True)
class NasControllerTrainingDefaults:
    sampling_epochs: int = 200
    samples_per_epoch: int = 8
    training_epochs: int = 5
    sampling_attempts_multiplier: int = 300
    sampling_attempts_minimum: int = 1000
    reward_baseline_strategy: str = "ema"
    reward_baseline_ema_decay: float = 0.7
    reward_standardize_advantage: bool = True
    rolling_window: Optional[int] = None
    rolling_window_multiplier: int = 8


@dataclass(frozen=True)
class NasSearchSpaceDefaults:
    layers: Tuple[int, ...] = (1, 2, 3)
    rnn: Tuple[str, ...] = ("gru", "lstm")
    units_0: Tuple[int, ...] = (8, 16, 32, 64, 128, 256, 300, 512, 700, 900, 1024)
    units_1: Tuple[int, ...] = (0, 8, 16, 32, 64, 128, 256, 300, 512, 700, 900, 1024)
    units_2: Tuple[int, ...] = (0, 8, 16, 32, 64, 128, 256, 300, 512, 700, 900, 1024)
    direction: Tuple[str, ...] = ("unidirectional", "bidirectional")
    memory_mode: Tuple[str, ...] = ("none", "carry_forward")
    seq: Tuple[int, ...] = (3, 6, 9, 12)
    head_units: Tuple[int, ...] = (64, 128, 256, 512, 1024)
    video_decision: Tuple[str, ...] = ("average", "majority", "max_prob")
    video_decision_input: Tuple[str, ...] = ("clip_logits", "clip_embeddings")
    cnn: Tuple[str, ...] = CNN_BACKBONES


@dataclass(frozen=True)
class NasDefaults:
    controller_model: NasControllerModelDefaults = NasControllerModelDefaults()
    controller_optimizer: OptimizerSpec = DEFAULT_NAS_CONTROLLER_OPTIMIZER_SPEC
    controller_scheduler: SchedulerSpec = DEFAULT_NAS_CONTROLLER_SCHEDULER_SPEC
    controller_training: NasControllerTrainingDefaults = NasControllerTrainingDefaults()
    search_space: NasSearchSpaceDefaults = NasSearchSpaceDefaults()


NAS_COMPONENT_DEFAULTS = NasDefaults()
