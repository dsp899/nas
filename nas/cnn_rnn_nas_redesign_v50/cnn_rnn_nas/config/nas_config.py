from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

from .defaults.nas_defaults import NAS_COMPONENT_DEFAULTS
from .rnn_config import RNN_DEFAULTS, normalize_memory_mode
from .specs.runtime import OptimizerSpec, SchedulerSpec
from .supported.cnn_supported import CNN_BACKBONES
from .supported.nas_supported import NAS_REWARD_BASELINE_STRATEGIES
from .supported.shared_supported import OPTIMIZER_NAMES


CANONICAL_SEARCH_DIMENSIONS: Tuple[str, ...] = (
    "layers",
    "rnn",
    "units_0",
    "units_1",
    "units_2",
    "direction",
    "memory_mode",
    "seq",
    "head_units",
    "video_decision",
    "video_decision_input",
    "cnn",
)


@dataclass(frozen=True)
class NasSupportedValues:
    reward_baseline_strategies: Tuple[str, ...] = NAS_REWARD_BASELINE_STRATEGIES
    controller_optimizers: Tuple[str, ...] = OPTIMIZER_NAMES


@dataclass(frozen=True)
class NasControllerModelConfig:
    lstm_dim: int = NAS_COMPONENT_DEFAULTS.controller_model.lstm_dim


NasControllerOptimizerConfig = OptimizerSpec
NasControllerSchedulerConfig = SchedulerSpec


@dataclass(frozen=True)
class NasControllerTrainingConfig:
    sampling_epochs: int = NAS_COMPONENT_DEFAULTS.controller_training.sampling_epochs
    samples_per_epoch: int = NAS_COMPONENT_DEFAULTS.controller_training.samples_per_epoch
    training_epochs: int = NAS_COMPONENT_DEFAULTS.controller_training.training_epochs
    sampling_attempts_multiplier: int = NAS_COMPONENT_DEFAULTS.controller_training.sampling_attempts_multiplier
    sampling_attempts_minimum: int = NAS_COMPONENT_DEFAULTS.controller_training.sampling_attempts_minimum
    reward_baseline_strategy: str = NAS_COMPONENT_DEFAULTS.controller_training.reward_baseline_strategy
    reward_baseline_ema_decay: float = NAS_COMPONENT_DEFAULTS.controller_training.reward_baseline_ema_decay
    reward_standardize_advantage: bool = NAS_COMPONENT_DEFAULTS.controller_training.reward_standardize_advantage
    rolling_window: Optional[int] = NAS_COMPONENT_DEFAULTS.controller_training.rolling_window
    rolling_window_multiplier: int = NAS_COMPONENT_DEFAULTS.controller_training.rolling_window_multiplier

    def __post_init__(self) -> None:
        strategy = self.reward_baseline_strategy.strip().lower()
        if strategy not in NAS_REWARD_BASELINE_STRATEGIES:
            raise ValueError("reward_baseline_strategy debe ser 'batch' o 'ema'")
        object.__setattr__(self, "reward_baseline_strategy", strategy)
        if self.rolling_window is not None and int(self.rolling_window) <= 0:
            raise ValueError("rolling_window debe ser > 0 cuando se especifica")
        if int(self.rolling_window_multiplier) <= 0:
            raise ValueError("rolling_window_multiplier debe ser > 0")
        if int(self.sampling_attempts_multiplier) <= 0:
            raise ValueError("sampling_attempts_multiplier debe ser > 0")
        if int(self.sampling_attempts_minimum) <= 0:
            raise ValueError("sampling_attempts_minimum debe ser > 0")

    @property
    def effective_rolling_window(self) -> int:
        if self.rolling_window is not None:
            return max(1, int(self.rolling_window))
        return max(1, int(self.samples_per_epoch) * int(self.rolling_window_multiplier))


@dataclass(frozen=True)
class NasControllerConfig:
    model: NasControllerModelConfig = NasControllerModelConfig()
    optimizer: OptimizerSpec = NAS_COMPONENT_DEFAULTS.controller_optimizer
    scheduler: SchedulerSpec = NAS_COMPONENT_DEFAULTS.controller_scheduler
    training: NasControllerTrainingConfig = NasControllerTrainingConfig()

    @property
    def controller_lstm_dim(self) -> int:
        return self.model.lstm_dim

    @property
    def controller_learning_rate(self) -> float:
        return self.optimizer.learning_rate

    @property
    def controller_sampling_epochs(self) -> int:
        return self.training.sampling_epochs

    @property
    def controller_samples_per_epoch(self) -> int:
        return self.training.samples_per_epoch

    @property
    def controller_training_epochs(self) -> int:
        return self.training.training_epochs

    @property
    def sampling_attempts_multiplier(self) -> int:
        return self.training.sampling_attempts_multiplier

    @property
    def sampling_attempts_minimum(self) -> int:
        return self.training.sampling_attempts_minimum

    @property
    def reward_baseline_strategy(self) -> str:
        return self.training.reward_baseline_strategy

    @property
    def reward_baseline_ema_decay(self) -> float:
        return self.training.reward_baseline_ema_decay

    @property
    def reward_standardize_advantage(self) -> bool:
        return self.training.reward_standardize_advantage

    @property
    def rolling_window(self) -> Optional[int]:
        return self.training.rolling_window

    @property
    def rolling_window_multiplier(self) -> int:
        return self.training.rolling_window_multiplier

    @property
    def effective_rolling_window(self) -> int:
        return self.training.effective_rolling_window

    @property
    def controller_reduce_lr_on_plateau(self) -> bool:
        return self.scheduler.reduce_lr_on_plateau

    @property
    def controller_reduce_lr_factor(self) -> float:
        return self.scheduler.reduce_lr_factor

    @property
    def controller_reduce_lr_patience(self) -> int:
        return self.scheduler.reduce_lr_patience

    @property
    def controller_min_learning_rate(self) -> float:
        return self.scheduler.min_learning_rate


@dataclass(frozen=True)
class NasSearchSpaceDefaults:
    layers: Tuple[int, ...] = NAS_COMPONENT_DEFAULTS.search_space.layers
    rnn: Tuple[str, ...] = NAS_COMPONENT_DEFAULTS.search_space.rnn
    units_0: Tuple[int, ...] = NAS_COMPONENT_DEFAULTS.search_space.units_0
    units_1: Tuple[int, ...] = NAS_COMPONENT_DEFAULTS.search_space.units_1
    units_2: Tuple[int, ...] = NAS_COMPONENT_DEFAULTS.search_space.units_2
    direction: Tuple[str, ...] = NAS_COMPONENT_DEFAULTS.search_space.direction
    memory_mode: Tuple[str, ...] = NAS_COMPONENT_DEFAULTS.search_space.memory_mode
    seq: Tuple[int, ...] = NAS_COMPONENT_DEFAULTS.search_space.seq
    head_units: Tuple[int, ...] = NAS_COMPONENT_DEFAULTS.search_space.head_units
    video_decision: Tuple[str, ...] = NAS_COMPONENT_DEFAULTS.search_space.video_decision
    video_decision_input: Tuple[str, ...] = NAS_COMPONENT_DEFAULTS.search_space.video_decision_input
    cnn: Tuple[str, ...] = NAS_COMPONENT_DEFAULTS.search_space.cnn


@dataclass(frozen=True)
class NasDefaults:
    supported: NasSupportedValues = NasSupportedValues()
    controller: NasControllerConfig = NasControllerConfig()
    search_space: NasSearchSpaceDefaults = NasSearchSpaceDefaults()


NAS_DEFAULTS = NasDefaults()


@dataclass(frozen=True)
class NasSearchSpaceConfig:
    layers: Tuple[int, ...] = NAS_DEFAULTS.search_space.layers
    rnn: Tuple[str, ...] = NAS_DEFAULTS.search_space.rnn
    units_0: Tuple[int, ...] = NAS_DEFAULTS.search_space.units_0
    units_1: Tuple[int, ...] = NAS_DEFAULTS.search_space.units_1
    units_2: Tuple[int, ...] = NAS_DEFAULTS.search_space.units_2
    direction: Tuple[str, ...] = NAS_DEFAULTS.search_space.direction
    memory_mode: Tuple[str, ...] = NAS_DEFAULTS.search_space.memory_mode
    seq: Tuple[int, ...] = NAS_DEFAULTS.search_space.seq
    head_units: Tuple[int, ...] = NAS_DEFAULTS.search_space.head_units
    video_decision: Tuple[str, ...] = NAS_DEFAULTS.search_space.video_decision
    video_decision_input: Tuple[str, ...] = NAS_DEFAULTS.search_space.video_decision_input
    cnn: Tuple[str, ...] = NAS_DEFAULTS.search_space.cnn

    def __post_init__(self) -> None:
        for dimension in CANONICAL_SEARCH_DIMENSIONS:
            values = tuple(getattr(self, dimension))
            if not values:
                raise ValueError(f"La dimensión '{dimension}' debe tener al menos una opción")
            if dimension == "layers" and any(int(value) < 1 or int(value) > 3 for value in values):
                raise ValueError("layers solo admite valores entre 1 y 3")
            if dimension == "rnn":
                unsupported = [value for value in values if str(value).lower() not in RNN_DEFAULTS.supported.rnn_types]
                if unsupported:
                    raise ValueError(f"rnn contiene opciones no soportadas: {unsupported!r}")
            if dimension == "direction":
                unsupported = [value for value in values if str(value).lower() not in RNN_DEFAULTS.supported.directions]
                if unsupported:
                    raise ValueError(f"direction contiene opciones no soportadas: {unsupported!r}")
            if dimension == "memory_mode":
                normalized = tuple(normalize_memory_mode(str(value)) for value in values)
                object.__setattr__(self, dimension, normalized)
            if dimension == "video_decision":
                unsupported = [value for value in values if str(value).lower() not in RNN_DEFAULTS.supported.video_decisions]
                if unsupported:
                    raise ValueError(f"video_decision contiene opciones no soportadas: {unsupported!r}")
            if dimension == "video_decision_input":
                unsupported = [value for value in values if str(value).lower() not in RNN_DEFAULTS.supported.video_decision_inputs]
                if unsupported:
                    raise ValueError(f"video_decision_input contiene opciones no soportadas: {unsupported!r}")
            if dimension == "cnn":
                unsupported = [value for value in values if str(value) not in CNN_BACKBONES]
                if unsupported:
                    raise ValueError(f"cnn contiene opciones no soportadas: {unsupported!r}")

    @property
    def variable_dimensions(self) -> Tuple[str, ...]:
        return tuple(name for name in CANONICAL_SEARCH_DIMENSIONS if len(getattr(self, name)) > 1)

    @property
    def fixed_dimensions(self) -> Tuple[str, ...]:
        return tuple(name for name in CANONICAL_SEARCH_DIMENSIONS if len(getattr(self, name)) == 1)

    def options(self, dimension: str) -> Tuple[Any, ...]:
        return tuple(getattr(self, dimension))

    def fixed_value(self, dimension: str) -> Any:
        values = self.options(dimension)
        if len(values) != 1:
            raise ValueError(f"La dimensión '{dimension}' no está fijada a un único valor")
        return values[0]

    def to_dict(self) -> Dict[str, Tuple[Any, ...]]:
        return {dimension: self.options(dimension) for dimension in CANONICAL_SEARCH_DIMENSIONS}

def default_nas_experiment() -> "RnnExperimentConfig":
    from .rnn_config import RnnArchitectureConfig, RnnDataConfig, RnnExperimentConfig, RnnOptimizerConfig, RnnRuntimeConfig

    search_space = NasSearchSpaceConfig()
    default_cnn = str(next(iter(search_space.cnn)))
    return RnnExperimentConfig(
        operation="search",
        data=RnnDataConfig(
            cnn=default_cnn,
            name=RNN_DEFAULTS.data.dataset_name,
            frames=RNN_DEFAULTS.data.frames,
            image_size=RNN_DEFAULTS.data.image_size,
            seq=next(iter(search_space.seq)),
            split=RNN_DEFAULTS.data.split,
            val_fraction=RNN_DEFAULTS.data.val_fraction,
            partition_mode=RNN_DEFAULTS.data.partition_mode,
            sampling=RNN_DEFAULTS.data.sampling,
            resize_mode=RNN_DEFAULTS.data.resize_mode,
            cnn_training_signature=RNN_DEFAULTS.data.cnn_training_signature,
            cnn_feature_export_signature=RNN_DEFAULTS.data.cnn_feature_export_signature,
        ),
        architecture=RnnArchitectureConfig(
            rnn=next(iter(search_space.rnn)),
            direction=next(iter(search_space.direction)),
            units=(
                int(next(iter(search_space.units_0))),
                int(next(iter(search_space.units_1))),
                int(next(iter(search_space.units_2))),
            ),
            memory_mode=next(iter(search_space.memory_mode)),
            head_units=int(next(iter(search_space.head_units))),
            video_decision=next(iter(search_space.video_decision)),
            video_decision_input=next(iter(search_space.video_decision_input)),
        ),
        runtime=RnnRuntimeConfig(),
        optimizer=RnnOptimizerConfig(),
        nas=NasControllerConfig(),
        search_space=search_space,
    )



__all__ = [
    "NAS_DEFAULTS",
    "NasDefaults",
    "CANONICAL_SEARCH_DIMENSIONS",
    "NasControllerModelConfig",
    "NasControllerOptimizerConfig",
    "NasControllerSchedulerConfig",
    "NasControllerTrainingConfig",
    "NasControllerConfig",
    "NasSearchSpaceConfig",
    "default_nas_experiment",
]
