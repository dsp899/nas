from dataclasses import asdict, dataclass, replace
from typing import Any, Dict, Optional, Tuple

from .defaults.shared_defaults import SHARED_DEFAULTS
from .defaults.rnn_defaults import RNN_COMPONENT_DEFAULTS
from .specs.data import CnnFeatureSourceRef, DatasetSpec, PartitionSpec
from .specs.modeling import (
    ClassificationHeadSpec,
    RnnEncoderSpec,
    RnnModelSpec,
    SequenceSpec,
    normalize_memory_mode,
)
from .specs.preprocess import FrameSourceSpec
from .specs.runtime import OptimizerSpec, RuntimeSpec, SchedulerSpec
from .supported.cnn_supported import CNN_BACKBONES
from .supported.shared_supported import OPTIMIZER_NAMES, PARTITION_MODES, RESIZE_MODES, SAMPLING_MODES
from .supported.rnn_supported import (
    RNN_DIRECTIONS,
    RNN_MEMORY_MODES,
    RNN_OPERATIONS,
    RNN_TEST_STRATEGIES,
    RNN_TYPES,
    RNN_VIDEO_DECISIONS,
    RNN_VIDEO_DECISION_INPUTS,
)


@dataclass(frozen=True)
class RnnSupportedValues:
    operations: Tuple[str, ...] = RNN_OPERATIONS
    rnn_types: Tuple[str, ...] = RNN_TYPES
    directions: Tuple[str, ...] = RNN_DIRECTIONS
    memory_modes: Tuple[str, ...] = RNN_MEMORY_MODES
    video_decisions: Tuple[str, ...] = RNN_VIDEO_DECISIONS
    video_decision_inputs: Tuple[str, ...] = RNN_VIDEO_DECISION_INPUTS
    test_strategies: Tuple[str, ...] = RNN_TEST_STRATEGIES
    backbones: Tuple[str, ...] = CNN_BACKBONES
    resize_modes: Tuple[str, ...] = RESIZE_MODES
    sampling_modes: Tuple[str, ...] = SAMPLING_MODES
    partition_modes: Tuple[str, ...] = PARTITION_MODES
    optimizers: Tuple[str, ...] = OPTIMIZER_NAMES


@dataclass(frozen=True)
class RnnDataDefaults:
    dataset_name: str = SHARED_DEFAULTS.dataset.name
    split: str = SHARED_DEFAULTS.dataset.split
    partition_mode: str = SHARED_DEFAULTS.partition.mode
    val_fraction: float = SHARED_DEFAULTS.partition.val_fraction
    frames: int = RNN_COMPONENT_DEFAULTS.data.frames
    image_size: int = RNN_COMPONENT_DEFAULTS.data.image_size
    backbone: str = RNN_COMPONENT_DEFAULTS.data.backbone
    sampling: str = RNN_COMPONENT_DEFAULTS.data.sampling
    resize_mode: str = RNN_COMPONENT_DEFAULTS.data.resize_mode
    cnn_training_signature: str = RNN_COMPONENT_DEFAULTS.data.cnn_training_signature
    cnn_feature_export_signature: str = RNN_COMPONENT_DEFAULTS.data.cnn_feature_export_signature


@dataclass(frozen=True)
class RnnArchitectureDefaults:
    rnn_type: str = RNN_COMPONENT_DEFAULTS.encoder.rnn
    direction: str = RNN_COMPONENT_DEFAULTS.encoder.direction
    units: Tuple[int, int, int] = RNN_COMPONENT_DEFAULTS.encoder.units
    memory_mode: str = RNN_COMPONENT_DEFAULTS.encoder.memory_mode
    seq_length: int = RNN_COMPONENT_DEFAULTS.encoder.seq_length
    head_units: int = RNN_COMPONENT_DEFAULTS.head.first_hidden_units
    video_decision: str = "average"
    video_decision_input: str = "clip_logits"


@dataclass(frozen=True)
class RnnOptimizerDefaults:
    spec: OptimizerSpec = RNN_COMPONENT_DEFAULTS.training.optimizer_spec

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def learning_rate(self) -> float:
        return self.spec.learning_rate

    @property
    def momentum(self) -> float:
        return self.spec.momentum

    @property
    def nesterov(self) -> bool:
        return self.spec.nesterov

    @property
    def weight_decay(self) -> float:
        return self.spec.weight_decay


@dataclass(frozen=True)
class RnnRuntimeDefaults:
    epochs: int = RNN_COMPONENT_DEFAULTS.training.epochs
    batch_size: int = RNN_COMPONENT_DEFAULTS.training.batch_size
    gpu: str = SHARED_DEFAULTS.runtime.gpu
    mixed_precision: bool = SHARED_DEFAULTS.runtime.mixed_precision
    random_seed: int = SHARED_DEFAULTS.runtime.random_seed
    project_root: str = SHARED_DEFAULTS.runtime.project_root
    test_strategy: str = RNN_COMPONENT_DEFAULTS.training.test_strategy
    allow_epoch_extension_resume: bool = RNN_COMPONENT_DEFAULTS.training.allow_epoch_extension_resume
    scheduler_spec: SchedulerSpec = RNN_COMPONENT_DEFAULTS.training.scheduler_spec

    @property
    def learning_rate(self) -> float:
        return RNN_COMPONENT_DEFAULTS.training.optimizer_spec.learning_rate

    @property
    def reduce_lr_on_plateau(self) -> bool:
        return self.scheduler_spec.reduce_lr_on_plateau

    @property
    def reduce_lr_factor(self) -> float:
        return self.scheduler_spec.reduce_lr_factor

    @property
    def reduce_lr_patience(self) -> int:
        return self.scheduler_spec.reduce_lr_patience

    @property
    def min_learning_rate(self) -> float:
        return self.scheduler_spec.min_learning_rate


@dataclass(frozen=True)
class RnnInternalDefaults:
    dense_dropout: float = RNN_COMPONENT_DEFAULTS.internal.dense_dropout
    recurrent_dropout: float = RNN_COMPONENT_DEFAULTS.internal.recurrent_dropout
    l2_reg: float = RNN_COMPONENT_DEFAULTS.internal.l2_reg
    surrogate_max_prob_temperature: float = RNN_COMPONENT_DEFAULTS.internal.surrogate_max_prob_temperature


@dataclass(frozen=True)
class RnnDefaults:
    supported: RnnSupportedValues = RnnSupportedValues()
    data: RnnDataDefaults = RnnDataDefaults()
    architecture: RnnArchitectureDefaults = RnnArchitectureDefaults()
    optimizer: RnnOptimizerDefaults = RnnOptimizerDefaults()
    runtime: RnnRuntimeDefaults = RnnRuntimeDefaults()
    internal: RnnInternalDefaults = RnnInternalDefaults()


RNN_DEFAULTS = RnnDefaults()


@dataclass(frozen=True)
class RnnDataConfig:
    cnn: str
    name: str
    frames: int
    image_size: int
    seq: int
    split: str = RNN_DEFAULTS.data.split
    val_fraction: float = RNN_DEFAULTS.data.val_fraction
    partition_mode: str = RNN_DEFAULTS.data.partition_mode
    sampling: str = RNN_DEFAULTS.data.sampling
    resize_mode: str = RNN_DEFAULTS.data.resize_mode
    cnn_training_signature: str = RNN_DEFAULTS.data.cnn_training_signature
    cnn_feature_export_signature: str = RNN_DEFAULTS.data.cnn_feature_export_signature

    def __post_init__(self) -> None:
        if self.cnn not in CNN_BACKBONES:
            raise ValueError(f"cnn no soportada: {self.cnn!r}")
        if self.sampling not in SAMPLING_MODES:
            raise ValueError(f"sampling no soportado: {self.sampling!r}")
        if self.resize_mode not in RESIZE_MODES:
            raise ValueError(f"resize_mode no soportado: {self.resize_mode!r}")
        partition_mode = self.partition_mode.strip().lower()
        if partition_mode not in PARTITION_MODES:
            raise ValueError(f"partition_mode no soportado: {self.partition_mode!r}")
        if not 0.0 <= float(self.val_fraction) < 0.5:
            raise ValueError("val_fraction debe estar en [0, 0.5)")
        if partition_mode == "train_val_test" and float(self.val_fraction) <= 0.0:
            raise ValueError("train_val_test requiere val_fraction > 0")
        object.__setattr__(self, "partition_mode", partition_mode)

    @property
    def size(self) -> int:
        return self.image_size

    @property
    def feature_sampling(self) -> str:
        return self.sampling

    @property
    def dataset_spec(self) -> DatasetSpec:
        return DatasetSpec(name=self.name, split=self.split)

    @property
    def partition_spec(self) -> PartitionSpec:
        return PartitionSpec(mode=self.partition_mode, val_fraction=self.val_fraction)

    @property
    def preprocess_spec(self) -> FrameSourceSpec:
        return FrameSourceSpec(image_size=self.image_size, frames=self.frames, sampling=self.sampling, resize_mode=self.resize_mode)

    @property
    def feature_source(self) -> CnnFeatureSourceRef:
        return CnnFeatureSourceRef(cnn_training_signature=self.cnn_training_signature, cnn_feature_export_signature=self.cnn_feature_export_signature)

    @property
    def sequence_spec(self) -> SequenceSpec:
        return SequenceSpec(seq_length=self.seq)

    @property
    def partition_tag(self) -> str:
        return f"dataset_{self.name}_{self.split}_{self.partition_mode}"

    @property
    def feature_spec_tag(self) -> str:
        return self.preprocess_spec.tag

    @property
    def sequence_spec_tag(self) -> str:
        return self.sequence_spec.tag


@dataclass(frozen=True)
class RnnArchitectureConfig:
    rnn: str
    direction: str
    units: Tuple[int, int, int]
    memory_mode: str = RNN_DEFAULTS.architecture.memory_mode
    head_units: int = RNN_DEFAULTS.architecture.head_units
    video_decision: str = RNN_DEFAULTS.architecture.video_decision
    video_decision_input: str = RNN_DEFAULTS.architecture.video_decision_input

    def __post_init__(self) -> None:
        normalize_memory_mode(self.memory_mode)
        model = self.model_spec
        object.__setattr__(self, "rnn", model.encoder.rnn)
        object.__setattr__(self, "direction", model.encoder.direction)
        object.__setattr__(self, "memory_mode", model.encoder.memory_mode)
        object.__setattr__(self, "video_decision", model.video_decision)
        object.__setattr__(self, "video_decision_input", model.video_decision_input)

    @property
    def encoder(self) -> RnnEncoderSpec:
        return RnnEncoderSpec(
            rnn=self.rnn,
            direction=self.direction,
            units=self.units,
            memory_mode=self.memory_mode,
            seq_length=RNN_DEFAULTS.architecture.seq_length,
        )

    @property
    def head(self) -> ClassificationHeadSpec:
        return ClassificationHeadSpec(hidden_units=(int(self.head_units),), dropouts=(RNN_DEFAULTS.internal.dense_dropout,))

    @property
    def model_spec(self) -> RnnModelSpec:
        return RnnModelSpec(encoder=self.encoder, head=self.head, video_decision=self.video_decision, video_decision_input=self.video_decision_input)

    @property
    def active_units(self) -> Tuple[int, ...]:
        return self.model_spec.encoder.active_units

    @property
    def encoder_output_dim(self) -> int:
        return self.model_spec.encoder.encoder_output_dim

    @property
    def tag(self) -> str:
        return self.model_spec.tag


@dataclass(frozen=True)
class RnnOptimizerConfig:
    name: str = RNN_DEFAULTS.optimizer.name
    momentum: float = RNN_DEFAULTS.optimizer.momentum
    nesterov: bool = RNN_DEFAULTS.optimizer.nesterov
    weight_decay: float = RNN_DEFAULTS.optimizer.weight_decay

    def __post_init__(self) -> None:
        name = self.name.strip().lower()
        if name not in OPTIMIZER_NAMES:
            raise ValueError(f"Optimizador RNN no soportado: {self.name!r}")
        object.__setattr__(self, "name", name)

    @property
    def base_spec(self) -> OptimizerSpec:
        return OptimizerSpec(name=self.name, learning_rate=RNN_DEFAULTS.runtime.learning_rate, momentum=self.momentum, nesterov=self.nesterov, weight_decay=self.weight_decay)


@dataclass(frozen=True)
class RnnRuntimeConfig:
    epochs: int = RNN_DEFAULTS.runtime.epochs
    batch_size: int = RNN_DEFAULTS.runtime.batch_size
    learning_rate: float = RNN_DEFAULTS.runtime.learning_rate
    gpu: str = RNN_DEFAULTS.runtime.gpu
    mixed_precision: bool = RNN_DEFAULTS.runtime.mixed_precision
    random_seed: int = RNN_DEFAULTS.runtime.random_seed
    project_root: str = RNN_DEFAULTS.runtime.project_root
    test_strategy: str = RNN_DEFAULTS.runtime.test_strategy
    allow_epoch_extension_resume: bool = RNN_DEFAULTS.runtime.allow_epoch_extension_resume
    reduce_lr_on_plateau: bool = RNN_DEFAULTS.runtime.reduce_lr_on_plateau
    reduce_lr_factor: float = RNN_DEFAULTS.runtime.reduce_lr_factor
    reduce_lr_patience: int = RNN_DEFAULTS.runtime.reduce_lr_patience
    min_learning_rate: float = RNN_DEFAULTS.runtime.min_learning_rate

    def __post_init__(self) -> None:
        strategy = self.test_strategy.strip().lower()
        if strategy not in RNN_TEST_STRATEGIES:
            raise ValueError(f"test_strategy no soportado: {self.test_strategy!r}")
        object.__setattr__(self, "test_strategy", strategy)

    @property
    def eval_strategy(self) -> str:
        return self.test_strategy

    @property
    def scheduler_spec(self) -> SchedulerSpec:
        return SchedulerSpec(
            reduce_lr_on_plateau=self.reduce_lr_on_plateau,
            reduce_lr_factor=self.reduce_lr_factor,
            reduce_lr_patience=self.reduce_lr_patience,
            min_learning_rate=self.min_learning_rate,
        )

    @property
    def runtime_spec(self) -> RuntimeSpec:
        return RuntimeSpec(gpu=self.gpu, mixed_precision=self.mixed_precision, random_seed=self.random_seed, project_root=self.project_root)


@dataclass(frozen=True)
class RnnExperimentConfig:
    operation: str
    data: RnnDataConfig
    architecture: RnnArchitectureConfig
    runtime: RnnRuntimeConfig
    optimizer: RnnOptimizerConfig = RnnOptimizerConfig()
    nas: Optional[Any] = None
    search_space: Optional[Any] = None

    @property
    def dataset(self) -> DatasetSpec:
        return self.data.dataset_spec

    @property
    def partition(self) -> PartitionSpec:
        return self.data.partition_spec

    @property
    def preprocess(self) -> FrameSourceSpec:
        return self.data.preprocess_spec

    @property
    def feature_source(self) -> CnnFeatureSourceRef:
        return self.data.feature_source

    @property
    def model(self) -> RnnModelSpec:
        encoder = RnnEncoderSpec(
            rnn=self.architecture.rnn,
            direction=self.architecture.direction,
            units=self.architecture.units,
            memory_mode=self.architecture.memory_mode,
            seq_length=self.data.seq,
        )
        head = ClassificationHeadSpec(hidden_units=(int(self.architecture.head_units),), dropouts=(RNN_DEFAULTS.internal.dense_dropout,))
        return RnnModelSpec(
            encoder=encoder,
            head=head,
            video_decision=self.architecture.video_decision,
            video_decision_input=self.architecture.video_decision_input,
        )

    @property
    def training(self) -> Dict[str, Any]:
        optimizer_payload = asdict(self.optimizer)
        optimizer_payload["learning_rate"] = self.runtime.learning_rate
        return {
            "epochs": self.runtime.epochs,
            "batch_size": self.runtime.batch_size,
            "optimizer": optimizer_payload,
            "scheduler": asdict(self.runtime.scheduler_spec),
            "test_strategy": self.runtime.test_strategy,
            "allow_epoch_extension_resume": self.runtime.allow_epoch_extension_resume,
        }

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["dataset"] = asdict(self.dataset)
        payload["partition"] = asdict(self.partition)
        payload["preprocess"] = asdict(self.preprocess)
        payload["feature_source"] = {
            "cnn": self.data.cnn,
            "cnn_training_signature": self.feature_source.cnn_training_signature,
            "cnn_feature_export_signature": self.feature_source.cnn_feature_export_signature,
        }
        payload["model"] = {
            "encoder": asdict(self.model.encoder),
            "head": asdict(self.model.head),
            "decision": {
                "video_decision": self.model.video_decision,
                "video_decision_input": self.model.video_decision_input,
            },
        }
        payload["training"] = self.training
        return payload

    @property
    def rnn_model_spec_tag(self) -> str:
        return f"{self.data.feature_spec_tag}_{self.data.sequence_spec_tag}_{self.architecture.tag}"

    def for_architecture(self, *, architecture: Optional[RnnArchitectureConfig] = None, data: Optional[RnnDataConfig] = None, operation: Optional[str] = None, runtime: Optional[RnnRuntimeConfig] = None, optimizer: Optional[RnnOptimizerConfig] = None) -> "RnnExperimentConfig":
        return replace(self, architecture=architecture or self.architecture, data=data or self.data, operation=operation or self.operation, runtime=runtime or self.runtime, optimizer=optimizer or self.optimizer)

def default_rnn_experiment(operation: str = "train") -> RnnExperimentConfig:
    return RnnExperimentConfig(
        operation=operation,
        data=RnnDataConfig(
            cnn=RNN_DEFAULTS.data.backbone,
            name=RNN_DEFAULTS.data.dataset_name,
            frames=RNN_DEFAULTS.data.frames,
            image_size=RNN_DEFAULTS.data.image_size,
            seq=RNN_DEFAULTS.architecture.seq_length,
            split=RNN_DEFAULTS.data.split,
            val_fraction=RNN_DEFAULTS.data.val_fraction,
            partition_mode=RNN_DEFAULTS.data.partition_mode,
            sampling=RNN_DEFAULTS.data.sampling,
            resize_mode=RNN_DEFAULTS.data.resize_mode,
            cnn_training_signature=RNN_DEFAULTS.data.cnn_training_signature,
            cnn_feature_export_signature=RNN_DEFAULTS.data.cnn_feature_export_signature,
        ),
        architecture=RnnArchitectureConfig(
            rnn=RNN_DEFAULTS.architecture.rnn_type,
            direction=RNN_DEFAULTS.architecture.direction,
            units=RNN_DEFAULTS.architecture.units,
            memory_mode=RNN_DEFAULTS.architecture.memory_mode,
            head_units=RNN_DEFAULTS.architecture.head_units,
            video_decision=RNN_DEFAULTS.architecture.video_decision,
            video_decision_input=RNN_DEFAULTS.architecture.video_decision_input,
        ),
        runtime=RnnRuntimeConfig(),
        optimizer=RnnOptimizerConfig(),
    )



__all__ = [
    "RnnDefaults",
    "RNN_DEFAULTS",
    "RnnDataConfig",
    "RnnArchitectureConfig",
    "RnnOptimizerConfig",
    "RnnRuntimeConfig",
    "RnnExperimentConfig",
    "default_rnn_experiment",
    "normalize_memory_mode",
]
