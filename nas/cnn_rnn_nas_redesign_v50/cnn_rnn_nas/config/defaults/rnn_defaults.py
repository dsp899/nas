from dataclasses import dataclass

from .shared_defaults import SHARED_DEFAULTS
from ..specs.modeling import ClassificationHeadSpec, RnnEncoderSpec
from ..specs.runtime import OptimizerSpec, SchedulerSpec


DEFAULT_RNN_ENCODER_SPEC = RnnEncoderSpec(rnn="gru", direction="unidirectional", units=(32, 0, 0), memory_mode="none", seq_length=6)
DEFAULT_RNN_HEAD_SPEC = ClassificationHeadSpec(hidden_units=(256,), dropouts=(0.5,))
DEFAULT_RNN_TRAINING_OPTIMIZER_SPEC = OptimizerSpec(
    name=SHARED_DEFAULTS.optimizer.name,
    learning_rate=SHARED_DEFAULTS.optimizer.learning_rate,
    momentum=SHARED_DEFAULTS.optimizer.momentum,
    nesterov=SHARED_DEFAULTS.optimizer.nesterov,
    weight_decay=SHARED_DEFAULTS.optimizer.weight_decay,
    clipnorm=SHARED_DEFAULTS.optimizer.clipnorm,
)
DEFAULT_RNN_TRAINING_SCHEDULER_SPEC = SchedulerSpec(
    reduce_lr_on_plateau=SHARED_DEFAULTS.scheduler.reduce_lr_on_plateau,
    reduce_lr_factor=SHARED_DEFAULTS.scheduler.reduce_lr_factor,
    reduce_lr_patience=SHARED_DEFAULTS.scheduler.reduce_lr_patience,
    min_learning_rate=SHARED_DEFAULTS.scheduler.min_learning_rate,
)


@dataclass(frozen=True)
class RnnDataDefaults:
    backbone: str = "inceptionV3"
    frames: int = 36
    image_size: int = 224
    sampling: str = "uniform"
    resize_mode: str = "pad"
    cnn_training_signature: str = ""
    cnn_feature_export_signature: str = ""


@dataclass(frozen=True)
class RnnTrainingDefaults:
    epochs: int = 10
    batch_size: int = 32
    optimizer_spec: OptimizerSpec = DEFAULT_RNN_TRAINING_OPTIMIZER_SPEC
    scheduler_spec: SchedulerSpec = DEFAULT_RNN_TRAINING_SCHEDULER_SPEC
    test_strategy: str = "average"
    allow_epoch_extension_resume: bool = False

    @property
    def learning_rate(self) -> float:
        return self.optimizer_spec.learning_rate

    @property
    def optimizer(self) -> str:
        return self.optimizer_spec.name

    @property
    def momentum(self) -> float:
        return self.optimizer_spec.momentum

    @property
    def nesterov(self) -> bool:
        return self.optimizer_spec.nesterov

    @property
    def weight_decay(self) -> float:
        return self.optimizer_spec.weight_decay

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
    dense_dropout: float = 0.5
    recurrent_dropout: float = 0.0
    l2_reg: float = 0.01
    surrogate_max_prob_temperature: float = 0.35


@dataclass(frozen=True)
class RnnDefaults:
    data: RnnDataDefaults = RnnDataDefaults()
    encoder: RnnEncoderSpec = DEFAULT_RNN_ENCODER_SPEC
    head: ClassificationHeadSpec = DEFAULT_RNN_HEAD_SPEC
    training: RnnTrainingDefaults = RnnTrainingDefaults()
    internal: RnnInternalDefaults = RnnInternalDefaults()


RNN_COMPONENT_DEFAULTS = RnnDefaults()
