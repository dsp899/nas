from dataclasses import dataclass

from .shared_defaults import SHARED_DEFAULTS
from ..specs.modeling import ClassificationHeadSpec, CnnExtractorSpec
from ..specs.runtime import OptimizerSpec, SchedulerSpec


DEFAULT_CNN_EXTRACTOR_SPEC = CnnExtractorSpec(backbone="resnet50", weights="imagenet", trainable=True, feature_dim=512)
DEFAULT_CNN_HEAD_SPEC = ClassificationHeadSpec(hidden_units=(1024,), dropouts=(0.5,))
DEFAULT_CNN_TRAINING_OPTIMIZER_SPEC = OptimizerSpec(
    name=SHARED_DEFAULTS.optimizer.name,
    learning_rate=SHARED_DEFAULTS.optimizer.learning_rate,
    momentum=SHARED_DEFAULTS.optimizer.momentum,
    nesterov=SHARED_DEFAULTS.optimizer.nesterov,
    weight_decay=SHARED_DEFAULTS.optimizer.weight_decay,
    clipnorm=SHARED_DEFAULTS.optimizer.clipnorm,
)
DEFAULT_CNN_TRAINING_SCHEDULER_SPEC = SchedulerSpec(
    reduce_lr_on_plateau=SHARED_DEFAULTS.scheduler.reduce_lr_on_plateau,
    reduce_lr_factor=SHARED_DEFAULTS.scheduler.reduce_lr_factor,
    reduce_lr_patience=SHARED_DEFAULTS.scheduler.reduce_lr_patience,
    min_learning_rate=SHARED_DEFAULTS.scheduler.min_learning_rate,
)


@dataclass(frozen=True)
class CnnTrainingDefaults:
    epochs: int = 10
    batch_size: int = 16
    feature_batch_size: int = 1
    optimizer_spec: OptimizerSpec = DEFAULT_CNN_TRAINING_OPTIMIZER_SPEC
    scheduler_spec: SchedulerSpec = DEFAULT_CNN_TRAINING_SCHEDULER_SPEC
    early_stopping_patience: int = 5

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
class CnnDefaults:
    extractor: CnnExtractorSpec = DEFAULT_CNN_EXTRACTOR_SPEC
    head: ClassificationHeadSpec = DEFAULT_CNN_HEAD_SPEC
    training: CnnTrainingDefaults = CnnTrainingDefaults()


CNN_COMPONENT_DEFAULTS = CnnDefaults()
