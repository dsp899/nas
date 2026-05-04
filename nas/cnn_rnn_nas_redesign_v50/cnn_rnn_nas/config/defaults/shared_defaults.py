from dataclasses import dataclass

from ..specs.data import DatasetSpec, PartitionSpec
from ..specs.preprocess import AugmentationSpec
from ..specs.runtime import OptimizerSpec, RuntimeSpec, SchedulerSpec


@dataclass(frozen=True)
class SharedDatasetDefaults:
    name: str = "pmi50"
    split: str = "split01"


@dataclass(frozen=True)
class SharedPartitionDefaults:
    mode: str = "train_test"
    val_fraction: float = 0.30


@dataclass(frozen=True)
class SharedPreprocessDefaults:
    image_size: int = 224
    sampling: str = "uniform"
    resize_mode: str = "pad"
    train_frames: int = 15
    predict_frames: int = 36
    shuffle_buffer_videos: int = 1024
    shuffle_buffer_frames: int = 5000


DEFAULT_DATASET_SPEC = DatasetSpec(name=SharedDatasetDefaults().name, split=SharedDatasetDefaults().split)
DEFAULT_PARTITION_SPEC = PartitionSpec(mode=SharedPartitionDefaults().mode, val_fraction=SharedPartitionDefaults().val_fraction)
DEFAULT_RUNTIME_SPEC = RuntimeSpec(gpu="1", mixed_precision=True, random_seed=1337, project_root=".")
DEFAULT_OPTIMIZER_SPEC = OptimizerSpec(name="adagrad", learning_rate=0.001, momentum=0.9, nesterov=True, weight_decay=0.0)
DEFAULT_SCHEDULER_SPEC = SchedulerSpec(reduce_lr_on_plateau=False, reduce_lr_factor=0.5, reduce_lr_patience=2, min_learning_rate=1e-6)
DEFAULT_AUGMENTATION_SPEC = AugmentationSpec()


@dataclass(frozen=True)
class SharedDefaults:
    dataset: SharedDatasetDefaults = SharedDatasetDefaults()
    partition: SharedPartitionDefaults = SharedPartitionDefaults()
    preprocess: SharedPreprocessDefaults = SharedPreprocessDefaults()
    augmentation: AugmentationSpec = DEFAULT_AUGMENTATION_SPEC
    runtime: RuntimeSpec = DEFAULT_RUNTIME_SPEC
    optimizer: OptimizerSpec = DEFAULT_OPTIMIZER_SPEC
    scheduler: SchedulerSpec = DEFAULT_SCHEDULER_SPEC


SHARED_DEFAULTS = SharedDefaults()
