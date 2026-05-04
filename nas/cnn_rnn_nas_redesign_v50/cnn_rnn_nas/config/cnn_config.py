from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

from .defaults.shared_defaults import SHARED_DEFAULTS
from .defaults.cnn_defaults import CNN_COMPONENT_DEFAULTS
from .specs.data import DatasetSpec, PartitionSpec, normalize_partition_mode
from .specs.modeling import ClassificationHeadSpec, CnnExtractorSpec, CnnModelSpec
from .specs.preprocess import AugmentationSpec, PreprocessSpec
from .specs.runtime import OptimizerSpec, RuntimeSpec, SchedulerSpec
from .supported.cnn_supported import CNN_BACKBONES, CNN_OPERATIONS
from .supported.shared_supported import OPTIMIZER_NAMES, PARTITION_MODES, RESIZE_MODES, SAMPLING_MODES


@dataclass(frozen=True)
class CnnSupportedValues:
    operations: Tuple[str, ...] = CNN_OPERATIONS
    backbones: Tuple[str, ...] = CNN_BACKBONES
    partition_modes: Tuple[str, ...] = PARTITION_MODES
    resize_modes: Tuple[str, ...] = RESIZE_MODES
    sampling_modes: Tuple[str, ...] = SAMPLING_MODES
    optimizers: Tuple[str, ...] = OPTIMIZER_NAMES


@dataclass(frozen=True)
class CnnDatasetDefaults:
    name: str = SHARED_DEFAULTS.dataset.name
    split: str = SHARED_DEFAULTS.dataset.split
    partition_mode: str = SHARED_DEFAULTS.partition.mode
    val_fraction: float = SHARED_DEFAULTS.partition.val_fraction


@dataclass(frozen=True)
class CnnPreprocessDefaults:
    image_size: int = SHARED_DEFAULTS.preprocess.image_size
    train_frames: int = SHARED_DEFAULTS.preprocess.train_frames
    predict_frames: int = SHARED_DEFAULTS.preprocess.predict_frames
    sampling: str = SHARED_DEFAULTS.preprocess.sampling
    resize_mode: str = SHARED_DEFAULTS.preprocess.resize_mode
    shuffle_buffer_videos: int = SHARED_DEFAULTS.preprocess.shuffle_buffer_videos
    shuffle_buffer_frames: int = SHARED_DEFAULTS.preprocess.shuffle_buffer_frames
    augmentation_enabled: bool = SHARED_DEFAULTS.augmentation.enabled
    random_flip: bool = SHARED_DEFAULTS.augmentation.random_flip
    random_crop_scale_min: float = SHARED_DEFAULTS.augmentation.random_crop_scale_min
    brightness_delta: float = SHARED_DEFAULTS.augmentation.brightness_delta
    contrast_lower: float = SHARED_DEFAULTS.augmentation.contrast_lower
    contrast_upper: float = SHARED_DEFAULTS.augmentation.contrast_upper
    saturation_lower: float = SHARED_DEFAULTS.augmentation.saturation_lower
    saturation_upper: float = SHARED_DEFAULTS.augmentation.saturation_upper


@dataclass(frozen=True)
class CnnExtractorDefaults:
    backbone: str = CNN_COMPONENT_DEFAULTS.extractor.backbone
    weights: Optional[str] = CNN_COMPONENT_DEFAULTS.extractor.weights
    trainable: bool = CNN_COMPONENT_DEFAULTS.extractor.trainable
    feature_dim: int = CNN_COMPONENT_DEFAULTS.extractor.feature_dim


@dataclass(frozen=True)
class CnnHeadDefaults:
    hidden_units: Tuple[int, ...] = CNN_COMPONENT_DEFAULTS.head.hidden_units
    dropouts: Tuple[float, ...] = CNN_COMPONENT_DEFAULTS.head.dropouts


@dataclass(frozen=True)
class CnnTrainingDefaults:
    epochs: int = CNN_COMPONENT_DEFAULTS.training.epochs
    batch_size: int = CNN_COMPONENT_DEFAULTS.training.batch_size
    feature_batch_size: int = CNN_COMPONENT_DEFAULTS.training.feature_batch_size
    optimizer_spec: OptimizerSpec = CNN_COMPONENT_DEFAULTS.training.optimizer_spec
    scheduler_spec: SchedulerSpec = CNN_COMPONENT_DEFAULTS.training.scheduler_spec
    early_stopping_patience: int = CNN_COMPONENT_DEFAULTS.training.early_stopping_patience

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
class CnnRuntimeDefaults:
    gpu: str = SHARED_DEFAULTS.runtime.gpu
    mixed_precision: bool = SHARED_DEFAULTS.runtime.mixed_precision
    random_seed: int = SHARED_DEFAULTS.runtime.random_seed
    project_root: str = SHARED_DEFAULTS.runtime.project_root


@dataclass(frozen=True)
class CnnDefaults:
    supported: CnnSupportedValues = CnnSupportedValues()
    dataset: CnnDatasetDefaults = CnnDatasetDefaults()
    preprocess: CnnPreprocessDefaults = CnnPreprocessDefaults()
    extractor: CnnExtractorDefaults = CnnExtractorDefaults()
    head: CnnHeadDefaults = CnnHeadDefaults()
    training: CnnTrainingDefaults = CnnTrainingDefaults()
    runtime: CnnRuntimeDefaults = CnnRuntimeDefaults()


CNN_DEFAULTS = CnnDefaults()


@dataclass(frozen=True)
class CnnDatasetSpec:
    name: str = CNN_DEFAULTS.dataset.name
    split: str = CNN_DEFAULTS.dataset.split
    partition_mode: str = CNN_DEFAULTS.dataset.partition_mode
    val_fraction: float = CNN_DEFAULTS.dataset.val_fraction

    def __post_init__(self) -> None:
        normalized = normalize_partition_mode(self.partition_mode)
        if not 0.0 <= float(self.val_fraction) < 0.5:
            raise ValueError("val_fraction debe estar en [0, 0.5)")
        if normalized == "train_val_test" and float(self.val_fraction) <= 0.0:
            raise ValueError("train_val_test requiere val_fraction > 0")
        object.__setattr__(self, "partition_mode", normalized)

    @property
    def dataset_spec(self) -> DatasetSpec:
        return DatasetSpec(name=self.name, split=self.split)

    @property
    def partition_spec(self) -> PartitionSpec:
        return PartitionSpec(mode=self.partition_mode, val_fraction=self.val_fraction)


@dataclass(frozen=True)
class CnnPreprocessSpec:
    image_size: int = CNN_DEFAULTS.preprocess.image_size
    train_frames: int = CNN_DEFAULTS.preprocess.train_frames
    predict_frames: int = CNN_DEFAULTS.preprocess.predict_frames
    sampling: str = CNN_DEFAULTS.preprocess.sampling
    resize_mode: str = CNN_DEFAULTS.preprocess.resize_mode
    shuffle_buffer_videos: int = CNN_DEFAULTS.preprocess.shuffle_buffer_videos
    shuffle_buffer_frames: int = CNN_DEFAULTS.preprocess.shuffle_buffer_frames
    augmentation_enabled: bool = CNN_DEFAULTS.preprocess.augmentation_enabled
    random_flip: bool = CNN_DEFAULTS.preprocess.random_flip
    random_crop_scale_min: float = CNN_DEFAULTS.preprocess.random_crop_scale_min
    brightness_delta: float = CNN_DEFAULTS.preprocess.brightness_delta
    contrast_lower: float = CNN_DEFAULTS.preprocess.contrast_lower
    contrast_upper: float = CNN_DEFAULTS.preprocess.contrast_upper
    saturation_lower: float = CNN_DEFAULTS.preprocess.saturation_lower
    saturation_upper: float = CNN_DEFAULTS.preprocess.saturation_upper

    def __post_init__(self) -> None:
        if self.sampling not in SAMPLING_MODES:
            raise ValueError(f"sampling no soportado: {self.sampling!r}")
        if self.resize_mode not in RESIZE_MODES:
            raise ValueError(f"resize_mode no soportado: {self.resize_mode!r}")
        if int(self.train_frames) <= 0 or int(self.predict_frames) <= 0:
            raise ValueError("train_frames y predict_frames deben ser > 0")
        if not 0.0 <= float(self.random_crop_scale_min) <= 1.0:
            raise ValueError("random_crop_scale_min debe estar en [0, 1]")
        if not 0.0 <= float(self.brightness_delta) <= 1.0:
            raise ValueError("brightness_delta debe estar en [0, 1]")

    @property
    def preprocess_spec(self) -> PreprocessSpec:
        return PreprocessSpec(
            image_size=self.image_size,
            train_frames=self.train_frames,
            predict_frames=self.predict_frames,
            sampling=self.sampling,
            resize_mode=self.resize_mode,
            shuffle_buffer_videos=self.shuffle_buffer_videos,
            shuffle_buffer_frames=self.shuffle_buffer_frames,
        )

    @property
    def augmentation(self) -> AugmentationSpec:
        return AugmentationSpec(
            enabled=self.augmentation_enabled,
            random_flip=self.random_flip,
            random_crop_scale_min=self.random_crop_scale_min,
            brightness_delta=self.brightness_delta,
            contrast_lower=self.contrast_lower,
            contrast_upper=self.contrast_upper,
            saturation_lower=self.saturation_lower,
            saturation_upper=self.saturation_upper,
        )

    @property
    def train_sampling(self) -> str:
        return self.sampling

    @property
    def test_sampling(self) -> str:
        return self.sampling

    @property
    def predict_sampling(self) -> str:
        return self.sampling

    @property
    def eval_sampling(self) -> str:
        return self.sampling

    @property
    def train_preprocess_tag(self) -> str:
        return self.preprocess_spec.train_tag

    @property
    def predict_preprocess_tag(self) -> str:
        return self.preprocess_spec.predict_tag


CnnExtractorSpec = CnnExtractorSpec
CnnHeadSpec = ClassificationHeadSpec


@dataclass(frozen=True)
class CnnTrainingSpec:
    epochs: int = CNN_DEFAULTS.training.epochs
    batch_size: int = CNN_DEFAULTS.training.batch_size
    feature_batch_size: int = CNN_DEFAULTS.training.feature_batch_size
    learning_rate: float = CNN_DEFAULTS.training.learning_rate
    optimizer: str = CNN_DEFAULTS.training.optimizer
    momentum: float = CNN_DEFAULTS.training.momentum
    nesterov: bool = CNN_DEFAULTS.training.nesterov
    weight_decay: float = CNN_DEFAULTS.training.weight_decay
    early_stopping_patience: int = CNN_DEFAULTS.training.early_stopping_patience
    reduce_lr_on_plateau: bool = CNN_DEFAULTS.training.reduce_lr_on_plateau
    reduce_lr_factor: float = CNN_DEFAULTS.training.reduce_lr_factor
    reduce_lr_patience: int = CNN_DEFAULTS.training.reduce_lr_patience
    min_learning_rate: float = CNN_DEFAULTS.training.min_learning_rate

    @property
    def optimizer_spec(self) -> OptimizerSpec:
        return OptimizerSpec(
            name=self.optimizer,
            learning_rate=self.learning_rate,
            momentum=self.momentum,
            nesterov=self.nesterov,
            weight_decay=self.weight_decay,
        )

    @property
    def scheduler_spec(self) -> SchedulerSpec:
        return SchedulerSpec(
            reduce_lr_on_plateau=self.reduce_lr_on_plateau,
            reduce_lr_factor=self.reduce_lr_factor,
            reduce_lr_patience=self.reduce_lr_patience,
            min_learning_rate=self.min_learning_rate,
        )


@dataclass(frozen=True)
class CnnRuntimeConfig:
    gpu: str = CNN_DEFAULTS.runtime.gpu
    mixed_precision: bool = CNN_DEFAULTS.runtime.mixed_precision
    random_seed: int = CNN_DEFAULTS.runtime.random_seed
    project_root: str = CNN_DEFAULTS.runtime.project_root

    @property
    def runtime_spec(self) -> RuntimeSpec:
        return RuntimeSpec(
            gpu=self.gpu,
            mixed_precision=self.mixed_precision,
            random_seed=self.random_seed,
            project_root=self.project_root,
        )


@dataclass(frozen=True)
class CnnExperimentConfig:
    operation: str
    dataset: CnnDatasetSpec
    preprocess: CnnPreprocessSpec
    extractor: CnnExtractorSpec
    head: CnnHeadSpec
    training: CnnTrainingSpec
    runtime: CnnRuntimeConfig

    @property
    def partition(self) -> PartitionSpec:
        return self.dataset.partition_spec

    @property
    def model(self) -> CnnModelSpec:
        return CnnModelSpec(extractor=self.extractor, head=self.head)

    @property
    def partition_tag(self) -> str:
        return f"dataset_{self.dataset.name}_{self.dataset.split}_{self.dataset.partition_mode}"

    @property
    def train_preprocess_tag(self) -> str:
        return self.preprocess.train_preprocess_tag

    @property
    def predict_preprocess_tag(self) -> str:
        return self.preprocess.predict_preprocess_tag

    @property
    def extractor_tag(self) -> str:
        return self.extractor.tag

    @property
    def head_tag(self) -> str:
        return self.head.tag

    @property
    def feature_spec_tag(self) -> str:
        return self.predict_preprocess_tag

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "operation": self.operation,
            "dataset": asdict(self.dataset),
            "partition": asdict(self.partition),
            "preprocess": asdict(self.preprocess),
            "extractor": asdict(self.extractor),
            "head": asdict(self.head),
            "training": {
                "epochs": self.training.epochs,
                "batch_size": self.training.batch_size,
                "feature_batch_size": self.training.feature_batch_size,
                "early_stopping_patience": self.training.early_stopping_patience,
                "optimizer": asdict(self.training.optimizer_spec),
                "scheduler": asdict(self.training.scheduler_spec),
            },
            "runtime": asdict(self.runtime),
            "model": {"extractor": asdict(self.extractor), "head": asdict(self.head)},
        }
        return payload


def default_cnn_experiment(operation: str = "train") -> CnnExperimentConfig:
    return CnnExperimentConfig(
        operation=operation,
        dataset=CnnDatasetSpec(),
        preprocess=CnnPreprocessSpec(),
        extractor=CnnExtractorSpec(
            backbone=CNN_DEFAULTS.extractor.backbone,
            weights=CNN_DEFAULTS.extractor.weights,
            trainable=CNN_DEFAULTS.extractor.trainable,
            feature_dim=CNN_DEFAULTS.extractor.feature_dim,
        ),
        head=CnnHeadSpec(
            hidden_units=CNN_DEFAULTS.head.hidden_units,
            dropouts=CNN_DEFAULTS.head.dropouts,
        ),
        training=CnnTrainingSpec(),
        runtime=CnnRuntimeConfig(),
    )


__all__ = [
    "CNN_DEFAULTS",
    "CnnDefaults",
    "CnnDatasetSpec",
    "CnnPreprocessSpec",
    "CnnExtractorSpec",
    "CnnHeadSpec",
    "CnnTrainingSpec",
    "CnnRuntimeConfig",
    "CnnExperimentConfig",
    "default_cnn_experiment",
    "normalize_partition_mode",
]
