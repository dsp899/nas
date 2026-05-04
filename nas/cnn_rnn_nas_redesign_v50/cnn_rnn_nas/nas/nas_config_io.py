import argparse
from dataclasses import replace
from typing import Any, Dict, Optional, Tuple

from ..common.config_io import load_optional_json_config, merge_with_defaults
from ..config.nas_config import NAS_DEFAULTS, NasControllerConfig, NasControllerModelConfig, NasControllerTrainingConfig, NasSearchSpaceConfig, default_nas_experiment
from ..config.rnn_config import RnnArchitectureConfig, RnnDataConfig, RnnExperimentConfig, RnnOptimizerConfig, RnnRuntimeConfig

NAS_APP_OPERATIONS: Tuple[str, ...] = ("search",)


def build_nas_app_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lanzador NAS basado en defaults internos y fichero de configuración JSON opcional")
    parser.add_argument("operation", choices=NAS_APP_OPERATIONS)
    parser.add_argument("--config", required=False, help="Ruta opcional al fichero JSON de configuración NAS")
    return parser


def _default_payload() -> Dict[str, Any]:
    base = default_nas_experiment()
    payload = base.to_dict()
    payload["controller"] = {
        "model": {"lstm_dim": NAS_DEFAULTS.controller.model.lstm_dim},
        "optimizer": {
            "name": NAS_DEFAULTS.controller.optimizer.name,
            "learning_rate": NAS_DEFAULTS.controller.optimizer.learning_rate,
            "momentum": NAS_DEFAULTS.controller.optimizer.momentum,
            "nesterov": NAS_DEFAULTS.controller.optimizer.nesterov,
            "weight_decay": NAS_DEFAULTS.controller.optimizer.weight_decay,
        },
        "scheduler": {
            "reduce_lr_on_plateau": NAS_DEFAULTS.controller.scheduler.reduce_lr_on_plateau,
            "reduce_lr_factor": NAS_DEFAULTS.controller.scheduler.reduce_lr_factor,
            "reduce_lr_patience": NAS_DEFAULTS.controller.scheduler.reduce_lr_patience,
            "min_learning_rate": NAS_DEFAULTS.controller.scheduler.min_learning_rate,
        },
        "training": {
            "sampling_epochs": NAS_DEFAULTS.controller.training.sampling_epochs,
            "samples_per_epoch": NAS_DEFAULTS.controller.training.samples_per_epoch,
            "training_epochs": NAS_DEFAULTS.controller.training.training_epochs,
            "sampling_attempts_multiplier": NAS_DEFAULTS.controller.training.sampling_attempts_multiplier,
            "sampling_attempts_minimum": NAS_DEFAULTS.controller.training.sampling_attempts_minimum,
            "reward_baseline_strategy": NAS_DEFAULTS.controller.training.reward_baseline_strategy,
            "reward_baseline_ema_decay": NAS_DEFAULTS.controller.training.reward_baseline_ema_decay,
            "reward_standardize_advantage": NAS_DEFAULTS.controller.training.reward_standardize_advantage,
            "rolling_window": NAS_DEFAULTS.controller.training.rolling_window,
            "rolling_window_multiplier": NAS_DEFAULTS.controller.training.rolling_window_multiplier,
        },
    }
    payload["search_space"] = base.search_space.to_dict()
    payload["architecture_training"] = payload.pop("training")
    payload["data_source"] = {
        "dataset": payload.pop("dataset"),
        "partition": payload.pop("partition"),
        "preprocess": payload.pop("preprocess"),
        "feature_source": payload.pop("feature_source"),
    }
    payload.pop("data", None)
    payload.pop("architecture", None)
    payload.pop("optimizer", None)
    payload.pop("nas", None)
    return payload


def load_nas_config(config_path: Optional[str] = None) -> RnnExperimentConfig:
    payload = merge_with_defaults(load_optional_json_config(config_path), _default_payload())
    data_source = payload["data_source"]
    dataset_payload = data_source["dataset"]
    partition_payload = data_source["partition"]
    preprocess_payload = data_source["preprocess"]
    feature_payload = data_source["feature_source"]
    arch_training = payload["architecture_training"]
    arch_optimizer = arch_training["optimizer"]
    arch_scheduler = arch_training["scheduler"]
    runtime_payload = payload["runtime"]
    controller_payload = payload["controller"]
    controller_model = controller_payload["model"]
    controller_optimizer = controller_payload["optimizer"]
    controller_scheduler = controller_payload["scheduler"]
    controller_training = controller_payload["training"]
    search_space_payload = payload["search_space"]

    search_space = NasSearchSpaceConfig(**search_space_payload)
    base_data = RnnDataConfig(
        cnn=str(search_space.options("cnn")[0]),
        name=dataset_payload["name"],
        frames=preprocess_payload["frames"],
        image_size=preprocess_payload["image_size"],
        seq=int(search_space.options("seq")[0]),
        split=dataset_payload["split"],
        val_fraction=partition_payload["val_fraction"],
        partition_mode=partition_payload["mode"],
        sampling=preprocess_payload["sampling"],
        resize_mode=preprocess_payload["resize_mode"],
        cnn_training_signature=feature_payload["cnn_training_signature"],
        cnn_feature_export_signature=feature_payload["cnn_feature_export_signature"],
    )
    units = (
        int(search_space.options("units_0")[0]),
        int(search_space.options("units_1")[0]),
        int(search_space.options("units_2")[0]),
    )
    base_arch = RnnArchitectureConfig(
        rnn=str(search_space.options("rnn")[0]),
        direction=str(search_space.options("direction")[0]),
        units=units,
        memory_mode=str(search_space.options("memory_mode")[0]),
        head_units=int(search_space.options("head_units")[0]),
        video_decision=str(search_space.options("video_decision")[0]),
        video_decision_input=str(search_space.options("video_decision_input")[0]),
    )
    runtime = RnnRuntimeConfig(
        epochs=arch_training["epochs"],
        batch_size=arch_training["batch_size"],
        learning_rate=arch_optimizer["learning_rate"],
        gpu=runtime_payload["gpu"],
        project_root=runtime_payload["project_root"],
        random_seed=runtime_payload["random_seed"],
        mixed_precision=runtime_payload["mixed_precision"],
        allow_epoch_extension_resume=arch_training["allow_epoch_extension_resume"],
        reduce_lr_on_plateau=arch_scheduler["reduce_lr_on_plateau"],
        reduce_lr_factor=arch_scheduler["reduce_lr_factor"],
        reduce_lr_patience=arch_scheduler["reduce_lr_patience"],
        min_learning_rate=arch_scheduler["min_learning_rate"],
    )
    nas_controller = NasControllerConfig(
        model=NasControllerModelConfig(lstm_dim=controller_model["lstm_dim"]),
        optimizer=type(NAS_DEFAULTS.controller.optimizer)(**controller_optimizer),
        scheduler=type(NAS_DEFAULTS.controller.scheduler)(**controller_scheduler),
        training=NasControllerTrainingConfig(**controller_training),
    )
    return RnnExperimentConfig(
        operation="search",
        data=base_data,
        architecture=base_arch,
        runtime=runtime,
        optimizer=RnnOptimizerConfig(
            name=arch_optimizer["name"],
            momentum=arch_optimizer["momentum"],
            nesterov=arch_optimizer["nesterov"],
            weight_decay=arch_optimizer["weight_decay"],
        ),
        nas=nas_controller,
        search_space=search_space,
    )


def run_nas(config_path: Optional[str] = None) -> Dict[str, Any]:
    from ..common.artifacts import ProjectPaths
    from ..common.registries import NasSearchRegistry, RnnExperimentRegistry
    from ..common.runtime import configure_runtime
    from .nas_engine import NasSearchEngine

    config = load_nas_config(config_path)
    effective_mixed_precision = configure_runtime(config.runtime.gpu, config.runtime.mixed_precision, config.runtime.random_seed)
    config = replace(config, runtime=replace(config.runtime, mixed_precision=effective_mixed_precision))
    paths = ProjectPaths(config.runtime.project_root)
    registry = RnnExperimentRegistry(paths.rnn_registry_path)
    search_registry = NasSearchRegistry(paths.nas_search_registry_path)
    engine = NasSearchEngine(config, paths, registry, search_registry)
    return engine.run()


load_nas_config_from_file = load_nas_config
run_nas_from_config_file = run_nas
