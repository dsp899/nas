import argparse
from dataclasses import replace
from typing import Any, Dict, Optional, Tuple

from ..common.config_io import load_optional_json_config, merge_with_defaults
from ..config.rnn_config import RnnArchitectureConfig, RnnDataConfig, RnnExperimentConfig, RnnOptimizerConfig, RnnRuntimeConfig, default_rnn_experiment

RNN_APP_OPERATIONS: Tuple[str, ...] = ("train", "test", "deploy")
RNN_DEPLOY_ACTIONS: Tuple[str, ...] = ("export", "test_runtime", "test_pipeline")


def build_rnn_app_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lanzador RNN basado en defaults internos y fichero de configuración JSON opcional")
    parser.add_argument("operation", choices=RNN_APP_OPERATIONS)
    parser.add_argument("--config", required=False, help="Ruta opcional al fichero JSON de configuración RNN")
    return parser




def _normalize_rnn_model_payload(model_payload: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(model_payload)
    if "decision" not in normalized:
        normalized["decision"] = {
            "video_decision": normalized.get("video_decision", "average"),
            "video_decision_input": normalized.get("video_decision_input", "clip_logits"),
        }
    return normalized


def _normalize_training_payload(training_payload: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(training_payload)
    optimizer_payload = normalized.get("optimizer", {})
    if not isinstance(optimizer_payload, dict):
        optimizer_payload = {
            "name": normalized.get("optimizer"),
            "learning_rate": normalized.get("learning_rate"),
            "momentum": normalized.get("momentum", 0.0),
            "nesterov": normalized.get("nesterov", False),
            "weight_decay": normalized.get("weight_decay", 0.0),
        }
    scheduler_payload = normalized.get("scheduler", {})
    if not isinstance(scheduler_payload, dict):
        scheduler_payload = {}
    scheduler_payload = {
        "reduce_lr_on_plateau": scheduler_payload.get("reduce_lr_on_plateau", normalized.get("reduce_lr_on_plateau", False)),
        "reduce_lr_factor": scheduler_payload.get("reduce_lr_factor", normalized.get("reduce_lr_factor", 0.5)),
        "reduce_lr_patience": scheduler_payload.get("reduce_lr_patience", normalized.get("reduce_lr_patience", 2)),
        "min_learning_rate": scheduler_payload.get("min_learning_rate", normalized.get("min_learning_rate", 1e-6)),
    }
    normalized["optimizer"] = optimizer_payload
    normalized["scheduler"] = scheduler_payload
    normalized["allow_epoch_extension_resume"] = bool(normalized.get("allow_epoch_extension_resume", False))
    return normalized

def _default_payload(operation: str) -> Dict[str, Any]:
    base = default_rnn_experiment(operation)
    payload = base.to_dict()
    payload["deploy"] = {
        "action": "export",
        "force": False,
        "signatures": {"experiment": None, "export": None},
        "export": {"runtime": "tflite", "tflite_precision": "fp32"},
        "test": {"test_split": "test", "save_clip_predictions": False, "save_video_predictions": True},
        "pipeline": {"cnn_deploy_signature": None, "save_quantized_sequences": True},
    }
    return payload


def load_rnn_config(operation: str, config_path: Optional[str] = None) -> Tuple[RnnExperimentConfig, Dict[str, Any]]:
    payload = merge_with_defaults(load_optional_json_config(config_path), _default_payload(operation))
    dataset_payload = payload["dataset"]
    partition_payload = payload["partition"]
    preprocess_payload = payload["preprocess"]
    feature_source_payload = payload["feature_source"]
    model_payload = _normalize_rnn_model_payload(payload["model"])
    encoder_payload = model_payload["encoder"]
    head_payload = model_payload.get("head", {})
    decision_payload = model_payload.get("decision", {})
    training_payload = _normalize_training_payload(payload["training"])
    optimizer_payload = training_payload.get("optimizer", {})
    scheduler_payload = training_payload.get("scheduler", {})
    runtime_payload = payload["runtime"]

    hidden_units = tuple(int(unit) for unit in head_payload.get("hidden_units", []))
    if len(hidden_units) > 1:
        raise ValueError("El head RNN actual admite una sola capa oculta; usa model.head.hidden_units con longitud 0 o 1")
    head_units = int(hidden_units[0]) if hidden_units else 0

    config = RnnExperimentConfig(
        operation=operation,
        data=RnnDataConfig(
            cnn=feature_source_payload["cnn"],
            name=dataset_payload["name"],
            frames=preprocess_payload["frames"],
            image_size=preprocess_payload["image_size"],
            seq=encoder_payload["seq_length"],
            split=dataset_payload["split"],
            val_fraction=partition_payload["val_fraction"],
            partition_mode=partition_payload["mode"],
            sampling=preprocess_payload["sampling"],
            resize_mode=preprocess_payload["resize_mode"],
            cnn_training_signature=feature_source_payload["cnn_training_signature"],
            cnn_feature_export_signature=feature_source_payload["cnn_feature_export_signature"],
        ),
        architecture=RnnArchitectureConfig(
            rnn=encoder_payload["rnn"],
            direction=encoder_payload["direction"],
            units=tuple(int(unit) for unit in encoder_payload["units"]),
            memory_mode=encoder_payload["memory_mode"],
            head_units=head_units,
            video_decision=decision_payload["video_decision"],
            video_decision_input=decision_payload["video_decision_input"],
        ),
        optimizer=RnnOptimizerConfig(
            name=optimizer_payload["name"],
            momentum=optimizer_payload["momentum"],
            nesterov=optimizer_payload["nesterov"],
            weight_decay=optimizer_payload["weight_decay"],
        ),
        runtime=RnnRuntimeConfig(
            epochs=training_payload["epochs"],
            batch_size=training_payload["batch_size"],
            learning_rate=optimizer_payload["learning_rate"],
            gpu=runtime_payload["gpu"],
            project_root=runtime_payload["project_root"],
            random_seed=runtime_payload["random_seed"],
            mixed_precision=runtime_payload["mixed_precision"],
            test_strategy=training_payload["test_strategy"],
            allow_epoch_extension_resume=training_payload["allow_epoch_extension_resume"],
            reduce_lr_on_plateau=scheduler_payload["reduce_lr_on_plateau"],
            reduce_lr_factor=scheduler_payload["reduce_lr_factor"],
            reduce_lr_patience=scheduler_payload["reduce_lr_patience"],
            min_learning_rate=scheduler_payload["min_learning_rate"],
        ),
    )
    return config, payload


def run_rnn(operation: str, config_path: Optional[str] = None) -> Dict[str, Any]:
    from ..common.runtime import configure_runtime
    from ..common.artifacts import ProjectPaths
    from ..common.registries import CnnDeployRegistry, RnnDeployEvalRegistry, RnnExperimentRegistry, RnnExportRegistry, RnnTfliteEvalRegistry
    from ..config.deploy_config import RnnDeployEvalConfig, RnnTfliteEvalConfig, RnnTfliteExportConfig
    from .rnn_data import SequenceRepository
    from .rnn_deploy import RnnDeployComparisonEvaluator, RnnTfliteEvaluator, RnnTfliteExporter
    from .rnn_test import test_rnn
    from .rnn_train import ArchitectureTrainer

    config, payload = load_rnn_config(operation, config_path)
    effective_mixed_precision = configure_runtime(config.runtime.gpu, config.runtime.mixed_precision, config.runtime.random_seed)
    config = replace(config, runtime=replace(config.runtime, mixed_precision=effective_mixed_precision))

    paths = ProjectPaths(config.runtime.project_root)
    registry = RnnExperimentRegistry(paths.rnn_registry_path)
    repo = SequenceRepository(paths)
    resolved_data, _ = repo.resolve_data_feature_source(config.data)
    config = replace(config, data=resolved_data)

    if operation == "train":
        bundle = repo.make_bundle(config.data, config.runtime.batch_size, config.runtime.random_seed)
        return ArchitectureTrainer(paths, registry).train_or_resume(config, bundle)
    if operation == "test":
        bundle = repo.make_bundle(config.data, config.runtime.batch_size, config.runtime.random_seed)
        return test_rnn(config, bundle, paths, registry)

    deploy_payload = payload.get("deploy", {})
    action = str(deploy_payload.get("action", "export")).strip().lower()
    if action not in RNN_DEPLOY_ACTIONS:
        raise ValueError(f"deploy.action RNN no soportado: {action!r}")

    export_registry = RnnExportRegistry(paths.rnn_export_registry_path)
    exporter = RnnTfliteExporter(paths, registry, export_registry)
    signatures = deploy_payload.get("signatures", {})
    export_payload = deploy_payload.get("export", {})
    test_payload = deploy_payload.get("test", {})
    pipeline_payload = deploy_payload.get("pipeline", {})
    export_config = RnnTfliteExportConfig(runtime=export_payload.get("runtime", "tflite"), tflite_precision=export_payload.get("tflite_precision", "fp32"))

    if action == "export":
        return exporter.export(
            config,
            export_config,
            explicit_experiment_signature=signatures.get("experiment"),
            force=bool(deploy_payload.get("force", False)),
        )

    if action == "test_runtime":
        eval_registry = RnnTfliteEvalRegistry(paths.rnn_tflite_eval_registry_path)
        evaluator = RnnTfliteEvaluator(paths, registry, export_registry, eval_registry, repo, exporter)
        test_config = RnnTfliteEvalConfig(
            eval_split=test_payload.get("test_split", "test"),
            save_clip_predictions=bool(test_payload.get("save_clip_predictions", False)),
            save_video_predictions=bool(test_payload.get("save_video_predictions", True)),
        )
        return evaluator.evaluate(
            config,
            export_config,
            test_config,
            explicit_experiment_signature=signatures.get("experiment"),
            explicit_export_signature=signatures.get("export"),
            force=bool(deploy_payload.get("force", False)),
        )

    eval_registry = RnnDeployEvalRegistry(paths.rnn_deploy_eval_registry_path)
    deploy_registry = CnnDeployRegistry(paths.cnn_deploy_registry_path)
    evaluator = RnnDeployComparisonEvaluator(paths, registry, export_registry, eval_registry, deploy_registry, repo, exporter)
    pipeline_test = RnnDeployEvalConfig(
        eval_split=test_payload.get("test_split", "test"),
        save_clip_predictions=bool(test_payload.get("save_clip_predictions", False)),
        save_video_predictions=bool(test_payload.get("save_video_predictions", True)),
        save_quantized_sequences=bool(pipeline_payload.get("save_quantized_sequences", True)),
    )
    return evaluator.evaluate(
        config,
        export_config,
        pipeline_test,
        explicit_experiment_signature=signatures.get("experiment"),
        explicit_export_signature=signatures.get("export"),
        explicit_cnn_deploy_signature=pipeline_payload.get("cnn_deploy_signature"),
        force=bool(deploy_payload.get("force", False)),
    )


load_rnn_config_from_file = load_rnn_config
run_rnn_from_config_file = run_rnn
