
from pathlib import Path

from ..config.rnn_config import RnnDataConfig, RnnExperimentConfig
from typing import Optional, Union


class ProjectPaths:
    def __init__(self, root: Union[str, Path] = ".") -> None:
        self.root = Path(root).resolve()

        # Raw source data only.
        self.data_root = self.root / "data"

        # Derived, versioned artifacts for every subsystem.
        self.artifacts_root = self.root / "artifacts"
        self.logs_root = self.artifacts_root / "logs"
        self.registries_root = self.artifacts_root / "registries"

        self.partitions_root = self.artifacts_root / "partitions"

        self.rnn_registry_path = self.registries_root / "rnn_experiment_registry.sqlite"
        self.cnn_registry_path = self.registries_root / "cnn_experiment_registry.sqlite"
        self.nas_search_registry_path = self.registries_root / "nas_search_registry.sqlite"
        self.cnn_deploy_registry_path = self.registries_root / "cnn_deploy_registry.sqlite"
        self.rnn_export_registry_path = self.registries_root / "rnn_export_registry.sqlite"
        self.rnn_tflite_eval_registry_path = self.registries_root / "rnn_tflite_eval_registry.sqlite"
        self.rnn_deploy_eval_registry_path = self.registries_root / "rnn_deploy_eval_registry.sqlite"
        self.cnn_quant_eval_registry_path = self.registries_root / "cnn_quant_eval_registry.sqlite"
        self.cnn_extractor_eval_registry_path = self.registries_root / "cnn_extractor_eval_registry.sqlite"
        self.cnn_deploy_eval_registry_path = self.registries_root / "cnn_deploy_eval_registry.sqlite"

        for path in (
            self.data_root,
            self.artifacts_root,
            self.logs_root,
            self.registries_root,
            self.partitions_root,
        ):
            path.mkdir(parents=True, exist_ok=True)


    def partition_root(self, partition_tag: str) -> Path:
        path = self.partitions_root / partition_tag
        path.mkdir(parents=True, exist_ok=True)
        return path

    def cnn_root(self, partition_tag: str) -> Path:
        path = self.partition_root(partition_tag) / "cnn"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def rnn_root(self, partition_tag: str) -> Path:
        path = self.partition_root(partition_tag) / "rnn"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def nas_root(self, partition_tag: str) -> Path:
        path = self.partition_root(partition_tag) / "nas"
        path.mkdir(parents=True, exist_ok=True)
        return path

    # ------------------------------------------------------------------
    # RNN / NAS data-derived artifacts
    # ------------------------------------------------------------------
    def feature_dir(self, data: RnnDataConfig) -> Path:
        if not data.cnn_training_signature or not data.cnn_feature_export_signature:
            raise ValueError("cnn_training_signature y cnn_feature_export_signature son obligatorios para resolver feature_dir")
        return (
            self.cnn_root(data.partition_tag)
            / "features"
            / data.cnn_training_signature
            / data.feature_spec_tag
            / data.cnn_feature_export_signature
        )

    def sequence_dir(self, data: RnnDataConfig) -> Path:
        return (
            self.rnn_root(data.partition_tag)
            / "sequence_cache"
            / data.cnn
            / data.feature_spec_tag
            / data.sequence_spec_tag
        )

    # ------------------------------------------------------------------
    # RNN experiment artifacts
    # ------------------------------------------------------------------
    def model_dir(self, config: RnnExperimentConfig, signature: str) -> Path:
        return (
            self.rnn_root(config.data.partition_tag)
            / "models"
            / config.data.cnn
            / config.data.feature_spec_tag
            / config.data.sequence_spec_tag
            / config.architecture.tag
            / signature
        )

    def model_path(self, config: RnnExperimentConfig, signature: str) -> Path:
        return self.model_dir(config, signature) / "best.keras"

    def model_last_path(self, config: RnnExperimentConfig, signature: str) -> Path:
        return self.model_dir(config, signature) / "last.keras"

    def model_training_state_path(self, config: RnnExperimentConfig, signature: str) -> Path:
        return self.model_dir(config, signature) / "training_state.json"

    def model_optimizer_state_path(self, config: RnnExperimentConfig, signature: str) -> Path:
        return self.model_dir(config, signature) / "optimizer_state.pkl"

    def model_manifest_path(self, config: RnnExperimentConfig, signature: str) -> Path:
        return self.model_dir(config, signature) / "artifact_manifest.json"

    def run_dir(self, config: RnnExperimentConfig, signature: str) -> Path:
        path = self.model_dir(config, signature)
        path.mkdir(parents=True, exist_ok=True)
        return path


    def rnn_export_dir(self, config: RnnExperimentConfig, experiment_signature: str, export_signature: str) -> Path:
        return (
            self.rnn_root(config.data.partition_tag)
            / "exports"
            / config.data.cnn
            / config.data.feature_spec_tag
            / config.data.sequence_spec_tag
            / config.architecture.tag
            / experiment_signature
            / export_signature
        )

    def rnn_export_manifest_path(self, config: RnnExperimentConfig, experiment_signature: str, export_signature: str) -> Path:
        return self.rnn_export_dir(config, experiment_signature, export_signature) / "export_manifest.json"

    def rnn_encoder_saved_model_dir(self, config: RnnExperimentConfig, experiment_signature: str, export_signature: str) -> Path:
        return self.rnn_export_dir(config, experiment_signature, export_signature) / "encoder_saved_model"

    def rnn_head_saved_model_dir(self, config: RnnExperimentConfig, experiment_signature: str, export_signature: str) -> Path:
        return self.rnn_export_dir(config, experiment_signature, export_signature) / "head_saved_model"

    def rnn_encoder_tflite_model_path(self, config: RnnExperimentConfig, experiment_signature: str, export_signature: str) -> Path:
        return self.rnn_export_dir(config, experiment_signature, export_signature) / "encoder.tflite"

    def rnn_head_tflite_model_path(self, config: RnnExperimentConfig, experiment_signature: str, export_signature: str) -> Path:
        return self.rnn_export_dir(config, experiment_signature, export_signature) / "head.tflite"

    def rnn_tflite_eval_dir(self, config: RnnExperimentConfig, experiment_signature: str, export_signature: str, eval_signature: str) -> Path:
        return self.rnn_export_dir(config, experiment_signature, export_signature) / "evaluations" / eval_signature

    def rnn_tflite_eval_manifest_path(self, config: RnnExperimentConfig, experiment_signature: str, export_signature: str, eval_signature: str) -> Path:
        return self.rnn_tflite_eval_dir(config, experiment_signature, export_signature, eval_signature) / "evaluation_manifest.json"

    def rnn_tflite_eval_clip_predictions_path(self, config: RnnExperimentConfig, experiment_signature: str, export_signature: str, eval_signature: str) -> Path:
        return self.rnn_tflite_eval_dir(config, experiment_signature, export_signature, eval_signature) / "clip_predictions.npz"

    def rnn_tflite_eval_video_predictions_path(self, config: RnnExperimentConfig, experiment_signature: str, export_signature: str, eval_signature: str) -> Path:
        return self.rnn_tflite_eval_dir(config, experiment_signature, export_signature, eval_signature) / "video_predictions.npz"



    def rnn_deploy_eval_dir(self, config: RnnExperimentConfig, experiment_signature: str, export_signature: str, eval_signature: str) -> Path:
        return self.rnn_export_dir(config, experiment_signature, export_signature) / "comparison_evaluations" / eval_signature

    def rnn_deploy_eval_manifest_path(self, config: RnnExperimentConfig, experiment_signature: str, export_signature: str, eval_signature: str) -> Path:
        return self.rnn_deploy_eval_dir(config, experiment_signature, export_signature, eval_signature) / "evaluation_manifest.json"

    def rnn_deploy_eval_clip_predictions_path(self, config: RnnExperimentConfig, experiment_signature: str, export_signature: str, eval_signature: str) -> Path:
        return self.rnn_deploy_eval_dir(config, experiment_signature, export_signature, eval_signature) / "clip_predictions.npz"

    def rnn_deploy_eval_video_predictions_path(self, config: RnnExperimentConfig, experiment_signature: str, export_signature: str, eval_signature: str) -> Path:
        return self.rnn_deploy_eval_dir(config, experiment_signature, export_signature, eval_signature) / "video_predictions.npz"

    def rnn_deploy_quantized_sequences_path(self, config: RnnExperimentConfig, experiment_signature: str, export_signature: str, eval_signature: str, mode: str) -> Path:
        return self.rnn_deploy_eval_dir(config, experiment_signature, export_signature, eval_signature) / f"quantized_sequences_{mode}.npy"

    def rnn_deploy_quantized_labels_path(self, config: RnnExperimentConfig, experiment_signature: str, export_signature: str, eval_signature: str, mode: str) -> Path:
        return self.rnn_deploy_eval_dir(config, experiment_signature, export_signature, eval_signature) / f"quantized_labels_{mode}.npy"

    def rnn_deploy_quantized_video_ids_path(self, config: RnnExperimentConfig, experiment_signature: str, export_signature: str, eval_signature: str, mode: str) -> Path:
        return self.rnn_deploy_eval_dir(config, experiment_signature, export_signature, eval_signature) / f"quantized_videos_id_{mode}.npy"

    # ------------------------------------------------------------------
    # CNN experiment artifacts
    # ------------------------------------------------------------------
    def cnn_feature_dir(self, config: "CnnExperimentConfig", training_signature: str, feature_export_signature: str) -> Path:
        return (
            self.cnn_root(config.partition_tag)
            / "features"
            / training_signature
            / config.predict_preprocess_tag
            / feature_export_signature
        )

    def cnn_model_dir(self, config: "CnnExperimentConfig", signature: str) -> Path:
        return (
            self.cnn_root(config.partition_tag)
            / "models"
            / config.train_preprocess_tag
            / config.extractor_tag
            / config.head_tag
            / signature
        )

    def cnn_best_model_path(self, config: "CnnExperimentConfig", signature: str) -> Path:
        return self.cnn_model_dir(config, signature) / "best_classifier.keras"

    def cnn_last_model_path(self, config: "CnnExperimentConfig", signature: str) -> Path:
        return self.cnn_model_dir(config, signature) / "latest_classifier.keras"

    def cnn_training_state_path(self, config: "CnnExperimentConfig", signature: str) -> Path:
        return self.cnn_model_dir(config, signature) / "training_state.json"

    def cnn_optimizer_state_path(self, config: "CnnExperimentConfig", signature: str) -> Path:
        return self.cnn_model_dir(config, signature) / "optimizer_state.pkl"

    def cnn_progress_path(self, config: "CnnExperimentConfig", signature: str) -> Path:
        return self.cnn_training_state_path(config, signature)

    def cnn_model_manifest_path(self, config: "CnnExperimentConfig", signature: str) -> Path:
        return self.cnn_model_dir(config, signature) / "model_artifact_manifest.json"

    def cnn_manifest_path(self, config: "CnnExperimentConfig", signature: str) -> Path:
        return self.cnn_model_manifest_path(config, signature)

    def cnn_feature_manifest_path(self, config: "CnnExperimentConfig", training_signature: str, feature_export_signature: str) -> Path:
        return self.cnn_feature_dir(config, training_signature, feature_export_signature) / "feature_export_manifest.json"

    def cnn_feature_array_path(self, config: "CnnExperimentConfig", training_signature: str, mode: str, kind: str, feature_export_signature: str) -> Path:
        mapping = {
            "features": f"video_features_{mode}.npy",
            "labels": f"video_labels_{mode}.npy",
            "video_ids": f"video_ids_{mode}.npy",
        }
        if kind not in mapping:
            raise ValueError(f"Tipo de artefacto de features no soportado: {kind!r}")
        return self.cnn_feature_dir(config, training_signature, feature_export_signature) / mapping[kind]

    def cnn_run_dir(self, config: "CnnExperimentConfig", signature: str) -> Path:
        path = self.cnn_model_dir(config, signature)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def cnn_deploy_dir(self, config: "CnnExperimentConfig", training_signature: str, deploy_signature: str) -> Path:
        return (
            self.cnn_root(config.partition_tag)
            / "deploy"
            / config.train_preprocess_tag
            / config.extractor_tag
            / config.head_tag
            / training_signature
            / deploy_signature
        )

    def cnn_deploy_manifest_path(self, config: "CnnExperimentConfig", training_signature: str, deploy_signature: str) -> Path:
        return self.cnn_deploy_dir(config, training_signature, deploy_signature) / "deploy_manifest.json"

    def cnn_saved_model_dir(self, config: "CnnExperimentConfig", training_signature: str, deploy_signature: str) -> Path:
        return self.cnn_deploy_dir(config, training_signature, deploy_signature) / "saved_model"

    def cnn_calibration_dir(self, config: "CnnExperimentConfig", training_signature: str, deploy_signature: str) -> Path:
        return self.cnn_deploy_dir(config, training_signature, deploy_signature) / "calibration"

    def cnn_inspector_script_path(self, config: "CnnExperimentConfig", training_signature: str, deploy_signature: str) -> Path:
        return self.cnn_deploy_dir(config, training_signature, deploy_signature) / "run_model_inspector.sh"

    def cnn_quantize_script_path(self, config: "CnnExperimentConfig", training_signature: str, deploy_signature: str) -> Path:
        return self.cnn_deploy_dir(config, training_signature, deploy_signature) / "run_quantize.sh"

    def cnn_compile_script_path(self, config: "CnnExperimentConfig", training_signature: str, deploy_signature: str) -> Path:
        return self.cnn_deploy_dir(config, training_signature, deploy_signature) / "run_compile.sh"

    def cnn_compiled_model_path(self, config: "CnnExperimentConfig", training_signature: str, deploy_signature: str) -> Path:
        return self.cnn_deploy_dir(config, training_signature, deploy_signature) / "compiled" / "model.xmodel"

    def cnn_quantized_dir(self, config: "CnnExperimentConfig", training_signature: str, deploy_signature: str) -> Path:
        return self.cnn_deploy_dir(config, training_signature, deploy_signature) / "quantized"

    def cnn_quantized_extractor_path(self, config: "CnnExperimentConfig", training_signature: str, deploy_signature: str) -> Path:
        return self.cnn_quantized_dir(config, training_signature, deploy_signature) / "quantized_extractor.h5"

    def cnn_quantized_classifier_path(self, config: "CnnExperimentConfig", training_signature: str, deploy_signature: str) -> Path:
        return self.cnn_quantized_dir(config, training_signature, deploy_signature) / "quantized_classifier.h5"

    def cnn_classifier_saved_model_dir(self, config: "CnnExperimentConfig", training_signature: str, deploy_signature: str) -> Path:
        return self.cnn_deploy_dir(config, training_signature, deploy_signature) / "saved_model_classifier"

    def cnn_classifier_float_model_path(self, config: "CnnExperimentConfig", training_signature: str, deploy_signature: str) -> Path:
        return self.cnn_deploy_dir(config, training_signature, deploy_signature) / "float_classifier.h5"

    def cnn_probe_head_float_model_path(self, config: "CnnExperimentConfig", training_signature: str, deploy_signature: str) -> Path:
        return self.cnn_deploy_dir(config, training_signature, deploy_signature) / "float_probe_head.h5"

    def cnn_quant_eval_dir(self, config: "CnnExperimentConfig", training_signature: str, deploy_signature: str, eval_signature: str) -> Path:
        return self.cnn_deploy_dir(config, training_signature, deploy_signature) / "evaluations" / eval_signature

    def cnn_quant_eval_manifest_path(self, config: "CnnExperimentConfig", training_signature: str, deploy_signature: str, eval_signature: str) -> Path:
        return self.cnn_quant_eval_dir(config, training_signature, deploy_signature, eval_signature) / "evaluation_manifest.json"

    def cnn_quant_eval_frame_predictions_path(self, config: "CnnExperimentConfig", training_signature: str, deploy_signature: str, eval_signature: str) -> Path:
        return self.cnn_quant_eval_dir(config, training_signature, deploy_signature, eval_signature) / "frame_predictions.npz"

    def cnn_quant_eval_video_predictions_path(self, config: "CnnExperimentConfig", training_signature: str, deploy_signature: str, eval_signature: str) -> Path:
        return self.cnn_quant_eval_dir(config, training_signature, deploy_signature, eval_signature) / "video_predictions.npz"

    def cnn_extractor_eval_dir(self, config: "CnnExperimentConfig", training_signature: str, deploy_signature: str, eval_signature: str) -> Path:
        return self.cnn_deploy_dir(config, training_signature, deploy_signature) / "extractor_evaluations" / eval_signature

    def cnn_extractor_eval_manifest_path(self, config: "CnnExperimentConfig", training_signature: str, deploy_signature: str, eval_signature: str) -> Path:
        return self.cnn_extractor_eval_dir(config, training_signature, deploy_signature, eval_signature) / "evaluation_manifest.json"

    def cnn_extractor_eval_frame_predictions_path(self, config: "CnnExperimentConfig", training_signature: str, deploy_signature: str, eval_signature: str) -> Path:
        return self.cnn_extractor_eval_dir(config, training_signature, deploy_signature, eval_signature) / "frame_predictions.npz"

    def cnn_extractor_eval_video_predictions_path(self, config: "CnnExperimentConfig", training_signature: str, deploy_signature: str, eval_signature: str) -> Path:
        return self.cnn_extractor_eval_dir(config, training_signature, deploy_signature, eval_signature) / "video_predictions.npz"

    def cnn_deploy_eval_dir(self, config: "CnnExperimentConfig", training_signature: str, deploy_signature: str, eval_signature: str) -> Path:
        return self.cnn_deploy_dir(config, training_signature, deploy_signature) / "comparison_evaluations" / eval_signature

    def cnn_deploy_eval_manifest_path(self, config: "CnnExperimentConfig", training_signature: str, deploy_signature: str, eval_signature: str) -> Path:
        return self.cnn_deploy_eval_dir(config, training_signature, deploy_signature, eval_signature) / "evaluation_manifest.json"

    def cnn_deploy_eval_frame_predictions_path(self, config: "CnnExperimentConfig", training_signature: str, deploy_signature: str, eval_signature: str) -> Path:
        return self.cnn_deploy_eval_dir(config, training_signature, deploy_signature, eval_signature) / "frame_predictions.npz"

    def cnn_deploy_eval_video_predictions_path(self, config: "CnnExperimentConfig", training_signature: str, deploy_signature: str, eval_signature: str) -> Path:
        return self.cnn_deploy_eval_dir(config, training_signature, deploy_signature, eval_signature) / "video_predictions.npz"

    # ------------------------------------------------------------------
    # NAS artifacts
    # ------------------------------------------------------------------
    def search_experiment_dir(self, partition_tag: str, search_signature: str) -> Path:
        path = self.nas_root(partition_tag) / "searches" / search_signature
        path.mkdir(parents=True, exist_ok=True)
        return path

    def search_run_dir(self, partition_tag: str, search_signature: str, run_id: str) -> Path:
        path = self.search_experiment_dir(partition_tag, search_signature) / "runs" / run_id
        path.mkdir(parents=True, exist_ok=True)
        return path
