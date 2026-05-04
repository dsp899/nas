import json
import shutil
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from .rnn_data import DataBundle
from .rnn_model import VideoAggregator, _next_state_for_next_clip, apply_head_from_clip_model, build_rnn_model, get_state_spec, zero_state_tensors
from ..common.runtime import process_memory_mb, release_memory
from ..common.model_io import save_model_without_compile_artifacts
from ..common.training import (
    ReduceLrPlateauState,
    apply_reduce_lr_on_plateau,
    get_optimizer_learning_rate,
    restore_optimizer_state,
    save_optimizer_state,
)
from ..common.artifacts import ProjectPaths
from ..config.rnn_config import RnnExperimentConfig
from ..common.registries import (
    RnnExperimentRegistry,
    RunArtifacts,
    rnn_architecture_signature,
    rnn_experiment_signature,
    rnn_data_signature,
    rnn_resume_signature,
    rnn_runtime_signature,
)
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

class ArchitectureTrainer:
    def __init__(self, paths: ProjectPaths, registry: RnnExperimentRegistry) -> None:
        self.paths = paths
        self.registry = registry

    def _state_entries(self, config: RnnExperimentConfig) -> List[Dict[str, Any]]:
        return get_state_spec(config)

    def _call_clip_model(
        self,
        model: tf.keras.Model,
        clip_x: tf.Tensor,
        state_values: Sequence[tf.Tensor],
        *,
        training: bool,
    ) -> Tuple[tf.Tensor, tf.Tensor, List[tf.Tensor]]:
        outputs = model([clip_x, *state_values], training=training)
        if not isinstance(outputs, (list, tuple)):
            raise ValueError("Se esperaban múltiples outputs del clip model")
        clip_embedding = tf.cast(outputs[0], tf.float32)
        clip_logits = tf.cast(outputs[1], tf.float32)
        next_states = [tf.cast(value, tf.float32) for value in outputs[2:]]
        return clip_embedding, clip_logits, next_states

    def _aggregate_video_probs_train(self, config: RnnExperimentConfig, model: tf.keras.Model, x_batch: tf.Tensor) -> tf.Tensor:
        batch_size = int(x_batch.shape[0])
        num_clips = int(x_batch.shape[1])
        state_values = zero_state_tensors(config, batch_size)
        clip_embeddings: List[tf.Tensor] = []
        clip_logits: List[tf.Tensor] = []
        for clip_index in range(num_clips):
            clip_x = tf.cast(x_batch[:, clip_index], tf.float32)
            embedding, logits, next_states = self._call_clip_model(model, clip_x, state_values, training=True)
            clip_embeddings.append(embedding)
            clip_logits.append(logits)
            state_values = _next_state_for_next_clip(config, next_states)
        if config.architecture.video_decision_input == "clip_embeddings":
            stacked_embeddings = tf.stack(clip_embeddings, axis=1)
            aggregated_embedding = tf.reduce_mean(stacked_embeddings, axis=1)
            aggregated_logits = apply_head_from_clip_model(model, aggregated_embedding)
            return tf.nn.softmax(aggregated_logits, axis=-1)
        stacked_logits = tf.stack(clip_logits, axis=1)
        return VideoAggregator.surrogate_probs_from_logits(stacked_logits, config.architecture.video_decision)

    def _aggregate_video_probs_eval(self, config: RnnExperimentConfig, model: tf.keras.Model, x_batch: tf.Tensor) -> np.ndarray:
        batch_size = int(x_batch.shape[0])
        num_clips = int(x_batch.shape[1])
        state_values = zero_state_tensors(config, batch_size)
        clip_embeddings: List[np.ndarray] = []
        clip_logits: List[np.ndarray] = []
        for clip_index in range(num_clips):
            clip_x = tf.cast(x_batch[:, clip_index], tf.float32)
            embedding, logits, next_states = self._call_clip_model(model, clip_x, state_values, training=False)
            clip_embeddings.append(np.asarray(embedding.numpy(), dtype=np.float32))
            clip_logits.append(np.asarray(logits.numpy(), dtype=np.float32))
            state_values = _next_state_for_next_clip(config, next_states)
        embeddings_np = np.stack(clip_embeddings, axis=1)
        logits_np = np.stack(clip_logits, axis=1)
        if config.architecture.video_decision_input == "clip_embeddings":
            aggregated_embedding = np.mean(embeddings_np, axis=1).astype(np.float32)
            aggregated_logits = apply_head_from_clip_model(model, tf.convert_to_tensor(aggregated_embedding, dtype=tf.float32))
            return tf.nn.softmax(aggregated_logits, axis=-1).numpy().astype(np.float32)
        probs = [
            VideoAggregator.exact_probs_from_logits(video_logits, config.architecture.video_decision, logits_np.shape[-1])
            for video_logits in logits_np
        ]
        return np.asarray(probs, dtype=np.float32)

    @staticmethod
    def _search_metric_source(config: RnnExperimentConfig) -> str:
        return "test" if config.data.partition_mode == "train_test" else "val"

    def _metric_row_from_sources(self, config: RnnExperimentConfig, train_row: Dict[str, float], val_row: Dict[str, float], test_row: Dict[str, float], epoch: int, epoch_seconds: float, learning_rate_before_schedule: float) -> Dict[str, Any]:
        search_source = self._search_metric_source(config)
        row = {
            "epoch": epoch,
            "train_loss": float(train_row["train_loss"]),
            "train_acc": float(train_row["train_acc"]),
            "val_loss": float(val_row["val_loss"]),
            "val_acc": float(val_row["val_acc"]),
            "test_loss": float(test_row["test_loss"]),
            "test_acc": float(test_row["test_acc"]),
            "search_metric_source": search_source,
            "search_metric_loss": float(test_row["test_loss"] if search_source == "test" else val_row["val_loss"]),
            "search_metric_acc": float(test_row["test_acc"] if search_source == "test" else val_row["val_acc"]),
            "report_metric_source": "test",
            "report_metric_loss": float(test_row["test_loss"]),
            "report_metric_acc": float(test_row["test_acc"]),
            "epoch_seconds": epoch_seconds,
            "memory_mb": process_memory_mb(),
            "learning_rate": learning_rate_before_schedule,
        }
        return row

    def _evaluate_dataset(self, config: RnnExperimentConfig, dataset: tf.data.Dataset, model: tf.keras.Model, loss_fn: tf.keras.losses.Loss, prefix: str) -> Dict[str, float]:
        metric_loss = tf.keras.metrics.Mean()
        metric_acc = tf.keras.metrics.CategoricalAccuracy()
        for x_batch, y_batch, _video_ids, sample_weight in dataset:
            probs = self._aggregate_video_probs_eval(config, model, x_batch)
            y_batch_tf = tf.cast(y_batch, tf.float32)
            probs_tf = tf.convert_to_tensor(probs, dtype=tf.float32)
            sample_weight_tf = tf.cast(sample_weight, tf.float32)
            per_example_loss = tf.keras.losses.categorical_crossentropy(y_batch_tf, probs_tf)
            metric_loss.update_state(per_example_loss, sample_weight=sample_weight_tf)
            metric_acc.update_state(y_batch_tf, probs_tf, sample_weight=sample_weight_tf)
        return {f"{prefix}_loss": float(metric_loss.result().numpy()), f"{prefix}_acc": float(metric_acc.result().numpy())}

    @staticmethod
    def _copy_if_exists(source: Union[str, Path, None], destination: Union[str, Path]) -> None:
        if not source:
            return
        source_path = Path(source)
        if not source_path.exists():
            return
        destination_path = Path(destination)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path)

    @staticmethod
    def _initial_plateau_state(resume_state: Optional[Dict[str, Any]]) -> ReduceLrPlateauState:
        if not resume_state:
            return ReduceLrPlateauState()
        best = resume_state.get("reduce_lr_plateau_best")
        if best is None:
            best = resume_state.get("best_search_metric_loss", resume_state.get("best_test_loss"))
        return ReduceLrPlateauState(
            best=float(best) if best is not None else None,
            bad_epochs=int(resume_state.get("reduce_lr_plateau_bad_epochs", 0)),
        )

    @staticmethod
    def _resume_learning_rate(optimizer: tf.keras.optimizers.Optimizer, resume_state: Optional[Dict[str, Any]]) -> None:
        if not resume_state:
            return
        value = resume_state.get("current_learning_rate")
        if value is None:
            return
        try:
            from ..common.training import set_optimizer_learning_rate
            set_optimizer_learning_rate(optimizer, float(value))
        except Exception:
            pass

    @staticmethod
    def _apply_rnn_lr_scheduler(
        config: RnnExperimentConfig,
        optimizer: tf.keras.optimizers.Optimizer,
        plateau_state: ReduceLrPlateauState,
        current_metric: float,
    ) -> bool:
        return apply_reduce_lr_on_plateau(
            enabled=config.runtime.reduce_lr_on_plateau,
            optimizer=optimizer,
            current_metric=current_metric,
            state=plateau_state,
            factor=config.runtime.reduce_lr_factor,
            patience=config.runtime.reduce_lr_patience,
            min_learning_rate=config.runtime.min_learning_rate,
            mode="min",
        )

    def _artifact_manifest_payload(
        self,
        config: RnnExperimentConfig,
        signature: str,
        best_model_path: Path,
        last_model_path: Path,
        training_state_path: Path,
        optimizer_state_path: Path,
        *,
        resumed_from_signature: Optional[str] = None,
        current_learning_rate: Optional[float] = None,
        reduce_lr_plateau_best: Optional[float] = None,
        reduce_lr_plateau_bad_epochs: int = 0,
    ) -> Dict[str, Any]:
        benchlib_bridge_spec = {
            "layers": len(config.architecture.active_units),
            "rnn": config.architecture.rnn,
            "units_0": int(config.architecture.units[0]),
            "units_1": int(config.architecture.units[1]),
            "units_2": int(config.architecture.units[2]),
            "direction": config.architecture.direction,
            "memory_mode": config.architecture.memory_mode,
            "seq": int(config.data.seq),
            "head_units": int(config.architecture.head_units),
            "video_decision": config.architecture.video_decision,
            "video_decision_input": config.architecture.video_decision_input,
        }
        return {
            "signature": signature,
            "signature_version": "v4",
            "model_family": "clip_encoder_plus_head_with_video_decision",
            "rnn_data_signature": rnn_data_signature(config),
            "rnn_architecture_signature": rnn_architecture_signature(config),
            "rnn_runtime_signature": rnn_runtime_signature(config),
            "rnn_resume_signature": rnn_resume_signature(config),
            "best_model_path": str(best_model_path),
            "last_model_path": str(last_model_path),
            "training_state_path": str(training_state_path),
            "optimizer_state_path": str(optimizer_state_path),
            "model_path": str(best_model_path),
            "benchlib_bridge_spec": benchlib_bridge_spec,
            "state_spec": get_state_spec(config),
            "output_order": ["clip_embedding", "clip_logits", *[entry["name"] for entry in get_state_spec(config)]],
            "resumed_from_signature": resumed_from_signature,
            "current_learning_rate": float(current_learning_rate) if current_learning_rate is not None else None,
            "reduce_lr_plateau_best": float(reduce_lr_plateau_best) if reduce_lr_plateau_best is not None else None,
            "reduce_lr_plateau_bad_epochs": int(reduce_lr_plateau_bad_epochs),
            "config": config.to_dict(),
        }

    def _write_model_manifest(
        self,
        config: RnnExperimentConfig,
        signature: str,
        best_model_path: Path,
        last_model_path: Path,
        training_state_path: Path,
        optimizer_state_path: Path,
        *,
        resumed_from_signature: Optional[str] = None,
        current_learning_rate: Optional[float] = None,
        reduce_lr_plateau_best: Optional[float] = None,
        reduce_lr_plateau_bad_epochs: int = 0,
    ) -> None:
        manifest_path = self.paths.model_manifest_path(config, signature)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        payload = self._artifact_manifest_payload(
            config,
            signature,
            best_model_path,
            last_model_path,
            training_state_path,
            optimizer_state_path,
            resumed_from_signature=resumed_from_signature,
            current_learning_rate=current_learning_rate,
            reduce_lr_plateau_best=reduce_lr_plateau_best,
            reduce_lr_plateau_bad_epochs=reduce_lr_plateau_bad_epochs,
        )
        manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def _artifact_is_consistent(self, config: RnnExperimentConfig, signature: str, best_model_path: Path) -> bool:
        if not best_model_path.exists():
            return False
        manifest_path = self.paths.model_manifest_path(config, signature)
        if not manifest_path.exists():
            return False
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return False
        return (
            payload.get("signature") == signature
            and payload.get("best_model_path") == str(best_model_path)
            and payload.get("rnn_data_signature") == rnn_data_signature(config)
            and payload.get("rnn_architecture_signature") == rnn_architecture_signature(config)
            and payload.get("rnn_runtime_signature") == rnn_runtime_signature(config)
        )

    def _resolve_model_path(self, config: RnnExperimentConfig, signature: str, existing: Optional[Dict[str, Any]] = None) -> Path:
        candidates: List[Path] = []
        if existing and existing.get("best_model_path"):
            candidates.append(Path(str(existing["best_model_path"])))
        if existing and existing.get("model_path"):
            candidates.append(Path(str(existing["model_path"])))
        candidates.append(self.paths.model_path(config, signature))
        for candidate in candidates:
            if self._artifact_is_consistent(config, signature, candidate):
                return candidate
        return self.paths.model_path(config, signature)

    @staticmethod
    def _load_training_state(training_state_path: Path) -> Optional[Dict[str, Any]]:
        if not training_state_path.exists():
            return None
        try:
            return json.loads(training_state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

    def _write_training_state(
        self,
        config: RnnExperimentConfig,
        signature: str,
        training_state_path: Path,
        best_model_path: Path,
        last_model_path: Path,
        optimizer_state_path: Path,
        *,
        best_search_metric_acc: float,
        best_search_metric_loss: float,
        best_val_acc: float,
        best_val_loss: float,
        best_test_acc: float,
        best_test_loss: float,
        best_epoch: int,
        initial_epoch: int,
        trained_epochs: int,
        resumed_from_signature: Optional[str],
        current_learning_rate: float,
        reduce_lr_plateau_best: Optional[float],
        reduce_lr_plateau_bad_epochs: int,
    ) -> Dict[str, Any]:
        payload = {
            "signature": signature,
            "signature_version": "v3",
            "rnn_resume_signature": rnn_resume_signature(config),
            "rnn_data_signature": rnn_data_signature(config),
            "rnn_architecture_signature": rnn_architecture_signature(config),
            "rnn_runtime_signature": rnn_runtime_signature(config),
            "best_model_path": str(best_model_path),
            "last_model_path": str(last_model_path),
            "optimizer_state_path": str(optimizer_state_path),
            "search_metric_source": self._search_metric_source(config),
            "report_metric_source": "test",
            "best_search_metric_acc": float(best_search_metric_acc),
            "best_search_metric_loss": float(best_search_metric_loss),
            "best_val_acc": float(best_val_acc),
            "best_val_loss": float(best_val_loss),
            "best_test_acc": float(best_test_acc),
            "best_test_loss": float(best_test_loss),
            "best_epoch": int(best_epoch),
            "initial_epoch": int(initial_epoch),
            "trained_epochs": int(trained_epochs),
            "resumed_from_signature": resumed_from_signature,
            "current_learning_rate": float(current_learning_rate),
            "reduce_lr_plateau_best": float(reduce_lr_plateau_best) if reduce_lr_plateau_best is not None else None,
            "reduce_lr_plateau_bad_epochs": int(reduce_lr_plateau_bad_epochs),
            "config": config.to_dict(),
        }
        training_state_path.parent.mkdir(parents=True, exist_ok=True)
        training_state_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return payload

    def _resolve_resume_state(self, config: RnnExperimentConfig) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        if not config.runtime.allow_epoch_extension_resume:
            return None, None
        candidate = self.registry.find_best_resume_candidate(config)
        if not candidate:
            return None, None
        last_model_path = Path(str(candidate.get("last_model_path") or ""))
        training_state_path = Path(str(candidate.get("training_state_path") or ""))
        optimizer_state_path = Path(str(candidate.get("optimizer_state_path") or ""))
        state = self._load_training_state(training_state_path)
        if not last_model_path.exists() or not optimizer_state_path.exists() or state is None:
            return None, None
        if state.get("rnn_resume_signature") != rnn_resume_signature(config):
            return None, None
        if int(state.get("trained_epochs", 0)) >= int(config.runtime.epochs):
            return None, None
        best_candidate_path = candidate.get("best_model_path") or candidate.get("model_path")
        if best_candidate_path and not Path(str(best_candidate_path)).exists():
            return None, None
        return candidate, state

    def _train_epochs(
        self,
        config: RnnExperimentConfig,
        bundle: DataBundle,
        model: tf.keras.Model,
        loss_fn: tf.keras.losses.Loss,
        optimizer: tf.keras.optimizers.Optimizer,
        artifacts: RunArtifacts,
        best_model_path: Path,
        last_model_path: Path,
        training_state_path: Path,
        optimizer_state_path: Path,
        *,
        signature: str,
        start_epoch: int = 0,
        resume_state: Optional[Dict[str, Any]] = None,
        resumed_from_signature: Optional[str] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        best_search_acc = float(resume_state.get("best_search_metric_acc", resume_state.get("best_test_acc", -1.0))) if resume_state else -1.0
        best_search_loss = float(resume_state.get("best_search_metric_loss", resume_state.get("best_test_loss", float("inf")))) if resume_state else float("inf")
        best_val_acc = float(resume_state.get("best_val_acc", resume_state.get("best_test_acc", -1.0))) if resume_state else -1.0
        best_val_loss = float(resume_state.get("best_val_loss", resume_state.get("best_test_loss", float("inf")))) if resume_state else float("inf")
        best_test_acc = float(resume_state.get("best_test_acc", -1.0)) if resume_state else -1.0
        best_test_loss = float(resume_state.get("best_test_loss", float("inf"))) if resume_state else float("inf")
        best_epoch = int(resume_state.get("best_epoch", 0)) if resume_state else 0
        plateau_state = self._initial_plateau_state(resume_state)

        for epoch in range(start_epoch, config.runtime.epochs):
            train_loss = tf.keras.metrics.Mean()
            train_acc = tf.keras.metrics.CategoricalAccuracy()
            epoch_start = time.time()
            train_iterator = bundle.train_ds
            train_bar = None
            if verbose:
                train_bar = tqdm(bundle.train_ds, total=bundle.metadata.train_batches, desc=f"train {epoch + 1}/{config.runtime.epochs}")
                train_iterator = train_bar
            for x_batch, y_batch, _video_ids, sample_weight in train_iterator:
                x_batch = tf.cast(x_batch, tf.float32)
                y_batch = tf.cast(y_batch, tf.float32)
                sample_weight = tf.cast(sample_weight, tf.float32)
                with tf.GradientTape() as tape:
                    probs = self._aggregate_video_probs_train(config, model, x_batch)
                    per_example_loss = tf.keras.losses.categorical_crossentropy(y_batch, probs)
                    weighted_loss = tf.reduce_sum(per_example_loss * sample_weight) / tf.maximum(tf.reduce_sum(sample_weight), tf.constant(1.0, dtype=tf.float32))
                    reg_loss = tf.add_n(model.losses) if model.losses else tf.constant(0.0, dtype=tf.float32)
                    total_loss = weighted_loss + reg_loss
                gradients = tape.gradient(total_loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                train_loss.update_state(per_example_loss, sample_weight=sample_weight)
                train_acc.update_state(y_batch, probs, sample_weight=sample_weight)
                if train_bar is not None:
                    train_bar.set_postfix(loss=f"{train_loss.result().numpy():.4f}", acc=f"{train_acc.result().numpy():.4f}")

            val_row = self._evaluate_dataset(config, bundle.val_ds, model, loss_fn, "val")
            test_row = self._evaluate_dataset(config, bundle.test_ds, model, loss_fn, "test")
            epoch_seconds = time.time() - epoch_start
            learning_rate_before_schedule = get_optimizer_learning_rate(optimizer)
            row = self._metric_row_from_sources(
                config,
                {"train_loss": float(train_loss.result().numpy()), "train_acc": float(train_acc.result().numpy())},
                val_row,
                test_row,
                epoch + 1,
                epoch_seconds,
                learning_rate_before_schedule,
            )
            lr_reduced = self._apply_rnn_lr_scheduler(config, optimizer, plateau_state, row["search_metric_loss"])
            row["learning_rate_reduced"] = bool(lr_reduced)
            row["learning_rate_after_schedule"] = get_optimizer_learning_rate(optimizer)
            artifacts.append_epoch(row)
            save_model_without_compile_artifacts(model, last_model_path)
            save_optimizer_state(optimizer_state_path, optimizer)
            if row["search_metric_acc"] > best_search_acc:
                best_search_acc = row["search_metric_acc"]
                best_search_loss = row["search_metric_loss"]
                best_val_acc = row["val_acc"]
                best_val_loss = row["val_loss"]
                best_test_acc = row["test_acc"]
                best_test_loss = row["test_loss"]
                best_epoch = epoch + 1
                save_model_without_compile_artifacts(model, best_model_path)

            self._write_training_state(
                config,
                signature,
                training_state_path,
                best_model_path,
                last_model_path,
                optimizer_state_path,
                best_search_metric_acc=best_search_acc,
                best_search_metric_loss=best_search_loss,
                best_val_acc=best_val_acc,
                best_val_loss=best_val_loss,
                best_test_acc=best_test_acc,
                best_test_loss=best_test_loss,
                best_epoch=best_epoch,
                initial_epoch=start_epoch,
                trained_epochs=epoch + 1,
                resumed_from_signature=resumed_from_signature,
                current_learning_rate=get_optimizer_learning_rate(optimizer),
                reduce_lr_plateau_best=plateau_state.best,
                reduce_lr_plateau_bad_epochs=plateau_state.bad_epochs,
            )

        return {
            "search_metric_source": self._search_metric_source(config),
            "report_metric_source": "test",
            "best_search_metric_acc": float(best_search_acc),
            "best_search_metric_loss": float(best_search_loss),
            "best_val_acc": float(best_val_acc),
            "best_val_loss": float(best_val_loss),
            "best_test_acc": float(best_test_acc),
            "best_test_loss": float(best_test_loss),
            "best_epoch": int(best_epoch),
            "initial_epoch": int(start_epoch),
            "final_epoch": int(config.runtime.epochs),
            "trained_epochs": int(config.runtime.epochs),
            "current_learning_rate": get_optimizer_learning_rate(optimizer),
            "reduce_lr_plateau_best": plateau_state.best,
            "reduce_lr_plateau_bad_epochs": plateau_state.bad_epochs,
        }

    def train_or_resume(self, config: RnnExperimentConfig, bundle: DataBundle, *, verbose: bool = True) -> Dict[str, Any]:
        signature = rnn_experiment_signature(config)
        existing = self.registry.get(signature)
        cached_model_path = self._resolve_model_path(config, signature, existing)
        if existing and existing["status"] == "completed" and existing["metrics"] and self._artifact_is_consistent(config, signature, cached_model_path):
            payload = dict(existing["metrics"])
            payload["cached"] = True
            payload["signature"] = signature
            payload["model_path"] = str(cached_model_path)
            payload["artifact_consistent"] = True
            return payload

        best_model_path = self.paths.model_path(config, signature)
        last_model_path = self.paths.model_last_path(config, signature)
        training_state_path = self.paths.model_training_state_path(config, signature)
        optimizer_state_path = self.paths.model_optimizer_state_path(config, signature)
        model_manifest_path = self.paths.model_manifest_path(config, signature)
        run_dir = self.paths.run_dir(config, signature)
        artifacts = RunArtifacts(run_dir)

        resume_entry, resume_state = self._resolve_resume_state(config)
        resumed_from_signature = str(resume_entry["rnn_experiment_signature"]) if resume_entry else None
        initial_epoch = int(resume_state.get("trained_epochs", 0)) if resume_state else 0

        self.registry.reserve(
            signature,
            config,
            best_model_path=best_model_path,
            last_model_path=last_model_path,
            training_state_path=training_state_path,
            optimizer_state_path=optimizer_state_path,
            model_manifest_path=model_manifest_path,
            run_dir=run_dir,
            resumed_from_signature=resumed_from_signature,
            initial_epoch=initial_epoch,
        )

        if resume_entry:
            previous_run_dir = Path(str(resume_entry.get("run_dir") or ""))
            artifacts.seed_history_from(previous_run_dir / "history.csv")
            self._copy_if_exists(resume_entry.get("best_model_path") or resume_entry.get("model_path"), best_model_path)
            self._copy_if_exists(resume_entry.get("last_model_path"), last_model_path)
            self._copy_if_exists(resume_entry.get("optimizer_state_path"), optimizer_state_path)
            self._write_training_state(
                config,
                signature,
                training_state_path,
                best_model_path,
                last_model_path,
                optimizer_state_path,
                best_search_metric_acc=float(resume_state.get("best_search_metric_acc", resume_state.get("best_test_acc", -1.0))),
                best_search_metric_loss=float(resume_state.get("best_search_metric_loss", resume_state.get("best_test_loss", float("inf")))),
                best_val_acc=float(resume_state.get("best_val_acc", resume_state.get("best_test_acc", -1.0))),
                best_val_loss=float(resume_state.get("best_val_loss", resume_state.get("best_test_loss", float("inf")))),
                best_test_acc=float(resume_state.get("best_test_acc", -1.0)),
                best_test_loss=float(resume_state.get("best_test_loss", float("inf"))),
                best_epoch=int(resume_state.get("best_epoch", 0)),
                initial_epoch=initial_epoch,
                trained_epochs=initial_epoch,
                resumed_from_signature=resumed_from_signature,
                current_learning_rate=float(resume_state.get("current_learning_rate", config.runtime.learning_rate)) if resume_state else config.runtime.learning_rate,
                reduce_lr_plateau_best=float(resume_state.get("reduce_lr_plateau_best", resume_state.get("best_test_loss", float("inf")))) if resume_state else None,
                reduce_lr_plateau_bad_epochs=int(resume_state.get("reduce_lr_plateau_bad_epochs", 0)) if resume_state else 0,
            )

        model = loss_fn = optimizer = None
        started = time.time()
        try:
            fresh_model, loss_fn, optimizer = build_rnn_model(config, bundle.metadata.num_features, bundle.metadata.num_classes)
            optimizer_state_restored = False
            if resume_entry:
                release_memory(fresh_model)
                model = tf.keras.models.load_model(Path(str(resume_entry["last_model_path"])), compile=False)
                optimizer_state_restored = restore_optimizer_state(optimizer_state_path, optimizer, model.trainable_variables)
                self._resume_learning_rate(optimizer, resume_state)
            else:
                model = fresh_model

            metrics = self._train_epochs(
                config,
                bundle,
                model,
                loss_fn,
                optimizer,
                artifacts,
                best_model_path,
                last_model_path,
                training_state_path,
                optimizer_state_path,
                signature=signature,
                start_epoch=initial_epoch,
                resume_state=resume_state,
                resumed_from_signature=resumed_from_signature,
                verbose=verbose,
            )

            final_state = self._load_training_state(training_state_path) or {}
            self._write_model_manifest(
                config,
                signature,
                best_model_path,
                last_model_path,
                training_state_path,
                optimizer_state_path,
                resumed_from_signature=resumed_from_signature,
                current_learning_rate=float(final_state.get("current_learning_rate", get_optimizer_learning_rate(optimizer))),
                reduce_lr_plateau_best=(float(final_state["reduce_lr_plateau_best"]) if final_state.get("reduce_lr_plateau_best") is not None else None),
                reduce_lr_plateau_bad_epochs=int(final_state.get("reduce_lr_plateau_bad_epochs", 0)),
            )
            metrics["cached"] = False
            metrics["signature"] = signature
            metrics["rnn_data_signature"] = rnn_data_signature(config)
            metrics["rnn_architecture_signature"] = rnn_architecture_signature(config)
            metrics["rnn_runtime_signature"] = rnn_runtime_signature(config)
            metrics["rnn_resume_signature"] = rnn_resume_signature(config)
            metrics["model_path"] = str(best_model_path)
            metrics["best_model_path"] = str(best_model_path)
            metrics["last_model_path"] = str(last_model_path)
            metrics["training_state_path"] = str(training_state_path)
            metrics["optimizer_state_path"] = str(optimizer_state_path)
            metrics["artifact_consistent"] = True
            metrics["runtime_seconds"] = time.time() - started
            metrics["memory_mb_end"] = process_memory_mb()
            metrics["resumed_from_signature"] = resumed_from_signature
            metrics["resumed"] = bool(resume_entry)
            metrics["optimizer_state_restored"] = bool(optimizer_state_restored)
            metrics["resume_mode"] = "last_model_and_optimizer_state" if optimizer_state_restored else ("last_model_weights_only" if resume_entry else None)

            artifacts.write_summary({"config": config.to_dict(), "metrics": metrics})
            artifacts.write_manifest(
                self._artifact_manifest_payload(
                    config,
                    signature,
                    best_model_path,
                    last_model_path,
                    training_state_path,
                    optimizer_state_path,
                    resumed_from_signature=resumed_from_signature,
                    current_learning_rate=metrics.get("current_learning_rate"),
                    reduce_lr_plateau_best=metrics.get("reduce_lr_plateau_best"),
                    reduce_lr_plateau_bad_epochs=metrics.get("reduce_lr_plateau_bad_epochs", 0),
                )
            )
            self.registry.complete(
                signature,
                metrics,
                best_model_path=best_model_path,
                last_model_path=last_model_path,
                training_state_path=training_state_path,
                optimizer_state_path=optimizer_state_path,
                model_manifest_path=model_manifest_path,
                run_dir=run_dir,
                resumed_from_signature=resumed_from_signature,
                initial_epoch=metrics["initial_epoch"],
                final_epoch=metrics["final_epoch"],
            )
            return metrics
        except Exception as exc:
            self.registry.fail(signature, str(exc))
            raise
        finally:
            release_memory(model, loss_fn, optimizer)

    def evaluate(self, config: RnnExperimentConfig, bundle: DataBundle) -> Dict[str, Any]:
        signature = rnn_experiment_signature(config)
        existing = self.registry.get(signature)
        model_path = self._resolve_model_path(config, signature, existing)
        if not self._artifact_is_consistent(config, signature, model_path):
            raise FileNotFoundError(
                f"No existe un artefacto consistente para {signature}. Ruta esperada: {model_path}. "
                "Si el registro existe pero el fichero fue borrado o pertenece a otra configuración, vuelve a entrenar."
            )
        model = tf.keras.models.load_model(model_path, compile=False)
        try:
            loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
            val_metrics = self._evaluate_dataset(config, bundle.val_ds, model, loss_fn, "val")
            test_metrics = self._evaluate_dataset(config, bundle.test_ds, model, loss_fn, "test")
            return {**val_metrics, **test_metrics, "signature": signature, "model_path": str(model_path)}
        finally:
            release_memory(model)
