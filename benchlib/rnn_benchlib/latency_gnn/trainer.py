from __future__ import annotations

import json
import math
import random
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Mapping, Sequence

# Keep TensorFlow sparse-gradient densification warnings from flooding the
# batch progress output. These warnings are noisy here and do not indicate
# a correctness problem in the current training pipeline.
warnings.filterwarnings(
    "ignore",
    message=r"Converting sparse IndexedSlices.*",
    category=UserWarning,
    module=r"tensorflow\.python\.framework\.indexed_slices",
)
warnings.filterwarnings(
    "ignore",
    message=r"Converting sparse IndexedSlices.*",
    category=UserWarning,
)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import mixed_precision as keras_mixed_precision

from rnn_benchlib.latency_gnn.artifacts import checkpoints_dir, eval_dir, run_dir
from rnn_benchlib.latency_gnn.evaluate import collect_predictions, compute_metrics, evaluate_model
from rnn_benchlib.latency_gnn.featurization import GRAPH_CAT_NAMES, LatencyGnnFeaturizer
from rnn_benchlib.latency_gnn.losses import MultiTargetLatencyLoss
from rnn_benchlib.latency_gnn.models import HeteroLatencyPredictor, HeteroLatencyPredictorConfig


@dataclass
class OptimizerConfig:
    name: str = "adam"
    learning_rate: float = 1e-3
    beta_1: float = 0.9
    beta_2: float = 0.999
    epsilon: float = 1e-7
    amsgrad: bool = False
    momentum: float = 0.0
    nesterov: bool = False
    rho: float = 0.9
    centered: bool = False
    clipnorm: float | None = None
    clipvalue: float | None = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrainingConfig:
    epochs: int = 40
    batch_size: int = 8
    weight_decay: float = 1e-4
    shuffle_train: bool = True
    device: str = "gpu"
    gpu_index: int = 0
    memory_growth: bool = True
    mixed_precision: bool = True
    enable_xla: bool = False
    batch_progress: bool = True
    batch_log_interval: int = 1
    val_interval_epochs: int = 1
    seed: int = 1234

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LatencyGnnTrainer:
    def __init__(
        self,
        train_config: TrainingConfig | None = None,
        model_config: HeteroLatencyPredictorConfig | None = None,
        optimizer_config: OptimizerConfig | None = None,
    ) -> None:
        self.train_config = train_config or TrainingConfig()
        self.model_config = model_config or HeteroLatencyPredictorConfig()
        self.optimizer_config = optimizer_config or OptimizerConfig()

    def fit(self, samples_by_split: Mapping[str, Sequence], output_root: str | Path, lot_id: str, dataset_id: str, run_id: str, run_name: str, split_payload: Mapping[str, Any]) -> Dict[str, Any]:
        out_dir = run_dir(output_root, lot_id, dataset_id, run_id)
        ckpt_dir = checkpoints_dir(output_root, lot_id, dataset_id, run_id)
        train_samples = list(samples_by_split.get("train", []))
        val_samples = list(samples_by_split.get("val", []))
        test_samples = list(samples_by_split.get("test", []))
        featurizer = LatencyGnnFeaturizer.build(train_samples if train_samples else train_samples + val_samples + test_samples)
        encoded = {name: featurizer.encode_many(list(rows)) for name, rows in samples_by_split.items()}
        _set_seeds(self.train_config.seed)
        device_name = _configure_tf_runtime(
            device=self.train_config.device,
            gpu_index=self.train_config.gpu_index,
            memory_growth=self.train_config.memory_growth,
            mixed_precision_enabled=self.train_config.mixed_precision,
            enable_xla=self.train_config.enable_xla,
        )
        model = HeteroLatencyPredictor(
            vocab_sizes={name: len(vocab) for name, vocab in featurizer.spec.vocabs.items()},
            op_num_dim=featurizer.spec.op_num_dim,
            tensor_num_dim=featurizer.spec.tensor_num_dim,
            graph_num_dim=featurizer.spec.graph_num_dim,
            config=self.model_config,
        )
        target_names = list(featurizer.spec.targets)
        wall_index = target_names.index("latency_clip_e2e_wall_ms")
        loss_fn = MultiTargetLatencyLoss(target_names=target_names)
        base_optimizer = _build_optimizer(self.optimizer_config)
        use_loss_scaling = bool(self.train_config.mixed_precision and device_name.startswith("/GPU"))
        optimizer = keras_mixed_precision.LossScaleOptimizer(base_optimizer) if use_loss_scaling else base_optimizer
        history_rows: List[Dict[str, Any]] = []
        best_val = float("inf")
        best_epoch = -1
        best_weights: List[np.ndarray] | None = None
        best_epoch_record: Dict[str, Any] | None = None
        last_val_wall = float("nan")

        @tf.function(reduce_retracing=True)
        def _compiled_train_step(batch: Mapping[str, Any]) -> tuple[tf.Tensor, tf.Tensor]:
            with tf.GradientTape() as tape:
                pred = model(batch, training=True)
                targets = {name: tf.cast(batch["targets"][:, idx], tf.float32) for idx, name in enumerate(target_names)}
                batch_loss, _ = loss_fn(predictions=pred, targets=targets, sample_weight=batch["sample_weight"], return_metrics=False)
                if self.train_config.weight_decay > 0:
                    batch_loss = batch_loss + self.train_config.weight_decay * _l2_penalty(model.trainable_variables)
                if use_loss_scaling:
                    scaled_loss = optimizer.get_scaled_loss(batch_loss)
            if use_loss_scaling:
                scaled_grads = tape.gradient(scaled_loss, model.trainable_variables)
                grads = optimizer.get_unscaled_gradients(scaled_grads)
            else:
                grads = tape.gradient(batch_loss, model.trainable_variables)
            grads_and_vars = [(g, v) for g, v in zip(grads, model.trainable_variables) if g is not None]
            optimizer.apply_gradients(grads_and_vars)
            pred_matrix = tf.stack([tf.cast(tf.reshape(pred[name], [-1]), tf.float32) for name in target_names], axis=1)
            return batch_loss, pred_matrix

        @tf.function(reduce_retracing=True)
        def _compiled_eval_step(batch: Mapping[str, Any]) -> tuple[tf.Tensor, tf.Tensor]:
            pred = model(batch, training=False)
            targets = {name: tf.cast(batch["targets"][:, idx], tf.float32) for idx, name in enumerate(target_names)}
            batch_loss, _ = loss_fn(predictions=pred, targets=targets, sample_weight=batch["sample_weight"], return_metrics=False)
            pred_matrix = tf.stack([tf.cast(tf.reshape(pred[name], [-1]), tf.float32) for name in target_names], axis=1)
            return batch_loss, pred_matrix

        def _summarize_epoch_from_arrays(loss_sum: float, sample_count: int, y_true: Dict[str, List[float]], y_pred: Dict[str, List[float]]) -> Dict[str, Any]:
            targets_payload: Dict[str, Dict[str, float]] = {}
            for name in target_names:
                targets_payload[name] = compute_metrics(y_true.get(name, []), y_pred.get(name, []))
            wall_metrics = targets_payload.get("latency_clip_e2e_wall_ms", {})
            return {
                "loss": float(loss_sum / max(1, sample_count)),
                "targets": targets_payload,
                "wall_mape": float(wall_metrics.get("mape", 0.0)),
                "wall_mae": float(wall_metrics.get("mae", 0.0)),
                "wall_rmse_log": float(wall_metrics.get("rmse_log", 0.0)),
                "wall_spearman": float(wall_metrics.get("spearman", 0.0)),
            }

        with tf.device(device_name):
            train_batches = _batchify_graphs(encoded["train"], self.train_config.batch_size, shuffle=True)
            if train_batches:
                model(train_batches[0], training=False)
            val_batches_template = _batchify_graphs(encoded["val"], self.train_config.batch_size, shuffle=False)
            for epoch in range(1, self.train_config.epochs + 1):
                epoch_t0 = perf_counter()
                train_loss_sum = 0.0
                train_sample_count = 0
                train_true = {name: [] for name in target_names}
                train_pred = {name: [] for name in target_names}
                train_batches = _batchify_graphs(encoded["train"], self.train_config.batch_size, shuffle=self.train_config.shuffle_train)
                progbar = None
                if self.train_config.batch_progress and train_batches:
                    progbar = tf.keras.utils.Progbar(target=len(train_batches), verbose=1, stateful_metrics=["loss", "train_mape", "val_mape"])
                for batch_index, batch in enumerate(train_batches, start=1):
                    batch_loss, pred_matrix = _compiled_train_step(batch)
                    pred_np = np.asarray(pred_matrix.numpy(), dtype=np.float32)
                    true_np = np.asarray(batch["targets"], dtype=np.float32)
                    batch_n = int(true_np.shape[0])
                    train_loss_sum += float(batch_loss.numpy()) * batch_n
                    train_sample_count += batch_n
                    for idx, name in enumerate(target_names):
                        train_true[name].extend(true_np[:, idx].astype(np.float64).tolist())
                        train_pred[name].extend(pred_np[:, idx].astype(np.float64).tolist())
                    if progbar is not None and (
                        batch_index % max(1, int(self.train_config.batch_log_interval)) == 0 or batch_index == len(train_batches)
                    ):
                        current_train = _summarize_epoch_from_arrays(train_loss_sum, train_sample_count, train_true, train_pred)
                        progbar.update(batch_index, values=[
                            ("loss", current_train["loss"]),
                            ("train_mape", current_train["wall_mape"]),
                            ("val_mape", 0.0 if math.isnan(last_val_wall) else last_val_wall),
                        ])
                train_summary = _summarize_epoch_from_arrays(train_loss_sum, train_sample_count, train_true, train_pred)

                val_summary = {
                    "loss": float("nan"),
                    "targets": {name: {"count": 0.0, "mape": float("nan"), "mae": float("nan"), "rmse_log": float("nan"), "spearman": float("nan")} for name in target_names},
                    "wall_mape": float("nan"),
                    "wall_mae": float("nan"),
                    "wall_rmse_log": float("nan"),
                    "wall_spearman": float("nan"),
                }
                if val_batches_template and (epoch % max(1, int(self.train_config.val_interval_epochs)) == 0 or epoch == self.train_config.epochs):
                    val_loss_sum = 0.0
                    val_sample_count = 0
                    val_true = {name: [] for name in target_names}
                    val_pred = {name: [] for name in target_names}
                    for batch in val_batches_template:
                        batch_loss, pred_matrix = _compiled_eval_step(batch)
                        pred_np = np.asarray(pred_matrix.numpy(), dtype=np.float32)
                        true_np = np.asarray(batch["targets"], dtype=np.float32)
                        batch_n = int(true_np.shape[0])
                        val_loss_sum += float(batch_loss.numpy()) * batch_n
                        val_sample_count += batch_n
                        for idx, name in enumerate(target_names):
                            val_true[name].extend(true_np[:, idx].astype(np.float64).tolist())
                            val_pred[name].extend(pred_np[:, idx].astype(np.float64).tolist())
                    val_summary = _summarize_epoch_from_arrays(val_loss_sum, val_sample_count, val_true, val_pred)
                    last_val_wall = val_summary["wall_mape"]
                elif not math.isnan(last_val_wall):
                    val_summary["wall_mape"] = last_val_wall

                epoch_payload = {
                    "epoch": epoch,
                    "epoch_sec": perf_counter() - epoch_t0,
                    "batches": len(train_batches),
                    "train": train_summary,
                    "val": val_summary,
                }
                history_rows.append(epoch_payload)
                print(
                    f"[train-gnn] epoch={epoch}/{self.train_config.epochs} "
                    f"train_loss={train_summary['loss']:.6f} "
                    f"val_loss={val_summary['loss']:.6f} "
                    f"train_wall_mape={train_summary['wall_mape']:.4f} "
                    f"val_wall_mape={val_summary['wall_mape']:.4f} "
                    f"epoch_sec={epoch_payload['epoch_sec']:.2f}",
                    flush=True,
                )
                self._save_weights(ckpt_dir / "last.weights.h5", model)
                current_val = val_summary["wall_mape"]
                if not math.isnan(current_val) and current_val <= best_val:
                    best_val = current_val
                    best_epoch = epoch
                    best_weights = model.get_weights()
                    best_epoch_record = json.loads(json.dumps(epoch_payload))
                    self._save_weights(ckpt_dir / "best.weights.h5", model)
            if best_weights is not None:
                model.set_weights(best_weights)
            test_metrics = evaluate_model(model, encoded["test"], featurizer.spec.targets, device=device_name) if encoded["test"] else {}
            predictions = collect_predictions(model, encoded["test"], featurizer.spec.targets, device=device_name) if encoded["test"] else {}
        config_payload = {
            "run_id": run_id,
            "run_name": run_name,
            "lot_id": lot_id,
            "dataset_id": dataset_id,
            "benchmark_id": split_payload.get("benchmark_id"),
            "split_id": split_payload["split_id"],
            "framework": "tensorflow",
            "framework_version": tf.__version__,
            "feature_spec": featurizer.spec.to_dict(),
            "model_config": self.model_config.to_dict(),
            "train_config": self.train_config.to_dict(),
            "optimizer_config": self.optimizer_config.to_dict(),
        }
        (out_dir / "config.json").write_text(json.dumps(config_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        run_payload = {
            "schema_version": "rnn_latency_gnn_run_v2",
            **config_payload,
        }
        (out_dir / "run.json").write_text(json.dumps(run_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        history_payload = {
            "schema_version": "rnn_latency_gnn_history_v2",
            "target_names": target_names,
            "epochs": history_rows,
        }
        (out_dir / "history.json").write_text(json.dumps(history_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        final_epoch_record = history_rows[-1] if history_rows else None
        metrics_payload = {
            "schema_version": "rnn_latency_gnn_metrics_v2",
            "best_epoch": best_epoch,
            "best_val_wall_mape": best_val,
            "train_count": len(encoded["train"]),
            "val_count": len(encoded["val"]),
            "test_count": len(encoded["test"]),
            "final_epoch": final_epoch_record,
            "best_epoch_metrics": best_epoch_record,
            "test_metrics": test_metrics,
        }
        (out_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        _export_training_figures(out_dir, history_rows, target_names)
        eval_payload = {
            "targets": {name: {"y_true": values["y_true"], "y_pred": values["y_pred"]} for name, values in predictions.items()},
        }
        e_dir = eval_dir(output_root, lot_id, dataset_id, run_id)
        (e_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        (e_dir / "predictions.json").write_text(json.dumps(eval_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return {**metrics_payload, "run_dir": str(out_dir), "checkpoint": str(ckpt_dir / "best.weights.h5")}

    def _save_weights(self, path: Path, model: HeteroLatencyPredictor) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        model.save_weights(path, overwrite=True)



def _export_training_figures(out_dir: Path, history_rows: Sequence[Mapping[str, Any]], target_names: Sequence[str]) -> None:
    if not history_rows:
        return
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    epochs = [int(row.get("epoch", idx + 1)) for idx, row in enumerate(history_rows)]

    train_loss = [float(row.get("train", {}).get("loss", float("nan"))) for row in history_rows]
    val_loss = [float(row.get("val", {}).get("loss", float("nan"))) for row in history_rows]
    _plot_series(
        figures_dir / "loss.png",
        title="Loss over epochs",
        xlabel="Epoch",
        ylabel="Loss",
        series={"train_loss": train_loss, "val_loss": val_loss},
        epochs=epochs,
    )

    overall_metrics = [
        ("wall_mape", "MAPE"),
        ("wall_mae", "MAE"),
        ("wall_rmse_log", "RMSE log"),
        ("wall_spearman", "Spearman"),
    ]
    for metric_key, metric_label in overall_metrics:
        _plot_series(
            figures_dir / f"{metric_key}.png",
            title=f"Wall {metric_label} over epochs",
            xlabel="Epoch",
            ylabel=metric_label,
            series={
                f"train_{metric_key}": [float(row.get("train", {}).get(metric_key, float("nan"))) for row in history_rows],
                f"val_{metric_key}": [float(row.get("val", {}).get(metric_key, float("nan"))) for row in history_rows],
            },
            epochs=epochs,
        )

    for target_name in target_names:
        alias = _target_alias(target_name)
        target_metrics = [
            ("mape", "MAPE"),
            ("mae", "MAE"),
            ("rmse_log", "RMSE log"),
            ("spearman", "Spearman"),
        ]
        for metric_key, metric_label in target_metrics:
            train_values = [float(row.get("train", {}).get("targets", {}).get(target_name, {}).get(metric_key, float("nan"))) for row in history_rows]
            val_values = [float(row.get("val", {}).get("targets", {}).get(target_name, {}).get(metric_key, float("nan"))) for row in history_rows]
            if all(math.isnan(v) for v in train_values) and all(math.isnan(v) for v in val_values):
                continue
            _plot_series(
                figures_dir / f"{alias}_{metric_key}.png",
                title=f"{alias} {metric_label} over epochs",
                xlabel="Epoch",
                ylabel=metric_label,
                series={f"train_{alias}_{metric_key}": train_values, f"val_{alias}_{metric_key}": val_values},
                epochs=epochs,
            )


def _target_alias(target_name: str) -> str:
    mapping = {
        "latency_clip_encoder_ms": "encoder",
        "latency_clip_bridge_ms": "bridge",
        "latency_clip_head_ms": "head",
        "latency_clip_e2e_wall_ms": "wall",
    }
    if target_name in mapping:
        return mapping[target_name]
    alias = str(target_name)
    for prefix in ("latency_clip_", "latency_", "clip_"):
        if alias.startswith(prefix):
            alias = alias[len(prefix):]
    for suffix in ("_ms",):
        if alias.endswith(suffix):
            alias = alias[: -len(suffix)]
    return alias


def _build_optimizer(config: OptimizerConfig) -> tf.keras.optimizers.Optimizer:
    name = str(config.name).strip().lower()
    common: Dict[str, Any] = {"learning_rate": float(config.learning_rate)}
    if config.clipnorm is not None:
        common["clipnorm"] = float(config.clipnorm)
    if config.clipvalue is not None:
        common["clipvalue"] = float(config.clipvalue)

    if name == "adam":
        return tf.keras.optimizers.Adam(
            beta_1=float(config.beta_1),
            beta_2=float(config.beta_2),
            epsilon=float(config.epsilon),
            amsgrad=bool(config.amsgrad),
            **common,
        )
    if name == "sgd":
        return tf.keras.optimizers.SGD(
            momentum=float(config.momentum),
            nesterov=bool(config.nesterov),
            **common,
        )
    if name == "rmsprop":
        return tf.keras.optimizers.RMSprop(
            rho=float(config.rho),
            momentum=float(config.momentum),
            epsilon=float(config.epsilon),
            centered=bool(config.centered),
            **common,
        )
    raise ValueError(f"optimizador GNN no soportado: {config.name}")

def _batchify_graphs(items: Sequence[Mapping[str, Any]], batch_size: int, shuffle: bool) -> List[Dict[str, Any]]:
    batch_size = max(1, int(batch_size))
    rows = list(items)
    if shuffle:
        random.shuffle(rows)
    return [_make_disjoint_batch(rows[i : i + batch_size]) for i in range(0, len(rows), batch_size)]


def _make_disjoint_batch(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not rows:
        raise ValueError("rows must not be empty")
    op_cat=[]; op_family=[]; op_component=[]; op_numeric=[]
    tensor_dtype=[]; tensor_component=[]; tensor_quant=[]; tensor_role=[]; tensor_numeric=[]
    encoder_mask=[]; head_mask=[]; bridge_mask=[]
    op_graph_index=[]; tensor_graph_index=[]
    edge_ot=[]; edge_to=[]; edge_oo=[]
    graph_numeric=[]; targets=[]; sample_weight=[]
    graph_cat={name: [] for name in GRAPH_CAT_NAMES}
    op_offset=0
    tensor_offset=0
    for graph_idx, row in enumerate(rows):
        n_op = int(np.asarray(row["op_cat"]).shape[0])
        n_tensor = int(np.asarray(row["tensor_dtype"]).shape[0])
        op_cat.append(np.asarray(row["op_cat"], dtype=np.int32))
        op_family.append(np.asarray(row["op_family"], dtype=np.int32))
        op_component.append(np.asarray(row["op_component"], dtype=np.int32))
        op_numeric.append(np.asarray(row["op_numeric"], dtype=np.float32))
        tensor_dtype.append(np.asarray(row["tensor_dtype"], dtype=np.int32))
        tensor_component.append(np.asarray(row["tensor_component"], dtype=np.int32))
        tensor_quant.append(np.asarray(row["tensor_quant"], dtype=np.int32))
        tensor_role.append(np.asarray(row["tensor_role"], dtype=np.int32))
        tensor_numeric.append(np.asarray(row["tensor_numeric"], dtype=np.float32))
        encoder_mask.append(np.asarray(row["encoder_op_mask"], dtype=np.bool_))
        head_mask.append(np.asarray(row["head_op_mask"], dtype=np.bool_))
        bridge_mask.append(np.asarray(row["bridge_tensor_mask"], dtype=np.bool_))
        op_graph_index.append(np.full((n_op,), graph_idx, dtype=np.int32))
        tensor_graph_index.append(np.full((n_tensor,), graph_idx, dtype=np.int32))
        e = np.asarray(row["edge_index_op_to_tensor"], dtype=np.int32)
        if e.size:
            edge_ot.append(np.vstack([e[0] + op_offset, e[1] + tensor_offset]))
        e = np.asarray(row["edge_index_tensor_to_op"], dtype=np.int32)
        if e.size:
            edge_to.append(np.vstack([e[0] + tensor_offset, e[1] + op_offset]))
        e = np.asarray(row["edge_index_op_to_op"], dtype=np.int32)
        if e.size:
            edge_oo.append(np.vstack([e[0] + op_offset, e[1] + op_offset]))
        graph_numeric.append(np.asarray(row["graph_numeric"], dtype=np.float32))
        for name in GRAPH_CAT_NAMES:
            graph_cat[name].append(np.asarray(row["graph_cat"][name], dtype=np.int32).reshape(()))
        targets.append(np.asarray(row["targets"], dtype=np.float32))
        sample_weight.append(np.asarray(row["sample_weight"], dtype=np.float32).reshape(()))
        op_offset += n_op
        tensor_offset += n_tensor

    def _cat(seq, dtype, axis=0):
        return np.concatenate(seq, axis=axis).astype(dtype, copy=False) if len(seq) > 1 else np.asarray(seq[0], dtype=dtype)

    return {
        "op_cat": _cat(op_cat, np.int32),
        "op_family": _cat(op_family, np.int32),
        "op_component": _cat(op_component, np.int32),
        "op_numeric": _cat(op_numeric, np.float32),
        "tensor_dtype": _cat(tensor_dtype, np.int32),
        "tensor_component": _cat(tensor_component, np.int32),
        "tensor_quant": _cat(tensor_quant, np.int32),
        "tensor_role": _cat(tensor_role, np.int32),
        "tensor_numeric": _cat(tensor_numeric, np.float32),
        "graph_numeric": np.stack(graph_numeric, axis=0).astype(np.float32, copy=False),
        "graph_cat": {name: np.asarray(values, dtype=np.int32) for name, values in graph_cat.items()},
        "encoder_op_mask": _cat(encoder_mask, np.bool_),
        "head_op_mask": _cat(head_mask, np.bool_),
        "bridge_tensor_mask": _cat(bridge_mask, np.bool_),
        "op_graph_index": _cat(op_graph_index, np.int32),
        "tensor_graph_index": _cat(tensor_graph_index, np.int32),
        "edge_index_op_to_tensor": np.concatenate(edge_ot, axis=1).astype(np.int32, copy=False) if edge_ot else np.zeros((2, 0), dtype=np.int32),
        "edge_index_tensor_to_op": np.concatenate(edge_to, axis=1).astype(np.int32, copy=False) if edge_to else np.zeros((2, 0), dtype=np.int32),
        "edge_index_op_to_op": np.concatenate(edge_oo, axis=1).astype(np.int32, copy=False) if edge_oo else np.zeros((2, 0), dtype=np.int32),
        "targets": np.stack(targets, axis=0).astype(np.float32, copy=False),
        "sample_weight": np.asarray(sample_weight, dtype=np.float32),
    }


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _configure_tf_runtime(
    device: str,
    gpu_index: int = 0,
    memory_growth: bool = True,
    mixed_precision_enabled: bool = True,
    enable_xla: bool = False,
) -> str:
    try:
        tf.config.optimizer.set_jit(bool(enable_xla))
    except Exception:
        pass
    value = str(device or "gpu").strip().lower()
    if value.startswith("/"):
        _set_precision_policy(device_name=value, mixed_precision_enabled=mixed_precision_enabled)
        return device
    if value == "cpu":
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass
        _set_precision_policy(device_name="/CPU:0", mixed_precision_enabled=False)
        return "/CPU:0"
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        _set_precision_policy(device_name="/CPU:0", mixed_precision_enabled=False)
        return "/CPU:0"
    index = max(0, min(int(gpu_index), len(gpus) - 1))
    selected = gpus[index]
    try:
        tf.config.set_visible_devices([selected], "GPU")
    except Exception:
        pass
    visible_gpus = tf.config.list_physical_devices("GPU")
    if memory_growth:
        try:
            for gpu in visible_gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass
    effective_mixed_precision = bool(mixed_precision_enabled)
    if effective_mixed_precision:
        try:
            details = tf.config.experimental.get_device_details(visible_gpus[0]) if visible_gpus else {}
            cc = details.get("compute_capability")
            if cc and tuple(cc) < (7, 0):
                print(f"[train-gnn] mixed_precision disabled automatically for GPU compute capability {cc[0]}.{cc[1]} (< 7.0)", flush=True)
                effective_mixed_precision = False
        except Exception:
            pass
    _set_precision_policy(device_name="/GPU:0", mixed_precision_enabled=effective_mixed_precision)
    return "/GPU:0"


def _set_precision_policy(device_name: str, mixed_precision_enabled: bool) -> None:
    if mixed_precision_enabled and str(device_name).startswith("/GPU"):
        keras_mixed_precision.set_global_policy("mixed_float16")
    else:
        keras_mixed_precision.set_global_policy("float32")


def _evaluate_wall_mape_batched(compiled_eval_step, batched_samples: Sequence[Mapping[str, Any]]) -> float:
    if not batched_samples:
        return 0.0
    total_rel = 0.0
    total_count = 0
    for batch in batched_samples:
        rel = compiled_eval_step(batch)
        count = int(np.asarray(batch["targets"]).shape[0])
        total_rel += float(rel.numpy()) * count
        total_count += count
    return float(total_rel / max(1, total_count))


class _null_device:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


def _resolve_tf_device(device: str) -> str:
    value = str(device or "gpu").strip().lower()
    if value.startswith("/"):
        return device
    if value.startswith("gpu"):
        suffix = value.split(":", 1)[1] if ":" in value else "0"
        return f"/GPU:{suffix}"
    return "/CPU:0"


def _l2_penalty(variables: Sequence[tf.Variable]) -> tf.Tensor:
    if not variables:
        return tf.constant(0.0, dtype=tf.float32)
    penalties = []
    for var in variables:
        name = var.name.lower()
        # Embedding gradients are sparse (IndexedSlices). Including them in an
        # explicit L2 penalty inside the tape forces densification and slows
        # training down a lot, especially on GPU. Keep weight decay on dense
        # kernels only.
        if "bias" in name or "embedding" in name or "/embeddings:" in name:
            continue
        penalties.append(tf.reduce_sum(tf.square(var)))
    if not penalties:
        return tf.constant(0.0, dtype=tf.float32)
    return tf.add_n(penalties)
