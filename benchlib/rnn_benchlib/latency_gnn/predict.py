from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import tensorflow as tf

from rnn_benchlib.latency_gnn.featurization import FeatureSpecBundle, GRAPH_CAT_NAMES, LatencyGnnFeaturizer
from rnn_benchlib.latency_gnn.models import HeteroLatencyPredictor, HeteroLatencyPredictorConfig


class LoadedLatencyPredictor:
    def __init__(self, model: HeteroLatencyPredictor, featurizer: LatencyGnnFeaturizer, device: str) -> None:
        self.model = model
        self.featurizer = featurizer
        self.device = device


def build_dummy_encoded_sample(feature_spec: FeatureSpecBundle) -> Dict[str, object]:
    return {
        "op_cat": np.asarray([0], dtype=np.int32),
        "op_family": np.asarray([0], dtype=np.int32),
        "op_component": np.asarray([0], dtype=np.int32),
        "op_numeric": np.zeros((1, feature_spec.op_num_dim), dtype=np.float32),
        "tensor_dtype": np.asarray([0], dtype=np.int32),
        "tensor_component": np.asarray([0], dtype=np.int32),
        "tensor_quant": np.asarray([0], dtype=np.int32),
        "tensor_role": np.asarray([0], dtype=np.int32),
        "tensor_numeric": np.zeros((1, feature_spec.tensor_num_dim), dtype=np.float32),
        "graph_numeric": np.zeros((feature_spec.graph_num_dim,), dtype=np.float32),
        "graph_cat": {name: np.asarray(0, dtype=np.int32) for name in GRAPH_CAT_NAMES},
        "encoder_op_mask": np.asarray([True], dtype=np.bool_),
        "head_op_mask": np.asarray([False], dtype=np.bool_),
        "bridge_tensor_mask": np.asarray([False], dtype=np.bool_),
        "edge_index_op_to_tensor": np.zeros((2, 0), dtype=np.int32),
        "edge_index_tensor_to_op": np.zeros((2, 0), dtype=np.int32),
        "edge_index_op_to_op": np.zeros((2, 0), dtype=np.int32),
        "targets": np.zeros((feature_spec.targets.__len__(),), dtype=np.float32),
        "sample_weight": np.asarray(1.0, dtype=np.float32),
        "target_names": list(feature_spec.targets),
        "metadata": {},
    }


def load_run_predictor(run_dir: str | Path, device: str = "cpu") -> LoadedLatencyPredictor:
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    feature_spec = FeatureSpecBundle.from_dict(config["feature_spec"])
    featurizer = LatencyGnnFeaturizer(feature_spec)
    model = HeteroLatencyPredictor(
        vocab_sizes={name: len(vocab) for name, vocab in feature_spec.vocabs.items()},
        op_num_dim=feature_spec.op_num_dim,
        tensor_num_dim=feature_spec.tensor_num_dim,
        graph_num_dim=feature_spec.graph_num_dim,
        config=HeteroLatencyPredictorConfig.from_dict(config["model_config"]),
    )
    device_name = _resolve_tf_device(device)
    with tf.device(device_name):
        model(build_dummy_encoded_sample(feature_spec), training=False)
        model.load_weights(run_dir / "checkpoints" / "best.weights.h5")
    return LoadedLatencyPredictor(model, featurizer, device_name)


def predict_sample(predictor: LoadedLatencyPredictor, sample) -> Dict[str, float]:
    encoded = predictor.featurizer.encode_sample(sample)
    with tf.device(predictor.device):
        outputs = predictor.model(encoded, training=False)
    return {name: float(tf.reshape(value, [-1])[0].numpy()) for name, value in outputs.items()}


def _resolve_tf_device(device: str) -> str:
    value = str(device or "cpu").strip().lower()
    if value.startswith("/"):
        return device
    if value.startswith("gpu"):
        suffix = value.split(":", 1)[1] if ":" in value else "0"
        return f"/GPU:{suffix}"
    return "/CPU:0"
