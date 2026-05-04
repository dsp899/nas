from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping

import tensorflow as tf

from rnn_benchlib.latency_gnn.featurization import GRAPH_CAT_NAMES


@dataclass
class HeteroLatencyPredictorConfig:
    hidden_dim: int = 96
    graph_hidden_dim: int = 64
    num_layers: int = 3
    dropout: float = 0.1
    embedding_dim: int = 16
    graph_embedding_dim: int = 8
    readout_dim: int = 64

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "HeteroLatencyPredictorConfig":
        keys = cls().__dict__.keys()
        return cls(**{key: payload[key] for key in keys if key in payload})


class MessagePassingLayer(tf.keras.layers.Layer):
    def __init__(self, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.op_update = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(hidden_dim, activation="relu"),
                tf.keras.layers.Dropout(dropout),
                tf.keras.layers.Dense(hidden_dim),
            ]
        )
        self.tensor_update = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(hidden_dim, activation="relu"),
                tf.keras.layers.Dropout(dropout),
                tf.keras.layers.Dense(hidden_dim),
            ]
        )
        self.op_norm = tf.keras.layers.LayerNormalization()
        self.tensor_norm = tf.keras.layers.LayerNormalization()

    def call(
        self,
        op_h: tf.Tensor,
        tensor_h: tf.Tensor,
        edge_index_op_to_tensor: tf.Tensor,
        edge_index_tensor_to_op: tf.Tensor,
        edge_index_op_to_op: tf.Tensor,
        training: bool = False,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        o2o = _aggregate_messages(op_h, edge_index_op_to_op, tf.shape(op_h)[0])
        t2o = _aggregate_messages(tensor_h, edge_index_tensor_to_op, tf.shape(op_h)[0])
        o2t = _aggregate_messages(op_h, edge_index_op_to_tensor, tf.shape(tensor_h)[0])
        new_op = self.op_norm(op_h + self.op_update(tf.concat([op_h, o2o, t2o], axis=-1), training=training))
        new_tensor = self.tensor_norm(tensor_h + self.tensor_update(tf.concat([tensor_h, o2t], axis=-1), training=training))
        return new_op, new_tensor


class HeteroLatencyPredictor(tf.keras.Model):
    def __init__(self, vocab_sizes: Mapping[str, int], op_num_dim: int, tensor_num_dim: int, graph_num_dim: int, config: HeteroLatencyPredictorConfig | None = None) -> None:
        super().__init__()
        self.config = config or HeteroLatencyPredictorConfig()
        emb = self.config.embedding_dim
        gemb = self.config.graph_embedding_dim
        half = max(1, emb // 2)
        self.op_category = tf.keras.layers.Embedding(int(vocab_sizes["op_category"]), emb)
        self.op_family = tf.keras.layers.Embedding(int(vocab_sizes["op_family"]), half)
        self.op_component = tf.keras.layers.Embedding(int(vocab_sizes["op_component"]), half)
        self.tensor_dtype = tf.keras.layers.Embedding(int(vocab_sizes["tensor_dtype"]), half)
        self.tensor_component = tf.keras.layers.Embedding(int(vocab_sizes["tensor_component"]), half)
        self.tensor_quant = tf.keras.layers.Embedding(int(vocab_sizes["tensor_quant"]), half)
        self.tensor_role = tf.keras.layers.Embedding(int(vocab_sizes["tensor_role"]), half)
        self.graph_embeddings = {
            name: tf.keras.layers.Embedding(int(vocab_sizes[name]), gemb)
            for name in GRAPH_CAT_NAMES
        }
        self.op_encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(self.config.hidden_dim, activation="relu"),
            tf.keras.layers.Dense(self.config.hidden_dim),
        ])
        self.tensor_encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(self.config.hidden_dim, activation="relu"),
            tf.keras.layers.Dense(self.config.hidden_dim),
        ])
        self.layers_mp = [MessagePassingLayer(self.config.hidden_dim, dropout=self.config.dropout) for _ in range(self.config.num_layers)]
        self.graph_mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(self.config.graph_hidden_dim, activation="relu"),
            tf.keras.layers.Dropout(self.config.dropout),
            tf.keras.layers.Dense(self.config.graph_hidden_dim),
        ])
        self.readout = tf.keras.Sequential([
            tf.keras.layers.Dense(self.config.readout_dim, activation="relu"),
            tf.keras.layers.Dropout(self.config.dropout),
            tf.keras.layers.Dense(self.config.readout_dim, activation="relu"),
        ])
        self.part_head = tf.keras.layers.Dense(3)
        self.wall_residual_head = tf.keras.layers.Dense(1)
        self._op_num_dim = int(op_num_dim)
        self._tensor_num_dim = int(tensor_num_dim)
        self._graph_num_dim = int(graph_num_dim)

    def call(self, batch: Mapping[str, Any], training: bool = False) -> Dict[str, tf.Tensor]:
        compute_dtype = tf.as_dtype(self.compute_dtype or tf.keras.backend.floatx())
        op_numeric = _to_tensor(batch["op_numeric"], compute_dtype)
        tensor_numeric = _to_tensor(batch["tensor_numeric"], compute_dtype)
        op_h = self.op_encoder(
            tf.concat(
                [
                    op_numeric,
                    _cast_like(self.op_category(_to_tensor(batch["op_cat"], tf.int32)), op_numeric),
                    _cast_like(self.op_family(_to_tensor(batch["op_family"], tf.int32)), op_numeric),
                    _cast_like(self.op_component(_to_tensor(batch["op_component"], tf.int32)), op_numeric),
                ],
                axis=-1,
            ),
            training=training,
        )
        tensor_h = self.tensor_encoder(
            tf.concat(
                [
                    tensor_numeric,
                    _cast_like(self.tensor_dtype(_to_tensor(batch["tensor_dtype"], tf.int32)), tensor_numeric),
                    _cast_like(self.tensor_component(_to_tensor(batch["tensor_component"], tf.int32)), tensor_numeric),
                    _cast_like(self.tensor_quant(_to_tensor(batch["tensor_quant"], tf.int32)), tensor_numeric),
                    _cast_like(self.tensor_role(_to_tensor(batch["tensor_role"], tf.int32)), tensor_numeric),
                ],
                axis=-1,
            ),
            training=training,
        )
        e_ot = _to_tensor(batch["edge_index_op_to_tensor"], tf.int32)
        e_to = _to_tensor(batch["edge_index_tensor_to_op"], tf.int32)
        e_oo = _to_tensor(batch["edge_index_op_to_op"], tf.int32)
        for layer in self.layers_mp:
            op_h, tensor_h = layer(op_h, tensor_h, e_ot, e_to, e_oo, training=training)
        graph_numeric = _ensure_2d(_to_tensor(batch["graph_numeric"], compute_dtype))
        num_graphs = tf.shape(graph_numeric)[0]
        op_graph_index = _graph_index_for(batch.get("op_graph_index"), tf.shape(op_h)[0], num_graphs)
        tensor_graph_index = _graph_index_for(batch.get("tensor_graph_index"), tf.shape(tensor_h)[0], num_graphs)
        pooled = tf.concat(
            [
                _cast_like(_pool_mask(op_h, _to_tensor(batch["encoder_op_mask"], tf.bool), op_graph_index, num_graphs), op_h),
                _cast_like(_pool_mask(op_h, _to_tensor(batch["head_op_mask"], tf.bool), op_graph_index, num_graphs), op_h),
                _cast_like(_pool_mask(tensor_h, _to_tensor(batch["bridge_tensor_mask"], tf.bool), tensor_graph_index, num_graphs), tensor_h),
                _cast_like(
                    _pool_global(
                        tf.concat([op_h, tensor_h], axis=0),
                        tf.concat([op_graph_index, tensor_graph_index], axis=0),
                        num_graphs,
                    ),
                    op_h,
                ),
            ],
            axis=-1,
        )
        graph_cat = batch["graph_cat"]
        graph_embed = [_ensure_2d(_cast_like(self.graph_embeddings[name](_to_tensor(graph_cat[name], tf.int32)), graph_numeric)) for name in GRAPH_CAT_NAMES]
        graph_context = self.graph_mlp(tf.concat([graph_numeric] + graph_embed, axis=-1), training=training)
        hidden = self.readout(tf.concat([pooled, graph_context], axis=-1), training=training)
        part_pred = tf.nn.softplus(self.part_head(hidden))
        sum_pred = tf.reduce_sum(part_pred, axis=-1, keepdims=True)
        wall_pred = tf.nn.softplus(sum_pred + self.wall_residual_head(hidden))
        return {
            "latency_clip_encoder_ms": part_pred[..., 0:1],
            "latency_clip_bridge_ms": part_pred[..., 1:2],
            "latency_clip_head_ms": part_pred[..., 2:3],
            "latency_clip_e2e_wall_ms": wall_pred,
        }



def _ensure_2d(tensor: tf.Tensor) -> tf.Tensor:
    tensor = tf.convert_to_tensor(tensor)
    rank = tensor.shape.rank
    if rank == 2:
        return tensor
    if rank == 1 or rank == 0:
        return tf.expand_dims(tensor, axis=0)
    if rank is None:
        return tf.cond(tf.equal(tf.rank(tensor), 2), lambda: tensor, lambda: tf.expand_dims(tensor, axis=0))
    raise ValueError(f"Expected rank <= 2 tensor, got rank={rank}")



def _to_tensor(value: Any, dtype: tf.dtypes.DType) -> tf.Tensor:
    tensor = tf.convert_to_tensor(value)
    return tf.cast(tensor, dtype)



def _cast_like(tensor: tf.Tensor, reference: tf.Tensor) -> tf.Tensor:
    return tf.cast(tf.convert_to_tensor(tensor), tf.convert_to_tensor(reference).dtype)



def _aggregate_messages(src_features: tf.Tensor, edge_index: tf.Tensor, dst_size: tf.Tensor) -> tf.Tensor:
    edge_index = tf.cast(edge_index, tf.int32)
    src = edge_index[0]
    dst = edge_index[1]
    gathered = tf.gather(src_features, src)
    return tf.math.unsorted_segment_mean(gathered, dst, num_segments=dst_size)



def _graph_index_for(value: Any, num_nodes: tf.Tensor, num_graphs: tf.Tensor) -> tf.Tensor:
    if value is None:
        return tf.zeros([num_nodes], dtype=tf.int32)
    return tf.cast(tf.reshape(tf.convert_to_tensor(value), [-1]), tf.int32)



def _segment_mean_max(features: tf.Tensor, graph_index: tf.Tensor, num_graphs: tf.Tensor) -> tf.Tensor:
    feat_dim = tf.shape(features)[-1]

    def _empty() -> tf.Tensor:
        return tf.zeros([num_graphs, feat_dim * 2], dtype=features.dtype)

    def _non_empty() -> tf.Tensor:
        sums = tf.math.unsorted_segment_sum(features, graph_index, num_segments=num_graphs)
        counts = tf.math.unsorted_segment_sum(tf.ones([tf.shape(features)[0], 1], dtype=features.dtype), graph_index, num_segments=num_graphs)
        means = sums / tf.maximum(counts, 1.0)
        maxs = tf.math.unsorted_segment_max(features, graph_index, num_segments=num_graphs)
        has_items = tf.squeeze(counts > 0, axis=-1)
        maxs = tf.where(tf.expand_dims(has_items, -1), maxs, tf.zeros_like(maxs))
        return tf.concat([means, maxs], axis=-1)

    return tf.cond(tf.equal(tf.shape(features)[0], 0), _empty, _non_empty)



def _pool_mask(features: tf.Tensor, mask: tf.Tensor, graph_index: tf.Tensor, num_graphs: tf.Tensor) -> tf.Tensor:
    mask = tf.cast(mask, tf.bool)
    idx = tf.reshape(tf.where(mask), [-1])
    selected = tf.gather(features, idx)
    selected_graph_index = tf.gather(graph_index, idx)
    return _segment_mean_max(selected, selected_graph_index, num_graphs)



def _pool_global(features: tf.Tensor, graph_index: tf.Tensor, num_graphs: tf.Tensor) -> tf.Tensor:
    return _segment_mean_max(features, graph_index, num_graphs)
