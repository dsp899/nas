import numpy as np
import tensorflow as tf

from ..config.rnn_config import RNN_DEFAULTS, RnnExperimentConfig
from ..common.training import OptimizerSpec, build_keras_optimizer
from typing import Any, Dict, List, Optional, Sequence, Tuple

def get_state_spec(config: RnnExperimentConfig) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    units_list = list(config.architecture.active_units)
    for layer_index, units in enumerate(units_list):
        if config.architecture.direction == "unidirectional":
            if config.architecture.rnn == "lstm":
                entries.extend(
                    [
                        {"name": f"layer{layer_index}_h", "units": units, "kind": "h", "layer_index": layer_index, "direction": "uni"},
                        {"name": f"layer{layer_index}_c", "units": units, "kind": "c", "layer_index": layer_index, "direction": "uni"},
                    ]
                )
            else:
                entries.append(
                    {"name": f"layer{layer_index}_h", "units": units, "kind": "h", "layer_index": layer_index, "direction": "uni"}
                )
        else:
            for direction in ("fw", "bw"):
                if config.architecture.rnn == "lstm":
                    entries.extend(
                        [
                            {"name": f"layer{layer_index}_{direction}_h", "units": units, "kind": "h", "layer_index": layer_index, "direction": direction},
                            {"name": f"layer{layer_index}_{direction}_c", "units": units, "kind": "c", "layer_index": layer_index, "direction": direction},
                        ]
                    )
                else:
                    entries.append(
                        {"name": f"layer{layer_index}_{direction}_h", "units": units, "kind": "h", "layer_index": layer_index, "direction": direction}
                    )
    return entries


def zero_state_tensors(config: RnnExperimentConfig, batch_size: int, *, dtype: tf.dtypes.DType = tf.float32) -> List[tf.Tensor]:
    return [tf.zeros((batch_size, int(entry["units"])), dtype=dtype) for entry in get_state_spec(config)]


def _next_state_for_next_clip(config: RnnExperimentConfig, next_states: Sequence[tf.Tensor]) -> List[tf.Tensor]:
    state_spec = get_state_spec(config)
    if config.architecture.memory_mode == "none":
        return [tf.zeros_like(value) for value in next_states]
    if config.architecture.direction == "unidirectional":
        return [tf.identity(value) for value in next_states]
    carried: List[tf.Tensor] = []
    for entry, value in zip(state_spec, next_states):
        if entry["direction"] == "bw":
            carried.append(tf.zeros_like(value))
        else:
            carried.append(tf.identity(value))
    return carried


def _make_rnn_cell_layer(
    config: RnnExperimentConfig,
    units: int,
    layer_index: int,
    return_sequences: bool,
    *,
    go_backwards: bool = False,
    name_prefix: Optional[str] = None,
) -> tf.keras.layers.Layer:
    layer_name = name_prefix or f"{config.architecture.rnn}_layer_{layer_index}"
    common_kwargs = dict(
        units=units,
        return_sequences=return_sequences,
        return_state=True,
        go_backwards=go_backwards,
        name=layer_name,
    )
    if config.architecture.rnn == "lstm":
        return tf.keras.layers.LSTM(
            activation="tanh",
            recurrent_activation="sigmoid",
            use_bias=True,
            dropout=0.0,
            recurrent_dropout=0.0,
            **common_kwargs,
        )
    return tf.keras.layers.GRU(
        activation="tanh",
        recurrent_activation="sigmoid",
        use_bias=True,
        reset_after=True,
        dropout=0.0,
        recurrent_dropout=0.0,
        **common_kwargs,
    )


def _consume_unidirectional_layer(
    x: tf.Tensor,
    config: RnnExperimentConfig,
    units: int,
    layer_index: int,
    return_sequences: bool,
    state_inputs: List[tf.keras.Input],
    state_outputs: List[tf.Tensor],
) -> tf.Tensor:
    layer = _make_rnn_cell_layer(config=config, units=units, layer_index=layer_index, return_sequences=return_sequences, go_backwards=False)
    if config.architecture.rnn == "lstm":
        h_in = state_inputs.pop(0)
        c_in = state_inputs.pop(0)
        y, h_out, c_out = layer(x, initial_state=[h_in, c_in])
        state_outputs.extend([h_out, c_out])
    else:
        h_in = state_inputs.pop(0)
        y, h_out = layer(x, initial_state=[h_in])
        state_outputs.append(h_out)
    return y


def _consume_bidirectional_layer(
    x: tf.Tensor,
    config: RnnExperimentConfig,
    units: int,
    layer_index: int,
    return_sequences: bool,
    state_inputs: List[tf.keras.Input],
    state_outputs: List[tf.Tensor],
) -> tf.Tensor:
    forward_layer = _make_rnn_cell_layer(
        config=config,
        units=units,
        layer_index=layer_index,
        return_sequences=return_sequences,
        go_backwards=False,
        name_prefix=f"fw_{config.architecture.rnn}_layer_{layer_index}",
    )
    backward_layer = _make_rnn_cell_layer(
        config=config,
        units=units,
        layer_index=layer_index,
        return_sequences=return_sequences,
        go_backwards=True,
        name_prefix=f"bw_{config.architecture.rnn}_layer_{layer_index}",
    )
    bidi = tf.keras.layers.Bidirectional(
        layer=forward_layer,
        backward_layer=backward_layer,
        merge_mode="concat",
        name=f"bidi_{config.architecture.rnn}_layer_{layer_index}",
    )
    if config.architecture.rnn == "lstm":
        fw_h = state_inputs.pop(0)
        fw_c = state_inputs.pop(0)
        bw_h = state_inputs.pop(0)
        bw_c = state_inputs.pop(0)
        outputs = bidi(x, initial_state=[fw_h, fw_c, bw_h, bw_c])
        y, fw_h_out, fw_c_out, bw_h_out, bw_c_out = outputs
        state_outputs.extend([fw_h_out, fw_c_out, bw_h_out, bw_c_out])
    else:
        fw_h = state_inputs.pop(0)
        bw_h = state_inputs.pop(0)
        outputs = bidi(x, initial_state=[fw_h, bw_h])
        y, fw_h_out, bw_h_out = outputs
        state_outputs.extend([fw_h_out, bw_h_out])
    return y


class VideoAggregator:
    @staticmethod
    def exact_probs_from_logits(clip_logits: np.ndarray, strategy: str, num_classes: int) -> np.ndarray:
        clip_logits = np.asarray(clip_logits, dtype=np.float32)
        probs = tf.nn.softmax(tf.convert_to_tensor(clip_logits), axis=-1).numpy()
        if strategy == "average":
            return np.mean(probs, axis=0).astype(np.float32)
        if strategy == "max_prob":
            confidences = np.max(probs, axis=1)
            return probs[int(np.argmax(confidences))].astype(np.float32)
        winners = np.argmax(clip_logits, axis=1)
        counts = np.bincount(winners, minlength=num_classes).astype(np.float32)
        total = np.maximum(np.sum(counts), 1.0)
        return (counts / total).astype(np.float32)

    @staticmethod
    def surrogate_probs_from_logits(clip_logits: tf.Tensor, strategy: str) -> tf.Tensor:
        probs = tf.nn.softmax(clip_logits, axis=-1)
        if strategy in {"average", "majority"}:
            return tf.reduce_mean(probs, axis=1)
        confidences = tf.reduce_max(probs, axis=-1)
        weights = tf.nn.softmax(confidences / RNN_DEFAULTS.internal.surrogate_max_prob_temperature, axis=1)
        return tf.reduce_sum(probs * weights[:, :, tf.newaxis], axis=1)



def apply_head_from_clip_model(model: tf.keras.Model, embeddings: tf.Tensor) -> tf.Tensor:
    hidden = model.get_layer("head_hidden")(embeddings)
    return model.get_layer("clip_logits")(hidden)



def build_rnn_model(config: RnnExperimentConfig, num_features: int, num_classes: int) -> Tuple[tf.keras.Model, tf.keras.losses.Loss, tf.keras.optimizers.Optimizer]:
    clip_x = tf.keras.Input(shape=(config.data.seq, num_features), name="clip_x", dtype=tf.float32)
    state_entries = get_state_spec(config)
    state_inputs: List[tf.keras.Input] = [
        tf.keras.Input(shape=(int(entry["units"]),), name=entry["name"], dtype=tf.float32)
        for entry in state_entries
    ]

    x = clip_x
    cursor = list(state_inputs)
    state_outputs: List[tf.Tensor] = []
    units_list = list(config.architecture.active_units)
    for layer_index, units in enumerate(units_list):
        return_sequences = layer_index < (len(units_list) - 1)
        if config.architecture.direction == "unidirectional":
            x = _consume_unidirectional_layer(
                x=x,
                config=config,
                units=int(units),
                layer_index=layer_index,
                return_sequences=return_sequences,
                state_inputs=cursor,
                state_outputs=state_outputs,
            )
        else:
            x = _consume_bidirectional_layer(
                x=x,
                config=config,
                units=int(units),
                layer_index=layer_index,
                return_sequences=return_sequences,
                state_inputs=cursor,
                state_outputs=state_outputs,
            )

    clip_embedding = tf.keras.layers.Activation("linear", name="clip_embedding")(x)
    hidden = tf.keras.layers.Dense(
        int(config.architecture.head_units),
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(RNN_DEFAULTS.internal.l2_reg),
        name="head_hidden",
    )(clip_embedding)
    clip_logits = tf.keras.layers.Dense(
        int(num_classes),
        activation=None,
        kernel_regularizer=tf.keras.regularizers.l2(RNN_DEFAULTS.internal.l2_reg),
        dtype="float32",
        name="clip_logits",
    )(hidden)
    model = tf.keras.Model(inputs=[clip_x, *state_inputs], outputs=[clip_embedding, clip_logits, *state_outputs], name="rnn_clip_model")
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    optimizer = build_keras_optimizer(
        OptimizerSpec(
            name=config.optimizer.name,
            learning_rate=config.runtime.learning_rate,
            momentum=config.optimizer.momentum,
            nesterov=config.optimizer.nesterov,
            weight_decay=config.optimizer.weight_decay,
            mixed_precision=config.runtime.mixed_precision,
            clipvalue=1.0,
        ),
        context="RNN",
    )
    dummy_inputs = [tf.zeros((1, config.data.seq, num_features), dtype=tf.float32)] + zero_state_tensors(config, 1)
    _ = model(dummy_inputs, training=False)
    return model, loss_fn, optimizer


