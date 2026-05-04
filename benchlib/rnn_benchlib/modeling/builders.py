from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import tensorflow as tf

from rnn_benchlib.config.schemas import FeatureSpec, ModelSpec
from rnn_benchlib.sampling.validators import assert_valid_model_spec


def get_state_spec(spec: ModelSpec) -> List[Dict[str, Any]]:
    state_entries: List[Dict[str, Any]] = []
    units_list = spec.normalized_units_list()
    for layer_index, units in enumerate(units_list):
        if spec.direction == "unidirectional":
            if spec.rnn == "lstm":
                state_entries.extend(
                    [
                        {"name": f"layer{layer_index}_h", "units": units, "kind": "h", "layer_index": layer_index, "direction": "uni"},
                        {"name": f"layer{layer_index}_c", "units": units, "kind": "c", "layer_index": layer_index, "direction": "uni"},
                    ]
                )
            else:
                state_entries.append(
                    {"name": f"layer{layer_index}_h", "units": units, "kind": "h", "layer_index": layer_index, "direction": "uni"}
                )
        else:
            for direction in ("fw", "bw"):
                if spec.rnn == "lstm":
                    state_entries.extend(
                        [
                            {"name": f"layer{layer_index}_{direction}_h", "units": units, "kind": "h", "layer_index": layer_index, "direction": direction},
                            {"name": f"layer{layer_index}_{direction}_c", "units": units, "kind": "c", "layer_index": layer_index, "direction": direction},
                        ]
                    )
                else:
                    state_entries.append(
                        {"name": f"layer{layer_index}_{direction}_h", "units": units, "kind": "h", "layer_index": layer_index, "direction": direction}
                    )
    return state_entries


def state_names(spec: ModelSpec) -> List[str]:
    return [entry["name"] for entry in get_state_spec(spec)]


def zero_state_numpy(spec: ModelSpec, batch_size: int, dtype: np.dtype = np.float32) -> Dict[str, np.ndarray]:
    result: Dict[str, np.ndarray] = {}
    for entry in get_state_spec(spec):
        result[entry["name"]] = np.zeros((batch_size, entry["units"]), dtype=dtype)
    return result


def ordered_state_arrays_from_dict(spec: ModelSpec, state_dict: Dict[str, np.ndarray]) -> List[np.ndarray]:
    arrays: List[np.ndarray] = []
    for entry in get_state_spec(spec):
        name = entry["name"]
        if name not in state_dict:
            raise KeyError(f"Falta el estado requerido: {name}")
        arrays.append(np.asarray(state_dict[name], dtype=np.float32))
    return arrays


def ordered_state_dict_from_outputs(spec: ModelSpec, outputs: Sequence[np.ndarray]) -> Dict[str, np.ndarray]:
    state_entries = get_state_spec(spec)
    state_outputs = list(outputs[1:])
    if len(state_outputs) != len(state_entries):
        raise ValueError(f"Número de estados devueltos inválido: recibido={len(state_outputs)}, esperado={len(state_entries)}")
    result: Dict[str, np.ndarray] = {}
    for entry, arr in zip(state_entries, state_outputs):
        result[entry["name"]] = np.asarray(arr, dtype=np.float32)
    return result


def _make_rnn_cell_layer(
    spec: ModelSpec,
    units: int,
    layer_index: int,
    return_sequences: bool,
    *,
    go_backwards: bool = False,
    name_prefix: str | None = None,
) -> tf.keras.layers.Layer:
    layer_name = name_prefix or f"{spec.rnn}_layer_{layer_index}"
    common_kwargs = dict(
        units=units,
        return_sequences=return_sequences,
        return_state=True,
        go_backwards=go_backwards,
        name=layer_name,
    )
    if spec.rnn == "lstm":
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
    spec: ModelSpec,
    units: int,
    layer_index: int,
    return_sequences: bool,
    state_inputs: List[tf.keras.Input],
    state_outputs: List[tf.Tensor],
) -> tf.Tensor:
    layer = _make_rnn_cell_layer(spec=spec, units=units, layer_index=layer_index, return_sequences=return_sequences, go_backwards=False)
    if spec.rnn == "lstm":
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
    spec: ModelSpec,
    units: int,
    layer_index: int,
    return_sequences: bool,
    state_inputs: List[tf.keras.Input],
    state_outputs: List[tf.Tensor],
) -> tf.Tensor:
    forward_layer = _make_rnn_cell_layer(
        spec=spec,
        units=units,
        layer_index=layer_index,
        return_sequences=return_sequences,
        go_backwards=False,
        name_prefix=f"fw_{spec.rnn}_layer_{layer_index}",
    )
    backward_layer = _make_rnn_cell_layer(
        spec=spec,
        units=units,
        layer_index=layer_index,
        return_sequences=return_sequences,
        go_backwards=True,
        name_prefix=f"bw_{spec.rnn}_layer_{layer_index}",
    )
    bidi = tf.keras.layers.Bidirectional(
        layer=forward_layer,
        backward_layer=backward_layer,
        merge_mode="concat",
        name=f"bidi_{spec.rnn}_layer_{layer_index}",
    )
    if spec.rnn == "lstm":
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


def build_encoder_model(spec: ModelSpec, feature_spec: FeatureSpec) -> Tuple[tf.keras.Model, Dict[str, Any]]:
    assert_valid_model_spec(spec=spec, video_steps=feature_spec.video_steps)
    clip_x = tf.keras.Input(shape=(spec.seq, feature_spec.feature_dim), name="clip_x", dtype=tf.float32)
    state_entries = get_state_spec(spec)
    state_inputs: List[tf.keras.Input] = [
        tf.keras.Input(shape=(entry["units"],), name=entry["name"], dtype=tf.float32)
        for entry in state_entries
    ]

    x = clip_x
    state_input_cursor = list(state_inputs)
    state_outputs: List[tf.Tensor] = []
    units_list = spec.normalized_units_list()
    for layer_index, units in enumerate(units_list):
        is_last = layer_index == (len(units_list) - 1)
        return_sequences = not is_last
        if spec.direction == "unidirectional":
            x = _consume_unidirectional_layer(
                x=x,
                spec=spec,
                units=units,
                layer_index=layer_index,
                return_sequences=return_sequences,
                state_inputs=state_input_cursor,
                state_outputs=state_outputs,
            )
        else:
            x = _consume_bidirectional_layer(
                x=x,
                spec=spec,
                units=units,
                layer_index=layer_index,
                return_sequences=return_sequences,
                state_inputs=state_input_cursor,
                state_outputs=state_outputs,
            )

    clip_embedding = tf.keras.layers.Activation("linear", name="clip_embedding")(x)
    outputs: List[tf.Tensor] = [clip_embedding] + state_outputs
    model_inputs: List[tf.Tensor] = [clip_x] + state_inputs
    model = tf.keras.Model(inputs=model_inputs, outputs=outputs, name=f"encoder_{spec.rnn}_{spec.direction}_L{spec.layers}_seq{spec.seq}")
    metadata = {
        "model_name": model.name,
        "output_names": ["clip_embedding"] + state_names(spec),
        "state_spec": state_entries,
        "clip_embedding_dim": spec.encoder_output_dim(),
    }
    return model, metadata


def build_head_model(spec: ModelSpec) -> Tuple[tf.keras.Model, Dict[str, Any]]:
    clip_embedding = tf.keras.Input(shape=(spec.encoder_output_dim(),), name="clip_embedding", dtype=tf.float32)
    x = tf.keras.layers.Dense(spec.head_units, activation="relu", name="head_hidden")(clip_embedding)
    clip_logits = tf.keras.layers.Dense(spec.num_classes, activation=None, name="clip_logits")(x)
    model = tf.keras.Model(inputs=[clip_embedding], outputs=[clip_logits], name=f"head_h{spec.head_units}_c{spec.num_classes}")
    metadata = {
        "model_name": model.name,
        "output_names": ["clip_logits"],
    }
    return model, metadata


def build_random_initialized_models(spec: ModelSpec, feature_spec: FeatureSpec) -> Tuple[tf.keras.Model, tf.keras.Model, Dict[str, Any]]:
    encoder_model, encoder_metadata = build_encoder_model(spec=spec, feature_spec=feature_spec)
    head_model, head_metadata = build_head_model(spec=spec)

    encoder_inputs: List[np.ndarray] = [np.zeros((1, spec.seq, feature_spec.feature_dim), dtype=np.float32)]
    dummy_state = zero_state_numpy(spec=spec, batch_size=1, dtype=np.float32)
    encoder_inputs.extend(ordered_state_arrays_from_dict(spec=spec, state_dict=dummy_state))
    encoder_outputs = encoder_model(encoder_inputs, training=False)
    if isinstance(encoder_outputs, (list, tuple)):
        clip_embedding = np.asarray(encoder_outputs[0].numpy())
    else:
        clip_embedding = np.asarray(encoder_outputs.numpy())
    _ = head_model([clip_embedding], training=False)

    metadata = {
        "encoder": encoder_metadata,
        "head": head_metadata,
    }
    return encoder_model, head_model, metadata
