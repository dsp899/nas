from __future__ import annotations

import os
import platform
import socket
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf

from rnn_benchlib.benchmark.metrics import compute_numeric_check, summarize_component_times
from rnn_benchlib.config.schemas import BenchmarkRecord, ClipTiming, ExperimentMeta, FeatureSpec, ModelSpec, VideoTiming
from rnn_benchlib.features.synthetic_video_features import VideoFeatureBatch
from rnn_benchlib.modeling.builders import ordered_state_arrays_from_dict, ordered_state_dict_from_outputs, state_names, zero_state_numpy


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def try_pin_to_cpu0() -> bool:
    try:
        if hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, {0})
            return True
    except Exception:
        pass
    return False


def configure_tensorflow_cpu_only_single_thread() -> Dict[str, object]:
    info: Dict[str, object] = {"gpu_disabled": False, "tf_single_thread": False, "cpu_affinity_cpu0": False}
    try:
        tf.config.set_visible_devices([], "GPU")
        info["gpu_disabled"] = True
    except Exception:
        pass
    try:
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        info["tf_single_thread"] = True
    except Exception:
        pass
    info["cpu_affinity_cpu0"] = try_pin_to_cpu0()
    return info


def create_experiment_meta(experiment_name: str, runtime: str, device_name: Optional[str] = None, notes: Optional[str] = None, extra: Optional[Dict[str, object]] = None) -> ExperimentMeta:
    experiment_id = f"exp_{uuid.uuid4().hex[:12]}"
    return ExperimentMeta(
        experiment_id=experiment_id,
        experiment_name=experiment_name,
        created_at_utc=utc_now_iso(),
        host_name=socket.gethostname(),
        device_name=device_name or platform.platform(),
        runtime=runtime,  # type: ignore[arg-type]
        notes=notes,
        extra=extra or {},
    )


def load_float_model(keras_dir: str) -> tf.keras.Model:
    return tf.keras.models.load_model(keras_dir)


def _normalize_tensor_name(name: str) -> str:
    if not name:
        return ""
    base = str(name)
    for sep in (":", ";"):
        if sep in base:
            base = base.split(sep, 1)[0]
    if base.startswith("serving_default_"):
        base = base[len("serving_default_"):]
    return base


def _resolve_named_inputs(interpreter: tf.lite.Interpreter) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    for d in interpreter.get_input_details():
        mapping[_normalize_tensor_name(d.get("name", ""))] = d["index"]
    return mapping


def _resolve_named_outputs(interpreter: tf.lite.Interpreter) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    for d in interpreter.get_output_details():
        mapping[_normalize_tensor_name(d.get("name", ""))] = d["index"]
    return mapping


def _resolve_encoder_outputs_by_shape(interpreter: tf.lite.Interpreter, spec: ModelSpec) -> List[int]:
    output_details = interpreter.get_output_details()
    if len(output_details) != (1 + len(state_names(spec))):
        raise ValueError(
            "No se pudieron resolver outputs del encoder por shape porque el número de outputs no coincide. "
            f"esperados={1 + len(state_names(spec))} reales={len(output_details)}"
        )
    details = []
    for d in output_details:
        shape = tuple(int(v) for v in np.asarray(d.get("shape", [])).tolist())
        details.append({"index": d["index"], "shape": shape})

    batch1_candidates = [item for item in details if len(item["shape"]) == 2 and item["shape"][0] == 1]
    emb_dim = spec.encoder_output_dim()
    emb_candidates = [item for item in batch1_candidates if item["shape"][1] == emb_dim]
    if not emb_candidates:
        raise ValueError(
            f"No se encontró candidato de clip_embedding por shape. emb_dim={emb_dim} shapes={[d['shape'] for d in details]}"
        )
    embedding_item = emb_candidates[-1]
    remaining = [item for item in details if item["index"] != embedding_item["index"]]

    ordered_state_indices: List[int] = []
    for state_name, arr in zero_state_numpy(spec=spec, batch_size=1, dtype=np.float32).items():
        expected_shape = tuple(arr.shape)
        matches = [item for item in remaining if item["shape"] == expected_shape]
        if not matches:
            raise ValueError(
                f"No se encontró output de estado compatible por shape para {state_name}. expected_shape={expected_shape} remaining_shapes={[d['shape'] for d in remaining]}"
            )
        chosen = matches[0]
        ordered_state_indices.append(chosen["index"])
        remaining.remove(chosen)

    return [embedding_item["index"]] + ordered_state_indices


def create_encoder_tflite_interpreter(model_path: str, spec: ModelSpec, feature_spec: FeatureSpec, num_threads: int = 1) -> Tuple[tf.lite.Interpreter, Dict[str, int], List[int]]:
    interpreter = tf.lite.Interpreter(model_path=model_path, num_threads=num_threads)
    input_name_to_index = _resolve_named_inputs(interpreter)
    dummy_clip = np.zeros((1, spec.seq, feature_spec.feature_dim), dtype=np.float32)
    dummy_state = zero_state_numpy(spec=spec, batch_size=1, dtype=np.float32)
    named_inputs = {"clip_x": dummy_clip}
    for name, arr in zip(state_names(spec), ordered_state_arrays_from_dict(spec=spec, state_dict=dummy_state)):
        named_inputs[name] = arr
    for name, arr in named_inputs.items():
        interpreter.resize_tensor_input(input_name_to_index[name], arr.shape, strict=False)
    interpreter.allocate_tensors()

    input_name_to_index = _resolve_named_inputs(interpreter)
    output_name_to_index = _resolve_named_outputs(interpreter)
    output_details = interpreter.get_output_details()
    expected_output_names = ["clip_embedding"] + state_names(spec)
    missing = [name for name in expected_output_names if name not in output_name_to_index]

    if not missing:
        ordered_output_indices = [output_name_to_index[name] for name in expected_output_names]
    else:
        try:
            ordered_output_indices = _resolve_encoder_outputs_by_shape(interpreter=interpreter, spec=spec)
        except Exception:
            if len(output_details) != len(expected_output_names):
                raise ValueError(
                    "No se pudieron resolver outputs del encoder por nombre y el número de outputs no coincide. "
                    f"Esperados={len(expected_output_names)} reales={len(output_details)} faltan={missing}"
                )
            ordered_output_indices = [d["index"] for d in output_details]

    return interpreter, input_name_to_index, ordered_output_indices


def create_head_tflite_interpreter(model_path: str, spec: ModelSpec, num_threads: int = 1) -> Tuple[tf.lite.Interpreter, int, int]:
    interpreter = tf.lite.Interpreter(model_path=model_path, num_threads=num_threads)
    input_details = interpreter.get_input_details()
    interpreter.resize_tensor_input(input_details[0]["index"], (1, spec.encoder_output_dim()), strict=False)
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    return interpreter, input_index, output_index


def timed_create_encoder_tflite_interpreter(model_path: str, spec: ModelSpec, feature_spec: FeatureSpec, num_threads: int = 1) -> Tuple[tf.lite.Interpreter, float, Dict[str, int], List[int]]:
    t0 = time.perf_counter_ns()
    interpreter, input_map, output_indices = create_encoder_tflite_interpreter(model_path=model_path, spec=spec, feature_spec=feature_spec, num_threads=num_threads)
    t1 = time.perf_counter_ns()
    return interpreter, float((t1 - t0) / 1e6), input_map, output_indices


def timed_create_head_tflite_interpreter(model_path: str, spec: ModelSpec, num_threads: int = 1) -> Tuple[tf.lite.Interpreter, float, int, int]:
    t0 = time.perf_counter_ns()
    interpreter, input_index, output_index = create_head_tflite_interpreter(model_path=model_path, spec=spec, num_threads=num_threads)
    t1 = time.perf_counter_ns()
    return interpreter, float((t1 - t0) / 1e6), input_index, output_index


def _single_clip_encoder_float(model: tf.keras.Model, spec: ModelSpec, clip_x: np.ndarray, state_dict: Dict[str, np.ndarray]) -> Tuple[np.ndarray, Dict[str, np.ndarray], float]:
    model_inputs = [clip_x] + ordered_state_arrays_from_dict(spec=spec, state_dict=state_dict)
    t0 = time.perf_counter_ns()
    outputs = model(model_inputs, training=False)
    t1 = time.perf_counter_ns()
    outputs_np = [np.asarray(o.numpy()) for o in outputs]
    return np.asarray(outputs_np[0]), ordered_state_dict_from_outputs(spec=spec, outputs=outputs_np), float((t1 - t0) / 1e6)


def _single_clip_head_float(model: tf.keras.Model, clip_embedding: np.ndarray) -> Tuple[np.ndarray, float]:
    t0 = time.perf_counter_ns()
    outputs = model([clip_embedding], training=False)
    t1 = time.perf_counter_ns()
    if isinstance(outputs, (list, tuple)):
        logits = np.asarray(outputs[0].numpy())
    else:
        logits = np.asarray(outputs.numpy())
    return logits, float((t1 - t0) / 1e6)




def resolve_encoder_tflite_output_indices_with_float_reference(
    interpreter: tf.lite.Interpreter,
    float_encoder_model: tf.keras.Model,
    spec: ModelSpec,
    feature_spec: FeatureSpec,
    input_name_to_index: Dict[str, int],
) -> List[int]:
    rng = np.random.default_rng(12345)
    clip_x = rng.standard_normal((1, spec.seq, feature_spec.feature_dim), dtype=np.float32)
    state_dict = {}
    for name, arr in zero_state_numpy(spec=spec, batch_size=1, dtype=np.float32).items():
        state_dict[name] = rng.standard_normal(arr.shape, dtype=np.float32)

    float_inputs = [clip_x] + ordered_state_arrays_from_dict(spec=spec, state_dict=state_dict)
    float_outputs = float_encoder_model(float_inputs, training=False)
    if isinstance(float_outputs, (list, tuple)):
        ref_outputs = [np.asarray(o.numpy()) for o in float_outputs]
    else:
        ref_outputs = [np.asarray(float_outputs.numpy())]

    named_inputs = {"clip_x": clip_x}
    for name, arr in zip(state_names(spec), ordered_state_arrays_from_dict(spec=spec, state_dict=state_dict)):
        named_inputs[name] = arr
    for name, arr in named_inputs.items():
        interpreter.set_tensor(input_name_to_index[name], arr)
    interpreter.invoke()

    raw_output_details = interpreter.get_output_details()
    raw_outputs = [np.asarray(interpreter.get_tensor(d["index"])) for d in raw_output_details]

    remaining = set(range(len(raw_outputs)))
    ordered_indices: List[int] = []
    for ref in ref_outputs:
        candidates = [i for i in remaining if raw_outputs[i].shape == ref.shape]
        if not candidates:
            raise ValueError(
                f"No hay outputs TFLite con shape compatible para resolver mapping. ref_shape={ref.shape} raw_shapes={[o.shape for o in raw_outputs]}"
            )
        best = min(candidates, key=lambda i: float(np.max(np.abs(raw_outputs[i] - ref))))
        ordered_indices.append(raw_output_details[best]["index"])
        remaining.remove(best)
    return ordered_indices


def _single_clip_encoder_tflite(interpreter: tf.lite.Interpreter, spec: ModelSpec, clip_x: np.ndarray, state_dict: Dict[str, np.ndarray], input_name_to_index: Dict[str, int], ordered_output_indices: List[int]) -> Tuple[np.ndarray, Dict[str, np.ndarray], float]:
    named_inputs = {"clip_x": clip_x}
    for name, arr in zip(state_names(spec), ordered_state_arrays_from_dict(spec=spec, state_dict=state_dict)):
        named_inputs[name] = arr
    for name, arr in named_inputs.items():
        interpreter.set_tensor(input_name_to_index[name], arr)
    t0 = time.perf_counter_ns()
    interpreter.invoke()
    t1 = time.perf_counter_ns()
    outputs_np = [interpreter.get_tensor(index) for index in ordered_output_indices]
    return np.asarray(outputs_np[0]), ordered_state_dict_from_outputs(spec=spec, outputs=outputs_np), float((t1 - t0) / 1e6)


def _single_clip_head_tflite(interpreter: tf.lite.Interpreter, input_index: int, output_index: int, clip_embedding: np.ndarray) -> Tuple[np.ndarray, float]:
    interpreter.set_tensor(input_index, clip_embedding)
    t0 = time.perf_counter_ns()
    interpreter.invoke()
    t1 = time.perf_counter_ns()
    logits = np.asarray(interpreter.get_tensor(output_index))
    return logits, float((t1 - t0) / 1e6)


def _next_state_for_next_clip(spec: ModelSpec, next_state: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    if spec.memory_mode == "none":
        batch_size = next(iter(next_state.values())).shape[0]
        return zero_state_numpy(spec=spec, batch_size=batch_size, dtype=np.float32)
    if spec.direction == "unidirectional":
        return next_state

    refreshed: Dict[str, np.ndarray] = {}
    for name, arr in next_state.items():
        if "_bw_" in name:
            refreshed[name] = np.zeros_like(arr)
        else:
            refreshed[name] = arr
    return refreshed


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def aggregate_video_from_logits(strategy: str, clip_logits: np.ndarray, num_classes: int) -> np.ndarray:
    if strategy == "average":
        return np.mean(clip_logits, axis=0)
    if strategy == "max_prob":
        confidences = np.max(np.apply_along_axis(_softmax, 1, clip_logits), axis=1)
        return clip_logits[int(np.argmax(confidences))]
    if strategy == "majority":
        winners = np.argmax(clip_logits, axis=1)
        counts = np.bincount(winners, minlength=num_classes).astype(np.float32)
        return counts
    raise ValueError(f"video_decision no soportado para logits: {strategy}")


def aggregate_video_from_embeddings(strategy: str, clip_embeddings: np.ndarray) -> np.ndarray:
    if strategy != "average":
        raise ValueError(f"video_decision no soportado para embeddings: {strategy}")
    return np.mean(clip_embeddings, axis=0)




def _run_video_pass_float(
    encoder_model: tf.keras.Model,
    head_model: tf.keras.Model,
    spec: ModelSpec,
    video_clips: np.ndarray,
) -> Dict[str, object]:
    batch_size = 1
    state = zero_state_numpy(spec=spec, batch_size=batch_size, dtype=np.float32)
    clip_timings: List[Dict[str, float]] = []
    clip_embeddings: List[np.ndarray] = []
    clip_logits: List[np.ndarray] = []
    video_t0 = time.perf_counter_ns()

    for clip_index in range(video_clips.shape[0]):
        clip_t0 = time.perf_counter_ns()
        clip_x = np.expand_dims(video_clips[clip_index], axis=0).astype(np.float32, copy=False)
        embedding, next_state, encoder_ms = _single_clip_encoder_float(encoder_model, spec, clip_x, state)
        bridge_ms = 0.0
        head_ms = 0.0
        if spec.video_decision_input == "clip_logits":
            bridge_t0 = time.perf_counter_ns()
            head_input = np.asarray(embedding, dtype=np.float32)
            bridge_ms = float((time.perf_counter_ns() - bridge_t0) / 1e6)
            logits, head_ms = _single_clip_head_float(head_model, head_input)
            clip_logits.append(np.asarray(logits[0]))
        clip_embeddings.append(np.asarray(embedding[0]))
        state = _next_state_for_next_clip(spec=spec, next_state=next_state)
        clip_e2e_sum_ms = float(encoder_ms + bridge_ms + head_ms)
        clip_e2e_wall_ms = float((time.perf_counter_ns() - clip_t0) / 1e6)
        clip_timings.append({
            "clip_index": float(clip_index),
            "carry_state": float(1.0 if spec.memory_mode == "carry_forward" else 0.0),
            "clip_encoder_ms": float(encoder_ms),
            "clip_bridge_ms": float(bridge_ms),
            "clip_head_ms": float(head_ms),
            "clip_e2e_sum_ms": clip_e2e_sum_ms,
            "clip_e2e_wall_ms": clip_e2e_wall_ms,
        })

    t0 = time.perf_counter_ns()
    if spec.video_decision_input == "clip_logits":
        video_logits = aggregate_video_from_logits(spec.video_decision, np.stack(clip_logits, axis=0), spec.num_classes)
        video_aggregation_ms = float((time.perf_counter_ns() - t0) / 1e6)
        video_head_ms = 0.0
    else:
        aggregated_embedding = aggregate_video_from_embeddings(spec.video_decision, np.stack(clip_embeddings, axis=0))
        video_aggregation_ms = float((time.perf_counter_ns() - t0) / 1e6)
        logits, video_head_ms = _single_clip_head_float(head_model, np.expand_dims(aggregated_embedding, axis=0).astype(np.float32))
        video_logits = np.asarray(logits[0])

    video_encoder_sum_ms = float(sum(item["clip_encoder_ms"] for item in clip_timings))
    video_bridge_sum_ms = float(sum(item["clip_bridge_ms"] for item in clip_timings))
    video_head_clip_sum_ms = float(sum(item["clip_head_ms"] for item in clip_timings))
    video_e2e_sum_ms = float(video_encoder_sum_ms + video_bridge_sum_ms + video_head_clip_sum_ms + video_aggregation_ms + video_head_ms)
    video_e2e_wall_ms = float((time.perf_counter_ns() - video_t0) / 1e6)
    return {
        "clip_timings": clip_timings,
        "video_logits": np.asarray(video_logits),
        "video_encoder_sum_ms": video_encoder_sum_ms,
        "video_bridge_sum_ms": video_bridge_sum_ms,
        "video_head_clip_sum_ms": video_head_clip_sum_ms,
        "video_aggregation_ms": video_aggregation_ms,
        "video_head_ms": float(video_head_ms),
        "video_e2e_sum_ms": video_e2e_sum_ms,
        "video_e2e_wall_ms": video_e2e_wall_ms,
    }


def _run_video_pass_tflite(
    encoder_interpreter: tf.lite.Interpreter,
    head_interpreter: tf.lite.Interpreter,
    spec: ModelSpec,
    video_clips: np.ndarray,
    encoder_input_name_to_index: Dict[str, int],
    encoder_ordered_output_indices: List[int],
    head_input_index: int,
    head_output_index: int,
) -> Dict[str, object]:
    batch_size = 1
    state = zero_state_numpy(spec=spec, batch_size=batch_size, dtype=np.float32)
    clip_timings: List[Dict[str, float]] = []
    clip_embeddings: List[np.ndarray] = []
    clip_logits: List[np.ndarray] = []
    video_t0 = time.perf_counter_ns()

    for clip_index in range(video_clips.shape[0]):
        clip_t0 = time.perf_counter_ns()
        clip_x = np.expand_dims(video_clips[clip_index], axis=0).astype(np.float32, copy=False)
        embedding, next_state, encoder_ms = _single_clip_encoder_tflite(
            encoder_interpreter,
            spec,
            clip_x,
            state,
            encoder_input_name_to_index,
            encoder_ordered_output_indices,
        )
        bridge_ms = 0.0
        head_ms = 0.0
        if spec.video_decision_input == "clip_logits":
            bridge_t0 = time.perf_counter_ns()
            head_input = np.asarray(embedding, dtype=np.float32)
            bridge_ms = float((time.perf_counter_ns() - bridge_t0) / 1e6)
            logits, head_ms = _single_clip_head_tflite(head_interpreter, head_input_index, head_output_index, head_input)
            clip_logits.append(np.asarray(logits[0]))
        clip_embeddings.append(np.asarray(embedding[0]))
        state = _next_state_for_next_clip(spec=spec, next_state=next_state)
        clip_e2e_sum_ms = float(encoder_ms + bridge_ms + head_ms)
        clip_e2e_wall_ms = float((time.perf_counter_ns() - clip_t0) / 1e6)
        clip_timings.append({
            "clip_index": float(clip_index),
            "carry_state": float(1.0 if spec.memory_mode == "carry_forward" else 0.0),
            "clip_encoder_ms": float(encoder_ms),
            "clip_bridge_ms": float(bridge_ms),
            "clip_head_ms": float(head_ms),
            "clip_e2e_sum_ms": clip_e2e_sum_ms,
            "clip_e2e_wall_ms": clip_e2e_wall_ms,
        })

    t0 = time.perf_counter_ns()
    if spec.video_decision_input == "clip_logits":
        video_logits = aggregate_video_from_logits(spec.video_decision, np.stack(clip_logits, axis=0), spec.num_classes)
        video_aggregation_ms = float((time.perf_counter_ns() - t0) / 1e6)
        video_head_ms = 0.0
    else:
        aggregated_embedding = aggregate_video_from_embeddings(spec.video_decision, np.stack(clip_embeddings, axis=0))
        video_aggregation_ms = float((time.perf_counter_ns() - t0) / 1e6)
        logits, video_head_ms = _single_clip_head_tflite(
            head_interpreter,
            head_input_index,
            head_output_index,
            np.expand_dims(aggregated_embedding, axis=0).astype(np.float32),
        )
        video_logits = np.asarray(logits[0])

    video_encoder_sum_ms = float(sum(item["clip_encoder_ms"] for item in clip_timings))
    video_bridge_sum_ms = float(sum(item["clip_bridge_ms"] for item in clip_timings))
    video_head_clip_sum_ms = float(sum(item["clip_head_ms"] for item in clip_timings))
    video_e2e_sum_ms = float(video_encoder_sum_ms + video_bridge_sum_ms + video_head_clip_sum_ms + video_aggregation_ms + video_head_ms)
    video_e2e_wall_ms = float((time.perf_counter_ns() - video_t0) / 1e6)
    return {
        "clip_timings": clip_timings,
        "video_logits": np.asarray(video_logits),
        "video_encoder_sum_ms": video_encoder_sum_ms,
        "video_bridge_sum_ms": video_bridge_sum_ms,
        "video_head_clip_sum_ms": video_head_clip_sum_ms,
        "video_aggregation_ms": video_aggregation_ms,
        "video_head_ms": float(video_head_ms),
        "video_e2e_sum_ms": video_e2e_sum_ms,
        "video_e2e_wall_ms": video_e2e_wall_ms,
    }


def _benchmark_runner(run_fn, warmup_runs: int, steady_runs: int) -> Dict[str, object]:
    for _ in range(warmup_runs):
        run_fn()
    representative = run_fn()
    steady_clip_encoder_ms: List[float] = []
    steady_clip_bridge_ms: List[float] = []
    steady_clip_head_ms: List[float] = []
    steady_clip_e2e_sum_ms: List[float] = []
    steady_clip_e2e_wall_ms: List[float] = []
    steady_video_encoder_sum_ms: List[float] = []
    steady_video_bridge_sum_ms: List[float] = []
    steady_video_head_clip_sum_ms: List[float] = []
    steady_video_aggregation_ms: List[float] = []
    steady_video_head_ms: List[float] = []
    steady_video_e2e_sum_ms: List[float] = []
    steady_video_e2e_wall_ms: List[float] = []

    for _ in range(steady_runs):
        result = run_fn()
        for timing in result["clip_timings"]:
            steady_clip_encoder_ms.append(float(timing["clip_encoder_ms"]))
            steady_clip_bridge_ms.append(float(timing["clip_bridge_ms"]))
            steady_clip_head_ms.append(float(timing["clip_head_ms"]))
            steady_clip_e2e_sum_ms.append(float(timing["clip_e2e_sum_ms"]))
            steady_clip_e2e_wall_ms.append(float(timing["clip_e2e_wall_ms"]))
        steady_video_encoder_sum_ms.append(float(result["video_encoder_sum_ms"]))
        steady_video_bridge_sum_ms.append(float(result["video_bridge_sum_ms"]))
        steady_video_head_clip_sum_ms.append(float(result["video_head_clip_sum_ms"]))
        steady_video_aggregation_ms.append(float(result["video_aggregation_ms"]))
        steady_video_head_ms.append(float(result["video_head_ms"]))
        steady_video_e2e_sum_ms.append(float(result["video_e2e_sum_ms"]))
        steady_video_e2e_wall_ms.append(float(result["video_e2e_wall_ms"]))

    if not steady_clip_e2e_sum_ms:
        for timing in representative["clip_timings"]:
            steady_clip_encoder_ms.append(float(timing["clip_encoder_ms"]))
            steady_clip_bridge_ms.append(float(timing["clip_bridge_ms"]))
            steady_clip_head_ms.append(float(timing["clip_head_ms"]))
            steady_clip_e2e_sum_ms.append(float(timing["clip_e2e_sum_ms"]))
            steady_clip_e2e_wall_ms.append(float(timing["clip_e2e_wall_ms"]))
        steady_video_encoder_sum_ms.append(float(representative["video_encoder_sum_ms"]))
        steady_video_bridge_sum_ms.append(float(representative["video_bridge_sum_ms"]))
        steady_video_head_clip_sum_ms.append(float(representative["video_head_clip_sum_ms"]))
        steady_video_aggregation_ms.append(float(representative["video_aggregation_ms"]))
        steady_video_head_ms.append(float(representative["video_head_ms"]))
        steady_video_e2e_sum_ms.append(float(representative["video_e2e_sum_ms"]))
        steady_video_e2e_wall_ms.append(float(representative["video_e2e_wall_ms"]))

    return {
        "representative": representative,
        "steady_clip_encoder_ms": steady_clip_encoder_ms,
        "steady_clip_bridge_ms": steady_clip_bridge_ms,
        "steady_clip_head_ms": steady_clip_head_ms,
        "steady_clip_e2e_sum_ms": steady_clip_e2e_sum_ms,
        "steady_clip_e2e_wall_ms": steady_clip_e2e_wall_ms,
        "steady_video_encoder_sum_ms": steady_video_encoder_sum_ms,
        "steady_video_bridge_sum_ms": steady_video_bridge_sum_ms,
        "steady_video_head_clip_sum_ms": steady_video_head_clip_sum_ms,
        "steady_video_aggregation_ms": steady_video_aggregation_ms,
        "steady_video_head_ms": steady_video_head_ms,
        "steady_video_e2e_sum_ms": steady_video_e2e_sum_ms,
        "steady_video_e2e_wall_ms": steady_video_e2e_wall_ms,
    }


def benchmark_float_video(encoder_model: tf.keras.Model, head_model: tf.keras.Model, spec: ModelSpec, video_clips: np.ndarray, warmup_runs: int = 5, steady_runs: int = 10) -> Dict[str, object]:
    return _benchmark_runner(lambda: _run_video_pass_float(encoder_model=encoder_model, head_model=head_model, spec=spec, video_clips=video_clips), warmup_runs=warmup_runs, steady_runs=steady_runs)


def benchmark_tflite_video(
    encoder_interpreter: tf.lite.Interpreter,
    head_interpreter: tf.lite.Interpreter,
    spec: ModelSpec,
    video_clips: np.ndarray,
    init_ms: float,
    encoder_input_name_to_index: Dict[str, int],
    encoder_ordered_output_indices: List[int],
    head_input_index: int,
    head_output_index: int,
    warmup_runs: int = 5,
    steady_runs: int = 10,
) -> Dict[str, object]:
    result = _benchmark_runner(
        lambda: _run_video_pass_tflite(
            encoder_interpreter=encoder_interpreter,
            head_interpreter=head_interpreter,
            spec=spec,
            video_clips=video_clips,
            encoder_input_name_to_index=encoder_input_name_to_index,
            encoder_ordered_output_indices=encoder_ordered_output_indices,
            head_input_index=head_input_index,
            head_output_index=head_output_index,
        ),
        warmup_runs=warmup_runs,
        steady_runs=steady_runs,
    )
    result["init_ms"] = init_ms
    return result


def create_benchmark_record(
    experiment: ExperimentMeta,
    model_id: str,
    runtime_kind: str,
    spec: ModelSpec,
    feature_spec: FeatureSpec,
    video_index: int,
    benchmark_result: Dict[str, object],
    threads: int,
    batch_size: int,
    numeric_check=None,
    init_ms: Optional[float] = None,
) -> BenchmarkRecord:
    representative = benchmark_result["representative"]
    clip_timings = [
        ClipTiming(
            clip_index=int(timing["clip_index"]),
            carry_state=bool(timing["carry_state"]),
            clip_encoder_ms=float(timing["clip_encoder_ms"]),
            clip_bridge_ms=float(timing["clip_bridge_ms"]),
            clip_head_ms=float(timing["clip_head_ms"]),
            clip_e2e_sum_ms=float(timing["clip_e2e_sum_ms"]),
            clip_e2e_wall_ms=float(timing["clip_e2e_wall_ms"]),
        )
        for timing in representative["clip_timings"]
    ]
    runtime_summary = summarize_component_times(
        clip_encoder_times_ms=benchmark_result["steady_clip_encoder_ms"],
        clip_bridge_times_ms=benchmark_result["steady_clip_bridge_ms"],
        clip_head_times_ms=benchmark_result["steady_clip_head_ms"],
        clip_e2e_sum_times_ms=benchmark_result["steady_clip_e2e_sum_ms"],
        clip_e2e_wall_times_ms=benchmark_result["steady_clip_e2e_wall_ms"],
        video_encoder_sum_times_ms=benchmark_result["steady_video_encoder_sum_ms"],
        video_bridge_sum_times_ms=benchmark_result["steady_video_bridge_sum_ms"],
        video_head_clip_sum_times_ms=benchmark_result["steady_video_head_clip_sum_ms"],
        video_aggregation_times_ms=benchmark_result["steady_video_aggregation_ms"],
        video_head_times_ms=benchmark_result["steady_video_head_ms"],
        video_e2e_sum_times_ms=benchmark_result["steady_video_e2e_sum_ms"],
        video_e2e_wall_times_ms=benchmark_result["steady_video_e2e_wall_ms"],
        init_ms=init_ms,
    )
    video_timing = VideoTiming(
        video_encoder_sum_ms=float(representative["video_encoder_sum_ms"]),
        video_bridge_sum_ms=float(representative["video_bridge_sum_ms"]),
        video_head_clip_sum_ms=float(representative["video_head_clip_sum_ms"]),
        video_aggregation_ms=float(representative["video_aggregation_ms"]),
        video_head_ms=float(representative["video_head_ms"]),
        video_e2e_sum_ms=float(representative["video_e2e_sum_ms"]),
        video_e2e_wall_ms=float(representative["video_e2e_wall_ms"]),
    )
    return BenchmarkRecord(
        experiment=experiment,
        model_id=model_id,
        runtime_kind=runtime_kind,  # type: ignore[arg-type]
        memory_mode=spec.memory_mode,
        seq=spec.seq,
        video_index=video_index,
        clips_per_video=spec.clips_per_video(feature_spec.video_steps),
        threads=threads,
        batch_size=batch_size,
        runtime_summary=runtime_summary,
        clip_timings=clip_timings,
        video_timing=video_timing,
        numeric_check=numeric_check,
        extra={
            "video_steps": feature_spec.video_steps,
            "feature_dim": feature_spec.feature_dim,
            "direction": spec.direction,
            "rnn": spec.rnn,
            "units_0": spec.units_0,
            "units_1": spec.units_1,
            "units_2": spec.units_2,
            "head_units": spec.head_units,
            "num_classes": spec.num_classes,
            "video_decision": spec.video_decision,
            "video_decision_input": spec.video_decision_input,
        },
    )

def benchmark_video_batch(
    experiment: ExperimentMeta,
    model_id: str,
    spec: ModelSpec,
    feature_spec: FeatureSpec,
    batch: VideoFeatureBatch,
    float_encoder_model: Optional[tf.keras.Model] = None,
    float_head_model: Optional[tf.keras.Model] = None,
    tflite_encoder_interpreter: Optional[tf.lite.Interpreter] = None,
    tflite_head_interpreter: Optional[tf.lite.Interpreter] = None,
    tflite_init_ms: Optional[float] = None,
    tflite_encoder_input_name_to_index: Optional[Dict[str, int]] = None,
    tflite_encoder_ordered_output_indices: Optional[List[int]] = None,
    tflite_head_input_index: Optional[int] = None,
    tflite_head_output_index: Optional[int] = None,
    runtime: str = "both",
    warmup_runs: int = 5,
    steady_runs: int = 10,
    threads: int = 1,
) -> List[BenchmarkRecord]:
    records: List[BenchmarkRecord] = []
    if runtime not in ("float", "tflite", "both"):
        raise ValueError(f"runtime inválido: {runtime}")
    if runtime in ("float", "both") and (float_encoder_model is None or float_head_model is None):
        raise ValueError("Se requieren float_encoder_model y float_head_model para runtime=float o both")
    if runtime in ("tflite", "both") and (
        tflite_encoder_interpreter is None or tflite_head_interpreter is None or tflite_init_ms is None or
        tflite_encoder_input_name_to_index is None or tflite_encoder_ordered_output_indices is None or
        tflite_head_input_index is None or tflite_head_output_index is None
    ):
        raise ValueError("Faltan dependencias TFLite para runtime=tflite o both")

    for video_index in range(batch.num_videos):
        video_clips = batch.clips[video_index]
        float_result = None
        if runtime in ("float", "both"):
            float_result = benchmark_float_video(
                encoder_model=float_encoder_model,  # type: ignore[arg-type]
                head_model=float_head_model,  # type: ignore[arg-type]
                spec=spec,
                video_clips=video_clips,
                warmup_runs=warmup_runs,
                steady_runs=steady_runs,
            )
            records.append(
                create_benchmark_record(
                    experiment=experiment,
                    model_id=model_id,
                    runtime_kind="float",
                    spec=spec,
                    feature_spec=feature_spec,
                    video_index=video_index,
                    benchmark_result=float_result,
                    threads=threads,
                    batch_size=1,
                    numeric_check=None,
                    init_ms=None,
                )
            )
        if runtime in ("tflite", "both"):
            tflite_result = benchmark_tflite_video(
                encoder_interpreter=tflite_encoder_interpreter,  # type: ignore[arg-type]
                head_interpreter=tflite_head_interpreter,  # type: ignore[arg-type]
                spec=spec,
                video_clips=video_clips,
                init_ms=float(tflite_init_ms),
                encoder_input_name_to_index=tflite_encoder_input_name_to_index,  # type: ignore[arg-type]
                encoder_ordered_output_indices=tflite_encoder_ordered_output_indices,  # type: ignore[arg-type]
                head_input_index=int(tflite_head_input_index),  # type: ignore[arg-type]
                head_output_index=int(tflite_head_output_index),  # type: ignore[arg-type]
                warmup_runs=warmup_runs,
                steady_runs=steady_runs,
            )
            numeric_check = None
            if float_result is not None:
                numeric_check = compute_numeric_check(
                    y_ref=np.asarray(float_result["representative"]["video_logits"]),
                    y_test=np.asarray(tflite_result["representative"]["video_logits"]),
                )
            records.append(
                create_benchmark_record(
                    experiment=experiment,
                    model_id=model_id,
                    runtime_kind="tflite",
                    spec=spec,
                    feature_spec=feature_spec,
                    video_index=video_index,
                    benchmark_result=tflite_result,
                    threads=threads,
                    batch_size=1,
                    numeric_check=numeric_check,
                    init_ms=float(tflite_init_ms),
                )
            )
    return records
