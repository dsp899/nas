from __future__ import annotations

import os
import shutil
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf

from rnn_benchlib.config.schemas import (
    ARTIFACT_SCHEMA_VERSION,
    ArtifactPaths,
    ConversionInfo,
    FeatureSpec,
    ModelRecord,
    ModelSpec,
)
from rnn_benchlib.conversion.graph_records import build_graph_record, save_graph_record
from rnn_benchlib.modeling.builders import build_random_initialized_models, ordered_state_arrays_from_dict, zero_state_numpy
from rnn_benchlib.sampling.sampler import model_spec_to_id
from rnn_benchlib.storage.jsonl import write_json


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def model_dir_from_id(output_root: str, model_id: str) -> str:
    return os.path.join(output_root, "models", model_id)


def build_artifact_paths(output_root: str, model_id: str) -> ArtifactPaths:
    model_dir = model_dir_from_id(output_root=output_root, model_id=model_id)
    meta_dir = os.path.join(model_dir, "meta")
    source_dir = os.path.join(model_dir, "source")
    compiled_dir = os.path.join(model_dir, "compiled")
    graphs_dir = os.path.join(model_dir, "graphs")
    reports_dir = os.path.join(model_dir, "reports")
    benchmark_dir = os.path.join(model_dir, "benchmarks")
    return ArtifactPaths(
        model_dir=model_dir,
        meta_dir=meta_dir,
        spec_path=os.path.join(meta_dir, "spec.json"),
        manifest_path=os.path.join(meta_dir, "manifest.json"),
        source_dir=source_dir,
        encoder_keras_dir=os.path.join(source_dir, "encoder.keras"),
        head_keras_dir=os.path.join(source_dir, "head.keras"),
        compiled_dir=compiled_dir,
        encoder_tflite_path=os.path.join(compiled_dir, "encoder.tflite"),
        head_tflite_path=os.path.join(compiled_dir, "head.tflite"),
        graphs_dir=graphs_dir,
        encoder_tflite_graph_path=os.path.join(graphs_dir, "encoder_tflite_graph.json"),
        head_tflite_graph_path=os.path.join(graphs_dir, "head_tflite_graph.json"),
        graph_record_path=os.path.join(graphs_dir, "graph_record.json"),
        reports_dir=reports_dir,
        conversion_report_path=os.path.join(reports_dir, "conversion_report.json"),
        benchmark_dir=benchmark_dir,
    )


def prepare_model_artifact_dir(paths: ArtifactPaths, overwrite: bool = False) -> None:
    if overwrite and os.path.exists(paths.model_dir):
        shutil.rmtree(paths.model_dir)
    ensure_dir(paths.model_dir)
    ensure_dir(paths.meta_dir)
    ensure_dir(paths.source_dir)
    ensure_dir(paths.compiled_dir)
    ensure_dir(paths.graphs_dir)
    ensure_dir(paths.reports_dir)
    ensure_dir(paths.benchmark_dir)


def _convert_once(model: tf.keras.Model, supported_ops: List[tf.lite.OpsSet], lower_tensor_list_ops: Optional[bool]) -> bytes:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = supported_ops
    if lower_tensor_list_ops is not None:
        converter._experimental_lower_tensor_list_ops = lower_tensor_list_ops
    return converter.convert()


def convert_model_with_fallback(model: tf.keras.Model) -> Tuple[bytes, str, Dict[str, Dict[str, Any]]]:
    attempts: Dict[str, Dict[str, Any]] = {}
    try:
        tflite_model = _convert_once(
            model=model,
            supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS],
            lower_tensor_list_ops=True,
        )
        attempts["builtin_only"] = {"status": "ok", "error": None}
        return tflite_model, "builtin_only", attempts
    except Exception as e_builtin:
        attempts["builtin_only"] = {"status": "failed", "error": repr(e_builtin)}

    tflite_model = _convert_once(
        model=model,
        supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS],
        lower_tensor_list_ops=False,
    )
    attempts["builtin_plus_select_tf_ops"] = {"status": "ok", "error": None}
    return tflite_model, "builtin_plus_select_tf_ops", attempts


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if hasattr(obj, "tolist") and callable(getattr(obj, "tolist", None)):
        try:
            return obj.tolist()
        except Exception:
            pass
    return repr(obj)


def _normalize_dtype_name(value: Any) -> str:
    try:
        return np.dtype(value).name
    except Exception:
        return str(value)


def _normalize_quantization_parameters(qparams: Any) -> Dict[str, Any]:
    qparams = qparams or {}
    scales = _jsonable(qparams.get("scales", []))
    zero_points = _jsonable(qparams.get("zero_points", []))
    return {
        "scales": scales,
        "zero_points": zero_points,
        "quantized_dimension": _jsonable(qparams.get("quantized_dimension", 0)),
    }


def _quant_mode_from_qparams(qparams: Dict[str, Any], dtype: str) -> str:
    if qparams.get("scales"):
        if dtype in {"int8", "uint8", "int16", "int32"}:
            return "int_quantized"
        return "per_tensor_or_channel"
    if dtype == "float16":
        return "float16"
    return "none"


def _tensor_detail_to_record(detail: Dict[str, Any], *, model_inputs: set[int], model_outputs: set[int]) -> Dict[str, Any]:
    tensor_id = int(detail.get("index", -1))
    shape = _jsonable(detail.get("shape", []))
    qparams = _normalize_quantization_parameters(detail.get("quantization_parameters", {}))
    dtype_name = _normalize_dtype_name(detail.get("dtype"))
    num_elements = None
    try:
        total = 1
        for dim in shape:
            dim_i = int(dim)
            if dim_i < 0:
                total = None
                break
            total *= dim_i
        num_elements = total
    except Exception:
        num_elements = None
    return {
        "tensor_id": tensor_id,
        "name": str(detail.get("name", f"tensor_{tensor_id}")),
        "shape": shape,
        "shape_signature": _jsonable(detail.get("shape_signature", detail.get("shape", []))),
        "rank": len(shape) if hasattr(shape, "__len__") else None,
        "num_elements": num_elements,
        "dtype": dtype_name,
        "quantization": _jsonable(detail.get("quantization", (0.0, 0))),
        "quantization_parameters": qparams,
        "quantization_mode": _quant_mode_from_qparams(qparams, dtype_name),
        "sparsity_parameters": _jsonable(detail.get("sparsity_parameters", {})),
        "is_model_input": tensor_id in model_inputs,
        "is_model_output": tensor_id in model_outputs,
        "producer_op": None,
        "consumer_ops": [],
    }


def _op_family(op_name: str) -> str:
    if op_name.startswith("Flex"):
        return "flex"
    return "builtin"


def _infer_quantization_mode(tensors: List[Dict[str, Any]]) -> str:
    modes = {tensor.get("quantization_mode", "none") for tensor in tensors}
    modes.discard("none")
    if not modes:
        return "none"
    if "int_quantized" in modes:
        return "int_quantized"
    if "float16" in modes:
        return "float16"
    return sorted(modes)[0]


def _infer_delegate_partitions(canonical_ops: List[Dict[str, Any]], runtime_delegate_ops: List[Dict[str, Any]], model_input_ids: set[int]) -> List[Dict[str, Any]]:
    producer_by_tensor: Dict[int, int] = {}
    op_by_id: Dict[int, Dict[str, Any]] = {op["op_id"]: op for op in canonical_ops}
    for op in canonical_ops:
        for tensor_id in op.get("output_tensors", []):
            producer_by_tensor[int(tensor_id)] = int(op["op_id"])

    partitions: List[Dict[str, Any]] = []
    for idx, delegate_op in enumerate(runtime_delegate_ops):
        boundary_inputs = [int(t) for t in delegate_op.get("input_tensors", []) if int(t) >= 0]
        boundary_outputs = [int(t) for t in delegate_op.get("output_tensors", []) if int(t) >= 0]
        stack = [producer_by_tensor[t] for t in boundary_outputs if t in producer_by_tensor]
        visited: set[int] = set()
        while stack:
            op_id = stack.pop()
            if op_id in visited:
                continue
            visited.add(op_id)
            op = op_by_id.get(op_id)
            if op is None:
                continue
            for tensor_id in op.get("input_tensors", []):
                if tensor_id in boundary_inputs or tensor_id in model_input_ids:
                    continue
                pred = producer_by_tensor.get(int(tensor_id))
                if pred is not None:
                    stack.append(pred)
        covered = sorted(visited)
        backend_hint = "flex" if any(op_by_id[op_id]["op_family"] == "flex" for op_id in covered) else "default_delegate"
        partitions.append(
            {
                "partition_id": idx,
                "delegate_backend_hint": backend_hint,
                "delegate_op_id": int(delegate_op.get("op_id", idx)),
                "boundary_input_tensor_ids": boundary_inputs,
                "boundary_output_tensor_ids": boundary_outputs,
                "covered_canonical_op_ids": covered,
                "covered_canonical_op_names": [op_by_id[op_id]["op_name"] for op_id in covered],
            }
        )
    return partitions


def export_tflite_graph_json(model_content: bytes, path: str, *, component_name: Optional[str] = None, conversion_mode: Optional[str] = None, uses_flex: Optional[bool] = None) -> Dict[str, Any]:
    interpreter = tf.lite.Interpreter(model_content=model_content, num_threads=1)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    tensor_details = interpreter.get_tensor_details()
    raw_ops = interpreter._get_ops_details()

    model_input_ids = {int(detail["index"]) for detail in input_details}
    model_output_ids = {int(detail["index"]) for detail in output_details}

    tensors_by_id: Dict[int, Dict[str, Any]] = {
        int(detail["index"]): _tensor_detail_to_record(detail, model_inputs=model_input_ids, model_outputs=model_output_ids)
        for detail in tensor_details
    }

    canonical_ops: List[Dict[str, Any]] = []
    runtime_delegate_ops: List[Dict[str, Any]] = []
    producer_map: Dict[int, int] = {}

    for raw in raw_ops:
        op_name = str(raw.get("op_name", "UNKNOWN"))
        input_tensors = [int(t) for t in raw.get("inputs", []) if isinstance(t, (int, np.integer)) and int(t) >= 0]
        output_tensors = [int(t) for t in raw.get("outputs", []) if isinstance(t, (int, np.integer)) and int(t) >= 0]
        op_record = {
            "op_id": int(raw.get("index", len(canonical_ops))),
            "op_name": op_name,
            "op_family": _op_family(op_name),
            "category_name": op_name,
            "category_id": op_name,
            "attrs": {},
            "input_tensors": input_tensors,
            "output_tensors": output_tensors,
            "num_inputs": len(input_tensors),
            "num_outputs": len(output_tensors),
        }
        if op_name == "DELEGATE":
            runtime_delegate_ops.append(op_record)
            continue
        canonical_ops.append(op_record)
        for tensor_id in output_tensors:
            producer_map[tensor_id] = op_record["op_id"]

    for op in canonical_ops:
        input_shapes = []
        input_dtypes = []
        output_shapes = []
        output_dtypes = []
        for tensor_id in op["input_tensors"]:
            tensor = tensors_by_id.get(tensor_id)
            if tensor is None:
                continue
            tensor["consumer_ops"].append(op["op_id"])
            input_shapes.append(tensor.get("shape"))
            input_dtypes.append(tensor.get("dtype"))
        for tensor_id in op["output_tensors"]:
            tensor = tensors_by_id.get(tensor_id)
            if tensor is None:
                continue
            tensor["producer_op"] = op["op_id"]
            output_shapes.append(tensor.get("shape"))
            output_dtypes.append(tensor.get("dtype"))
        op["input_shapes"] = input_shapes
        op["input_dtypes"] = input_dtypes
        op["output_shapes"] = output_shapes
        op["output_dtypes"] = output_dtypes

    op_to_tensor_edges: List[Dict[str, int]] = []
    tensor_to_op_edges: List[Dict[str, int]] = []
    op_to_op_edges: List[Dict[str, int]] = []
    for op in canonical_ops:
        for tensor_id in op["output_tensors"]:
            op_to_tensor_edges.append({"src_op": op["op_id"], "dst_tensor": tensor_id})
        for tensor_id in op["input_tensors"]:
            tensor_to_op_edges.append({"src_tensor": tensor_id, "dst_op": op["op_id"]})
            producer = producer_map.get(tensor_id)
            if producer is not None:
                op_to_op_edges.append({"src_op": producer, "dst_op": op["op_id"], "via_tensor": tensor_id})

    tensors = [tensors_by_id[tensor_id] for tensor_id in sorted(tensors_by_id.keys())]
    for tensor in tensors:
        tensor["num_consumers"] = len(tensor["consumer_ops"])

    quantization_mode = _infer_quantization_mode(tensors)
    delegate_partitions = _infer_delegate_partitions(canonical_ops, runtime_delegate_ops, model_input_ids)
    payload = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "graph_meta": {
            "component": component_name,
            "format": "tflite",
            "conversion_mode": conversion_mode,
            "uses_flex": bool(uses_flex),
            "quantization_mode": quantization_mode,
            "num_ops": len(canonical_ops),
            "num_tensors": len(tensors),
            "num_inputs": len(input_details),
            "num_outputs": len(output_details),
        },
        "inputs": _jsonable(input_details),
        "outputs": _jsonable(output_details),
        "tensors": _jsonable(tensors),
        "ops": _jsonable(canonical_ops),
        "edges": {
            "op_to_tensor": op_to_tensor_edges,
            "tensor_to_op": tensor_to_op_edges,
            "op_to_op": op_to_op_edges,
        },
        "runtime_annotations": {
            "delegate_ops": _jsonable(runtime_delegate_ops),
            "delegate_partitions": _jsonable(delegate_partitions),
            "raw_op_count": len(raw_ops),
        },
    }
    write_json(path, payload, indent=2)
    return payload


def inspect_encoder_tflite_bytes(tflite_model: bytes, spec: ModelSpec, feature_spec: FeatureSpec) -> Dict[str, Any]:
    interpreter = tf.lite.Interpreter(model_content=tflite_model, num_threads=1)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    raw_ops = interpreter._get_ops_details()
    ops = [str(d["op_name"]) for d in raw_ops]
    uses_flex = any(op.startswith("Flex") for op in ops)
    warnings: List[str] = []
    try:
        dummy_clip = np.zeros((1, spec.seq, feature_spec.feature_dim), dtype=np.float32)
        dummy_state = zero_state_numpy(spec=spec, batch_size=1, dtype=np.float32)
        all_inputs = [dummy_clip] + ordered_state_arrays_from_dict(spec=spec, state_dict=dummy_state)
        for i, arr in enumerate(all_inputs):
            interpreter.resize_tensor_input(input_details[i]["index"], arr.shape, strict=False)
        interpreter.allocate_tensors()
        for i, arr in enumerate(all_inputs):
            interpreter.set_tensor(interpreter.get_input_details()[i]["index"], arr)
        interpreter.invoke()
        output_details = interpreter.get_output_details()
    except Exception as e:
        warnings.append(repr(e))
    return {
        "input_details": _jsonable(input_details),
        "output_details": _jsonable(output_details),
        "ops": ops,
        "uses_flex": uses_flex,
        "warning": " | ".join(warnings) if warnings else None,
    }


def inspect_head_tflite_bytes(tflite_model: bytes, spec: ModelSpec) -> Dict[str, Any]:
    interpreter = tf.lite.Interpreter(model_content=tflite_model, num_threads=1)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    raw_ops = interpreter._get_ops_details()
    ops = [str(d["op_name"]) for d in raw_ops]
    uses_flex = any(op.startswith("Flex") for op in ops)
    warnings: List[str] = []
    try:
        dummy_embedding = np.zeros((1, spec.encoder_output_dim()), dtype=np.float32)
        interpreter.resize_tensor_input(input_details[0]["index"], dummy_embedding.shape, strict=False)
        interpreter.allocate_tensors()
        interpreter.set_tensor(interpreter.get_input_details()[0]["index"], dummy_embedding)
        interpreter.invoke()
        output_details = interpreter.get_output_details()
    except Exception as e:
        warnings.append(repr(e))
    return {
        "input_details": _jsonable(input_details),
        "output_details": _jsonable(output_details),
        "ops": ops,
        "uses_flex": uses_flex,
        "warning": " | ".join(warnings) if warnings else None,
    }


def recommend_target_runtime(conversion_mode: str, uses_flex: bool) -> str:
    if conversion_mode == "builtin_only" and not uses_flex:
        return "tflite_runtime_or_tensorflow"
    return "tensorflow_full"


def save_conversion_report(path: str, payload: Dict[str, Any]) -> None:
    write_json(path, _jsonable(payload), indent=2)


def _component_failed_report(component_name: str, error: Exception, *, attempts: Optional[Dict[str, Dict[str, Any]]] = None, tflite_path: str, tflite_graph_path: str) -> Dict[str, Any]:
    return {
        "name": component_name,
        "status": "failed",
        "conversion_mode": "failed",
        "attempts": attempts or {},
        "uses_flex": False,
        "ops": [],
        "graph": None,
        "inspection": None,
        "warnings": [],
        "error": repr(error),
        "quantization_mode": "none",
        "tflite_path": tflite_path if os.path.exists(tflite_path) else None,
        "tflite_graph_path": tflite_graph_path if os.path.exists(tflite_graph_path) else None,
    }


def _aggregate_component_conversion(component_name: str, model: tf.keras.Model, tflite_path: str, tflite_graph_path: str, inspect_fn) -> Dict[str, Any]:
    try:
        tflite_model, conversion_mode, attempts = convert_model_with_fallback(model=model)
    except Exception as e:
        return _component_failed_report(component_name, e, attempts={}, tflite_path=tflite_path, tflite_graph_path=tflite_graph_path)

    try:
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)
    except Exception as e:
        return _component_failed_report(component_name, e, attempts=attempts, tflite_path=tflite_path, tflite_graph_path=tflite_graph_path)

    warnings: List[str] = []
    graph_payload: Optional[Dict[str, Any]] = None
    try:
        graph_payload = export_tflite_graph_json(
            tflite_model,
            tflite_graph_path,
            component_name=component_name,
            conversion_mode=conversion_mode,
            uses_flex=(conversion_mode == "builtin_plus_select_tf_ops"),
        )
    except Exception as e:
        warnings.append(f"graph_export: {repr(e)}")

    inspection: Dict[str, Any]
    try:
        inspection = inspect_fn(tflite_model)
        if inspection.get("warning"):
            warnings.append(f"inspection: {inspection['warning']}")
    except Exception as e:
        warning = repr(e)
        warnings.append(f"inspection: {warning}")
        inspection = {"warning": warning, "uses_flex": False, "ops": []}

    graph_ops = []
    if graph_payload is not None:
        graph_ops = [op.get("op_name") for op in graph_payload.get("ops", []) if op.get("op_name")]
    uses_flex = bool(
        inspection.get("uses_flex", False)
        or conversion_mode == "builtin_plus_select_tf_ops"
        or any(str(op).startswith("Flex") for op in graph_ops)
    )
    return {
        "name": component_name,
        "status": "ok",
        "conversion_mode": conversion_mode,
        "attempts": attempts,
        "uses_flex": uses_flex,
        "ops": sorted(set(graph_ops or list(inspection.get("ops", [])))),
        "graph": graph_payload,
        "inspection": inspection,
        "warnings": warnings,
        "error": None,
        "quantization_mode": (graph_payload or {}).get("graph_meta", {}).get("quantization_mode", "none"),
        "tflite_path": tflite_path if os.path.exists(tflite_path) else None,
        "tflite_graph_path": tflite_graph_path if os.path.exists(tflite_graph_path) else None,
    }


def write_model_spec(paths: ArtifactPaths, spec: ModelSpec, feature_spec: FeatureSpec, seed: int, model_id: str) -> None:
    write_json(
        paths.spec_path,
        {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "model_id": model_id,
            "seed": seed,
            "spec": spec.as_key_dict(),
            "feature_spec": feature_spec.__dict__,
        },
        indent=2,
    )


def model_spec_to_stable_key_for_manifest(spec: ModelSpec, feature_spec: FeatureSpec) -> str:
    from rnn_benchlib.sampling.sampler import model_spec_to_stable_key
    return model_spec_to_stable_key(spec=spec, feature_spec=feature_spec)


def _compact_artifact_layout(paths: ArtifactPaths) -> Dict[str, Any]:
    return {
        "model_dir": paths.model_dir,
        "meta_dir": paths.meta_dir,
        "source_dir": paths.source_dir,
        "compiled_dir": paths.compiled_dir,
        "graphs_dir": paths.graphs_dir,
        "reports_dir": paths.reports_dir,
        "benchmark_dir": paths.benchmark_dir,
    }


def _compact_conversion_summary(conversion: ConversionInfo) -> Dict[str, Any]:
    return {
        "status": conversion.status,
        "conversion_mode": conversion.conversion_mode,
        "uses_flex": conversion.uses_flex,
        "target_runtime_recommendation": conversion.target_runtime_recommendation,
        "quantization_mode": conversion.quantization_mode,
        "warning": conversion.warning,
        "components_present": {
            "encoder_tflite": bool(conversion.encoder_tflite_path),
            "head_tflite": bool(conversion.head_tflite_path),
        },
    }


def write_model_manifest(record: ModelRecord) -> None:
    payload = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_layout": "rnn_grouped_v2",
        "model_id": record.model_id,
        "model_key": model_spec_to_stable_key_for_manifest(record.spec, record.feature_spec),
        "seed": record.seed,
        "created_at_utc": record.created_at_utc,
        "conversion": _compact_conversion_summary(record.conversion),
    }
    write_json(record.artifacts.manifest_path, payload, indent=2)


def build_model_record(
    output_root: str,
    seed: int,
    spec: ModelSpec,
    feature_spec: FeatureSpec,
    model_id: Optional[str] = None,
    overwrite: bool = False,
) -> ModelRecord:
    model_id = model_id or model_spec_to_id(spec=spec, feature_spec=feature_spec)
    paths = build_artifact_paths(output_root=output_root, model_id=model_id)
    prepare_model_artifact_dir(paths=paths, overwrite=overwrite)
    write_model_spec(paths, spec, feature_spec, seed, model_id)

    encoder_model, head_model, model_metadata = build_random_initialized_models(spec=spec, feature_spec=feature_spec)
    encoder_model.save(paths.encoder_keras_dir)
    head_model.save(paths.head_keras_dir)

    encoder_component = _aggregate_component_conversion(
        component_name="encoder",
        model=encoder_model,
        tflite_path=paths.encoder_tflite_path,
        tflite_graph_path=paths.encoder_tflite_graph_path,
        inspect_fn=lambda tflite_model: inspect_encoder_tflite_bytes(tflite_model=tflite_model, spec=spec, feature_spec=feature_spec),
    )
    head_component = _aggregate_component_conversion(
        component_name="head",
        model=head_model,
        tflite_path=paths.head_tflite_path,
        tflite_graph_path=paths.head_tflite_graph_path,
        inspect_fn=lambda tflite_model: inspect_head_tflite_bytes(tflite_model=tflite_model, spec=spec),
    )

    component_reports = {"encoder": encoder_component, "head": head_component}
    successful_components = [component for component in component_reports.values() if component.get("status") == "ok"]
    failed_components = [component for component in component_reports.values() if component.get("status") != "ok"]

    encoder_tflite_path = encoder_component.get("tflite_path")
    head_tflite_path = head_component.get("tflite_path")

    if successful_components and not failed_components:
        component_modes = [component["conversion_mode"] for component in successful_components]
        overall_mode = "builtin_only" if all(mode == "builtin_only" for mode in component_modes) else "builtin_plus_select_tf_ops"
        uses_flex = bool(any(component["uses_flex"] for component in successful_components))
        ops = sorted(set(op for component in successful_components for op in component["ops"]))
        warning_parts = [part for component in successful_components for part in component.get("warnings", []) if part]
        conversion = ConversionInfo(
            status="ok",
            conversion_mode=overall_mode,  # type: ignore[arg-type]
            uses_flex=uses_flex,
            ops=ops,
            target_runtime_recommendation=recommend_target_runtime(conversion_mode=overall_mode, uses_flex=uses_flex),  # type: ignore[arg-type]
            error=None,
            warning=" | ".join(warning_parts) if warning_parts else None,
            quantization_mode="none",
            encoder_tflite_path=encoder_tflite_path,
            head_tflite_path=head_tflite_path,
        )
    else:
        successful_modes = [component["conversion_mode"] for component in successful_components if component.get("conversion_mode") != "failed"]
        overall_mode = (
            "builtin_plus_select_tf_ops"
            if any(mode == "builtin_plus_select_tf_ops" for mode in successful_modes)
            else ("builtin_only" if successful_modes else "failed")
        )
        uses_flex = bool(any(component.get("uses_flex", False) for component in successful_components))
        ops = sorted(set(op for component in successful_components for op in component.get("ops", [])))
        error_parts = [part for part in [component.get("error") for component in component_reports.values()] if part]
        warning_parts = [part for component in component_reports.values() for part in component.get("warnings", []) if part]
        status = "partial" if successful_components else "failed"
        conversion = ConversionInfo(
            status=status,
            conversion_mode=overall_mode,  # type: ignore[arg-type]
            uses_flex=uses_flex,
            ops=ops,
            target_runtime_recommendation=(recommend_target_runtime(conversion_mode=overall_mode, uses_flex=uses_flex) if successful_modes else "tensorflow_full"),  # type: ignore[arg-type]
            error=" | ".join(error_parts) if error_parts else None,
            warning=" | ".join(warning_parts) if warning_parts else None,
            quantization_mode="none",
            encoder_tflite_path=encoder_tflite_path,
            head_tflite_path=head_tflite_path,
        )

    compact_component_reports = {
        name: {
            "name": component.get("name"),
            "status": component.get("status"),
            "conversion_mode": component.get("conversion_mode"),
            "attempts": component.get("attempts", {}),
            "uses_flex": component.get("uses_flex", False),
            "ops": component.get("ops", []),
            "warnings": component.get("warnings", []),
            "error": component.get("error"),
            "quantization_mode": component.get("quantization_mode", "none"),
            "tflite_path": component.get("tflite_path"),
            "tflite_graph_path": component.get("tflite_graph_path"),
        }
        for name, component in component_reports.items()
    }
    report = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "model_id": model_id,
        "created_at_utc": utc_now_iso(),
        "tensorflow_version": tf.__version__,
        "components": compact_component_reports,
        "final_conversion": conversion.to_dict(),
    }
    save_conversion_report(paths.conversion_report_path, report)

    graph_record = build_graph_record(
        model_id=model_id,
        spec=spec,
        feature_spec=feature_spec,
        component_reports=component_reports,
        paths=paths,
    )
    if graph_record is not None:
        save_graph_record(paths.graph_record_path, graph_record)

    record = ModelRecord(
        model_id=model_id,
        seed=seed,
        spec=spec,
        feature_spec=feature_spec,
        conversion=conversion,
        artifacts=paths,
        created_at_utc=utc_now_iso(),
    )
    write_model_manifest(record)
    return record
