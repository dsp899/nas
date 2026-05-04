from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from rnn_benchlib.config.schemas import ArtifactPaths, FeatureSpec, GRAPH_SCHEMA_VERSION, ModelSpec
from rnn_benchlib.storage.jsonl import write_json


def _state_layout(spec: ModelSpec) -> List[Dict[str, Any]]:
    layout: List[Dict[str, Any]] = []
    for layer_index, units in enumerate(spec.normalized_units_list()):
        if spec.rnn == "lstm":
            kinds = ["h", "c"]
        else:
            kinds = ["h"]
        directions = ["fw", "bw"] if spec.direction == "bidirectional" else ["fw"]
        for direction in directions:
            for kind in kinds:
                layout.append(
                    {
                        "name": f"layer{layer_index}_{direction}_{kind}",
                        "units": units,
                        "kind": kind,
                        "layer_index": layer_index,
                        "direction": direction,
                    }
                )
    return layout


def _component_backend_hint(component_report: Dict[str, Any]) -> str:
    if component_report.get("uses_flex"):
        return "flex"
    if component_report.get("conversion_mode") == "builtin_only":
        return "xnnpack_or_default"
    return "tflite_or_default"


def _namespace_op_id(component: str, op_id: int) -> str:
    return f"op:{component}:{op_id}"


def _namespace_tensor_id(component: str, tensor_id: int) -> str:
    return f"tensor:{component}:{tensor_id}"


def _num_elements(shape: List[Any]) -> Optional[int]:
    try:
        total = 1
        for dim in shape:
            dim_i = int(dim)
            if dim_i < 0:
                return None
            total *= dim_i
        return int(total)
    except Exception:
        return None


def _merge_component_graph(component: str, payload: Dict[str, Any], spec: ModelSpec) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]], Dict[str, Any], Dict[str, str]]:
    input_ids = {int(item["index"]) for item in payload.get("inputs", [])}
    output_ids = {int(item["index"]) for item in payload.get("outputs", [])}

    op_nodes: List[Dict[str, Any]] = []
    tensor_nodes: List[Dict[str, Any]] = []
    op_to_tensor: List[Dict[str, Any]] = []
    tensor_to_op: List[Dict[str, Any]] = []
    op_to_op: List[Dict[str, Any]] = []
    op_id_map: Dict[int, str] = {}
    tensor_id_map: Dict[int, str] = {}

    for op in payload.get("ops", []):
        old_id = int(op["op_id"])
        new_id = _namespace_op_id(component, old_id)
        op_id_map[old_id] = new_id
        op_nodes.append(
            {
                "op_id": new_id,
                "component": component,
                "topo_index": old_id,
                "op_name": op.get("op_name"),
                "op_family": op.get("op_family", "builtin"),
                "category_name": op.get("category_name", op.get("op_name")),
                "category_id": op.get("category_id", op.get("op_name")),
                "attrs": op.get("attrs", {}),
                "input_tensor_ids": [_namespace_tensor_id(component, int(t)) for t in op.get("input_tensors", [])],
                "output_tensor_ids": [_namespace_tensor_id(component, int(t)) for t in op.get("output_tensors", [])],
                "features": {
                    "num_inputs": op.get("num_inputs", len(op.get("input_tensors", []))),
                    "num_outputs": op.get("num_outputs", len(op.get("output_tensors", []))),
                    "is_control_flow": op.get("op_name") in {"WHILE", "IF"},
                    "is_flex_op": op.get("op_family") == "flex",
                    "is_state_boundary": False,
                    "is_bridge_boundary": False,
                },
                "input_shapes": op.get("input_shapes", []),
                "input_dtypes": op.get("input_dtypes", []),
                "output_shapes": op.get("output_shapes", []),
                "output_dtypes": op.get("output_dtypes", []),
            }
        )

    for tensor in payload.get("tensors", []):
        old_id = int(tensor["tensor_id"])
        new_id = _namespace_tensor_id(component, old_id)
        tensor_id_map[old_id] = new_id
        name = str(tensor.get("name", f"tensor_{old_id}"))
        is_bridge_tensor = component == "encoder" and name.endswith("clip_embedding")
        tensor_nodes.append(
            {
                "tensor_id": new_id,
                "component": component,
                "name": name,
                "shape": tensor.get("shape", []),
                "shape_signature": tensor.get("shape_signature", tensor.get("shape", [])),
                "rank": tensor.get("rank"),
                "num_elements": tensor.get("num_elements") if tensor.get("num_elements") is not None else _num_elements(tensor.get("shape", [])),
                "dtype": tensor.get("dtype"),
                "quantization_mode": tensor.get("quantization_mode", "none"),
                "quantization_parameters": tensor.get("quantization_parameters", {}),
                "is_model_input": bool(tensor.get("is_model_input", old_id in input_ids)),
                "is_model_output": bool(tensor.get("is_model_output", old_id in output_ids)),
                "is_state_tensor": name.startswith("serving_default_layer") or name.startswith("StatefulPartitionedCall:"),
                "is_bridge_tensor": is_bridge_tensor,
                "producer_op_id": None if tensor.get("producer_op") is None else _namespace_op_id(component, int(tensor["producer_op"])),
                "consumer_op_ids": [_namespace_op_id(component, int(op_id)) for op_id in tensor.get("consumer_ops", [])],
                "num_consumers": tensor.get("num_consumers", len(tensor.get("consumer_ops", []))),
            }
        )

    for edge in (payload.get("edges", {}) or {}).get("op_to_tensor", []):
        op_to_tensor.append({"src_op_id": _namespace_op_id(component, int(edge["src_op"])), "dst_tensor_id": _namespace_tensor_id(component, int(edge["dst_tensor"])), "role": "produces"})
    for edge in (payload.get("edges", {}) or {}).get("tensor_to_op", []):
        tensor_to_op.append({"src_tensor_id": _namespace_tensor_id(component, int(edge["src_tensor"])), "dst_op_id": _namespace_op_id(component, int(edge["dst_op"])), "role": "consumes"})
    for edge in (payload.get("edges", {}) or {}).get("op_to_op", []):
        op_to_op.append({"src_op_id": _namespace_op_id(component, int(edge["src_op"])), "dst_op_id": _namespace_op_id(component, int(edge["dst_op"])), "via_tensor_id": _namespace_tensor_id(component, int(edge["via_tensor"]))})

    component_meta = {
        "component_id": component,
        "source_format": (payload.get("graph_meta") or {}).get("format", "tflite"),
        "entry_op_ids": [node["op_id"] for node in op_nodes[:1]],
        "exit_op_ids": [node["op_id"] for node in op_nodes[-1:]],
        "input_tensor_ids": [tensor_id_map[i] for i in sorted(input_ids) if i in tensor_id_map],
        "output_tensor_ids": [tensor_id_map[i] for i in sorted(output_ids) if i in tensor_id_map],
    }
    return op_nodes, tensor_nodes, {"op_to_tensor": op_to_tensor, "tensor_to_op": tensor_to_op, "op_to_op": op_to_op}, component_meta, op_id_map


def _build_bridge_info(spec: ModelSpec, encoder_tensors: List[Dict[str, Any]], head_tensors: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    encoder_bridge = next((tensor for tensor in encoder_tensors if tensor.get("is_bridge_tensor")), None)
    head_bridge = next((tensor for tensor in head_tensors if tensor.get("is_model_input")), None)
    if encoder_bridge is None or head_bridge is None:
        return [], [], []
    bridge_tensor_id = "tensor:bridge:clip_embedding"
    bridge_tensor = {
        "tensor_id": bridge_tensor_id,
        "component": "bridge",
        "name": "clip_embedding",
        "shape": encoder_bridge.get("shape", []),
        "shape_signature": encoder_bridge.get("shape_signature", encoder_bridge.get("shape", [])),
        "rank": encoder_bridge.get("rank"),
        "num_elements": encoder_bridge.get("num_elements"),
        "dtype": encoder_bridge.get("dtype"),
        "quantization_mode": encoder_bridge.get("quantization_mode", "none"),
        "quantization_parameters": encoder_bridge.get("quantization_parameters", {}),
        "is_model_input": False,
        "is_model_output": False,
        "is_state_tensor": False,
        "is_bridge_tensor": True,
        "producer_op_id": encoder_bridge.get("producer_op_id"),
        "consumer_op_ids": head_bridge.get("consumer_op_ids", []),
        "num_consumers": len(head_bridge.get("consumer_op_ids", [])),
        "semantic_role": "clip_embedding",
    }
    bridge_info = [{"tensor_id": bridge_tensor_id, "from_component": "encoder", "to_component": "head", "semantic_role": "clip_embedding"}]
    bridge_edges = []
    if encoder_bridge.get("producer_op_id"):
        bridge_edges.append({"src_op_id": encoder_bridge["producer_op_id"], "dst_tensor_id": bridge_tensor_id, "role": "produces_bridge"})
    for consumer in head_bridge.get("consumer_op_ids", []):
        bridge_edges.append({"src_tensor_id": bridge_tensor_id, "dst_op_id": consumer, "role": "consumes_bridge"})
    return bridge_info, [bridge_tensor], bridge_edges


def _merge_delegate_partitions(component_name: str, component_report: Dict[str, Any], op_id_map: Dict[int, str], tensor_prefix: str) -> List[Dict[str, Any]]:
    graph = component_report.get("graph") or {}
    partitions = (graph.get("runtime_annotations") or {}).get("delegate_partitions", [])
    merged: List[Dict[str, Any]] = []
    for part in partitions:
        merged.append(
            {
                "component": component_name,
                "partition_id": part.get("partition_id"),
                "delegate_backend_hint": part.get("delegate_backend_hint") or _component_backend_hint(component_report),
                "delegate_op_id": part.get("delegate_op_id"),
                "covered_canonical_op_ids": [op_id_map[int(op_id)] for op_id in part.get("covered_canonical_op_ids", []) if int(op_id) in op_id_map],
                "covered_canonical_op_names": list(part.get("covered_canonical_op_names", [])),
                "boundary_input_tensor_ids": [f"{tensor_prefix}:{int(tid)}" for tid in part.get("boundary_input_tensor_ids", [])],
                "boundary_output_tensor_ids": [f"{tensor_prefix}:{int(tid)}" for tid in part.get("boundary_output_tensor_ids", [])],
            }
        )
    return merged


def build_graph_record(*, model_id: str, spec: ModelSpec, feature_spec: FeatureSpec, component_reports: Dict[str, Dict[str, Any]], paths: ArtifactPaths) -> Optional[Dict[str, Any]]:
    encoder_graph = (component_reports.get("encoder") or {}).get("graph")
    head_graph = (component_reports.get("head") or {}).get("graph")
    if not encoder_graph or not head_graph:
        return None

    encoder_ops, encoder_tensors, encoder_edges, encoder_meta, encoder_op_id_map = _merge_component_graph("encoder", encoder_graph, spec)
    head_ops, head_tensors, head_edges, head_meta, head_op_id_map = _merge_component_graph("head", head_graph, spec)
    bridge_info, bridge_tensor_nodes, bridge_edges = _build_bridge_info(spec, encoder_tensors, head_tensors)

    op_nodes = encoder_ops + head_ops
    tensor_nodes = encoder_tensors + head_tensors + bridge_tensor_nodes
    op_to_tensor = encoder_edges["op_to_tensor"] + head_edges["op_to_tensor"]
    tensor_to_op = encoder_edges["tensor_to_op"] + head_edges["tensor_to_op"]
    op_to_op = encoder_edges["op_to_op"] + head_edges["op_to_op"]

    for edge in bridge_edges:
        if "src_op_id" in edge:
            op_to_tensor.append(edge)
        elif "src_tensor_id" in edge:
            tensor_to_op.append(edge)
    bridge_tensor = bridge_tensor_nodes[0] if bridge_tensor_nodes else None
    if bridge_tensor is not None and bridge_tensor.get("producer_op_id"):
        for consumer_op_id in bridge_tensor.get("consumer_op_ids", []):
            op_to_op.append({"src_op_id": bridge_tensor["producer_op_id"], "dst_op_id": consumer_op_id, "via_tensor_id": bridge_tensor["tensor_id"]})

    delegate_partitions = []
    delegate_partitions.extend(_merge_delegate_partitions("encoder", component_reports.get("encoder") or {}, encoder_op_id_map, "tensor:encoder"))
    delegate_partitions.extend(_merge_delegate_partitions("head", component_reports.get("head") or {}, head_op_id_map, "tensor:head"))

    graph_record = {
        "schema_version": GRAPH_SCHEMA_VERSION,
        "artifact_schema_version": GRAPH_SCHEMA_VERSION,
        "graph_id": f"graph_{model_id}",
        "model_id": model_id,
        "task": "video_clip_classification",
        "prediction_scope": {
            "primary_target": "latency_clip_e2e_wall_ms",
            "unit": "clip_window",
            "secondary_targets": ["latency_clip_e2e_sum_ms", "latency_clip_encoder_ms", "latency_clip_bridge_ms", "latency_clip_head_ms", "latency_video_e2e_wall_ms", "latency_video_e2e_sum_ms"],
        },
        "model_config": {
            "feature_spec": {
                "source": feature_spec.source,
                "feature_dim": feature_spec.feature_dim,
                "video_steps": feature_spec.video_steps,
                "frame_size": feature_spec.frame_size,
                "pooling": feature_spec.pooling,
            },
            "encoder_spec": {
                "rnn": spec.rnn,
                "layers": spec.layers,
                "units": spec.normalized_units_list(),
                "direction": spec.direction,
                "memory_mode": spec.memory_mode,
                "seq": spec.seq,
                "clip_embedding_dim": spec.encoder_output_dim(),
                "state_layout": _state_layout(spec),
            },
            "head_spec": {"head_units": spec.head_units, "num_classes": spec.num_classes, "output_name": "clip_logits"},
            "decision_spec": {"video_decision": spec.video_decision, "video_decision_input": spec.video_decision_input},
        },
        "graph_meta": {
            "graph_kind": "merged_encoder_head_tflite",
            "representation": "heterogeneous_bipartite",
            "node_types": ["op", "tensor"],
            "num_op_nodes": len(op_nodes),
            "num_tensor_nodes": len(tensor_nodes),
            "num_components": 2,
        },
        "components": {"encoder": encoder_meta, "head": head_meta},
        "bridge_tensors": bridge_info,
        "op_nodes": op_nodes,
        "tensor_nodes": tensor_nodes,
        "edges": {"op_to_tensor": op_to_tensor, "tensor_to_op": tensor_to_op, "op_to_op": op_to_op},
        "execution_annotations": {
            "quantization_mode": "none",
            "runtime_capabilities": {
                "encoder": {
                    "conversion_mode": (component_reports.get("encoder") or {}).get("conversion_mode"),
                    "uses_flex": bool((component_reports.get("encoder") or {}).get("uses_flex", False)),
                    "target_runtime_recommendation": "tensorflow_full" if (component_reports.get("encoder") or {}).get("uses_flex") else "tflite_runtime_or_tensorflow",
                    "delegate_backend_hint": _component_backend_hint(component_reports.get("encoder") or {}),
                },
                "head": {
                    "conversion_mode": (component_reports.get("head") or {}).get("conversion_mode"),
                    "uses_flex": bool((component_reports.get("head") or {}).get("uses_flex", False)),
                    "target_runtime_recommendation": "tensorflow_full" if (component_reports.get("head") or {}).get("uses_flex") else "tflite_runtime_or_tensorflow",
                    "delegate_backend_hint": _component_backend_hint(component_reports.get("head") or {}),
                },
            },
            "delegate_partitions": delegate_partitions,
        },
    }
    return graph_record


def save_graph_record(path: str, payload: Dict[str, Any]) -> None:
    write_json(path, payload, indent=2)
