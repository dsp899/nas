from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np

from rnn_benchlib.latency_gnn.dataset import ALL_TARGETS, AggregatedLatencySample


GRAPH_CAT_NAMES = [
    "graph_rnn",
    "graph_direction",
    "graph_memory",
    "graph_decision",
    "graph_decision_input",
    "graph_pooling",
    "runtime_kind",
    "runtime_hw",
    "runtime_exec",
    "runtime_quant",
    "runtime_encoder_delegate",
    "runtime_head_delegate",
]


def _make_vocab(values: Iterable[str]) -> Dict[str, int]:
    ordered = sorted({str(v) for v in values if v is not None})
    return {token: idx for idx, token in enumerate(["<unk>"] + ordered)}


@dataclass
class FeatureSpecBundle:
    op_num_dim: int
    tensor_num_dim: int
    graph_num_dim: int
    targets: list[str]
    vocabs: Dict[str, Dict[str, int]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "op_num_dim": self.op_num_dim,
            "tensor_num_dim": self.tensor_num_dim,
            "graph_num_dim": self.graph_num_dim,
            "targets": list(self.targets),
            "vocabs": {name: dict(vocab) for name, vocab in self.vocabs.items()},
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FeatureSpecBundle":
        return cls(
            op_num_dim=int(payload["op_num_dim"]),
            tensor_num_dim=int(payload["tensor_num_dim"]),
            graph_num_dim=int(payload["graph_num_dim"]),
            targets=list(payload["targets"]),
            vocabs={name: {k: int(v) for k, v in vocab.items()} for name, vocab in payload["vocabs"].items()},
        )


class LatencyGnnFeaturizer:
    def __init__(self, spec: FeatureSpecBundle) -> None:
        self.spec = spec

    @staticmethod
    def build(samples: Sequence[AggregatedLatencySample]) -> "LatencyGnnFeaturizer":
        names = [
            "op_category", "op_family", "op_component",
            "tensor_dtype", "tensor_component", "tensor_quant", "tensor_role",
            *GRAPH_CAT_NAMES,
        ]
        buckets = {name: [] for name in names}
        for sample in samples:
            graph = sample.graph_record
            for op in graph.get("op_nodes", []):
                buckets["op_category"].append(str(op.get("category_id") or op.get("category_name") or "unknown"))
                buckets["op_family"].append(str(op.get("op_family") or "unknown"))
                buckets["op_component"].append(str(op.get("component") or "unknown"))
            for tensor in graph.get("tensor_nodes", []):
                buckets["tensor_dtype"].append(str(tensor.get("dtype") or "unknown"))
                buckets["tensor_component"].append(str(tensor.get("component") or "unknown"))
                buckets["tensor_quant"].append(str(tensor.get("quantization_mode") or "none"))
                buckets["tensor_role"].append(_tensor_role_name(tensor))
            gf, rf = sample.graph_features, sample.runtime_features
            buckets["graph_rnn"].append(str(gf.get("rnn", "unknown")))
            buckets["graph_direction"].append(str(gf.get("direction", "unknown")))
            buckets["graph_memory"].append(str(gf.get("memory_mode", "unknown")))
            buckets["graph_decision"].append(str(gf.get("video_decision", "unknown")))
            buckets["graph_decision_input"].append(str(gf.get("video_decision_input", "unknown")))
            buckets["graph_pooling"].append(str(gf.get("pooling", "unknown")))
            buckets["runtime_kind"].append(str(rf.get("runtime_kind", "unknown")))
            buckets["runtime_hw"].append(str(rf.get("hardware_target", "unknown")))
            buckets["runtime_exec"].append(str(rf.get("execution_site", "unknown")))
            buckets["runtime_quant"].append(str(rf.get("quantization_mode", "none")))
            buckets["runtime_encoder_delegate"].append(str(rf.get("encoder_delegate_backend", "unknown")))
            buckets["runtime_head_delegate"].append(str(rf.get("head_delegate_backend", "unknown")))
        spec = FeatureSpecBundle(
            op_num_dim=8,
            tensor_num_dim=8,
            graph_num_dim=18,
            targets=list(ALL_TARGETS),
            vocabs={name: _make_vocab(values) for name, values in buckets.items()},
        )
        return LatencyGnnFeaturizer(spec)

    def _index(self, vocab_name: str, value: Any) -> int:
        return self.spec.vocabs[vocab_name].get(str(value), 0)

    def encode_sample(self, sample: AggregatedLatencySample) -> Dict[str, Any]:
        graph = sample.graph_record
        op_nodes = graph.get("op_nodes", [])
        tensor_nodes = graph.get("tensor_nodes", [])
        op_id_map = {str(node["op_id"]): idx for idx, node in enumerate(op_nodes)}
        tensor_id_map = {str(node["tensor_id"]): idx for idx, node in enumerate(tensor_nodes)}

        op_cat, op_family, op_component, op_numeric = [], [], [], []
        for op in op_nodes:
            feats = op.get("features", {})
            op_cat.append(self._index("op_category", op.get("category_id") or op.get("category_name") or "unknown"))
            op_family.append(self._index("op_family", op.get("op_family") or "unknown"))
            op_component.append(self._index("op_component", op.get("component") or "unknown"))
            op_numeric.append([
                _log1p(feats.get("num_inputs", 0)),
                _log1p(feats.get("num_outputs", 0)),
                float(bool(feats.get("is_control_flow", False))),
                float(bool(feats.get("is_flex_op", False))),
                float(bool(feats.get("is_state_boundary", False))),
                float(bool(feats.get("is_bridge_boundary", False))),
                _shape_stat(op.get("input_shapes", []), "rank"),
                _shape_stat(op.get("output_shapes", []), "elements"),
            ])

        tensor_dtype, tensor_component, tensor_quant, tensor_role, tensor_numeric = [], [], [], [], []
        for tensor in tensor_nodes:
            tensor_dtype.append(self._index("tensor_dtype", tensor.get("dtype") or "unknown"))
            tensor_component.append(self._index("tensor_component", tensor.get("component") or "unknown"))
            tensor_quant.append(self._index("tensor_quant", tensor.get("quantization_mode") or "none"))
            tensor_role.append(self._index("tensor_role", _tensor_role_name(tensor)))
            tensor_numeric.append([
                _log1p(tensor.get("rank", 0)),
                _log1p(tensor.get("num_elements", 0)),
                _log1p(tensor.get("num_consumers", 0)),
                float(bool(tensor.get("is_model_input", False))),
                float(bool(tensor.get("is_model_output", False))),
                float(bool(tensor.get("is_state_tensor", False))),
                float(bool(tensor.get("is_bridge_tensor", False))),
                _shape_stat([tensor.get("shape", [])], "known_ratio"),
            ])

        gf, rf = sample.graph_features, sample.runtime_features
        graph_numeric = np.asarray([
            _log1p(gf.get("layers", 0)),
            _log1p(gf.get("seq", 0)),
            _log1p(gf.get("clip_embedding_dim", 0)),
            _log1p(gf.get("head_units", 0)),
            _log1p(gf.get("num_classes", 0)),
            _log1p(gf.get("feature_dim", 0)),
            _log1p(gf.get("video_steps", 0)),
            _log1p(gf.get("clips_per_video", 0)),
            _log1p(gf.get("total_units", 0)),
            _log1p(gf.get("max_units", 0)),
            _log1p(gf.get("num_op_nodes", 0)),
            _log1p(gf.get("num_tensor_nodes", 0)),
            _log1p(gf.get("num_op_to_tensor_edges", 0)),
            _log1p(gf.get("num_tensor_to_op_edges", 0)),
            _log1p(gf.get("num_op_to_op_edges", 0)),
            _log1p(gf.get("num_bridge_tensors", 0)),
            _log1p(gf.get("num_delegate_partitions", 0)),
            _log1p(rf.get("threads", 1)),
        ], dtype=np.float32)
        graph_cat = {
            "graph_rnn": self._index("graph_rnn", gf.get("rnn", "unknown")),
            "graph_direction": self._index("graph_direction", gf.get("direction", "unknown")),
            "graph_memory": self._index("graph_memory", gf.get("memory_mode", "unknown")),
            "graph_decision": self._index("graph_decision", gf.get("video_decision", "unknown")),
            "graph_decision_input": self._index("graph_decision_input", gf.get("video_decision_input", "unknown")),
            "graph_pooling": self._index("graph_pooling", gf.get("pooling", "unknown")),
            "runtime_kind": self._index("runtime_kind", rf.get("runtime_kind", "unknown")),
            "runtime_hw": self._index("runtime_hw", rf.get("hardware_target", "unknown")),
            "runtime_exec": self._index("runtime_exec", rf.get("execution_site", "unknown")),
            "runtime_quant": self._index("runtime_quant", rf.get("quantization_mode", "none")),
            "runtime_encoder_delegate": self._index("runtime_encoder_delegate", rf.get("encoder_delegate_backend", "unknown")),
            "runtime_head_delegate": self._index("runtime_head_delegate", rf.get("head_delegate_backend", "unknown")),
        }

        return {
            "sample_id": sample.sample_id,
            "graph_id": sample.graph_id,
            "model_id": sample.model_id,
            "op_cat": np.asarray(op_cat or [0], dtype=np.int32),
            "op_family": np.asarray(op_family or [0], dtype=np.int32),
            "op_component": np.asarray(op_component or [0], dtype=np.int32),
            "op_numeric": np.asarray(op_numeric or [[0.0] * self.spec.op_num_dim], dtype=np.float32),
            "tensor_dtype": np.asarray(tensor_dtype or [0], dtype=np.int32),
            "tensor_component": np.asarray(tensor_component or [0], dtype=np.int32),
            "tensor_quant": np.asarray(tensor_quant or [0], dtype=np.int32),
            "tensor_role": np.asarray(tensor_role or [0], dtype=np.int32),
            "tensor_numeric": np.asarray(tensor_numeric or [[0.0] * self.spec.tensor_num_dim], dtype=np.float32),
            "graph_numeric": graph_numeric,
            "graph_cat": {key: np.asarray(value, dtype=np.int32) for key, value in graph_cat.items()},
            "encoder_op_mask": np.asarray([bool(node.get("component") == "encoder") for node in op_nodes] or [False], dtype=np.bool_),
            "head_op_mask": np.asarray([bool(node.get("component") == "head") for node in op_nodes] or [False], dtype=np.bool_),
            "bridge_tensor_mask": np.asarray([bool(node.get("component") == "bridge") for node in tensor_nodes] or [False], dtype=np.bool_),
            "edge_index_op_to_tensor": _edge_index(graph.get("edges", {}).get("op_to_tensor", []), op_id_map, tensor_id_map, "src_op_id", "dst_tensor_id"),
            "edge_index_tensor_to_op": _edge_index(graph.get("edges", {}).get("tensor_to_op", []), tensor_id_map, op_id_map, "src_tensor_id", "dst_op_id"),
            "edge_index_op_to_op": _edge_index(graph.get("edges", {}).get("op_to_op", []), op_id_map, op_id_map, "src_op_id", "dst_op_id"),
            "targets": np.asarray([float(sample.targets[name]) for name in self.spec.targets], dtype=np.float32),
            "sample_weight": np.asarray(float(sample.weight), dtype=np.float32),
            "target_names": list(self.spec.targets),
            "metadata": {"runtime_features": dict(sample.runtime_features), "graph_features": dict(sample.graph_features)},
        }

    def encode_many(self, samples: Sequence[AggregatedLatencySample]) -> List[Dict[str, Any]]:
        return [self.encode_sample(sample) for sample in samples]



def _edge_index(edges: Sequence[Mapping[str, Any]], src_map: Mapping[str, int], dst_map: Mapping[str, int], src_key: str, dst_key: str) -> np.ndarray:
    pairs = []
    for edge in edges:
        src = src_map.get(str(edge.get(src_key)))
        dst = dst_map.get(str(edge.get(dst_key)))
        if src is None or dst is None:
            continue
        pairs.append((src, dst))
    if not pairs:
        return np.zeros((2, 0), dtype=np.int32)
    return np.asarray(pairs, dtype=np.int32).T.copy()



def _tensor_role_name(tensor: Mapping[str, Any]) -> str:
    if tensor.get("is_bridge_tensor"):
        return "bridge"
    if tensor.get("is_state_tensor"):
        return "state"
    if tensor.get("is_model_input"):
        return "model_input"
    if tensor.get("is_model_output"):
        return "model_output"
    return str(tensor.get("semantic_role") or "internal")



def _shape_stat(shapes: Sequence[Any], which: str) -> float:
    dims = []
    known = 0
    total = 0
    for shape in shapes:
        if not isinstance(shape, (list, tuple)):
            continue
        for dim in shape:
            total += 1
            try:
                val = int(dim)
            except Exception:
                continue
            if val >= 0:
                known += 1
                dims.append(val)
    if which == "rank":
        return _log1p(len(dims))
    if which == "elements":
        if not dims:
            return 0.0
        prod = 1
        for val in dims:
            prod *= max(1, val)
        return _log1p(prod)
    if which == "known_ratio":
        return float(known / max(1, total))
    return 0.0



def _log1p(value: Any) -> float:
    try:
        return float(math.log1p(max(0.0, float(value))))
    except Exception:
        return 0.0
