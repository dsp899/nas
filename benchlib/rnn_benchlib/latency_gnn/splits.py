from __future__ import annotations

import hashlib
import json
import random
from typing import Dict, List, Sequence

from rnn_benchlib.latency_gnn.artifacts import split_path
from rnn_benchlib.latency_gnn.dataset import AggregatedLatencySample
from rnn_benchlib.storage.jsonl import read_json, write_json


def _normalize_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> tuple[float, float, float]:
    total = float(train_ratio) + float(val_ratio) + float(test_ratio)
    if total <= 0:
        raise ValueError("La suma de train/val/test debe ser > 0")
    return float(train_ratio) / total, float(val_ratio) / total, float(test_ratio) / total


def create_split_payload(
    samples: Sequence[AggregatedLatencySample],
    lot_id: str,
    dataset_id: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict:
    train_ratio, val_ratio, test_ratio = _normalize_ratios(train_ratio, val_ratio, test_ratio)
    graph_ids = sorted({sample.graph_id for sample in samples})
    rng = random.Random(seed)
    rng.shuffle(graph_ids)
    n = len(graph_ids)
    n_train = max(1, int(round(n * train_ratio))) if n else 0
    n_val = int(round(n * val_ratio))
    if n_train + n_val > n:
        n_val = max(0, n - n_train)
    train_graphs = set(graph_ids[:n_train])
    val_graphs = set(graph_ids[n_train:n_train + n_val])
    test_graphs = set(graph_ids[n_train + n_val:])
    if not test_graphs and val_graphs:
        test_graphs.add(sorted(val_graphs)[-1])
        val_graphs.remove(sorted(val_graphs)[-1])
    split_id = hashlib.sha1(json.dumps({"lot_id": lot_id, "dataset_id": dataset_id, "seed": seed, "ratios": [train_ratio, val_ratio, test_ratio], "graphs": graph_ids}, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    train_ids = [sample.sample_id for sample in samples if sample.graph_id in train_graphs]
    val_ids = [sample.sample_id for sample in samples if sample.graph_id in val_graphs]
    test_ids = [sample.sample_id for sample in samples if sample.graph_id in test_graphs]
    return {
        "schema_version": "rnn_latency_gnn_split_v2",
        "split_id": split_id,
        "lot_id": lot_id,
        "dataset_id": dataset_id,
        "benchmark_id": samples[0].benchmark_id if samples else None,
        "seed": seed,
        "ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "graph_counts": {"train": len(train_graphs), "val": len(val_graphs), "test": len(test_graphs)},
        "sample_counts": {"train": len(train_ids), "val": len(val_ids), "test": len(test_ids)},
        "train_sample_ids": train_ids,
        "val_sample_ids": val_ids,
        "test_sample_ids": test_ids,
    }


def materialize_split(
    output_root: str,
    lot_id: str,
    dataset_id: str,
    samples: Sequence[AggregatedLatencySample],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict:
    payload = create_split_payload(samples, lot_id, dataset_id, train_ratio, val_ratio, test_ratio, seed)
    write_json(str(split_path(output_root, lot_id, dataset_id, payload["split_id"])), payload, indent=2)
    return payload


def load_split(output_root: str, lot_id: str, dataset_id: str, split_id: str) -> Dict:
    path = split_path(output_root, lot_id, dataset_id, split_id)
    payload = read_json(str(path), default=None)
    if payload is None:
        raise FileNotFoundError(f"No existe el split solicitado: {path}")
    return payload


def apply_split(samples: Sequence[AggregatedLatencySample], split_payload: Dict) -> Dict[str, List[AggregatedLatencySample]]:
    ids = {
        "train": set(split_payload.get("train_sample_ids", [])),
        "val": set(split_payload.get("val_sample_ids", [])),
        "test": set(split_payload.get("test_sample_ids", [])),
    }
    return {
        name: [sample for sample in samples if sample.sample_id in sample_ids]
        for name, sample_ids in ids.items()
    }
