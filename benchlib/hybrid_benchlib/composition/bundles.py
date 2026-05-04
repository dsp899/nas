from __future__ import annotations

from typing import Any, Dict
from cnn_benchlib.storage.layout import build_artifact_paths as build_cnn_paths
from cnn_benchlib.storage.registry import CnnModelRegistry
from rnn_benchlib.storage.registry import ModelRegistry


def load_cnn_bundle(output_root: str, model_id: str) -> Dict[str, Any]:
    paths = build_cnn_paths(output_root, model_id)
    registry = CnnModelRegistry(paths.registry_path)
    return registry.require(model_id)


def load_rnn_bundle(output_root: str, model_id: str) -> Dict[str, Any]:
    registry = ModelRegistry(f"{output_root}/registry.json")
    records = {r["model_id"]: r for r in registry.list_model_records_dicts()}
    if model_id not in records:
        raise KeyError(f"No existe modelo RNN con id {model_id!r}")
    return records[model_id]
