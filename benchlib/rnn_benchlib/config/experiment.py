from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Optional

from rnn_benchlib.config.schemas import ExperimentConfig, SearchSpace
from rnn_benchlib.storage.jsonl import read_json, write_json


_DATASET_NUM_CLASSES = {
    "ucf50": 50,
    "ucf101": 101,
}


def default_experiment_config() -> ExperimentConfig:
    return ExperimentConfig(
        dataset_profile="ucf101",
        num_classes=101,
        feature_dim=256,
        video_steps=36,
        search_space=SearchSpace(),
    )


def _coerce_search_space(raw: Dict[str, Any]) -> SearchSpace:
    defaults = default_experiment_config().search_space
    payload = {}
    for field_name in defaults.__dataclass_fields__.keys():
        value = raw.get(field_name, getattr(defaults, field_name))
        payload[field_name] = tuple(value)
    return SearchSpace(**payload)


def load_experiment_config(path: Optional[str]) -> ExperimentConfig:
    if path is None:
        return default_experiment_config()

    raw = read_json(path, default=None)
    if raw is None:
        raise FileNotFoundError(f"No existe experiment_config: {path}")

    dataset_profile = raw.get("dataset_profile", "ucf101")
    if dataset_profile not in _DATASET_NUM_CLASSES:
        raise ValueError(f"dataset_profile no soportado: {dataset_profile}")

    search_space = _coerce_search_space(raw.get("search_space", {}))
    num_classes = int(raw.get("num_classes", _DATASET_NUM_CLASSES[dataset_profile]))

    return ExperimentConfig(
        dataset_profile=dataset_profile,
        num_classes=num_classes,
        feature_dim=int(raw.get("feature_dim", 256)),
        video_steps=int(raw.get("video_steps", 36)),
        search_space=search_space,
    )


def apply_overrides(
    config: ExperimentConfig,
    *,
    feature_dim: Optional[int] = None,
    video_steps: Optional[int] = None,
) -> ExperimentConfig:
    payload = {}
    if feature_dim is not None:
        payload["feature_dim"] = int(feature_dim)
    if video_steps is not None:
        payload["video_steps"] = int(video_steps)
    if not payload:
        return config
    return replace(config, **payload)


def save_experiment_config(path: str, config: ExperimentConfig) -> None:
    write_json(path, config.to_dict(), indent=2)
