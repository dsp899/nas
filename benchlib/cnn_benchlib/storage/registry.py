from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional
from benchlib_common.artifacts.ids import stable_hash
from benchlib_common.io.jsonl import read_json, write_json
from cnn_benchlib.config.schemas import CnnArtifactRecord, CnnExperimentConfig, CnnModelSpec


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def model_spec_to_id(spec: CnnModelSpec, experiment: CnnExperimentConfig) -> str:
    payload = {**spec.to_key_dict(), "dataset_profile": experiment.dataset_profile, "num_classes": experiment.num_classes}
    return f"cnn_{stable_hash(payload)}"


class CnnModelRegistry:
    def __init__(self, registry_path: str):
        self.registry_path = registry_path
        self._data = self._load()

    def _load(self) -> Dict[str, Any]:
        data = read_json(self.registry_path, default=None)
        if data is None:
            return {"version": 1, "created_at_utc": utc_now_iso(), "updated_at_utc": utc_now_iso(), "models_by_id": {}}
        data.setdefault("version", 1)
        data.setdefault("created_at_utc", utc_now_iso())
        data.setdefault("updated_at_utc", utc_now_iso())
        data.setdefault("models_by_id", {})
        return data

    def save(self) -> None:
        self._data["updated_at_utc"] = utc_now_iso()
        write_json(self.registry_path, self._data, indent=2)

    def register(self, record: CnnArtifactRecord) -> None:
        self._data["models_by_id"][record.model_id] = record.to_dict()
        self.save()

    def get(self, model_id: str) -> Optional[Dict[str, Any]]:
        return self._data["models_by_id"].get(model_id)

    def require(self, model_id: str) -> Dict[str, Any]:
        record = self.get(model_id)
        if record is None:
            raise KeyError(f"No existe modelo CNN con id {model_id!r}")
        return record
