from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict
from benchlib_common.io.jsonl import read_json, write_json
from hybrid_benchlib.config.schemas import HybridBundleRecord


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class HybridRegistry:
    def __init__(self, registry_path: str):
        self.registry_path = registry_path
        self._data = self._load()

    def _load(self) -> Dict[str, Any]:
        data = read_json(self.registry_path, default=None)
        if data is None:
            return {"version": 1, "created_at_utc": utc_now_iso(), "updated_at_utc": utc_now_iso(), "hybrids_by_id": {}}
        data.setdefault("version", 1)
        data.setdefault("created_at_utc", utc_now_iso())
        data.setdefault("updated_at_utc", utc_now_iso())
        data.setdefault("hybrids_by_id", {})
        return data

    def save(self) -> None:
        self._data["updated_at_utc"] = utc_now_iso()
        write_json(self.registry_path, self._data, indent=2)

    def register(self, record: HybridBundleRecord) -> None:
        self._data["hybrids_by_id"][record.hybrid_model_id] = record.to_dict()
        self.save()
