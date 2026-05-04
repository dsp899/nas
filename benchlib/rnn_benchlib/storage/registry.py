from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from rnn_benchlib.storage.state import ensure_parent_dir, stable_hash, utc_now_iso


@dataclass(frozen=True)
class ModelRow:
    model_id: str
    model_key: str
    family: str
    feature_dim: int
    video_steps: int
    spec_json: str
    model_dir: str
    manifest_path: str
    has_float: bool
    has_tflite: bool
    created_at: str


@dataclass(frozen=True)
class LotRow:
    lot_id: str
    kind: str
    config_json: str
    seed: Optional[int]
    requested_count: int
    created_at: str


@dataclass(frozen=True)
class MeasurementRow:
    measurement_id: str
    model_id: str
    runtime: str
    profile_id: str
    config_json: str
    result_path: str
    status: str
    created_at: str


class RnnStateStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        ensure_parent_dir(db_path)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=30000;")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS models (
                    model_id TEXT PRIMARY KEY,
                    model_key TEXT NOT NULL UNIQUE,
                    family TEXT NOT NULL,
                    feature_dim INTEGER NOT NULL,
                    video_steps INTEGER NOT NULL,
                    spec_json TEXT NOT NULL,
                    model_dir TEXT NOT NULL,
                    manifest_path TEXT NOT NULL,
                    has_float INTEGER NOT NULL,
                    has_tflite INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS lots (
                    lot_id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    seed INTEGER,
                    requested_count INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS lot_models (
                    lot_id TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    position INTEGER NOT NULL,
                    source TEXT NOT NULL,
                    PRIMARY KEY (lot_id, model_id),
                    UNIQUE (lot_id, position),
                    FOREIGN KEY (lot_id) REFERENCES lots(lot_id),
                    FOREIGN KEY (model_id) REFERENCES models(model_id)
                );

                CREATE TABLE IF NOT EXISTS measurements (
                    measurement_id TEXT PRIMARY KEY,
                    model_id TEXT NOT NULL,
                    runtime TEXT NOT NULL,
                    profile_id TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    result_path TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(model_id, runtime, profile_id),
                    FOREIGN KEY (model_id) REFERENCES models(model_id)
                );
                """
            )

    def make_stable_lot_id(self, *, kind: str, signature_fields: Dict[str, Any]) -> str:
        return stable_hash({"kind": kind, "signature_fields": signature_fields}, prefix="lot")

    def upsert_lot(self, *, lot_id: str, kind: str, config_json: Dict[str, Any], seed: Optional[int], requested_count: int) -> LotRow:
        created_at = utc_now_iso()
        raw_config = json.dumps(config_json, sort_keys=True, ensure_ascii=False)
        with self._connect() as conn:
            existing = conn.execute("SELECT created_at FROM lots WHERE lot_id = ?", (lot_id,)).fetchone()
            preserved_created_at = existing["created_at"] if existing is not None else created_at
            conn.execute(
                """
                INSERT INTO lots (lot_id, kind, config_json, seed, requested_count, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(lot_id) DO UPDATE SET
                    kind=excluded.kind,
                    config_json=excluded.config_json,
                    seed=excluded.seed,
                    requested_count=excluded.requested_count
                """,
                (lot_id, kind, raw_config, seed, requested_count, preserved_created_at),
            )
        return LotRow(lot_id=lot_id, kind=kind, config_json=raw_config, seed=seed, requested_count=requested_count, created_at=preserved_created_at)

    def get_lot(self, lot_id: str) -> Optional[LotRow]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM lots WHERE lot_id = ?", (lot_id,)).fetchone()
        if row is None:
            return None
        return LotRow(**dict(row))

    def upsert_model(self, *, model_id: str, model_key: str, family: str, feature_dim: int, video_steps: int, spec_json: Dict[str, Any], model_dir: str, manifest_path: str, has_float: bool, has_tflite: bool) -> None:
        created_at = utc_now_iso()
        raw_spec = json.dumps(spec_json, sort_keys=True, ensure_ascii=False)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO models (model_id, model_key, family, feature_dim, video_steps, spec_json, model_dir, manifest_path, has_float, has_tflite, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(model_id) DO UPDATE SET
                    model_key=excluded.model_key,
                    family=excluded.family,
                    feature_dim=excluded.feature_dim,
                    video_steps=excluded.video_steps,
                    spec_json=excluded.spec_json,
                    model_dir=excluded.model_dir,
                    manifest_path=excluded.manifest_path,
                    has_float=excluded.has_float,
                    has_tflite=excluded.has_tflite
                """,
                (model_id, model_key, family, feature_dim, video_steps, raw_spec, model_dir, manifest_path, int(has_float), int(has_tflite), created_at),
            )

    def get_model(self, model_id: str) -> Optional[ModelRow]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM models WHERE model_id = ?", (model_id,)).fetchone()
        if row is None:
            return None
        data = dict(row)
        data["has_float"] = bool(data["has_float"])
        data["has_tflite"] = bool(data["has_tflite"])
        return ModelRow(**data)

    def get_model_by_key(self, model_key: str) -> Optional[ModelRow]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM models WHERE model_key = ?", (model_key,)).fetchone()
        if row is None:
            return None
        data = dict(row)
        data["has_float"] = bool(data["has_float"])
        data["has_tflite"] = bool(data["has_tflite"])
        return ModelRow(**data)

    def list_models(self) -> List[ModelRow]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM models ORDER BY created_at ASC").fetchall()
        result: List[ModelRow] = []
        for row in rows:
            data = dict(row)
            data["has_float"] = bool(data["has_float"])
            data["has_tflite"] = bool(data["has_tflite"])
            result.append(ModelRow(**data))
        return result

    def clear_lot_models(self, lot_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM lot_models WHERE lot_id = ?", (lot_id,))

    def add_model_to_lot(self, *, lot_id: str, model_id: str, position: int, source: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO lot_models (lot_id, model_id, position, source) VALUES (?, ?, ?, ?)",
                (lot_id, model_id, position, source),
            )

    def list_lot_models(self, lot_id: str) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT m.*, lm.position, lm.source
                FROM lot_models lm
                JOIN models m ON lm.model_id = m.model_id
                WHERE lm.lot_id = ?
                ORDER BY lm.position ASC
                """,
                (lot_id,),
            ).fetchall()
        result: List[Dict[str, Any]] = []
        for row in rows:
            data = dict(row)
            data["has_float"] = bool(data["has_float"])
            data["has_tflite"] = bool(data["has_tflite"])
            result.append(data)
        return result

    def make_profile_id(self, *, runtime: str, config_json: Dict[str, Any]) -> str:
        return stable_hash({"runtime": runtime, "config": config_json}, prefix="profile")

    def get_measurement(self, *, model_id: str, runtime: str, profile_id: str) -> Optional[MeasurementRow]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM measurements WHERE model_id = ? AND runtime = ? AND profile_id = ?",
                (model_id, runtime, profile_id),
            ).fetchone()
        if row is None:
            return None
        return MeasurementRow(**dict(row))

    def upsert_measurement(self, *, measurement_id: str, model_id: str, runtime: str, profile_id: str, config_json: Dict[str, Any], result_path: str, status: str) -> None:
        raw_config = json.dumps(config_json, sort_keys=True, ensure_ascii=False)
        created_at = utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO measurements (measurement_id, model_id, runtime, profile_id, config_json, result_path, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(measurement_id) DO UPDATE SET
                    model_id=excluded.model_id,
                    runtime=excluded.runtime,
                    profile_id=excluded.profile_id,
                    config_json=excluded.config_json,
                    result_path=excluded.result_path,
                    status=excluded.status
                """,
                (measurement_id, model_id, runtime, profile_id, raw_config, result_path, status, created_at),
            )
