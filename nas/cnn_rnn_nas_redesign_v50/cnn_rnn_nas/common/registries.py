
import csv
import hashlib
import json
import sqlite3
import shutil
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path

from ..config.cnn_config import CnnExperimentConfig
from ..config.rnn_config import RnnDataConfig, RnnExperimentConfig
from typing import Any, Dict, List, Optional, Union

RNN_SIGNATURE_VERSION = "v12"
CNN_SIGNATURE_VERSION = "v15"
NAS_SIGNATURE_VERSION = "v6"


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"No serializable: {type(value)!r}")


def canonical_json(data: Dict[str, Any]) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), default=_json_default)


def hash_payload(payload: Dict[str, Any], *, prefix: str, signature_version: str) -> str:
    material = {"signature_version": signature_version, "prefix": prefix, "payload": payload}
    return hashlib.sha256(canonical_json(material).encode("utf-8")).hexdigest()[:16]


def _ensure_sqlite_column(conn: sqlite3.Connection, table: str, column: str, declaration: str) -> None:
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")



def nas_search_tag(config: RnnExperimentConfig) -> str:
    search_space = config.search_space
    cnn_scope = config.data.cnn
    if search_space is not None and len(getattr(search_space, "cnn", (config.data.cnn,))) > 1:
        cnn_scope = "multi_cnn"
    reward_source = "test" if config.data.partition_mode == "train_test" else "val"
    dims_count = len(search_space.variable_dimensions) if search_space is not None else 0
    return f"{cnn_scope}_{config.data.feature_spec_tag}_dims{dims_count}_reward{reward_source}"

# ---------------------------------------------------------------------------
# NAS search signatures and registry
# ---------------------------------------------------------------------------

def nas_base_data_signature(config: RnnExperimentConfig) -> str:
    return hash_payload(asdict(config.data), prefix="nas-base-data", signature_version=NAS_SIGNATURE_VERSION)


def nas_search_space_signature(config: RnnExperimentConfig) -> str:
    search_space = config.search_space.to_dict() if config.search_space is not None else {}
    return hash_payload(search_space, prefix="nas-search-space", signature_version=NAS_SIGNATURE_VERSION)


def nas_controller_signature(config: RnnExperimentConfig) -> str:
    controller_payload = asdict(config.nas) if config.nas is not None else {}
    return hash_payload(controller_payload, prefix="nas-controller", signature_version=NAS_SIGNATURE_VERSION)


def nas_candidate_runtime_signature(config: RnnExperimentConfig) -> str:
    payload = {
        "epochs": config.runtime.epochs,
        "batch_size": config.runtime.batch_size,
        "learning_rate": config.runtime.learning_rate,
        "test_strategy": config.runtime.test_strategy,
        "mixed_precision": config.runtime.mixed_precision,
        "random_seed": config.runtime.random_seed,
        "allow_epoch_extension_resume": config.runtime.allow_epoch_extension_resume,
        "reduce_lr_on_plateau": config.runtime.reduce_lr_on_plateau,
        "reduce_lr_factor": config.runtime.reduce_lr_factor,
        "reduce_lr_patience": config.runtime.reduce_lr_patience,
        "min_learning_rate": config.runtime.min_learning_rate,
    }
    return hash_payload(payload, prefix="nas-candidate-runtime", signature_version=NAS_SIGNATURE_VERSION)


def nas_candidate_optimizer_signature(config: RnnExperimentConfig) -> str:
    return hash_payload(asdict(config.optimizer), prefix="nas-candidate-optimizer", signature_version=NAS_SIGNATURE_VERSION)


def nas_search_signature(config: RnnExperimentConfig) -> str:
    payload = {
        "nas_base_data_signature": nas_base_data_signature(config),
        "nas_search_space_signature": nas_search_space_signature(config),
        "nas_controller_signature": nas_controller_signature(config),
        "nas_candidate_runtime_signature": nas_candidate_runtime_signature(config),
        "nas_candidate_optimizer_signature": nas_candidate_optimizer_signature(config),
    }
    return hash_payload(payload, prefix="nas-search-experiment", signature_version=NAS_SIGNATURE_VERSION)


class NasSearchRegistry:
    def __init__(self, db_path: Union[str, Path]) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS nas_search_runs (
                    nas_search_signature TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    nas_base_data_signature TEXT,
                    nas_search_space_signature TEXT,
                    nas_controller_signature TEXT,
                    nas_candidate_runtime_signature TEXT,
                    nas_candidate_optimizer_signature TEXT,
                    search_run_dir TEXT,
                    search_log_path TEXT,
                    architectures_csv_path TEXT,
                    controller_history_csv_path TEXT,
                    summary_path TEXT,
                    last_error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    signature_version TEXT
                )
                """
            )
            conn.commit()

    def get(self, nas_search_signature_value: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT nas_search_signature, status, config_json, nas_base_data_signature, nas_search_space_signature,
                       nas_controller_signature, nas_candidate_runtime_signature, nas_candidate_optimizer_signature,
                       search_run_dir, search_log_path, architectures_csv_path, controller_history_csv_path, summary_path,
                       last_error, created_at, updated_at, signature_version
                FROM nas_search_runs
                WHERE nas_search_signature = ?
                """,
                (nas_search_signature_value,),
            ).fetchone()
        if not row:
            return None
        return {
            "nas_search_signature": row[0],
            "status": row[1],
            "config": json.loads(row[2]),
            "nas_base_data_signature": row[3],
            "nas_search_space_signature": row[4],
            "nas_controller_signature": row[5],
            "nas_candidate_runtime_signature": row[6],
            "nas_candidate_optimizer_signature": row[7],
            "search_run_dir": row[8],
            "search_log_path": row[9],
            "architectures_csv_path": row[10],
            "controller_history_csv_path": row[11],
            "summary_path": row[12],
            "last_error": row[13],
            "created_at": row[14],
            "updated_at": row[15],
            "signature_version": row[16],
        }

    def reserve(self, config: RnnExperimentConfig, *, search_run_dir: Union[str, Path], search_log_path: Union[str, Path], architectures_csv_path: Union[str, Path], controller_history_csv_path: Union[str, Path], summary_path: Union[str, Path]) -> str:
        signature = nas_search_signature(config)
        now = datetime.utcnow().isoformat()
        payload = canonical_json(config.to_dict())
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO nas_search_runs(
                    nas_search_signature, status, config_json, nas_base_data_signature, nas_search_space_signature,
                    nas_controller_signature, nas_candidate_runtime_signature, nas_candidate_optimizer_signature,
                    search_run_dir, search_log_path, architectures_csv_path, controller_history_csv_path, summary_path,
                    last_error, created_at, updated_at, signature_version
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?)
                ON CONFLICT(nas_search_signature) DO UPDATE SET
                    status=excluded.status,
                    config_json=excluded.config_json,
                    nas_base_data_signature=excluded.nas_base_data_signature,
                    nas_search_space_signature=excluded.nas_search_space_signature,
                    nas_controller_signature=excluded.nas_controller_signature,
                    nas_candidate_runtime_signature=excluded.nas_candidate_runtime_signature,
                    nas_candidate_optimizer_signature=excluded.nas_candidate_optimizer_signature,
                    search_run_dir=excluded.search_run_dir,
                    search_log_path=excluded.search_log_path,
                    architectures_csv_path=excluded.architectures_csv_path,
                    controller_history_csv_path=excluded.controller_history_csv_path,
                    summary_path=excluded.summary_path,
                    updated_at=excluded.updated_at,
                    signature_version=excluded.signature_version,
                    last_error=NULL
                """,
                (
                    signature, "running", payload, nas_base_data_signature(config), nas_search_space_signature(config),
                    nas_controller_signature(config), nas_candidate_runtime_signature(config), nas_candidate_optimizer_signature(config),
                    str(search_run_dir), str(search_log_path), str(architectures_csv_path), str(controller_history_csv_path), str(summary_path),
                    now, now, NAS_SIGNATURE_VERSION,
                ),
            )
            conn.commit()
        return signature

    def complete(self, nas_search_signature_value: str, *, search_run_dir: Union[str, Path], search_log_path: Union[str, Path], architectures_csv_path: Union[str, Path], controller_history_csv_path: Union[str, Path], summary_path: Union[str, Path]) -> None:
        now = datetime.utcnow().isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE nas_search_runs
                SET status=?, search_run_dir=?, search_log_path=?, architectures_csv_path=?, controller_history_csv_path=?, summary_path=?, updated_at=?, last_error=NULL
                WHERE nas_search_signature=?
                """,
                ("completed", str(search_run_dir), str(search_log_path), str(architectures_csv_path), str(controller_history_csv_path), str(summary_path), now, nas_search_signature_value),
            )
            conn.commit()

    def fail(self, nas_search_signature_value: str, error: str) -> None:
        now = datetime.utcnow().isoformat()
        with self._connect() as conn:
            conn.execute(
                "UPDATE nas_search_runs SET status=?, last_error=?, updated_at=? WHERE nas_search_signature=?",
                ("failed", error, now, nas_search_signature_value),
            )
            conn.commit()


# ---------------------------------------------------------------------------
# RNN / NAS signatures and registry
# ---------------------------------------------------------------------------

def rnn_data_signature(config: RnnExperimentConfig) -> str:
    return hash_payload(asdict(config.data), prefix="data", signature_version=RNN_SIGNATURE_VERSION)


def rnn_architecture_signature(config: RnnExperimentConfig) -> str:
    return hash_payload(asdict(config.architecture), prefix="architecture", signature_version=RNN_SIGNATURE_VERSION)


def rnn_runtime_signature(config: RnnExperimentConfig) -> str:
    payload = {
        "epochs": config.runtime.epochs,
        "batch_size": config.runtime.batch_size,
        "learning_rate": config.runtime.learning_rate,
        "test_strategy": config.runtime.test_strategy,
        "mixed_precision": config.runtime.mixed_precision,
        "random_seed": config.runtime.random_seed,
        "reduce_lr_on_plateau": config.runtime.reduce_lr_on_plateau,
        "reduce_lr_factor": config.runtime.reduce_lr_factor,
        "reduce_lr_patience": config.runtime.reduce_lr_patience,
        "min_learning_rate": config.runtime.min_learning_rate,
        "optimizer": asdict(config.optimizer),
    }
    return hash_payload(payload, prefix="runtime", signature_version=RNN_SIGNATURE_VERSION)


def rnn_resume_runtime_signature(config: RnnExperimentConfig) -> str:
    payload = {
        "batch_size": config.runtime.batch_size,
        "learning_rate": config.runtime.learning_rate,
        "test_strategy": config.runtime.test_strategy,
        "mixed_precision": config.runtime.mixed_precision,
        "random_seed": config.runtime.random_seed,
        "reduce_lr_on_plateau": config.runtime.reduce_lr_on_plateau,
        "reduce_lr_factor": config.runtime.reduce_lr_factor,
        "reduce_lr_patience": config.runtime.reduce_lr_patience,
        "min_learning_rate": config.runtime.min_learning_rate,
        "optimizer": asdict(config.optimizer),
    }
    return hash_payload(payload, prefix="resume-runtime", signature_version=RNN_SIGNATURE_VERSION)


def rnn_resume_signature(config: RnnExperimentConfig) -> str:
    payload = {
        "rnn_data_signature": rnn_data_signature(config),
        "rnn_architecture_signature": rnn_architecture_signature(config),
        "rnn_resume_runtime_signature": rnn_resume_runtime_signature(config),
    }
    return hash_payload(payload, prefix="resume-experiment", signature_version=RNN_SIGNATURE_VERSION)


def rnn_experiment_signature(config: RnnExperimentConfig) -> str:
    payload = {
        "rnn_data_signature": rnn_data_signature(config),
        "rnn_architecture_signature": rnn_architecture_signature(config),
        "rnn_runtime_signature": rnn_runtime_signature(config),
    }
    return hash_payload(payload, prefix="experiment", signature_version=RNN_SIGNATURE_VERSION)


# ---------------------------------------------------------------------------
# CNN signatures and registry
# ---------------------------------------------------------------------------

def cnn_training_data_signature(config: CnnExperimentConfig) -> str:
    payload = {
        "dataset": asdict(config.dataset),
        "preprocess": asdict(config.preprocess),
    }
    return hash_payload(payload, prefix="cnn-train-data", signature_version=CNN_SIGNATURE_VERSION)


def cnn_model_signature(config: CnnExperimentConfig) -> str:
    payload = {
        "extractor": asdict(config.extractor),
        "head": asdict(config.head),
    }
    return hash_payload(payload, prefix="cnn-model", signature_version=CNN_SIGNATURE_VERSION)


def cnn_training_runtime_signature(config: CnnExperimentConfig) -> str:
    payload = {
        "training": asdict(config.training),
        "runtime": asdict(config.runtime),
    }
    return hash_payload(payload, prefix="cnn-train-runtime", signature_version=CNN_SIGNATURE_VERSION)


def cnn_training_signature(config: CnnExperimentConfig) -> str:
    payload = {
        "training_data_signature": cnn_training_data_signature(config),
        "model_signature": cnn_model_signature(config),
        "training_runtime_signature": cnn_training_runtime_signature(config),
    }
    return hash_payload(payload, prefix="cnn-training-experiment", signature_version=CNN_SIGNATURE_VERSION)


def cnn_feature_export_signature(config: CnnExperimentConfig, training_signature: str) -> str:
    payload = {
        "training_signature": training_signature,
        "dataset": asdict(config.dataset),
        "predict_preprocess": {
            "image_size": config.preprocess.image_size,
            "predict_frames": config.preprocess.predict_frames,
            "resize_mode": config.preprocess.resize_mode,
            "predict_sampling": config.preprocess.predict_sampling,
        },
        "extractor": {
            "backbone": config.extractor.backbone,
            "feature_dim": config.extractor.feature_dim,
        },
    }
    return hash_payload(payload, prefix="cnn-feature-export", signature_version=CNN_SIGNATURE_VERSION)


class RunArtifacts:
    def __init__(self, run_dir: Union[str, Path]) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.history_csv = self.run_dir / "history.csv"
        self.summary_json = self.run_dir / "summary.json"
        self.manifest_json = self.run_dir / "run_manifest.json"

    def append_epoch(self, row: Dict[str, Any]) -> None:
        file_exists = self.history_csv.exists()
        with self.history_csv.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def seed_history_from(self, source_history_csv: Union[str, Path, None]) -> None:
        if not source_history_csv:
            return
        source = Path(source_history_csv)
        if not source.exists() or self.history_csv.exists():
            return
        self.history_csv.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, self.history_csv)

    def write_summary(self, payload: Dict[str, Any]) -> None:
        self.summary_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def write_manifest(self, payload: Dict[str, Any]) -> None:
        self.manifest_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


class RnnExperimentRegistry:
    def __init__(self, db_path: Union[str, Path]) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS rnn_experiment_runs (
                    rnn_experiment_signature TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    metrics_json TEXT,
                    best_model_path TEXT,
                    last_model_path TEXT,
                    training_state_path TEXT,
                    optimizer_state_path TEXT,
                    model_manifest_path TEXT,
                    run_dir TEXT,
                    last_error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    rnn_data_signature TEXT NOT NULL,
                    rnn_architecture_signature TEXT NOT NULL,
                    rnn_runtime_signature TEXT NOT NULL,
                    rnn_resume_runtime_signature TEXT NOT NULL,
                    rnn_resume_signature TEXT NOT NULL,
                    resumed_from_signature TEXT,
                    initial_epoch INTEGER NOT NULL,
                    final_epoch INTEGER NOT NULL,
                    signature_version TEXT NOT NULL
                )
                """
            )
            _ensure_sqlite_column(conn, "rnn_experiment_runs", "optimizer_state_path", "TEXT")
            conn.commit()

    def get(self, rnn_experiment_signature: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT rnn_experiment_signature, status, operation, config_json, metrics_json,
                       best_model_path, last_model_path, training_state_path, optimizer_state_path, model_manifest_path,
                       run_dir, last_error, created_at, updated_at, rnn_data_signature, rnn_architecture_signature,
                       rnn_runtime_signature, rnn_resume_runtime_signature, rnn_resume_signature,
                       resumed_from_signature, initial_epoch, final_epoch, signature_version
                FROM rnn_experiment_runs
                WHERE rnn_experiment_signature = ?
                """,
                (rnn_experiment_signature,),
            ).fetchone()
        if not row:
            return None
        return {
            "rnn_experiment_signature": row[0],
            "status": row[1],
            "operation": row[2],
            "config": json.loads(row[3]),
            "metrics": json.loads(row[4]) if row[4] else None,
            "best_model_path": row[5],
            "last_model_path": row[6],
            "training_state_path": row[7],
            "optimizer_state_path": row[8],
            "model_manifest_path": row[9],
            "run_dir": row[10],
            "last_error": row[11],
            "created_at": row[12],
            "updated_at": row[13],
            "rnn_data_signature": row[14],
            "rnn_architecture_signature": row[15],
            "rnn_runtime_signature": row[16],
            "rnn_resume_runtime_signature": row[17],
            "rnn_resume_signature": row[18],
            "resumed_from_signature": row[19],
            "initial_epoch": int(row[20] or 0),
            "final_epoch": int(row[21] or 0),
            "signature_version": row[22],
        }

    def reserve(self, rnn_experiment_signature: str, config: RnnExperimentConfig, *, best_model_path: Union[str, Path], last_model_path: Union[str, Path], training_state_path: Union[str, Path], optimizer_state_path: Union[str, Path], model_manifest_path: Union[str, Path], run_dir: Union[str, Path], resumed_from_signature: Optional[str] = None, initial_epoch: int = 0) -> None:
        now = datetime.utcnow().isoformat()
        payload = canonical_json(config.to_dict())
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO rnn_experiment_runs(
                    rnn_experiment_signature, status, operation, config_json, metrics_json,
                    best_model_path, last_model_path, training_state_path, optimizer_state_path, model_manifest_path, run_dir,
                    last_error, created_at, updated_at, rnn_data_signature, rnn_architecture_signature,
                    rnn_runtime_signature, rnn_resume_runtime_signature, rnn_resume_signature,
                    resumed_from_signature, initial_epoch, final_epoch, signature_version
                )
                VALUES(?, ?, ?, ?, NULL, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(rnn_experiment_signature) DO UPDATE SET
                    status=excluded.status,
                    operation=excluded.operation,
                    config_json=excluded.config_json,
                    best_model_path=excluded.best_model_path,
                    last_model_path=excluded.last_model_path,
                    training_state_path=excluded.training_state_path,
                    optimizer_state_path=excluded.optimizer_state_path,
                    model_manifest_path=excluded.model_manifest_path,
                    run_dir=excluded.run_dir,
                    updated_at=excluded.updated_at,
                    rnn_data_signature=excluded.rnn_data_signature,
                    rnn_architecture_signature=excluded.rnn_architecture_signature,
                    rnn_runtime_signature=excluded.rnn_runtime_signature,
                    rnn_resume_runtime_signature=excluded.rnn_resume_runtime_signature,
                    rnn_resume_signature=excluded.rnn_resume_signature,
                    resumed_from_signature=excluded.resumed_from_signature,
                    initial_epoch=excluded.initial_epoch,
                    final_epoch=excluded.final_epoch,
                    signature_version=excluded.signature_version,
                    last_error=NULL
                """,
                (
                    rnn_experiment_signature, "running", config.operation, payload,
                    str(best_model_path), str(last_model_path), str(training_state_path), str(optimizer_state_path),
                    str(model_manifest_path), str(run_dir), now, now,
                    rnn_data_signature(config), rnn_architecture_signature(config), rnn_runtime_signature(config),
                    rnn_resume_runtime_signature(config), rnn_resume_signature(config), resumed_from_signature,
                    int(initial_epoch), int(initial_epoch), RNN_SIGNATURE_VERSION,
                ),
            )
            conn.commit()

    def complete(self, rnn_experiment_signature: str, metrics: Dict[str, Any], *, best_model_path: Union[str, Path], last_model_path: Union[str, Path], training_state_path: Union[str, Path], optimizer_state_path: Union[str, Path], model_manifest_path: Union[str, Path], run_dir: Union[str, Path], resumed_from_signature: Optional[str] = None, initial_epoch: int = 0, final_epoch: int = 0) -> None:
        now = datetime.utcnow().isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE rnn_experiment_runs
                SET status=?, metrics_json=?, best_model_path=?, last_model_path=?,
                    training_state_path=?, optimizer_state_path=?, model_manifest_path=?, run_dir=?, resumed_from_signature=?,
                    initial_epoch=?, final_epoch=?, updated_at=?, last_error=NULL
                WHERE rnn_experiment_signature=?
                """,
                (
                    "completed", json.dumps(metrics), str(best_model_path), str(last_model_path),
                    str(training_state_path), str(optimizer_state_path), str(model_manifest_path), str(run_dir),
                    resumed_from_signature, int(initial_epoch), int(final_epoch), now, rnn_experiment_signature,
                ),
            )
            conn.commit()

    def fail(self, rnn_experiment_signature: str, error: str) -> None:
        now = datetime.utcnow().isoformat()
        with self._connect() as conn:
            conn.execute(
                "UPDATE rnn_experiment_runs SET status=?, last_error=?, updated_at=? WHERE rnn_experiment_signature=?",
                ("failed", error, now, rnn_experiment_signature),
            )
            conn.commit()

    def find_best_resume_candidate(self, config: RnnExperimentConfig) -> Optional[Dict[str, Any]]:
        target_epochs = int(config.runtime.epochs)
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT rnn_experiment_signature
                FROM rnn_experiment_runs
                WHERE status='completed'
                  AND rnn_resume_signature=?
                  AND final_epoch < ?
                ORDER BY final_epoch DESC, updated_at DESC
                LIMIT 1
                """,
                (rnn_resume_signature(config), target_epochs),
            ).fetchone()
        return self.get(row[0]) if row else None

    def top_completed(self, limit: int = 10) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT rnn_experiment_signature, config_json, metrics_json, best_model_path,
                       rnn_data_signature, rnn_architecture_signature, rnn_runtime_signature, signature_version
                FROM rnn_experiment_runs
                WHERE status='completed'
                """
            ).fetchall()
        parsed = []
        for signature, config_json, metrics_json, best_model_path, data_sig, arch_sig, runtime_sig, sig_ver in rows:
            metrics = json.loads(metrics_json or "{}")
            parsed.append({
                "rnn_experiment_signature": signature,
                "config": json.loads(config_json),
                "metrics": metrics,
                "best_model_path": best_model_path,
                "rnn_data_signature": data_sig,
                "rnn_architecture_signature": arch_sig,
                "rnn_runtime_signature": runtime_sig,
                "signature_version": sig_ver,
            })
        parsed.sort(key=lambda item: item["metrics"].get("best_test_acc", 0.0), reverse=True)
        return parsed[:limit]


class CnnExperimentRegistry:
    def __init__(self, db_path: Union[str, Path]) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cnn_training_runs (
                    training_signature TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    metrics_json TEXT,
                    best_model_path TEXT,
                    last_model_path TEXT,
                    training_state_path TEXT,
                    optimizer_state_path TEXT,
                    model_manifest_path TEXT,
                    run_dir TEXT,
                    last_error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    training_data_signature TEXT,
                    model_signature TEXT,
                    training_runtime_signature TEXT,
                    signature_version TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cnn_feature_exports (
                    feature_signature TEXT PRIMARY KEY,
                    training_signature TEXT NOT NULL,
                    status TEXT NOT NULL,
                    export_config_json TEXT NOT NULL,
                    feature_dir TEXT,
                    feature_manifest_path TEXT,
                    export_summary_json TEXT,
                    last_error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    signature_version TEXT,
                    FOREIGN KEY(training_signature) REFERENCES cnn_training_runs(training_signature)
                )
                """
            )
            _ensure_sqlite_column(conn, "cnn_training_runs", "optimizer_state_path", "TEXT")
            conn.commit()

    def get_training(self, training_signature: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT training_signature, status, config_json, metrics_json, best_model_path, last_model_path,
                       training_state_path, optimizer_state_path, model_manifest_path, run_dir, last_error, created_at, updated_at,
                       training_data_signature, model_signature, training_runtime_signature, signature_version
                FROM cnn_training_runs
                WHERE training_signature = ?
                """,
                (training_signature,),
            ).fetchone()
        if not row:
            return None
        return {
            "training_signature": row[0],
            "status": row[1],
            "config": json.loads(row[2]),
            "metrics": json.loads(row[3]) if row[3] else None,
            "best_model_path": row[4],
            "last_model_path": row[5],
            "training_state_path": row[6],
            "optimizer_state_path": row[7],
            "model_manifest_path": row[8],
            "run_dir": row[9],
            "last_error": row[10],
            "created_at": row[11],
            "updated_at": row[12],
            "training_data_signature": row[13],
            "model_signature": row[14],
            "training_runtime_signature": row[15],
            "signature_version": row[16],
        }

    def reserve_training(self, training_signature: str, config: CnnExperimentConfig, *, best_model_path: Union[str, Path], last_model_path: Union[str, Path], training_state_path: Union[str, Path], optimizer_state_path: Union[str, Path], model_manifest_path: Union[str, Path], run_dir: Union[str, Path]) -> None:
        now = datetime.utcnow().isoformat()
        payload = canonical_json(config.to_dict())
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO cnn_training_runs(
                    training_signature, status, config_json, metrics_json, best_model_path, last_model_path,
                    training_state_path, optimizer_state_path, model_manifest_path, run_dir, last_error, created_at, updated_at,
                    training_data_signature, model_signature, training_runtime_signature, signature_version
                )
                VALUES(?, ?, ?, NULL, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(training_signature) DO UPDATE SET
                    status=excluded.status,
                    config_json=excluded.config_json,
                    best_model_path=excluded.best_model_path,
                    last_model_path=excluded.last_model_path,
                    training_state_path=excluded.training_state_path,
                    optimizer_state_path=excluded.optimizer_state_path,
                    model_manifest_path=excluded.model_manifest_path,
                    run_dir=excluded.run_dir,
                    updated_at=excluded.updated_at,
                    training_data_signature=excluded.training_data_signature,
                    model_signature=excluded.model_signature,
                    training_runtime_signature=excluded.training_runtime_signature,
                    signature_version=excluded.signature_version,
                    last_error=NULL
                """,
                (
                    training_signature, "running", payload,
                    str(best_model_path), str(last_model_path), str(training_state_path), str(optimizer_state_path),
                    str(model_manifest_path), str(run_dir), now, now,
                    cnn_training_data_signature(config), cnn_model_signature(config), cnn_training_runtime_signature(config),
                    CNN_SIGNATURE_VERSION,
                ),
            )
            conn.commit()

    def complete_training(self, training_signature: str, metrics: Dict[str, Any], *, best_model_path: Union[str, Path], last_model_path: Union[str, Path], training_state_path: Union[str, Path], optimizer_state_path: Union[str, Path], model_manifest_path: Union[str, Path], run_dir: Union[str, Path]) -> None:
        now = datetime.utcnow().isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE cnn_training_runs
                SET status=?, metrics_json=?, best_model_path=?, last_model_path=?,
                    training_state_path=?, optimizer_state_path=?, model_manifest_path=?, run_dir=?, updated_at=?, last_error=NULL
                WHERE training_signature=?
                """,
                (
                    "completed", json.dumps(metrics), str(best_model_path), str(last_model_path),
                    str(training_state_path), str(optimizer_state_path), str(model_manifest_path), str(run_dir), now,
                    training_signature,
                ),
            )
            conn.commit()

    def fail_training(self, training_signature: str, error: str) -> None:
        now = datetime.utcnow().isoformat()
        with self._connect() as conn:
            conn.execute(
                "UPDATE cnn_training_runs SET status=?, last_error=?, updated_at=? WHERE training_signature=?",
                ("failed", error, now, training_signature),
            )
            conn.commit()

    def find_latest_completed_training(self, config: CnnExperimentConfig) -> Optional[Dict[str, Any]]:
        train_data_sig = cnn_training_data_signature(config)
        model_sig = cnn_model_signature(config)
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT training_signature
                FROM cnn_training_runs
                WHERE status='completed' AND training_data_signature=? AND model_signature=?
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (train_data_sig, model_sig),
            ).fetchone()
        return self.get_training(row[0]) if row else None

    def get_feature_export(self, feature_signature: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT feature_signature, training_signature, status, export_config_json, feature_dir,
                       feature_manifest_path, export_summary_json, last_error, created_at, updated_at, signature_version
                FROM cnn_feature_exports
                WHERE feature_signature = ?
                """,
                (feature_signature,),
            ).fetchone()
        if not row:
            return None
        return {
            "feature_signature": row[0],
            "training_signature": row[1],
            "status": row[2],
            "config": json.loads(row[3]),
            "feature_dir": row[4],
            "feature_manifest_path": row[5],
            "summary": json.loads(row[6]) if row[6] else None,
            "last_error": row[7],
            "created_at": row[8],
            "updated_at": row[9],
            "signature_version": row[10],
        }

    def find_latest_completed_feature_export_for_rnn(
        self,
        data: RnnDataConfig,
        *,
        training_signature: Optional[str] = None,
        feature_signature: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT feature_signature, training_signature, export_config_json
                FROM cnn_feature_exports
                WHERE status='completed'
                ORDER BY updated_at DESC
                """
            ).fetchall()
        for row_feature_signature, row_training_signature, export_config_json in rows:
            if training_signature and str(row_training_signature) != str(training_signature):
                continue
            if feature_signature and str(row_feature_signature) != str(feature_signature):
                continue
            try:
                payload = json.loads(export_config_json)
            except json.JSONDecodeError:
                continue
            dataset_cfg = payload.get("dataset", {})
            predict_preprocess_cfg = payload.get("predict_preprocess") or payload.get("preprocess", {})
            extractor_cfg = payload.get("extractor") or payload.get("model", {}).get("extractor", {})
            predict_frames = predict_preprocess_cfg.get("predict_frames", predict_preprocess_cfg.get("frames", -1))
            sampling = predict_preprocess_cfg.get("sampling", predict_preprocess_cfg.get("predict_sampling"))
            if (
                extractor_cfg.get("backbone") == data.cnn
                and dataset_cfg.get("name") == data.name
                and dataset_cfg.get("split") == data.split
                and dataset_cfg.get("partition_mode") == data.partition_mode
                and float(dataset_cfg.get("val_fraction", 0.0)) == float(data.val_fraction)
                and int(predict_frames) == int(data.frames)
                and int(predict_preprocess_cfg.get("image_size", -1)) == int(data.image_size)
                and predict_preprocess_cfg.get("resize_mode") == data.resize_mode
                and sampling == data.sampling
            ):
                return self.get_feature_export(row_feature_signature)
        return None

    def reserve_feature_export(self, feature_signature: str, training_signature: str, config: CnnExperimentConfig, *, feature_dir: Union[str, Path], feature_manifest_path: Union[str, Path]) -> None:
        now = datetime.utcnow().isoformat()
        payload = canonical_json(config.to_dict())
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO cnn_feature_exports(
                    feature_signature, training_signature, status, export_config_json, feature_dir,
                    feature_manifest_path, export_summary_json, last_error, created_at, updated_at, signature_version
                )
                VALUES(?, ?, ?, ?, ?, ?, NULL, NULL, ?, ?, ?)
                ON CONFLICT(feature_signature) DO UPDATE SET
                    training_signature=excluded.training_signature,
                    status=excluded.status,
                    export_config_json=excluded.export_config_json,
                    feature_dir=excluded.feature_dir,
                    feature_manifest_path=excluded.feature_manifest_path,
                    updated_at=excluded.updated_at,
                    signature_version=excluded.signature_version,
                    last_error=NULL
                """,
                (
                    feature_signature,
                    training_signature,
                    "running",
                    payload,
                    str(feature_dir),
                    str(feature_manifest_path),
                    now,
                    now,
                    CNN_SIGNATURE_VERSION,
                ),
            )
            conn.commit()

    def complete_feature_export(self, feature_signature: str, summary: Dict[str, Any], *, feature_dir: Union[str, Path], feature_manifest_path: Union[str, Path]) -> None:
        now = datetime.utcnow().isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE cnn_feature_exports
                SET status=?, export_summary_json=?, feature_dir=?, feature_manifest_path=?, updated_at=?, last_error=NULL
                WHERE feature_signature=?
                """,
                ("completed", json.dumps(summary), str(feature_dir), str(feature_manifest_path), now, feature_signature),
            )
            conn.commit()

    def fail_feature_export(self, feature_signature: str, error: str) -> None:
        now = datetime.utcnow().isoformat()
        with self._connect() as conn:
            conn.execute(
                "UPDATE cnn_feature_exports SET status=?, last_error=?, updated_at=? WHERE feature_signature=?",
                ("failed", error, now, feature_signature),
            )
            conn.commit()

# ---------------------------------------------------------------------------
# Deploy signatures and registries
# ---------------------------------------------------------------------------

import json
import sqlite3
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

from ..config.cnn_config import CnnExperimentConfig
from ..config.deploy_config import (
    CnnExtractorEvalConfig,
    CnnQuantEvalConfig,
    CnnVitisAiDeployConfig,
    RnnTfliteEvalConfig,
    RnnTfliteExportConfig,
    RnnDeployEvalConfig,
)
from ..config.rnn_config import RnnExperimentConfig

CNN_DEPLOY_SIGNATURE_VERSION = "v3"
RNN_EXPORT_SIGNATURE_VERSION = "v3"
CNN_QUANT_EVAL_SIGNATURE_VERSION = "v2"
CNN_EXTRACTOR_EVAL_SIGNATURE_VERSION = "v1"
CNN_DEPLOY_EVAL_SIGNATURE_VERSION = "v1"
RNN_TFLITE_EVAL_SIGNATURE_VERSION = "v2"
RNN_DEPLOY_EVAL_SIGNATURE_VERSION = "v2"


def cnn_deploy_signature(config: CnnExperimentConfig, training_signature: str, deploy_config: CnnVitisAiDeployConfig) -> str:
    payload = {
        "training_signature": training_signature,
        "partition_tag": config.partition_tag,
        "extractor": config.extractor_tag,
        "train_preprocess_tag": config.train_preprocess_tag,
        "head_tag": config.head_tag,
        "predict_preprocess_tag": config.predict_preprocess_tag,
        "deploy_config": deploy_config.to_dict(),
    }
    return hash_payload(payload, prefix="cnn-vitis-ai-deploy", signature_version=CNN_DEPLOY_SIGNATURE_VERSION)


def cnn_quant_eval_signature(config: CnnExperimentConfig, training_signature: str, deploy_signature: str, eval_config: CnnQuantEvalConfig) -> str:
    payload = {
        "training_signature": training_signature,
        "cnn_deploy_signature": deploy_signature,
        "partition_tag": config.partition_tag,
        "extractor": config.extractor_tag,
        "train_preprocess_tag": config.train_preprocess_tag,
        "head_tag": config.head_tag,
        "predict_preprocess_tag": config.predict_preprocess_tag,
        "eval_config": eval_config.to_dict(),
    }
    return hash_payload(payload, prefix="cnn-quant-eval", signature_version=CNN_QUANT_EVAL_SIGNATURE_VERSION)


def cnn_extractor_eval_signature(config: CnnExperimentConfig, training_signature: str, deploy_signature: str, eval_config: CnnExtractorEvalConfig) -> str:
    payload = {
        "training_signature": training_signature,
        "cnn_deploy_signature": deploy_signature,
        "partition_tag": config.partition_tag,
        "extractor": config.extractor_tag,
        "train_preprocess_tag": config.train_preprocess_tag,
        "head_tag": config.head_tag,
        "predict_preprocess_tag": config.predict_preprocess_tag,
        "eval_config": eval_config.to_dict(),
    }
    return hash_payload(payload, prefix="cnn-extractor-eval", signature_version=CNN_EXTRACTOR_EVAL_SIGNATURE_VERSION)


def cnn_deploy_eval_signature(config: CnnExperimentConfig, training_signature: str, deploy_signature: str, eval_config: CnnExtractorEvalConfig) -> str:
    payload = {
        "training_signature": training_signature,
        "cnn_deploy_signature": deploy_signature,
        "partition_tag": config.partition_tag,
        "extractor": config.extractor_tag,
        "train_preprocess_tag": config.train_preprocess_tag,
        "head_tag": config.head_tag,
        "predict_preprocess_tag": config.predict_preprocess_tag,
        "eval_config": eval_config.to_dict(),
    }
    return hash_payload(payload, prefix="cnn-deploy-eval", signature_version=CNN_DEPLOY_EVAL_SIGNATURE_VERSION)


def rnn_export_signature(config: RnnExperimentConfig, experiment_signature: str, export_config: RnnTfliteExportConfig) -> str:
    payload = {
        "rnn_experiment_signature": experiment_signature,
        "partition_tag": config.data.partition_tag,
        "feature_spec_tag": config.data.feature_spec_tag,
        "sequence_spec_tag": config.data.sequence_spec_tag,
        "architecture_tag": config.architecture.tag,
        "export_config": export_config.to_dict(),
        "export_layout": "encoder_plus_head",
    }
    return hash_payload(payload, prefix="rnn-tflite-export", signature_version=RNN_EXPORT_SIGNATURE_VERSION)


def rnn_tflite_eval_signature(config: RnnExperimentConfig, experiment_signature: str, export_signature: str, eval_config: RnnTfliteEvalConfig) -> str:
    payload = {
        "rnn_experiment_signature": experiment_signature,
        "rnn_export_signature": export_signature,
        "partition_tag": config.data.partition_tag,
        "feature_spec_tag": config.data.feature_spec_tag,
        "sequence_spec_tag": config.data.sequence_spec_tag,
        "architecture_tag": config.architecture.tag,
        "eval_config": eval_config.to_dict(),
    }
    return hash_payload(payload, prefix="rnn-tflite-eval", signature_version=RNN_TFLITE_EVAL_SIGNATURE_VERSION)


def rnn_deploy_eval_signature(config: RnnExperimentConfig, experiment_signature: str, export_signature: str, cnn_deploy_signature: str, eval_config: RnnDeployEvalConfig) -> str:
    payload = {
        "rnn_experiment_signature": experiment_signature,
        "rnn_export_signature": export_signature,
        "cnn_deploy_signature": cnn_deploy_signature,
        "partition_tag": config.data.partition_tag,
        "feature_spec_tag": config.data.feature_spec_tag,
        "sequence_spec_tag": config.data.sequence_spec_tag,
        "architecture_tag": config.architecture.tag,
        "eval_config": eval_config.to_dict(),
    }
    return hash_payload(payload, prefix="rnn-deploy-eval", signature_version=RNN_DEPLOY_EVAL_SIGNATURE_VERSION)


class _BaseRegistry:
    table_name: str
    create_sql: str

    def __init__(self, db_path: Union[str, Path]) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(self.create_sql)
            conn.commit()


class CnnDeployRegistry(_BaseRegistry):
    table_name = "cnn_deploy_runs"
    create_sql = """
        CREATE TABLE IF NOT EXISTS cnn_deploy_runs (
            cnn_deploy_signature TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            training_signature TEXT NOT NULL,
            config_json TEXT NOT NULL,
            deploy_config_json TEXT NOT NULL,
            deploy_dir TEXT,
            deploy_manifest_path TEXT,
            saved_model_dir TEXT,
            calibration_dir TEXT,
            inspector_script_path TEXT,
            quantize_script_path TEXT,
            compile_script_path TEXT,
            last_error TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            signature_version TEXT NOT NULL
        )
    """

    def get(self, deploy_signature: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                f"""
                SELECT cnn_deploy_signature, status, training_signature, config_json, deploy_config_json,
                       deploy_dir, deploy_manifest_path, saved_model_dir, calibration_dir,
                       inspector_script_path, quantize_script_path, compile_script_path,
                       last_error, created_at, updated_at, signature_version
                FROM {self.table_name}
                WHERE cnn_deploy_signature = ?
                """,
                (deploy_signature,),
            ).fetchone()
        if not row:
            return None
        return {
            "cnn_deploy_signature": row[0], "status": row[1], "training_signature": row[2],
            "config": json.loads(row[3]), "deploy_config": json.loads(row[4]),
            "deploy_dir": row[5], "deploy_manifest_path": row[6], "saved_model_dir": row[7],
            "calibration_dir": row[8], "inspector_script_path": row[9], "quantize_script_path": row[10],
            "compile_script_path": row[11], "last_error": row[12], "created_at": row[13],
            "updated_at": row[14], "signature_version": row[15],
        }

    def reserve(self, deploy_signature: str, training_signature: str, config: CnnExperimentConfig, deploy_config: CnnVitisAiDeployConfig, *, deploy_dir: Union[str, Path], deploy_manifest_path: Union[str, Path], saved_model_dir: Union[str, Path], calibration_dir: Union[str, Path], inspector_script_path: Union[str, Path], quantize_script_path: Union[str, Path], compile_script_path: Union[str, Path]) -> None:
        now = datetime.utcnow().isoformat()
        with self._connect() as conn:
            conn.execute(
                f"""
                INSERT INTO {self.table_name}(
                    cnn_deploy_signature, status, training_signature, config_json, deploy_config_json,
                    deploy_dir, deploy_manifest_path, saved_model_dir, calibration_dir,
                    inspector_script_path, quantize_script_path, compile_script_path,
                    last_error, created_at, updated_at, signature_version
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?)
                ON CONFLICT(cnn_deploy_signature) DO UPDATE SET
                    status=excluded.status,
                    training_signature=excluded.training_signature,
                    config_json=excluded.config_json,
                    deploy_config_json=excluded.deploy_config_json,
                    deploy_dir=excluded.deploy_dir,
                    deploy_manifest_path=excluded.deploy_manifest_path,
                    saved_model_dir=excluded.saved_model_dir,
                    calibration_dir=excluded.calibration_dir,
                    inspector_script_path=excluded.inspector_script_path,
                    quantize_script_path=excluded.quantize_script_path,
                    compile_script_path=excluded.compile_script_path,
                    updated_at=excluded.updated_at,
                    signature_version=excluded.signature_version,
                    last_error=NULL
                """,
                (
                    deploy_signature, "running", training_signature,
                    canonical_json(config.to_dict()), canonical_json(deploy_config.to_dict()),
                    str(deploy_dir), str(deploy_manifest_path), str(saved_model_dir), str(calibration_dir),
                    str(inspector_script_path), str(quantize_script_path), str(compile_script_path),
                    now, now, CNN_DEPLOY_SIGNATURE_VERSION,
                ),
            )
            conn.commit()

    def find_latest_completed_for_training(self, training_signature: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                f"""
                SELECT cnn_deploy_signature
                FROM {self.table_name}
                WHERE status='completed' AND training_signature=?
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (training_signature,),
            ).fetchone()
        return self.get(row[0]) if row else None

    def complete(self, deploy_signature: str) -> None:
        now = datetime.utcnow().isoformat()
        with self._connect() as conn:
            conn.execute(f"UPDATE {self.table_name} SET status=?, updated_at=?, last_error=NULL WHERE cnn_deploy_signature=?", ("completed", now, deploy_signature))
            conn.commit()

    def fail(self, deploy_signature: str, error: str) -> None:
        now = datetime.utcnow().isoformat()
        with self._connect() as conn:
            conn.execute(f"UPDATE {self.table_name} SET status=?, last_error=?, updated_at=? WHERE cnn_deploy_signature=?", ("failed", error, now, deploy_signature))
            conn.commit()


class RnnExportRegistry(_BaseRegistry):
    table_name = "rnn_tflite_exports"
    create_sql = """
        CREATE TABLE IF NOT EXISTS rnn_tflite_exports (
            rnn_export_signature TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            rnn_experiment_signature TEXT NOT NULL,
            config_json TEXT NOT NULL,
            export_config_json TEXT NOT NULL,
            export_dir TEXT,
            export_manifest_path TEXT,
            saved_model_dir TEXT,
            tflite_model_path TEXT,
            last_error TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            signature_version TEXT NOT NULL
        )
    """

    def get(self, export_signature: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                f"""
                SELECT rnn_export_signature, status, rnn_experiment_signature, config_json, export_config_json,
                       export_dir, export_manifest_path, saved_model_dir, tflite_model_path,
                       last_error, created_at, updated_at, signature_version
                FROM {self.table_name}
                WHERE rnn_export_signature = ?
                """,
                (export_signature,),
            ).fetchone()
        if not row:
            return None
        return {
            "rnn_export_signature": row[0], "status": row[1], "rnn_experiment_signature": row[2],
            "config": json.loads(row[3]), "export_config": json.loads(row[4]), "export_dir": row[5],
            "export_manifest_path": row[6], "saved_model_dir": row[7], "tflite_model_path": row[8],
            "last_error": row[9], "created_at": row[10], "updated_at": row[11], "signature_version": row[12],
        }

    def reserve(self, export_signature: str, experiment_signature: str, config: RnnExperimentConfig, export_config: RnnTfliteExportConfig, *, export_dir: Union[str, Path], export_manifest_path: Union[str, Path], saved_model_dir: Union[str, Path], tflite_model_path: Union[str, Path]) -> None:
        now = datetime.utcnow().isoformat()
        with self._connect() as conn:
            conn.execute(
                f"""
                INSERT INTO {self.table_name}(
                    rnn_export_signature, status, rnn_experiment_signature, config_json, export_config_json,
                    export_dir, export_manifest_path, saved_model_dir, tflite_model_path,
                    last_error, created_at, updated_at, signature_version
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?)
                ON CONFLICT(rnn_export_signature) DO UPDATE SET
                    status=excluded.status,
                    rnn_experiment_signature=excluded.rnn_experiment_signature,
                    config_json=excluded.config_json,
                    export_config_json=excluded.export_config_json,
                    export_dir=excluded.export_dir,
                    export_manifest_path=excluded.export_manifest_path,
                    saved_model_dir=excluded.saved_model_dir,
                    tflite_model_path=excluded.tflite_model_path,
                    updated_at=excluded.updated_at,
                    signature_version=excluded.signature_version,
                    last_error=NULL
                """,
                (
                    export_signature, "running", experiment_signature,
                    canonical_json(config.to_dict()), canonical_json(export_config.to_dict()),
                    str(export_dir), str(export_manifest_path), str(saved_model_dir), str(tflite_model_path),
                    now, now, RNN_EXPORT_SIGNATURE_VERSION,
                ),
            )
            conn.commit()

    def complete(self, export_signature: str) -> None:
        now = datetime.utcnow().isoformat()
        with self._connect() as conn:
            conn.execute(f"UPDATE {self.table_name} SET status=?, updated_at=?, last_error=NULL WHERE rnn_export_signature=?", ("completed", now, export_signature))
            conn.commit()

    def fail(self, export_signature: str, error: str) -> None:
        now = datetime.utcnow().isoformat()
        with self._connect() as conn:
            conn.execute(f"UPDATE {self.table_name} SET status=?, last_error=?, updated_at=? WHERE rnn_export_signature=?", ("failed", error, now, export_signature))
            conn.commit()


class CnnQuantEvalRegistry(_BaseRegistry):
    table_name = "cnn_quant_evals"
    create_sql = """
        CREATE TABLE IF NOT EXISTS cnn_quant_evals (
            cnn_quant_eval_signature TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            training_signature TEXT NOT NULL,
            cnn_deploy_signature TEXT NOT NULL,
            config_json TEXT NOT NULL,
            eval_config_json TEXT NOT NULL,
            eval_dir TEXT,
            eval_manifest_path TEXT,
            quantized_classifier_path TEXT,
            last_error TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            signature_version TEXT NOT NULL
        )
    """

    def get(self, eval_signature: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                f"""
                SELECT cnn_quant_eval_signature, status, training_signature, cnn_deploy_signature,
                       config_json, eval_config_json, eval_dir, eval_manifest_path,
                       quantized_classifier_path, last_error, created_at, updated_at, signature_version
                FROM {self.table_name}
                WHERE cnn_quant_eval_signature = ?
                """,
                (eval_signature,),
            ).fetchone()
        if not row:
            return None
        return {
            "cnn_quant_eval_signature": row[0], "status": row[1], "training_signature": row[2],
            "cnn_deploy_signature": row[3], "config": json.loads(row[4]), "eval_config": json.loads(row[5]),
            "eval_dir": row[6], "eval_manifest_path": row[7], "quantized_classifier_path": row[8],
            "last_error": row[9], "created_at": row[10], "updated_at": row[11], "signature_version": row[12],
        }

    def reserve(self, eval_signature: str, training_signature: str, deploy_signature: str, config: CnnExperimentConfig, eval_config: CnnQuantEvalConfig, *, eval_dir: Union[str, Path], eval_manifest_path: Union[str, Path], quantized_classifier_path: Union[str, Path]) -> None:
        now = datetime.utcnow().isoformat()
        with self._connect() as conn:
            conn.execute(
                f"""
                INSERT INTO {self.table_name}(
                    cnn_quant_eval_signature, status, training_signature, cnn_deploy_signature, config_json, eval_config_json,
                    eval_dir, eval_manifest_path, quantized_classifier_path, last_error, created_at, updated_at, signature_version
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?)
                ON CONFLICT(cnn_quant_eval_signature) DO UPDATE SET
                    status=excluded.status,
                    training_signature=excluded.training_signature,
                    cnn_deploy_signature=excluded.cnn_deploy_signature,
                    config_json=excluded.config_json,
                    eval_config_json=excluded.eval_config_json,
                    eval_dir=excluded.eval_dir,
                    eval_manifest_path=excluded.eval_manifest_path,
                    quantized_classifier_path=excluded.quantized_classifier_path,
                    updated_at=excluded.updated_at,
                    signature_version=excluded.signature_version,
                    last_error=NULL
                """,
                (
                    eval_signature, "running", training_signature, deploy_signature,
                    canonical_json(config.to_dict()), canonical_json(eval_config.to_dict()),
                    str(eval_dir), str(eval_manifest_path), str(quantized_classifier_path),
                    now, now, CNN_QUANT_EVAL_SIGNATURE_VERSION,
                ),
            )
            conn.commit()

    def complete(self, eval_signature: str) -> None:
        now = datetime.utcnow().isoformat()
        with self._connect() as conn:
            conn.execute(f"UPDATE {self.table_name} SET status=?, updated_at=?, last_error=NULL WHERE cnn_quant_eval_signature=?", ("completed", now, eval_signature))
            conn.commit()

    def fail(self, eval_signature: str, error: str) -> None:
        now = datetime.utcnow().isoformat()
        with self._connect() as conn:
            conn.execute(f"UPDATE {self.table_name} SET status=?, last_error=?, updated_at=? WHERE cnn_quant_eval_signature=?", ("failed", error, now, eval_signature))
            conn.commit()


class CnnExtractorEvalRegistry(_BaseRegistry):
    table_name = "cnn_extractor_evals"
    create_sql = """
        CREATE TABLE IF NOT EXISTS cnn_extractor_evals (
            cnn_extractor_eval_signature TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            training_signature TEXT NOT NULL,
            cnn_deploy_signature TEXT NOT NULL,
            config_json TEXT NOT NULL,
            eval_config_json TEXT NOT NULL,
            eval_dir TEXT,
            eval_manifest_path TEXT,
            quantized_extractor_path TEXT,
            float_classifier_path TEXT,
            float_probe_head_path TEXT,
            last_error TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            signature_version TEXT NOT NULL
        )
    """

    def get(self, eval_signature: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                f"""
                SELECT cnn_extractor_eval_signature, status, training_signature, cnn_deploy_signature,
                       config_json, eval_config_json, eval_dir, eval_manifest_path,
                       quantized_extractor_path, float_classifier_path, float_probe_head_path, last_error,
                       created_at, updated_at, signature_version
                FROM {self.table_name}
                WHERE cnn_extractor_eval_signature = ?
                """,
                (eval_signature,),
            ).fetchone()
        if not row:
            return None
        return {
            "cnn_extractor_eval_signature": row[0], "status": row[1], "training_signature": row[2],
            "cnn_deploy_signature": row[3], "config": json.loads(row[4]), "eval_config": json.loads(row[5]),
            "eval_dir": row[6], "eval_manifest_path": row[7], "quantized_extractor_path": row[8],
            "float_classifier_path": row[9], "float_probe_head_path": row[10], "last_error": row[11],
            "created_at": row[12], "updated_at": row[13], "signature_version": row[14],
        }

    def reserve(self, eval_signature: str, training_signature: str, deploy_signature: str, config: CnnExperimentConfig, eval_config: CnnExtractorEvalConfig, *, eval_dir: Union[str, Path], eval_manifest_path: Union[str, Path], quantized_extractor_path: Union[str, Path], float_classifier_path: Union[str, Path], float_probe_head_path: Union[str, Path]) -> None:
        now = datetime.utcnow().isoformat()
        with self._connect() as conn:
            conn.execute(
                f"""
                INSERT INTO {self.table_name}(
                    cnn_extractor_eval_signature, status, training_signature, cnn_deploy_signature, config_json, eval_config_json,
                    eval_dir, eval_manifest_path, quantized_extractor_path, float_classifier_path, float_probe_head_path,
                    last_error, created_at, updated_at, signature_version
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?)
                ON CONFLICT(cnn_extractor_eval_signature) DO UPDATE SET
                    status=excluded.status,
                    training_signature=excluded.training_signature,
                    cnn_deploy_signature=excluded.cnn_deploy_signature,
                    config_json=excluded.config_json,
                    eval_config_json=excluded.eval_config_json,
                    eval_dir=excluded.eval_dir,
                    eval_manifest_path=excluded.eval_manifest_path,
                    quantized_extractor_path=excluded.quantized_extractor_path,
                    float_classifier_path=excluded.float_classifier_path,
                    float_probe_head_path=excluded.float_probe_head_path,
                    updated_at=excluded.updated_at,
                    signature_version=excluded.signature_version,
                    last_error=NULL
                """,
                (
                    eval_signature, "running", training_signature, deploy_signature,
                    canonical_json(config.to_dict()), canonical_json(eval_config.to_dict()),
                    str(eval_dir), str(eval_manifest_path), str(quantized_extractor_path), str(float_classifier_path), str(float_probe_head_path),
                    now, now, CNN_EXTRACTOR_EVAL_SIGNATURE_VERSION,
                ),
            )
            conn.commit()

    def complete(self, eval_signature: str) -> None:
        now = datetime.utcnow().isoformat()
        with self._connect() as conn:
            conn.execute(f"UPDATE {self.table_name} SET status=?, updated_at=?, last_error=NULL WHERE cnn_extractor_eval_signature=?", ("completed", now, eval_signature))
            conn.commit()

    def fail(self, eval_signature: str, error: str) -> None:
        now = datetime.utcnow().isoformat()
        with self._connect() as conn:
            conn.execute(f"UPDATE {self.table_name} SET status=?, last_error=?, updated_at=? WHERE cnn_extractor_eval_signature=?", ("failed", error, now, eval_signature))
            conn.commit()


class CnnDeployEvalRegistry(_BaseRegistry):
    table_name = "cnn_deploy_evals"
    create_sql = """
        CREATE TABLE IF NOT EXISTS cnn_deploy_evals (
            cnn_deploy_eval_signature TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            training_signature TEXT NOT NULL,
            cnn_deploy_signature TEXT NOT NULL,
            config_json TEXT NOT NULL,
            eval_config_json TEXT NOT NULL,
            eval_dir TEXT,
            eval_manifest_path TEXT,
            quantized_classifier_path TEXT,
            quantized_extractor_path TEXT,
            float_classifier_path TEXT,
            float_probe_head_path TEXT,
            last_error TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            signature_version TEXT NOT NULL
        )
    """

    def get(self, eval_signature: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                f"""
                SELECT cnn_deploy_eval_signature, status, training_signature, cnn_deploy_signature,
                       config_json, eval_config_json, eval_dir, eval_manifest_path,
                       quantized_classifier_path, quantized_extractor_path, float_classifier_path, float_probe_head_path,
                       last_error, created_at, updated_at, signature_version
                FROM {self.table_name}
                WHERE cnn_deploy_eval_signature = ?
                """,
                (eval_signature,),
            ).fetchone()
        if not row:
            return None
        return {
            "cnn_deploy_eval_signature": row[0], "status": row[1], "training_signature": row[2],
            "cnn_deploy_signature": row[3], "config": json.loads(row[4]), "eval_config": json.loads(row[5]),
            "eval_dir": row[6], "eval_manifest_path": row[7], "quantized_classifier_path": row[8],
            "quantized_extractor_path": row[9], "float_classifier_path": row[10], "float_probe_head_path": row[11],
            "last_error": row[12], "created_at": row[13], "updated_at": row[14], "signature_version": row[15],
        }

    def reserve(self, eval_signature: str, training_signature: str, deploy_signature: str, config: CnnExperimentConfig, eval_config: CnnExtractorEvalConfig, *, eval_dir: Union[str, Path], eval_manifest_path: Union[str, Path], quantized_classifier_path: Union[str, Path], quantized_extractor_path: Union[str, Path], float_classifier_path: Union[str, Path], float_probe_head_path: Union[str, Path]) -> None:
        now = datetime.utcnow().isoformat()
        with self._connect() as conn:
            conn.execute(
                f"""
                INSERT INTO {self.table_name}(
                    cnn_deploy_eval_signature, status, training_signature, cnn_deploy_signature, config_json, eval_config_json,
                    eval_dir, eval_manifest_path, quantized_classifier_path, quantized_extractor_path, float_classifier_path, float_probe_head_path,
                    last_error, created_at, updated_at, signature_version
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?)
                ON CONFLICT(cnn_deploy_eval_signature) DO UPDATE SET
                    status=excluded.status,
                    training_signature=excluded.training_signature,
                    cnn_deploy_signature=excluded.cnn_deploy_signature,
                    config_json=excluded.config_json,
                    eval_config_json=excluded.eval_config_json,
                    eval_dir=excluded.eval_dir,
                    eval_manifest_path=excluded.eval_manifest_path,
                    quantized_classifier_path=excluded.quantized_classifier_path,
                    quantized_extractor_path=excluded.quantized_extractor_path,
                    float_classifier_path=excluded.float_classifier_path,
                    float_probe_head_path=excluded.float_probe_head_path,
                    updated_at=excluded.updated_at,
                    signature_version=excluded.signature_version,
                    last_error=NULL
                """,
                (
                    eval_signature, "running", training_signature, deploy_signature,
                    canonical_json(config.to_dict()), canonical_json(eval_config.to_dict()),
                    str(eval_dir), str(eval_manifest_path), str(quantized_classifier_path), str(quantized_extractor_path), str(float_classifier_path), str(float_probe_head_path),
                    now, now, CNN_DEPLOY_EVAL_SIGNATURE_VERSION,
                ),
            )
            conn.commit()

    def complete(self, eval_signature: str) -> None:
        now = datetime.utcnow().isoformat()
        with self._connect() as conn:
            conn.execute(f"UPDATE {self.table_name} SET status=?, updated_at=?, last_error=NULL WHERE cnn_deploy_eval_signature=?", ("completed", now, eval_signature))
            conn.commit()

    def fail(self, eval_signature: str, error: str) -> None:
        now = datetime.utcnow().isoformat()
        with self._connect() as conn:
            conn.execute(f"UPDATE {self.table_name} SET status=?, last_error=?, updated_at=? WHERE cnn_deploy_eval_signature=?", ("failed", error, now, eval_signature))
            conn.commit()


class RnnTfliteEvalRegistry(_BaseRegistry):
    table_name = "rnn_tflite_evals"
    create_sql = """
        CREATE TABLE IF NOT EXISTS rnn_tflite_evals (
            rnn_tflite_eval_signature TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            rnn_experiment_signature TEXT NOT NULL,
            rnn_export_signature TEXT NOT NULL,
            config_json TEXT NOT NULL,
            eval_config_json TEXT NOT NULL,
            eval_dir TEXT,
            eval_manifest_path TEXT,
            tflite_model_path TEXT,
            last_error TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            signature_version TEXT NOT NULL
        )
    """

    def get(self, eval_signature: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                f"""
                SELECT rnn_tflite_eval_signature, status, rnn_experiment_signature, rnn_export_signature,
                       config_json, eval_config_json, eval_dir, eval_manifest_path,
                       tflite_model_path, last_error, created_at, updated_at, signature_version
                FROM {self.table_name}
                WHERE rnn_tflite_eval_signature = ?
                """,
                (eval_signature,),
            ).fetchone()
        if not row:
            return None
        return {
            "rnn_tflite_eval_signature": row[0], "status": row[1], "rnn_experiment_signature": row[2],
            "rnn_export_signature": row[3], "config": json.loads(row[4]), "eval_config": json.loads(row[5]),
            "eval_dir": row[6], "eval_manifest_path": row[7], "tflite_model_path": row[8],
            "last_error": row[9], "created_at": row[10], "updated_at": row[11], "signature_version": row[12],
        }

    def reserve(self, eval_signature: str, experiment_signature: str, export_signature: str, config: RnnExperimentConfig, eval_config: RnnTfliteEvalConfig, *, eval_dir: Union[str, Path], eval_manifest_path: Union[str, Path], tflite_model_path: Union[str, Path]) -> None:
        now = datetime.utcnow().isoformat()
        with self._connect() as conn:
            conn.execute(
                f"""
                INSERT INTO {self.table_name}(
                    rnn_tflite_eval_signature, status, rnn_experiment_signature, rnn_export_signature,
                    config_json, eval_config_json, eval_dir, eval_manifest_path, tflite_model_path,
                    last_error, created_at, updated_at, signature_version
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?)
                ON CONFLICT(rnn_tflite_eval_signature) DO UPDATE SET
                    status=excluded.status,
                    rnn_experiment_signature=excluded.rnn_experiment_signature,
                    rnn_export_signature=excluded.rnn_export_signature,
                    config_json=excluded.config_json,
                    eval_config_json=excluded.eval_config_json,
                    eval_dir=excluded.eval_dir,
                    eval_manifest_path=excluded.eval_manifest_path,
                    tflite_model_path=excluded.tflite_model_path,
                    updated_at=excluded.updated_at,
                    signature_version=excluded.signature_version,
                    last_error=NULL
                """,
                (
                    eval_signature, "running", experiment_signature, export_signature,
                    canonical_json(config.to_dict()), canonical_json(eval_config.to_dict()),
                    str(eval_dir), str(eval_manifest_path), str(tflite_model_path),
                    now, now, RNN_TFLITE_EVAL_SIGNATURE_VERSION,
                ),
            )
            conn.commit()

    def complete(self, eval_signature: str) -> None:
        now = datetime.utcnow().isoformat()
        with self._connect() as conn:
            conn.execute(f"UPDATE {self.table_name} SET status=?, updated_at=?, last_error=NULL WHERE rnn_tflite_eval_signature=?", ("completed", now, eval_signature))
            conn.commit()

    def fail(self, eval_signature: str, error: str) -> None:
        now = datetime.utcnow().isoformat()
        with self._connect() as conn:
            conn.execute(f"UPDATE {self.table_name} SET status=?, last_error=?, updated_at=? WHERE rnn_tflite_eval_signature=?", ("failed", error, now, eval_signature))
            conn.commit()


class RnnDeployEvalRegistry(_BaseRegistry):
    table_name = "rnn_deploy_evals"
    create_sql = """
        CREATE TABLE IF NOT EXISTS rnn_deploy_evals (
            rnn_deploy_eval_signature TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            rnn_experiment_signature TEXT NOT NULL,
            rnn_export_signature TEXT NOT NULL,
            cnn_deploy_signature TEXT NOT NULL,
            config_json TEXT NOT NULL,
            eval_config_json TEXT NOT NULL,
            eval_dir TEXT,
            eval_manifest_path TEXT,
            tflite_model_path TEXT,
            quantized_extractor_path TEXT,
            last_error TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            signature_version TEXT NOT NULL
        )
    """

    def get(self, eval_signature: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                f"""
                SELECT rnn_deploy_eval_signature, status, rnn_experiment_signature, rnn_export_signature, cnn_deploy_signature,
                       config_json, eval_config_json, eval_dir, eval_manifest_path, tflite_model_path, quantized_extractor_path,
                       last_error, created_at, updated_at, signature_version
                FROM {self.table_name}
                WHERE rnn_deploy_eval_signature = ?
                """,
                (eval_signature,),
            ).fetchone()
        if not row:
            return None
        return {
            "rnn_deploy_eval_signature": row[0], "status": row[1], "rnn_experiment_signature": row[2],
            "rnn_export_signature": row[3], "cnn_deploy_signature": row[4], "config": json.loads(row[5]),
            "eval_config": json.loads(row[6]), "eval_dir": row[7], "eval_manifest_path": row[8],
            "tflite_model_path": row[9], "quantized_extractor_path": row[10], "last_error": row[11],
            "created_at": row[12], "updated_at": row[13], "signature_version": row[14],
        }

    def reserve(self, eval_signature: str, experiment_signature: str, export_signature: str, cnn_deploy_signature: str, config: RnnExperimentConfig, eval_config: RnnDeployEvalConfig, *, eval_dir: Union[str, Path], eval_manifest_path: Union[str, Path], tflite_model_path: Union[str, Path], quantized_extractor_path: Union[str, Path]) -> None:
        now = datetime.utcnow().isoformat()
        with self._connect() as conn:
            conn.execute(
                f"""
                INSERT INTO {self.table_name}(
                    rnn_deploy_eval_signature, status, rnn_experiment_signature, rnn_export_signature, cnn_deploy_signature,
                    config_json, eval_config_json, eval_dir, eval_manifest_path, tflite_model_path, quantized_extractor_path,
                    last_error, created_at, updated_at, signature_version
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?)
                ON CONFLICT(rnn_deploy_eval_signature) DO UPDATE SET
                    status=excluded.status,
                    rnn_experiment_signature=excluded.rnn_experiment_signature,
                    rnn_export_signature=excluded.rnn_export_signature,
                    cnn_deploy_signature=excluded.cnn_deploy_signature,
                    config_json=excluded.config_json,
                    eval_config_json=excluded.eval_config_json,
                    eval_dir=excluded.eval_dir,
                    eval_manifest_path=excluded.eval_manifest_path,
                    tflite_model_path=excluded.tflite_model_path,
                    quantized_extractor_path=excluded.quantized_extractor_path,
                    updated_at=excluded.updated_at,
                    signature_version=excluded.signature_version,
                    last_error=NULL
                """,
                (
                    eval_signature, "running", experiment_signature, export_signature, cnn_deploy_signature,
                    canonical_json(config.to_dict()), canonical_json(eval_config.to_dict()),
                    str(eval_dir), str(eval_manifest_path), str(tflite_model_path), str(quantized_extractor_path),
                    now, now, RNN_DEPLOY_EVAL_SIGNATURE_VERSION,
                ),
            )
            conn.commit()

    def complete(self, eval_signature: str) -> None:
        now = datetime.utcnow().isoformat()
        with self._connect() as conn:
            conn.execute(f"UPDATE {self.table_name} SET status=?, updated_at=?, last_error=NULL WHERE rnn_deploy_eval_signature=?", ("completed", now, eval_signature))
            conn.commit()

    def fail(self, eval_signature: str, error: str) -> None:
        now = datetime.utcnow().isoformat()
        with self._connect() as conn:
            conn.execute(f"UPDATE {self.table_name} SET status=?, last_error=?, updated_at=? WHERE rnn_deploy_eval_signature=?", ("failed", error, now, eval_signature))
            conn.commit()