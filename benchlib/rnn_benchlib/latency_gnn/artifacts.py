from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

from rnn_benchlib.storage.jsonl import read_json, read_jsonl, write_json, write_jsonl
from rnn_benchlib.storage.layout import lot_benchmark_config_file, lot_benchmark_profile_file

RUNTIME_FILENAMES = {
    "float": "float_measurements.jsonl",
    "tflite": "tflite_measurements.jsonl",
}


@dataclass(frozen=True)
class ModelArtifactBundle:
    lot_id: str
    model_id: str
    model_dir: Path
    graph_record_path: Path
    spec_path: Optional[Path]
    manifest_path: Optional[Path]
    member_record: Dict


@dataclass(frozen=True)
class MeasurementBundle:
    lot_id: str
    benchmark_id: str
    model_id: str
    runtime_kind: str
    profile_id: str
    path: Path
    graph_record: Dict
    rows: list[Dict]
    spec: Dict | None
    manifest: Dict | None
    member_record: Dict


def _stable_dataset_id(lot_id: str, benchmark_id: str, runtime_scope: str, profile_ids: Dict[str, str]) -> str:
    payload = {
        "lot_id": lot_id,
        "benchmark_id": benchmark_id,
        "runtime_scope": runtime_scope,
        "profile_ids": profile_ids,
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return "dataset_" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def lot_root(output_root: str | Path, lot_id: str) -> Path:
    return Path(output_root).expanduser().resolve() / "lots" / lot_id


def lot_members_path(output_root: str | Path, lot_id: str) -> Path:
    return lot_root(output_root, lot_id) / "members.jsonl"


def gnn_latency_root(output_root: str | Path, lot_id: str) -> Path:
    path = lot_root(output_root, lot_id) / "gnn_latency"
    path.mkdir(parents=True, exist_ok=True)
    return path


def datasets_dir(output_root: str | Path, lot_id: str) -> Path:
    path = gnn_latency_root(output_root, lot_id) / "datasets"
    path.mkdir(parents=True, exist_ok=True)
    return path


def dataset_dir(output_root: str | Path, lot_id: str, dataset_id: str) -> Path:
    path = datasets_dir(output_root, lot_id) / dataset_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def dataset_export_path(output_root: str | Path, lot_id: str, dataset_id: str) -> Path:
    return dataset_dir(output_root, lot_id, dataset_id) / "export.json"


def dataset_rows_path(output_root: str | Path, lot_id: str, dataset_id: str) -> Path:
    return dataset_dir(output_root, lot_id, dataset_id) / "rows.jsonl"


def splits_dir(output_root: str | Path, lot_id: str, dataset_id: str) -> Path:
    path = dataset_dir(output_root, lot_id, dataset_id) / "splits"
    path.mkdir(parents=True, exist_ok=True)
    return path


def split_path(output_root: str | Path, lot_id: str, dataset_id: str, split_id: str) -> Path:
    return splits_dir(output_root, lot_id, dataset_id) / f"split_{split_id}.json"


def runs_dir(output_root: str | Path, lot_id: str, dataset_id: str) -> Path:
    path = dataset_dir(output_root, lot_id, dataset_id) / "runs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_dir(output_root: str | Path, lot_id: str, dataset_id: str, run_id: str) -> Path:
    path = runs_dir(output_root, lot_id, dataset_id) / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def checkpoints_dir(output_root: str | Path, lot_id: str, dataset_id: str, run_id: str) -> Path:
    path = run_dir(output_root, lot_id, dataset_id, run_id) / "checkpoints"
    path.mkdir(parents=True, exist_ok=True)
    return path


def eval_dir(output_root: str | Path, lot_id: str, dataset_id: str, run_id: str) -> Path:
    path = run_dir(output_root, lot_id, dataset_id, run_id) / "eval"
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_lot_members(output_root: str | Path, lot_id: str) -> List[Dict]:
    path = lot_members_path(output_root, lot_id)
    members = read_jsonl(str(path))
    if not members:
        raise FileNotFoundError(f"No existe o está vacío el members.jsonl del lote: {path}")
    return members


def read_benchmark_config(output_root: str | Path, lot_id: str, benchmark_id: str) -> Dict:
    path = Path(lot_benchmark_config_file(str(Path(output_root).expanduser().resolve()), lot_id, benchmark_id))
    payload = read_json(str(path), default=None)
    if payload is None:
        raise FileNotFoundError(f"No existe la configuración de benchmark solicitada: {path}")
    return payload


def resolve_profile_ids(output_root: str | Path, lot_id: str, benchmark_id: str, runtime_filter: Iterable[str] | None = None) -> Dict[str, str]:
    config_payload = read_benchmark_config(output_root, lot_id, benchmark_id)
    all_profile_ids = dict(config_payload.get("profile_ids") or {})
    allowed = set(runtime_filter or all_profile_ids.keys())
    selected = {runtime: profile_id for runtime, profile_id in all_profile_ids.items() if runtime in allowed}
    if not selected:
        raise RuntimeError(
            f"La configuración de benchmark {benchmark_id} no expone perfiles para runtime_filter={sorted(allowed)}"
        )
    return selected


def make_dataset_id(output_root: str | Path, lot_id: str, benchmark_id: str, runtime_filter: Iterable[str] | None = None) -> str:
    profile_ids = resolve_profile_ids(output_root, lot_id, benchmark_id, runtime_filter=runtime_filter)
    runtime_scope = "both" if set(profile_ids.keys()) == {"float", "tflite"} else next(iter(profile_ids.keys()))
    return _stable_dataset_id(lot_id, benchmark_id, runtime_scope, profile_ids)


def iter_model_bundles_for_lot(output_root: str | Path, lot_id: str) -> Iterator[ModelArtifactBundle]:
    root = Path(output_root).expanduser().resolve()
    for member in read_lot_members(root, lot_id):
        model_id = str(member.get("model_id") or "").strip()
        if not model_id:
            continue
        model_dir = root / "models" / model_id
        graph_path = model_dir / "graphs" / "graph_record.json"
        if not graph_path.exists():
            continue
        yield ModelArtifactBundle(
            lot_id=lot_id,
            model_id=model_id,
            model_dir=model_dir,
            graph_record_path=graph_path,
            spec_path=model_dir / "meta" / "spec.json",
            manifest_path=model_dir / "meta" / "manifest.json",
            member_record=dict(member),
        )


def iter_measurement_bundles_for_dataset(
    output_root: str | Path,
    lot_id: str,
    benchmark_id: str,
    runtime_filter: Iterable[str] | None = None,
) -> Iterator[MeasurementBundle]:
    root = Path(output_root).expanduser().resolve()
    selected_profiles = resolve_profile_ids(root, lot_id, benchmark_id, runtime_filter=runtime_filter)
    for bundle in iter_model_bundles_for_lot(root, lot_id):
        graph_record = read_json(str(bundle.graph_record_path), default=None)
        if not graph_record:
            continue
        spec = read_json(str(bundle.spec_path), default=None) if bundle.spec_path else None
        manifest = read_json(str(bundle.manifest_path), default=None) if bundle.manifest_path else None
        benchmark_root = bundle.model_dir / "benchmarks"
        if not benchmark_root.exists():
            continue
        for runtime_kind, profile_id in selected_profiles.items():
            profile_dir = benchmark_root / profile_id
            if not profile_dir.exists():
                continue
            filename = RUNTIME_FILENAMES[runtime_kind]
            rows = read_jsonl(str(profile_dir / filename))
            if not rows:
                continue
            yield MeasurementBundle(
                lot_id=lot_id,
                benchmark_id=benchmark_id,
                model_id=bundle.model_id,
                runtime_kind=runtime_kind,
                profile_id=profile_id,
                path=profile_dir / filename,
                graph_record=graph_record,
                rows=rows,
                spec=spec,
                manifest=manifest,
                member_record=bundle.member_record,
            )


def write_dataset_artifacts(output_root: str | Path, lot_id: str, dataset_id: str, export_payload: Dict, rows: List[Dict]) -> None:
    write_json(str(dataset_export_path(output_root, lot_id, dataset_id)), export_payload, indent=2)
    write_jsonl(str(dataset_rows_path(output_root, lot_id, dataset_id)), rows)


def read_dataset_rows(output_root: str | Path, lot_id: str, dataset_id: str) -> List[Dict]:
    return read_jsonl(str(dataset_rows_path(output_root, lot_id, dataset_id)))


def read_dataset_export(output_root: str | Path, lot_id: str, dataset_id: str) -> Dict:
    return read_json(str(dataset_export_path(output_root, lot_id, dataset_id)), default={})
