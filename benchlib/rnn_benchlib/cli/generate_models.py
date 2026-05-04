from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, Dict

from rnn_benchlib.config.schemas import FeatureSpec, SearchSpace
from rnn_benchlib.config.script_loader import load_rnn_config, search_space_to_dict
from rnn_benchlib.generation.scheduler import GenerationRunResult, GenerationSettings, GenerationTask, run_generation
from rnn_benchlib.sampling.sampler import enumerate_valid_specs, model_spec_to_id, model_spec_to_stable_key, summarize_sampling_pool
from rnn_benchlib.storage.jsonl import append_jsonl, read_json, read_jsonl, write_json, write_jsonl
from rnn_benchlib.storage.layout import (
    build_root_layout,
    lot_generation_config_file,
    lot_generation_runtime_file,
    lot_generation_summary_file,
    lot_json_path,
    lot_members_path,
    model_dir,
)
from rnn_benchlib.storage.registry import RnnStateStore
from rnn_benchlib.storage.state import stable_hash, utc_now_iso


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genera o recalcula un lote RNN definido por firma estable. Usa siempre ./artifacts como raíz de persistencia.")
    parser.add_argument("--config", required=True, help="Fichero de configuración centralizado de la rama RNN. Se recomienda ubicarlo en ./configs y diferenciarlo en el nombre como rnn_*.json.")
    return parser.parse_args()


def _manifest_summary(output_root: str, model_id: str) -> Dict[str, Any] | None:
    manifest_path = Path(model_dir(output_root, model_id)) / "meta" / "manifest.json"
    manifest = read_json(str(manifest_path), default=None)
    if not isinstance(manifest, dict) or manifest.get("model_id") != model_id:
        return None
    conversion = manifest.get("conversion", {}) if isinstance(manifest.get("conversion"), dict) else {}
    components_present = conversion.get("components_present", {}) if isinstance(conversion.get("components_present"), dict) else {}
    return {
        "model_id": model_id,
        "manifest_path": str(manifest_path),
        "model_dir": str(Path(output_root) / "models" / model_id),
        "conversion_status": conversion.get("status", "unknown"),
        "conversion_mode": conversion.get("conversion_mode", "unknown"),
        "uses_flex": bool(conversion.get("uses_flex", False)),
        "has_tflite": bool(components_present.get("encoder_tflite") and components_present.get("head_tflite")),
        "model_key": manifest.get("model_key"),
    }


def _repair_global_model_registry(store: RnnStateStore, output_root: str) -> int:
    models_root = Path(output_root) / "models"
    repaired = 0
    if not models_root.exists():
        return repaired
    for model_path in sorted(models_root.iterdir()):
        if not model_path.is_dir():
            continue
        model_id = model_path.name
        if store.get_model(model_id) is not None:
            continue
        manifest = _manifest_summary(output_root, model_id)
        spec_path = model_path / "meta" / "spec.json"
        spec_payload = read_json(str(spec_path), default=None)
        if manifest is None or not isinstance(spec_payload, dict):
            continue
        feature_spec = spec_payload.get("feature_spec", {}) if isinstance(spec_payload.get("feature_spec"), dict) else {}
        spec_json = spec_payload.get("spec", {}) if isinstance(spec_payload.get("spec"), dict) else {}
        model_key = manifest.get("model_key")
        if not spec_json or not model_key:
            continue
        store.upsert_model(
            model_id=model_id,
            model_key=model_key,
            family="rnn",
            feature_dim=int(feature_spec.get("feature_dim", 512)),
            video_steps=int(feature_spec.get("video_steps", 36)),
            spec_json=spec_json,
            model_dir=manifest["model_dir"],
            manifest_path=manifest["manifest_path"],
            has_float=True,
            has_tflite=bool(manifest.get("has_tflite", False)),
        )
        repaired += 1
    return repaired


def main() -> None:
    args = parse_args()
    output_root = "artifacts"
    layout = build_root_layout(output_root)
    store = RnnStateStore(layout["db_path"])

    cfg = load_rnn_config(args.config)
    search_space: SearchSpace = cfg["search_space"]
    generation_seed = int(cfg["generation_seed"])
    requested_count = int(cfg["requested_count"])
    generation_cfg = dict(cfg.get("generation") or {})
    runtime = dict(generation_cfg.get("runtime") or cfg.get("generation_runtime") or {})
    resource_manager = dict(cfg.get("resource_manager") or {})
    storage_cfg = dict(cfg.get("storage") or {})
    experiment = dict(cfg.get("experiment") or {})

    feature_spec = FeatureSpec(
        source="synthetic",
        backbone=None,
        pooling="avg",
        feature_dim=int(experiment.get("feature_dim", 512)),
        video_steps=int(experiment.get("video_steps", 36)),
    )
    num_classes = int(experiment.get("num_classes", 101))

    signature_fields = {
        "search_space": search_space_to_dict(search_space),
        "generation_seed": generation_seed,
        "requested_count": requested_count,
    }
    lot_signature = stable_hash(signature_fields, prefix="lsig")
    lot_id = store.make_stable_lot_id(kind="generation", signature_fields=signature_fields)

    repaired = _repair_global_model_registry(store, output_root)

    config_payload = {
        "schema_version": "generation_config_v2",
        "signature": {"value": lot_signature, "fields": signature_fields},
    }
    store.upsert_lot(lot_id=lot_id, kind="generation", config_json=config_payload, seed=generation_seed, requested_count=requested_count)
    write_json(lot_generation_config_file(output_root, lot_id), config_payload, indent=2)

    # Same lot => rewrite lot view from scratch, but reuse globally persisted models.
    store.clear_lot_models(lot_id)
    write_jsonl(lot_members_path(output_root, lot_id), [])

    progress = {"requested": requested_count, "resolved": 0, "created": 0, "reused": 0, "failed": 0}
    members_seen = set()
    created_at = utc_now_iso()

    def write_views(state: str, completed: bool = False) -> None:
        members = read_jsonl(lot_members_path(output_root, lot_id))
        lot_payload = {
            "lot_id": lot_id,
            "state": state,
            "signature": {"value": lot_signature, "fields_ref": "generation/config.json#signature.fields"},
            "counts": progress.copy(),
            "created_at": created_at,
            "updated_at": utc_now_iso(),
        }
        summary_payload = {
            "lot_id": lot_id,
            "signature": {"value": lot_signature, "fields_ref": "config.json#signature.fields"},
            "counts": progress.copy(),
            "completed": bool(completed),
            "updated_at": utc_now_iso(),
        }
        write_json(lot_json_path(output_root, lot_id), lot_payload, indent=2)
        write_json(lot_generation_summary_file(output_root, lot_id), summary_payload, indent=2)

    def register_member(*, position: int, model_id: str, model_key: str, spec_json: Dict[str, Any], source: str, manifest: Dict[str, Any]) -> None:
        if model_id in members_seen:
            return
        members_seen.add(model_id)
        store.upsert_model(
            model_id=model_id,
            model_key=model_key,
            family="rnn",
            feature_dim=feature_spec.feature_dim,
            video_steps=feature_spec.video_steps,
            spec_json=spec_json,
            model_dir=manifest["model_dir"],
            manifest_path=manifest["manifest_path"],
            has_float=bool(storage_cfg.get("persist_float_model", True)),
            has_tflite=bool(manifest.get("has_tflite", False)),
        )
        store.add_model_to_lot(lot_id=lot_id, model_id=model_id, position=position, source=source)
        member_record = {"member_index": position - 1, "model_id": model_id, "source": source}
        append_jsonl(lot_members_path(output_root, lot_id), member_record)
        progress["resolved"] += 1
        if source == "created":
            progress["created"] += 1
        else:
            progress["reused"] += 1
        write_views(state="running", completed=False)

    write_views(state="running", completed=False)

    pool_summary = summarize_sampling_pool(
        space=search_space,
        feature_spec=feature_spec,
        num_classes=num_classes,
        existing_keys={row.model_key for row in store.list_models()},
    )
    print("\n=== Pool de sampleo ===")
    for k, v in pool_summary.items():
        print(f"{k}: {v}")
    if repaired:
        print(f"registry_repaired_models: {repaired}")

    valid_specs = enumerate_valid_specs(space=search_space, num_classes=num_classes, video_steps=feature_spec.video_steps)
    rng = random.Random(generation_seed)
    rng.shuffle(valid_specs)
    selected_specs = valid_specs[: min(requested_count, len(valid_specs))]

    tasks = []
    for idx, spec in enumerate(selected_specs, start=1):
        model_key = model_spec_to_stable_key(spec=spec, feature_spec=feature_spec)
        model_id = model_spec_to_id(spec=spec, feature_spec=feature_spec)
        manifest = _manifest_summary(output_root, model_id)
        if manifest is not None:
            register_member(position=idx, model_id=model_id, model_key=model_key, spec_json=spec.as_key_dict(), source="reused", manifest=manifest)
            continue
        tasks.append(GenerationTask(position=idx, spec=spec, model_id=model_id, model_key=model_key))

    runtime_payload = {
        "started_at": utc_now_iso(),
        "runtime": {
            "jobs": runtime.get("jobs", "auto"),
            "estimated_worker_ram_mb": int(runtime.get("estimated_worker_ram_mb", resource_manager.get("generation_estimated_worker_ram_mb", 5500))),
            "max_ram_fraction": float(resource_manager.get("max_ram_fraction", runtime.get("max_ram_fraction", 0.65))),
            "ram_reserve_mb": int(resource_manager.get("ram_reserve_mb", runtime.get("ram_reserve_mb", 32768))),
            "worker_max_tasks": int(runtime.get("worker_max_tasks", 4)),
            "intra_op_threads": int(runtime.get("intra_op_threads", 1)),
            "inter_op_threads": int(runtime.get("inter_op_threads", 1)),
        },
    }
    write_json(lot_generation_runtime_file(output_root, lot_id), runtime_payload, indent=2)

    print(f"\nLote de generación: {lot_id}")
    print(f"lot_signature: {lot_signature}")
    print(f"Modelos solicitados: {len(selected_specs)}")
    print(f"Ya reutilizados al arrancar: {progress['reused']}")

    def handle_result(payload: Dict[str, Any]) -> None:
        task = payload["task"]
        if not payload.get("ok"):
            progress["failed"] += 1
            write_views(state="running", completed=False)
            return
        result = payload["result"]
        manifest = _manifest_summary(output_root, result["model_id"])
        if manifest is None:
            progress["failed"] += 1
            write_views(state="running", completed=False)
            return
        register_member(
            position=int(payload["position"]),
            model_id=result["model_id"],
            model_key=task.model_key,
            spec_json=task.spec.as_key_dict(),
            source="created" if result.get("status") == "created" else "reused",
            manifest=manifest,
        )

    settings = GenerationSettings(
        output_root=output_root,
        lot_id=lot_id,
        seed=generation_seed,
        feature_spec=feature_spec,
        overwrite_existing_artifacts=False,
        jobs=runtime.get("jobs", "auto"),
        estimated_worker_ram_mb=int(runtime.get("estimated_worker_ram_mb", resource_manager.get("generation_estimated_worker_ram_mb", 5500))),
        max_ram_fraction=float(resource_manager.get("max_ram_fraction", runtime.get("max_ram_fraction", 0.65))),
        ram_reserve_mb=int(resource_manager.get("ram_reserve_mb", runtime.get("ram_reserve_mb", 32768))),
        ram_check_interval_sec=float(runtime.get("ram_check_interval_sec", 0.5)),
        worker_max_tasks=int(runtime.get("worker_max_tasks", 4)),
        progress_report_interval_sec=float(runtime.get("progress_report_interval_sec", 1.0)),
        stall_warning_sec=float(runtime.get("stall_warning_sec", 60.0)),
    )
    run_result: GenerationRunResult = run_generation(settings=settings, tasks=tasks, on_result=handle_result)

    final_state = "failed" if progress["failed"] else "completed"
    write_views(state=final_state, completed=(progress["resolved"] + progress["failed"]) >= requested_count)

    print("\n=== Resumen de generación ===")
    print(f"lot_id: {lot_id}")
    print(f"lot_signature: {lot_signature}")
    print(f"resolved: {progress['resolved']}")
    print(f"created: {progress['created']}")
    print(f"reused: {progress['reused']}")
    print(f"failed: {progress['failed']}")
    print(f"jobs: {run_result.resolved_jobs}")
    print(f"runtime_path: {run_result.runtime_summary.get('runtime_path')}")
    print(f"status_path: {run_result.runtime_summary.get('status_path')}")
    print(f"progress_path: {run_result.runtime_summary.get('progress_path')}")
    print(f"errors_path: {run_result.runtime_summary.get('errors_path')}")


if __name__ == "__main__":
    main()
