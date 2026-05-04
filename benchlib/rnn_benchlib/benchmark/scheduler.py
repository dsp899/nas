from __future__ import annotations

import gc
import inspect
import math
import os
import time
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
from multiprocessing import get_context
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import psutil

from rnn_benchlib.storage.jsonl import append_jsonl, read_json, read_jsonl, write_json, write_jsonl
from rnn_benchlib.storage.layout import (
    lot_benchmark_config_dir,
    lot_benchmark_config_file,
    lot_benchmark_lot_summary_file,
    lot_benchmark_profile_file,
    lot_benchmark_run_dir,
    lot_benchmark_runtime_file,
    lot_benchmark_status_file,
    lot_benchmark_progress_file,
    lot_benchmark_errors_file,
    model_profile_dir,
)
from rnn_benchlib.storage.locks import exclusive_file_lock
from rnn_benchlib.storage.state import RnnStateStore, stable_hash
from rnn_benchlib.benchmark.signature import canonical_benchmark_signature, benchmark_id_from_signature


MB = 1024 * 1024


@dataclass(frozen=True)
class BenchmarkTask:
    position: int
    model_id: str
    manifest_path: str
    has_tflite: bool


@dataclass(frozen=True)
class BenchmarkSlot:
    slot_id: int
    cpus: Tuple[int, ...]

    @property
    def threads(self) -> int:
        return len(self.cpus)


@dataclass(frozen=True)
class BenchmarkSettings:
    output_root: str
    lot_id: str
    runtime: str
    feature_source: str
    feature_npy: Optional[str]
    num_videos: int
    feature_seed: int
    distribution: str
    warmup_runs: int
    steady_runs: int
    threads: int
    experiment_name: str
    device_name: Optional[str]
    hardware_target: Optional[str]
    notes: Optional[str]
    cpu_policy: str
    cpu_freq_khz: Optional[int]
    disable_turbo: bool
    cpu_slots: Optional[str]
    cpu_reserve_cores: int
    jobs: int | str
    estimated_worker_ram_mb: int
    max_ram_fraction: float
    ram_reserve_mb: int
    ram_check_interval_sec: float
    worker_max_tasks: int
    progress_report_interval_sec: float = 1.0
    stall_warning_sec: float = 60.0
    task_timeout_sec: float = 1800.0


@dataclass(frozen=True)
class BenchmarkRunResult:
    lot_id: str
    benchmark_id: str
    profile_ids: Dict[str, str]
    processed_models: int
    reused_measurements: int
    failed_models: List[str]
    resolved_jobs: int
    slots: List[BenchmarkSlot]
    runtime_summary: Dict[str, Any]


@dataclass(frozen=True)
class RuntimePaths:
    benchmark_id: str
    runtime_dir: str
    runtime_path: str
    status_path: str
    events_path: str
    resources_path: str
    workers_path: str
    results_path: str
    errors_path: str
    log_lock_path: str


def _profile_config_identity(benchmark_id: str, profile_signature: Dict[str, Any]) -> Dict[str, Any]:
    return {"benchmark_id": benchmark_id, "signature": profile_signature}


def _runtime_paths(output_root: str, lot_id: str, benchmark_id: str) -> RuntimePaths:
    runtime_dir = lot_benchmark_run_dir(output_root, lot_id, benchmark_id, "latest")
    os.makedirs(runtime_dir, exist_ok=True)
    return RuntimePaths(
        benchmark_id=benchmark_id,
        runtime_dir=runtime_dir,
        runtime_path=lot_benchmark_runtime_file(output_root, lot_id, benchmark_id),
        status_path=lot_benchmark_status_file(output_root, lot_id, benchmark_id),
        events_path=os.path.join(runtime_dir, "events.jsonl"),
        resources_path=os.path.join(runtime_dir, "resources.jsonl"),
        workers_path=os.path.join(runtime_dir, "workers.jsonl"),
        results_path=lot_benchmark_progress_file(output_root, lot_id, benchmark_id),
        errors_path=lot_benchmark_errors_file(output_root, lot_id, benchmark_id),
        log_lock_path=os.path.join(runtime_dir, ".log.lock"),
    )


def _append_locked_jsonl(paths: RuntimePaths, path: str, record: Dict[str, Any]) -> None:
    with exclusive_file_lock(paths.log_lock_path, poll_interval_s=0.02, stale_after_s=30.0):
        append_jsonl(path, record)


def _log_event(paths: RuntimePaths, event_type: str, **payload: Any) -> None:
    record = {"ts": _safe_round(time.time()), "event": event_type, **payload}
    _append_locked_jsonl(paths, paths.events_path, record)


def _write_status(paths: RuntimePaths, payload: Dict[str, Any]) -> None:
    write_json(paths.status_path, payload, indent=2)


def _safe_round(value: float) -> float:
    return round(float(value), 4)


def _format_eta(seconds: Optional[float]) -> Optional[str]:
    if seconds is None or not math.isfinite(seconds):
        return None
    seconds_i = max(0, int(seconds))
    hours, rem = divmod(seconds_i, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _available_cpus() -> List[int]:
    try:
        return sorted(int(v) for v in os.sched_getaffinity(0))
    except Exception:
        return list(range(os.cpu_count() or 1))


def _parse_cpu_slots(spec: str) -> List[BenchmarkSlot]:
    slots: List[BenchmarkSlot] = []
    for slot_idx, chunk in enumerate(spec.split(";")):
        chunk = chunk.strip()
        if not chunk:
            continue
        cpus = tuple(sorted({int(v.strip()) for v in chunk.split(",") if v.strip() != ""}))
        if not cpus:
            continue
        slots.append(BenchmarkSlot(slot_id=slot_idx, cpus=cpus))
    if not slots:
        raise ValueError("--cpu-slots no produjo ningún slot válido.")
    return slots


def _auto_slots(*, threads: int, reserve_cores: int) -> List[BenchmarkSlot]:
    cpus = _available_cpus()
    if reserve_cores > 0 and len(cpus) > reserve_cores:
        usable = cpus[:-reserve_cores]
    else:
        usable = cpus
    if len(usable) < threads:
        usable = cpus[:threads] if len(cpus) >= threads else cpus
    slots: List[BenchmarkSlot] = []
    for idx in range(0, len(usable), threads):
        group = tuple(usable[idx : idx + threads])
        if len(group) != threads:
            break
        slots.append(BenchmarkSlot(slot_id=len(slots), cpus=group))
    if not slots and cpus:
        slots.append(BenchmarkSlot(slot_id=0, cpus=(cpus[0],)))
    return slots


def resolve_benchmark_slots(settings: BenchmarkSettings, total_tasks: int) -> List[BenchmarkSlot]:
    if settings.cpu_slots:
        slots = _parse_cpu_slots(settings.cpu_slots)
        invalid = [slot for slot in slots if len(slot.cpus) != settings.threads]
        if invalid:
            raise ValueError(
                "Todos los slots manuales deben tener exactamente el mismo número de CPUs que --threads. "
                f"threads={settings.threads} slots_invalidos={[slot.cpus for slot in invalid]}"
            )
    else:
        slots = _auto_slots(threads=max(1, settings.threads), reserve_cores=max(0, settings.cpu_reserve_cores))
    if not slots:
        raise RuntimeError("No se pudieron resolver slots de CPU para benchmark.")
    if isinstance(settings.jobs, str) and settings.jobs == "auto":
        resolved_jobs = min(len(slots), max(1, total_tasks))
    else:
        resolved_jobs = min(len(slots), max(1, int(settings.jobs)), max(1, total_tasks))
    return slots[:resolved_jobs]


def _build_executor(*, max_workers: int, worker_max_tasks: int) -> ProcessPoolExecutor:
    kwargs = {
        "max_workers": max_workers,
        "mp_context": get_context("spawn"),
    }
    try:
        params = inspect.signature(ProcessPoolExecutor).parameters
    except (TypeError, ValueError):
        params = {}
    if "max_tasks_per_child" in params:
        kwargs["max_tasks_per_child"] = max(1, worker_max_tasks)
    return ProcessPoolExecutor(**kwargs)


def _read_resource_snapshot() -> Dict[str, Any]:
    vm = psutil.virtual_memory()
    sm = psutil.swap_memory()
    total_mb = vm.total / MB
    available_mb = vm.available / MB
    used_mb = total_mb - available_mb
    return {
        "total_mb": _safe_round(total_mb),
        "available_mb": _safe_round(available_mb),
        "used_mb": _safe_round(used_mb),
        "used_fraction": _safe_round((used_mb / total_mb) if total_mb else 0.0),
        "buffers_cache_mb": _safe_round(getattr(vm, "buffers", 0) / MB + getattr(vm, "cached", 0) / MB),
        "swap_total_mb": _safe_round(sm.total / MB),
        "swap_used_mb": _safe_round(sm.used / MB),
        "swap_used_fraction": _safe_round((sm.used / sm.total) if sm.total else 0.0),
        "loadavg_1m": _safe_round(os.getloadavg()[0]) if hasattr(os, "getloadavg") else 0.0,
    }


def _executor_process_rss_mb(executor: ProcessPoolExecutor) -> Dict[str, Any]:
    pids: List[int] = []
    rss_map: Dict[str, float] = {}
    total_rss_mb = 0.0
    processes = getattr(executor, "_processes", {}) or {}
    for pid in list(processes.keys()):
        pids.append(int(pid))
        try:
            rss_mb = psutil.Process(pid).memory_info().rss / MB
        except Exception:
            rss_mb = 0.0
        rss_mb = _safe_round(rss_mb)
        rss_map[str(pid)] = rss_mb
        total_rss_mb += rss_mb
    return {
        "worker_pids": sorted(pids),
        "worker_rss_mb": rss_map,
        "worker_total_rss_mb": _safe_round(total_rss_mb),
    }


def _dispatch_state(settings: BenchmarkSettings, active_workers: int, resolved_jobs: int, *, benchmark_id: str | None = None) -> Tuple[str, str, Dict[str, Any]]:
    res = _read_resource_snapshot()
    details = {
        "active_workers": active_workers,
        "benchmark_id": benchmark_id,
        "job_limit": int(resolved_jobs),
        "estimated_worker_ram_mb": int(settings.estimated_worker_ram_mb),
        "ram_reserve_mb": int(settings.ram_reserve_mb),
        **res,
    }
    soft_fraction_limit_mb = res["total_mb"] * settings.max_ram_fraction
    details["soft_fraction_limit_mb"] = _safe_round(soft_fraction_limit_mb)
    details["soft_fraction_exceeded"] = bool(res["used_mb"] >= soft_fraction_limit_mb)
    if active_workers >= int(resolved_jobs):
        return "PAUSED", "ACTIVE_LIMIT", details
    if res["available_mb"] < settings.ram_reserve_mb:
        return "PAUSED", "RAM_RESERVE_HIT", details
    if (res["available_mb"] - settings.ram_reserve_mb) < settings.estimated_worker_ram_mb:
        return "PAUSED", "RAM_AVAILABLE_LOW", details
    if details["soft_fraction_exceeded"]:
        return "READY", "RAM_FRACTION_WARN", details
    return "READY", "READY", details


def _configure_benchmark_worker_cpu(slot: BenchmarkSlot) -> Dict[str, Any]:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    info: Dict[str, Any] = {
        "gpu_disabled_env": True,
        "requested_cpus": list(slot.cpus),
        "requested_threads": slot.threads,
        "cpu_affinity_applied": False,
        "tf_gpu_disabled": False,
        "tf_threads": slot.threads,
    }
    try:
        if hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, set(slot.cpus))
            info["cpu_affinity_applied"] = True
    except Exception as exc:
        info["cpu_affinity_error"] = repr(exc)

    import tensorflow as tf

    try:
        tf.config.set_visible_devices([], "GPU")
        info["tf_gpu_disabled"] = True
    except Exception as exc:
        info["tf_gpu_disable_error"] = repr(exc)
    try:
        tf.config.threading.set_intra_op_parallelism_threads(slot.threads)
        tf.config.threading.set_inter_op_parallelism_threads(slot.threads)
    except Exception as exc:
        info["tf_thread_config_error"] = repr(exc)
    try:
        visible = sorted(int(v) for v in os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
        info["visible_cpus_after_pin"] = visible
    except Exception:
        pass
    return info


def _measurement_profile_config(settings: BenchmarkSettings) -> Dict[str, object]:
    return canonical_benchmark_signature({
        "feature_source": settings.feature_source,
        "feature_npy": settings.feature_npy,
        "num_videos": settings.num_videos,
        "feature_seed": settings.feature_seed,
        "distribution": settings.distribution,
        "warmup_runs": settings.warmup_runs,
        "steady_runs": settings.steady_runs,
        "threads": settings.threads,
        "hardware_target": settings.hardware_target,
        "benchmark_mode": "host_parallel",
    })


def _selected_runtime_list(requested: str) -> List[str]:
    if requested == "both":
        return ["float", "tflite"]
    return [requested]


def _load_manifest(manifest_path: str) -> Dict[str, Any]:
    raw = read_json(manifest_path, default=None)
    if raw is None:
        raise FileNotFoundError(f"No existe manifest del modelo: {manifest_path}")
    return raw


def _load_model_spec_bundle(output_root: str, model_id: str) -> Dict[str, Any]:
    from rnn_benchlib.conversion.convert import build_artifact_paths

    artifacts = build_artifact_paths(output_root, model_id)
    raw = read_json(artifacts.spec_path, default=None)
    if raw is None:
        raise FileNotFoundError(f"No existe spec del modelo: {artifacts.spec_path}")
    return raw


def _reconstruct_model_spec(spec_dict: Dict[str, Any]):
    from rnn_benchlib.config.schemas import ModelSpec

    return ModelSpec(**spec_dict)


def _reconstruct_feature_spec(feature_spec_dict: Dict[str, Any]):
    from rnn_benchlib.config.schemas import FeatureSpec

    return FeatureSpec(**feature_spec_dict)


def _reconstruct_benchmark_record(record_dict: Dict[str, Any]):
    from rnn_benchlib.config.schemas import BenchmarkRecord, ClipTiming, ExperimentMeta, NumericCheck, RuntimeSummary, VideoTiming

    experiment = ExperimentMeta(**record_dict["experiment"])
    runtime_summary = RuntimeSummary(**record_dict["runtime_summary"])
    clip_timings = [ClipTiming(**item) for item in record_dict["clip_timings"]]
    video_timing = VideoTiming(**record_dict["video_timing"])
    numeric = NumericCheck(**record_dict["numeric_check"]) if record_dict.get("numeric_check") else None
    return BenchmarkRecord(
        experiment=experiment,
        model_id=record_dict["model_id"],
        runtime_kind=record_dict["runtime_kind"],
        memory_mode=record_dict["memory_mode"],
        seq=record_dict["seq"],
        video_index=record_dict["video_index"],
        clips_per_video=record_dict["clips_per_video"],
        threads=record_dict["threads"],
        batch_size=record_dict["batch_size"],
        runtime_summary=runtime_summary,
        clip_timings=clip_timings,
        video_timing=video_timing,
        numeric_check=numeric,
        extra=record_dict.get("extra", {}),
    )


def _benchmark_model_for_runtime(*, settings: BenchmarkSettings, runtime_kind: str, model_id: str, manifest: Dict[str, Any], spec, feature_spec, batch, experiment_extra: Dict[str, object]) -> List[Dict[str, Any]]:
    from rnn_benchlib.benchmark.runners import (
        benchmark_video_batch,
        create_experiment_meta,
        load_float_model,
        resolve_encoder_tflite_output_indices_with_float_reference,
        timed_create_encoder_tflite_interpreter,
        timed_create_head_tflite_interpreter,
    )
    from rnn_benchlib.conversion.convert import build_artifact_paths

    experiment = create_experiment_meta(
        experiment_name=settings.experiment_name,
        runtime=runtime_kind,
        device_name=settings.device_name,
        notes=settings.notes,
        extra=experiment_extra,
    )

    artifacts = build_artifact_paths(settings.output_root, model_id)
    float_encoder_model = None
    float_head_model = None
    if runtime_kind == "float":
        float_encoder_model = load_float_model(artifacts.encoder_keras_dir)
        float_head_model = load_float_model(artifacts.head_keras_dir)

    tflite_encoder_interpreter = None
    tflite_head_interpreter = None
    tflite_init_ms = None
    encoder_input_map = None
    encoder_output_indices = None
    head_input_index = None
    head_output_index = None
    if runtime_kind == "tflite":
        tflite_encoder_interpreter, encoder_init_ms, encoder_input_map, encoder_output_indices = timed_create_encoder_tflite_interpreter(
            model_path=artifacts.encoder_tflite_path,
            spec=spec,
            feature_spec=feature_spec,
            num_threads=settings.threads,
        )
        float_encoder_model = load_float_model(artifacts.encoder_keras_dir)
        encoder_output_indices = resolve_encoder_tflite_output_indices_with_float_reference(
            interpreter=tflite_encoder_interpreter,
            float_encoder_model=float_encoder_model,
            spec=spec,
            feature_spec=feature_spec,
            input_name_to_index=encoder_input_map,
        )
        tflite_head_interpreter, head_init_ms, head_input_index, head_output_index = timed_create_head_tflite_interpreter(
            model_path=artifacts.head_tflite_path,
            spec=spec,
            num_threads=settings.threads,
        )
        tflite_init_ms = float(encoder_init_ms + head_init_ms)

    records = benchmark_video_batch(
        experiment=experiment,
        model_id=model_id,
        spec=spec,
        feature_spec=feature_spec,
        batch=batch,
        float_encoder_model=float_encoder_model if runtime_kind == "float" else None,
        float_head_model=float_head_model if runtime_kind == "float" else None,
        tflite_encoder_interpreter=tflite_encoder_interpreter,
        tflite_head_interpreter=tflite_head_interpreter,
        tflite_init_ms=tflite_init_ms,
        tflite_encoder_input_name_to_index=encoder_input_map,
        tflite_encoder_ordered_output_indices=encoder_output_indices,
        tflite_head_input_index=head_input_index,
        tflite_head_output_index=head_output_index,
        runtime=runtime_kind,
        warmup_runs=settings.warmup_runs,
        steady_runs=settings.steady_runs,
        threads=settings.threads,
    )
    return [record.to_dict() for record in records]


def _measurement_lock_path(output_root: str, profile_id: str, model_id: str, runtime_kind: str) -> str:
    return os.path.join(output_root, "state", "locks", "benchmarks", profile_id, f"{model_id}.{runtime_kind}.lock")


def _worker_benchmark_one(payload: Dict[str, Any]) -> Dict[str, Any]:
    started_at = time.time()
    task = payload["task"]
    settings: BenchmarkSettings = payload["settings"]
    slot: BenchmarkSlot = payload["slot"]

    cpu_info = _configure_benchmark_worker_cpu(slot)
    store = RnnStateStore(os.path.join(settings.output_root, "state", "benchlib.db"))
    profile_config = _measurement_profile_config(settings)
    benchmark_id = benchmark_id_from_signature(profile_config)
    profile_identity = _profile_config_identity(benchmark_id, profile_config)
    profile_ids = {runtime: store.make_profile_id(runtime=runtime, config_json=profile_identity) for runtime in _selected_runtime_list(settings.runtime)}

    from rnn_benchlib.benchmark.measurement_schema import benchmark_record_to_measurement, write_measurement_jsonl
    from rnn_benchlib.features.synthetic_video_features import generate_synthetic_video_batch, load_video_batch_from_npy

    manifest = _load_manifest(task.manifest_path)
    spec_bundle = _load_model_spec_bundle(settings.output_root, task.model_id)
    spec = _reconstruct_model_spec(spec_bundle["spec"])
    feature_spec = _reconstruct_feature_spec(spec_bundle["feature_spec"])

    if settings.feature_source == "synthetic":
        batch = generate_synthetic_video_batch(
            num_videos=settings.num_videos,
            feature_spec=feature_spec,
            model_spec=spec,
            seed=settings.feature_seed,
            distribution=settings.distribution,
        )
    else:
        if not settings.feature_npy:
            raise ValueError("--feature-npy es obligatorio si feature-source=npy")
        batch = load_video_batch_from_npy(path=settings.feature_npy, model_spec=spec)

    entries: List[Dict[str, Any]] = []
    benchmark_rows = 0
    runtime_env_info = {
        "gpu_disabled": cpu_info.get("tf_gpu_disabled", False),
        "tf_single_thread": settings.threads == 1,
        "cpu_affinity": list(slot.cpus),
        "slot_id": slot.slot_id,
        "slot_threads": slot.threads,
        **cpu_info,
    }
    experiment_extra_base: Dict[str, object] = {
        "runtime_env": runtime_env_info,
        "cpu_control": payload.get("cpu_control", {}),
        **profile_config,
        "execution_site": "host_parallel",
        "lot_id": settings.lot_id,
        "slot_id": slot.slot_id,
        "slot_cpus": list(slot.cpus),
        "benchmark_id": benchmark_id,
    }

    for runtime_kind in _selected_runtime_list(settings.runtime):
        profile_id = profile_ids[runtime_kind]
        result_root = model_profile_dir(settings.output_root, task.model_id, profile_id)
        result_path = os.path.join(result_root, f"{runtime_kind}.jsonl")
        measurement_path = os.path.join(result_root, f"{runtime_kind}_measurements.jsonl")

        if runtime_kind == "tflite" and not task.has_tflite:
            entries.append({
                "model_id": task.model_id,
                "runtime": runtime_kind,
                "profile_id": profile_id,
                "result_path": None,
                "measurement_path": None,
                "source": "skipped_no_tflite",
                "slot_id": slot.slot_id,
            })
            continue

        lock_path = _measurement_lock_path(settings.output_root, profile_id, task.model_id, runtime_kind)
        with exclusive_file_lock(lock_path, poll_interval_s=0.1, stale_after_s=3600.0):
            existing = store.get_measurement(model_id=task.model_id, runtime=runtime_kind, profile_id=profile_id)
            if existing is not None and existing.status == "ok" and os.path.exists(existing.result_path):
                existing_result_path = existing.result_path
                existing_measurement_path = os.path.join(os.path.dirname(existing_result_path), f"{runtime_kind}_measurements.jsonl")
                entries.append({
                    "model_id": task.model_id,
                    "runtime": runtime_kind,
                    "profile_id": profile_id,
                    "result_path": existing_result_path,
                    "measurement_path": existing_measurement_path if os.path.exists(existing_measurement_path) else None,
                    "source": "reused",
                    "slot_id": slot.slot_id,
                })
                continue

            experiment_extra = dict(experiment_extra_base)
            experiment_extra["runtime"] = runtime_kind
            records = _benchmark_model_for_runtime(
                settings=settings,
                runtime_kind=runtime_kind,
                model_id=task.model_id,
                manifest=manifest,
                spec=spec,
                feature_spec=feature_spec,
                batch=batch,
                experiment_extra=experiment_extra,
            )
            write_jsonl(result_path, records)
            graph_id = f"graph_{task.model_id}"
            uses_flex = bool(manifest.get("conversion", {}).get("uses_flex", False))
            measurement_records = [
                benchmark_record_to_measurement(
                    _reconstruct_benchmark_record(record_dict),
                    graph_id=graph_id,
                    uses_flex=uses_flex,
                    quantization_mode=manifest.get("conversion", {}).get("quantization_mode", "none"),
                )
                for record_dict in records
            ]
            write_measurement_jsonl(measurement_path, measurement_records)
            store.upsert_measurement(
                model_id=task.model_id,
                runtime=runtime_kind,
                profile_id=profile_id,
                config_json=profile_identity,
                result_path=result_path,
                status="ok",
            )
            entries.append({
                "model_id": task.model_id,
                "runtime": runtime_kind,
                "profile_id": profile_id,
                "result_path": result_path,
                "measurement_path": measurement_path,
                "source": "new",
                "slot_id": slot.slot_id,
                "rows": len(records),
            })
            benchmark_rows += len(records)

    return {
        "ok": True,
        "pid": os.getpid(),
        "position": task.position,
        "model_id": task.model_id,
        "elapsed_sec": _safe_round(time.time() - started_at),
        "slot_id": slot.slot_id,
        "slot_cpus": list(slot.cpus),
        "entries": entries,
        "benchmark_rows": benchmark_rows,
    }


def _worker_entry(payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return _worker_benchmark_one(payload)
    except Exception as exc:
        return {
            "ok": False,
            "pid": os.getpid(),
            "position": int(payload["task"].position),
            "model_id": payload["task"].model_id,
            "elapsed_sec": _safe_round(time.time() - payload.get("started_at", time.time())),
            "slot_id": int(payload["slot"].slot_id),
            "slot_cpus": list(payload["slot"].cpus),
            "error": repr(exc),
        }
    finally:
        try:
            import tensorflow as tf
            tf.keras.backend.clear_session()
        except Exception:
            pass
        gc.collect()


def _spec_payload(task: BenchmarkTask, settings: BenchmarkSettings, slot: BenchmarkSlot, cpu_control: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "task": task,
        "settings": settings,
        "slot": slot,
        "cpu_control": cpu_control,
        "started_at": time.time(),
    }


def _aggregate_runtime_records(records: List[Dict[str, Any]]) -> Dict[str, object]:
    if not records:
        return {
            "num_videos": 0,
            "num_total_clips": 0,
            "clip_e2e_wall_stats_global": {
                "count": 0,
                "mean_ms": 0.0,
                "median_ms": 0.0,
                "std_ms": 0.0,
                "min_ms": 0.0,
                "max_ms": 0.0,
                "p95_ms": 0.0,
                "p99_ms": 0.0,
            },
        }
    import numpy as np

    vals = [float(clip.get("clip_e2e_wall_ms", 0.0)) for row in records for clip in (row.get("clip_timings", []) or [])]
    arr = np.asarray(vals, dtype=np.float64) if vals else np.asarray([], dtype=np.float64)
    stats = {
        "count": int(arr.size),
        "mean_ms": float(np.mean(arr)) if arr.size else 0.0,
        "median_ms": float(np.median(arr)) if arr.size else 0.0,
        "std_ms": float(np.std(arr, ddof=0)) if arr.size else 0.0,
        "min_ms": float(np.min(arr)) if arr.size else 0.0,
        "max_ms": float(np.max(arr)) if arr.size else 0.0,
        "p95_ms": float(np.percentile(arr, 95)) if arr.size else 0.0,
        "p99_ms": float(np.percentile(arr, 99)) if arr.size else 0.0,
    }
    return {
        "num_videos": len(records),
        "num_total_clips": sum(len(row.get("clip_timings", []) or []) for row in records),
        "clip_e2e_wall_stats_global": stats,
    }


def _aggregate_from_result_paths(paths: Sequence[str]) -> Dict[str, object]:
    combined: List[Dict[str, Any]] = []
    for path in paths:
        combined.extend(read_jsonl(path) if os.path.exists(path) else [])
    return _aggregate_runtime_records(combined)


def _build_status_payload(*, settings: BenchmarkSettings, resolved_jobs: int, total_tasks: int, pending: int, active: Dict[Future, Dict[str, Any]], results: List[Dict[str, Any]], task_durations: List[float], executor: ProcessPoolExecutor, started_at: float, last_progress_at: float, dispatch_state: str, dispatch_reason: str, dispatch_details: Dict[str, Any], slots: List[BenchmarkSlot]) -> Dict[str, Any]:
    completed = len(results)
    failed = sum(1 for item in results if not item.get("ok"))
    succeeded = completed - failed
    processed_models = sum(1 for item in results if item.get("ok"))
    reused_count = sum(1 for item in results if item.get("ok") and any(entry.get("source") == "reused" for entry in item.get("entries", [])))
    benchmarked_count = sum(1 for item in results if item.get("ok") and any(entry.get("source") == "new" for entry in item.get("entries", [])))
    skipped_tflite = sum(1 for item in results if item.get("ok") and any(entry.get("source") == "skipped_no_tflite" for entry in item.get("entries", [])))
    elapsed = max(0.000001, time.time() - started_at)
    throughput_per_min = succeeded / elapsed * 60.0
    eta_sec = (pending / succeeded * elapsed) if succeeded > 0 and pending > 0 else None
    memory = {
        **dispatch_details,
        "ram_reserve_mb": int(settings.ram_reserve_mb),
        "estimated_worker_ram_mb": int(settings.estimated_worker_ram_mb),
        "max_ram_fraction": float(settings.max_ram_fraction),
    }
    workers = {
        **_executor_process_rss_mb(executor),
        "active_tasks": [
            {
                "position": meta["task"].position,
                "model_id": meta["task"].model_id,
                "submitted_at": _safe_round(meta["submitted_at"]),
                "elapsed_sec": _safe_round(time.time() - meta["submitted_at"]),
                "slot_id": meta["slot"].slot_id,
                "slot_cpus": list(meta["slot"].cpus),
            }
            for meta in sorted(active.values(), key=lambda item: item["task"].position)
        ],
        "slots": [{"slot_id": slot.slot_id, "cpus": list(slot.cpus)} for slot in slots],
    }
    avg_task_sec = (sum(task_durations) / len(task_durations)) if task_durations else 0.0
    status = {
        "schema_version": "rnn_benchlib_benchmark_runtime_v2",
        "lot_id": settings.lot_id,
        "benchmark_id": dispatch_details.get("benchmark_id"),
        "started_at": _safe_round(started_at),
        "updated_at": _safe_round(time.time()),
        "seconds_since_progress": _safe_round(time.time() - last_progress_at),
        "progress": {
            "total": total_tasks,
            "completed": completed,
            "succeeded": succeeded,
            "failed": failed,
            "processed_models": processed_models,
            "benchmarked_models": benchmarked_count,
            "reused_models": reused_count,
            "skipped_no_tflite_models": skipped_tflite,
            "pending": pending,
            "active": len(active),
            "throughput_per_min": _safe_round(throughput_per_min),
            "avg_task_sec": _safe_round(avg_task_sec),
            "eta_sec": _safe_round(eta_sec) if eta_sec is not None else None,
            "eta_human": _format_eta(eta_sec),
        },
        "dispatch": {
            "state": dispatch_state,
            "reason": dispatch_reason,
        },
        "resources": memory,
        "workers": workers,
        "scheduler": {
            "resolved_jobs": resolved_jobs,
            "ram_check_interval_sec": _safe_round(settings.ram_check_interval_sec),
            "progress_report_interval_sec": _safe_round(settings.progress_report_interval_sec),
            "worker_max_tasks": int(settings.worker_max_tasks),
            "task_timeout_sec": _safe_round(settings.task_timeout_sec),
            "threads_per_model": int(settings.threads),
        },
    }
    return status


def _console_line(status: Dict[str, Any]) -> str:
    progress = status["progress"]
    dispatch = status["dispatch"]
    resources = status["resources"]
    workers = status["workers"]
    return (
        f"[{status['lot_id']}] done={progress['completed']}/{progress['total']} "
        f"bench={progress['benchmarked_models']} reused={progress['reused_models']} failed={progress['failed']} "
        f"pending={progress['pending']} active={progress['active']} "
        f"throughput={progress['throughput_per_min']:.2f}/min eta={progress.get('eta_human') or 'n/a'} | "
        f"ram_avail={resources['available_mb'] / 1024:.1f}Gi swap_used={resources['swap_used_mb'] / 1024:.1f}Gi "
        f"load1={resources['loadavg_1m']:.1f} worker_rss={workers['worker_total_rss_mb'] / 1024:.1f}Gi | "
        f"dispatch={dispatch['state']} reason={dispatch['reason']}"
    )


def _write_benchmark_config_artifacts(output_root: str, lot_id: str, benchmark_id: str, profile_signature: Dict[str, Any], profile_ids: Dict[str, str]) -> None:
    lot_benchmark_config_dir(output_root, lot_id, benchmark_id)
    payload = {
        "schema_version": "rnn_benchlib_benchmark_config_v2",
        "lot_id": lot_id,
        "benchmark_id": benchmark_id,
        "signature": {
            "value": benchmark_id,
            "fields": profile_signature,
        },
        "profile_ids": profile_ids,
    }
    write_json(lot_benchmark_config_file(output_root, lot_id, benchmark_id), payload, indent=2)


def _write_lot_profile_summaries(output_root: str, lot_id: str, benchmark_id: str, profile_signature: Dict[str, Any], profile_ids: Dict[str, str], runtime_entries: Dict[str, List[Dict[str, Any]]], failed_models: List[str], selected_model_ids: List[str]) -> None:
    profile_summaries: Dict[str, Dict[str, Any]] = {}
    for runtime, profile_id in profile_ids.items():
        entries = runtime_entries.get(runtime, [])
        result_paths = [entry["result_path"] for entry in entries if entry.get("result_path")]
        payload = {
            "schema_version": "rnn_benchlib_lot_benchmark_profile_v2",
            "lot_id": lot_id,
            "benchmark_id": benchmark_id,
            "profile_id": profile_id,
            "profile": {
                "profile_id": profile_id,
                "runtime": runtime,
                "benchmark_id": benchmark_id,
                "signature": profile_signature,
            },
            "selection": {
                "policy": "parallel_all_models_with_required_runtime",
                "selected_model_ids": selected_model_ids,
            },
            "run_summary": {
                "lot_id": lot_id,
                "runtime": runtime,
                "processed_models": sum(1 for entry in entries if entry.get("source") in {"new", "reused"}),
                "reused_measurements": sum(1 for entry in entries if entry.get("source") == "reused"),
                "failed_models": failed_models,
                **_aggregate_from_result_paths(result_paths),
            },
            "model_results": entries,
        }
        write_json(lot_benchmark_profile_file(output_root, lot_id, benchmark_id, profile_id), payload, indent=2)
        profile_summaries[runtime] = {
            "profile_id": profile_id,
            "summary_path": lot_benchmark_profile_file(output_root, lot_id, benchmark_id, profile_id),
            "processed_models": payload["run_summary"]["processed_models"],
            "reused_measurements": payload["run_summary"]["reused_measurements"],
        }
    lot_summary = {
        "schema_version": "rnn_benchlib_lot_benchmark_v5",
        "lot_id": lot_id,
        "benchmark_id": benchmark_id,
        "signature": profile_signature,
        "profile_ids": profile_ids,
        "selection": {
            "selected_model_ids": selected_model_ids,
        },
        "failed_models": failed_models,
        "profiles": profile_summaries,
    }
    write_json(lot_benchmark_lot_summary_file(output_root, lot_id, benchmark_id), lot_summary, indent=2)


def run_benchmark(*, settings: BenchmarkSettings, tasks: List[BenchmarkTask], cpu_control: Dict[str, Any]) -> BenchmarkRunResult:
    started_at = time.time()
    profile_config = _measurement_profile_config(settings)
    benchmark_id = benchmark_id_from_signature(profile_config)
    store = RnnStateStore(os.path.join(settings.output_root, "state", "benchlib.db"))
    profile_identity = _profile_config_identity(benchmark_id, profile_config)
    profile_ids = {runtime: store.make_profile_id(runtime=runtime, config_json=profile_identity) for runtime in _selected_runtime_list(settings.runtime)}
    _write_benchmark_config_artifacts(settings.output_root, settings.lot_id, benchmark_id, profile_config, profile_ids)
    paths = _runtime_paths(settings.output_root, settings.lot_id, benchmark_id)
    write_json(paths.runtime_path, {"started_at": _safe_round(started_at), "runtime": {"jobs": settings.jobs, "estimated_worker_ram_mb": settings.estimated_worker_ram_mb, "max_ram_fraction": settings.max_ram_fraction, "ram_reserve_mb": settings.ram_reserve_mb, "worker_max_tasks": settings.worker_max_tasks, "threads": settings.threads, "cpu_slots": settings.cpu_slots, "cpu_reserve_cores": settings.cpu_reserve_cores}}, indent=2)
    _log_event(paths, "lot_benchmark_started", lot_id=settings.lot_id, benchmark_id=benchmark_id, submitted_tasks=len(tasks), runtime=settings.runtime)

    if not tasks:
        status = {
            "schema_version": "rnn_benchlib_benchmark_runtime_v2",
            "lot_id": settings.lot_id,
            "benchmark_id": benchmark_id,
            "started_at": _safe_round(started_at),
            "updated_at": _safe_round(time.time()),
            "progress": {"total": 0, "completed": 0, "succeeded": 0, "failed": 0, "processed_models": 0, "benchmarked_models": 0, "reused_models": 0, "skipped_no_tflite_models": 0, "pending": 0, "active": 0, "throughput_per_min": 0.0, "avg_task_sec": 0.0, "eta_sec": None, "eta_human": None},
            "dispatch": {"state": "READY", "reason": "NO_PENDING_TASKS"},
            "resources": _read_resource_snapshot(),
            "workers": {"worker_pids": [], "worker_rss_mb": {}, "worker_total_rss_mb": 0.0, "active_tasks": [], "slots": []},
            "scheduler": {"resolved_jobs": 1, "ram_check_interval_sec": _safe_round(settings.ram_check_interval_sec), "progress_report_interval_sec": _safe_round(settings.progress_report_interval_sec), "worker_max_tasks": int(settings.worker_max_tasks), "task_timeout_sec": _safe_round(settings.task_timeout_sec), "threads_per_model": int(settings.threads)},
        }
        _write_status(paths, status)
        return BenchmarkRunResult(lot_id=settings.lot_id, benchmark_id=benchmark_id, profile_ids=profile_ids, processed_models=0, reused_measurements=0, failed_models=[], resolved_jobs=1, slots=[], runtime_summary={"runtime_dir": paths.runtime_dir, "runtime_path": paths.runtime_path, "status_path": paths.status_path, "progress_path": paths.results_path, "errors_path": paths.errors_path})

    slots = resolve_benchmark_slots(settings, total_tasks=len(tasks))
    resolved_jobs = len(slots)
    effective_settings = BenchmarkSettings(**{**settings.__dict__, "jobs": resolved_jobs, "progress_report_interval_sec": max(1.0, settings.progress_report_interval_sec), "stall_warning_sec": max(10.0, settings.stall_warning_sec), "task_timeout_sec": max(10.0, settings.task_timeout_sec)})

    pending = list(tasks)
    active: Dict[Future, Dict[str, Any]] = {}
    free_slots: List[BenchmarkSlot] = list(slots)
    results: List[Dict[str, Any]] = []
    task_durations: List[float] = []
    last_report_at = 0.0
    last_progress_at = started_at
    last_dispatch_reason: Optional[str] = None
    last_stall_warning_at = 0.0
    timed_out_tasks: set[Tuple[int, str]] = set()

    runtime_entries: Dict[str, List[Dict[str, Any]]] = {runtime: [] for runtime in profile_ids.keys()}
    failed_models: List[str] = []
    selected_model_ids = [task.model_id for task in tasks]

    with _build_executor(max_workers=resolved_jobs, worker_max_tasks=effective_settings.worker_max_tasks) as executor:
        while pending or active:
            submitted_this_round = False
            while pending and free_slots:
                state, reason, details = _dispatch_state(effective_settings, len(active), resolved_jobs, benchmark_id=benchmark_id)
                if state != "READY":
                    break
                task = pending.pop(0)
                slot = free_slots.pop(0)
                future = executor.submit(_worker_entry, _spec_payload(task, effective_settings, slot, cpu_control))
                active[future] = {"task": task, "slot": slot, "submitted_at": time.time()}
                submitted_this_round = True
                _log_event(paths, "task_submitted", position=task.position, model_id=task.model_id, slot_id=slot.slot_id, slot_cpus=list(slot.cpus), active=len(active), pending=len(pending))

            state, reason, details = _dispatch_state(effective_settings, len(active), resolved_jobs, benchmark_id=benchmark_id)
            now = time.time()
            should_report = (now - last_report_at) >= effective_settings.progress_report_interval_sec
            if reason != last_dispatch_reason:
                _log_event(paths, "dispatch_state_changed", state=state, reason=reason, active=len(active), pending=len(pending), resources=details)
                last_dispatch_reason = reason
                should_report = True

            for meta in active.values():
                elapsed = now - meta["submitted_at"]
                key = (meta["task"].position, meta["task"].model_id)
                if elapsed >= effective_settings.task_timeout_sec and key not in timed_out_tasks:
                    _log_event(paths, "task_timeout_warning", position=meta["task"].position, model_id=meta["task"].model_id, slot_id=meta["slot"].slot_id, elapsed_sec=_safe_round(elapsed))
                    timed_out_tasks.add(key)
                    should_report = True

            if pending and (now - last_progress_at) >= effective_settings.stall_warning_sec and (now - last_stall_warning_at) >= effective_settings.stall_warning_sec:
                status = _build_status_payload(
                    settings=effective_settings,
                    resolved_jobs=resolved_jobs,
                    total_tasks=len(tasks),
                    pending=len(pending),
                    active=active,
                    results=results,
                    task_durations=task_durations,
                    executor=executor,
                    started_at=started_at,
                    last_progress_at=last_progress_at,
                    dispatch_state=state,
                    dispatch_reason=reason,
                    dispatch_details=details,
                    slots=slots,
                )
                _log_event(paths, "stall_warning", seconds_since_progress=status["seconds_since_progress"], dispatch_reason=reason, active=len(active), pending=len(pending))
                print("STALL WARNING:", _console_line(status), flush=True)
                last_stall_warning_at = now
                should_report = True

            if should_report:
                status = _build_status_payload(
                    settings=effective_settings,
                    resolved_jobs=resolved_jobs,
                    total_tasks=len(tasks),
                    pending=len(pending),
                    active=active,
                    results=results,
                    task_durations=task_durations,
                    executor=executor,
                    started_at=started_at,
                    last_progress_at=last_progress_at,
                    dispatch_state=state,
                    dispatch_reason=reason,
                    dispatch_details=details,
                    slots=slots,
                )
                _write_status(paths, status)
                _append_locked_jsonl(paths, paths.resources_path, {
                    "ts": _safe_round(now),
                    "lot_id": effective_settings.lot_id,
                    **status["resources"],
                    "pending": status["progress"]["pending"],
                    "active": status["progress"]["active"],
                    "dispatch_reason": reason,
                })
                for task_info in status["workers"]["active_tasks"]:
                    _append_locked_jsonl(paths, paths.workers_path, {"ts": _safe_round(now), "lot_id": effective_settings.lot_id, **task_info})
                print(_console_line(status), flush=True)
                last_report_at = now

            if not active:
                if pending and not submitted_this_round:
                    time.sleep(effective_settings.ram_check_interval_sec)
                    continue
                break

            done, _ = wait(active.keys(), timeout=effective_settings.ram_check_interval_sec, return_when=FIRST_COMPLETED)
            if not done:
                continue
            for future in done:
                meta = active.pop(future)
                free_slots.append(meta["slot"])
                free_slots.sort(key=lambda item: item.slot_id)
                payload = future.result()
                task_durations.append(float(payload.get("elapsed_sec", 0.0)))
                results.append(payload)
                last_progress_at = time.time()
                if payload.get("ok"):
                    for entry in payload.get("entries", []):
                        runtime_entries.setdefault(entry["runtime"], []).append(entry)
                    _append_locked_jsonl(paths, paths.results_path, {"ts": _safe_round(time.time()), **payload})
                    _log_event(paths, "task_completed", position=payload["position"], model_id=payload["model_id"], elapsed_sec=payload.get("elapsed_sec"), pid=payload.get("pid"), slot_id=payload.get("slot_id"))
                else:
                    failed_models.append(payload["model_id"])
                    _append_locked_jsonl(paths, paths.errors_path, {"ts": _safe_round(time.time()), **payload})
                    _log_event(paths, "task_failed", position=payload["position"], model_id=payload["model_id"], elapsed_sec=payload.get("elapsed_sec"), pid=payload.get("pid"), slot_id=payload.get("slot_id"), error=payload.get("error"))
                _write_lot_profile_summaries(settings.output_root, settings.lot_id, benchmark_id, profile_config, profile_ids, runtime_entries, failed_models, selected_model_ids)

    avg_task_sec = (sum(task_durations) / len(task_durations)) if task_durations else 0.0
    final_state, final_reason, final_details = ("READY", "NO_PENDING_TASKS", _dispatch_state(effective_settings, 0, resolved_jobs, benchmark_id=benchmark_id)[2])
    final_status = {
        **_build_status_payload(
            settings=effective_settings,
            resolved_jobs=resolved_jobs,
            total_tasks=len(tasks),
            pending=0,
            active={},
            results=results,
            task_durations=task_durations,
            executor=type("Dummy", (), {"_processes": {}})(),
            started_at=started_at,
            last_progress_at=last_progress_at,
            dispatch_state=final_state,
            dispatch_reason=final_reason,
            dispatch_details=final_details,
            slots=slots,
        ),
        "finished_at": _safe_round(time.time()),
    }
    _write_status(paths, final_status)
    _log_event(paths, "lot_benchmark_finished", lot_id=effective_settings.lot_id, benchmark_id=benchmark_id, processed_models=len(results), failed_models=len(failed_models))

    reused_measurements = sum(1 for entries in runtime_entries.values() for entry in entries if entry.get("source") == "reused")
    processed_models = sum(1 for item in results if item.get("ok"))
    runtime_summary = {
        "jobs": resolved_jobs,
        "threads_per_model": effective_settings.threads,
        "cpu_slots": [list(slot.cpus) for slot in slots],
        "submitted_tasks": len(tasks),
        "completed_tasks": len(results),
        "avg_task_sec": round(avg_task_sec, 6),
        "estimated_worker_ram_mb": effective_settings.estimated_worker_ram_mb,
        "max_ram_fraction": effective_settings.max_ram_fraction,
        "ram_reserve_mb": effective_settings.ram_reserve_mb,
        "worker_max_tasks": effective_settings.worker_max_tasks,
        "task_timeout_sec": effective_settings.task_timeout_sec,
        "benchmark_id": benchmark_id,
        "runtime_dir": paths.runtime_dir,
        "runtime_path": paths.runtime_path,
        "status_path": paths.status_path,
        "events_path": paths.events_path,
        "resources_path": paths.resources_path,
        "workers_path": paths.workers_path,
        "progress_path": paths.results_path,
        "errors_path": paths.errors_path,
    }
    return BenchmarkRunResult(
        lot_id=settings.lot_id,
        benchmark_id=benchmark_id,
        profile_ids=profile_ids,
        processed_models=processed_models,
        reused_measurements=reused_measurements,
        failed_models=sorted(set(failed_models)),
        resolved_jobs=resolved_jobs,
        slots=slots,
        runtime_summary=runtime_summary,
    )
