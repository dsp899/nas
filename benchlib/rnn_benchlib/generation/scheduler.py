from __future__ import annotations

import gc
import inspect
import math
import os
import shutil
import time
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
from multiprocessing import get_context
from typing import Any, Callable, Dict, List, Optional, Tuple

import psutil

from rnn_benchlib.config.schemas import FeatureSpec, ModelSpec
from rnn_benchlib.storage.jsonl import append_jsonl, write_json
from rnn_benchlib.storage.layout import (
    lot_generation_errors_file,
    lot_generation_logs_dir,
    lot_generation_progress_file,
    lot_generation_status_file,
)
from rnn_benchlib.storage.locks import exclusive_file_lock


MB = 1024 * 1024


def _disable_tensorflow_gpu_for_generation() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


@dataclass(frozen=True)
class GenerationTask:
    position: int
    spec: ModelSpec
    model_id: str
    model_key: str


@dataclass(frozen=True)
class GenerationSettings:
    output_root: str
    lot_id: str
    seed: int
    feature_spec: FeatureSpec
    overwrite_existing_artifacts: bool
    jobs: int | str
    estimated_worker_ram_mb: int
    max_ram_fraction: float
    ram_reserve_mb: int
    ram_check_interval_sec: float
    worker_max_tasks: int
    progress_report_interval_sec: float = 1.0
    stall_warning_sec: float = 60.0


@dataclass(frozen=True)
class GenerationRunResult:
    covered_count: int
    new_count: int
    reused_count: int
    failed_count: int
    resolved_jobs: int
    runtime_summary: Dict[str, Any]


@dataclass(frozen=True)
class RuntimePaths:
    runtime_dir: str
    runtime_path: str
    status_path: str
    progress_path: str
    errors_path: str
    lock_path: str


ResultCallback = Callable[[Dict[str, Any]], None]


def _runtime_paths(output_root: str, lot_id: str) -> RuntimePaths:
    runtime_dir = lot_generation_logs_dir(output_root, lot_id)
    if os.path.isdir(runtime_dir):
        shutil.rmtree(runtime_dir)
    os.makedirs(runtime_dir, exist_ok=True)
    return RuntimePaths(
        runtime_dir=runtime_dir,
        runtime_path=os.path.join(runtime_dir, "runtime.json"),
        status_path=lot_generation_status_file(output_root, lot_id),
        progress_path=lot_generation_progress_file(output_root, lot_id),
        errors_path=lot_generation_errors_file(output_root, lot_id),
        lock_path=os.path.join(runtime_dir, ".log.lock"),
    )


def _manifest_path(output_root: str, model_id: str) -> str:
    return os.path.join(output_root, "models", model_id, "meta", "manifest.json")


def _read_manifest_summary(output_root: str, model_id: str) -> Optional[Dict[str, Any]]:
    from rnn_benchlib.storage.jsonl import read_json

    manifest = read_json(_manifest_path(output_root, model_id), default=None)
    if not isinstance(manifest, dict):
        return None
    if manifest.get("model_id") != model_id:
        return None
    conversion = manifest.get("conversion", {}) if isinstance(manifest.get("conversion"), dict) else {}
    components_present = conversion.get("components_present", {}) if isinstance(conversion.get("components_present"), dict) else {}
    return {
        "model_id": model_id,
        "status": conversion.get("status", "unknown"),
        "conversion_status": conversion.get("status", "unknown"),
        "conversion_mode": conversion.get("conversion_mode", "unknown"),
        "uses_flex": bool(conversion.get("uses_flex", False)),
        "has_tflite": bool(components_present.get("encoder_tflite") and components_present.get("head_tflite")),
        "manifest_path": _manifest_path(output_root, model_id),
        "model_dir": os.path.join(output_root, "models", model_id),
    }


def _configure_worker_tensorflow_threads() -> None:
    import tensorflow as tf

    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass
    try:
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except Exception:
        pass


def _worker_generate_one(output_root: str, seed: int, spec: ModelSpec, feature_spec: FeatureSpec, model_id: str, overwrite_existing_artifacts: bool) -> Dict[str, Any]:
    _disable_tensorflow_gpu_for_generation()
    _configure_worker_tensorflow_threads()

    from rnn_benchlib.conversion.convert import build_model_record
    from rnn_benchlib.storage.layout import model_dir
    from rnn_benchlib.storage.locks import exclusive_file_lock

    lock_path = os.path.join(output_root, "state", "locks", "models", f"{model_id}.lock")
    with exclusive_file_lock(lock_path):
        manifest_summary = _read_manifest_summary(output_root, model_id)
        if manifest_summary is not None:
            return {"status": "reused", **manifest_summary}

        force_overwrite = overwrite_existing_artifacts or os.path.isdir(model_dir(output_root, model_id))
        record = build_model_record(
            output_root=output_root,
            seed=seed,
            spec=spec,
            feature_spec=feature_spec,
            model_id=model_id,
            overwrite=force_overwrite,
        )
        has_tflite = bool(record.conversion.encoder_tflite_path and record.conversion.head_tflite_path)
        return {
            "status": "created",
            "model_id": record.model_id,
            "conversion_status": record.conversion.status,
            "conversion_mode": record.conversion.conversion_mode,
            "uses_flex": bool(record.conversion.uses_flex),
            "has_tflite": has_tflite,
            "manifest_path": record.artifacts.manifest_path,
            "model_dir": record.artifacts.model_dir,
        }


def _worker_entry(payload: Dict[str, Any]) -> Dict[str, Any]:
    started_at = time.time()
    result: Optional[Dict[str, Any]] = None
    try:
        result = _worker_generate_one(
            output_root=payload["output_root"],
            seed=int(payload["seed"]),
            spec=payload["spec"],
            feature_spec=payload["feature_spec"],
            model_id=payload["model_id"],
            overwrite_existing_artifacts=bool(payload["overwrite_existing_artifacts"]),
        )
        return {
            "ok": True,
            "pid": os.getpid(),
            "position": int(payload["position"]),
            "model_id": payload["model_id"],
            "elapsed_sec": round(time.time() - started_at, 6),
            "result": result,
        }
    except Exception as exc:
        return {
            "ok": False,
            "pid": os.getpid(),
            "position": int(payload["position"]),
            "model_id": payload["model_id"],
            "elapsed_sec": round(time.time() - started_at, 6),
            "error": repr(exc),
        }
    finally:
        try:
            result = None
            import tensorflow as tf
            tf.keras.backend.clear_session()
        except Exception:
            pass
        gc.collect()
        gc.collect()


def _resolve_jobs(requested_jobs: int | str, *, estimated_worker_ram_mb: int, max_ram_fraction: float, ram_reserve_mb: int, num_tasks: int) -> int:
    cpu_total = os.cpu_count() or 1
    cpu_physical = psutil.cpu_count(logical=False) or cpu_total
    # Para cargas pesadas de TensorFlow/TFLite suele ser mejor limitar AUTO a cores físicos.
    cpu_cap = max(1, cpu_physical - 1)
    if isinstance(requested_jobs, str) and requested_jobs == "auto":
        vm = psutil.virtual_memory()
        available_mb = vm.available / MB
        hard_available_budget_mb = max(0.0, available_mb - ram_reserve_mb)
        by_available = max(1, int(hard_available_budget_mb // max(1, estimated_worker_ram_mb))) if hard_available_budget_mb >= estimated_worker_ram_mb else 1
        total_soft_budget_mb = max(0.0, vm.total / MB * max_ram_fraction)
        by_fraction = max(1, int(total_soft_budget_mb // max(1, estimated_worker_ram_mb))) if total_soft_budget_mb >= estimated_worker_ram_mb else 1
        resolved = min(cpu_cap, by_available, by_fraction)
    else:
        resolved = int(requested_jobs)
    return max(1, min(resolved, max(1, num_tasks)))


def _supports_max_tasks_per_child() -> bool:
    try:
        params = inspect.signature(ProcessPoolExecutor).parameters
    except (TypeError, ValueError):
        params = {}
    return "max_tasks_per_child" in params


def _build_executor(*, max_workers: int, worker_max_tasks: int) -> ProcessPoolExecutor:
    kwargs = {"max_workers": max_workers, "mp_context": get_context("spawn")}
    if _supports_max_tasks_per_child():
        kwargs["max_tasks_per_child"] = max(1, worker_max_tasks)
    return ProcessPoolExecutor(**kwargs)


def _shutdown_executor(executor: Optional[ProcessPoolExecutor]) -> None:
    if executor is None:
        return
    try:
        executor.shutdown(wait=True)
    except Exception:
        pass


def _append_scheduler_event(paths: RuntimePaths, phase: str, reason: str, extra: Optional[Dict[str, Any]] = None) -> None:
    record = {
        "ts": _safe_round(time.time()),
        "phase": phase,
        "reason": reason,
    }
    if extra:
        record.update(extra)
    _append_locked_jsonl(paths, paths.progress_path, record)


def _spec_payload(task: GenerationTask, settings: GenerationSettings) -> Dict[str, Any]:
    return {
        "output_root": settings.output_root,
        "seed": settings.seed,
        "feature_spec": settings.feature_spec,
        "spec": task.spec,
        "position": task.position,
        "model_id": task.model_id,
        "overwrite_existing_artifacts": settings.overwrite_existing_artifacts,
    }


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


def _read_memory_snapshot() -> Dict[str, Any]:
    vm = psutil.virtual_memory()
    sm = psutil.swap_memory()
    total_mb = vm.total / MB
    available_mb = vm.available / MB
    used_mb = total_mb - available_mb
    return {
        "total_mb": _safe_round(total_mb),
        "available_mb": _safe_round(available_mb),
        "used_mb": _safe_round(used_mb),
        "swap_used_mb": _safe_round(sm.used / MB),
    }


def _dispatch_state(settings: GenerationSettings, active_workers: int) -> Tuple[str, str, Dict[str, Any]]:
    mem = _read_memory_snapshot()
    details = {
        "active_workers": active_workers,
        "job_limit": int(settings.jobs),
        "estimated_worker_ram_mb": int(settings.estimated_worker_ram_mb),
        "ram_reserve_mb": int(settings.ram_reserve_mb),
        **mem,
    }
    soft_fraction_limit_mb = mem["total_mb"] * settings.max_ram_fraction
    if active_workers >= int(settings.jobs):
        return "PAUSED", "ACTIVE_LIMIT", details
    if mem["available_mb"] < settings.ram_reserve_mb:
        return "PAUSED", "RAM_RESERVE_HIT", details
    if (mem["available_mb"] - settings.ram_reserve_mb) < settings.estimated_worker_ram_mb:
        return "PAUSED", "RAM_AVAILABLE_LOW", details
    if mem["used_mb"] >= soft_fraction_limit_mb:
        return "READY", "RAM_FRACTION_WARN", details
    return "READY", "READY", details


def _executor_process_rss_mb(executor: ProcessPoolExecutor) -> Dict[str, Any]:
    total_rss_mb = 0.0
    processes = getattr(executor, "_processes", {}) or {}
    for pid in list(processes.keys()):
        try:
            total_rss_mb += psutil.Process(pid).memory_info().rss / MB
        except Exception:
            pass
    return {"worker_total_rss_mb": _safe_round(total_rss_mb)}


def _append_locked_jsonl(paths: RuntimePaths, path: str, record: Dict[str, Any]) -> None:
    with exclusive_file_lock(paths.lock_path, poll_interval_s=0.02, stale_after_s=30.0):
        append_jsonl(path, record)


def _write_status(paths: RuntimePaths, payload: Dict[str, Any]) -> None:
    write_json(paths.status_path, payload, indent=2)


def _console_line(status: Dict[str, Any]) -> str:
    return (
        f"[{status['lot_id']}] done={status['done']}/{status['total']} "
        f"created={status['created']} reused={status['reused']} failed={status['failed']} "
        f"pending={status['pending']} active={status['active']} throughput={status['throughput_models_per_min']:.2f}/min "
        f"eta={status.get('eta_human') or 'n/a'} | "
        f"ram_avail={status['memory']['available_mb'] / 1024:.1f}Gi reserve={status['runtime']['ram_reserve_mb'] / 1024:.1f}Gi "
        f"worker_rss={status['memory']['worker_total_rss_mb'] / 1024:.1f}Gi | "
        f"dispatch={status['dispatch_state']} reason={status['dispatch_reason']}"
    )


def _build_status_payload(*, settings: GenerationSettings, total_tasks: int, pending: int, active: Dict[Future, Dict[str, Any]], results: List[Dict[str, Any]], task_durations: List[float], executor: ProcessPoolExecutor, started_at: float, last_progress_at: float, dispatch_state: str, dispatch_reason: str, dispatch_details: Dict[str, Any]) -> Dict[str, Any]:
    completed = len(results)
    failed = sum(1 for item in results if not item.get("ok"))
    created = sum(1 for item in results if item.get("ok") and item.get("result", {}).get("status") == "created")
    reused = sum(1 for item in results if item.get("ok") and item.get("result", {}).get("status") == "reused")
    elapsed = max(0.000001, time.time() - started_at)
    done_ok = created + reused
    throughput_per_min = done_ok / elapsed * 60.0
    eta_sec = (pending / done_ok * elapsed) if done_ok > 0 and pending > 0 else None
    mem = {**dispatch_details, **_executor_process_rss_mb(executor)}
    peak_task = 0.0
    if results:
        peak_task = max(float(item.get("elapsed_sec", 0.0)) for item in results)  # placeholder overwritten in caller summary only if wanted
    return {
        "lot_id": settings.lot_id,
        "state": "running" if pending or active else "completed",
        "total": total_tasks,
        "done": completed,
        "created": created,
        "reused": reused,
        "failed": failed,
        "pending": pending,
        "active": len(active),
        "duration_s": _safe_round(elapsed),
        "throughput_models_per_min": _safe_round(throughput_per_min),
        "eta_sec": _safe_round(eta_sec) if eta_sec is not None else None,
        "eta_human": _format_eta(eta_sec),
        "last_progress_at": _safe_round(last_progress_at),
        "updated_at": _safe_round(time.time()),
        "dispatch_state": dispatch_state,
        "dispatch_reason": dispatch_reason,
        "runtime": {
            "jobs": int(settings.jobs),
            "estimated_worker_ram_mb": int(settings.estimated_worker_ram_mb),
            "max_ram_fraction": float(settings.max_ram_fraction),
            "ram_reserve_mb": int(settings.ram_reserve_mb),
            "worker_max_tasks": int(settings.worker_max_tasks),
            "ram_check_interval_sec": _safe_round(settings.ram_check_interval_sec),
        },
        "memory": mem,
    }


def run_generation(*, settings: GenerationSettings, tasks: List[GenerationTask], on_result: Optional[ResultCallback] = None) -> GenerationRunResult:
    paths = _runtime_paths(settings.output_root, settings.lot_id)
    started_at = time.time()
    if not tasks:
        runtime_summary = {
            "jobs": 1,
            "runtime_dir": paths.runtime_dir,
            "runtime_path": paths.runtime_path,
            "status_path": paths.status_path,
            "progress_path": paths.progress_path,
            "errors_path": paths.errors_path,
        }
        _write_status(paths, {
            "lot_id": settings.lot_id,
            "state": "completed",
            "total": 0,
            "done": 0,
            "created": 0,
            "reused": 0,
            "failed": 0,
            "pending": 0,
            "active": 0,
            "duration_s": 0.0,
            "throughput_models_per_min": 0.0,
            "eta_sec": None,
            "eta_human": None,
            "last_progress_at": _safe_round(started_at),
            "updated_at": _safe_round(time.time()),
            "dispatch_state": "READY",
            "dispatch_reason": "NO_PENDING_TASKS",
            "runtime": runtime_summary,
            "memory": {**_read_memory_snapshot(), "worker_total_rss_mb": 0.0},
        })
        return GenerationRunResult(0, 0, 0, 0, 1, runtime_summary)

    resolved_jobs = _resolve_jobs(
        settings.jobs,
        estimated_worker_ram_mb=settings.estimated_worker_ram_mb,
        max_ram_fraction=settings.max_ram_fraction,
        ram_reserve_mb=settings.ram_reserve_mb,
        num_tasks=len(tasks),
    )
    effective_settings = GenerationSettings(**{**settings.__dict__, "jobs": resolved_jobs})
    native_max_tasks = _supports_max_tasks_per_child()
    manual_recycle_interval = None
    if not native_max_tasks and effective_settings.worker_max_tasks > 0:
        manual_recycle_interval = max(1, resolved_jobs * max(1, int(effective_settings.worker_max_tasks)))

    write_json(paths.runtime_path, {
        "started_at": _safe_round(started_at),
        "runtime": {
            "jobs": resolved_jobs,
            "estimated_worker_ram_mb": int(effective_settings.estimated_worker_ram_mb),
            "max_ram_fraction": float(effective_settings.max_ram_fraction),
            "ram_reserve_mb": int(effective_settings.ram_reserve_mb),
            "worker_max_tasks": int(effective_settings.worker_max_tasks),
            "native_max_tasks_per_child": bool(native_max_tasks),
            "manual_recycle_interval_tasks": int(manual_recycle_interval or 0),
            "intra_op_threads": 1,
            "inter_op_threads": 1,
        },
    }, indent=2)

    pending = list(tasks)
    active: Dict[Future, Dict[str, Any]] = {}
    results: List[Dict[str, Any]] = []
    task_durations: List[float] = []
    last_report_at = 0.0
    last_progress_at = started_at
    last_dispatch_reason: Optional[str] = None
    completed_since_recycle = 0
    idle_ram_pause_since: Optional[float] = None
    recycle_due_to_idle_after_sec = max(10.0, min(float(effective_settings.stall_warning_sec), float(effective_settings.ram_check_interval_sec) * 8.0))

    executor: Optional[ProcessPoolExecutor] = _build_executor(max_workers=resolved_jobs, worker_max_tasks=effective_settings.worker_max_tasks)
    _append_scheduler_event(paths, "scheduler", "POOL_STARTED", {
        "resolved_jobs": resolved_jobs,
        "native_max_tasks_per_child": bool(native_max_tasks),
        "manual_recycle_interval_tasks": int(manual_recycle_interval or 0),
    })
    try:
        while pending or active:
            submitted_this_round = False
            while pending:
                state, reason, details = _dispatch_state(effective_settings, len(active))
                if state != "READY":
                    break
                task = pending.pop(0)
                try:
                    future = executor.submit(_worker_entry, _spec_payload(task, effective_settings))
                except Exception as exc:
                    pending.insert(0, task)
                    _append_scheduler_event(paths, "scheduler", "POOL_SUBMIT_FAILED", {"error": repr(exc)})
                    _shutdown_executor(executor)
                    executor = _build_executor(max_workers=resolved_jobs, worker_max_tasks=effective_settings.worker_max_tasks)
                    _append_scheduler_event(paths, "scheduler", "POOL_RESTARTED", {"reason": "submit_failed"})
                    time.sleep(min(1.0, effective_settings.ram_check_interval_sec))
                    break
                active[future] = {"task": task, "submitted_at": time.time()}
                submitted_this_round = True

            state, reason, details = _dispatch_state(effective_settings, len(active))
            now = time.time()

            recycle_due_to_tasks = bool(
                manual_recycle_interval
                and not active
                and pending
                and completed_since_recycle >= manual_recycle_interval
            )
            if pending and not active and reason == "RAM_RESERVE_HIT":
                if idle_ram_pause_since is None:
                    idle_ram_pause_since = now
                recycle_due_to_idle = (now - idle_ram_pause_since) >= recycle_due_to_idle_after_sec
            else:
                idle_ram_pause_since = None
                recycle_due_to_idle = False

            if recycle_due_to_tasks or recycle_due_to_idle:
                recycle_reason = "task_budget" if recycle_due_to_tasks else "idle_ram_retention"
                _append_scheduler_event(paths, "scheduler", "POOL_RECYCLE", {
                    "reason": recycle_reason,
                    "completed_since_recycle": completed_since_recycle,
                    "pending": len(pending),
                    "available_mb": details.get("available_mb"),
                    "worker_total_rss_mb": _executor_process_rss_mb(executor).get("worker_total_rss_mb", 0.0),
                })
                _shutdown_executor(executor)
                executor = _build_executor(max_workers=resolved_jobs, worker_max_tasks=effective_settings.worker_max_tasks)
                completed_since_recycle = 0
                idle_ram_pause_since = None
                last_dispatch_reason = None
                time.sleep(min(1.0, effective_settings.ram_check_interval_sec))
                continue

            should_report = (now - last_report_at) >= effective_settings.progress_report_interval_sec or reason != last_dispatch_reason
            last_dispatch_reason = reason
            if should_report:
                status = _build_status_payload(
                    settings=effective_settings,
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
                )
                _write_status(paths, status)
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
                task = meta["task"]
                try:
                    payload = future.result()
                except Exception as exc:
                    payload = {
                        "ok": False,
                        "pid": None,
                        "position": int(task.position),
                        "model_id": task.model_id,
                        "elapsed_sec": 0.0,
                        "error": repr(exc),
                    }
                task_durations.append(float(payload.get("elapsed_sec", 0.0)))
                payload["task"] = task
                results.append(payload)
                completed_since_recycle += 1
                last_progress_at = time.time()
                if payload.get("ok"):
                    row = {
                        "ts": _safe_round(time.time()),
                        "position": task.position,
                        "model_id": task.model_id,
                        "status": payload["result"].get("status"),
                        "elapsed_sec": payload.get("elapsed_sec"),
                        "pid": payload.get("pid"),
                    }
                    _append_locked_jsonl(paths, paths.progress_path, row)
                else:
                    row = {
                        "ts": _safe_round(time.time()),
                        "position": task.position,
                        "model_id": task.model_id,
                        "phase": "worker_task",
                        "error": payload.get("error"),
                        "elapsed_sec": payload.get("elapsed_sec"),
                        "pid": payload.get("pid"),
                    }
                    _append_locked_jsonl(paths, paths.errors_path, row)
                if on_result is not None:
                    on_result(payload)
    finally:
        _shutdown_executor(executor)

    final_state, final_reason, final_details = _dispatch_state(effective_settings, 0)
    final_status = _build_status_payload(
        settings=effective_settings,
        total_tasks=len(tasks),
        pending=0,
        active={},
        results=results,
        task_durations=task_durations,
        executor=type("Dummy", (), {"_processes": {}})(),
        started_at=started_at,
        last_progress_at=last_progress_at,
        dispatch_state="READY",
        dispatch_reason="NO_PENDING_TASKS",
        dispatch_details=final_details,
    )
    final_status["state"] = "failed" if any(not item.get("ok") for item in results) else "completed"
    _write_status(paths, final_status)

    created_count = sum(1 for item in results if item.get("ok") and item.get("result", {}).get("status") == "created")
    reused_count = sum(1 for item in results if item.get("ok") and item.get("result", {}).get("status") == "reused")
    failed_count = sum(1 for item in results if not item.get("ok"))
    runtime_summary = {
        "jobs": resolved_jobs,
        "runtime_dir": paths.runtime_dir,
        "runtime_path": paths.runtime_path,
        "status_path": paths.status_path,
        "progress_path": paths.progress_path,
        "errors_path": paths.errors_path,
        "avg_task_sec": _safe_round(sum(task_durations) / len(task_durations)) if task_durations else 0.0,
        "native_max_tasks_per_child": bool(native_max_tasks),
        "manual_recycle_interval_tasks": int(manual_recycle_interval or 0),
    }
    return GenerationRunResult(
        covered_count=created_count + reused_count,
        new_count=created_count,
        reused_count=reused_count,
        failed_count=failed_count,
        resolved_jobs=resolved_jobs,
        runtime_summary=runtime_summary,
    )


    resolved_jobs = _resolve_jobs(
        settings.jobs,
        estimated_worker_ram_mb=settings.estimated_worker_ram_mb,
        max_ram_fraction=settings.max_ram_fraction,
        ram_reserve_mb=settings.ram_reserve_mb,
        num_tasks=len(tasks),
    )
    effective_settings = GenerationSettings(**{**settings.__dict__, "jobs": resolved_jobs})
    write_json(paths.runtime_path, {
        "started_at": _safe_round(started_at),
        "runtime": {
            "jobs": resolved_jobs,
            "estimated_worker_ram_mb": int(effective_settings.estimated_worker_ram_mb),
            "max_ram_fraction": float(effective_settings.max_ram_fraction),
            "ram_reserve_mb": int(effective_settings.ram_reserve_mb),
            "worker_max_tasks": int(effective_settings.worker_max_tasks),
            "intra_op_threads": 1,
            "inter_op_threads": 1,
        },
    }, indent=2)

    pending = list(tasks)
    active: Dict[Future, Dict[str, Any]] = {}
    results: List[Dict[str, Any]] = []
    task_durations: List[float] = []
    last_report_at = 0.0
    last_progress_at = started_at
    last_dispatch_reason: Optional[str] = None

    with _build_executor(max_workers=resolved_jobs, worker_max_tasks=effective_settings.worker_max_tasks) as executor:
        while pending or active:
            submitted_this_round = False
            while pending:
                state, reason, details = _dispatch_state(effective_settings, len(active))
                if state != "READY":
                    break
                task = pending.pop(0)
                future = executor.submit(_worker_entry, _spec_payload(task, effective_settings))
                active[future] = {"task": task, "submitted_at": time.time()}
                submitted_this_round = True

            state, reason, details = _dispatch_state(effective_settings, len(active))
            now = time.time()
            should_report = (now - last_report_at) >= effective_settings.progress_report_interval_sec or reason != last_dispatch_reason
            last_dispatch_reason = reason
            if should_report:
                status = _build_status_payload(
                    settings=effective_settings,
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
                )
                _write_status(paths, status)
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
                task = meta["task"]
                payload = future.result()
                task_durations.append(float(payload.get("elapsed_sec", 0.0)))
                payload["task"] = task
                results.append(payload)
                last_progress_at = time.time()
                if payload.get("ok"):
                    row = {
                        "ts": _safe_round(time.time()),
                        "position": task.position,
                        "model_id": task.model_id,
                        "status": payload["result"].get("status"),
                        "elapsed_sec": payload.get("elapsed_sec"),
                    }
                    _append_locked_jsonl(paths, paths.progress_path, row)
                else:
                    row = {
                        "ts": _safe_round(time.time()),
                        "position": task.position,
                        "model_id": task.model_id,
                        "phase": "worker_task",
                        "error": payload.get("error"),
                        "elapsed_sec": payload.get("elapsed_sec"),
                    }
                    _append_locked_jsonl(paths, paths.errors_path, row)
                if on_result is not None:
                    on_result(payload)

    final_state, final_reason, final_details = _dispatch_state(effective_settings, 0)
    final_status = _build_status_payload(
        settings=effective_settings,
        total_tasks=len(tasks),
        pending=0,
        active={},
        results=results,
        task_durations=task_durations,
        executor=type("Dummy", (), {"_processes": {}})(),
        started_at=started_at,
        last_progress_at=last_progress_at,
        dispatch_state="READY",
        dispatch_reason="NO_PENDING_TASKS",
        dispatch_details=final_details,
    )
    final_status["state"] = "failed" if any(not item.get("ok") for item in results) else "completed"
    _write_status(paths, final_status)

    created_count = sum(1 for item in results if item.get("ok") and item.get("result", {}).get("status") == "created")
    reused_count = sum(1 for item in results if item.get("ok") and item.get("result", {}).get("status") == "reused")
    failed_count = sum(1 for item in results if not item.get("ok"))
    runtime_summary = {
        "jobs": resolved_jobs,
        "runtime_dir": paths.runtime_dir,
        "runtime_path": paths.runtime_path,
        "status_path": paths.status_path,
        "progress_path": paths.progress_path,
        "errors_path": paths.errors_path,
        "avg_task_sec": _safe_round(sum(task_durations) / len(task_durations)) if task_durations else 0.0,
    }
    return GenerationRunResult(
        covered_count=created_count + reused_count,
        new_count=created_count,
        reused_count=reused_count,
        failed_count=failed_count,
        resolved_jobs=resolved_jobs,
        runtime_summary=runtime_summary,
    )
