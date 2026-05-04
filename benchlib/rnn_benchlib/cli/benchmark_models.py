from __future__ import annotations

import argparse

from rnn_benchlib.benchmark.cpu_control import cpu_policy_scope
from rnn_benchlib.benchmark.scheduler import BenchmarkSettings, BenchmarkTask, run_benchmark
from rnn_benchlib.benchmark.signature import canonical_benchmark_signature
from rnn_benchlib.config.experiment import default_experiment_config
from rnn_benchlib.config.script_loader import load_rnn_config
from rnn_benchlib.storage.layout import build_root_layout
from rnn_benchlib.storage.registry import RnnStateStore
from rnn_benchlib.storage.state import stable_hash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmarkea un lote RNN usando firma lógica y runtime cargados desde un fichero de configuración centralizado. Usa siempre ./artifacts como raíz de persistencia.")
    parser.add_argument("--config", required=True, help="Fichero de configuración centralizado de la rama RNN. Se recomienda ubicarlo en ./configs y diferenciarlo en el nombre como rnn_*.json.")
    parser.add_argument("--lot-id", default=None, help="Opcional. Si se omite, se calcula a partir del bloque LOT del config.")
    parser.add_argument("--model-id", default=None, help="Opcional. Restringe el benchmark a un único modelo.")
    return parser.parse_args()


def _stable_lot_id(signature_fields: dict) -> str:
    return stable_hash({"kind": "generation", "signature_fields": signature_fields}, prefix="lot")


def main() -> None:
    args = parse_args()
    output_root = "artifacts"
    cfg = load_rnn_config(args.config)
    layout = build_root_layout(output_root)
    store = RnnStateStore(layout["db_path"])

    experiment_defaults = default_experiment_config()
    signature_fields = {
        "search_space": {k: list(v) for k, v in cfg["search_space"].__dict__.items()},
        "generation_seed": int(cfg["generation_seed"]),
        "requested_count": int(cfg["requested_count"]),
    }
    lot_id = args.lot_id or _stable_lot_id(signature_fields)

    benchmark_cfg = dict(cfg.get("benchmark") or {})
    raw_benchmark_signature = dict(benchmark_cfg.get("signature") or {})
    benchmark_signature = canonical_benchmark_signature(raw_benchmark_signature)
    runtime_cfg = dict(benchmark_cfg.get("runtime") or {})
    storage_cfg = dict(cfg.get("storage") or {})
    resource_manager = dict(cfg.get("resource_manager") or {})
    if not benchmark_signature:
        raise RuntimeError("El config debe definir BENCHMARK['signature'] para benchmarkear.")

    selected_rows = store.list_lot_models(lot_id)
    if args.model_id is not None:
        selected_rows = [row for row in selected_rows if row["model_id"] == args.model_id]
    if not selected_rows:
        raise RuntimeError(f"No hay modelos seleccionados para benchmark en el lote {lot_id}.")

    tasks = [
        BenchmarkTask(
            position=int(row["position"]),
            model_id=row["model_id"],
            manifest_path=row["manifest_path"],
            has_tflite=bool(row["has_tflite"]),
        )
        for row in selected_rows
    ]

    runtime_request = runtime_cfg.get("runtime", benchmark_cfg.get("runtime", "tflite"))
    if isinstance(runtime_request, list):
        runtime_request = "both" if set(runtime_request) == {"float", "tflite"} else runtime_request[0]
    runtime_request_str = str(runtime_request or "tflite")
    if runtime_request_str in {"float", "both"} and not bool(storage_cfg.get("persist_float_model", False)):
        raise RuntimeError(
            "El benchmark runtime=float/both requiere STORAGE.persist_float_model=true en la configuración del lote. "
            "Con la configuración actual solo está soportado runtime='tflite'."
        )

    settings = BenchmarkSettings(
        output_root=output_root,
        lot_id=lot_id,
        runtime=runtime_request_str,
        feature_source=str(benchmark_signature.get("feature_source", "synthetic")),
        feature_npy=raw_benchmark_signature.get("feature_npy"),
        num_videos=int(benchmark_signature.get("num_videos", 8)),
        feature_seed=int(benchmark_signature.get("feature_seed", int(cfg["generation_seed"]))),
        distribution=str(benchmark_signature.get("distribution", "normal")),
        warmup_runs=int(benchmark_signature.get("warmup_runs", 5)),
        steady_runs=int(benchmark_signature.get("steady_runs", 10)),
        threads=max(1, int(benchmark_signature.get("threads", 1))),
        experiment_name=str(raw_benchmark_signature.get("experiment_name", f"benchmark_{lot_id}")),
        device_name=raw_benchmark_signature.get("device_name"),
        hardware_target=benchmark_signature.get("hardware_target"),
        notes=raw_benchmark_signature.get("notes"),
        cpu_policy=str(runtime_cfg.get("cpu_policy", "none")),
        cpu_freq_khz=runtime_cfg.get("cpu_freq_khz"),
        disable_turbo=bool(runtime_cfg.get("disable_turbo", False)),
        cpu_slots=runtime_cfg.get("cpu_slots"),
        cpu_reserve_cores=max(0, int(runtime_cfg.get("cpu_reserve_cores", 1))),
        jobs=runtime_cfg.get("jobs", "auto"),
        estimated_worker_ram_mb=max(1, int(runtime_cfg.get("estimated_worker_ram_mb", resource_manager.get("benchmark_estimated_worker_ram_mb", 5500)))),
        max_ram_fraction=float(resource_manager.get("max_ram_fraction", runtime_cfg.get("max_ram_fraction", 0.65))),
        ram_reserve_mb=max(0, int(resource_manager.get("ram_reserve_mb", runtime_cfg.get("ram_reserve_mb", 32768)))),
        ram_check_interval_sec=max(0.1, float(runtime_cfg.get("ram_check_interval_sec", 1.0))),
        worker_max_tasks=max(1, int(runtime_cfg.get("worker_max_tasks", 4))),
        progress_report_interval_sec=max(0.5, float(runtime_cfg.get("progress_report_interval_sec", 1.0))),
        stall_warning_sec=max(5.0, float(runtime_cfg.get("stall_warning_sec", 60.0))),
        task_timeout_sec=max(10.0, float(runtime_cfg.get("task_timeout_sec", 1800.0))),
    )

    print("\n=== Benchmark paralelo de lote ===")
    print(f"lot_id     : {lot_id}")
    print(f"models     : {len(tasks)}")
    print(f"runtime    : {settings.runtime}")
    print(f"threads    : {settings.threads}")
    print(f"jobs       : {settings.jobs}")

    with cpu_policy_scope(policy=settings.cpu_policy, freq_khz=settings.cpu_freq_khz, disable_turbo=settings.disable_turbo) as cpu_result:
        run_result = run_benchmark(settings=settings, tasks=tasks, cpu_control=cpu_result.to_dict())

    print("\n=== Resumen benchmark ===")
    print(f"benchmark_id        : {run_result.benchmark_id}")
    print(f"processed_models    : {run_result.processed_models}")
    print(f"reused_measurements : {run_result.reused_measurements}")
    print(f"failed_models       : {len(run_result.failed_models)}")
    print(f"resolved_jobs       : {run_result.resolved_jobs}")
    print(f"latest_logs         : {run_result.runtime_summary.get('runtime_dir')}")


if __name__ == "__main__":
    main()
