from __future__ import annotations

import os
from typing import Dict


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def build_root_layout(output_root: str) -> Dict[str, str]:
    return {
        "output_root": ensure_dir(output_root),
        "models_root": ensure_dir(os.path.join(output_root, "models")),
        "lots_root": ensure_dir(os.path.join(output_root, "lots")),
        "state_root": ensure_dir(os.path.join(output_root, "state")),
        "db_path": os.path.join(output_root, "state", "benchlib.db"),
    }


def model_dir(output_root: str, model_id: str) -> str:
    return os.path.join(output_root, "models", model_id)


def lot_dir(output_root: str, lot_id: str) -> str:
    return os.path.join(output_root, "lots", lot_id)


def lot_json_path(output_root: str, lot_id: str) -> str:
    return os.path.join(lot_dir(output_root, lot_id), "lot.json")


def lot_members_path(output_root: str, lot_id: str) -> str:
    return os.path.join(lot_dir(output_root, lot_id), "members.jsonl")


# Generation

def lot_generation_dir(output_root: str, lot_id: str) -> str:
    return ensure_dir(os.path.join(lot_dir(output_root, lot_id), "generation"))


def lot_generation_config_file(output_root: str, lot_id: str) -> str:
    return os.path.join(lot_generation_dir(output_root, lot_id), "config.json")


def lot_generation_summary_file(output_root: str, lot_id: str) -> str:
    return os.path.join(lot_generation_dir(output_root, lot_id), "summary.json")


def lot_generation_logs_dir(output_root: str, lot_id: str) -> str:
    return ensure_dir(os.path.join(lot_generation_dir(output_root, lot_id), "logs", "latest"))


def lot_generation_runtime_file(output_root: str, lot_id: str) -> str:
    return os.path.join(lot_generation_logs_dir(output_root, lot_id), "runtime.json")


def lot_generation_status_file(output_root: str, lot_id: str) -> str:
    return os.path.join(lot_generation_logs_dir(output_root, lot_id), "status.json")


def lot_generation_progress_file(output_root: str, lot_id: str) -> str:
    return os.path.join(lot_generation_logs_dir(output_root, lot_id), "progress.jsonl")


def lot_generation_errors_file(output_root: str, lot_id: str) -> str:
    return os.path.join(lot_generation_logs_dir(output_root, lot_id), "errors.jsonl")


# Benchmark

def lot_benchmarks_dir(output_root: str, lot_id: str) -> str:
    return ensure_dir(os.path.join(lot_dir(output_root, lot_id), "benchmarks"))


def lot_benchmark_dir(output_root: str, lot_id: str, benchmark_id: str) -> str:
    return ensure_dir(os.path.join(lot_benchmarks_dir(output_root, lot_id), benchmark_id))


def lot_benchmark_config_file(output_root: str, lot_id: str, benchmark_id: str) -> str:
    return os.path.join(lot_benchmark_dir(output_root, lot_id, benchmark_id), "config.json")


def lot_benchmark_summary_file(output_root: str, lot_id: str, benchmark_id: str) -> str:
    return os.path.join(lot_benchmark_dir(output_root, lot_id, benchmark_id), "summary.json")


def lot_benchmark_profiles_dir(output_root: str, lot_id: str, benchmark_id: str) -> str:
    return ensure_dir(os.path.join(lot_benchmark_dir(output_root, lot_id, benchmark_id), "profiles"))


def lot_benchmark_profile_file(output_root: str, lot_id: str, benchmark_id: str, profile_id: str) -> str:
    return os.path.join(lot_benchmark_profiles_dir(output_root, lot_id, benchmark_id), f"{profile_id}.json")


def lot_benchmark_logs_dir(output_root: str, lot_id: str, benchmark_id: str) -> str:
    return ensure_dir(os.path.join(lot_benchmark_dir(output_root, lot_id, benchmark_id), "logs", "latest"))


def lot_benchmark_runtime_file(output_root: str, lot_id: str, benchmark_id: str) -> str:
    return os.path.join(lot_benchmark_logs_dir(output_root, lot_id, benchmark_id), "runtime.json")


def lot_benchmark_status_file(output_root: str, lot_id: str, benchmark_id: str) -> str:
    return os.path.join(lot_benchmark_logs_dir(output_root, lot_id, benchmark_id), "status.json")


def lot_benchmark_progress_file(output_root: str, lot_id: str, benchmark_id: str) -> str:
    return os.path.join(lot_benchmark_logs_dir(output_root, lot_id, benchmark_id), "progress.jsonl")


def lot_benchmark_errors_file(output_root: str, lot_id: str, benchmark_id: str) -> str:
    return os.path.join(lot_benchmark_logs_dir(output_root, lot_id, benchmark_id), "errors.jsonl")


# Kept for backward-compat internal imports, but now points to logs/latest

def lot_benchmark_run_dir(output_root: str, lot_id: str, benchmark_id: str, benchmark_run_id: str) -> str:
    return lot_benchmark_logs_dir(output_root, lot_id, benchmark_id)


def lot_benchmark_lot_summary_file(output_root: str, lot_id: str, benchmark_id: str) -> str:
    return lot_benchmark_summary_file(output_root, lot_id, benchmark_id)


def lot_benchmark_config_dir(output_root: str, lot_id: str, benchmark_id: str) -> str:
    return lot_benchmark_dir(output_root, lot_id, benchmark_id)


def lot_benchmark_file(output_root: str, lot_id: str, profile_id: str) -> str:
    return os.path.join(lot_benchmarks_dir(output_root, lot_id), f"{profile_id}.json")


# Model benchmark artifacts

def model_benchmark_dir(output_root: str, model_id: str) -> str:
    return ensure_dir(os.path.join(model_dir(output_root, model_id), "benchmarks"))


def model_profile_dir(output_root: str, model_id: str, profile_id: str) -> str:
    return ensure_dir(os.path.join(model_benchmark_dir(output_root, model_id), profile_id))
