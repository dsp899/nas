from __future__ import annotations

import platform
import socket
from typing import Any, Dict, List, Tuple

from benchlib_common.artifacts.ids import stable_hash
from benchlib_common.io.jsonl import append_jsonl, write_json
from benchlib_common.synthetic.images import SyntheticVideoImageSpec, generate_synthetic_video_frames
from hybrid_benchlib.composition.bundles import load_cnn_bundle, load_rnn_bundle
from hybrid_benchlib.config.schemas import HybridBenchmarkConfig
from hybrid_benchlib.pipeline.engine import HybridPipelineEngine
from hybrid_benchlib.runtime.backends import create_cnn_backend_from_record, create_rnn_backend_from_record
from hybrid_benchlib.storage.layout import build_artifact_paths, build_experiment_paths
from hybrid_benchlib.storage.registry import HybridRegistry, utc_now_iso


RUNTIME_PRESET_TO_BACKENDS = {
    "float_all": ("float", "float"),
    "tflite_all": ("tflite", "tflite"),
    "xmodel_tflite": ("xmodel", "tflite"),
}


def _resolve_backends(config: HybridBenchmarkConfig) -> Tuple[str, str]:
    try:
        return RUNTIME_PRESET_TO_BACKENDS[config.runtime_preset]
    except KeyError as exc:
        raise ValueError(f"runtime_preset no soportado: {config.runtime_preset!r}") from exc


def _build_experiment_meta(experiment_name: str, config: HybridBenchmarkConfig, hybrid_record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "experiment_id": f"hybrid_exp_{stable_hash({'name': experiment_name, 'config': config.to_dict(), 'hybrid': hybrid_record['hybrid_model_id']})}",
        "experiment_name": experiment_name,
        "created_at_utc": utc_now_iso(),
        "host_name": socket.gethostname(),
        "device_name": platform.platform(),
        "runtime_preset": config.runtime_preset,
        "pipeline_mode": config.overlap_mode,
        "extra": {
            "cnn_workers": config.cnn_workers,
            "sample_stride_frames": config.sample_stride_frames,
            "video_fps": config.video_fps,
            "threads": config.threads,
            "num_videos": config.num_videos,
            "frames_per_video": config.frames_per_video,
            "hybrid_model_id": hybrid_record["hybrid_model_id"],
            "cnn_model_id": hybrid_record["cnn_model_id"],
            "rnn_model_id": hybrid_record["rnn_model_id"],
        },
    }


def _build_engine(cnn_record: Dict[str, Any], rnn_record: Dict[str, Any], *, cnn_backend: str, rnn_backend: str, config: HybridBenchmarkConfig) -> HybridPipelineEngine:
    spec = rnn_record["spec"]
    seq = int(spec["seq"])
    hop = int(rnn_record["feature_spec"]["video_steps"] // (rnn_record["feature_spec"]["video_steps"] // seq)) if False else int(1)
    hop = int(config.__dict__.get("hop", 1) or 1)
    return HybridPipelineEngine(
        cnn_backend_factory=lambda: create_cnn_backend_from_record(cnn_record, cnn_backend, threads=config.threads),
        rnn_backend_factory=lambda: create_rnn_backend_from_record(rnn_record, rnn_backend, threads=config.threads),
        seq=seq,
        hop=hop,
        cnn_workers=config.cnn_workers,
        pipeline_mode=config.overlap_mode,
    )


def benchmark_hybrid_pipeline_host(
    output_root: str,
    hybrid_model_id: str,
    config: HybridBenchmarkConfig,
    *,
    experiment_name: str = "hybrid_pipeline_benchmark",
) -> Dict[str, Any]:
    paths = build_artifact_paths(output_root, hybrid_model_id)
    registry = HybridRegistry(paths.registry_path)
    hybrid_record = registry._data["hybrids_by_id"].get(hybrid_model_id)
    if hybrid_record is None:
        raise KeyError(f"No existe bundle híbrido con id {hybrid_model_id!r}")

    cnn_record = load_cnn_bundle(output_root, hybrid_record["cnn_model_id"])
    rnn_record = load_rnn_bundle(output_root, hybrid_record["rnn_model_id"])
    cnn_backend, rnn_backend = _resolve_backends(config)

    input_size = int(cnn_record.get("notes", {}).get("input_size", cnn_record["spec"]["input_size"]))
    frames = generate_synthetic_video_frames(
        SyntheticVideoImageSpec(
            num_videos=config.num_videos,
            frames_per_video=config.frames_per_video,
            image_size=input_size,
            seed=config.seed,
        )
    )

    experiment_meta = _build_experiment_meta(experiment_name, config, hybrid_record)
    experiment_paths = build_experiment_paths(output_root, experiment_meta["experiment_id"])

    def make_engine() -> HybridPipelineEngine:
        return HybridPipelineEngine(
            cnn_backend_factory=lambda: create_cnn_backend_from_record(cnn_record, cnn_backend, threads=config.threads),
            rnn_backend_factory=lambda: create_rnn_backend_from_record(rnn_record, rnn_backend, threads=config.threads),
            seq=int(rnn_record["spec"]["seq"]),
            hop=int(config.hop),
            cnn_workers=config.cnn_workers,
            pipeline_mode=config.overlap_mode,
        )

    # Warmup without recording.
    for warmup_idx in range(max(0, config.warmup_runs)):
        video = frames[warmup_idx % config.num_videos]
        make_engine().run_video(video, sample_stride_frames=config.sample_stride_frames)

    rows: List[Dict[str, Any]] = []
    event_count = 0
    t_firsts: List[float] = []
    t_updates: List[float] = []
    total_video_ms: List[float] = []
    cnn_total_ms: List[float] = []
    rnn_total_ms: List[float] = []
    queue_wait_ms: List[float] = []

    for video_index in range(config.num_videos):
        summary = make_engine().run_video(frames[video_index], sample_stride_frames=config.sample_stride_frames)
        row = {
            "experiment": experiment_meta,
            "hybrid_model_id": hybrid_model_id,
            "cnn_model_id": hybrid_record["cnn_model_id"],
            "rnn_model_id": hybrid_record["rnn_model_id"],
            "runtime_preset": config.runtime_preset,
            "cnn_backend": cnn_backend,
            "rnn_backend": rnn_backend,
            "pipeline_mode": config.overlap_mode,
            "video_index": video_index,
            "seq": int(rnn_record["spec"]["seq"]),
            "hop": int(config.hop),
            "num_sampled_frames": summary.num_sampled_frames,
            "num_clips": summary.num_clips,
            "t_first_ms": summary.t_first_ms,
            "t_update_mean_ms": summary.t_update_mean_ms,
            "t_update_min_ms": summary.t_update_min_ms,
            "t_update_max_ms": summary.t_update_max_ms,
            "cnn_total_ms": summary.cnn_total_ms,
            "rnn_total_ms": summary.rnn_total_ms,
            "queue_wait_total_ms": summary.queue_wait_total_ms,
            "video_total_ms": summary.video_total_ms,
            "feature_records": summary.feature_records,
            "clip_records": summary.clip_records,
            "extra": {
                "feature_dim": hybrid_record["feature_dim"],
                "num_classes": hybrid_record["num_classes"],
                "cnn_workers": config.cnn_workers,
                "sample_stride_frames": config.sample_stride_frames,
                "video_fps": config.video_fps,
                "frames_per_video": config.frames_per_video,
                "memory_mode": rnn_record["spec"]["memory_mode"],
                "direction": rnn_record["spec"]["direction"],
                "video_decision": rnn_record["spec"]["video_decision"],
                "video_decision_input": rnn_record["spec"]["video_decision_input"],
                "rnn": rnn_record["spec"]["rnn"],
                "head_units": rnn_record["spec"]["head_units"],
                "cnn_backbone": cnn_record["spec"]["backbone_name"],
                "execution_runtime": "unified_real_pipeline",
            },
        }
        rows.append(row)
        append_jsonl(experiment_paths.results_jsonl_path, row)
        for item in summary.feature_records:
            append_jsonl(
                experiment_paths.event_trace_jsonl_path,
                {
                    "experiment_id": experiment_meta["experiment_id"],
                    "hybrid_model_id": hybrid_model_id,
                    "video_index": video_index,
                    "event_kind": "feature",
                    **item,
                },
            )
            event_count += 1
        for item in summary.clip_records:
            append_jsonl(
                experiment_paths.event_trace_jsonl_path,
                {
                    "experiment_id": experiment_meta["experiment_id"],
                    "hybrid_model_id": hybrid_model_id,
                    "video_index": video_index,
                    "event_kind": "clip",
                    **item,
                },
            )
            event_count += 1

        t_firsts.append(summary.t_first_ms)
        t_updates.append(summary.t_update_mean_ms)
        total_video_ms.append(summary.video_total_ms)
        cnn_total_ms.append(summary.cnn_total_ms)
        rnn_total_ms.append(summary.rnn_total_ms)
        queue_wait_ms.append(summary.queue_wait_total_ms)

    summary_payload = {
        "experiment": experiment_meta,
        "hybrid_model_id": hybrid_model_id,
        "cnn_model_id": hybrid_record["cnn_model_id"],
        "rnn_model_id": hybrid_record["rnn_model_id"],
        "runtime_preset": config.runtime_preset,
        "cnn_backend": cnn_backend,
        "rnn_backend": rnn_backend,
        "pipeline_mode": config.overlap_mode,
        "num_videos": config.num_videos,
        "warmup_runs": config.warmup_runs,
        "results_rows_written": len(rows),
        "event_rows_written": event_count,
        "t_first_ms": {
            "mean": float(sum(t_firsts) / len(t_firsts)),
            "min": float(min(t_firsts)),
            "max": float(max(t_firsts)),
        },
        "t_update_mean_ms": {
            "mean": float(sum(t_updates) / len(t_updates)),
            "min": float(min(t_updates)),
            "max": float(max(t_updates)),
        },
        "video_total_ms": {
            "mean": float(sum(total_video_ms) / len(total_video_ms)),
            "min": float(min(total_video_ms)),
            "max": float(max(total_video_ms)),
        },
        "cnn_total_ms": {
            "mean": float(sum(cnn_total_ms) / len(cnn_total_ms)),
            "min": float(min(cnn_total_ms)),
            "max": float(max(cnn_total_ms)),
        },
        "rnn_total_ms": {
            "mean": float(sum(rnn_total_ms) / len(rnn_total_ms)),
            "min": float(min(rnn_total_ms)),
            "max": float(max(rnn_total_ms)),
        },
        "queue_wait_total_ms": {
            "mean": float(sum(queue_wait_ms) / len(queue_wait_ms)),
            "min": float(min(queue_wait_ms)),
            "max": float(max(queue_wait_ms)),
        },
        "results_jsonl_path": experiment_paths.results_jsonl_path,
        "event_trace_jsonl_path": experiment_paths.event_trace_jsonl_path,
    }
    write_json(experiment_paths.summary_json_path, summary_payload, indent=2)
    return summary_payload
