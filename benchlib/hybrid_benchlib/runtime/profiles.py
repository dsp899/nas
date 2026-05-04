from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from benchlib_common.io.jsonl import read_json
from benchlib_common.synthetic.images import SyntheticVideoImageSpec, generate_synthetic_video_frames
from cnn_benchlib.runtime.backends import FloatCnnRuntime, TfliteCnnRuntime, XmodelCnnRuntime
from cnn_benchlib.storage.layout import build_artifact_paths as build_cnn_paths
from rnn_benchlib.config.schemas import FeatureSpec, ModelSpec
from rnn_benchlib.features.synthetic_video_features import generate_synthetic_video_batch


@dataclass(frozen=True)
class CnnComponentProfile:
    backend: str
    source: str
    preprocess_ms: float
    infer_ms: float
    postprocess_ms: float
    sample_total_ms: float
    init_ms: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RnnComponentProfile:
    backend: str
    source: str
    init_ms: Optional[float]
    encoder_ms: float
    clip_head_ms: float
    clip_total_ms: float
    aggregation_ms: float
    video_head_ms: float
    video_total_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _load_json_if_exists(path: str | Path) -> Optional[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return None
    return read_json(p, default=None)


def _profile_from_cnn_summary(summary: Dict[str, Any], backend: str) -> CnnComponentProfile:
    return CnnComponentProfile(
        backend=backend,
        source="cached_benchmark",
        preprocess_ms=float(summary.get("preprocess", {}).get("mean_ms", 0.0)),
        infer_ms=float(summary.get("infer", {}).get("mean_ms", 0.0)),
        postprocess_ms=float(summary.get("postprocess", {}).get("mean_ms", 0.0)),
        sample_total_ms=float(summary.get("sample_total", {}).get("mean_ms", 0.0)),
        init_ms=float(summary.get("init_ms")) if summary.get("init_ms") is not None else None,
    )


def measure_cnn_component_profile(
    output_root: str,
    cnn_record: Dict[str, Any],
    backend: str,
    *,
    threads: int = 1,
    seed: int = 1234,
    prefer_cached: bool = True,
    samples: int = 8,
    xmodel_summary_path: Optional[str] = None,
) -> CnnComponentProfile:
    model_id = str(cnn_record["model_id"])
    paths = build_cnn_paths(output_root, model_id)

    if backend == "xmodel":
        candidates = []
        if xmodel_summary_path:
            candidates.append(Path(xmodel_summary_path))
        candidates.append(Path(paths.benchmark_dir) / "summary_xmodel.json")
        for candidate in candidates:
            summary = _load_json_if_exists(candidate)
            if summary is not None:
                return _profile_from_cnn_summary(summary, backend="xmodel")

    if prefer_cached:
        summary = _load_json_if_exists(Path(paths.benchmark_dir) / f"summary_{backend}.json")
        if summary is not None:
            return _profile_from_cnn_summary(summary, backend=backend)

    input_size = int(cnn_record.get("notes", {}).get("input_size", cnn_record["spec"]["input_size"]))
    frames = generate_synthetic_video_frames(
        SyntheticVideoImageSpec(num_videos=1, frames_per_video=samples, image_size=input_size, seed=seed)
    )[0]

    if backend == "float":
        runner = FloatCnnRuntime(cnn_record["float_extractor_path"])
        init_ms = None
    elif backend == "tflite":
        tflite_path = cnn_record.get("tflite_extractor", {}).get("path") or paths.tflite_extractor_path
        runner = TfliteCnnRuntime(tflite_path, num_threads=threads)
        init_ms = None
    elif backend == "xmodel":
        xmodel_path = cnn_record.get("xmodel_extractor", {}).get("path") or paths.xmodel_extractor_path
        runner = XmodelCnnRuntime(xmodel_path)
        init_ms = None
    else:
        raise ValueError(f"backend CNN no soportado: {backend!r}")

    preprocess_ms = []
    infer_ms = []
    postprocess_ms = []
    total_ms = []
    for frame in frames:
        result = runner.run(frame)
        preprocess_ms.append(result.preprocess_ms)
        infer_ms.append(result.infer_ms)
        postprocess_ms.append(result.postprocess_ms)
        total_ms.append(result.total_ms)

    return CnnComponentProfile(
        backend=backend,
        source="measured_host",
        preprocess_ms=float(np.mean(preprocess_ms)),
        infer_ms=float(np.mean(infer_ms)),
        postprocess_ms=float(np.mean(postprocess_ms)),
        sample_total_ms=float(np.mean(total_ms)),
        init_ms=init_ms,
    )


def measure_rnn_component_profile(
    rnn_record: Dict[str, Any],
    backend: str,
    *,
    threads: int = 1,
    seed: int = 1234,
    warmup_runs: int = 3,
    steady_runs: int = 5,
) -> RnnComponentProfile:
    from rnn_benchlib.benchmark.runners import (
        benchmark_float_video,
        benchmark_tflite_video,
        load_float_model,
        timed_create_encoder_tflite_interpreter,
        timed_create_head_tflite_interpreter,
    )

    spec = ModelSpec(**rnn_record["spec"])
    feature_spec = FeatureSpec(**rnn_record["feature_spec"])
    batch = generate_synthetic_video_batch(
        num_videos=1,
        feature_spec=feature_spec,
        model_spec=spec,
        seed=seed,
    )
    video_clips = batch.clips[0]
    artifacts = rnn_record["artifacts"]

    if backend == "float":
        encoder_model = load_float_model(artifacts["encoder_keras_dir"])
        head_model = load_float_model(artifacts["head_keras_dir"])
        result = benchmark_float_video(
            encoder_model=encoder_model,
            head_model=head_model,
            spec=spec,
            video_clips=video_clips,
            warmup_runs=warmup_runs,
            steady_runs=steady_runs,
        )
        init_ms = None
        source = "measured_host"
    elif backend == "tflite":
        encoder_path = artifacts.get("encoder_tflite_path")
        head_path = artifacts.get("head_tflite_path")
        if not encoder_path or not head_path:
            raise ValueError("El artefacto RNN no tiene encoder/head TFLite listos")
        encoder_interpreter, init_encoder_ms, input_map, output_indices = timed_create_encoder_tflite_interpreter(
            encoder_path,
            spec,
            feature_spec,
            num_threads=threads,
        )
        head_interpreter, init_head_ms, head_input_index, head_output_index = timed_create_head_tflite_interpreter(
            head_path,
            spec,
            num_threads=threads,
        )
        result = benchmark_tflite_video(
            encoder_interpreter=encoder_interpreter,
            head_interpreter=head_interpreter,
            spec=spec,
            video_clips=video_clips,
            init_ms=float(init_encoder_ms + init_head_ms),
            encoder_input_name_to_index=input_map,
            encoder_ordered_output_indices=output_indices,
            head_input_index=head_input_index,
            head_output_index=head_output_index,
            warmup_runs=warmup_runs,
            steady_runs=steady_runs,
        )
        init_ms = float(init_encoder_ms + init_head_ms)
        source = "measured_host"
    else:
        raise ValueError(f"backend RNN no soportado: {backend!r}")

    return RnnComponentProfile(
        backend=backend,
        source=source,
        init_ms=init_ms,
        encoder_ms=float(np.mean(result["steady_encoder_ms"])),
        clip_head_ms=float(np.mean(result["steady_head_ms"])),
        clip_total_ms=float(np.mean(result["steady_clip_total_ms"])),
        aggregation_ms=float(np.mean(result["steady_aggregation_ms"])),
        video_head_ms=float(np.mean(result["steady_video_head_ms"])),
        video_total_ms=float(np.mean(result["steady_video_total_ms"])),
    )
