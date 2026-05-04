from __future__ import annotations

from pathlib import Path
from typing import Dict, List
from benchlib_common.io.jsonl import append_jsonl, write_json
from benchlib_common.synthetic.images import SyntheticVideoImageSpec, generate_synthetic_video_frames
from benchlib_common.timing.stats import summarize_ms
from cnn_benchlib.runtime.backends import FloatCnnRuntime, TfliteCnnRuntime, XmodelCnnRuntime
from cnn_benchlib.storage.layout import build_artifact_paths
from cnn_benchlib.storage.registry import CnnModelRegistry


def benchmark_cnn_model(output_root: str, model_id: str, runtime: str, num_videos: int = 8, frames_per_video: int = 16, seed: int = 1234, threads: int = 1):
    paths = build_artifact_paths(output_root, model_id)
    registry = CnnModelRegistry(paths.registry_path)
    record = registry.require(model_id)
    input_size = int(record["notes"].get("input_size", record["spec"]["input_size"]))
    if runtime == "float":
        runner = FloatCnnRuntime(record["float_extractor_path"])
    elif runtime == "tflite":
        tflite_path = record.get("tflite_extractor", {}).get("path") or paths.tflite_extractor_path
        runner = TfliteCnnRuntime(tflite_path, num_threads=threads)
    elif runtime == "xmodel":
        xmodel_path = record.get("xmodel_extractor", {}).get("path") or paths.xmodel_extractor_path
        runner = XmodelCnnRuntime(xmodel_path)
    else:
        raise ValueError(f"Runtime CNN no soportado: {runtime!r}")
    videos = generate_synthetic_video_frames(SyntheticVideoImageSpec(num_videos=num_videos, frames_per_video=frames_per_video, image_size=input_size, seed=seed))
    rows: List[Dict[str, object]] = []
    total_samples = 0
    sample_total_ms: List[float] = []
    video_total_ms: List[float] = []
    preprocess_ms: List[float] = []
    infer_ms: List[float] = []
    postprocess_ms: List[float] = []
    for video_idx, video in enumerate(videos):
        per_video = 0.0
        for frame_idx, frame in enumerate(video):
            result = runner.run(frame)
            rows.append({"model_id": model_id, "runtime": runtime, "video_index": int(video_idx), "frame_index": int(frame_idx), "preprocess_ms": result.preprocess_ms, "infer_ms": result.infer_ms, "postprocess_ms": result.postprocess_ms, "total_ms": result.total_ms, "output_shape": list(result.output.shape)})
            total_samples += 1
            per_video += result.total_ms
            sample_total_ms.append(result.total_ms)
            preprocess_ms.append(result.preprocess_ms)
            infer_ms.append(result.infer_ms)
            postprocess_ms.append(result.postprocess_ms)
        video_total_ms.append(per_video)
    results_path = Path(paths.benchmark_dir) / f"benchmark_{runtime}.jsonl"
    for row in rows:
        append_jsonl(results_path, row)
    summary = {"model_id": model_id, "runtime": runtime, "num_videos": num_videos, "frames_per_video": frames_per_video, "threads": threads, "total_samples": total_samples, "preprocess": summarize_ms(preprocess_ms), "infer": summarize_ms(infer_ms), "postprocess": summarize_ms(postprocess_ms), "sample_total": summarize_ms(sample_total_ms), "video_total": summarize_ms(video_total_ms), "results_jsonl": str(results_path)}
    write_json(Path(paths.benchmark_dir) / f"summary_{runtime}.json", summary, indent=2)
    return summary
