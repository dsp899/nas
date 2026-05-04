from __future__ import annotations

import argparse
import json

from hybrid_benchlib.benchmark.host import benchmark_hybrid_pipeline_host
from hybrid_benchlib.config.schemas import HybridBenchmarkConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark del pipeline híbrido CNN-RNN con runtime unificado real")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--hybrid-model-id", required=True)
    parser.add_argument("--experiment-name", default="hybrid_pipeline_compare")
    parser.add_argument("--runtime-preset", choices=("float_all", "tflite_all", "xmodel_tflite"), default="tflite_all")
    parser.add_argument("--overlap-mode", choices=("cnn_rnn_overlap", "cnn_rnn_serialized"), default="cnn_rnn_overlap")
    parser.add_argument("--cnn-workers", type=int, default=3)
    parser.add_argument("--hop", type=int, default=1)
    parser.add_argument("--sample-stride-frames", type=int, default=1)
    parser.add_argument("--video-fps", type=float, default=30.0)
    parser.add_argument("--num-videos", type=int, default=8)
    parser.add_argument("--frames-per-video", type=int, default=64)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--steady-runs", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--xmodel-summary-path")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = HybridBenchmarkConfig(
        runtime_preset=args.runtime_preset,
        overlap_mode=args.overlap_mode,
        cnn_workers=args.cnn_workers,
        hop=args.hop,
        sample_stride_frames=args.sample_stride_frames,
        video_fps=args.video_fps,
        num_videos=args.num_videos,
        frames_per_video=args.frames_per_video,
        warmup_runs=args.warmup_runs,
        steady_runs=args.steady_runs,
        threads=args.threads,
        seed=args.seed,
        xmodel_summary_path=args.xmodel_summary_path,
    )
    payload = benchmark_hybrid_pipeline_host(
        output_root=args.output_root,
        hybrid_model_id=args.hybrid_model_id,
        config=config,
        experiment_name=args.experiment_name,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
