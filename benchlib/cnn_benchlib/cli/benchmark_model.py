from __future__ import annotations

import argparse
import json
from cnn_benchlib.benchmark.runners import benchmark_cnn_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark individualizado de un extractor CNN.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--runtime", choices=("float", "tflite", "xmodel"), default="float")
    parser.add_argument("--num-videos", type=int, default=8)
    parser.add_argument("--frames-per-video", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    payload = benchmark_cnn_model(args.output_root, args.model_id, args.runtime, num_videos=args.num_videos, frames_per_video=args.frames_per_video, seed=args.seed, threads=args.threads)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
