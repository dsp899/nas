from __future__ import annotations

import argparse
import json
from hybrid_benchlib.composition.service import compose_hybrid_bundle
from hybrid_benchlib.config.schemas import HybridPipelineConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Compone artefactos CNN + RNN en un bundle híbrido.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--cnn-model-id", required=True)
    parser.add_argument("--rnn-model-id", required=True)
    parser.add_argument("--cnn-backend", choices=("float", "tflite", "xmodel"), default="xmodel")
    parser.add_argument("--rnn-backend", choices=("float", "tflite"), default="tflite")
    parser.add_argument("--cnn-workers", type=int, default=3)
    parser.add_argument("--hop", type=int, default=3)
    args = parser.parse_args()
    payload = compose_hybrid_bundle(args.output_root, args.cnn_model_id, args.rnn_model_id, HybridPipelineConfig(cnn_backend=args.cnn_backend, rnn_backend=args.rnn_backend, cnn_workers=args.cnn_workers, hop=args.hop))
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
