from __future__ import annotations

import argparse

from rnn_benchlib.latency_gnn.artifacts import dataset_dir
from rnn_benchlib.latency_gnn.dataset import export_benchmark_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exporta un dataset de latency GNN ligado a un lote y a un benchmark lógico concretos. Usa siempre ./artifacts.")
    parser.add_argument("--lot-id", required=True)
    parser.add_argument("--benchmark-id", required=True)
    parser.add_argument("--runtime", choices=["float", "tflite", "both"], default="both")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = "artifacts"
    runtime_filter = None if args.runtime == "both" else [args.runtime]
    payload = export_benchmark_dataset(output_root, args.lot_id, args.benchmark_id, runtime_filter=runtime_filter)
    print(f"dataset_id: {payload['dataset_id']}")
    print(f"Dataset exportado en: {dataset_dir(output_root, args.lot_id, payload['dataset_id'])}")
    print(f"num_samples: {payload['num_samples']}")


if __name__ == "__main__":
    main()
