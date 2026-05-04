from __future__ import annotations

import argparse

from rnn_benchlib.latency_gnn.dataset import load_exported_samples
from rnn_benchlib.latency_gnn.splits import materialize_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crea un split train/val/test persistente para un dataset concreto de latency GNN. Usa siempre ./artifacts.")
    parser.add_argument("--lot-id", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = "artifacts"
    samples = load_exported_samples(output_root, args.lot_id, args.dataset_id)
    payload = materialize_split(output_root, args.lot_id, args.dataset_id, samples, args.train_ratio, args.val_ratio, args.test_ratio, args.seed)
    print(f"split_id: {payload['split_id']}")
    print(f"sample_counts: {payload['sample_counts']}")


if __name__ == "__main__":
    main()
