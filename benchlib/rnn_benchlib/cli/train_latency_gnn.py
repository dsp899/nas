from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone

from rnn_benchlib.latency_gnn.dataset import load_exported_samples
from rnn_benchlib.latency_gnn.models import HeteroLatencyPredictorConfig
from rnn_benchlib.latency_gnn.splits import apply_split, load_split
from rnn_benchlib.latency_gnn.trainer import LatencyGnnTrainer, TrainingConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrena un predictor de latency GNN a partir de un dataset y un split persistentes. Usa siempre ./artifacts.")
    parser.add_argument("--lot-id", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--split-id", required=True)
    parser.add_argument("--run-name", default="baseline_tf210")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--graph-hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def _make_run_id(args: argparse.Namespace) -> str:
    payload = {
        "lot_id": args.lot_id,
        "dataset_id": args.dataset_id,
        "split_id": args.split_id,
        "run_name": args.run_name,
        "epochs": args.epochs,
        "seed": args.seed,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    return "run_" + hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def main() -> None:
    args = parse_args()
    output_root = "artifacts"
    samples = load_exported_samples(output_root, args.lot_id, args.dataset_id)
    split_payload = load_split(output_root, args.lot_id, args.dataset_id, args.split_id)
    samples_by_split = apply_split(samples, split_payload)
    trainer = LatencyGnnTrainer(
        train_config=TrainingConfig(
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            device=args.device,
            seed=args.seed,
        ),
        model_config=HeteroLatencyPredictorConfig(
            hidden_dim=args.hidden_dim,
            graph_hidden_dim=args.graph_hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
        ),
    )
    run_id = _make_run_id(args)
    metrics = trainer.fit(samples_by_split, output_root, args.lot_id, args.dataset_id, run_id, args.run_name, split_payload)
    print(f"run_id: {run_id}")
    print(f"best_val_wall_mape: {metrics['best_val_wall_mape']:.6f}")
    print(f"checkpoint: {metrics['checkpoint']}")


if __name__ == "__main__":
    main()
