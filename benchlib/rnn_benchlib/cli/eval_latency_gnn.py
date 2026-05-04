from __future__ import annotations

import argparse
import json

from rnn_benchlib.latency_gnn.artifacts import eval_dir, run_dir
from rnn_benchlib.latency_gnn.dataset import load_exported_samples
from rnn_benchlib.latency_gnn.evaluate import evaluate_model
from rnn_benchlib.latency_gnn.predict import load_run_predictor
from rnn_benchlib.latency_gnn.splits import apply_split, load_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evalúa un run de latency GNN ya entrenado dentro de un dataset concreto. Usa siempre ./artifacts.")
    parser.add_argument("--lot-id", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = "artifacts"
    current_run_dir = run_dir(output_root, args.lot_id, args.dataset_id, args.run_id)
    run_payload = json.loads((current_run_dir / "run.json").read_text(encoding="utf-8"))
    predictor = load_run_predictor(current_run_dir, device=args.device)
    samples = load_exported_samples(output_root, args.lot_id, args.dataset_id)
    split_payload = load_split(output_root, args.lot_id, args.dataset_id, run_payload["split_id"])
    samples_by_split = apply_split(samples, split_payload)
    encoded = predictor.featurizer.encode_many(samples_by_split["test"])
    metrics = evaluate_model(predictor.model, encoded, predictor.featurizer.spec.targets, device=predictor.device)
    out_dir = eval_dir(output_root, args.lot_id, args.dataset_id, args.run_id)
    (out_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Evaluación escrita en: {out_dir / 'test_metrics.json'}")
    for target_name, target_metrics in metrics.items():
        print(target_name)
        for metric_name, metric_value in target_metrics.items():
            print(f"  {metric_name}: {metric_value:.6f}")


if __name__ == "__main__":
    main()
