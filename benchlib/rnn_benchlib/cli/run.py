from __future__ import annotations

import argparse
import warnings
from typing import Any, Dict

from rnn_benchlib.config.script_loader import load_rnn_config, search_space_to_dict
from pathlib import Path

from rnn_benchlib.benchmark.signature import benchmark_id_from_signature
from rnn_benchlib.latency_gnn.artifacts import make_dataset_id
from rnn_benchlib.latency_gnn.dataset import export_benchmark_dataset, load_exported_samples
from rnn_benchlib.latency_gnn.splits import materialize_split, apply_split
from rnn_benchlib.storage.state import stable_hash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CLI mínima de la rama RNN: la operación y el fichero de configuración mandan. La persistencia usa siempre ./artifacts.")
    parser.add_argument("--op", required=True, choices=["generate", "benchmark", "export-dataset", "make-split", "train-gnn", "eval-gnn"])
    parser.add_argument("--config", required=True, help="Fichero de configuración centralizado de la rama RNN. Se recomienda ubicarlo en ./configs y diferenciarlo en el nombre como rnn_*.json.")
    return parser.parse_args()


def _lot_signature_fields(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "search_space": search_space_to_dict(cfg["search_space"]),
        "generation_seed": int(cfg["generation_seed"]),
        "requested_count": int(cfg["requested_count"]),
    }


def _lot_id(cfg: Dict[str, Any]) -> str:
    return stable_hash({"kind": "generation", "signature_fields": _lot_signature_fields(cfg)}, prefix="lot")


def _benchmark_signature(cfg: Dict[str, Any]) -> Dict[str, Any]:
    bench = dict(cfg.get("benchmark") or {})
    sig = dict(bench.get("signature") or {})
    if not sig:
        raise RuntimeError("El script debe definir BENCHMARK['signature'] para esta operación.")
    return sig


def _benchmark_id(cfg: Dict[str, Any]) -> str:
    return benchmark_id_from_signature(_benchmark_signature(cfg))


def main() -> None:
    args = parse_args()
    output_root = "artifacts"
    cfg = load_rnn_config(args.config)
    lot_id = _lot_id(cfg)
    benchmark_id = _benchmark_id(cfg) if cfg.get("benchmark") else None

    if args.op == "generate":
        from rnn_benchlib.cli.generate_models import main as generate_main
        import sys
        sys.argv = [sys.argv[0], "--config", args.config]
        generate_main()
        return

    if args.op == "benchmark":
        from rnn_benchlib.cli.benchmark_models import main as benchmark_main
        import sys
        sys.argv = [sys.argv[0], "--config", args.config, "--lot-id", lot_id]
        benchmark_main()
        return

    if args.op == "export-dataset":
        payload = export_benchmark_dataset(output_root, lot_id, benchmark_id, runtime_filter=["tflite"])
        print(f"lot_id: {lot_id}")
        print(f"benchmark_id: {benchmark_id}")
        print(f"dataset_id: {payload['dataset_id']}")
        print(f"num_samples: {payload['num_samples']}")
        return

    if args.op == "make-split":
        dataset_id = make_dataset_id(output_root, lot_id, benchmark_id, runtime_filter=["tflite"])
        samples = load_exported_samples(output_root, lot_id, dataset_id)
        split_cfg = dict(cfg.get("split") or {})
        payload = materialize_split(
            output_root=output_root,
            lot_id=lot_id,
            dataset_id=dataset_id,
            samples=samples,
            train_ratio=float(split_cfg.get("train_ratio", 0.70)),
            val_ratio=float(split_cfg.get("val_ratio", 0.15)),
            test_ratio=float(split_cfg.get("test_ratio", 0.15)),
            seed=int(split_cfg.get("seed", cfg["generation_seed"])),
        )
        print(f"dataset_id: {dataset_id}")
        print(f"split_id: {payload['split_id']}")
        return

    if args.op in {"train-gnn", "eval-gnn"}:
        dataset_id = make_dataset_id(output_root, lot_id, benchmark_id, runtime_filter=["tflite"])
        samples = load_exported_samples(output_root, lot_id, dataset_id)
        split_cfg = dict(cfg.get("split") or {})
        split_payload = materialize_split(
            output_root=output_root,
            lot_id=lot_id,
            dataset_id=dataset_id,
            samples=samples,
            train_ratio=float(split_cfg.get("train_ratio", 0.70)),
            val_ratio=float(split_cfg.get("val_ratio", 0.15)),
            test_ratio=float(split_cfg.get("test_ratio", 0.15)),
            seed=int(split_cfg.get("seed", cfg["generation_seed"])),
        )
        split_id = split_payload["split_id"]
        samples_by_split = apply_split(samples, split_payload)

        gnn_cfg = dict(cfg.get("gnn") or {})
        gnn_training = dict(gnn_cfg.get("training") or {})
        gnn_optimizer = dict(gnn_cfg.get("optimizer") or {})
        gnn_runtime = dict(gnn_cfg.get("runtime") or {})
        run_name = str(gnn_cfg.get("run_name", "latency_gnn"))
        run_id = stable_hash({"dataset_id": dataset_id, "split_id": split_id, "run_name": run_name, "train_config": gnn_cfg}, prefix="run")

        if args.op == "train-gnn":
            from rnn_benchlib.latency_gnn.models import HeteroLatencyPredictorConfig
            from rnn_benchlib.latency_gnn.trainer import LatencyGnnTrainer, OptimizerConfig, TrainingConfig
            trainer = LatencyGnnTrainer(
                train_config=TrainingConfig(
                    epochs=int(gnn_training.get("epochs", gnn_cfg.get("epochs", 40))),
                    weight_decay=float(gnn_training.get("weight_decay", gnn_cfg.get("weight_decay", 1e-4))),
                    batch_size=int(gnn_training.get("batch_size", gnn_cfg.get("batch_size", 8))),
                    shuffle_train=bool(gnn_training.get("shuffle_train", True)),
                    device=str(gnn_runtime.get("device", gnn_cfg.get("device", "gpu"))),
                    gpu_index=int(gnn_runtime.get("gpu_index", gnn_cfg.get("gpu_index", 0))),
                    memory_growth=bool(gnn_runtime.get("memory_growth", gnn_cfg.get("memory_growth", True))),
                    mixed_precision=bool(gnn_runtime.get("mixed_precision", gnn_cfg.get("mixed_precision", True))),
                    enable_xla=bool(gnn_runtime.get("enable_xla", gnn_cfg.get("enable_xla", False))),
                    batch_progress=bool(gnn_runtime.get("batch_progress", gnn_cfg.get("batch_progress", True))),
                    batch_log_interval=int(gnn_runtime.get("batch_log_interval", gnn_cfg.get("batch_log_interval", 1))),
                    val_interval_epochs=int(gnn_training.get("val_interval_epochs", gnn_runtime.get("val_interval_epochs", gnn_cfg.get("val_interval_epochs", 1)))),
                    seed=int(gnn_cfg.get("seed", cfg["generation_seed"])),
                ),
                optimizer_config=OptimizerConfig(
                    name=str(gnn_optimizer.get("name", "adam")),
                    learning_rate=float(gnn_optimizer.get("learning_rate", gnn_cfg.get("learning_rate", 1e-3))),
                    beta_1=float(gnn_optimizer.get("beta_1", 0.9)),
                    beta_2=float(gnn_optimizer.get("beta_2", 0.999)),
                    epsilon=float(gnn_optimizer.get("epsilon", 1e-7)),
                    amsgrad=bool(gnn_optimizer.get("amsgrad", False)),
                    momentum=float(gnn_optimizer.get("momentum", 0.0)),
                    nesterov=bool(gnn_optimizer.get("nesterov", False)),
                    rho=float(gnn_optimizer.get("rho", 0.9)),
                    centered=bool(gnn_optimizer.get("centered", False)),
                    clipnorm=(None if gnn_optimizer.get("clipnorm") is None else float(gnn_optimizer.get("clipnorm"))),
                    clipvalue=(None if gnn_optimizer.get("clipvalue") is None else float(gnn_optimizer.get("clipvalue"))),
                ),
                model_config=HeteroLatencyPredictorConfig.from_dict(dict(gnn_cfg.get("model", gnn_cfg))),
            )
            result = trainer.fit(samples_by_split, output_root, lot_id, dataset_id, run_id, run_name, split_payload)
            print(f"dataset_id: {dataset_id}")
            print(f"split_id: {split_id}")
            print(f"run_id: {run_id}")
            print(f"best_checkpoint: {result['checkpoint']}")
            return

        if args.op == "eval-gnn":
            from rnn_benchlib.latency_gnn.artifacts import eval_dir, run_dir
            print(f"dataset_id: {dataset_id}")
            print(f"split_id: {split_id}")
            print(f"run_id: {run_id}")
            print(f"run_dir: {run_dir(output_root, lot_id, dataset_id, run_id)}")
            print(f"eval_dir: {eval_dir(output_root, lot_id, dataset_id, run_id)}")
            return


if __name__ == "__main__":
    main()
