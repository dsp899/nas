from __future__ import annotations

import argparse
import os

from rnn_benchlib.analysis.report import flatten_records, load_results_from_paths, write_analysis_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analiza resultados de benchmark host y genera tablas, resumen y gráficas. Usa siempre ./artifacts como raíz por defecto.")
    parser.add_argument("--results-jsonl", type=str, default=None, help="Ruta directa a benchmark_results.jsonl")
    parser.add_argument("--experiment-root", type=str, default=None, help="Ruta al directorio del experimento benchmark_runs/exp_xxx")
    parser.add_argument("--experiment-id", type=str, default=None, help="ID de experimento para resolver dentro de ./artifacts/benchmark_runs")
    parser.add_argument("--analysis-dir", type=str, default=None, help="Directorio de salida del análisis")
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[str, str, str]:
    output_root = os.path.abspath("artifacts")
    if args.results_jsonl:
        results_jsonl = args.results_jsonl
        experiment_root = os.path.dirname(os.path.abspath(results_jsonl))
    elif args.experiment_root:
        experiment_root = os.path.abspath(args.experiment_root)
        results_jsonl = os.path.join(experiment_root, "benchmark_results.jsonl")
    elif args.experiment_id:
        experiment_root = os.path.join(output_root, "benchmark_runs", args.experiment_id)
        results_jsonl = os.path.join(experiment_root, "benchmark_results.jsonl")
    else:
        raise SystemExit("Debes indicar --results-jsonl, o --experiment-root, o --experiment-id.")

    if not os.path.exists(results_jsonl):
        raise FileNotFoundError(f"No existe results_jsonl: {results_jsonl}")

    analysis_dir = args.analysis_dir or os.path.join(experiment_root, "analysis")
    return experiment_root, results_jsonl, analysis_dir


def main() -> None:
    args = parse_args()
    experiment_root, results_jsonl, analysis_dir = resolve_paths(args)
    experiment_meta_path = os.path.join(experiment_root, "experiment_meta.json")
    rows, meta = load_results_from_paths(results_jsonl, experiment_meta_path)
    if not rows:
        raise RuntimeError("No hay filas en benchmark_results.jsonl")
    df = flatten_records(rows)
    outputs = write_analysis_outputs(analysis_dir, df, meta)

    print("\n=== Benchmark analysis ===")
    print(f"experiment_root: {experiment_root}")
    print(f"results_jsonl  : {results_jsonl}")
    print(f"analysis_dir   : {analysis_dir}")
    for key, value in outputs.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
