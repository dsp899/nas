import argparse
import json
import os
from pprint import pprint

# Silence TensorFlow C++ logs before importing project modules that may import TensorFlow.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified entrypoint for CNN, RNN, and NAS workflows")
    subparsers = parser.add_subparsers(dest="component", required=True)

    cnn_parser = subparsers.add_parser("cnn", help="CNN operations")
    cnn_parser.add_argument("operation", choices=("train", "test", "export_features", "deploy"))
    cnn_parser.add_argument("--config", required=False, help="Optional JSON config path for CNN")

    rnn_parser = subparsers.add_parser("rnn", help="RNN operations")
    rnn_parser.add_argument("operation", choices=("train", "test", "deploy"))
    rnn_parser.add_argument("--config", required=False, help="Optional JSON config path for RNN")

    nas_parser = subparsers.add_parser("nas", help="NAS operations")
    nas_subparsers = nas_parser.add_subparsers(dest="operation", required=True)

    nas_search = nas_subparsers.add_parser("search", help="Run NAS search")
    nas_search.add_argument("--config", required=False, help="Optional JSON config path for NAS search")

    nas_plot = nas_subparsers.add_parser("plot", help="Generate NAS plots from a completed run summary")
    nas_plot.add_argument("--summary-json", required=True, help="Path to a NAS search summary JSON")
    nas_plot.add_argument("--output-dir", default=None, help="Optional output directory for generated plots")

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.component == "cnn":
        from cnn_rnn_nas.cnn.cnn_config_io import run_cnn

        pprint(run_cnn(args.operation, args.config))
        return

    if args.component == "rnn":
        from cnn_rnn_nas.rnn.rnn_config_io import run_rnn

        pprint(run_rnn(args.operation, args.config))
        return

    if args.component == "nas" and args.operation == "search":
        from cnn_rnn_nas.nas.nas_config_io import run_nas

        pprint(run_nas(args.config))
        return

    if args.component == "nas" and args.operation == "plot":
        from cnn_rnn_nas.nas.nas_plotting import generate_search_analysis

        artifacts = generate_search_analysis(summary_json=args.summary_json, output_dir=args.output_dir)
        print(json.dumps({
            "output_dir": str(artifacts.output_dir),
            "tables_dir": str(artifacts.tables_dir),
            "overview_dir": str(artifacts.overview_dir),
            "search_space_dir": str(artifacts.search_space_dir),
            "dimensions_dir": str(artifacts.dimensions_dir),
            "correlations_dir": str(artifacts.correlations_dir),
            "metrics_by_epoch_csv": str(artifacts.metrics_by_epoch_csv),
            "metrics_by_sample_csv": str(artifacts.metrics_by_sample_csv),
            "dimension_value_stats_csv": str(artifacts.dimension_value_stats_csv),
            "dimension_importance_csv": str(artifacts.dimension_importance_csv),
            "pairwise_interactions_csv": str(artifacts.pairwise_interactions_csv),
            "manifest_json": str(artifacts.manifest_json),
            "epoch_cumulative_plot": str(artifacts.epoch_cumulative_plot),
            "epoch_rolling_plot": str(artifacts.epoch_rolling_plot),
            "controller_loss_plot": str(artifacts.controller_loss_plot),
            "dimension_importance_plot": str(artifacts.dimension_importance_plot),
            "pairwise_interactions_plot": str(artifacts.pairwise_interactions_plot),
            "layer_distribution_plot": str(artifacts.layer_distribution_plot),
            "cumulative_layer_distribution_plot": str(artifacts.cumulative_layer_distribution_plot),
            "dimension_profiles_dir": str(artifacts.dimension_profiles_dir),
        }, indent=2, sort_keys=True))
        return

    raise ValueError("Unsupported component/operation combination")


if __name__ == "__main__":
    main()
