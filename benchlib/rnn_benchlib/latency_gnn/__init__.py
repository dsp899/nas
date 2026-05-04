"""Latency GNN submodule for RNN artifacts using TensorFlow 2.10.

This module keeps imports light on purpose so dataset export does not pull TensorFlow.
"""

from rnn_benchlib.latency_gnn.dataset import (
    AggregatedLatencySample,
    discover_latency_samples_for_dataset,
    export_benchmark_dataset,
    load_exported_samples,
)


def load_run_predictor(*args, **kwargs):
    from rnn_benchlib.latency_gnn.predict import load_run_predictor as _impl
    return _impl(*args, **kwargs)


def predict_sample(*args, **kwargs):
    from rnn_benchlib.latency_gnn.predict import predict_sample as _impl
    return _impl(*args, **kwargs)


__all__ = [
    "AggregatedLatencySample",
    "discover_latency_samples_for_dataset",
    "export_benchmark_dataset",
    "load_exported_samples",
    "load_run_predictor",
    "predict_sample",
]
