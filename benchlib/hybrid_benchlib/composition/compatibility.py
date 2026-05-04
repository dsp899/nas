from __future__ import annotations

from typing import Any, Dict


def validate_compatibility(cnn_record: Dict[str, Any], rnn_record: Dict[str, Any]) -> Dict[str, Any]:
    cnn_feature_dim = int(cnn_record["feature_dim"])
    rnn_feature_dim = int(rnn_record["feature_spec"]["feature_dim"])
    cnn_classes = int(cnn_record["experiment"]["num_classes"])
    rnn_classes = int(rnn_record["spec"]["num_classes"])
    checks = {
        "feature_dim_match": cnn_feature_dim == rnn_feature_dim,
        "num_classes_match": cnn_classes == rnn_classes,
        "cnn_feature_dim": cnn_feature_dim,
        "rnn_feature_dim": rnn_feature_dim,
        "cnn_num_classes": cnn_classes,
        "rnn_num_classes": rnn_classes,
    }
    checks["compatible"] = bool(checks["feature_dim_match"] and checks["num_classes_match"])
    if not checks["compatible"]:
        raise ValueError(f"Incompatibilidad CNN/RNN: {checks}")
    return checks
