"""Runtime unificado de hybrid_benchlib."""

from hybrid_benchlib.runtime.backends import (
    CnnFeatureResult,
    CnnStageTiming,
    RnnDecisionResult,
    create_cnn_backend_from_record,
    create_rnn_backend_from_record,
)

__all__ = [
    "CnnFeatureResult",
    "CnnStageTiming",
    "RnnDecisionResult",
    "create_cnn_backend_from_record",
    "create_rnn_backend_from_record",
]
