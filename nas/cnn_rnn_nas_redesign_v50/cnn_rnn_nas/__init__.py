"""cnn_rnn_nas package."""

import os

# Reduce TensorFlow C++ log noise before TensorFlow is imported anywhere else.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
