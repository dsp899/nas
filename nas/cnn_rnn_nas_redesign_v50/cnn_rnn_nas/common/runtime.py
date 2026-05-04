import gc
import logging
import os
import random
import warnings
from typing import Any

import numpy as np
import psutil


def process_memory_mb() -> float:
    return psutil.Process().memory_info().rss / (1024 ** 2)


def prepare_tensorflow_logging() -> None:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    warnings.filterwarnings(
        "ignore",
        message=r"Compiled the loaded model, but the compiled metrics have yet to be built.*",
    )
    try:
        import absl.logging  # type: ignore

        absl.logging.set_verbosity(absl.logging.ERROR)
        absl.logging.set_stderrthreshold("error")
    except Exception:
        pass


def _tensorflow():
    prepare_tensorflow_logging()
    import tensorflow as tf

    try:
        tf.get_logger().setLevel("ERROR")
    except Exception:
        pass
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    return tf


def configure_runtime(gpu: str = "0", mixed_precision: bool = True, seed: int = 1337) -> bool:
    tf = _tensorflow()
    tf.keras.utils.set_random_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        tf.keras.mixed_precision.set_global_policy("float32")
        return False

    index = int(gpu)
    if index >= len(gpus):
        raise ValueError("GPU {} no disponible. GPUs detectadas: {}".format(gpu, len(gpus)))

    tf.config.set_visible_devices(gpus[index], "GPU")
    tf.config.experimental.set_memory_growth(gpus[index], True)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    effective_mixed_precision = bool(mixed_precision)
    if mixed_precision:
        details = tf.config.experimental.get_device_details(gpus[index])
        capability = details.get("compute_capability") if isinstance(details, dict) else None
        if capability is not None and tuple(capability) < (7, 0):
            effective_mixed_precision = False

    tf.keras.mixed_precision.set_global_policy("mixed_float16" if effective_mixed_precision else "float32")
    return effective_mixed_precision


def release_memory(*objects: Any) -> None:
    tf = _tensorflow()
    for obj in objects:
        try:
            del obj
        except Exception:
            pass
    tf.keras.backend.clear_session()
    gc.collect()
