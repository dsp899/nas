
import shutil
from pathlib import Path

import tensorflow as tf
from typing import Union


def save_model_without_compile_artifacts(model: tf.keras.Model, path: Union[str, Path]) -> None:
    target = str(path)
    try:
        model.save(target, include_optimizer=False)
    except TypeError:
        model.save(target)


def export_saved_model(model: tf.keras.Model, export_dir: Union[str, Path]) -> Path:
    target = Path(export_dir)
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    export_method = getattr(model, "export", None)
    if callable(export_method):
        export_method(str(target))
    else:
        tf.saved_model.save(model, str(target))
    return target
