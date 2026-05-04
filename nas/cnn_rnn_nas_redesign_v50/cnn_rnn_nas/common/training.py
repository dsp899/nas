
import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf
from typing import Iterable, List, Optional, Union


@dataclass(frozen=True)
class OptimizerSpec:
    name: str
    learning_rate: float
    momentum: float = 0.0
    nesterov: bool = False
    weight_decay: float = 0.0
    mixed_precision: bool = False
    clipnorm: Optional[float] = None
    clipvalue: Optional[float] = None


def _legacy_namespace():
    return getattr(tf.keras.optimizers, "legacy", None)


def build_keras_optimizer(spec: OptimizerSpec, *, context: str) -> tf.keras.optimizers.Optimizer:
    name = spec.name.strip().lower()
    lr = spec.learning_rate
    legacy = _legacy_namespace()
    common_kwargs = {}
    if spec.clipnorm is not None:
        common_kwargs["clipnorm"] = spec.clipnorm
    if spec.clipvalue is not None:
        common_kwargs["clipvalue"] = spec.clipvalue

    if name == "adam":
        if legacy is not None and hasattr(legacy, "Adam"):
            return legacy.Adam(learning_rate=lr, **common_kwargs)
        return tf.keras.optimizers.Adam(learning_rate=lr, **common_kwargs)

    if name == "adamw":
        if legacy is not None and hasattr(legacy, "AdamW"):
            return legacy.AdamW(learning_rate=lr, weight_decay=spec.weight_decay, **common_kwargs)
        if spec.mixed_precision:
            warnings.warn(
                f"AdamW puede ser inestable con mixed precision en algunos entornos para {context}; se usará Adam estable.",
                RuntimeWarning,
            )
            fallback = OptimizerSpec(
                name="adam",
                learning_rate=lr,
                momentum=spec.momentum,
                nesterov=spec.nesterov,
                weight_decay=spec.weight_decay,
                mixed_precision=spec.mixed_precision,
                clipnorm=spec.clipnorm,
                clipvalue=spec.clipvalue,
            )
            return build_keras_optimizer(fallback, context=context)
        if hasattr(tf.keras.optimizers, "AdamW"):
            return tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=spec.weight_decay, **common_kwargs)
        experimental = getattr(tf.keras.optimizers, "experimental", None)
        if experimental is not None and hasattr(experimental, "AdamW"):
            return experimental.AdamW(learning_rate=lr, weight_decay=spec.weight_decay, **common_kwargs)
        fallback = OptimizerSpec(
            name="adam",
            learning_rate=lr,
            momentum=spec.momentum,
            nesterov=spec.nesterov,
            weight_decay=spec.weight_decay,
            mixed_precision=spec.mixed_precision,
            clipnorm=spec.clipnorm,
            clipvalue=spec.clipvalue,
        )
        return build_keras_optimizer(fallback, context=context)

    if name == "sgd":
        kwargs = dict(learning_rate=lr, momentum=spec.momentum, nesterov=spec.nesterov, **common_kwargs)
        if legacy is not None and hasattr(legacy, "SGD"):
            return legacy.SGD(**kwargs)
        return tf.keras.optimizers.SGD(**kwargs)

    if name == "rmsprop":
        kwargs = dict(learning_rate=lr, momentum=spec.momentum, **common_kwargs)
        if legacy is not None and hasattr(legacy, "RMSprop"):
            return legacy.RMSprop(**kwargs)
        return tf.keras.optimizers.RMSprop(**kwargs)

    if name == "adagrad":
        kwargs = dict(learning_rate=lr, **common_kwargs)
        if legacy is not None and hasattr(legacy, "Adagrad"):
            return legacy.Adagrad(**kwargs)
        return tf.keras.optimizers.Adagrad(**kwargs)

    raise ValueError(f"Optimizador no soportado en {context}: {spec.name!r}")


def get_optimizer_learning_rate(optimizer: tf.keras.optimizers.Optimizer) -> float:
    lr = optimizer.learning_rate
    if hasattr(lr, "numpy"):
        return float(lr.numpy())
    return float(tf.keras.backend.get_value(lr))


def set_optimizer_learning_rate(optimizer: tf.keras.optimizers.Optimizer, value: float) -> None:
    lr = optimizer.learning_rate
    if hasattr(lr, "assign"):
        lr.assign(value)
    else:
        tf.keras.backend.set_value(lr, value)


@dataclass
class ReduceLrPlateauState:
    best: Optional[float] = None
    bad_epochs: int = 0


def apply_reduce_lr_on_plateau(
    *,
    enabled: bool,
    optimizer: tf.keras.optimizers.Optimizer,
    current_metric: float,
    state: ReduceLrPlateauState,
    factor: float,
    patience: int,
    min_learning_rate: float,
    mode: str = "min",
    min_delta: float = 0.0,
) -> bool:
    if not enabled:
        return False
    if factor >= 1.0:
        raise ValueError("ReduceLROnPlateau requiere factor < 1.0")
    if patience < 1:
        raise ValueError("ReduceLROnPlateau requiere patience >= 1")

    improved = False
    if state.best is None:
        improved = True
    elif mode == "min":
        improved = current_metric < (state.best - min_delta)
    else:
        improved = current_metric > (state.best + min_delta)

    if improved:
        state.best = float(current_metric)
        state.bad_epochs = 0
        return False

    state.bad_epochs += 1
    if state.bad_epochs < patience:
        return False

    current_lr = get_optimizer_learning_rate(optimizer)
    new_lr = max(current_lr * factor, min_learning_rate)
    state.bad_epochs = 0
    if new_lr >= current_lr - 1e-15:
        return False
    set_optimizer_learning_rate(optimizer, new_lr)
    return True



def _optimizer_variables(optimizer: tf.keras.optimizers.Optimizer) -> List[tf.Variable]:
    variables_attr = getattr(optimizer, "variables", None)
    if callable(variables_attr):
        try:
            return list(variables_attr())
        except TypeError:
            pass
    if variables_attr is not None:
        return list(variables_attr)
    weights_attr = getattr(optimizer, "weights", None)
    if weights_attr is not None:
        return list(weights_attr)
    return []


def ensure_optimizer_state_initialized(optimizer: tf.keras.optimizers.Optimizer, variables: Iterable[tf.Variable]) -> None:
    variables = list(variables)
    if not variables:
        return
    build = getattr(optimizer, "build", None)
    if callable(build):
        try:
            build(variables)
            return
        except Exception:
            pass
    legacy_create = getattr(optimizer, "_create_all_weights", None)
    if callable(legacy_create):
        try:
            legacy_create(variables)
            return
        except Exception:
            pass
    try:
        zero_grads = [tf.zeros_like(variable) for variable in variables]
        optimizer.apply_gradients(zip(zero_grads, variables))
    except Exception:
        pass


def _optimizer_weight_values(optimizer: tf.keras.optimizers.Optimizer) -> List[np.ndarray]:
    get_weights = getattr(optimizer, "get_weights", None)
    if callable(get_weights):
        try:
            return [np.asarray(weight) for weight in get_weights()]
        except Exception:
            pass
    return [np.asarray(variable.numpy()) for variable in _optimizer_variables(optimizer)]


def save_optimizer_state(state_path: Union[str, Path], optimizer: tf.keras.optimizers.Optimizer) -> Path:
    state_path = Path(state_path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "optimizer_class": optimizer.__class__.__name__,
        "weights": _optimizer_weight_values(optimizer),
    }
    with state_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return state_path


def restore_optimizer_state(
    state_path: Union[str, Path],
    optimizer: tf.keras.optimizers.Optimizer,
    variables: Iterable[tf.Variable],
) -> bool:
    state_path = Path(state_path)
    if not state_path.exists():
        return False
    with state_path.open("rb") as handle:
        payload = pickle.load(handle)
    weights = payload.get("weights")
    if not isinstance(weights, list):
        return False
    ensure_optimizer_state_initialized(optimizer, variables)
    set_weights = getattr(optimizer, "set_weights", None)
    if callable(set_weights):
        try:
            set_weights(weights)
            return True
        except Exception:
            pass
    optimizer_variables = _optimizer_variables(optimizer)
    if len(optimizer_variables) != len(weights):
        return False
    try:
        for variable, value in zip(optimizer_variables, weights):
            variable.assign(value)
    except Exception:
        return False
    return True
