from __future__ import annotations

from typing import Dict, Iterable, Mapping

import tensorflow as tf


class MultiTargetLatencyLoss:
    def __init__(self, target_names: Iterable[str], wall_weight: float = 1.0, sum_weight: float = 0.4, part_weight: float = 0.2, huber_weight: float = 0.2) -> None:
        self.target_names = list(target_names)
        self.weights = {
            "latency_clip_e2e_wall_ms": wall_weight,
            "latency_clip_e2e_sum_ms": sum_weight,
            "latency_clip_encoder_ms": part_weight,
            "latency_clip_bridge_ms": part_weight,
            "latency_clip_head_ms": part_weight,
        }
        self.huber_weight = huber_weight
        self.huber = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.NONE, delta=0.25)

    def __call__(
        self,
        predictions: Mapping[str, tf.Tensor],
        targets: Mapping[str, tf.Tensor],
        sample_weight: tf.Tensor | None = None,
        return_metrics: bool = True,
    ) -> tuple[tf.Tensor, Dict[str, float]]:
        total = tf.constant(0.0, dtype=tf.float32)
        metrics: Dict[str, float] = {}
        eager = bool(tf.executing_eagerly())
        for name in self.target_names:
            pred = tf.cast(tf.reshape(predictions[name], [-1]), tf.float32)
            target = tf.cast(tf.reshape(targets[name], [-1]), tf.float32)
            rel = tf.abs(pred - target) / tf.maximum(target, 1e-3)
            huber = self.huber(tf.math.log1p(target), tf.math.log1p(pred))
            loss_vec = rel + self.huber_weight * tf.reshape(huber, [-1])
            if sample_weight is not None:
                sw = tf.cast(tf.reshape(sample_weight, [-1]), tf.float32)
                loss_vec = loss_vec * sw
            loss = tf.reduce_mean(loss_vec) * float(self.weights.get(name, 1.0))
            total = total + loss
            if return_metrics and eager:
                metrics[f"loss_{name}"] = float(loss.numpy())
        if return_metrics and eager:
            metrics["loss_total"] = float(total.numpy())
        return total, metrics


def compute_multitask_loss(predictions: Mapping[str, tf.Tensor], encoded_sample: Mapping[str, object]) -> tf.Tensor:
    """Compute multitask latency loss from one encoded sample.

    Works both in eager mode and inside ``tf.function`` by avoiding Python
    iteration over symbolic tensors and by not materializing numpy values.
    """
    target_names_obj = encoded_sample["target_names"]
    if isinstance(target_names_obj, tf.Tensor):
        target_names = [
            name.decode("utf-8") if isinstance(name, (bytes, bytearray)) else str(name)
            for name in (tf.get_static_value(target_names_obj) or [])
        ]
    else:
        target_names = list(target_names_obj)

    targets_tensor = tf.cast(tf.reshape(encoded_sample["targets"], [-1]), tf.float32)
    target_values = tf.unstack(targets_tensor, num=len(target_names))
    targets = {name: tensor for name, tensor in zip(target_names, target_values)}
    sample_weight = encoded_sample.get("sample_weight") if isinstance(encoded_sample, Mapping) else None
    loss_fn = MultiTargetLatencyLoss(target_names=target_names)
    total, _ = loss_fn(predictions=predictions, targets=targets, sample_weight=sample_weight, return_metrics=False)
    return total
