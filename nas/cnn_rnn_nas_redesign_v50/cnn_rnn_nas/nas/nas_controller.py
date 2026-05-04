from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .nas_search_space import SearchSpace
from ..common.training import (
    ReduceLrPlateauState,
    OptimizerSpec,
    apply_reduce_lr_on_plateau,
    build_keras_optimizer,
    get_optimizer_learning_rate,
)


def _tensorflow():
    from ..common.runtime import _tensorflow  # lazy import

    return _tensorflow()


tf = _tensorflow()


@dataclass(frozen=True)
class ControllerBatch:
    x: np.ndarray
    y_tokens: np.ndarray
    advantages: np.ndarray


@dataclass(frozen=True)
class ControllerTrainHistory:
    history: Dict[str, List[float]]


@dataclass(frozen=True)
class ControllerSampleResult:
    sequences: List[List[int]]
    guided_attempts: int
    fallback_attempts: int
    duplicate_hits: int
    used_fallback: bool

    @property
    def total_attempts(self) -> int:
        return int(self.guided_attempts + self.fallback_attempts)


class SearchController:
    def __init__(self, nas_config: Any, space: SearchSpace) -> None:
        self.nas_config = nas_config
        self.space = space
        self.decision_steps = self.space.sequence_length
        self.controller_steps = self.space.controller_sequence_length
        self.model = self._build_model()
        self.optimizer = build_keras_optimizer(
            OptimizerSpec(
                name=self.nas_config.optimizer.name,
                learning_rate=self.nas_config.optimizer.learning_rate,
                momentum=self.nas_config.optimizer.momentum,
                nesterov=self.nas_config.optimizer.nesterov,
                weight_decay=self.nas_config.optimizer.weight_decay,
                mixed_precision=False,
            ),
            context="nas_controller",
        )
        self.train_step = self._build_train_step()
        self.plateau_state = ReduceLrPlateauState()

    def _build_model(self) -> tf.keras.Model:
        input_tokens = tf.keras.Input(shape=(self.controller_steps,), dtype=tf.int32, name="controller_input")
        x = tf.keras.layers.Embedding(len(self.space.controller_vocab), self.nas_config.model.lstm_dim, mask_zero=True)(input_tokens)
        x = tf.keras.layers.LSTM(self.nas_config.model.lstm_dim, return_sequences=True)(x)
        output = tf.keras.layers.Dense(
            len(self.space.controller_vocab),
            activation="softmax",
            name="controller_output",
        )(x)
        return tf.keras.Model(inputs=input_tokens, outputs=output, name="nas_controller")

    def _build_train_step(self):
        @tf.function
        def train_step(x: tf.Tensor, y_tokens: tf.Tensor, advantages: tf.Tensor) -> tf.Tensor:
            x = tf.cast(x, tf.int32)
            y_tokens = tf.cast(y_tokens, tf.int32)
            advantages = tf.cast(advantages, tf.float32)
            with tf.GradientTape() as tape:
                output = self.model(x, training=True)
                output = tf.cast(output, tf.float32)

                batch_size = tf.shape(y_tokens)[0]
                time_steps = tf.shape(y_tokens)[1]
                batch_indices = tf.tile(tf.range(batch_size, dtype=tf.int32)[:, None], [1, time_steps])
                time_indices = tf.tile(tf.range(time_steps, dtype=tf.int32)[None, :], [batch_size, 1])
                gather_indices = tf.stack([batch_indices, time_indices, y_tokens], axis=-1)
                action_probs = tf.gather_nd(output, gather_indices)
                action_probs = tf.clip_by_value(action_probs, 1e-8, 1.0)

                loss = -tf.math.log(action_probs) * advantages
                loss = tf.reduce_mean(loss)
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            return loss

        return train_step

    def _allowed_ids(self, step: int, partial: List[int]) -> List[int]:
        return self.space.controller_allowed_ids(step, partial)

    @staticmethod
    def _sample_from_logits(valid_ids: List[int], step_probs: np.ndarray) -> int:
        probs = np.asarray(step_probs[valid_ids], dtype=np.float64)
        probs = np.maximum(probs, 0.0)
        total = probs.sum()
        if total <= 0:
            probs = np.full(len(valid_ids), 1.0 / len(valid_ids), dtype=np.float64)
        else:
            probs = probs / total
            probs[-1] += 1.0 - probs.sum()
        return int(np.random.choice(valid_ids, p=probs))

    @staticmethod
    def _sample_uniform(valid_ids: List[int]) -> int:
        return int(np.random.choice(valid_ids))

    def _guided_sequence(self) -> List[int]:
        decisions: List[int] = []
        prefix: List[int] = [self.space.START_TOKEN_ID]
        for step in range(self.controller_steps):
            controller_input = np.asarray([self.space.pad_prefix(prefix)], dtype=np.int32)
            predictions = self.model.predict(controller_input, verbose=0)[0]
            time_index = len(prefix) - 1
            step_probs = predictions[time_index]
            allowed_ids = self._allowed_ids(step, decisions)
            next_token = self._sample_from_logits(allowed_ids, step_probs)
            prefix.append(next_token)
            if step < self.decision_steps:
                decisions.append(next_token)
            elif next_token != self.space.END_TOKEN_ID:
                raise ValueError("El controlador intentó cerrar la secuencia con un token distinto de END")
        return decisions

    def _uniform_sequence(self) -> List[int]:
        decisions: List[int] = []
        for step in range(self.controller_steps):
            allowed_ids = self._allowed_ids(step, decisions)
            next_token = self._sample_uniform(allowed_ids)
            if step < self.decision_steps:
                decisions.append(next_token)
            elif next_token != self.space.END_TOKEN_ID:
                raise ValueError("El sampler uniforme intentó cerrar la secuencia con un token distinto de END")
        return decisions

    def sample_sequences(self, number_of_samples: int, seen: Optional[Set[Tuple[int, ...]]] = None) -> ControllerSampleResult:
        if seen is None:
            seen = set()
        samples: List[List[int]] = []
        duplicate_hits = 0
        guided_attempts = 0
        fallback_attempts = 0
        max_attempts = max(
            int(number_of_samples) * int(self.nas_config.sampling_attempts_multiplier),
            int(self.nas_config.sampling_attempts_minimum),
        )

        while len(samples) < number_of_samples and guided_attempts < max_attempts:
            guided_attempts += 1
            decisions = self._guided_sequence()
            sample_key = tuple(decisions)
            if sample_key in seen:
                duplicate_hits += 1
                continue
            seen.add(sample_key)
            samples.append(decisions)

        used_fallback = False
        if len(samples) < number_of_samples:
            used_fallback = True
            fallback_budget = max(max_attempts, int(number_of_samples) * int(self.nas_config.sampling_attempts_multiplier) * 2)
            while len(samples) < number_of_samples and fallback_attempts < fallback_budget:
                fallback_attempts += 1
                decisions = self._uniform_sequence()
                sample_key = tuple(decisions)
                if sample_key in seen:
                    duplicate_hits += 1
                    continue
                seen.add(sample_key)
                samples.append(decisions)

        return ControllerSampleResult(
            sequences=samples,
            guided_attempts=guided_attempts,
            fallback_attempts=fallback_attempts,
            duplicate_hits=duplicate_hits,
            used_fallback=used_fallback,
        )

    def prepare_training_batch(self, sequences: List[List[int]], advantages: List[float]) -> ControllerBatch:
        x = np.asarray([self.space.teacher_forcing_input(sequence) for sequence in sequences], dtype=np.int32)
        y_tokens = np.asarray([self.space.teacher_forcing_target(sequence) for sequence in sequences], dtype=np.int32)
        repeated = np.repeat(np.asarray(advantages, dtype=np.float32)[:, None], self.controller_steps, axis=1)
        repeated[:, -1] = 0.0
        return ControllerBatch(x=x, y_tokens=y_tokens, advantages=repeated)

    def train(self, batch: ControllerBatch) -> ControllerTrainHistory:
        x = tf.convert_to_tensor(batch.x, dtype=tf.int32)
        y_tokens = tf.convert_to_tensor(batch.y_tokens, dtype=tf.int32)
        advantages = tf.convert_to_tensor(batch.advantages, dtype=tf.float32)

        loss_history: List[float] = []
        lr_history: List[float] = []
        reduced_history: List[float] = []
        for _ in range(self.nas_config.controller_training_epochs):
            loss = float(self.train_step(x, y_tokens, advantages).numpy())
            loss_history.append(loss)
            reduced = apply_reduce_lr_on_plateau(
                enabled=self.nas_config.controller_reduce_lr_on_plateau,
                optimizer=self.optimizer,
                current_metric=loss,
                state=self.plateau_state,
                factor=self.nas_config.controller_reduce_lr_factor,
                patience=self.nas_config.controller_reduce_lr_patience,
                min_learning_rate=self.nas_config.controller_min_learning_rate,
                mode="min",
            )
            reduced_history.append(1.0 if reduced else 0.0)
            lr_history.append(get_optimizer_learning_rate(self.optimizer))
        return ControllerTrainHistory(
            history={
                "loss": loss_history,
                "learning_rate": lr_history,
                "learning_rate_reduced": reduced_history,
            }
        )

    @property
    def current_learning_rate(self) -> float:
        return get_optimizer_learning_rate(self.optimizer)
