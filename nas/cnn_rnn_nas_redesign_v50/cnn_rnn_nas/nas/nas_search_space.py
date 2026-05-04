
from dataclasses import dataclass

from ..config.nas_config import CANONICAL_SEARCH_DIMENSIONS, NasSearchSpaceConfig
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class SearchSpace:
    config: NasSearchSpaceConfig

    PAD_TOKEN_ID: int = 0
    START_TOKEN_ID: int = 1
    END_TOKEN_ID: int = 2

    def __post_init__(self) -> None:
        offsets: Dict[str, Dict[int, Any]] = {}
        start = self.END_TOKEN_ID + 1
        for name in self.dimensions:
            values = self.config.options(name)
            ids = list(range(start, start + len(values)))
            offsets[name] = dict(zip(ids, values))
            start += len(values)

        decision_vocab = {key: value for group in offsets.values() for key, value in group.items()}
        controller_vocab = {
            self.PAD_TOKEN_ID: "<PAD>",
            self.START_TOKEN_ID: "<START>",
            self.END_TOKEN_ID: "<END>",
            **decision_vocab,
        }

        object.__setattr__(self, "groups", offsets)
        object.__setattr__(self, "vocab", decision_vocab)
        object.__setattr__(self, "controller_vocab", controller_vocab)
        object.__setattr__(self, "sequence_length", len(self.dimensions))
        object.__setattr__(self, "controller_sequence_length", len(self.dimensions) + 1)
        object.__setattr__(self, "controller_input_length", len(self.dimensions) + 1)

    @property
    def dimensions(self) -> Tuple[str, ...]:
        return self.config.variable_dimensions

    @property
    def fixed_dimensions(self) -> Tuple[str, ...]:
        return self.config.fixed_dimensions

    @property
    def all_dimensions(self) -> Tuple[str, ...]:
        return CANONICAL_SEARCH_DIMENSIONS

    def options(self, dimension: str) -> Tuple[Any, ...]:
        return self.config.options(dimension)

    def encode(self, decoded_by_dimension: Dict[str, Any]) -> List[int]:
        encoded: List[int] = []
        for dimension in self.dimensions:
            inverse = {value: key for key, value in self.groups[dimension].items()}
            encoded.append(inverse[decoded_by_dimension[dimension]])
        return encoded

    def decode(self, encoded: List[int]) -> List[Any]:
        return [self.vocab[item] for item in encoded]

    def decode_dict(self, encoded: List[int]) -> Dict[str, Any]:
        return {dimension: self.vocab[token] for dimension, token in zip(self.dimensions, encoded)}

    def teacher_forcing_input(self, encoded_decisions: List[int]) -> List[int]:
        return [self.START_TOKEN_ID, *encoded_decisions]

    def teacher_forcing_target(self, encoded_decisions: List[int]) -> List[int]:
        return [*encoded_decisions, self.END_TOKEN_ID]

    def pad_prefix(self, prefix: List[int]) -> List[int]:
        if len(prefix) > self.controller_input_length:
            raise ValueError("El prefijo del controlador excede la longitud máxima esperada")
        return prefix + [self.PAD_TOKEN_ID] * (self.controller_input_length - len(prefix))

    def controller_allowed_ids(self, step: int, partial_tokens: List[int]) -> List[int]:
        if step < self.sequence_length:
            return self.allowed_ids(step, partial_tokens)
        if step == self.sequence_length:
            return [self.END_TOKEN_ID]
        raise IndexError(f"Paso del controlador fuera de rango: {step}")

    def _layer_context(self, partial: Dict[str, Any]) -> Optional[int]:
        if "layers" in partial:
            return int(partial["layers"])
        if "layers" in self.fixed_dimensions:
            return int(self.config.fixed_value("layers"))
        return None

    def allowed_ids(self, step: int, partial_tokens: List[int]) -> List[int]:
        dimension = self.dimensions[step]
        partial = self.decode_dict(partial_tokens)

        if dimension == "units_1":
            chosen_layers = self._layer_context(partial)
            if chosen_layers == 1:
                return [token for token, value in self.groups[dimension].items() if int(value) == 0]
        if dimension == "units_2":
            chosen_layers = self._layer_context(partial)
            if chosen_layers is not None and chosen_layers <= 2:
                return [token for token, value in self.groups[dimension].items() if int(value) == 0]
        if dimension == "video_decision_input":
            chosen_video_decision = partial.get("video_decision")
            if chosen_video_decision is not None and str(chosen_video_decision) != "average":
                return [token for token, value in self.groups[dimension].items() if str(value) == "clip_logits"]
        return list(self.groups[dimension].keys())

    def count_valid_sequences(self) -> int:
        memo: Dict[Tuple[int, Tuple[int, ...]], int] = {}

        def _count(step: int, partial_tokens: Tuple[int, ...]) -> int:
            key = (step, partial_tokens)
            if key in memo:
                return memo[key]
            if step >= self.sequence_length:
                memo[key] = 1
                return 1
            total = 0
            for token_id in self.allowed_ids(step, list(partial_tokens)):
                total += _count(step + 1, partial_tokens + (token_id,))
            memo[key] = total
            return total

        return _count(0, tuple())
