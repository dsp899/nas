from __future__ import annotations

from typing import Iterable, List, Tuple

from rnn_benchlib.config.schemas import ModelSpec


_SUPPORTED_EMBEDDING_DECISIONS = {"average"}


def validate_model_spec(spec: ModelSpec, video_steps: int = 36) -> Tuple[bool, List[str]]:
    errors: List[str] = []

    if spec.layers not in (1, 2, 3):
        errors.append(f"layers inválido: {spec.layers}")

    if spec.units_0 <= 0:
        errors.append(f"units_0 debe ser > 0, recibido: {spec.units_0}")

    if spec.layers == 1:
        if spec.units_1 != 0:
            errors.append(f"units_1 debe ser 0 cuando layers=1, recibido: {spec.units_1}")
        if spec.units_2 != 0:
            errors.append(f"units_2 debe ser 0 cuando layers=1, recibido: {spec.units_2}")

    if spec.layers == 2:
        if spec.units_1 <= 0:
            errors.append(f"units_1 debe ser > 0 cuando layers=2, recibido: {spec.units_1}")
        if spec.units_2 != 0:
            errors.append(f"units_2 debe ser 0 cuando layers=2, recibido: {spec.units_2}")

    if spec.layers == 3:
        if spec.units_1 <= 0:
            errors.append(f"units_1 debe ser > 0 cuando layers=3, recibido: {spec.units_1}")
        if spec.units_2 <= 0:
            errors.append(f"units_2 debe ser > 0 cuando layers=3, recibido: {spec.units_2}")

    if spec.seq <= 0:
        errors.append(f"seq debe ser > 0, recibido: {spec.seq}")
    elif video_steps % spec.seq != 0:
        errors.append(f"seq={spec.seq} no divide video_steps={video_steps} exactamente")

    if spec.direction not in ("unidirectional", "bidirectional"):
        errors.append(f"direction inválido: {spec.direction}")

    if spec.rnn not in ("lstm", "gru"):
        errors.append(f"rnn inválido: {spec.rnn}")

    if spec.memory_mode not in ("none", "carry_forward"):
        errors.append(f"memory_mode inválido: {spec.memory_mode}")

    if spec.video_decision_input == "clip_embeddings" and spec.video_decision not in _SUPPORTED_EMBEDDING_DECISIONS:
        errors.append(
            f"video_decision={spec.video_decision} no está soportado con clip_embeddings; usa una de {sorted(_SUPPORTED_EMBEDDING_DECISIONS)}"
        )

    if spec.head_units <= 0:
        errors.append(f"head_units debe ser > 0, recibido: {spec.head_units}")

    if spec.num_classes <= 1:
        errors.append(f"num_classes debe ser > 1, recibido: {spec.num_classes}")

    return (len(errors) == 0, errors)


def assert_valid_model_spec(spec: ModelSpec, video_steps: int = 36) -> None:
    ok, errors = validate_model_spec(spec=spec, video_steps=video_steps)
    if not ok:
        raise ValueError("ModelSpec inválido:\n- " + "\n- ".join(errors))


def filter_valid_specs(specs: Iterable[ModelSpec], video_steps: int = 36) -> List[ModelSpec]:
    valid_specs: List[ModelSpec] = []
    for spec in specs:
        ok, _ = validate_model_spec(spec=spec, video_steps=video_steps)
        if ok:
            valid_specs.append(spec)
    return valid_specs
