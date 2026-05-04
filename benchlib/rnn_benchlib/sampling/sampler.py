from __future__ import annotations

import hashlib
import itertools
import json
import random
from typing import Iterable, List, Optional, Sequence, Set

from rnn_benchlib.config.schemas import FeatureSpec, ModelSpec, SearchSpace
from rnn_benchlib.sampling.validators import filter_valid_specs


def model_spec_to_stable_key(spec: ModelSpec, feature_spec: FeatureSpec) -> str:
    payload = {
        "spec": spec.as_key_dict(),
        "feature_spec": {
            "feature_dim": feature_spec.feature_dim,
            "video_steps": feature_spec.video_steps,
        },
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def model_spec_to_id(spec: ModelSpec, feature_spec: FeatureSpec, prefix: str = "rnn") -> str:
    stable_key = model_spec_to_stable_key(spec=spec, feature_spec=feature_spec)
    digest = hashlib.sha1(stable_key.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def enumerate_search_space(space: SearchSpace, num_classes: int) -> List[ModelSpec]:
    specs: List[ModelSpec] = []
    for values in itertools.product(
        space.layers,
        space.rnn,
        space.units_0,
        space.units_1,
        space.units_2,
        space.direction,
        space.memory_mode,
        space.seq,
        space.head_units,
        space.video_decision,
        space.video_decision_input,
    ):
        (
            layers,
            rnn,
            units_0,
            units_1,
            units_2,
            direction,
            memory_mode,
            seq,
            head_units,
            video_decision,
            video_decision_input,
        ) = values
        specs.append(
            ModelSpec(
                layers=layers,
                rnn=rnn,
                units_0=units_0,
                units_1=units_1,
                units_2=units_2,
                direction=direction,
                memory_mode=memory_mode,
                seq=seq,
                head_units=head_units,
                num_classes=num_classes,
                video_decision=video_decision,
                video_decision_input=video_decision_input,
            )
        )
    return specs


def enumerate_valid_specs(space: SearchSpace, num_classes: int, video_steps: int = 36) -> List[ModelSpec]:
    return filter_valid_specs(enumerate_search_space(space=space, num_classes=num_classes), video_steps=video_steps)


def filter_out_existing_specs(
    specs: Iterable[ModelSpec],
    feature_spec: FeatureSpec,
    existing_keys: Optional[Set[str]] = None,
) -> List[ModelSpec]:
    existing_keys = existing_keys or set()
    result: List[ModelSpec] = []
    for spec in specs:
        key = model_spec_to_stable_key(spec=spec, feature_spec=feature_spec)
        if key not in existing_keys:
            result.append(spec)
    return result


def sample_model_specs(
    count: int,
    seed: int,
    space: SearchSpace,
    feature_spec: FeatureSpec,
    num_classes: int,
    existing_keys: Optional[Set[str]] = None,
) -> List[ModelSpec]:
    rng = random.Random(seed)
    valid_specs = enumerate_valid_specs(space=space, num_classes=num_classes, video_steps=feature_spec.video_steps)
    remaining_specs = filter_out_existing_specs(specs=valid_specs, feature_spec=feature_spec, existing_keys=existing_keys)
    if not remaining_specs:
        return []
    rng.shuffle(remaining_specs)
    if count >= len(remaining_specs):
        return remaining_specs
    return remaining_specs[:count]


def summarize_sampling_pool(
    space: SearchSpace,
    feature_spec: FeatureSpec,
    num_classes: int,
    existing_keys: Optional[Set[str]] = None,
) -> dict:
    all_specs = enumerate_search_space(space=space, num_classes=num_classes)
    valid_specs = enumerate_valid_specs(space=space, num_classes=num_classes, video_steps=feature_spec.video_steps)
    remaining_specs = filter_out_existing_specs(specs=valid_specs, feature_spec=feature_spec, existing_keys=existing_keys)
    return {
        "search_space_total": len(all_specs),
        "valid_total": len(valid_specs),
        "existing_total": len(existing_keys or set()),
        "remaining_total": len(remaining_specs),
    }


def sort_specs_deterministically(specs: Sequence[ModelSpec], feature_spec: FeatureSpec) -> List[ModelSpec]:
    return sorted(specs, key=lambda spec: model_spec_to_stable_key(spec=spec, feature_spec=feature_spec))
