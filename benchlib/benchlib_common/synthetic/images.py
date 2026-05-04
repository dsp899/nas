from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SyntheticVideoImageSpec:
    num_videos: int = 8
    frames_per_video: int = 16
    image_size: int = 224
    channels: int = 3
    seed: int = 1234
    distribution: str = "normal"


def generate_synthetic_video_frames(spec: SyntheticVideoImageSpec) -> np.ndarray:
    rng = np.random.default_rng(spec.seed)
    shape = (spec.num_videos, spec.frames_per_video, spec.image_size, spec.image_size, spec.channels)
    if spec.distribution == "uniform":
        return rng.uniform(0.0, 255.0, size=shape).astype(np.float32)
    return rng.normal(127.5, 50.0, size=shape).clip(0.0, 255.0).astype(np.float32)
