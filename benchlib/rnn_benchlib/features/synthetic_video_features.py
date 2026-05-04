from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from rnn_benchlib.config.schemas import FeatureSpec, ModelSpec


@dataclass(frozen=True)
class VideoFeatureBatch:
    """
    Contenedor simple para features sintéticas.

    videos:
      shape = [num_videos, video_steps, feature_dim]

    clips:
      shape = [num_videos, clips_per_video, seq, feature_dim]
    """
    videos: np.ndarray
    clips: np.ndarray
    num_videos: int
    video_steps: int
    feature_dim: int
    seq: int
    clips_per_video: int

    def to_dict(self) -> Dict[str, int]:
        return {
            "num_videos": self.num_videos,
            "video_steps": self.video_steps,
            "feature_dim": self.feature_dim,
            "seq": self.seq,
            "clips_per_video": self.clips_per_video,
        }


def generate_synthetic_videos(
    num_videos: int,
    feature_spec: FeatureSpec,
    seed: int,
    distribution: str = "normal",
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """
    Genera un tensor de vídeos sintéticos con shape:
      [num_videos, video_steps, feature_dim]
    """
    rng = np.random.default_rng(seed)

    shape = (num_videos, feature_spec.video_steps, feature_spec.feature_dim)

    if distribution == "normal":
        videos = rng.standard_normal(size=shape, dtype=dtype)
    elif distribution == "uniform":
        videos = rng.uniform(low=-1.0, high=1.0, size=shape).astype(dtype)
    else:
        raise ValueError(f"distribution no soportada: {distribution}")

    return videos.astype(dtype, copy=False)


def split_videos_into_clips(
    videos: np.ndarray,
    seq: int,
) -> np.ndarray:
    """
    Convierte vídeos [N, video_steps, feature_dim] en clips:
      [N, clips_per_video, seq, feature_dim]

    Requiere que video_steps sea múltiplo de seq.
    """
    if videos.ndim != 3:
        raise ValueError(
            f"Se esperaba videos.ndim == 3, recibido shape={videos.shape}"
        )

    num_videos, video_steps, feature_dim = videos.shape

    if seq <= 0:
        raise ValueError(f"seq debe ser > 0, recibido: {seq}")

    if video_steps % seq != 0:
        raise ValueError(
            f"video_steps={video_steps} no es múltiplo de seq={seq}"
        )

    clips_per_video = video_steps // seq
    return videos.reshape(num_videos, clips_per_video, seq, feature_dim)


def generate_synthetic_video_batch(
    num_videos: int,
    feature_spec: FeatureSpec,
    model_spec: ModelSpec,
    seed: int,
    distribution: str = "normal",
    dtype: np.dtype = np.float32,
) -> VideoFeatureBatch:
    """
    Genera vídeos sintéticos y su vista ya recortada en clips.
    """
    videos = generate_synthetic_videos(
        num_videos=num_videos,
        feature_spec=feature_spec,
        seed=seed,
        distribution=distribution,
        dtype=dtype,
    )

    clips = split_videos_into_clips(
        videos=videos,
        seq=model_spec.seq,
    )

    return VideoFeatureBatch(
        videos=videos,
        clips=clips,
        num_videos=num_videos,
        video_steps=feature_spec.video_steps,
        feature_dim=feature_spec.feature_dim,
        seq=model_spec.seq,
        clips_per_video=model_spec.clips_per_video(feature_spec.video_steps),
    )


def load_video_features_npy(path: str) -> np.ndarray:
    """
    Carga features precomputadas desde .npy.

    Formato esperado:
      [num_videos, video_steps, feature_dim]
    """
    videos = np.load(path)
    if videos.ndim != 3:
        raise ValueError(
            f"El .npy debe tener shape [num_videos, video_steps, feature_dim], "
            f"recibido {videos.shape}"
        )
    return videos.astype(np.float32, copy=False)


def load_video_batch_from_npy(
    path: str,
    model_spec: ModelSpec,
) -> VideoFeatureBatch:
    """
    Carga vídeos desde .npy y genera su vista por clips.
    """
    videos = load_video_features_npy(path)
    clips = split_videos_into_clips(videos=videos, seq=model_spec.seq)

    num_videos, video_steps, feature_dim = videos.shape

    return VideoFeatureBatch(
        videos=videos,
        clips=clips,
        num_videos=num_videos,
        video_steps=video_steps,
        feature_dim=feature_dim,
        seq=model_spec.seq,
        clips_per_video=model_spec.clips_per_video(video_steps),
    )


def get_single_video_clips(
    batch: VideoFeatureBatch,
    video_index: int,
) -> np.ndarray:
    """
    Devuelve los clips de un único vídeo:
      [clips_per_video, seq, feature_dim]
    """
    if video_index < 0 or video_index >= batch.num_videos:
        raise IndexError(
            f"video_index fuera de rango: {video_index}, num_videos={batch.num_videos}"
        )
    return batch.clips[video_index]


def get_single_clip_input(
    batch: VideoFeatureBatch,
    video_index: int,
    clip_index: int,
) -> np.ndarray:
    """
    Devuelve un único clip con batch=1:
      [1, seq, feature_dim]
    """
    video_clips = get_single_video_clips(batch=batch, video_index=video_index)

    if clip_index < 0 or clip_index >= video_clips.shape[0]:
        raise IndexError(
            f"clip_index fuera de rango: {clip_index}, clips_per_video={video_clips.shape[0]}"
        )

    clip = video_clips[clip_index]
    return np.expand_dims(clip, axis=0).astype(np.float32, copy=False)


def describe_video_tensor(videos: np.ndarray) -> Dict[str, object]:
    return {
        "shape": list(videos.shape),
        "dtype": str(videos.dtype),
        "min": float(np.min(videos)),
        "max": float(np.max(videos)),
        "mean": float(np.mean(videos)),
        "std": float(np.std(videos)),
    }