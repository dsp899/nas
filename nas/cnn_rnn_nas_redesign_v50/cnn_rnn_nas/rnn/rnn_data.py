import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Generator, Optional, Tuple

import numpy as np
import tensorflow as tf

from ..common.artifacts import ProjectPaths
from ..common.registries import CnnExperimentRegistry
from ..config.rnn_config import RnnDataConfig


TF_AUTOTUNE = tf.data.AUTOTUNE


@dataclass(frozen=True)
class SequenceMetadata:
    num_features: int
    num_classes: int
    sequence_size: int
    num_sequences_per_video: int
    train_videos: int
    val_videos: int
    test_videos: int
    train_samples: int
    val_samples: int
    test_samples: int
    train_batches: int
    val_batches: int
    test_batches: int


@dataclass
class DataBundle:
    train_ds: tf.data.Dataset
    val_ds: tf.data.Dataset
    test_ds: tf.data.Dataset
    metadata: SequenceMetadata


class SequenceRepository:
    def __init__(self, paths: ProjectPaths) -> None:
        self.paths = paths
        self.cnn_registry = CnnExperimentRegistry(paths.cnn_registry_path)

    def resolve_data_feature_source(self, data: RnnDataConfig) -> Tuple[RnnDataConfig, Dict[str, Any]]:
        explicit_training_signature = data.cnn_training_signature or None
        explicit_feature_signature = data.cnn_feature_export_signature or None
        if explicit_training_signature or explicit_feature_signature:
            record = self.cnn_registry.find_latest_completed_feature_export_for_rnn(
                data,
                training_signature=explicit_training_signature,
                feature_signature=explicit_feature_signature,
            )
            if not record:
                raise FileNotFoundError(
                    "No se encontró la exportación de features CNN solicitada explícitamente para el bloque RNN/NAS. "
                    f"cnn_training_signature={data.cnn_training_signature!r}, "
                    f"cnn_feature_export_signature={data.cnn_feature_export_signature!r}"
                )
            resolved = replace(
                data,
                cnn_training_signature=str(record["training_signature"]),
                cnn_feature_export_signature=str(record["feature_signature"]),
            )
            return resolved, record

        record = self.cnn_registry.find_latest_completed_feature_export_for_rnn(data)
        if not record:
            raise FileNotFoundError(
                "No se encontró ninguna exportación de features CNN compatible para el bloque RNN/NAS. "
                f"Esperado: cnn={data.cnn}, dataset={data.name}, split={data.split}, partition_mode={data.partition_mode}, "
                f"frames={data.frames}, image_size={data.image_size}, sampling={data.sampling}, resize_mode={data.resize_mode}. "
                "Genera primero las features con 'python3 run_cnn.py export_features' usando la misma configuración efectiva."
            )
        resolved = replace(
            data,
            cnn_training_signature=str(record["training_signature"]),
            cnn_feature_export_signature=str(record["feature_signature"]),
        )
        return resolved, record

    def _load_feature_arrays(self, mode: str, data: RnnDataConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        resolved_data, record = self.resolve_data_feature_source(data)
        feature_dir = Path(record["feature_dir"])
        paths = {
            "features": feature_dir / f"video_features_{mode}.npy",
            "labels": feature_dir / f"video_labels_{mode}.npy",
            "video_ids": feature_dir / f"video_ids_{mode}.npy",
        }
        missing = [str(path) for path in paths.values() if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "No se encontraron los artefactos de features esperados para el bloque RNN/NAS. "
                f"Faltan: {missing}. Vuelve a ejecutar 'python3 run_cnn.py export_features' con la misma configuración efectiva."
            )
        return (
            np.load(paths["features"], mmap_mode="r"),
            np.load(paths["labels"], mmap_mode="r"),
            np.load(paths["video_ids"], mmap_mode="r"),
        )

    @staticmethod
    def _non_overlapping_windows(sequence: np.ndarray, window_size: int) -> np.ndarray:
        windows = []
        max_start = len(sequence) - window_size + 1
        for start in range(0, max(1, max_start), window_size):
            end = start + window_size
            if end > len(sequence):
                break
            windows.append(sequence[start:end])
        if not windows:
            windows.append(sequence[:window_size])
        return np.stack(windows)

    @staticmethod
    def _window_videos(videos: np.ndarray, seq_length: int) -> np.ndarray:
        windows = [SequenceRepository._non_overlapping_windows(video, seq_length) for video in videos]
        return np.stack(windows).astype(np.float32) if windows else np.zeros((0, 1, seq_length, videos.shape[-1]), dtype=np.float32)

    def load_windowed_split(self, mode: str, data: RnnDataConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        videos, labels, video_ids = self._load_feature_arrays(mode, data)
        windowed = self._window_videos(videos, data.seq)
        label_view = labels[:, 0, :] if labels.ndim == 3 else labels
        id_view = video_ids[:, 0] if video_ids.ndim > 1 else video_ids
        return windowed, np.asarray(label_view, dtype=np.float32), np.asarray(id_view, dtype=np.int64)

    def make_bundle(self, data: RnnDataConfig, batch_size: int, seed: int = 1337) -> DataBundle:
        data, _ = self.resolve_data_feature_source(data)
        train_videos, train_labels, train_ids = self.load_windowed_split("train", data)
        val_videos, val_labels, val_ids = self.load_windowed_split("val", data)
        test_videos, test_labels, test_ids = self.load_windowed_split("test", data)

        num_features = int(train_videos.shape[-1])
        num_classes = int(train_labels.shape[-1])
        sequence_size = int(train_videos.shape[2])
        num_sequences_per_video = int(train_videos.shape[1])

        train_samples = int(train_videos.shape[0])
        val_samples = int(val_videos.shape[0])
        test_samples = int(test_videos.shape[0])
        train_batches = math.ceil(train_samples / batch_size)
        val_batches = math.ceil(val_samples / batch_size)
        test_batches = math.ceil(test_samples / batch_size)
        train_ds = self._build_video_train(train_videos, train_labels, train_ids, batch_size, seed)
        val_ds = self._build_video_test(val_videos, val_labels, val_ids, batch_size)
        test_ds = self._build_video_test(test_videos, test_labels, test_ids, batch_size)

        metadata = SequenceMetadata(
            num_features=num_features,
            num_classes=num_classes,
            sequence_size=sequence_size,
            num_sequences_per_video=num_sequences_per_video,
            train_videos=int(train_videos.shape[0]),
            val_videos=int(val_videos.shape[0]),
            test_videos=int(test_videos.shape[0]),
            train_samples=train_samples,
            val_samples=val_samples,
            test_samples=test_samples,
            train_batches=train_batches,
            val_batches=val_batches,
            test_batches=test_batches,
        )
        return DataBundle(train_ds=train_ds, val_ds=val_ds, test_ds=test_ds, metadata=metadata)

    @staticmethod
    def _video_batches(
        videos: np.ndarray,
        labels: np.ndarray,
        video_ids: np.ndarray,
        batch_size: int,
        *,
        shuffle: bool,
        seed: Optional[int] = None,
    ) -> Generator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], None, None]:
        indices = np.arange(len(labels))
        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            valid_count = len(batch_indices)
            if valid_count < batch_size:
                pad_count = batch_size - valid_count
                pad_indices = np.repeat(batch_indices[-1:], pad_count)
                batch_indices = np.concatenate([batch_indices, pad_indices])
            x_batch = np.asarray(videos[batch_indices], dtype=np.float32)
            y_batch = np.asarray(labels[batch_indices], dtype=np.float32)
            id_batch = np.asarray(video_ids[batch_indices], dtype=np.int64)
            sample_weight = np.concatenate([np.ones(valid_count, dtype=np.float32), np.zeros(batch_size - valid_count, dtype=np.float32)]).astype(np.float32, copy=False)
            yield x_batch, y_batch, id_batch, sample_weight

    @staticmethod
    def _build_video_train(videos: np.ndarray, labels: np.ndarray, video_ids: np.ndarray, batch_size: int, seed: int) -> tf.data.Dataset:
        output_signature = (
            tf.TensorSpec(shape=(batch_size, videos.shape[1], videos.shape[2], videos.shape[3]), dtype=tf.float32),
            tf.TensorSpec(shape=(batch_size, labels.shape[-1]), dtype=tf.float32),
            tf.TensorSpec(shape=(batch_size,), dtype=tf.int64),
            tf.TensorSpec(shape=(batch_size,), dtype=tf.float32),
        )
        ds = tf.data.Dataset.from_generator(
            lambda: SequenceRepository._video_batches(videos, labels, video_ids, batch_size, shuffle=True, seed=seed),
            output_signature=output_signature,
        )
        return ds.prefetch(TF_AUTOTUNE)

    @staticmethod
    def _build_video_test(videos: np.ndarray, labels: np.ndarray, video_ids: np.ndarray, batch_size: int) -> tf.data.Dataset:
        output_signature = (
            tf.TensorSpec(shape=(batch_size, videos.shape[1], videos.shape[2], videos.shape[3]), dtype=tf.float32),
            tf.TensorSpec(shape=(batch_size, labels.shape[-1]), dtype=tf.float32),
            tf.TensorSpec(shape=(batch_size,), dtype=tf.int64),
            tf.TensorSpec(shape=(batch_size,), dtype=tf.float32),
        )
        ds = tf.data.Dataset.from_generator(
            lambda: SequenceRepository._video_batches(videos, labels, video_ids, batch_size, shuffle=False),
            output_signature=output_signature,
        )
        return ds.prefetch(TF_AUTOTUNE)
