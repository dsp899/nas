from __future__ import annotations

import queue
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Literal, Optional

import numpy as np

from hybrid_benchlib.runtime.backends import CnnBackend, RnnBackend

PipelineMode = Literal["cnn_rnn_overlap", "cnn_rnn_serialized"]


@dataclass(frozen=True)
class FrameTask:
    sample_index: int
    frame_index: int
    image: np.ndarray


@dataclass(frozen=True)
class FeatureTask:
    sample_index: int
    frame_index: int
    feature: np.ndarray
    produced_ms: float
    preprocess_ms: float
    infer_ms: float
    postprocess_ms: float


@dataclass(frozen=True)
class ClipTask:
    clip_index: int
    sample_start_index: int
    sample_end_index: int
    clip_x: np.ndarray
    clip_ready_ms: float


@dataclass(frozen=True)
class FeatureRecord:
    sample_index: int
    frame_index: int
    produced_ms: float
    preprocess_ms: float
    infer_ms: float
    postprocess_ms: float
    total_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ClipRecord:
    clip_index: int
    sample_start_index: int
    sample_end_index: int
    clip_ready_ms: float
    rnn_queue_wait_ms: float
    rnn_start_ms: float
    encoder_ms: float
    clip_head_ms: float
    aggregation_ms: float
    video_head_ms: float
    decision_ready_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VideoPipelineSummary:
    num_sampled_frames: int
    num_clips: int
    t_first_ms: float
    t_update_mean_ms: float
    t_update_min_ms: float
    t_update_max_ms: float
    cnn_total_ms: float
    rnn_total_ms: float
    queue_wait_total_ms: float
    video_total_ms: float
    feature_records: List[Dict[str, Any]]
    clip_records: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class _Sentinel:
    pass


SENTINEL = _Sentinel()


class HybridPipelineEngine:
    def __init__(
        self,
        cnn_backend_factory,
        rnn_backend_factory,
        *,
        seq: int,
        hop: int,
        cnn_workers: int,
        pipeline_mode: PipelineMode,
    ):
        self.cnn_backend_factory = cnn_backend_factory
        self.rnn_backend_factory = rnn_backend_factory
        self.seq = int(seq)
        self.hop = int(hop)
        self.cnn_workers = int(cnn_workers)
        self.pipeline_mode = pipeline_mode

    def run_video(self, frames: np.ndarray, sample_stride_frames: int = 1) -> VideoPipelineSummary:
        sample_stride_frames = max(1, int(sample_stride_frames))
        sampled_indices = list(range(0, int(frames.shape[0]), sample_stride_frames))
        if len(sampled_indices) < self.seq:
            raise ValueError(
                f"No hay suficientes frames muestreados para formar un clip: sampled={len(sampled_indices)} seq={self.seq}"
            )
        sampled_frames = [FrameTask(i, frame_idx, frames[frame_idx]) for i, frame_idx in enumerate(sampled_indices)]

        t0 = time.perf_counter_ns()

        frame_queue: queue.Queue[Any] = queue.Queue(maxsize=max(8, self.cnn_workers * 2))
        feature_queue: queue.Queue[Any] = queue.Queue(maxsize=max(8, self.cnn_workers * 2))
        clip_queue: queue.Queue[Any] = queue.Queue(maxsize=max(8, self.cnn_workers * 2))

        feature_records: List[FeatureRecord] = []
        feature_records_lock = threading.Lock()
        clip_records: List[ClipRecord] = []
        clip_records_lock = threading.Lock()
        prepared_clips: List[ClipTask] = []
        prepared_clips_lock = threading.Lock()

        def now_ms() -> float:
            return (time.perf_counter_ns() - t0) / 1e6

        def cnn_worker() -> None:
            backend: CnnBackend = self.cnn_backend_factory()
            while True:
                task = frame_queue.get()
                try:
                    if task is SENTINEL:
                        feature_queue.put(SENTINEL)
                        return
                    assert isinstance(task, FrameTask)
                    result = backend.run_feature(task.image)
                    produced_ms = now_ms()
                    record = FeatureRecord(
                        sample_index=task.sample_index,
                        frame_index=task.frame_index,
                        produced_ms=produced_ms,
                        preprocess_ms=float(result.timing.preprocess_ms),
                        infer_ms=float(result.timing.infer_ms),
                        postprocess_ms=float(result.timing.postprocess_ms),
                        total_ms=float(result.timing.total_ms),
                    )
                    with feature_records_lock:
                        feature_records.append(record)
                    feature_queue.put(
                        FeatureTask(
                            sample_index=task.sample_index,
                            frame_index=task.frame_index,
                            feature=np.asarray(result.feature, dtype=np.float32),
                            produced_ms=produced_ms,
                            preprocess_ms=record.preprocess_ms,
                            infer_ms=record.infer_ms,
                            postprocess_ms=record.postprocess_ms,
                        )
                    )
                finally:
                    frame_queue.task_done()

        def assembler_overlap() -> None:
            pending: Dict[int, FeatureTask] = {}
            ordered_features: List[np.ndarray] = []
            next_expected = 0
            next_clip_end = self.seq - 1
            clip_index = 0
            sentinels_seen = 0
            while True:
                item = feature_queue.get()
                try:
                    if item is SENTINEL:
                        sentinels_seen += 1
                        if sentinels_seen == self.cnn_workers:
                            break
                        continue
                    assert isinstance(item, FeatureTask)
                    pending[item.sample_index] = item
                    while next_expected in pending:
                        ft = pending.pop(next_expected)
                        ordered_features.append(np.asarray(ft.feature, dtype=np.float32))
                        while len(ordered_features) - 1 >= next_clip_end:
                            start_idx = next_clip_end - self.seq + 1
                            clip = np.stack(ordered_features[start_idx : next_clip_end + 1], axis=0).astype(np.float32)
                            clip_queue.put(
                                ClipTask(
                                    clip_index=clip_index,
                                    sample_start_index=start_idx,
                                    sample_end_index=next_clip_end,
                                    clip_x=clip,
                                    clip_ready_ms=float(ft.produced_ms),
                                )
                            )
                            clip_index += 1
                            next_clip_end += self.hop
                        next_expected += 1
                finally:
                    feature_queue.task_done()
            clip_queue.put(SENTINEL)

        def assembler_serialized() -> None:
            pending: Dict[int, FeatureTask] = {}
            ordered_features: List[np.ndarray] = []
            next_expected = 0
            next_clip_end = self.seq - 1
            clip_index = 0
            sentinels_seen = 0
            while True:
                item = feature_queue.get()
                try:
                    if item is SENTINEL:
                        sentinels_seen += 1
                        if sentinels_seen == self.cnn_workers:
                            break
                        continue
                    assert isinstance(item, FeatureTask)
                    pending[item.sample_index] = item
                    while next_expected in pending:
                        ft = pending.pop(next_expected)
                        ordered_features.append(np.asarray(ft.feature, dtype=np.float32))
                        while len(ordered_features) - 1 >= next_clip_end:
                            start_idx = next_clip_end - self.seq + 1
                            clip = np.stack(ordered_features[start_idx : next_clip_end + 1], axis=0).astype(np.float32)
                            with prepared_clips_lock:
                                prepared_clips.append(
                                    ClipTask(
                                        clip_index=clip_index,
                                        sample_start_index=start_idx,
                                        sample_end_index=next_clip_end,
                                        clip_x=clip,
                                        clip_ready_ms=float(ft.produced_ms),
                                    )
                                )
                            clip_index += 1
                            next_clip_end += self.hop
                        next_expected += 1
                finally:
                    feature_queue.task_done()

        def rnn_consumer_from_queue() -> None:
            backend: RnnBackend = self.rnn_backend_factory()
            backend.reset_video()
            while True:
                item = clip_queue.get()
                try:
                    if item is SENTINEL:
                        return
                    assert isinstance(item, ClipTask)
                    start_ms = now_ms()
                    result = backend.run_clip(np.expand_dims(item.clip_x, axis=0).astype(np.float32, copy=False))
                    end_ms = now_ms()
                    with clip_records_lock:
                        clip_records.append(
                            ClipRecord(
                                clip_index=item.clip_index,
                                sample_start_index=item.sample_start_index,
                                sample_end_index=item.sample_end_index,
                                clip_ready_ms=float(item.clip_ready_ms),
                                rnn_queue_wait_ms=float(start_ms - item.clip_ready_ms),
                                rnn_start_ms=float(start_ms),
                                encoder_ms=float(result.encoder_ms),
                                clip_head_ms=float(result.clip_head_ms),
                                aggregation_ms=float(result.aggregation_ms),
                                video_head_ms=float(result.video_head_ms),
                                decision_ready_ms=float(end_ms),
                            )
                        )
                finally:
                    clip_queue.task_done()

        def rnn_consumer_from_list() -> None:
            backend: RnnBackend = self.rnn_backend_factory()
            backend.reset_video()
            for item in prepared_clips:
                start_ms = now_ms()
                result = backend.run_clip(np.expand_dims(item.clip_x, axis=0).astype(np.float32, copy=False))
                end_ms = now_ms()
                with clip_records_lock:
                    clip_records.append(
                        ClipRecord(
                            clip_index=item.clip_index,
                            sample_start_index=item.sample_start_index,
                            sample_end_index=item.sample_end_index,
                            clip_ready_ms=float(item.clip_ready_ms),
                            rnn_queue_wait_ms=float(start_ms - item.clip_ready_ms),
                            rnn_start_ms=float(start_ms),
                            encoder_ms=float(result.encoder_ms),
                            clip_head_ms=float(result.clip_head_ms),
                            aggregation_ms=float(result.aggregation_ms),
                            video_head_ms=float(result.video_head_ms),
                            decision_ready_ms=float(end_ms),
                        )
                    )

        assembler_target = assembler_overlap if self.pipeline_mode == "cnn_rnn_overlap" else assembler_serialized
        assembler_thread = threading.Thread(target=assembler_target, daemon=True, name="hybrid-assembler")
        cnn_threads = [threading.Thread(target=cnn_worker, daemon=True, name=f"hybrid-cnn-{i}") for i in range(self.cnn_workers)]

        rnn_thread: Optional[threading.Thread] = None
        if self.pipeline_mode == "cnn_rnn_overlap":
            rnn_thread = threading.Thread(target=rnn_consumer_from_queue, daemon=True, name="hybrid-rnn")
            rnn_thread.start()

        assembler_thread.start()
        for thread in cnn_threads:
            thread.start()

        for task in sampled_frames:
            frame_queue.put(task)
        for _ in range(self.cnn_workers):
            frame_queue.put(SENTINEL)

        frame_queue.join()
        feature_queue.join()
        assembler_thread.join()
        for thread in cnn_threads:
            thread.join()

        if self.pipeline_mode == "cnn_rnn_overlap":
            clip_queue.join()
            assert rnn_thread is not None
            rnn_thread.join()
        else:
            with prepared_clips_lock:
                prepared_clips.sort(key=lambda item: item.clip_index)
            rnn_consumer_from_list()

        feature_records.sort(key=lambda r: r.sample_index)
        clip_records.sort(key=lambda r: r.clip_index)
        if not clip_records:
            raise RuntimeError("El pipeline híbrido no produjo ningún clip")

        decision_times = [record.decision_ready_ms for record in clip_records]
        updates = [decision_times[0]] if len(decision_times) == 1 else [decision_times[i] - decision_times[i - 1] for i in range(1, len(decision_times))]
        cnn_total_ms = float(sum(r.total_ms for r in feature_records))
        rnn_total_ms = float(sum(r.encoder_ms + r.clip_head_ms + r.aggregation_ms + r.video_head_ms for r in clip_records))
        queue_wait_total_ms = float(sum(r.rnn_queue_wait_ms for r in clip_records))
        video_total_ms = float(clip_records[-1].decision_ready_ms)

        return VideoPipelineSummary(
            num_sampled_frames=len(feature_records),
            num_clips=len(clip_records),
            t_first_ms=float(clip_records[0].decision_ready_ms),
            t_update_mean_ms=float(sum(updates) / len(updates)),
            t_update_min_ms=float(min(updates)),
            t_update_max_ms=float(max(updates)),
            cnn_total_ms=cnn_total_ms,
            rnn_total_ms=rnn_total_ms,
            queue_wait_total_ms=queue_wait_total_ms,
            video_total_ms=video_total_ms,
            feature_records=[item.to_dict() for item in feature_records],
            clip_records=[item.to_dict() for item in clip_records],
        )
