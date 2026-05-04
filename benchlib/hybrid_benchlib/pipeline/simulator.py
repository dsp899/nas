from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Literal

from hybrid_benchlib.runtime.profiles import CnnComponentProfile, RnnComponentProfile


OverlapMode = Literal["cnn_rnn_overlap", "cnn_rnn_serialized"]


@dataclass(frozen=True)
class ClipPipelineRecord:
    clip_index: int
    frame_start_index: int
    frame_end_index: int
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
    video_index: int
    num_features: int
    num_clips: int
    t_first_ms: float
    t_update_mean_ms: float
    t_update_min_ms: float
    t_update_max_ms: float
    cnn_total_ms: float
    rnn_total_ms: float
    queue_wait_total_ms: float
    video_total_ms: float
    clip_records: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ComponentLatencyBreakdown:
    cnn_profile: Dict[str, Any]
    rnn_profile: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _build_feature_ready_times(
    num_features: int,
    frame_interval_ms: float,
    cnn_service_ms: float,
    cnn_workers: int,
    start_time_ms: float = 0.0,
) -> List[float]:
    worker_available = [start_time_ms for _ in range(max(1, cnn_workers))]
    ready_times: List[float] = []
    for feature_index in range(num_features):
        arrival_ms = start_time_ms + feature_index * frame_interval_ms
        worker_idx = min(range(len(worker_available)), key=lambda idx: worker_available[idx])
        begin_ms = max(arrival_ms, worker_available[worker_idx])
        finish_ms = begin_ms + cnn_service_ms
        worker_available[worker_idx] = finish_ms
        ready_times.append(finish_ms)
    return ready_times


def _compute_num_clips(num_features: int, seq: int, hop: int) -> int:
    if num_features < seq:
        return 0
    return 1 + ((num_features - seq) // hop)


def simulate_video_pipeline(
    *,
    video_index: int,
    num_features: int,
    seq: int,
    hop: int,
    frame_interval_ms: float,
    cnn_workers: int,
    overlap_mode: OverlapMode,
    cnn_profile: CnnComponentProfile,
    rnn_profile: RnnComponentProfile,
) -> VideoPipelineSummary:
    if seq <= 0:
        raise ValueError("seq debe ser > 0")
    if hop <= 0:
        raise ValueError("hop debe ser > 0")

    num_clips = _compute_num_clips(num_features=num_features, seq=seq, hop=hop)
    if num_clips <= 0:
        raise ValueError(
            f"No hay suficientes features para construir clips: num_features={num_features}, seq={seq}, hop={hop}"
        )

    cnn_service_ms = float(cnn_profile.sample_total_ms)
    per_clip_rnn_ms = float(rnn_profile.encoder_ms + rnn_profile.clip_head_ms + rnn_profile.aggregation_ms + rnn_profile.video_head_ms)

    clip_records: List[ClipPipelineRecord] = []
    decision_times: List[float] = []
    queue_wait_total_ms = 0.0
    cnn_total_ms = 0.0
    rnn_total_ms = 0.0

    if overlap_mode == "cnn_rnn_overlap":
        feature_ready = _build_feature_ready_times(
            num_features=num_features,
            frame_interval_ms=frame_interval_ms,
            cnn_service_ms=cnn_service_ms,
            cnn_workers=cnn_workers,
            start_time_ms=0.0,
        )
        cnn_total_ms = float(max(feature_ready) if feature_ready else 0.0)
        rnn_available_ms = 0.0
        for clip_index in range(num_clips):
            start_idx = clip_index * hop
            end_idx = start_idx + seq - 1
            clip_ready_ms = float(max(feature_ready[start_idx : end_idx + 1]))
            rnn_start_ms = max(clip_ready_ms, rnn_available_ms)
            queue_wait_ms = max(0.0, rnn_available_ms - clip_ready_ms)
            encoder_ms = float(rnn_profile.encoder_ms)
            clip_head_ms = float(rnn_profile.clip_head_ms)
            aggregation_ms = float(rnn_profile.aggregation_ms)
            video_head_ms = float(rnn_profile.video_head_ms)
            decision_ready_ms = rnn_start_ms + encoder_ms + clip_head_ms + aggregation_ms + video_head_ms
            rnn_available_ms = decision_ready_ms
            queue_wait_total_ms += queue_wait_ms
            rnn_total_ms += encoder_ms + clip_head_ms + aggregation_ms + video_head_ms
            decision_times.append(decision_ready_ms)
            clip_records.append(
                ClipPipelineRecord(
                    clip_index=clip_index,
                    frame_start_index=start_idx,
                    frame_end_index=end_idx,
                    clip_ready_ms=clip_ready_ms,
                    rnn_queue_wait_ms=queue_wait_ms,
                    rnn_start_ms=rnn_start_ms,
                    encoder_ms=encoder_ms,
                    clip_head_ms=clip_head_ms,
                    aggregation_ms=aggregation_ms,
                    video_head_ms=video_head_ms,
                    decision_ready_ms=decision_ready_ms,
                )
            )
    elif overlap_mode == "cnn_rnn_serialized":
        current_time_ms = 0.0
        feature_ready: Dict[int, float] = {}
        next_feature_to_produce = 0
        for clip_index in range(num_clips):
            start_idx = clip_index * hop
            end_idx = start_idx + seq - 1
            required_last = end_idx
            if required_last >= next_feature_to_produce:
                missing_count = required_last - next_feature_to_produce + 1
                cnn_chunk_start_ms = current_time_ms
                new_ready = _build_feature_ready_times(
                    num_features=missing_count,
                    frame_interval_ms=frame_interval_ms,
                    cnn_service_ms=cnn_service_ms,
                    cnn_workers=cnn_workers,
                    start_time_ms=current_time_ms,
                )
                for offset, ready_ms in enumerate(new_ready):
                    feature_ready[next_feature_to_produce + offset] = ready_ms
                next_feature_to_produce = required_last + 1
                current_time_ms = max(new_ready)
                cnn_total_ms += max(0.0, current_time_ms - cnn_chunk_start_ms)
            clip_ready_ms = max(feature_ready[i] for i in range(start_idx, end_idx + 1))
            queue_wait_ms = 0.0
            rnn_start_ms = max(current_time_ms, clip_ready_ms)
            encoder_ms = float(rnn_profile.encoder_ms)
            clip_head_ms = float(rnn_profile.clip_head_ms)
            aggregation_ms = float(rnn_profile.aggregation_ms)
            video_head_ms = float(rnn_profile.video_head_ms)
            decision_ready_ms = rnn_start_ms + encoder_ms + clip_head_ms + aggregation_ms + video_head_ms
            current_time_ms = decision_ready_ms
            rnn_total_ms += encoder_ms + clip_head_ms + aggregation_ms + video_head_ms
            decision_times.append(decision_ready_ms)
            clip_records.append(
                ClipPipelineRecord(
                    clip_index=clip_index,
                    frame_start_index=start_idx,
                    frame_end_index=end_idx,
                    clip_ready_ms=clip_ready_ms,
                    rnn_queue_wait_ms=queue_wait_ms,
                    rnn_start_ms=rnn_start_ms,
                    encoder_ms=encoder_ms,
                    clip_head_ms=clip_head_ms,
                    aggregation_ms=aggregation_ms,
                    video_head_ms=video_head_ms,
                    decision_ready_ms=decision_ready_ms,
                )
            )
    else:
        raise ValueError(f"overlap_mode no soportado: {overlap_mode!r}")

    if len(decision_times) == 1:
        updates = [decision_times[0]]
    else:
        updates = [decision_times[i] - decision_times[i - 1] for i in range(1, len(decision_times))]

    return VideoPipelineSummary(
        video_index=video_index,
        num_features=num_features,
        num_clips=num_clips,
        t_first_ms=float(decision_times[0]),
        t_update_mean_ms=float(sum(updates) / len(updates)),
        t_update_min_ms=float(min(updates)),
        t_update_max_ms=float(max(updates)),
        cnn_total_ms=float(cnn_total_ms),
        rnn_total_ms=float(rnn_total_ms),
        queue_wait_total_ms=float(queue_wait_total_ms),
        video_total_ms=float(decision_times[-1]),
        clip_records=[record.to_dict() for record in clip_records],
    )
