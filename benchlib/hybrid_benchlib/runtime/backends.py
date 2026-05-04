from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Tuple

import numpy as np

from cnn_benchlib.runtime.backends import FloatCnnRuntime, TfliteCnnRuntime, XmodelCnnRuntime
from rnn_benchlib.config.schemas import FeatureSpec, ModelSpec


@dataclass(frozen=True)
class CnnStageTiming:
    preprocess_ms: float
    infer_ms: float
    postprocess_ms: float

    @property
    def total_ms(self) -> float:
        return self.preprocess_ms + self.infer_ms + self.postprocess_ms


@dataclass(frozen=True)
class CnnFeatureResult:
    feature: np.ndarray
    timing: CnnStageTiming


@dataclass(frozen=True)
class RnnDecisionResult:
    embedding: np.ndarray
    logits: np.ndarray
    encoder_ms: float
    clip_head_ms: float
    aggregation_ms: float
    video_head_ms: float

    @property
    def total_ms(self) -> float:
        return self.encoder_ms + self.clip_head_ms + self.aggregation_ms + self.video_head_ms




def _rnn_runtime_api():
    from rnn_benchlib.benchmark import runners as _r
    return _r


def _get_state_spec(spec: ModelSpec) -> List[Dict[str, object]]:
    state_entries: List[Dict[str, object]] = []
    units_list = spec.normalized_units_list()
    for layer_index, units in enumerate(units_list):
        if spec.direction == "unidirectional":
            if spec.rnn == "lstm":
                state_entries.extend([
                    {"name": f"layer{layer_index}_h", "units": units},
                    {"name": f"layer{layer_index}_c", "units": units},
                ])
            else:
                state_entries.append({"name": f"layer{layer_index}_h", "units": units})
        else:
            for direction in ("fw", "bw"):
                if spec.rnn == "lstm":
                    state_entries.extend([
                        {"name": f"layer{layer_index}_{direction}_h", "units": units},
                        {"name": f"layer{layer_index}_{direction}_c", "units": units},
                    ])
                else:
                    state_entries.append({"name": f"layer{layer_index}_{direction}_h", "units": units})
    return state_entries


def _zero_state_numpy(spec: ModelSpec, batch_size: int, dtype: np.dtype = np.float32) -> Dict[str, np.ndarray]:
    result: Dict[str, np.ndarray] = {}
    for entry in _get_state_spec(spec):
        result[str(entry["name"])] = np.zeros((batch_size, int(entry["units"])), dtype=dtype)
    return result

class CnnBackend(Protocol):
    def run_feature(self, image: np.ndarray) -> CnnFeatureResult:
        ...


class RnnBackend(Protocol):
    spec: ModelSpec

    def reset_video(self) -> None:
        ...

    def run_clip(self, clip_x: np.ndarray) -> RnnDecisionResult:
        ...


class FloatCnnFeatureBackend:
    def __init__(self, extractor_path: str):
        self._runtime = FloatCnnRuntime(extractor_path)

    def run_feature(self, image: np.ndarray) -> CnnFeatureResult:
        result = self._runtime.run(image)
        feature = np.asarray(result.output)
        if feature.ndim >= 2:
            feature = np.asarray(feature[0])
        return CnnFeatureResult(
            feature=feature.astype(np.float32, copy=False),
            timing=CnnStageTiming(
                preprocess_ms=float(result.preprocess_ms),
                infer_ms=float(result.infer_ms),
                postprocess_ms=float(result.postprocess_ms),
            ),
        )


class TfliteCnnFeatureBackend:
    def __init__(self, extractor_path: str, threads: int = 1):
        self._runtime = TfliteCnnRuntime(extractor_path, num_threads=threads)

    def run_feature(self, image: np.ndarray) -> CnnFeatureResult:
        result = self._runtime.run(image)
        feature = np.asarray(result.output)
        if feature.ndim >= 2:
            feature = np.asarray(feature[0])
        return CnnFeatureResult(
            feature=feature.astype(np.float32, copy=False),
            timing=CnnStageTiming(
                preprocess_ms=float(result.preprocess_ms),
                infer_ms=float(result.infer_ms),
                postprocess_ms=float(result.postprocess_ms),
            ),
        )


class XmodelCnnFeatureBackend:
    def __init__(self, xmodel_path: str):
        self._runtime = XmodelCnnRuntime(xmodel_path)

    def run_feature(self, image: np.ndarray) -> CnnFeatureResult:
        result = self._runtime.run(image)
        feature = np.asarray(result.output)
        if feature.ndim >= 2:
            feature = np.asarray(feature[0])
        return CnnFeatureResult(
            feature=feature.astype(np.float32, copy=False),
            timing=CnnStageTiming(
                preprocess_ms=float(result.preprocess_ms),
                infer_ms=float(result.infer_ms),
                postprocess_ms=float(result.postprocess_ms),
            ),
        )


class _BaseRnnBackend:
    def __init__(self, spec: ModelSpec, feature_spec: FeatureSpec):
        self.spec = spec
        self.feature_spec = feature_spec
        self._state: Dict[str, np.ndarray] = {}
        self._clip_embeddings: List[np.ndarray] = []
        self._clip_logits: List[np.ndarray] = []
        self.reset_video()

    def reset_video(self) -> None:
        self._state = _zero_state_numpy(spec=self.spec, batch_size=1, dtype=np.float32)
        self._clip_embeddings = []
        self._clip_logits = []

    def _aggregate_logits(self) -> Tuple[np.ndarray, float, float]:
        t0 = time.perf_counter_ns()
        video_logits = _rnn_runtime_api().aggregate_video_from_logits(
            self.spec.video_decision,
            np.stack(self._clip_logits, axis=0),
            self.spec.num_classes,
        )
        t1 = time.perf_counter_ns()
        return np.asarray(video_logits), float((t1 - t0) / 1e6), 0.0

    def _aggregate_embeddings(self) -> Tuple[np.ndarray, float, float]:
        t0 = time.perf_counter_ns()
        aggregated_embedding = _rnn_runtime_api().aggregate_video_from_embeddings(
            self.spec.video_decision,
            np.stack(self._clip_embeddings, axis=0),
        )
        t1 = time.perf_counter_ns()
        logits, video_head_ms = self._run_video_head(aggregated_embedding)
        return np.asarray(logits), float((t1 - t0) / 1e6), video_head_ms

    def _run_video_head(self, aggregated_embedding: np.ndarray) -> Tuple[np.ndarray, float]:
        raise NotImplementedError

    def _run_single_clip(self, clip_x: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray], float, Optional[np.ndarray], float]:
        raise NotImplementedError

    def run_clip(self, clip_x: np.ndarray) -> RnnDecisionResult:
        embedding, next_state, encoder_ms, maybe_clip_logits, clip_head_ms = self._run_single_clip(clip_x)
        self._clip_embeddings.append(np.asarray(embedding[0], dtype=np.float32))
        if maybe_clip_logits is not None:
            self._clip_logits.append(np.asarray(maybe_clip_logits[0], dtype=np.float32))
        self._state = _rnn_runtime_api()._next_state_for_next_clip(spec=self.spec, next_state=next_state)

        if self.spec.video_decision_input == "clip_logits":
            logits, aggregation_ms, video_head_ms = self._aggregate_logits()
        else:
            logits, aggregation_ms, video_head_ms = self._aggregate_embeddings()

        return RnnDecisionResult(
            embedding=np.asarray(embedding[0], dtype=np.float32),
            logits=np.asarray(logits, dtype=np.float32),
            encoder_ms=float(encoder_ms),
            clip_head_ms=float(clip_head_ms),
            aggregation_ms=float(aggregation_ms),
            video_head_ms=float(video_head_ms),
        )


class FloatRnnBackend(_BaseRnnBackend):
    def __init__(self, encoder_model_path: str, head_model_path: str, spec: ModelSpec, feature_spec: FeatureSpec):
        super().__init__(spec=spec, feature_spec=feature_spec)
        self.encoder_model = _rnn_runtime_api().load_float_model(encoder_model_path)
        self.head_model = _rnn_runtime_api().load_float_model(head_model_path)

    def _run_video_head(self, aggregated_embedding: np.ndarray) -> Tuple[np.ndarray, float]:
        logits, ms = _rnn_runtime_api()._single_clip_head_float(
            self.head_model,
            np.expand_dims(aggregated_embedding, axis=0).astype(np.float32, copy=False),
        )
        return np.asarray(logits[0], dtype=np.float32), float(ms)

    def _run_single_clip(self, clip_x: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray], float, Optional[np.ndarray], float]:
        embedding, next_state, encoder_ms = _rnn_runtime_api()._single_clip_encoder_float(
            self.encoder_model,
            self.spec,
            clip_x,
            self._state,
        )
        if self.spec.video_decision_input == "clip_logits":
            logits, head_ms = _rnn_runtime_api()._single_clip_head_float(self.head_model, embedding)
            return embedding, next_state, encoder_ms, logits, head_ms
        return embedding, next_state, encoder_ms, None, 0.0


class TfliteRnnBackend(_BaseRnnBackend):
    def __init__(
        self,
        encoder_model_path: str,
        head_model_path: str,
        float_encoder_reference_path: str,
        spec: ModelSpec,
        feature_spec: FeatureSpec,
        threads: int = 1,
    ):
        super().__init__(spec=spec, feature_spec=feature_spec)
        self.encoder_interpreter, self.encoder_input_map, self.encoder_output_indices = _rnn_runtime_api().create_encoder_tflite_interpreter(
            encoder_model_path,
            spec,
            feature_spec,
            num_threads=threads,
        )
        try:
            float_encoder_model = _rnn_runtime_api().load_float_model(float_encoder_reference_path)
            self.encoder_output_indices = _rnn_runtime_api().resolve_encoder_tflite_output_indices_with_float_reference(
                self.encoder_interpreter,
                float_encoder_model,
                spec,
                feature_spec,
                self.encoder_input_map,
            )
        except Exception:
            # fallback al mapping ya calculado por nombre/shape
            pass
        self.head_interpreter, self.head_input_index, self.head_output_index = _rnn_runtime_api().create_head_tflite_interpreter(
            head_model_path,
            spec,
            num_threads=threads,
        )

    def _run_video_head(self, aggregated_embedding: np.ndarray) -> Tuple[np.ndarray, float]:
        logits, ms = _rnn_runtime_api()._single_clip_head_tflite(
            self.head_interpreter,
            self.head_input_index,
            self.head_output_index,
            np.expand_dims(aggregated_embedding, axis=0).astype(np.float32, copy=False),
        )
        return np.asarray(logits[0], dtype=np.float32), float(ms)

    def _run_single_clip(self, clip_x: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray], float, Optional[np.ndarray], float]:
        embedding, next_state, encoder_ms = _rnn_runtime_api()._single_clip_encoder_tflite(
            self.encoder_interpreter,
            self.spec,
            clip_x,
            self._state,
            self.encoder_input_map,
            self.encoder_output_indices,
        )
        if self.spec.video_decision_input == "clip_logits":
            logits, head_ms = _rnn_runtime_api()._single_clip_head_tflite(
                self.head_interpreter,
                self.head_input_index,
                self.head_output_index,
                embedding,
            )
            return embedding, next_state, encoder_ms, logits, head_ms
        return embedding, next_state, encoder_ms, None, 0.0


def create_cnn_backend_from_record(record: Dict[str, object], backend: str, threads: int = 1) -> CnnBackend:
    if backend == "float":
        return FloatCnnFeatureBackend(str(record["float_extractor_path"]))
    if backend == "tflite":
        status = record.get("tflite_extractor", {})  # type: ignore[assignment]
        path = None
        if isinstance(status, dict):
            path = status.get("path")
        path = path or record.get("tflite_extractor_path")
        if not path:
            raise ValueError("El bundle CNN no tiene extractor TFLite disponible")
        return TfliteCnnFeatureBackend(str(path), threads=threads)
    if backend == "xmodel":
        status = record.get("xmodel_extractor", {})  # type: ignore[assignment]
        path = None
        if isinstance(status, dict):
            path = status.get("path")
        path = path or record.get("xmodel_extractor_path")
        if not path:
            raise ValueError("El bundle CNN no tiene extractor XModel disponible")
        return XmodelCnnFeatureBackend(str(path))
    raise ValueError(f"Backend CNN no soportado: {backend!r}")


def create_rnn_backend_from_record(record: Dict[str, object], backend: str, threads: int = 1) -> RnnBackend:
    spec = ModelSpec(**record["spec"])  # type: ignore[arg-type]
    feature_spec = FeatureSpec(**record["feature_spec"])  # type: ignore[arg-type]
    artifacts = record["artifacts"]  # type: ignore[index]
    if not isinstance(artifacts, dict):
        raise ValueError("Formato inesperado de artifacts en bundle RNN")

    if backend == "float":
        return FloatRnnBackend(
            encoder_model_path=str(artifacts["encoder_keras_dir"]),
            head_model_path=str(artifacts["head_keras_dir"]),
            spec=spec,
            feature_spec=feature_spec,
        )
    if backend == "tflite":
        encoder_tflite = artifacts.get("encoder_tflite_path")
        head_tflite = artifacts.get("head_tflite_path")
        if not encoder_tflite or not head_tflite:
            raise ValueError("El bundle RNN no tiene encoder/head TFLite disponibles")
        return TfliteRnnBackend(
            encoder_model_path=str(encoder_tflite),
            head_model_path=str(head_tflite),
            float_encoder_reference_path=str(artifacts["encoder_keras_dir"]),
            spec=spec,
            feature_spec=feature_spec,
            threads=threads,
        )
    raise ValueError(f"Backend RNN no soportado: {backend!r}")
