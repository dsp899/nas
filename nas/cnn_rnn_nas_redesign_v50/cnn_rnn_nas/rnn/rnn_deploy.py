import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import tensorflow as tf

from ..common.artifacts import ProjectPaths
from ..common.model_io import export_saved_model
from ..common.registries import (
    CnnDeployRegistry,
    RnnDeployEvalRegistry,
    RnnExperimentRegistry,
    RnnExportRegistry,
    RnnTfliteEvalRegistry,
    rnn_deploy_eval_signature,
    rnn_experiment_signature,
    rnn_export_signature,
    rnn_tflite_eval_signature,
)
from ..config.deploy_config import RnnDeployEvalConfig, RnnTfliteEvalConfig, RnnTfliteExportConfig
from ..config.rnn_config import RnnExperimentConfig
from .rnn_data import SequenceRepository
from .rnn_model import VideoAggregator, get_state_spec

class RnnTfliteExporter:
    def __init__(self, paths: ProjectPaths, registry: RnnExperimentRegistry, export_registry: RnnExportRegistry) -> None:
        self.paths = paths
        self.registry = registry
        self.export_registry = export_registry

    def _resolve_experiment_record(self, config: RnnExperimentConfig, explicit_signature: Optional[str] = None) -> Tuple[str, Optional[Dict[str, Any]]]:
        if explicit_signature:
            return explicit_signature, self.registry.get(explicit_signature)
        exact_signature = rnn_experiment_signature(config)
        exact_record = self.registry.get(exact_signature)
        if exact_record and exact_record.get("status") == "completed":
            return exact_signature, exact_record
        for candidate in self.registry.top_completed(limit=100):
            cfg = candidate.get("config", {})
            if (
                cfg.get("data", {}).get("cnn") == config.data.cnn
                and cfg.get("data", {}).get("name") == config.data.name
                and int(cfg.get("data", {}).get("frames", -1)) == int(config.data.frames)
                and int(cfg.get("data", {}).get("image_size", cfg.get("data", {}).get("size", -1))) == int(config.data.image_size)
                and int(cfg.get("data", {}).get("seq", -1)) == int(config.data.seq)
                and cfg.get("architecture", {}).get("memory_mode") == config.architecture.memory_mode
                and cfg.get("architecture", {}).get("rnn") == config.architecture.rnn
                and cfg.get("architecture", {}).get("direction") == config.architecture.direction
                and tuple(cfg.get("architecture", {}).get("units", ())) == tuple(config.architecture.units)
                and int(cfg.get("architecture", {}).get("head_units", -1)) == int(config.architecture.head_units)
                and str(cfg.get("architecture", {}).get("video_decision")) == config.architecture.video_decision
                and str(cfg.get("architecture", {}).get("video_decision_input")) == config.architecture.video_decision_input
            ):
                return str(candidate["rnn_experiment_signature"]), candidate
        return exact_signature, exact_record

    @staticmethod
    def build_inference_components(source_model: tf.keras.Model, config: RnnExperimentConfig) -> Tuple[tf.keras.Model, tf.keras.Model, Dict[str, Any]]:
        clip_embedding = source_model.get_layer("clip_embedding").output
        state_outputs = list(source_model.outputs[2:])
        encoder_model = tf.keras.Model(
            inputs=source_model.inputs,
            outputs=[clip_embedding, *state_outputs],
            name="rnn_encoder",
        )

        embedding_input = tf.keras.Input(
            shape=(int(config.architecture.encoder_output_dim),),
            name="clip_embedding",
            dtype=tf.float32,
        )
        hidden = source_model.get_layer("head_hidden")(embedding_input)
        logits = source_model.get_layer("clip_logits")(hidden)
        head_model = tf.keras.Model(inputs=[embedding_input], outputs=[logits], name="rnn_head")

        state_spec = get_state_spec(config)
        metadata = {
            "encoder_input_names": [tensor.name for tensor in encoder_model.inputs],
            "encoder_output_names": [tensor.name for tensor in encoder_model.outputs],
            "head_input_names": [tensor.name for tensor in head_model.inputs],
            "head_output_names": [tensor.name for tensor in head_model.outputs],
            "state_spec": state_spec,
            "clip_embedding_dim": int(config.architecture.encoder_output_dim),
            "num_classes": int(source_model.get_layer("clip_logits").units),
        }
        return encoder_model, head_model, metadata

    @staticmethod
    def _warm_up_components(
        encoder_model: tf.keras.Model,
        head_model: tf.keras.Model,
        config: RnnExperimentConfig,
        feature_dim: int,
    ) -> None:
        encoder_inputs: List[np.ndarray] = [np.zeros((1, config.data.seq, feature_dim), dtype=np.float32)]
        for entry in get_state_spec(config):
            encoder_inputs.append(np.zeros((1, int(entry["units"])), dtype=np.float32))
        encoder_outputs = encoder_model(encoder_inputs, training=False)
        if isinstance(encoder_outputs, (list, tuple)):
            clip_embedding = np.asarray(encoder_outputs[0], dtype=np.float32)
        else:
            clip_embedding = np.asarray(encoder_outputs, dtype=np.float32)
        _ = head_model([clip_embedding], training=False)

    @staticmethod
    def _convert_to_tflite(model: tf.keras.Model) -> Tuple[bytes, str, Optional[str]]:
        def _convert(supported_ops, lower_tensor_list_ops: Optional[bool]) -> bytes:
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.target_spec.supported_ops = supported_ops
            if lower_tensor_list_ops is not None:
                converter._experimental_lower_tensor_list_ops = lower_tensor_list_ops
            return converter.convert()

        try:
            return (
                _convert([tf.lite.OpsSet.TFLITE_BUILTINS], True),
                "builtin_only",
                None,
            )
        except Exception as exc_builtin:
            builtin_error = repr(exc_builtin)
        return (
            _convert([tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS], False),
            "builtin_plus_select_tf_ops",
            builtin_error,
        )

    @staticmethod
    def _inspect_tflite_component(model_bytes: bytes) -> Dict[str, Any]:
        interpreter = tf.lite.Interpreter(model_content=model_bytes, num_threads=1)
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        ops: List[str] = []
        uses_flex = False
        try:
            ops = [str(detail["op_name"]) for detail in interpreter._get_ops_details()]
            uses_flex = any(op.startswith("Flex") for op in ops)
        except Exception:
            pass
        return {
            "input_details": [
                {"name": detail.get("name"), "shape": np.asarray(detail.get("shape")).tolist()}
                for detail in input_details
            ],
            "output_details": [
                {"name": detail.get("name"), "shape": np.asarray(detail.get("shape")).tolist()}
                for detail in output_details
            ],
            "ops": ops,
            "uses_flex": bool(uses_flex),
        }

    def export(
        self,
        config: RnnExperimentConfig,
        export_config: RnnTfliteExportConfig,
        *,
        explicit_experiment_signature: Optional[str] = None,
        force: bool = False,
    ) -> Dict[str, Any]:
        experiment_signature, experiment_record = self._resolve_experiment_record(config, explicit_experiment_signature)
        if not experiment_record or experiment_record.get("status") != "completed":
            raise FileNotFoundError("No se encontró un experimento RNN completado compatible para exportar a TFLite.")

        model_path = Path(experiment_record.get("best_model_path") or self.paths.model_path(config, experiment_signature))
        if not model_path.exists():
            raise FileNotFoundError(f"No existe el mejor modelo RNN en {model_path}")

        export_signature = rnn_export_signature(config, experiment_signature, export_config)
        manifest_path = self.paths.rnn_export_manifest_path(config, experiment_signature, export_signature)
        existing = self.export_registry.get(export_signature)
        if not force and existing and existing.get("status") == "completed" and manifest_path.exists():
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload["cached"] = True
            return payload

        export_dir = self.paths.rnn_export_dir(config, experiment_signature, export_signature)
        encoder_saved_model_dir = self.paths.rnn_encoder_saved_model_dir(config, experiment_signature, export_signature)
        head_saved_model_dir = self.paths.rnn_head_saved_model_dir(config, experiment_signature, export_signature)
        encoder_tflite_path = self.paths.rnn_encoder_tflite_model_path(config, experiment_signature, export_signature)
        head_tflite_path = self.paths.rnn_head_tflite_model_path(config, experiment_signature, export_signature)
        if force and export_dir.exists():
            shutil.rmtree(export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)

        # Registry keeps the encoder paths in its generic columns. Full component layout lives in the manifest.
        self.export_registry.reserve(
            export_signature,
            experiment_signature,
            config,
            export_config,
            export_dir=export_dir,
            export_manifest_path=manifest_path,
            saved_model_dir=encoder_saved_model_dir,
            tflite_model_path=encoder_tflite_path,
        )

        source_model = encoder_model = head_model = None
        try:
            source_model = tf.keras.models.load_model(str(model_path), compile=False)
            encoder_model, head_model, component_meta = self.build_inference_components(source_model, config)
            input_feature_dim = int(source_model.input_shape[0][-1])
            self._warm_up_components(encoder_model, head_model, config, input_feature_dim)

            export_saved_model(encoder_model, encoder_saved_model_dir)
            export_saved_model(head_model, head_saved_model_dir)

            encoder_bytes, encoder_mode, encoder_builtin_error = self._convert_to_tflite(encoder_model)
            head_bytes, head_mode, head_builtin_error = self._convert_to_tflite(head_model)
            encoder_tflite_path.write_bytes(encoder_bytes)
            head_tflite_path.write_bytes(head_bytes)

            encoder_inspection = self._inspect_tflite_component(encoder_bytes)
            head_inspection = self._inspect_tflite_component(head_bytes)
            uses_flex = bool(
                encoder_mode == "builtin_plus_select_tf_ops"
                or head_mode == "builtin_plus_select_tf_ops"
                or encoder_inspection.get("uses_flex", False)
                or head_inspection.get("uses_flex", False)
            )
            conversion_mode = (
                "builtin_only"
                if encoder_mode == "builtin_only" and head_mode == "builtin_only"
                else "builtin_plus_select_tf_ops"
            )
            summary = {
                "rnn_export_signature": export_signature,
                "rnn_experiment_signature": experiment_signature,
                "export_layout": "encoder_plus_head",
                "export_dir": str(export_dir),
                "export_manifest_path": str(manifest_path),
                "encoder_saved_model_dir": str(encoder_saved_model_dir),
                "head_saved_model_dir": str(head_saved_model_dir),
                "encoder_tflite_path": str(encoder_tflite_path),
                "head_tflite_path": str(head_tflite_path),
                "export_config": export_config.to_dict(),
                "state_spec": get_state_spec(config),
                "component_metadata": component_meta,
                "conversion": {
                    "status": "ok",
                    "conversion_mode": conversion_mode,
                    "uses_flex": uses_flex,
                    "target_runtime_recommendation": (
                        "tflite_runtime_or_tensorflow"
                        if conversion_mode == "builtin_only" and not uses_flex
                        else "tensorflow_full"
                    ),
                    "components": {
                        "encoder": {
                            "conversion_mode": encoder_mode,
                            "builtin_error": encoder_builtin_error,
                            "inspection": encoder_inspection,
                        },
                        "head": {
                            "conversion_mode": head_mode,
                            "builtin_error": head_builtin_error,
                            "inspection": head_inspection,
                        },
                    },
                },
                "cached": False,
            }
            manifest_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
            self.export_registry.complete(export_signature)
            return summary
        except Exception as exc:
            self.export_registry.fail(export_signature, str(exc))
            raise
        finally:
            del source_model
            del encoder_model
            del head_model

class _FloatRunnerPair:
    def __init__(self, encoder_model: tf.keras.Model, head_model: tf.keras.Model) -> None:
        self.encoder_model = encoder_model
        self.head_model = head_model

    def run_encoder(self, clip_x: np.ndarray, states: Sequence[np.ndarray]) -> Tuple[np.ndarray, List[np.ndarray]]:
        outputs = self.encoder_model([clip_x, *states], training=False)
        if not isinstance(outputs, (list, tuple)):
            raise ValueError("El encoder float debe devolver embedding y estados")
        embedding = np.asarray(outputs[0], dtype=np.float32)
        next_states = [np.asarray(value, dtype=np.float32) for value in outputs[1:]]
        return embedding, next_states

    def run_head(self, clip_embedding: np.ndarray) -> np.ndarray:
        outputs = self.head_model([clip_embedding], training=False)
        if isinstance(outputs, (list, tuple)):
            return np.asarray(outputs[0], dtype=np.float32)
        return np.asarray(outputs, dtype=np.float32)


class _TfliteEncoderRunner:
    def __init__(self, model_path: Path, config: RnnExperimentConfig) -> None:
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            Interpreter = tf.lite.Interpreter
        self.config = config
        self.state_spec = get_state_spec(config)
        self.interpreter = Interpreter(model_path=str(model_path), num_threads=1)
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.expected_input_names = ["clip_x", *[entry["name"] for entry in self.state_spec]]
        self.expected_output_names = ["clip_embedding", *[entry["name"] for entry in self.state_spec]]
        self.input_order = self._resolve_tensor_order(self.input_details, self.expected_input_names)
        self.output_order = self._resolve_tensor_order(self.output_details, self.expected_output_names, prefer_order_fallback=True)
        self.interpreter.allocate_tensors()

    @staticmethod
    def _resolve_tensor_order(details: Sequence[Dict[str, Any]], expected_names: Sequence[str], prefer_order_fallback: bool = False) -> List[int]:
        name_to_index: Dict[str, int] = {}
        for pos, detail in enumerate(details):
            raw_name = str(detail.get("name", ""))
            normalized = raw_name.split(":")[0].split("/")[-1]
            name_to_index[normalized] = pos
            name_to_index[raw_name] = pos
        order: List[int] = []
        used: Set[int] = set()
        for idx, expected_name in enumerate(expected_names):
            match = None
            for key, pos in name_to_index.items():
                if pos in used:
                    continue
                if key == expected_name or key.endswith(expected_name) or expected_name in key:
                    match = pos
                    break
            if match is None:
                if prefer_order_fallback and idx < len(details):
                    match = idx
                else:
                    remaining = [pos for pos in range(len(details)) if pos not in used]
                    if len(remaining) == 1:
                        match = remaining[0]
            if match is None:
                raise ValueError(f"No se pudo resolver el tensor esperado: {expected_name}")
            order.append(match)
            used.add(match)
        return order

    def _ensure_input_shapes(self, values: Sequence[np.ndarray]) -> None:
        resized = False
        for detail_pos, value in zip(self.input_order, values):
            detail = self.input_details[detail_pos]
            current = tuple(int(dim) for dim in np.asarray(detail["shape"]).tolist())
            target = tuple(int(dim) for dim in np.asarray(value.shape).tolist())
            if current != target:
                self.interpreter.resize_tensor_input(detail["index"], target, strict=False)
                resized = True
        if resized:
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()

    def run_encoder(self, clip_x: np.ndarray, states: Sequence[np.ndarray]) -> Tuple[np.ndarray, List[np.ndarray]]:
        values = [np.asarray(clip_x, dtype=np.float32), *[np.asarray(state, dtype=np.float32) for state in states]]
        self._ensure_input_shapes(values)
        for detail_pos, value in zip(self.input_order, values):
            self.interpreter.set_tensor(self.input_details[detail_pos]["index"], value)
        self.interpreter.invoke()
        outputs = [self.interpreter.get_tensor(self.output_details[detail_pos]["index"]) for detail_pos in self.output_order]
        return np.asarray(outputs[0], dtype=np.float32), [np.asarray(value, dtype=np.float32) for value in outputs[1:]]


class _TfliteHeadRunner:
    def __init__(self, model_path: Path) -> None:
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            Interpreter = tf.lite.Interpreter
        self.interpreter = Interpreter(model_path=str(model_path), num_threads=1)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def run_head(self, clip_embedding: np.ndarray) -> np.ndarray:
        clip_embedding = np.asarray(clip_embedding, dtype=np.float32)
        current = tuple(int(dim) for dim in np.asarray(self.input_details[0]["shape"]).tolist())
        target = tuple(int(dim) for dim in np.asarray(clip_embedding.shape).tolist())
        if current != target:
            self.interpreter.resize_tensor_input(self.input_details[0]["index"], target, strict=False)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
        self.interpreter.set_tensor(self.input_details[0]["index"], clip_embedding)
        self.interpreter.invoke()
        return np.asarray(self.interpreter.get_tensor(self.output_details[0]["index"]), dtype=np.float32)


class _TfliteRunnerPair:
    def __init__(self, encoder_model_path: Path, head_model_path: Path, config: RnnExperimentConfig) -> None:
        self.encoder_runner = _TfliteEncoderRunner(encoder_model_path, config)
        self.head_runner = _TfliteHeadRunner(head_model_path)

    def run_encoder(self, clip_x: np.ndarray, states: Sequence[np.ndarray]) -> Tuple[np.ndarray, List[np.ndarray]]:
        return self.encoder_runner.run_encoder(clip_x, states)

    def run_head(self, clip_embedding: np.ndarray) -> np.ndarray:
        return self.head_runner.run_head(clip_embedding)


class RnnTfliteEvaluator:
    def __init__(
        self,
        paths: ProjectPaths,
        registry: RnnExperimentRegistry,
        export_registry: RnnExportRegistry,
        eval_registry: RnnTfliteEvalRegistry,
        repo: SequenceRepository,
        exporter: RnnTfliteExporter,
    ) -> None:
        self.paths = paths
        self.registry = registry
        self.export_registry = export_registry
        self.eval_registry = eval_registry
        self.repo = repo
        self.exporter = exporter

    def _resolve_experiment_record(self, config: RnnExperimentConfig, explicit_signature: Optional[str] = None) -> Tuple[str, Optional[Dict[str, Any]]]:
        if explicit_signature:
            return explicit_signature, self.registry.get(explicit_signature)
        exact_signature = rnn_experiment_signature(config)
        exact_record = self.registry.get(exact_signature)
        if exact_record and exact_record.get("status") == "completed":
            return exact_signature, exact_record
        for candidate in self.registry.top_completed(limit=100):
            cfg = candidate.get("config", {})
            if (
                cfg.get("data", {}).get("cnn") == config.data.cnn
                and cfg.get("data", {}).get("name") == config.dataset.name
                and int(cfg.get("data", {}).get("frames", -1)) == int(config.data.frames)
                and int(cfg.get("data", {}).get("size", -1)) == int(config.preprocess.image_size)
                and int(cfg.get("data", {}).get("seq", -1)) == int(config.data.seq)
                and cfg.get("architecture", {}).get("memory_mode") == config.architecture.memory_mode
                and cfg.get("architecture", {}).get("rnn") == config.architecture.rnn
                and cfg.get("architecture", {}).get("direction") == config.architecture.direction
                and tuple(cfg.get("architecture", {}).get("units", ())) == tuple(config.architecture.units)
                and int(cfg.get("architecture", {}).get("head_units", -1)) == int(config.architecture.head_units)
                and str(cfg.get("architecture", {}).get("video_decision")) == config.architecture.video_decision
                and str(cfg.get("architecture", {}).get("video_decision_input")) == config.architecture.video_decision_input
            ):
                return str(candidate["rnn_experiment_signature"]), candidate
        return exact_signature, exact_record

    def _resolve_export_record(
        self,
        config: RnnExperimentConfig,
        export_config: RnnTfliteExportConfig,
        experiment_signature: str,
        explicit_export_signature: Optional[str] = None,
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        if explicit_export_signature:
            return explicit_export_signature, self.export_registry.get(explicit_export_signature)
        export_signature = rnn_export_signature(config, experiment_signature, export_config)
        return export_signature, self.export_registry.get(export_signature)

    def _resolve_export_artifacts(
        self,
        config: RnnExperimentConfig,
        experiment_signature: str,
        export_signature: str,
    ) -> Dict[str, Path]:
        manifest_path = self.paths.rnn_export_manifest_path(config, experiment_signature, export_signature)
        manifest_payload = {}
        if manifest_path.exists():
            manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        candidates = {
            "encoder_tflite_path": [
                manifest_payload.get("encoder_tflite_path"),
                str(self.paths.rnn_encoder_tflite_model_path(config, experiment_signature, export_signature)),
            ],
            "head_tflite_path": [
                manifest_payload.get("head_tflite_path"),
                str(self.paths.rnn_head_tflite_model_path(config, experiment_signature, export_signature)),
            ],
        }
        resolved: Dict[str, Path] = {}
        for key, values in candidates.items():
            found = None
            for value in values:
                if not value:
                    continue
                candidate = Path(str(value))
                if candidate.exists():
                    found = candidate
                    break
            if found is None:
                raise FileNotFoundError(f"No se encontró el artefacto de exportación requerido: {key}")
            resolved[key] = found
        resolved["manifest_path"] = manifest_path
        return resolved

    @staticmethod
    def _zero_state_numpy(config: RnnExperimentConfig, batch_size: int = 1) -> List[np.ndarray]:
        return [np.zeros((batch_size, int(entry["units"])), dtype=np.float32) for entry in get_state_spec(config)]

    @staticmethod
    def _next_state_for_next_clip_numpy(config: RnnExperimentConfig, next_states: Sequence[np.ndarray]) -> List[np.ndarray]:
        if config.architecture.memory_mode == "none":
            return [np.zeros_like(value, dtype=np.float32) for value in next_states]
        if config.architecture.direction == "unidirectional":
            return [np.asarray(value, dtype=np.float32) for value in next_states]
        carried: List[np.ndarray] = []
        for entry, value in zip(get_state_spec(config), next_states):
            if entry["direction"] == "bw":
                carried.append(np.zeros_like(value, dtype=np.float32))
            else:
                carried.append(np.asarray(value, dtype=np.float32))
        return carried

    def _load_split_arrays(self, config: RnnExperimentConfig, eval_split: str):
        return self.repo.load_windowed_split(eval_split, config.data)

    @staticmethod
    def _decision_rule(config: RnnExperimentConfig) -> str:
        if config.architecture.video_decision_input == "clip_logits":
            return f"{config.architecture.video_decision}(clip_logits)"
        return "average(clip_embeddings)->head"

    def _evaluate_runner_pair(
        self,
        runner_pair: Any,
        config: RnnExperimentConfig,
        videos: np.ndarray,
        labels: np.ndarray,
        video_ids: np.ndarray,
    ) -> Dict[str, Any]:
        clip_video_ids: List[int] = []
        clip_labels: List[int] = []
        clip_preds: List[int] = []
        clip_probs: List[np.ndarray] = []
        video_pred_rows: List[Tuple[int, int, int, float, np.ndarray]] = []
        correct_videos = 0

        for video_index in range(videos.shape[0]):
            true_probs = np.asarray(labels[video_index], dtype=np.float32)
            true_class = int(np.argmax(true_probs))
            states = self._zero_state_numpy(config, batch_size=1)
            clip_embeddings: List[np.ndarray] = []
            clip_logits: List[np.ndarray] = []
            for clip_index in range(videos.shape[1]):
                clip = np.asarray(videos[video_index, clip_index : clip_index + 1], dtype=np.float32)
                embedding, next_states = runner_pair.run_encoder(clip, states)
                logits = runner_pair.run_head(embedding)
                logits_1d = np.asarray(logits[0], dtype=np.float32)
                probs_1d = tf.nn.softmax(tf.convert_to_tensor(logits_1d), axis=-1).numpy().astype(np.float32)
                clip_embeddings.append(np.asarray(embedding[0], dtype=np.float32))
                clip_logits.append(logits_1d)
                clip_video_ids.append(int(video_ids[video_index]))
                clip_labels.append(true_class)
                clip_preds.append(int(np.argmax(probs_1d)))
                clip_probs.append(probs_1d)
                states = self._next_state_for_next_clip_numpy(config, next_states)

            if config.architecture.video_decision_input == "clip_logits":
                video_probs = VideoAggregator.exact_probs_from_logits(
                    np.asarray(clip_logits, dtype=np.float32),
                    config.architecture.video_decision,
                    int(clip_logits[0].shape[-1]),
                )
            else:
                aggregated_embedding = np.mean(np.asarray(clip_embeddings, dtype=np.float32), axis=0, keepdims=True).astype(np.float32)
                aggregated_logits = runner_pair.run_head(aggregated_embedding)
                video_probs = tf.nn.softmax(tf.convert_to_tensor(aggregated_logits[0]), axis=-1).numpy().astype(np.float32)

            pred_class = int(np.argmax(video_probs))
            correct_videos += int(pred_class == true_class)
            video_pred_rows.append(
                (
                    int(video_ids[video_index]),
                    true_class,
                    pred_class,
                    float(np.max(video_probs)),
                    np.asarray(video_probs, dtype=np.float32),
                )
            )

        clip_accuracy = float(np.mean(np.asarray(clip_preds) == np.asarray(clip_labels))) if clip_labels else 0.0
        video_accuracy = float(correct_videos / max(1, videos.shape[0]))
        return {
            "clip_accuracy": clip_accuracy,
            "video_accuracy": video_accuracy,
            "clip_predictions": {
                "video_ids": np.asarray(clip_video_ids, dtype=np.int64),
                "true_classes": np.asarray(clip_labels, dtype=np.int64),
                "pred_classes": np.asarray(clip_preds, dtype=np.int64),
                "probs": np.asarray(clip_probs, dtype=np.float32),
            },
            "video_predictions": {
                "video_ids": np.asarray([row[0] for row in video_pred_rows], dtype=np.int64),
                "true_classes": np.asarray([row[1] for row in video_pred_rows], dtype=np.int64),
                "pred_classes": np.asarray([row[2] for row in video_pred_rows], dtype=np.int64),
                "confidences": np.asarray([row[3] for row in video_pred_rows], dtype=np.float32),
                "probs": np.asarray([row[4] for row in video_pred_rows], dtype=np.float32),
            },
            "decision_rule": self._decision_rule(config),
        }

    def evaluate(
        self,
        config: RnnExperimentConfig,
        export_config: RnnTfliteExportConfig,
        eval_config: RnnTfliteEvalConfig,
        *,
        explicit_experiment_signature: Optional[str] = None,
        explicit_export_signature: Optional[str] = None,
        force: bool = False,
    ) -> Dict[str, Any]:
        experiment_signature, experiment_record = self._resolve_experiment_record(config, explicit_experiment_signature)
        if not experiment_record or experiment_record.get("status") != "completed":
            raise FileNotFoundError("No se encontró un experimento RNN completado compatible para evaluar el modelo TFLite.")
        export_signature, export_record = self._resolve_export_record(config, export_config, experiment_signature, explicit_export_signature)
        if not export_record or export_record.get("status") != "completed":
            raise FileNotFoundError("No se encontró un export TFLite completado para esta RNN. Ejecuta primero `run_rnn.py deploy --config <rnn_deploy.json>` con `deploy.action=export`.")

        artifacts = self._resolve_export_artifacts(config, experiment_signature, export_signature)
        eval_signature = rnn_tflite_eval_signature(config, experiment_signature, export_signature, eval_config)
        eval_manifest_path = self.paths.rnn_tflite_eval_manifest_path(config, experiment_signature, export_signature, eval_signature)
        if not force and eval_manifest_path.exists():
            payload = json.loads(eval_manifest_path.read_text(encoding="utf-8"))
            payload["cached"] = True
            return payload

        eval_dir = self.paths.rnn_tflite_eval_dir(config, experiment_signature, export_signature, eval_signature)
        if force and eval_dir.exists():
            import shutil
            shutil.rmtree(str(eval_dir))
        eval_dir.mkdir(parents=True, exist_ok=True)
        self.eval_registry.reserve(
            eval_signature,
            experiment_signature,
            export_signature,
            config,
            eval_config,
            eval_dir=eval_dir,
            eval_manifest_path=eval_manifest_path,
            tflite_model_path=artifacts["encoder_tflite_path"],
        )

        try:
            videos, labels, video_ids = self._load_split_arrays(config, eval_config.eval_split)
            runner_pair = _TfliteRunnerPair(artifacts["encoder_tflite_path"], artifacts["head_tflite_path"], config)
            metrics = self._evaluate_runner_pair(runner_pair, config, videos, labels, video_ids)
            clip_predictions_path = self.paths.rnn_tflite_eval_clip_predictions_path(config, experiment_signature, export_signature, eval_signature) if eval_config.save_clip_predictions else None
            video_predictions_path = self.paths.rnn_tflite_eval_video_predictions_path(config, experiment_signature, export_signature, eval_signature) if eval_config.save_video_predictions else None
            if clip_predictions_path is not None:
                np.savez(str(clip_predictions_path), **metrics["clip_predictions"])
            if video_predictions_path is not None:
                np.savez(str(video_predictions_path), **metrics["video_predictions"])
            summary = {
                "rnn_tflite_eval_signature": eval_signature,
                "rnn_experiment_signature": experiment_signature,
                "rnn_export_signature": export_signature,
                "export_layout": "encoder_plus_head",
                "encoder_tflite_path": str(artifacts["encoder_tflite_path"]),
                "head_tflite_path": str(artifacts["head_tflite_path"]),
                "eval_dir": str(eval_dir),
                "eval_manifest_path": str(eval_manifest_path),
                "eval_config": eval_config.to_dict(),
                "model_video_decision": config.architecture.video_decision,
                "model_video_decision_input": config.architecture.video_decision_input,
                "clip_eval": {
                    "split": eval_config.eval_split,
                    "accuracy": metrics["clip_accuracy"],
                    "num_predictions": int(metrics["clip_predictions"]["pred_classes"].shape[0]),
                },
                "video_eval": {
                    "split": eval_config.eval_split,
                    "accuracy": metrics["video_accuracy"],
                    "num_videos": int(metrics["video_predictions"]["pred_classes"].shape[0]),
                    "decision_rule": metrics["decision_rule"],
                },
                "saved_clip_predictions_path": str(clip_predictions_path) if clip_predictions_path is not None else None,
                "saved_video_predictions_path": str(video_predictions_path) if video_predictions_path is not None else None,
                "cached": False,
            }
            eval_manifest_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
            self.eval_registry.complete(eval_signature)
            return summary
        except Exception as exc:
            self.eval_registry.fail(eval_signature, str(exc))
            raise


class RnnDeployComparisonEvaluator(RnnTfliteEvaluator):
    def __init__(
        self,
        paths: ProjectPaths,
        registry: RnnExperimentRegistry,
        export_registry: RnnExportRegistry,
        deploy_eval_registry: RnnDeployEvalRegistry,
        cnn_deploy_registry: CnnDeployRegistry,
        repo: SequenceRepository,
        exporter: RnnTfliteExporter,
    ) -> None:
        super().__init__(paths, registry, export_registry, RnnTfliteEvalRegistry(paths.rnn_tflite_eval_registry_path), repo, exporter)
        self.deploy_eval_registry = deploy_eval_registry
        self.cnn_deploy_registry = cnn_deploy_registry

    @staticmethod
    def _cnn_config_from_payload(payload: Dict[str, Any]) -> CnnExperimentConfig:
        dataset = CnnDatasetSpec(**payload.get("dataset", {}))
        preprocess = CnnPreprocessSpec(**payload.get("preprocess", {}))
        extractor = CnnExtractorSpec(**payload.get("extractor", {}))
        head = CnnHeadSpec(**payload.get("head", {}))
        training = CnnTrainingSpec(**payload.get("training", {}))
        runtime = CnnRuntimeConfig(**payload.get("runtime", {}))
        return CnnExperimentConfig(
            operation=payload.get("operation", "predict"),
            dataset=dataset,
            preprocess=preprocess,
            extractor=extractor,
            head=head,
            training=training,
            runtime=runtime,
        )

    @staticmethod
    def _load_keras_model(model_path: Path) -> tf.keras.Model:
        errors = []
        attempts = [
            {"compile": False},
            {"compile": False, "custom_objects": {"Functional": tf.keras.Model}},
        ]
        for kwargs in attempts:
            try:
                return tf.keras.models.load_model(str(model_path), **kwargs)
            except Exception as exc:
                errors.append(str(exc))
        try:
            from tensorflow_model_optimization.quantization.keras import vitis_quantize
            with vitis_quantize.quantize_scope():
                return tf.keras.models.load_model(str(model_path), compile=False)
        except Exception as exc:
            errors.append(str(exc))
        try:
            from tensorflow_model_optimization.quantization.keras import vitis_quantize
            with vitis_quantize.quantize_scope():
                return tf.keras.models.load_model(str(model_path), compile=False, custom_objects={"Functional": tf.keras.Model})
        except Exception as exc:
            errors.append(str(exc))
        raise ValueError("No se pudo cargar el modelo {}: {}".format(model_path, " | ".join(errors)))

    @staticmethod
    def _catalog(paths: ProjectPaths, config: CnnExperimentConfig) -> UCF101Catalog:
        return UCF101Catalog(
            paths,
            config.dataset.name,
            config.dataset.split,
            val_fraction=config.dataset.val_fraction,
            seed=config.runtime.random_seed,
        )

    @staticmethod
    def _resolve_records_for_split(catalog: UCF101Catalog, config: CnnExperimentConfig, split_name: str) -> list:
        split_name = split_name.strip().lower()
        if split_name == "train":
            if config.dataset.partition_mode == "train_test":
                return list(catalog.official_train_records)
            return list(catalog.train_records)
        if split_name == "val":
            if config.dataset.partition_mode == "train_test":
                return list(catalog.test_records)
            return list(catalog.val_records)
        if split_name == "test":
            return list(catalog.test_records)
        raise ValueError("Split no soportado: {!r}".format(split_name))

    def _resolve_cnn_deploy_record(
        self,
        cnn_config: CnnExperimentConfig,
        training_signature: str,
        explicit_deploy_signature: Optional[str] = None,
    ) -> Tuple[str, Optional[Dict[str, Any]], CnnVitisAiDeployConfig]:
        deploy_config = CnnVitisAiDeployConfig()
        if explicit_deploy_signature:
            return explicit_deploy_signature, self.cnn_deploy_registry.get(explicit_deploy_signature), deploy_config
        exact_signature = cnn_deploy_signature(cnn_config, training_signature, deploy_config)
        exact_record = self.cnn_deploy_registry.get(exact_signature)
        if exact_record and exact_record.get("status") == "completed":
            return exact_signature, exact_record, deploy_config
        fallback = self.cnn_deploy_registry.find_latest_completed_for_training(training_signature)
        if fallback:
            return str(fallback["cnn_deploy_signature"]), fallback, deploy_config
        return exact_signature, exact_record, deploy_config

    def _resolve_quantized_extractor_path(
        self,
        cnn_config: CnnExperimentConfig,
        training_signature: str,
        deploy_signature: str,
        deploy_record: Optional[Dict[str, Any]],
    ) -> Path:
        candidates = [self.paths.cnn_quantized_extractor_path(cnn_config, training_signature, deploy_signature)]
        if deploy_record and deploy_record.get("deploy_dir"):
            candidates.append(Path(deploy_record["deploy_dir"]) / "quantized" / "quantized_extractor.h5")
        checked = []
        for candidate in candidates:
            candidate = Path(candidate)
            if candidate in checked:
                continue
            checked.append(candidate)
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            "No existe el extractor cuantizado. Ejecuta primero export_vitis_ai y run_quantize.sh. Rutas comprobadas:\n - {}".format(
                "\n - ".join(str(path) for path in checked)
            )
        )

    def _resolve_float_feature_source(self, config: RnnExperimentConfig) -> Tuple[RnnExperimentConfig, Dict[str, Any]]:
        resolved_data, record = self.repo.resolve_data_feature_source(config.data)
        return replace(config, data=resolved_data), record

    def _build_quantized_sequences(
        self,
        config: RnnExperimentConfig,
        cnn_config: CnnExperimentConfig,
        split_name: str,
        quantized_extractor: tf.keras.Model,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        catalog = self._catalog(self.paths, cnn_config)
        records = self._resolve_records_for_split(catalog, cnn_config, split_name)
        builder = FrameDatasetBuilder(
            records,
            cnn_config.preprocess.predict_frames,
            cnn_config.preprocess.image_size,
            cnn_config.training.batch_size,
            catalog.num_classes,
            seed=cnn_config.runtime.random_seed,
            sampling_policy=cnn_config.preprocess.predict_sampling,
            resize_mode=cnn_config.preprocess.resize_mode,
            shuffle_buffer_videos=cnn_config.preprocess.shuffle_buffer_videos,
            shuffle_buffer_frames=cnn_config.preprocess.shuffle_buffer_frames,
            preprocess=cnn_config.preprocess,
        )
        ds = builder.video_feature_dataset(batch_size_videos=cnn_config.training.feature_batch_size)
        features_list = []
        labels_list = []
        video_ids_list = []
        feature_dim = int(quantized_extractor.output_shape[-1])
        frames_size_mb = 0.0
        for video_frames, label, video_id in ds:
            video_frames_np = np.asarray(video_frames, dtype=np.float32)
            batch_videos = int(video_frames_np.shape[0])
            batch_frames = int(video_frames_np.shape[1])
            frames_size_mb += float(video_frames_np.nbytes / (1024 ** 2))
            flat_frames = tf.reshape(video_frames, (batch_videos * batch_frames, cnn_config.preprocess.image_size, cnn_config.preprocess.image_size, 3))
            flat_features = quantized_extractor(flat_frames, training=False)
            feature_dim = int(flat_features.shape[-1])
            features = tf.reshape(flat_features, (batch_videos, batch_frames, feature_dim)).numpy().astype(np.float32)
            repeated_labels = np.repeat(np.asarray(label, dtype=np.float32)[:, None, :], repeats=batch_frames, axis=1).astype(np.float32)
            repeated_video_ids = np.repeat(np.asarray(video_id, dtype=np.int64)[:, None], repeats=batch_frames, axis=1).astype(np.int64)
            features_list.append(features)
            labels_list.append(repeated_labels)
            video_ids_list.append(repeated_video_ids)
        if features_list:
            video_features = np.concatenate(features_list, axis=0)
            video_labels = np.concatenate(labels_list, axis=0)
            video_ids = np.concatenate(video_ids_list, axis=0)
        else:
            video_features = np.zeros((0, config.data.frames, feature_dim), dtype=np.float32)
            video_labels = np.zeros((0, config.data.frames, catalog.num_classes), dtype=np.float32)
            video_ids = np.zeros((0, config.data.frames), dtype=np.int64)
        windows = [self.repo._sliding_windows(video, config.data.seq) for video in video_features]
        sequences = np.stack(windows).astype(np.float32) if windows else np.zeros((0, 1, config.data.seq, feature_dim), dtype=np.float32)
        label_view = video_labels[:, 0, :] if video_labels.ndim == 3 else video_labels
        id_view = video_ids[:, 0] if video_ids.ndim > 1 else video_ids
        summary = {
            "num_videos": int(video_features.shape[0]),
            "num_frames": int(video_features.shape[1]) if video_features.ndim >= 2 else 0,
            "feature_dim": int(feature_dim),
            "frames_size_mb": frames_size_mb,
            "features_size_mb": float(video_features.nbytes / (1024 ** 2)),
        }
        return sequences, label_view.astype(np.float32), id_view.astype(np.int64), summary

    def _load_or_build_quantized_sequences(
        self,
        config: RnnExperimentConfig,
        cnn_config: CnnExperimentConfig,
        experiment_signature: str,
        export_signature: str,
        eval_signature: str,
        split_name: str,
        quantized_extractor: tf.keras.Model,
        save_sequences: bool,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        seq_path = self.paths.rnn_deploy_quantized_sequences_path(config, experiment_signature, export_signature, eval_signature, split_name)
        labels_path = self.paths.rnn_deploy_quantized_labels_path(config, experiment_signature, export_signature, eval_signature, split_name)
        ids_path = self.paths.rnn_deploy_quantized_video_ids_path(config, experiment_signature, export_signature, eval_signature, split_name)
        if seq_path.exists() and labels_path.exists() and ids_path.exists():
            sequences = np.load(seq_path, mmap_mode=None)
            labels = np.load(labels_path, mmap_mode=None)
            video_ids = np.load(ids_path, mmap_mode=None)
            summary = {
                "num_videos": int(labels.shape[0]),
                "num_sequences_per_video": int(sequences.shape[1]) if sequences.ndim >= 2 else 0,
                "sequence_shape": list(sequences.shape),
                "cached": True,
            }
            return sequences, labels, video_ids, summary
        sequences, labels, video_ids, summary = self._build_quantized_sequences(config, cnn_config, split_name, quantized_extractor)
        if save_sequences:
            np.save(seq_path, sequences)
            np.save(labels_path, labels)
            np.save(ids_path, video_ids)
        return sequences, labels, video_ids, summary

    @staticmethod
    def _prediction_payload(prefix: str, metrics: Dict[str, Any]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        clip_payload = {
            f"{prefix}_video_ids": metrics["clip_predictions"]["video_ids"],
            f"{prefix}_true_classes": metrics["clip_predictions"]["true_classes"],
            f"{prefix}_pred_classes": metrics["clip_predictions"]["pred_classes"],
            f"{prefix}_probs": metrics["clip_predictions"]["probs"],
        }
        video_payload = {
            f"{prefix}_video_ids": metrics["video_predictions"]["video_ids"],
            f"{prefix}_true_classes": metrics["video_predictions"]["true_classes"],
            f"{prefix}_pred_classes": metrics["video_predictions"]["pred_classes"],
            f"{prefix}_confidences": metrics["video_predictions"]["confidences"],
            f"{prefix}_probs": metrics["video_predictions"]["probs"],
        }
        return clip_payload, video_payload

    @staticmethod
    def _save_predictions(path: Optional[Path], payload: Dict[str, np.ndarray]) -> Optional[str]:
        if path is None:
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(path), **payload)
        return str(path)

    @staticmethod
    def _comparison_block(ref: Dict[str, Any], other: Dict[str, Any]) -> Dict[str, float]:
        return {
            "clip_accuracy_delta": float(other["clip_eval"]["accuracy"] - ref["clip_eval"]["accuracy"]),
            "video_accuracy_delta": float(other["video_eval"]["accuracy"] - ref["video_eval"]["accuracy"]),
        }

    def evaluate(
        self,
        config: RnnExperimentConfig,
        export_config: RnnTfliteExportConfig,
        eval_config: RnnDeployEvalConfig,
        *,
        explicit_experiment_signature: Optional[str] = None,
        explicit_export_signature: Optional[str] = None,
        explicit_cnn_deploy_signature: Optional[str] = None,
        force: bool = False,
    ) -> Dict[str, Any]:
        config, feature_record = self._resolve_float_feature_source(config)
        experiment_signature, experiment_record = self._resolve_experiment_record(config, explicit_experiment_signature)
        if not experiment_record or experiment_record.get("status") != "completed":
            raise FileNotFoundError("No se encontró un experimento RNN completado compatible para evaluar el deploy RNN.")
        export_signature, export_record = self._resolve_export_record(config, export_config, experiment_signature, explicit_export_signature)
        if not export_record or export_record.get("status") != "completed":
            raise FileNotFoundError("No se encontró un export TFLite completado para esta RNN. Ejecuta primero `run_rnn.py deploy --config <rnn_deploy.json>` con `deploy.action=export`.")

        artifacts = self._resolve_export_artifacts(config, experiment_signature, export_signature)
        cnn_config = self._cnn_config_from_payload(feature_record["config"])
        cnn_training_signature = str(feature_record["training_signature"])
        cnn_deploy_signature_value, cnn_deploy_record, _cnn_deploy_config = self._resolve_cnn_deploy_record(cnn_config, cnn_training_signature, explicit_cnn_deploy_signature)
        if not cnn_deploy_record or cnn_deploy_record.get("status") != "completed":
            raise FileNotFoundError("No se encontró un deploy CNN completado compatible para generar features cuantizadas. Ejecuta primero `run_cnn.py deploy --config <cnn_deploy.json>` con `deploy.action=export` y después `run_quantize.sh`.")
        quantized_extractor_path = self._resolve_quantized_extractor_path(cnn_config, cnn_training_signature, cnn_deploy_signature_value, cnn_deploy_record)

        eval_signature = rnn_deploy_eval_signature(config, experiment_signature, export_signature, cnn_deploy_signature_value, eval_config)
        eval_manifest_path = self.paths.rnn_deploy_eval_manifest_path(config, experiment_signature, export_signature, eval_signature)
        if not force and eval_manifest_path.exists():
            payload = json.loads(eval_manifest_path.read_text(encoding="utf-8"))
            payload["cached"] = True
            return payload

        eval_dir = self.paths.rnn_deploy_eval_dir(config, experiment_signature, export_signature, eval_signature)
        if force and eval_dir.exists():
            import shutil
            shutil.rmtree(str(eval_dir))
        eval_dir.mkdir(parents=True, exist_ok=True)
        self.deploy_eval_registry.reserve(
            eval_signature,
            experiment_signature,
            export_signature,
            cnn_deploy_signature_value,
            config,
            eval_config,
            eval_dir=eval_dir,
            eval_manifest_path=eval_manifest_path,
            tflite_model_path=artifacts["encoder_tflite_path"],
            quantized_extractor_path=quantized_extractor_path,
        )

        source_model = quantized_extractor = None
        try:
            float_videos, float_labels, float_video_ids = self._load_split_arrays(config, eval_config.eval_split)
            source_model_path = Path(experiment_record.get("best_model_path") or self.paths.model_path(config, experiment_signature))
            source_model = tf.keras.models.load_model(str(source_model_path), compile=False)
            encoder_model, head_model, _component_meta = self.exporter.build_inference_components(source_model, config)
            input_feature_dim = int(source_model.input_shape[0][-1])
            self.exporter._warm_up_components(encoder_model, head_model, config, input_feature_dim)
            float_runner = _FloatRunnerPair(encoder_model, head_model)
            tflite_runner = _TfliteRunnerPair(artifacts["encoder_tflite_path"], artifacts["head_tflite_path"], config)
            quantized_extractor = self._load_keras_model(quantized_extractor_path)
            quant_videos, quant_labels, quant_video_ids, quant_feature_summary = self._load_or_build_quantized_sequences(
                config,
                cnn_config,
                experiment_signature,
                export_signature,
                eval_signature,
                eval_config.eval_split,
                quantized_extractor,
                eval_config.save_quantized_sequences,
            )
            ff_float = self._evaluate_runner_pair(float_runner, config, float_videos, float_labels, float_video_ids)
            ff_tflite = self._evaluate_runner_pair(tflite_runner, config, float_videos, float_labels, float_video_ids)
            qf_float = self._evaluate_runner_pair(float_runner, config, quant_videos, quant_labels, quant_video_ids)
            qf_tflite = self._evaluate_runner_pair(tflite_runner, config, quant_videos, quant_labels, quant_video_ids)
            branches = {
                "float_features_float_rnn": ff_float,
                "float_features_tflite_rnn": ff_tflite,
                "quantized_features_float_rnn": qf_float,
                "quantized_features_tflite_rnn": qf_tflite,
            }
            clip_payload: Dict[str, np.ndarray] = {}
            video_payload: Dict[str, np.ndarray] = {}
            for name, metrics in branches.items():
                c_payload, v_payload = self._prediction_payload(name, metrics)
                clip_payload.update(c_payload)
                video_payload.update(v_payload)
            clip_predictions_path = self.paths.rnn_deploy_eval_clip_predictions_path(config, experiment_signature, export_signature, eval_signature) if eval_config.save_clip_predictions else None
            video_predictions_path = self.paths.rnn_deploy_eval_video_predictions_path(config, experiment_signature, export_signature, eval_signature) if eval_config.save_video_predictions else None
            saved_clip = self._save_predictions(clip_predictions_path, clip_payload)
            saved_video = self._save_predictions(video_predictions_path, video_payload)
            summary_branches = {}
            for name, metrics in branches.items():
                summary_branches[name] = {
                    "clip_eval": {
                        "split": eval_config.eval_split,
                        "accuracy": metrics["clip_accuracy"],
                        "num_predictions": int(metrics["clip_predictions"]["pred_classes"].shape[0]),
                    },
                    "video_eval": {
                        "split": eval_config.eval_split,
                        "accuracy": metrics["video_accuracy"],
                        "num_videos": int(metrics["video_predictions"]["pred_classes"].shape[0]),
                        "decision_rule": metrics["decision_rule"],
                    },
                }
            comparisons = {
                "float_features_tflite_vs_float": self._comparison_block(summary_branches["float_features_float_rnn"], summary_branches["float_features_tflite_rnn"]),
                "quantized_features_float_vs_float_features_float": self._comparison_block(summary_branches["float_features_float_rnn"], summary_branches["quantized_features_float_rnn"]),
                "quantized_features_tflite_vs_float_features_tflite": self._comparison_block(summary_branches["float_features_tflite_rnn"], summary_branches["quantized_features_tflite_rnn"]),
                "quantized_features_tflite_vs_quantized_features_float": self._comparison_block(summary_branches["quantized_features_float_rnn"], summary_branches["quantized_features_tflite_rnn"]),
            }
            summary = {
                "rnn_deploy_eval_signature": eval_signature,
                "rnn_experiment_signature": experiment_signature,
                "rnn_export_signature": export_signature,
                "cnn_deploy_signature": cnn_deploy_signature_value,
                "export_layout": "encoder_plus_head",
                "encoder_tflite_path": str(artifacts["encoder_tflite_path"]),
                "head_tflite_path": str(artifacts["head_tflite_path"]),
                "quantized_extractor_path": str(quantized_extractor_path),
                "eval_dir": str(eval_dir),
                "eval_manifest_path": str(eval_manifest_path),
                "eval_config": eval_config.to_dict(),
                "cnn_training_signature": config.data.cnn_training_signature,
                "cnn_feature_export_signature": config.data.cnn_feature_export_signature,
                "model_video_decision": config.architecture.video_decision,
                "model_video_decision_input": config.architecture.video_decision_input,
                "quantized_feature_summary": quant_feature_summary,
                "float_features_float_rnn": summary_branches["float_features_float_rnn"],
                "float_features_tflite_rnn": summary_branches["float_features_tflite_rnn"],
                "quantized_features_float_rnn": summary_branches["quantized_features_float_rnn"],
                "quantized_features_tflite_rnn": summary_branches["quantized_features_tflite_rnn"],
                "comparisons": comparisons,
                "saved_clip_predictions_path": saved_clip,
                "saved_video_predictions_path": saved_video,
                "cached": False,
            }
            eval_manifest_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
            self.deploy_eval_registry.complete(eval_signature)
            return summary
        except Exception as exc:
            self.deploy_eval_registry.fail(eval_signature, str(exc))
            raise
        finally:
            del source_model
            del quantized_extractor
