from __future__ import annotations

from pathlib import Path
from benchlib_common.io.jsonl import write_json
from cnn_benchlib.config.schemas import ConversionStatus, TfliteExportConfig
from cnn_benchlib.storage.layout import build_artifact_paths
from cnn_benchlib.storage.registry import CnnModelRegistry


def _convert_model(model_path: str, output_path: str, config: TfliteExportConfig):
    import tensorflow as tf

    model = tf.keras.models.load_model(model_path, compile=False)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if config.optimize_default:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if config.allow_select_tf_ops:
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    blob = converter.convert()
    Path(output_path).write_bytes(blob)
    return {"output_path": output_path, "size_bytes": len(blob)}


def export_cnn_tflite(output_root: str, model_id: str, config: TfliteExportConfig):
    paths = build_artifact_paths(output_root, model_id)
    registry = CnnModelRegistry(paths.registry_path)
    record = registry.require(model_id)
    extractor_info = _convert_model(record["float_extractor_path"], paths.tflite_extractor_path, config)
    classifier_info = _convert_model(record["float_classifier_path"], paths.tflite_classifier_path, config)
    record["tflite_extractor"] = ConversionStatus(status="ready", path=paths.tflite_extractor_path).to_dict()
    record["tflite_classifier"] = ConversionStatus(status="ready", path=paths.tflite_classifier_path).to_dict()
    registry._data["models_by_id"][model_id] = record
    registry.save()
    report = {"model_id": model_id, "config": config.to_dict(), "extractor": extractor_info, "classifier": classifier_info}
    write_json(Path(paths.tflite_dir) / "conversion_report.json", report, indent=2)
    return report
