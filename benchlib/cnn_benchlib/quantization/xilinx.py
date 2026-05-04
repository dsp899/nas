from __future__ import annotations

import subprocess
from pathlib import Path
import numpy as np
from benchlib_common.io.jsonl import write_json
from benchlib_common.synthetic.images import SyntheticVideoImageSpec, generate_synthetic_video_frames
from cnn_benchlib.config.schemas import ConversionStatus, XilinxQuantConfig
from cnn_benchlib.storage.layout import build_artifact_paths
from cnn_benchlib.storage.registry import CnnModelRegistry


def _write_executable_script(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | 0o111)


def prepare_xilinx_quantization(output_root: str, model_id: str, config: XilinxQuantConfig, execute: bool = False):
    paths = build_artifact_paths(output_root, model_id)
    registry = CnnModelRegistry(paths.registry_path)
    record = registry.require(model_id)
    xilinx_dir = Path(paths.xilinx_dir)
    calib_dir = xilinx_dir / "calibration"
    calib_dir.mkdir(parents=True, exist_ok=True)
    input_size = int(record["notes"].get("input_size", record["spec"]["input_size"]))
    frames = generate_synthetic_video_frames(SyntheticVideoImageSpec(num_videos=max(1, config.calibration_samples // 8), frames_per_video=8, image_size=input_size, seed=config.calibration_seed)).reshape(-1, input_size, input_size, 3)[: config.calibration_samples]
    calib_path = calib_dir / "frames.npy"
    np.save(calib_path, frames.astype("float32"))
    quant_py = xilinx_dir / "quantize_model.py"
    run_quantize = xilinx_dir / "run_quantize.sh"
    quant_body = f'''#!/usr/bin/env python3
import os
import numpy as np
import tensorflow as tf
FLOAT_EXTRACTOR_PATH = r"{record['float_extractor_path']}"
FLOAT_CLASSIFIER_PATH = r"{record['float_classifier_path']}"
CALIB_FRAMES_PATH = r"{calib_path}"
OUTPUT_EXTRACTOR_H5 = r"{paths.quantized_extractor_path}"
OUTPUT_CLASSIFIER_H5 = r"{paths.quantized_classifier_path}"
INPUT_SHAPE = [1, {input_size}, {input_size}, 3]
try:
    from tensorflow_model_optimization.quantization.keras import vitis_quantize
except Exception as exc:
    raise SystemExit(f"No se pudo importar vitis_quantize: {{exc}}")
frames = np.load(CALIB_FRAMES_PATH).astype("float32")
limit = int(os.environ.get("CALIB_LIMIT", "0") or 0)
if limit > 0:
    frames = frames[:limit]
if frames.size == 0:
    raise SystemExit(f"El dataset de calibración está vacío: {{CALIB_FRAMES_PATH}}")
def quantize_to_path(source_model_path, output_h5):
    model = tf.keras.models.load_model(source_model_path, compile=False)
    quantizer = vitis_quantize.VitisQuantizer(model)
    quantized_model = quantizer.quantize_model(calib_dataset=frames, input_shape=INPUT_SHAPE)
    os.makedirs(os.path.dirname(output_h5), exist_ok=True)
    quantized_model.save(output_h5)
    print(f"Quantized model written to {{output_h5}}")
quantize_to_path(FLOAT_EXTRACTOR_PATH, OUTPUT_EXTRACTOR_H5)
quantize_to_path(FLOAT_CLASSIFIER_PATH, OUTPUT_CLASSIFIER_H5)
'''
    shell_body = f'''#!/usr/bin/env bash
set -euo pipefail
python3 "{quant_py}"
'''
    _write_executable_script(quant_py, quant_body)
    _write_executable_script(run_quantize, shell_body)
    if execute:
        subprocess.run([str(run_quantize)], check=True)
    record["quantized_extractor"] = ConversionStatus(status="ready", path=paths.quantized_extractor_path).to_dict()
    record["quantized_classifier"] = ConversionStatus(status="ready", path=paths.quantized_classifier_path).to_dict()
    record["xilinx_bundle_dir"] = str(xilinx_dir)
    registry._data["models_by_id"][model_id] = record
    registry.save()
    manifest = {"model_id": model_id, "config": config.to_dict(), "calibration_frames": str(calib_path), "quantize_python_script": str(quant_py), "run_quantize_script": str(run_quantize), "quantized_extractor_path": paths.quantized_extractor_path, "quantized_classifier_path": paths.quantized_classifier_path, "executed": execute}
    write_json(xilinx_dir / "quantization_manifest.json", manifest, indent=2)
    return manifest
