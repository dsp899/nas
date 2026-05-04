from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict
from benchlib_common.artifacts.layout import ensure_dir


@dataclass(frozen=True)
class CnnArtifactPaths:
    model_dir: str
    float_extractor_path: str
    float_classifier_path: str
    tflite_extractor_path: str
    tflite_classifier_path: str
    quantized_extractor_path: str
    quantized_classifier_path: str
    xmodel_extractor_path: str
    metadata_path: str
    registry_path: str
    benchmark_dir: str
    xilinx_dir: str
    tflite_dir: str

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


def build_artifact_paths(output_root: str, model_id: str) -> CnnArtifactPaths:
    root = Path(output_root)
    model_dir = root / "cnn_models" / model_id
    float_dir = model_dir / "float"
    tflite_dir = model_dir / "tflite"
    xilinx_dir = model_dir / "xilinx"
    quantized_dir = xilinx_dir / "quantized"
    xmodel_dir = xilinx_dir / "xmodel"
    benchmark_dir = model_dir / "benchmark"
    for p in [model_dir, float_dir, tflite_dir, xilinx_dir, quantized_dir, xmodel_dir, benchmark_dir]:
        ensure_dir(p)
    return CnnArtifactPaths(
        model_dir=str(model_dir),
        float_extractor_path=str(float_dir / "extractor.keras"),
        float_classifier_path=str(float_dir / "classifier.keras"),
        tflite_extractor_path=str(tflite_dir / "extractor.tflite"),
        tflite_classifier_path=str(tflite_dir / "classifier.tflite"),
        quantized_extractor_path=str(quantized_dir / "quantized_extractor.h5"),
        quantized_classifier_path=str(quantized_dir / "quantized_classifier.h5"),
        xmodel_extractor_path=str(xmodel_dir / f"{model_id}.xmodel"),
        metadata_path=str(model_dir / "metadata.json"),
        registry_path=str(root / "cnn_registry.json"),
        benchmark_dir=str(benchmark_dir),
        xilinx_dir=str(xilinx_dir),
        tflite_dir=str(tflite_dir),
    )
