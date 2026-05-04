from __future__ import annotations

import argparse
import json
from cnn_benchlib.config.schemas import TfliteExportConfig
from cnn_benchlib.conversion.tflite import export_cnn_tflite


def main() -> None:
    parser = argparse.ArgumentParser(description="Exporta extractor y clasificador CNN a TFLite.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--optimize-default", action="store_true")
    parser.add_argument("--allow-select-tf-ops", action="store_true")
    args = parser.parse_args()
    payload = export_cnn_tflite(args.output_root, args.model_id, TfliteExportConfig(optimize_default=args.optimize_default, allow_select_tf_ops=args.allow_select_tf_ops))
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
