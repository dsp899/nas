from __future__ import annotations

import argparse
import json
from cnn_benchlib.config.schemas import XilinxQuantConfig
from cnn_benchlib.quantization.xilinx import prepare_xilinx_quantization


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepara cuantización Vitis AI para un modelo CNN.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--calibration-samples", type=int, default=64)
    parser.add_argument("--calibration-seed", type=int, default=1234)
    parser.add_argument("--arch-json", default="/opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU102/arch.json")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    payload = prepare_xilinx_quantization(args.output_root, args.model_id, XilinxQuantConfig(calibration_samples=args.calibration_samples, calibration_seed=args.calibration_seed, arch_json=args.arch_json), execute=args.execute)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
