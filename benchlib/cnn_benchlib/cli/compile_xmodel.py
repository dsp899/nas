from __future__ import annotations

import argparse
import json
from cnn_benchlib.compilation.xilinx import prepare_xilinx_compilation
from cnn_benchlib.config.schemas import XilinxCompileConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepara compilación de extractor CNN cuantizado a XModel.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--arch-json", default="/opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU102/arch.json")
    parser.add_argument("--vai-c-bin", default="vai_c_tensorflow2")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    payload = prepare_xilinx_compilation(args.output_root, args.model_id, XilinxCompileConfig(arch_json=args.arch_json, vai_c_bin=args.vai_c_bin), execute=args.execute)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
