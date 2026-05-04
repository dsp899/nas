from __future__ import annotations

import subprocess
from pathlib import Path
from benchlib_common.io.jsonl import write_json
from cnn_benchlib.config.schemas import ConversionStatus, XilinxCompileConfig
from cnn_benchlib.storage.layout import build_artifact_paths
from cnn_benchlib.storage.registry import CnnModelRegistry


def _write_executable_script(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | 0o111)


def prepare_xilinx_compilation(output_root: str, model_id: str, config: XilinxCompileConfig, execute: bool = False):
    paths = build_artifact_paths(output_root, model_id)
    registry = CnnModelRegistry(paths.registry_path)
    record = registry.require(model_id)
    xilinx_dir = Path(paths.xilinx_dir)
    compile_script = xilinx_dir / "run_compile.sh"
    quantized_extractor = record.get("quantized_extractor", {}).get("path") or paths.quantized_extractor_path
    compile_body = f'''#!/usr/bin/env bash
set -euo pipefail
ARCH_JSON="${{1:-${{ARCH_JSON:-{config.arch_json}}}}}"
if [[ -z "$ARCH_JSON" ]]; then
  echo "Debes pasar el arch.json como primer argumento o exportar ARCH_JSON." >&2
  exit 1
fi
mkdir -p "{Path(paths.xmodel_extractor_path).parent}"
: "${{VAI_C_BIN:={config.vai_c_bin}}}"
"${{VAI_C_BIN}}" -m "{quantized_extractor}" -a "$ARCH_JSON" -o "{Path(paths.xmodel_extractor_path).parent}" -n "{model_id}"
echo "Expected xmodel path: {paths.xmodel_extractor_path}"
'''
    _write_executable_script(compile_script, compile_body)
    if execute:
        subprocess.run([str(compile_script)], check=True)
    record["xmodel_extractor"] = ConversionStatus(status="ready", path=paths.xmodel_extractor_path).to_dict()
    registry._data["models_by_id"][model_id] = record
    registry.save()
    manifest = {"model_id": model_id, "config": config.to_dict(), "compile_script": str(compile_script), "quantized_extractor_path": quantized_extractor, "xmodel_extractor_path": paths.xmodel_extractor_path, "executed": execute}
    write_json(Path(paths.xilinx_dir) / "compile_manifest.json", manifest, indent=2)
    return manifest
