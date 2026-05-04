from __future__ import annotations

import argparse
import json
from hybrid_benchlib.storage.layout import build_artifact_paths
from hybrid_benchlib.storage.registry import HybridRegistry


def main() -> None:
    parser = argparse.ArgumentParser(description="Describe un bundle híbrido persistido.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--hybrid-model-id", required=True)
    args = parser.parse_args()
    paths = build_artifact_paths(args.output_root, args.hybrid_model_id)
    registry = HybridRegistry(paths.registry_path)
    payload = registry._data["hybrids_by_id"][args.hybrid_model_id]
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
