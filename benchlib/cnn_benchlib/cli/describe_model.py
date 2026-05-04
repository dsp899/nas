from __future__ import annotations

import argparse
import json
from cnn_benchlib.storage.layout import build_artifact_paths
from cnn_benchlib.storage.registry import CnnModelRegistry


def main() -> None:
    parser = argparse.ArgumentParser(description="Describe un modelo CNN persistido.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--model-id", required=True)
    args = parser.parse_args()
    paths = build_artifact_paths(args.output_root, args.model_id)
    registry = CnnModelRegistry(paths.registry_path)
    payload = registry.require(args.model_id)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
