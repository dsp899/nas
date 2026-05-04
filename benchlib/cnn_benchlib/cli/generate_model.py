from __future__ import annotations

import argparse
import json
from cnn_benchlib.config.schemas import CnnExperimentConfig, CnnModelSpec
from cnn_benchlib.generation.service import generate_float_model
from cnn_benchlib.modeling.backbone_specs import get_backbone_spec


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera y persiste un modelo CNN float.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--dataset-profile", choices=("ucf50", "ucf101"), default="ucf101")
    parser.add_argument("--input-size", type=int, default=None)
    parser.add_argument("--pooling-mode", choices=("avg", "max"), default="avg")
    parser.add_argument("--projection-dim", type=int, default=256)
    args = parser.parse_args()
    backbone = get_backbone_spec(args.backbone)
    experiment = CnnExperimentConfig.from_dataset_profile(args.dataset_profile)
    spec = CnnModelSpec(backbone_name=backbone.name, input_size=int(args.input_size or backbone.recommended_size), pooling_mode=args.pooling_mode, projection_dim=args.projection_dim, num_classes=experiment.num_classes)
    payload = generate_float_model(args.output_root, spec, experiment)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
