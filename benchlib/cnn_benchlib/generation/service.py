from __future__ import annotations

from benchlib_common.io.jsonl import write_json
from cnn_benchlib.config.schemas import CnnArtifactRecord, CnnExperimentConfig, CnnModelSpec
from cnn_benchlib.modeling.builders import build_feature_extractor_and_classifier
from cnn_benchlib.storage.layout import build_artifact_paths
from cnn_benchlib.storage.registry import CnnModelRegistry, model_spec_to_id, utc_now_iso


def generate_float_model(output_root: str, spec: CnnModelSpec, experiment: CnnExperimentConfig):
    model_id = model_spec_to_id(spec, experiment)
    paths = build_artifact_paths(output_root, model_id)
    extractor, classifier, metadata = build_feature_extractor_and_classifier(spec, experiment)
    extractor.save(paths.float_extractor_path)
    classifier.save(paths.float_classifier_path)
    record = CnnArtifactRecord(
        model_id=model_id,
        spec=spec,
        experiment=experiment,
        created_at_utc=utc_now_iso(),
        model_dir=paths.model_dir,
        float_extractor_path=paths.float_extractor_path,
        float_classifier_path=paths.float_classifier_path,
        feature_dim=metadata["feature_dim"],
        notes={"input_size": metadata["input_size"]},
    )
    registry = CnnModelRegistry(paths.registry_path)
    registry.register(record)
    write_json(paths.metadata_path, record.to_dict(), indent=2)
    return {"model_id": model_id, "paths": paths.to_dict(), "record": record.to_dict(), "metadata": metadata}
