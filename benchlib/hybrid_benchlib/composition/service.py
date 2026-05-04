from __future__ import annotations

from benchlib_common.artifacts.ids import stable_hash
from benchlib_common.io.jsonl import write_json
from hybrid_benchlib.composition.bundles import load_cnn_bundle, load_rnn_bundle
from hybrid_benchlib.composition.compatibility import validate_compatibility
from hybrid_benchlib.config.schemas import HybridBundleRecord, HybridPipelineConfig
from hybrid_benchlib.storage.layout import build_artifact_paths
from hybrid_benchlib.storage.registry import HybridRegistry, utc_now_iso


def compose_hybrid_bundle(output_root: str, cnn_model_id: str, rnn_model_id: str, pipeline_config: HybridPipelineConfig):
    cnn_record = load_cnn_bundle(output_root, cnn_model_id)
    rnn_record = load_rnn_bundle(output_root, rnn_model_id)
    compatibility = validate_compatibility(cnn_record, rnn_record)
    hybrid_model_id = f"hybrid_{stable_hash({'cnn': cnn_model_id, 'rnn': rnn_model_id, 'pipeline': pipeline_config.to_dict()})}"
    paths = build_artifact_paths(output_root, hybrid_model_id)
    record = HybridBundleRecord(
        hybrid_model_id=hybrid_model_id,
        cnn_model_id=cnn_model_id,
        rnn_model_id=rnn_model_id,
        created_at_utc=utc_now_iso(),
        feature_dim=int(compatibility["cnn_feature_dim"]),
        num_classes=int(compatibility["cnn_num_classes"]),
        pipeline_config=pipeline_config.to_dict(),
        compatibility=compatibility,
        references={"cnn_model_dir": cnn_record["model_dir"], "rnn_model_dir": rnn_record["artifacts"]["model_dir"]},
    )
    registry = HybridRegistry(paths.registry_path)
    registry.register(record)
    write_json(paths.metadata_path, record.to_dict(), indent=2)
    return {"hybrid_model_id": hybrid_model_id, "paths": paths.to_dict(), "record": record.to_dict()}
