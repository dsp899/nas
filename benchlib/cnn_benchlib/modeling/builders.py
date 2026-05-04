from __future__ import annotations

from typing import Dict, Tuple, Any
from cnn_benchlib.config.schemas import CnnExperimentConfig, CnnModelSpec
from cnn_benchlib.modeling.backbone_specs import get_backbone_spec


def build_feature_extractor_and_classifier(spec: CnnModelSpec, experiment: CnnExperimentConfig) -> Tuple[Any, Any, Dict[str, int]]:
    import tensorflow as tf

    backbone_spec = get_backbone_spec(spec.backbone_name)
    inputs = tf.keras.Input(shape=(spec.input_size, spec.input_size, 3), name="image", dtype=tf.float32)
    x = backbone_spec.preprocess_input(inputs)
    base = backbone_spec.builder(include_top=False, weights="imagenet", input_tensor=x)
    if spec.pooling_mode == "avg":
        features = tf.keras.layers.GlobalAveragePooling2D(name="backbone_pool")(base.output)
    else:
        features = tf.keras.layers.GlobalMaxPooling2D(name="backbone_pool")(base.output)
    clip_embedding = tf.keras.layers.Dense(spec.projection_dim, activation=None, name="clip_embedding")(features)
    extractor = tf.keras.Model(inputs=inputs, outputs=clip_embedding, name=f"{spec.backbone_name}_extractor")

    classifier_inputs = tf.keras.Input(shape=(spec.input_size, spec.input_size, 3), name="image", dtype=tf.float32)
    classifier_features = extractor(classifier_inputs)
    hidden = tf.keras.layers.Dense(spec.projection_dim, activation="relu", name="head_dense")(classifier_features)
    logits = tf.keras.layers.Dense(experiment.num_classes, activation=None, name="classifier_logits")(hidden)
    classifier = tf.keras.Model(inputs=classifier_inputs, outputs=logits, name=f"{spec.backbone_name}_classifier")

    metadata = {"feature_dim": int(spec.projection_dim), "num_classes": int(experiment.num_classes), "input_size": int(spec.input_size)}
    return extractor, classifier, metadata
