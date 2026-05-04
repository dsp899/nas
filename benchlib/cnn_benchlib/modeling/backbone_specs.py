from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Callable, Tuple, Any


@dataclass(frozen=True)
class BackboneSpec:
    name: str
    builder_path: str
    preprocess_input_path: str
    recommended_size: int
    aliases: Tuple[str, ...] = ()

    @property
    def builder(self) -> Callable[..., Any]:
        module_name, attr_name = self.builder_path.rsplit('.', 1)
        return getattr(import_module(module_name), attr_name)

    @property
    def preprocess_input(self) -> Callable[..., Any]:
        module_name, attr_name = self.preprocess_input_path.rsplit('.', 1)
        return getattr(import_module(module_name), attr_name)


BACKBONE_SPECS: Tuple[BackboneSpec, ...] = (
    BackboneSpec(
        name="vgg16",
        builder_path="tensorflow.keras.applications.VGG16",
        preprocess_input_path="tensorflow.keras.applications.vgg16.preprocess_input",
        recommended_size=224,
    ),
    BackboneSpec(
        name="resnet50",
        builder_path="tensorflow.keras.applications.ResNet50",
        preprocess_input_path="tensorflow.keras.applications.resnet50.preprocess_input",
        recommended_size=224,
    ),
    BackboneSpec(
        name="inceptionV3",
        builder_path="tensorflow.keras.applications.InceptionV3",
        preprocess_input_path="tensorflow.keras.applications.inception_v3.preprocess_input",
        recommended_size=299,
        aliases=("inceptionv3",),
    ),
)

BACKBONE_REGISTRY = {spec.name: spec for spec in BACKBONE_SPECS}
for spec in BACKBONE_SPECS:
    for alias in spec.aliases:
        BACKBONE_REGISTRY[alias] = spec

BACKBONE_NAMES = tuple(spec.name for spec in BACKBONE_SPECS)


def get_backbone_spec(name: str) -> BackboneSpec:
    try:
        return BACKBONE_REGISTRY[name]
    except KeyError as exc:
        supported = ", ".join(BACKBONE_NAMES)
        raise ValueError(f"CNN no soportada: {name!r}. Opciones disponibles: {supported}") from exc
