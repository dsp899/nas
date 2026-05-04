from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ArtifactRef:
    artifact_type: str
    model_id: str
    path: str
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RuntimeArtifactStatus:
    float_ready: bool = False
    tflite_ready: bool = False
    quantized_ready: bool = False
    xmodel_ready: bool = False
    latest_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
