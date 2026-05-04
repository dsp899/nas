import json
from pathlib import Path
from typing import Any, Dict, Optional, Union



def load_json_config(path: Union[str, Path]) -> Dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"No existe el fichero de configuración: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("El fichero de configuración debe contener un objeto JSON en la raíz")
    return payload



def load_optional_json_config(path: Optional[Union[str, Path]]) -> Dict[str, Any]:
    if path is None:
        return {}
    if isinstance(path, str) and not path.strip():
        return {}
    return load_json_config(path)



def nested_get(mapping: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current



def merge_with_defaults(payload: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge that keeps defaults when payload omits a branch."""
    result: Dict[str, Any] = dict(defaults)
    for key, value in payload.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_with_defaults(value, result[key])
        else:
            result[key] = value
    return result



def resolve_config_payload(config_path: Optional[Union[str, Path]], defaults: Dict[str, Any]) -> Dict[str, Any]:
    return merge_with_defaults(load_optional_json_config(config_path), defaults)
