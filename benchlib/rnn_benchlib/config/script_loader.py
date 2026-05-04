from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, Dict

from rnn_benchlib.config.schemas import SearchSpace


def load_python_module(path: str):
    resolved = str(Path(path).expanduser().resolve())
    spec = importlib.util.spec_from_file_location("rnn_benchlib_user_config", resolved)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"No se pudo cargar fichero de configuración Python: {resolved}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _coerce_search_space(raw: Any) -> SearchSpace:
    defaults = SearchSpace()
    if isinstance(raw, SearchSpace):
        return raw
    if not isinstance(raw, dict):
        raise TypeError("search_space debe ser dict o SearchSpace")
    payload = {}
    for field_name in defaults.__dataclass_fields__.keys():
        value = raw.get(field_name, getattr(defaults, field_name))
        payload[field_name] = tuple(value)
    return SearchSpace(**payload)


def search_space_to_dict(space: SearchSpace) -> Dict[str, Any]:
    return {field_name: list(getattr(space, field_name)) for field_name in space.__dataclass_fields__.keys()}


def _normalize_top_level_sections(raw: Dict[str, Any]) -> Dict[str, Any]:
    lot = dict(raw.get('LOT', raw.get('lot', {})) or {})
    generation = dict(raw.get('GENERATION', raw.get('generation', {})) or {})
    runtime_generation = dict(generation.get('runtime', generation.get('RUNTIME', {})) or {})
    if not runtime_generation:
        runtime_generation = dict(raw.get('GENERATION_RUNTIME', raw.get('generation_runtime', {})) or {})
    benchmark = dict(raw.get('BENCHMARK', raw.get('benchmark', {})) or {})
    dataset = dict(raw.get('DATASET', raw.get('dataset', {})) or {})
    split = dict(raw.get('SPLIT', raw.get('split', {})) or {})
    gnn = dict(raw.get('GNN', raw.get('gnn', {})) or {})
    storage = dict(raw.get('STORAGE', raw.get('storage', {})) or {})
    resource_manager = dict(raw.get('RESOURCE_MANAGER', raw.get('resource_manager', {})) or {})

    if not lot and {'search_space', 'generation_seed', 'requested_count'} <= set(raw.keys()):
        lot = {
            'search_space': raw['search_space'],
            'generation_seed': raw['generation_seed'],
            'requested_count': raw['requested_count'],
            'experiment': dict(raw.get('experiment', {})),
        }
        runtime_generation = dict(raw.get('generation_runtime', raw.get('runtime', runtime_generation)) or {})
        benchmark = dict(raw.get('benchmark', benchmark) or {})
        dataset = dict(raw.get('dataset', dataset) or {})
        split = dict(raw.get('split', split) or {})
        gnn = dict(raw.get('gnn', gnn) or {})
        storage = dict(raw.get('storage', storage) or {})
        resource_manager = dict(raw.get('resource_manager', resource_manager) or {})
        generation = {'runtime': runtime_generation}

    if not lot:
        raise RuntimeError("El fichero de configuración debe definir la sección LOT.")

    search_space = _coerce_search_space(lot['search_space'])
    experiment = dict(lot.get('experiment', {}))
    return {
        'search_space': search_space,
        'generation_seed': int(lot['generation_seed']),
        'requested_count': int(lot['requested_count']),
        'experiment': experiment,
        'generation': {'runtime': runtime_generation},
        'generation_runtime': runtime_generation,
        'benchmark': benchmark,
        'dataset': dataset,
        'split': split,
        'gnn': gnn,
        'storage': storage,
        'resource_manager': resource_manager,
    }


def _load_json_config(path: str) -> Dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    try:
        payload = json.loads(resolved.read_text(encoding='utf-8'))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"No se pudo parsear el fichero JSON de configuración: {resolved}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("El fichero JSON de configuración debe contener un objeto en la raíz.")
    return _normalize_top_level_sections(payload)


def load_rnn_config(path: str) -> Dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    suffix = resolved.suffix.lower()
    if suffix == '.json':
        return _load_json_config(str(resolved))

    # Fallback de compatibilidad: configuración Python
    module = load_python_module(str(resolved))
    if hasattr(module, 'LOT'):
        lot = dict(getattr(module, 'LOT'))
        generation = dict(getattr(module, 'GENERATION', {}))
        runtime_generation = dict(getattr(module, 'GENERATION_RUNTIME', {}))
        if generation and 'runtime' not in generation and runtime_generation:
            generation['runtime'] = runtime_generation
        benchmark = dict(getattr(module, 'BENCHMARK', {}))
        dataset = dict(getattr(module, 'DATASET', {}))
        split = dict(getattr(module, 'SPLIT', {}))
        gnn = dict(getattr(module, 'GNN', {}))
        storage = dict(getattr(module, 'STORAGE', {}))
        resource_manager = dict(getattr(module, 'RESOURCE_MANAGER', {}))
        return _normalize_top_level_sections({
            'LOT': lot,
            'GENERATION': generation,
            'GENERATION_RUNTIME': runtime_generation,
            'BENCHMARK': benchmark,
            'DATASET': dataset,
            'SPLIT': split,
            'GNN': gnn,
            'STORAGE': storage,
            'RESOURCE_MANAGER': resource_manager,
        })

    search_space = _coerce_search_space(getattr(module, 'SEARCH_SPACE'))
    generation_seed = int(getattr(module, 'GENERATION_SEED'))
    requested_count = int(getattr(module, 'REQUESTED_COUNT'))
    return _normalize_top_level_sections({
        'LOT': {
            'search_space': search_space,
            'generation_seed': generation_seed,
            'requested_count': requested_count,
            'experiment': dict(getattr(module, 'EXPERIMENT', {})),
        },
        'GENERATION': {'runtime': dict(getattr(module, 'RUNTIME', {}))},
        'BENCHMARK': dict(getattr(module, 'BENCHMARK', {})),
        'DATASET': dict(getattr(module, 'DATASET', {})),
        'SPLIT': dict(getattr(module, 'SPLIT', {})),
        'GNN': dict(getattr(module, 'GNN', {})),
        'STORAGE': dict(getattr(module, 'STORAGE', {})),
        'RESOURCE_MANAGER': dict(getattr(module, 'RESOURCE_MANAGER', {})),
    })
