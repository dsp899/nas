from typing import Any, Dict

from ..config.rnn_config import RnnExperimentConfig
from .rnn_data import DataBundle
from ..common.artifacts import ProjectPaths
from ..common.registries import RnnExperimentRegistry
from .rnn_train import ArchitectureTrainer


def test_rnn(config: RnnExperimentConfig, bundle: DataBundle, paths: ProjectPaths, registry: RnnExperimentRegistry) -> Dict[str, Any]:
    return ArchitectureTrainer(paths, registry).evaluate(config, bundle)
