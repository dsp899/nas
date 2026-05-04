
import csv
import json
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import numpy as np

from .nas_controller import SearchController
from ..rnn.rnn_data import SequenceRepository
from ..common.runtime import process_memory_mb
from ..rnn.rnn_train import ArchitectureTrainer
from ..common.artifacts import ProjectPaths
from .nas_search_space import SearchSpace
from ..config.nas_config import NasSearchSpaceConfig
from ..config.rnn_config import RnnArchitectureConfig, RnnDataConfig, RnnExperimentConfig
from ..common.registries import NasSearchRegistry, RnnExperimentRegistry, nas_search_signature, nas_search_tag
from typing import Set, Any, Dict, List, Tuple, Union
from collections import Counter


class SearchRunLogger:
    def __init__(self, search_run_dir: Union[str, Path], *, search_signature: str, run_id: str, search_experiment_dir: Union[str, Path], rolling_window: int, search_space: NasSearchSpaceConfig, baseline_strategy: str, controller_config: Dict[str, Any], candidate_runtime: Dict[str, Any], candidate_optimizer: Dict[str, Any]) -> None:
        search_run_dir = Path(search_run_dir)
        search_run_dir.mkdir(parents=True, exist_ok=True)
        self.search_signature = search_signature
        self.run_id = run_id
        self.experiment_dir = Path(search_experiment_dir)
        self.run_dir = search_run_dir
        self.json_path = search_run_dir / "search_run.json"
        self.architectures_csv = search_run_dir / "search_architectures.csv"
        self.controller_csv = search_run_dir / "search_controller_history.csv"
        self.summary_path = search_run_dir / "search_summary.json"
        self.rolling_window = max(1, rolling_window)
        self.sample_counter = 0
        self.controller_step_counter = 0
        self.all_samples: List[Dict[str, Any]] = []
        self.payload: Dict[str, Any] = {
            "started_at": datetime.utcnow().isoformat(),
            "nas_search_signature": search_signature,
            "nas_run_id": run_id,
            "search_experiment_dir": str(self.experiment_dir),
            "search_run_dir": str(self.run_dir),
            "rolling_window": self.rolling_window,
            "search_space": {
                "variable_dimensions": list(search_space.variable_dimensions),
                "fixed_dimensions": list(search_space.fixed_dimensions),
                "options": search_space.to_dict(),
            },
            "reward_strategy": {
                "baseline_strategy": baseline_strategy,
            },
            "controller_config": controller_config,
            "candidate_runtime": candidate_runtime,
            "candidate_optimizer": candidate_optimizer,
            "epochs": [],
            "best_architectures": [],
            "global_stats": {},
            "artifacts": {
                "architectures_csv": str(self.architectures_csv),
                "controller_history_csv": str(self.controller_csv),
                "summary_json": str(self.summary_path),
            },
        }

    @staticmethod
    def _append_csv(path: Path, row: Dict[str, Any]) -> None:
        file_exists = path.exists()
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def _global_stats(self) -> Dict[str, Any]:
        if not self.all_samples:
            return {
                "num_samples": 0,
                "mean_accuracy": 0.0,
                "median_accuracy": 0.0,
                "max_accuracy": 0.0,
                "min_accuracy": 0.0,
                "rolling_mean_accuracy": 0.0,
                "rolling_median_accuracy": 0.0,
                "rolling_max_accuracy": 0.0,
                "mean_advantage": 0.0,
                "mean_advantage_used": 0.0,
            }
        accuracies = np.asarray([sample["metrics"]["accuracy"] for sample in self.all_samples], dtype=np.float64)
        advantages = np.asarray([sample["metrics"]["raw_reward"] for sample in self.all_samples], dtype=np.float64)
        advantages_used = np.asarray([sample["metrics"]["normalized_reward"] for sample in self.all_samples], dtype=np.float64)
        rolling = accuracies[-self.rolling_window :]
        return {
            "num_samples": int(accuracies.size),
            "mean_accuracy": float(np.mean(accuracies)),
            "median_accuracy": float(np.median(accuracies)),
            "max_accuracy": float(np.max(accuracies)),
            "min_accuracy": float(np.min(accuracies)),
            "rolling_mean_accuracy": float(np.mean(rolling)),
            "rolling_median_accuracy": float(np.median(rolling)),
            "rolling_max_accuracy": float(np.max(rolling)),
            "mean_advantage": float(np.mean(advantages)),
            "mean_advantage_used": float(np.mean(advantages_used)),
        }

    @staticmethod
    def _layers_from_units(units: Tuple[int, int, int]) -> int:
        return int(sum(1 for unit in units if int(unit) != 0))

    @staticmethod
    def _format_distribution(values: List[int]) -> str:
        if not values:
            return "-"
        counts = Counter(int(value) for value in values)
        return ", ".join(f"{key}:{counts[key]}" for key in sorted(counts))

    def append_epoch(self, epoch_payload: Dict[str, Any]) -> None:
        self.payload["epochs"].append(epoch_payload)

        for architecture in epoch_payload["architectures"]:
            self.sample_counter += 1
            config = architecture["config"]
            metrics = architecture["metrics"]
            units = tuple(config["architecture"]["units"])
            csv_row = {
                "global_sample_order": self.sample_counter,
                "search_epoch": epoch_payload["epoch"],
                "sample_order_in_epoch": architecture["sample_order_in_epoch"],
                "sampled_dimensions": json.dumps(sorted(architecture.get("sampled_dimensions", {}).keys())),
                "sampled_values": json.dumps(architecture.get("sampled_dimensions", {}), sort_keys=True),
                "layers": self._layers_from_units(units),
                "rnn": config["architecture"]["rnn"],
                "units_0": units[0],
                "units_1": units[1],
                "units_2": units[2],
                "direction": config["architecture"]["direction"],
                "memory_mode": config["architecture"]["memory_mode"],
                "seq": config["data"]["seq"],
                "head_units": config["architecture"]["head_units"],
                "video_decision": config["architecture"]["video_decision"],
                "video_decision_input": config["architecture"]["video_decision_input"],
                "cnn": config["data"]["cnn"],
                "encoded": json.dumps(architecture["encoded"]),
                "decoded": json.dumps(architecture.get("sampled_dimensions", {}), sort_keys=True),
                "signature": metrics["signature"],
                "rnn_data_signature": metrics["rnn_data_signature"],
                "rnn_architecture_signature": metrics["rnn_architecture_signature"],
                "rnn_runtime_signature": metrics["rnn_runtime_signature"],
                "accuracy": metrics["accuracy"],
                "search_accuracy": metrics.get("search_accuracy"),
                "val_accuracy": metrics.get("val_accuracy"),
                "test_accuracy": metrics.get("test_accuracy"),
                "search_metric_source": metrics.get("search_metric_source"),
                "raw_reward": metrics["raw_reward"],
                "normalized_reward": metrics["normalized_reward"],
                "baseline_value_used": metrics.get("baseline_value_used"),
                "cached": metrics["cached"],
                "artifact_consistent": metrics.get("artifact_consistent", False),
                "model_path": metrics["model_path"],
                "baseline_before": epoch_payload.get("baseline_before"),
                "baseline_after": epoch_payload.get("baseline_after"),
                "baseline_strategy": epoch_payload.get("baseline_strategy"),
            }
            self._append_csv(self.architectures_csv, csv_row)
            self.all_samples.append(architecture)

        controller_losses = epoch_payload.get("controller_loss", [])
        for controller_epoch, loss in enumerate(controller_losses, start=1):
            self.controller_step_counter += 1
            self._append_csv(
                self.controller_csv,
                {
                    "global_controller_step": self.controller_step_counter,
                    "search_epoch": epoch_payload["epoch"],
                    "controller_epoch": controller_epoch,
                    "loss": loss,
                    "learning_rate": epoch_payload.get("controller_learning_rate_history", [None] * len(controller_losses))[controller_epoch - 1],
                    "learning_rate_reduced": epoch_payload.get("controller_learning_rate_reduced_history", [None] * len(controller_losses))[controller_epoch - 1],
                    "num_sampled_architectures": len(epoch_payload["architectures"]),
                },
            )

        all_architectures = sorted(self.all_samples, key=lambda item: item["metrics"]["accuracy"], reverse=True)
        self.payload["best_architectures"] = all_architectures[:20]
        self.payload["global_stats"] = self._global_stats()
        self.json_path.write_text(json.dumps(self.payload, indent=2, sort_keys=True), encoding="utf-8")
        self.summary_path.write_text(json.dumps(self._build_summary(), indent=2, sort_keys=True), encoding="utf-8")

    def _build_summary(self) -> Dict[str, Any]:
        per_epoch = []
        cumulative: List[float] = []
        for epoch in self.payload["epochs"]:
            epoch_accuracies = [item["metrics"]["accuracy"] for item in epoch["architectures"]]
            cumulative.extend(epoch_accuracies)
            cumulative_np = np.asarray(cumulative, dtype=np.float64)
            rolling_np = cumulative_np[-self.rolling_window :]
            per_epoch.append(
                {
                    "epoch": epoch["epoch"],
                    "num_sampled_architectures": len(epoch_accuracies),
                    "epoch_mean_accuracy": float(np.mean(epoch_accuracies)) if epoch_accuracies else 0.0,
                    "epoch_median_accuracy": float(np.median(epoch_accuracies)) if epoch_accuracies else 0.0,
                    "epoch_max_accuracy": float(np.max(epoch_accuracies)) if epoch_accuracies else 0.0,
                    "cumulative_mean_accuracy": float(np.mean(cumulative_np)) if cumulative else 0.0,
                    "cumulative_median_accuracy": float(np.median(cumulative_np)) if cumulative else 0.0,
                    "cumulative_max_accuracy": float(np.max(cumulative_np)) if cumulative else 0.0,
                    "rolling_mean_accuracy": float(np.mean(rolling_np)) if cumulative else 0.0,
                    "rolling_median_accuracy": float(np.median(rolling_np)) if cumulative else 0.0,
                    "rolling_max_accuracy": float(np.max(rolling_np)) if cumulative else 0.0,
                    "baseline_strategy": epoch.get("baseline_strategy"),
                    "baseline_value_used": epoch.get("baseline_value_used"),
                    "controller_loss": list(epoch.get("controller_loss", [])),
                    "controller_learning_rate_history": list(epoch.get("controller_learning_rate_history", [])),
                    "controller_last_loss": float(epoch.get("controller_loss", [])[-1]) if epoch.get("controller_loss") else None,
                    "controller_last_learning_rate": float(epoch.get("controller_learning_rate_history", [])[-1]) if epoch.get("controller_learning_rate_history") else None,
                    "sampling_attempts": int(epoch.get("sampling_attempts", 0)),
                    "sampling_guided_attempts": int(epoch.get("sampling_guided_attempts", 0)),
                    "sampling_fallback_attempts": int(epoch.get("sampling_fallback_attempts", 0)),
                    "sampling_duplicate_hits": int(epoch.get("sampling_duplicate_hits", 0)),
                    "sampling_used_fallback": bool(epoch.get("sampling_used_fallback", False)),
                    "epoch_layer_distribution": dict(epoch.get("epoch_layer_distribution", {})),
                    "global_layer_distribution": dict(epoch.get("global_layer_distribution", {})),
                }
            )
        return {
            "started_at": self.payload["started_at"],
            "rolling_window": self.rolling_window,
            "search_space": self.payload["search_space"],
            "reward_strategy": self.payload["reward_strategy"],
            "global_stats": self._global_stats(),
            "per_epoch": per_epoch,
            "artifact_paths": self.payload["artifacts"],
        }


class NasSearchEngine:
    def __init__(self, config: RnnExperimentConfig, paths: ProjectPaths, registry: RnnExperimentRegistry, search_registry: NasSearchRegistry) -> None:
        self.config = config
        self.paths = paths
        self.registry = registry
        self.search_registry = search_registry
        self.search_signature = nas_search_signature(config)
        self.search_tag = nas_search_tag(config)
        self.search_space = SearchSpace(config.search_space)
        self.total_valid_architectures = self.search_space.count_valid_sequences()
        self.planned_total_samples = int(config.nas.controller_sampling_epochs) * int(config.nas.controller_samples_per_epoch)
        self.run_id = datetime.utcnow().strftime("run_%Y%m%dT%H%M%S_%fZ")
        self.search_experiment_dir = paths.search_experiment_dir(config.data.partition_tag, self.search_signature)
        self.search_run_dir = paths.search_run_dir(config.data.partition_tag, self.search_signature, self.run_id)
        self.controller = SearchController(config.nas, self.search_space)
        self.repo = SequenceRepository(paths)
        self.trainer = ArchitectureTrainer(paths, registry)
        self.logger = SearchRunLogger(
            self.search_run_dir,
            search_signature=self.search_signature,
            run_id=self.run_id,
            search_experiment_dir=self.search_experiment_dir,
            rolling_window=config.nas.effective_rolling_window,
            search_space=config.search_space,
            baseline_strategy=config.nas.reward_baseline_strategy,
            controller_config=asdict(config.nas),
            candidate_runtime=asdict(config.runtime),
            candidate_optimizer=asdict(config.optimizer),
        )
        self.baseline = 0.0
        self.seen_sequences: Set[Tuple[int, ...]] = set()

    def _apply_layers(self, units: List[int], layer_count: int) -> List[int]:
        adjusted = list(units)
        for index in range(len(adjusted)):
            if index >= layer_count:
                adjusted[index] = 0

        for index in range(min(layer_count, len(adjusted))):
            if adjusted[index] != 0:
                continue
            dimension = f"units_{index}"
            options = [int(value) for value in self.config.search_space.options(dimension) if int(value) != 0]
            if not options:
                raise ValueError(
                    f"La configuración del search space es inconsistente: '{dimension}' no tiene valores no nulos "
                    f"pero se ha solicitado una arquitectura de {layer_count} capas."
                )
            adjusted[index] = options[0]
        return adjusted

    def _child_config_from_sequence(self, sequence: List[int]) -> Tuple[RnnExperimentConfig, Dict[str, Any]]:
        sampled = self.search_space.decode_dict(sequence)
        data = RnnDataConfig(
            cnn=str(sampled.get("cnn", self.config.data.cnn)),
            name=self.config.data.name,
            frames=self.config.data.frames,
            image_size=self.config.data.image_size,
            seq=int(sampled.get("seq", self.config.data.seq)),
            split=self.config.data.split,
            val_fraction=self.config.data.val_fraction,
            partition_mode=self.config.data.partition_mode,
            sampling=self.config.data.sampling,
            resize_mode=self.config.data.resize_mode,
            cnn_training_signature=self.config.data.cnn_training_signature,
            cnn_feature_export_signature=self.config.data.cnn_feature_export_signature,
        )
        units = [int(unit) for unit in self.config.architecture.units]
        if "units_0" in sampled:
            units[0] = int(sampled["units_0"])
        if "units_1" in sampled:
            units[1] = int(sampled["units_1"])
        if "units_2" in sampled:
            units[2] = int(sampled["units_2"])
        layer_count = int(sampled["layers"]) if "layers" in sampled else sum(1 for unit in units if unit != 0)
        units = self._apply_layers(units, layer_count)
        architecture = RnnArchitectureConfig(
            rnn=str(sampled.get("rnn", self.config.architecture.rnn)),
            direction=str(sampled.get("direction", self.config.architecture.direction)),
            units=tuple(units),
            memory_mode=str(sampled.get("memory_mode", self.config.architecture.memory_mode)),
            head_units=int(sampled.get("head_units", self.config.architecture.head_units)),
            video_decision=str(sampled.get("video_decision", self.config.architecture.video_decision)),
            video_decision_input=str(sampled.get("video_decision_input", self.config.architecture.video_decision_input)),
        )
        runtime = replace(self.config.runtime)
        return self.config.for_architecture(operation="train", architecture=architecture, data=data, runtime=runtime), sampled

    def _batch_baseline(self, accuracies: List[float]) -> float:
        return float(np.mean(accuracies)) if accuracies else 0.0

    def _ema_baseline(self, accuracies: List[float]) -> float:
        if not accuracies:
            return self.baseline
        if self.baseline == 0.0:
            return float(np.mean(accuracies))
        return self.baseline

    def _compute_advantages(self, accuracies: List[float]) -> Tuple[List[float], List[float], float, float]:
        accuracies_np = np.asarray(accuracies, dtype=np.float32)
        strategy = self.config.nas.reward_baseline_strategy
        if strategy == "batch":
            baseline_used = self._batch_baseline(accuracies)
            raw_advantages = accuracies_np - baseline_used
            baseline_after = baseline_used
        else:
            baseline_used = self._ema_baseline(accuracies)
            raw_advantages = accuracies_np - baseline_used
            epoch_mean = float(np.mean(accuracies_np)) if accuracies else 0.0
            if self.baseline == 0.0:
                baseline_after = epoch_mean
            else:
                decay = self.config.nas.reward_baseline_ema_decay
                baseline_after = decay * self.baseline + (1.0 - decay) * epoch_mean

        if self.config.nas.reward_standardize_advantage:
            std = float(np.std(raw_advantages))
            if std > 1e-8:
                used_advantages = raw_advantages / std
            else:
                used_advantages = raw_advantages.copy()
        else:
            used_advantages = raw_advantages.copy()

        return (
            raw_advantages.astype(np.float32).tolist(),
            used_advantages.astype(np.float32).tolist(),
            float(baseline_used),
            float(baseline_after),
        )

    @staticmethod
    def _candidate_label(child_config: RnnExperimentConfig) -> str:
        units = tuple(int(unit) for unit in child_config.architecture.units)
        return (
            f"cnn={child_config.data.cnn} "
            f"rnn={child_config.architecture.rnn} "
            f"units={units} "
            f"dir={child_config.architecture.direction} "
            f"mem={child_config.architecture.memory_mode} "
            f"seq={child_config.data.seq} "
            f"head={child_config.architecture.head_units} "
            f"vd={child_config.architecture.video_decision}/"
            f"{child_config.architecture.video_decision_input}"
        )

    def _print_search_header(self) -> None:
        print("NAS search")
        print("  signature:", self.search_signature)
        print("  run_id:", self.run_id)
        print("  experiment_dir:", self.search_experiment_dir)
        print("  run_dir:", self.search_run_dir)
        print("  variable_dimensions:", len(self.search_space.dimensions))
        print("  valid_architectures:", self.total_valid_architectures)
        print("  samples_per_epoch:", self.config.nas.controller_samples_per_epoch)
        print("  search_epochs:", self.config.nas.controller_sampling_epochs)
        print("  planned_samples:", self.planned_total_samples)
        print("  sampling_attempts_multiplier:", self.config.nas.sampling_attempts_multiplier)
        print("  sampling_attempts_minimum:", self.config.nas.sampling_attempts_minimum)

    def run(self) -> Dict[str, Any]:
        self.search_registry.reserve(
            self.config,
            search_run_dir=self.logger.run_dir,
            search_log_path=self.logger.json_path,
            architectures_csv_path=self.logger.architectures_csv,
            controller_history_csv_path=self.logger.controller_csv,
            summary_path=self.logger.summary_path,
        )
        try:
            self._print_search_header()
            global_sample_index = 0
            total_epochs = int(self.config.nas.controller_sampling_epochs)
            samples_per_epoch = int(self.config.nas.controller_samples_per_epoch)
            global_layer_counts: Counter = Counter()
            for epoch in range(total_epochs):
                sample_result = self.controller.sample_sequences(
                    self.config.nas.controller_samples_per_epoch,
                    seen=self.seen_sequences,
                )
                sequences = sample_result.sequences
                sampled_before_epoch = len(self.logger.all_samples)
                epoch_global_start = sampled_before_epoch + 1
                epoch_global_end = sampled_before_epoch + len(sequences)
                planned_done = sampled_before_epoch
                planned_remaining_before = max(0, self.planned_total_samples - planned_done)
                print(
                    f"[search {epoch + 1}/{total_epochs}] "
                    f"sampling {len(sequences)} candidates | "
                    f"global {epoch_global_start}-{epoch_global_end}/{self.planned_total_samples} | "
                    f"valid_seen={len(self.seen_sequences)}/{self.total_valid_architectures} | "
                    f"valid_remaining={max(0, self.total_valid_architectures - len(self.seen_sequences))} | "
                    f"planned_remaining={planned_remaining_before} | "
                    f"attempts={sample_result.total_attempts} (guided={sample_result.guided_attempts}, fallback={sample_result.fallback_attempts}, duplicates={sample_result.duplicate_hits})"
                )
                if not sequences:
                    print(
                        f"[search {epoch + 1}/{total_epochs}] sampler could not discover new unique candidates "
                        f"within the attempt budget; stopping early at {len(self.logger.all_samples)}/{self.planned_total_samples} sampled"
                    )
                    break
                epoch_payload = {
                    "epoch": epoch + 1,
                    "baseline_strategy": self.config.nas.reward_baseline_strategy,
                    "baseline_before": self.baseline if self.config.nas.reward_baseline_strategy == "ema" else None,
                    "memory_mb_before": process_memory_mb(),
                    "sampling_attempts": sample_result.total_attempts,
                    "sampling_guided_attempts": sample_result.guided_attempts,
                    "sampling_fallback_attempts": sample_result.fallback_attempts,
                    "sampling_duplicate_hits": sample_result.duplicate_hits,
                    "sampling_used_fallback": bool(sample_result.used_fallback),
                    "architectures": [],
                }
                for sample_order, sequence in enumerate(sequences, start=1):
                    child_config, sampled_dimensions = self._child_config_from_sequence(sequence)
                    resolved_data, _ = self.repo.resolve_data_feature_source(child_config.data)
                    child_config = replace(child_config, data=resolved_data)
                    bundle = self.repo.make_bundle(child_config.data, child_config.runtime.batch_size, child_config.runtime.random_seed)
                    metrics = self.trainer.train_or_resume(child_config, bundle, verbose=False)
                    reward_accuracy = float(metrics.get("best_search_metric_acc", metrics["best_test_acc"]))
                    report_accuracy = float(metrics.get("best_test_acc", reward_accuracy))
                    epoch_payload["architectures"].append(
                        {
                            "sample_order_in_epoch": sample_order,
                            "encoded": sequence,
                            "decoded": list(sampled_dimensions.values()),
                            "sampled_dimensions": sampled_dimensions,
                            "config": child_config.to_dict(),
                            "metrics": {
                                "accuracy": report_accuracy,
                                "search_accuracy": reward_accuracy,
                                "val_accuracy": float(metrics.get("best_val_acc", report_accuracy)),
                                "test_accuracy": float(metrics.get("best_test_acc", report_accuracy)),
                                "search_metric_source": str(metrics.get("search_metric_source", "test")),
                                "raw_reward": 0.0,
                                "normalized_reward": 0.0,
                                "baseline_value_used": None,
                                "cached": bool(metrics.get("cached", False)),
                                "signature": metrics["signature"],
                                "rnn_data_signature": metrics["rnn_data_signature"],
                                "rnn_architecture_signature": metrics["rnn_architecture_signature"],
                                "rnn_runtime_signature": metrics["rnn_runtime_signature"],
                                "artifact_consistent": bool(metrics.get("artifact_consistent", False)),
                                "model_path": metrics["model_path"],
                            },
                        }
                    )
                    global_sample_index += 1
                    status = "cached" if bool(metrics.get("cached", False)) else "trained"
                    print(
                        f"  [candidate {sample_order}/{len(sequences)} | global {global_sample_index}/{self.planned_total_samples}] "
                        f"{self._candidate_label(child_config)} | acc={report_accuracy:.4f} | {status}"
                    )

                report_accuracies = [item["metrics"]["accuracy"] for item in epoch_payload["architectures"]]
                search_accuracies = [item["metrics"]["search_accuracy"] for item in epoch_payload["architectures"]]
                epoch_layer_counts = Counter(
                    self.logger._layers_from_units(tuple(item["config"]["architecture"]["units"]))
                    for item in epoch_payload["architectures"]
                )
                for key, value in epoch_layer_counts.items():
                    global_layer_counts[key] += value
                epoch_payload["epoch_layer_distribution"] = {str(key): int(epoch_layer_counts[key]) for key in sorted(epoch_layer_counts)}
                epoch_payload["global_layer_distribution"] = {str(key): int(global_layer_counts[key]) for key in sorted(global_layer_counts)}
                raw_advantages, used_advantages, baseline_used, baseline_after = self._compute_advantages(search_accuracies)
                for architecture, raw_advantage, used_advantage in zip(epoch_payload["architectures"], raw_advantages, used_advantages):
                    architecture["metrics"]["raw_reward"] = float(raw_advantage)
                    architecture["metrics"]["normalized_reward"] = float(used_advantage)
                    architecture["metrics"]["baseline_value_used"] = float(baseline_used)

                if self.config.nas.reward_baseline_strategy == "ema":
                    self.baseline = baseline_after
                else:
                    self.baseline = baseline_used

                epoch_payload["baseline_value_used"] = baseline_used
                epoch_payload["baseline_after"] = baseline_after
                epoch_payload["mean_accuracy"] = float(np.mean(report_accuracies))
                epoch_payload["median_accuracy"] = float(np.median(report_accuracies))
                epoch_payload["max_accuracy"] = float(np.max(report_accuracies))
                epoch_payload["min_accuracy"] = float(np.min(report_accuracies))
                epoch_payload["mean_search_accuracy"] = float(np.mean(search_accuracies))
                epoch_payload["median_search_accuracy"] = float(np.median(search_accuracies))
                epoch_payload["max_search_accuracy"] = float(np.max(search_accuracies))
                epoch_payload["min_search_accuracy"] = float(np.min(search_accuracies))
                epoch_payload["memory_mb_after"] = process_memory_mb()
                batch = self.controller.prepare_training_batch(sequences, used_advantages)
                history = self.controller.train(batch)
                epoch_payload["controller_loss"] = [float(item) for item in history.history.get("loss", [])]
                epoch_payload["controller_learning_rate_history"] = [float(item) for item in history.history.get("learning_rate", [])]
                epoch_payload["controller_learning_rate_reduced_history"] = [bool(item) for item in history.history.get("learning_rate_reduced", [])]
                self.logger.append_epoch(epoch_payload)
                print(
                    f"[search {epoch + 1}/{total_epochs}] done | "
                    f"epoch_best={epoch_payload['max_accuracy']:.4f} | "
                    f"epoch_mean={epoch_payload['mean_accuracy']:.4f} | "
                    f"global_sampled={len(self.logger.all_samples)}/{self.planned_total_samples} | "
                    f"valid_seen={len(self.seen_sequences)}/{self.total_valid_architectures} | "
                    f"valid_remaining={max(0, self.total_valid_architectures - len(self.seen_sequences))} | "
                    f"epoch_layers={self.logger._format_distribution(list(epoch_layer_counts.elements()))} | "
                    f"global_layers={self.logger._format_distribution(list(global_layer_counts.elements()))}"
                )

            self.search_registry.complete(
                self.search_signature,
                search_run_dir=self.logger.run_dir,
                search_log_path=self.logger.json_path,
                architectures_csv_path=self.logger.architectures_csv,
                controller_history_csv_path=self.logger.controller_csv,
                summary_path=self.logger.summary_path,
            )
            return {
                "nas_search_signature": self.search_signature,
                "nas_search_tag": self.search_tag,
                "search_experiment_dir": str(self.search_experiment_dir),
                "search_run_dir": str(self.logger.run_dir),
                "nas_run_id": self.run_id,
                "search_log": str(self.logger.json_path),
                "search_summary": str(self.logger.summary_path),
                "sampled_architectures_csv": str(self.logger.architectures_csv),
                "controller_history_csv": str(self.logger.controller_csv),
                "best_architectures": self.registry.top_completed(limit=10),
                "search_space": self.config.search_space.to_dict(),
                "variable_dimensions": list(self.config.search_space.variable_dimensions),
                "baseline_strategy": self.config.nas.reward_baseline_strategy,
            }
        except Exception as exc:
            self.search_registry.fail(self.search_signature, str(exc))
            raise
