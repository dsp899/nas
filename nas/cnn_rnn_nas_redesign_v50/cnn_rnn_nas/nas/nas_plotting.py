import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SearchExperimentFiles:
    experiment_id: str
    summary_json: Path
    architectures_csv: Path
    controller_csv: Path


@dataclass(frozen=True)
class SearchAnalysisArtifacts:
    output_dir: Path
    tables_dir: Path
    overview_dir: Path
    search_space_dir: Path
    dimensions_dir: Path
    correlations_dir: Path
    metrics_by_epoch_csv: Path
    metrics_by_sample_csv: Path
    dimension_value_stats_csv: Path
    dimension_importance_csv: Path
    pairwise_interactions_csv: Path
    manifest_json: Path
    epoch_cumulative_plot: Path
    epoch_rolling_plot: Path
    controller_loss_plot: Path
    dimension_importance_plot: Path
    pairwise_interactions_plot: Path
    layer_distribution_plot: Path
    cumulative_layer_distribution_plot: Path
    dimension_profiles_dir: Path


_METRIC_ORDER = ("max", "min", "mean", "median")
_EPOCH_BEST_METRIC_ORDER = ("best_max", "best_min")
MIN_VALID_EPOCHS_FOR_CORRELATION = 3


def _extract_experiment_id(path: Path) -> str:
    name = path.name
    if name.endswith("_summary.json"):
        return name[: -len("_summary.json")]
    if name.endswith(".json"):
        return path.stem
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem)


def _safe_name(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "item"


def _series_slope(x: pd.Series, y: pd.Series) -> float:
    x_numeric = pd.to_numeric(x, errors="coerce")
    y_numeric = pd.to_numeric(y, errors="coerce")
    frame = pd.DataFrame({"x": x_numeric, "y": y_numeric}).dropna()
    if len(frame) <= 1:
        return 0.0
    x_values = frame["x"].astype(float).to_numpy()
    y_values = frame["y"].astype(float).to_numpy()
    if np.allclose(x_values, x_values[0]):
        return 0.0
    return float(np.polyfit(x_values, y_values, deg=1)[0])


def _resolve_existing(path: Union[str, Path], base_dir: Optional[Path] = None) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        resolved = candidate.resolve()
        if not resolved.exists():
            raise FileNotFoundError("No existe el artefacto requerido: {}".format(resolved))
        return resolved

    direct = candidate.resolve()
    if direct.exists():
        return direct

    if base_dir is not None:
        relative_to_base = (base_dir / candidate).resolve()
        if relative_to_base.exists():
            return relative_to_base
        raise FileNotFoundError("No existe el artefacto requerido: {}".format(relative_to_base))

    raise FileNotFoundError("No existe el artefacto requerido: {}".format(direct))


def resolve_search_experiment(summary_json: Union[str, Path]) -> SearchExperimentFiles:
    summary_path = Path(summary_json).resolve()
    if not summary_path.exists():
        raise FileNotFoundError("No existe el summary JSON: {}".format(summary_path))

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    artifact_paths = payload.get("artifact_paths", {})
    base_dir = summary_path.parent
    architectures_csv = artifact_paths.get("architectures_csv")
    controller_csv = artifact_paths.get("controller_history_csv")
    if architectures_csv is None or controller_csv is None:
        experiment_id = _extract_experiment_id(summary_path)
        architectures_csv = base_dir / (experiment_id + "_architectures.csv")
        controller_csv = base_dir / (experiment_id + "_controller_history.csv")

    return SearchExperimentFiles(
        experiment_id=_extract_experiment_id(summary_path),
        summary_json=summary_path,
        architectures_csv=_resolve_existing(architectures_csv, base_dir=base_dir),
        controller_csv=_resolve_existing(controller_csv, base_dir=base_dir),
    )


def _default_output_dir(files: SearchExperimentFiles) -> Path:
    return files.summary_json.parent / "analysis"


def _load_summary_payload(files: SearchExperimentFiles) -> Dict[str, Any]:
    return json.loads(files.summary_json.read_text(encoding="utf-8"))


def _search_space_options(summary_payload: Dict[str, Any]) -> Dict[str, List[Any]]:
    search_space = summary_payload.get("search_space", {})
    options = search_space.get("options", {})
    return {str(key): list(value) for key, value in options.items()}


def _normalize_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _safe_std(series: pd.Series) -> float:
    if series.empty or len(series) <= 1:
        return 0.0
    return float(series.std(ddof=0))


def _safe_eta_squared(values: pd.Series, groups: pd.Series) -> float:
    if values.empty:
        return 0.0
    overall_mean = float(values.mean())
    total_ss = float(((values - overall_mean) ** 2).sum())
    if total_ss <= 0.0:
        return 0.0
    between_ss = 0.0
    grouped = pd.DataFrame({"value": values, "group": groups}).groupby("group")
    for _, frame in grouped:
        group_mean = float(frame["value"].mean())
        between_ss += float(len(frame)) * ((group_mean - overall_mean) ** 2)
    return float(between_ss / total_ss)


def _normalized_entropy(counts: pd.Series) -> float:
    total = float(counts.sum())
    if total <= 0.0 or len(counts) <= 1:
        return 0.0
    probs = counts.astype(float) / total
    probs = probs[probs > 0.0]
    entropy = float(-(probs * np.log(probs)).sum())
    return float(entropy / np.log(len(counts)))


def _rolling_window_from_summary(summary_payload: Dict[str, Any], sample_count: int) -> int:
    window = summary_payload.get("rolling_window")
    if window is None:
        return max(1, min(64, sample_count))
    try:
        return max(1, int(window))
    except (TypeError, ValueError):
        return max(1, min(64, sample_count))


def _augment_series_metrics(df: pd.DataFrame, value_column: str, prefix: str, window: int) -> pd.DataFrame:
    series = pd.to_numeric(df[value_column], errors="coerce").astype(float)
    rolling = series.rolling(window=window, min_periods=1)
    df["rolling_mean_{}".format(prefix)] = rolling.mean()
    df["rolling_median_{}".format(prefix)] = rolling.median()
    df["rolling_max_{}".format(prefix)] = rolling.max()
    df["rolling_min_{}".format(prefix)] = rolling.min()
    df["cumulative_mean_{}".format(prefix)] = series.expanding(min_periods=1).mean()
    df["cumulative_median_{}".format(prefix)] = series.expanding(min_periods=1).median()
    df["cumulative_max_{}".format(prefix)] = series.cummax()
    df["cumulative_min_{}".format(prefix)] = series.cummin()
    return df


def _build_controller_step_df(files: SearchExperimentFiles) -> pd.DataFrame:
    controller_df = pd.read_csv(files.controller_csv)
    if controller_df.empty:
        return controller_df
    controller_df = controller_df.sort_values(["search_epoch", "controller_epoch", "global_controller_step"]).reset_index(drop=True)
    for column in ("loss", "learning_rate", "search_epoch", "controller_epoch", "global_controller_step"):
        if column in controller_df.columns:
            controller_df[column] = pd.to_numeric(controller_df[column], errors="coerce")
    return controller_df


def load_sample_metrics(files: SearchExperimentFiles, summary_payload: Dict[str, Any]) -> pd.DataFrame:
    architecture_df = pd.read_csv(files.architectures_csv)
    if architecture_df.empty:
        raise ValueError("El CSV de arquitecturas está vacío: {}".format(files.architectures_csv))

    architecture_df = architecture_df.sort_values(["global_sample_order", "search_epoch", "sample_order_in_epoch"]).reset_index(drop=True)

    numeric_cols = [
        "accuracy",
        "search_accuracy",
        "val_accuracy",
        "test_accuracy",
        "raw_reward",
        "normalized_reward",
        "baseline_value_used",
        "layers",
        "units_0",
        "units_1",
        "units_2",
        "seq",
        "head_units",
        "global_sample_order",
        "search_epoch",
        "sample_order_in_epoch",
    ]
    for column in numeric_cols:
        if column in architecture_df.columns:
            architecture_df[column] = pd.to_numeric(architecture_df[column], errors="coerce")

    architecture_df["cached"] = architecture_df["cached"].map(_normalize_bool)
    architecture_df["cumulative_cache_hits"] = architecture_df["cached"].astype(int).cumsum()
    architecture_df["cumulative_cache_hit_rate"] = architecture_df["cumulative_cache_hits"] / architecture_df.index.to_series().add(1)

    rolling_window = _rolling_window_from_summary(summary_payload, len(architecture_df))
    architecture_df = _augment_series_metrics(architecture_df, "accuracy", "accuracy", rolling_window)

    keep_cols = [
        "global_sample_order",
        "search_epoch",
        "sample_order_in_epoch",
        "accuracy",
        "search_accuracy",
        "val_accuracy",
        "test_accuracy",
        "raw_reward",
        "normalized_reward",
        "baseline_value_used",
        "cached",
        "cumulative_cache_hits",
        "cumulative_cache_hit_rate",
        "layers",
        "rnn",
        "units_0",
        "units_1",
        "units_2",
        "direction",
        "memory_mode",
        "seq",
        "head_units",
        "video_decision",
        "video_decision_input",
        "cnn",
        "signature",
        "model_path",
    ]
    dynamic_cols = [
        "rolling_mean_accuracy",
        "rolling_median_accuracy",
        "rolling_max_accuracy",
        "rolling_min_accuracy",
        "cumulative_mean_accuracy",
        "cumulative_median_accuracy",
        "cumulative_max_accuracy",
        "cumulative_min_accuracy",
    ]
    existing = [column for column in keep_cols + dynamic_cols if column in architecture_df.columns]
    return architecture_df[existing].copy()


def load_epoch_metrics(files: SearchExperimentFiles, summary_payload: Dict[str, Any], sample_df: pd.DataFrame, controller_df: pd.DataFrame) -> pd.DataFrame:
    epoch_grouped = sample_df.groupby("search_epoch", dropna=False)
    epoch_df = epoch_grouped["accuracy"].agg([("num_sampled_architectures", "count"), ("epoch_mean_accuracy", "mean"), ("epoch_median_accuracy", "median"), ("epoch_max_accuracy", "max"), ("epoch_min_accuracy", "min")]).reset_index()
    epoch_df = epoch_df.rename(columns={"search_epoch": "epoch"}).sort_values("epoch").reset_index(drop=True)

    rolling_window = _rolling_window_from_summary(summary_payload, len(epoch_df))
    # Epoch-level trend columns should represent rolling/cumulative behaviour of the
    # per-epoch summary metrics themselves, using the same public names consumed by
    # the plotting layer (rolling_*_accuracy / cumulative_*_accuracy).
    epoch_mean_series = pd.to_numeric(epoch_df["epoch_mean_accuracy"], errors="coerce").astype(float)
    epoch_median_series = pd.to_numeric(epoch_df["epoch_median_accuracy"], errors="coerce").astype(float)
    epoch_max_series = pd.to_numeric(epoch_df["epoch_max_accuracy"], errors="coerce").astype(float)
    epoch_min_series = pd.to_numeric(epoch_df["epoch_min_accuracy"], errors="coerce").astype(float)

    epoch_df["rolling_mean_accuracy"] = epoch_mean_series.rolling(window=rolling_window, min_periods=1).mean()
    epoch_df["rolling_median_accuracy"] = epoch_median_series.rolling(window=rolling_window, min_periods=1).median()
    epoch_df["rolling_max_accuracy"] = epoch_max_series.rolling(window=rolling_window, min_periods=1).mean()
    epoch_df["rolling_min_accuracy"] = epoch_min_series.rolling(window=rolling_window, min_periods=1).mean()
    epoch_df["rolling_best_max_accuracy"] = epoch_max_series.rolling(window=rolling_window, min_periods=1).max()
    epoch_df["rolling_best_min_accuracy"] = epoch_min_series.rolling(window=rolling_window, min_periods=1).max()

    epoch_df["cumulative_mean_accuracy"] = epoch_mean_series.expanding(min_periods=1).mean()
    epoch_df["cumulative_median_accuracy"] = epoch_median_series.expanding(min_periods=1).median()
    epoch_df["cumulative_max_accuracy"] = epoch_max_series.expanding(min_periods=1).mean()
    epoch_df["cumulative_min_accuracy"] = epoch_min_series.expanding(min_periods=1).mean()
    epoch_df["cumulative_best_max_accuracy"] = epoch_max_series.cummax()
    epoch_df["cumulative_best_min_accuracy"] = epoch_min_series.cummax()

    summary_by_epoch = {}
    for item in summary_payload.get("per_epoch", []):
        summary_by_epoch[int(item.get("epoch", 0))] = item

    last_loss_by_epoch = {}
    if not controller_df.empty:
        grouped_controller = controller_df.groupby("search_epoch", dropna=False)
        for epoch, frame in grouped_controller:
            last_loss = frame["loss"].dropna().iloc[-1] if not frame["loss"].dropna().empty else np.nan
            last_lr = frame["learning_rate"].dropna().iloc[-1] if "learning_rate" in frame and not frame["learning_rate"].dropna().empty else np.nan
            last_loss_by_epoch[int(epoch)] = (float(last_loss) if pd.notna(last_loss) else np.nan, float(last_lr) if pd.notna(last_lr) else np.nan)

    sampling_columns = [
        "sampling_attempts",
        "sampling_guided_attempts",
        "sampling_fallback_attempts",
        "sampling_duplicate_hits",
        "sampling_used_fallback",
    ]
    for column in sampling_columns:
        epoch_df[column] = 0
    epoch_df["controller_last_loss"] = np.nan
    epoch_df["controller_last_learning_rate"] = np.nan
    for layer in (1, 2, 3):
        epoch_df["layers_{}_count".format(layer)] = 0

    cumulative_layer_counts = {1: 0, 2: 0, 3: 0}
    for idx, row in epoch_df.iterrows():
        epoch = int(row["epoch"])
        summary_item = summary_by_epoch.get(epoch, {})
        for column in sampling_columns:
            epoch_df.at[idx, column] = int(summary_item.get(column, 0) or 0)
        epoch_layer_distribution = summary_item.get("epoch_layer_distribution", {}) or {}
        for layer in (1, 2, 3):
            count = int(epoch_layer_distribution.get(str(layer), epoch_layer_distribution.get(layer, 0)) or 0)
            cumulative_layer_counts[layer] += count
            epoch_df.at[idx, "layers_{}_count".format(layer)] = count
            epoch_df.at[idx, "cumulative_layers_{}_count".format(layer)] = cumulative_layer_counts[layer]
        if epoch in last_loss_by_epoch:
            epoch_df.at[idx, "controller_last_loss"] = last_loss_by_epoch[epoch][0]
            epoch_df.at[idx, "controller_last_learning_rate"] = last_loss_by_epoch[epoch][1]
        else:
            epoch_df.at[idx, "controller_last_loss"] = float(summary_item.get("controller_last_loss")) if summary_item.get("controller_last_loss") is not None else np.nan
            epoch_df.at[idx, "controller_last_learning_rate"] = float(summary_item.get("controller_last_learning_rate")) if summary_item.get("controller_last_learning_rate") is not None else np.nan

    return epoch_df


def compute_dimension_value_stats(sample_df: pd.DataFrame, search_space_options: Dict[str, List[Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    dimensions = [dimension for dimension in search_space_options.keys() if dimension in sample_df.columns]
    total_samples = max(1, len(sample_df))
    for dimension in dimensions:
        grouped = sample_df.groupby(dimension, dropna=False)
        for value, frame in grouped:
            rows.append(
                {
                    "dimension": dimension,
                    "value": value,
                    "count": int(len(frame)),
                    "sample_fraction": float(len(frame) / total_samples),
                    "cached_fraction": float(frame["cached"].mean()) if len(frame) else 0.0,
                    "mean_accuracy": float(frame["accuracy"].mean()) if len(frame) else 0.0,
                    "median_accuracy": float(frame["accuracy"].median()) if len(frame) else 0.0,
                    "std_accuracy": _safe_std(frame["accuracy"]),
                    "max_accuracy": float(frame["accuracy"].max()) if len(frame) else 0.0,
                    "min_accuracy": float(frame["accuracy"].min()) if len(frame) else 0.0,
                    "mean_search_accuracy": float(frame["search_accuracy"].mean()) if len(frame) and "search_accuracy" in frame else 0.0,
                    "best_signature": str(frame.sort_values("accuracy", ascending=False).iloc[0]["signature"]) if len(frame) else "",
                }
            )
    if not rows:
        return pd.DataFrame(columns=["dimension", "value", "count", "sample_fraction", "cached_fraction", "mean_accuracy", "median_accuracy", "std_accuracy", "max_accuracy", "min_accuracy", "mean_search_accuracy", "best_signature"])
    return pd.DataFrame(rows).sort_values(["dimension", "mean_accuracy", "count"], ascending=[True, False, False]).reset_index(drop=True)


def compute_dimension_importance(sample_df: pd.DataFrame, search_space_options: Dict[str, List[Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    dimensions = [dimension for dimension in search_space_options.keys() if dimension in sample_df.columns]
    for dimension in dimensions:
        grouped = sample_df.groupby(dimension, dropna=False)
        count_series = grouped.size()
        mean_accuracy = grouped["accuracy"].mean()
        max_accuracy = grouped["accuracy"].max()
        rows.append(
            {
                "dimension": dimension,
                "configured_option_count": int(len(search_space_options.get(dimension, []))),
                "sampled_unique_values": int(count_series.shape[0]),
                "sample_coverage": float(count_series.shape[0] / max(1, len(search_space_options.get(dimension, [])))),
                "sampling_entropy": _normalized_entropy(count_series),
                "eta_squared_accuracy": _safe_eta_squared(sample_df["accuracy"], sample_df[dimension]),
                "eta_squared_search_accuracy": _safe_eta_squared(sample_df["search_accuracy"], sample_df[dimension]) if "search_accuracy" in sample_df.columns else 0.0,
                "mean_accuracy_range": float(mean_accuracy.max() - mean_accuracy.min()) if len(mean_accuracy) else 0.0,
                "max_accuracy_range": float(max_accuracy.max() - max_accuracy.min()) if len(max_accuracy) else 0.0,
                "best_value_by_mean": str(mean_accuracy.idxmax()) if len(mean_accuracy) else "",
                "best_mean_accuracy": float(mean_accuracy.max()) if len(mean_accuracy) else 0.0,
                "best_value_by_max": str(max_accuracy.idxmax()) if len(max_accuracy) else "",
                "best_max_accuracy": float(max_accuracy.max()) if len(max_accuracy) else 0.0,
            }
        )
    if not rows:
        return pd.DataFrame(columns=["dimension", "configured_option_count", "sampled_unique_values", "sample_coverage", "sampling_entropy", "eta_squared_accuracy", "eta_squared_search_accuracy", "mean_accuracy_range", "max_accuracy_range", "best_value_by_mean", "best_mean_accuracy", "best_value_by_max", "best_max_accuracy", "relative_importance", "importance_rank"])
    result = pd.DataFrame(rows).sort_values("eta_squared_accuracy", ascending=False).reset_index(drop=True)
    total = float(result["eta_squared_accuracy"].sum()) if not result.empty else 0.0
    result["relative_importance"] = result["eta_squared_accuracy"] / total if total > 0.0 else 0.0
    result["importance_rank"] = np.arange(1, len(result) + 1)
    return result


def compute_pairwise_interactions(sample_df: pd.DataFrame, search_space_options: Dict[str, List[Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    dimensions = [dimension for dimension in search_space_options.keys() if dimension in sample_df.columns]
    for idx_a, dim_a in enumerate(dimensions):
        for dim_b in dimensions[idx_a + 1 :]:
            pair_series = sample_df[[dim_a, dim_b]].astype(str).agg(" | ".join, axis=1)
            grouped = pd.DataFrame({"pair": pair_series, "accuracy": sample_df["accuracy"]}).groupby("pair")
            mean_accuracy = grouped["accuracy"].mean()
            max_accuracy = grouped["accuracy"].max()
            rows.append(
                {
                    "dimension_a": dim_a,
                    "dimension_b": dim_b,
                    "pair_count": int(mean_accuracy.shape[0]),
                    "eta_squared_accuracy": _safe_eta_squared(sample_df["accuracy"], pair_series),
                    "best_pair_by_mean": str(mean_accuracy.idxmax()) if len(mean_accuracy) else "",
                    "best_pair_mean_accuracy": float(mean_accuracy.max()) if len(mean_accuracy) else 0.0,
                    "best_pair_by_max": str(max_accuracy.idxmax()) if len(max_accuracy) else "",
                    "best_pair_max_accuracy": float(max_accuracy.max()) if len(max_accuracy) else 0.0,
                }
            )
    if not rows:
        return pd.DataFrame(columns=["dimension_a", "dimension_b", "pair_count", "eta_squared_accuracy", "best_pair_by_mean", "best_pair_mean_accuracy", "best_pair_by_max", "best_pair_max_accuracy"])
    return pd.DataFrame(rows).sort_values("eta_squared_accuracy", ascending=False).reset_index(drop=True)


def _save_single_axes_figure(output_path: Path, width: float = 12.0, height: float = 6.0):
    fig, ax = plt.subplots(figsize=(width, height))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return fig, ax


def _plot_accuracy_cumulative(sample_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = _save_single_axes_figure(output_path, width=14, height=6)
    x = sample_df["global_sample_order"]
    for metric in _METRIC_ORDER:
        ax.plot(x, sample_df["cumulative_{}_accuracy".format(metric)], linewidth=2, label="cumulative_{}".format(metric))
    ax.set_title("Accuracy por candidata: métricas acumuladas")
    ax.set_xlabel("global_sample_order")
    ax.set_ylabel("accuracy")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_accuracy_rolling(sample_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = _save_single_axes_figure(output_path, width=14, height=6)
    x = sample_df["global_sample_order"]
    for metric in _METRIC_ORDER:
        ax.plot(x, sample_df["rolling_{}_accuracy".format(metric)], linewidth=2, label="rolling_{}".format(metric))
    ax.set_title("Accuracy por candidata: métricas rolling")
    ax.set_xlabel("global_sample_order")
    ax.set_ylabel("accuracy")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_epoch_cumulative(epoch_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = _save_single_axes_figure(output_path, width=14, height=6)
    x = epoch_df["epoch"]
    for metric in _METRIC_ORDER:
        ax.plot(x, epoch_df["cumulative_{}_accuracy".format(metric)], marker="o", linewidth=2, label="cumulative_{}".format(metric))
    for metric in _EPOCH_BEST_METRIC_ORDER:
        ax.plot(x, epoch_df["cumulative_{}_accuracy".format(metric)], marker="o", linewidth=2, linestyle="--", label="cumulative_{}".format(metric))
    ax.set_title("Accuracy por epoch: métricas acumuladas")
    ax.set_xlabel("search_epoch")
    ax.set_ylabel("accuracy")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_epoch_rolling(epoch_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = _save_single_axes_figure(output_path, width=14, height=6)
    x = epoch_df["epoch"]
    for metric in _METRIC_ORDER:
        ax.plot(x, epoch_df["rolling_{}_accuracy".format(metric)], marker="o", linewidth=2, label="rolling_{}".format(metric))
    for metric in _EPOCH_BEST_METRIC_ORDER:
        ax.plot(x, epoch_df["rolling_{}_accuracy".format(metric)], marker="o", linewidth=2, linestyle="--", label="rolling_{}".format(metric))
    ax.set_title("Accuracy por epoch: métricas rolling")
    ax.set_xlabel("search_epoch")
    ax.set_ylabel("accuracy")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_controller_loss(controller_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = _save_single_axes_figure(output_path, width=12, height=6)
    if not controller_df.empty and "loss" in controller_df.columns and controller_df["loss"].notna().any():
        loss_series = pd.to_numeric(controller_df["loss"], errors="coerce")
        rolling_window = max(1, min(25, len(loss_series)))
        rolling_mean = loss_series.rolling(window=rolling_window, min_periods=1).mean()
        ax.plot(controller_df["global_controller_step"], loss_series, linewidth=1.2, alpha=0.55, label="loss")
        ax.plot(controller_df["global_controller_step"], rolling_mean, linewidth=2.2, label="rolling_mean_loss")
        ax.set_title("Pérdida del controller por step")
        ax.set_xlabel("global_controller_step")
        ax.set_ylabel("loss")
        ax.grid(True, alpha=0.3)
        ax.legend()
    else:
        ax.text(0.5, 0.5, "Sin historial de loss del controller", ha="center", va="center")
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_dimension_importance(df: pd.DataFrame, output_path: Path) -> None:
    plot_df = df.sort_values("relative_importance", ascending=True)
    fig, ax = _save_single_axes_figure(output_path, width=10, height=max(6, 0.45 * max(1, len(plot_df))))
    if not plot_df.empty:
        ax.barh(plot_df["dimension"], plot_df["relative_importance"])
    ax.set_title("Importancia relativa de dimensiones del search space")
    ax.set_xlabel("relative_importance")
    ax.set_ylabel("dimension")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_pairwise_interactions(df: pd.DataFrame, output_path: Path) -> None:
    plot_df = df.head(15).copy()
    if not plot_df.empty:
        plot_df["pair"] = plot_df["dimension_a"] + " × " + plot_df["dimension_b"]
        plot_df = plot_df.sort_values("eta_squared_accuracy", ascending=True)
    fig, ax = _save_single_axes_figure(output_path, width=12, height=max(6, 0.45 * max(1, len(plot_df))))
    if not plot_df.empty:
        ax.barh(plot_df["pair"], plot_df["eta_squared_accuracy"])
    ax.set_title("Interacciones por pares más informativas")
    ax.set_xlabel("eta_squared_accuracy")
    ax.set_ylabel("pair")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_layer_distribution(epoch_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = _save_single_axes_figure(output_path, width=11, height=6)
    for layer in (1, 2, 3):
        column = "layers_{}_count".format(layer)
        if column in epoch_df.columns:
            ax.plot(epoch_df["epoch"], epoch_df[column], marker="o", label="layers={}".format(layer))
    ax.set_title("Distribución de número de capas por epoch")
    ax.set_xlabel("search_epoch")
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_cumulative_layer_distribution(epoch_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = _save_single_axes_figure(output_path, width=11, height=6)
    for layer in (1, 2, 3):
        cumulative_column = "cumulative_layers_{}_count".format(layer)
        if cumulative_column in epoch_df.columns:
            ax.plot(epoch_df["epoch"], epoch_df[cumulative_column], marker="o", label="layers={}".format(layer))
    ax.set_title("Distribución acumulada de capas")
    ax.set_xlabel("search_epoch")
    ax.set_ylabel("cumulative_count")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _build_dimension_epoch_profile_frame(sample_df: pd.DataFrame, dimension: str) -> pd.DataFrame:
    if dimension not in sample_df.columns:
        return pd.DataFrame()
    dim_df = sample_df[["search_epoch", dimension, "accuracy"]].copy()
    dim_df = dim_df.dropna(subset=["search_epoch", "accuracy"])
    if dim_df.empty:
        return pd.DataFrame()
    dim_df["search_epoch"] = pd.to_numeric(dim_df["search_epoch"], errors="coerce")
    dim_df = dim_df.dropna(subset=["search_epoch"]).sort_values(["search_epoch"]).reset_index(drop=True)
    if dim_df.empty:
        return pd.DataFrame()
    dim_df["search_epoch"] = dim_df["search_epoch"].astype(int)
    all_epochs = sorted(dim_df["search_epoch"].unique().tolist())
    total_by_epoch = dim_df.groupby("search_epoch").size().reindex(all_epochs, fill_value=0).cumsum()

    rows = []
    for value, value_df in dim_df.groupby(dimension, dropna=False):
        value_df = value_df.sort_values("search_epoch")
        grouped = value_df.groupby("search_epoch")["accuracy"].apply(list).to_dict()
        cumulative_accuracies: List[float] = []
        cumulative_count = 0
        first_epoch_seen = None
        for epoch in all_epochs:
            if epoch in grouped:
                cumulative_accuracies.extend(grouped[epoch])
                cumulative_count += len(grouped[epoch])
                if first_epoch_seen is None:
                    first_epoch_seen = epoch
            if cumulative_count == 0:
                continue
            rows.append({
                "search_epoch": int(epoch),
                "value": value,
                "cumulative_mean_accuracy": float(np.mean(cumulative_accuracies)),
                "cumulative_median_accuracy": float(np.median(cumulative_accuracies)),
                "cumulative_selection_count": int(cumulative_count),
                "cumulative_selection_share": float(cumulative_count / total_by_epoch.loc[epoch]) if total_by_epoch.loc[epoch] else 0.0,
                "first_epoch_seen": int(first_epoch_seen),
            })
    return pd.DataFrame(rows)


def _build_dimension_local_epoch_profile_frame(sample_df: pd.DataFrame, dimension: str) -> pd.DataFrame:
    if dimension not in sample_df.columns:
        return pd.DataFrame()
    dim_df = sample_df[["search_epoch", dimension, "accuracy"]].copy()
    dim_df = dim_df.dropna(subset=["search_epoch", "accuracy"])
    if dim_df.empty:
        return pd.DataFrame()
    dim_df["search_epoch"] = pd.to_numeric(dim_df["search_epoch"], errors="coerce")
    dim_df = dim_df.dropna(subset=["search_epoch"]).sort_values(["search_epoch"]).reset_index(drop=True)
    if dim_df.empty:
        return pd.DataFrame()
    dim_df["search_epoch"] = dim_df["search_epoch"].astype(int)
    total_by_epoch = dim_df.groupby("search_epoch").size()
    epoch_rows = dim_df.groupby(["search_epoch", dimension], dropna=False).agg(
        epoch_mean_accuracy=("accuracy", "mean"),
        epoch_median_accuracy=("accuracy", "median"),
        epoch_selection_count=("accuracy", "count"),
    ).reset_index().rename(columns={dimension: "value"})
    epoch_rows["epoch_selection_share"] = epoch_rows.apply(
        lambda row: float(row["epoch_selection_count"] / total_by_epoch.loc[row["search_epoch"]]) if total_by_epoch.loc[row["search_epoch"]] else 0.0, axis=1
    )
    return epoch_rows


def _corr_or_zero(frame: pd.DataFrame, left: str, right: str, min_valid_epochs: int = MIN_VALID_EPOCHS_FOR_CORRELATION) -> float:
    valid = frame[[left, right]].dropna()
    if len(valid) < min_valid_epochs:
        return 0.0
    corr = valid.corr().iloc[0, 1]
    return float(0.0 if pd.isna(corr) else corr)


def _build_value_trajectory_summary(profile_df: pd.DataFrame, min_valid_epochs: int = MIN_VALID_EPOCHS_FOR_CORRELATION) -> pd.DataFrame:
    rows = []
    for value, value_df in profile_df.groupby("value", dropna=False):
        value_df = value_df.sort_values("search_epoch").reset_index(drop=True)
        if value_df.empty:
            continue
        num_valid_epochs = int(len(value_df))
        rows.append({
            "value": value,
            "first_epoch_seen": int(value_df["first_epoch_seen"].iloc[0]) if "first_epoch_seen" in value_df.columns else int(value_df["search_epoch"].iloc[0]),
            "last_epoch_seen": int(value_df["search_epoch"].iloc[-1]),
            "num_valid_epochs": num_valid_epochs,
            "valid_for_correlation": bool(num_valid_epochs >= min_valid_epochs),
            "final_cumulative_mean_accuracy": float(value_df["cumulative_mean_accuracy"].iloc[-1]) if "cumulative_mean_accuracy" in value_df.columns else float(value_df["epoch_mean_accuracy"].iloc[-1]),
            "final_cumulative_median_accuracy": float(value_df["cumulative_median_accuracy"].iloc[-1]) if "cumulative_median_accuracy" in value_df.columns else float(value_df["epoch_median_accuracy"].iloc[-1]),
            "final_selection_count": int(value_df["cumulative_selection_count"].iloc[-1]) if "cumulative_selection_count" in value_df.columns else int(value_df["epoch_selection_count"].sum()),
            "final_selection_share": float(value_df["cumulative_selection_share"].iloc[-1]) if "cumulative_selection_share" in value_df.columns else float(value_df["epoch_selection_share"].iloc[-1]),
            "peak_selection_share": float(value_df["cumulative_selection_share"].max()) if "cumulative_selection_share" in value_df.columns else float(value_df["epoch_selection_share"].max()),
            "mean_accuracy_slope": _series_slope(value_df["search_epoch"], value_df["cumulative_mean_accuracy"] if "cumulative_mean_accuracy" in value_df.columns else value_df["epoch_mean_accuracy"]),
            "median_accuracy_slope": _series_slope(value_df["search_epoch"], value_df["cumulative_median_accuracy"] if "cumulative_median_accuracy" in value_df.columns else value_df["epoch_median_accuracy"]),
            "selection_count_slope": _series_slope(value_df["search_epoch"], value_df["cumulative_selection_count"] if "cumulative_selection_count" in value_df.columns else value_df["epoch_selection_count"]),
            "selection_share_slope": _series_slope(value_df["search_epoch"], value_df["cumulative_selection_share"] if "cumulative_selection_share" in value_df.columns else value_df["epoch_selection_share"]),
            "corr_selection_count_vs_cum_mean_accuracy": _corr_or_zero(value_df.rename(columns={"epoch_selection_count": "cumulative_selection_count", "epoch_mean_accuracy": "cumulative_mean_accuracy"}), "cumulative_selection_count", "cumulative_mean_accuracy", min_valid_epochs),
            "corr_selection_count_vs_cum_median_accuracy": _corr_or_zero(value_df.rename(columns={"epoch_selection_count": "cumulative_selection_count", "epoch_median_accuracy": "cumulative_median_accuracy"}), "cumulative_selection_count", "cumulative_median_accuracy", min_valid_epochs),
            "corr_selection_share_vs_cum_mean_accuracy": _corr_or_zero(value_df.rename(columns={"epoch_selection_share": "cumulative_selection_share", "epoch_mean_accuracy": "cumulative_mean_accuracy"}), "cumulative_selection_share", "cumulative_mean_accuracy", min_valid_epochs),
            "corr_selection_share_vs_cum_median_accuracy": _corr_or_zero(value_df.rename(columns={"epoch_selection_share": "cumulative_selection_share", "epoch_median_accuracy": "cumulative_median_accuracy"}), "cumulative_selection_share", "cumulative_median_accuracy", min_valid_epochs),
        })
    summary_df = pd.DataFrame(rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["final_cumulative_median_accuracy", "final_selection_share"], ascending=[False, False]).reset_index(drop=True)
    return summary_df


def _plot_dimension_profile(profile_df: pd.DataFrame, dimension: str, ordered_values: List[Any], accuracy_column: str, selection_column: str, output_path: Path) -> None:
    if profile_df.empty:
        return
    fig, ax_left = _save_single_axes_figure(output_path, width=14, height=7)
    ax_right = ax_left.twinx()
    legend_handles = []
    legend_labels = []
    for value in ordered_values:
        value_df = profile_df[profile_df["value"] == value].sort_values("search_epoch")
        if value_df.empty:
            continue
        value_label = str(value)
        line_acc, = ax_left.plot(
            value_df["search_epoch"],
            value_df[accuracy_column],
            marker="o",
            linewidth=2,
            label="{} {}".format(value_label, accuracy_column),
        )
        line_sel, = ax_right.plot(
            value_df["search_epoch"],
            value_df[selection_column],
            linestyle="--",
            linewidth=1.75,
            alpha=0.85,
            color=line_acc.get_color(),
            label="{} {}".format(value_label, selection_column),
        )
        legend_handles.extend([line_acc, line_sel])
        legend_labels.extend(["{} {}".format(value_label, accuracy_column), "{} {}".format(value_label, selection_column)])

    ax_left.set_title("{}: {} vs {}".format(dimension, accuracy_column, selection_column))
    ax_left.set_xlabel("search_epoch")
    ax_left.set_ylabel(accuracy_column)
    ax_right.set_ylabel(selection_column)
    ax_left.grid(True, alpha=0.3)
    if legend_handles:
        ax_left.legend(legend_handles, legend_labels, loc="best", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_corr_bars(df: pd.DataFrame, columns: List[str], title: str, output_path: Path) -> None:
    if df.empty:
        return
    plot_df = df[["value"] + columns].copy().set_index("value")
    fig, ax = _save_single_axes_figure(output_path, width=12, height=max(6, 0.5 * len(plot_df) + 2))
    y = np.arange(len(plot_df))
    width = 0.8 / max(1, len(columns))
    offsets = np.linspace(-0.4 + width / 2.0, 0.4 - width / 2.0, num=len(columns)) if len(columns) > 1 else [0.0]
    for offset, column in zip(offsets, columns):
        ax.barh(y + offset, plot_df[column].fillna(0.0), height=width, label=column)
    ax.set_yticks(y)
    ax.set_yticklabels([str(v) for v in plot_df.index])
    ax.set_xlabel("correlation")
    ax.set_ylabel("value")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_corr_heatmap(matrix: pd.DataFrame, title: str, output_path: Path) -> None:
    if matrix.empty:
        return
    labels = [str(c) for c in matrix.columns]
    fig, ax = _save_single_axes_figure(output_path, width=max(8, 0.8 * len(labels)), height=max(6, 0.7 * len(labels)))
    im = ax.imshow(matrix.to_numpy(dtype=float), vmin=-1.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_dimension_alignment(summary_df: pd.DataFrame, output_path: Path) -> None:
    if summary_df.empty:
        return
    plot_df = summary_df.sort_values("corr_final_selection_share_vs_final_cum_median_accuracy", ascending=False)
    fig, ax = _save_single_axes_figure(output_path, width=12, height=max(6, 0.5 * len(plot_df) + 2))
    y = np.arange(len(plot_df))
    ax.barh(y - 0.18, plot_df["corr_final_selection_share_vs_final_cum_median_accuracy"], height=0.35, label="median")
    ax.barh(y + 0.18, plot_df["corr_final_selection_share_vs_final_cum_mean_accuracy"], height=0.35, label="mean")
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["dimension"].tolist())
    ax.set_xlabel("correlation")
    ax.set_ylabel("dimension")
    ax.set_title("Selection-accuracy alignment por dimensión")
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _build_dimension_over_time_alignment(profile_df: pd.DataFrame, summary_df: pd.DataFrame, dimension: str, selection_column: str, mean_column: str, median_column: str) -> pd.DataFrame:
    if profile_df.empty or summary_df.empty:
        return pd.DataFrame()
    valid_values = set(summary_df.loc[summary_df["valid_for_correlation"].astype(bool), "value"].tolist()) if "valid_for_correlation" in summary_df.columns else set(summary_df["value"].tolist())
    if not valid_values:
        return pd.DataFrame()
    filtered = profile_df[profile_df["value"].isin(valid_values)].copy()
    if filtered.empty:
        return pd.DataFrame()
    rows = []
    for epoch, epoch_df in filtered.groupby("search_epoch", dropna=False):
        valid_mean = epoch_df[["value", selection_column, mean_column]].dropna()
        valid_median = epoch_df[["value", selection_column, median_column]].dropna()
        corr_mean = 0.0
        corr_median = 0.0
        num_values_mean = int(len(valid_mean))
        num_values_median = int(len(valid_median))
        if num_values_mean >= 2:
            corr = valid_mean[[selection_column, mean_column]].corr().iloc[0, 1]
            corr_mean = float(0.0 if pd.isna(corr) else corr)
        if num_values_median >= 2:
            corr = valid_median[[selection_column, median_column]].corr().iloc[0, 1]
            corr_median = float(0.0 if pd.isna(corr) else corr)
        rows.append({
            "dimension": dimension,
            "search_epoch": int(epoch),
            "corr_selection_share_vs_mean_accuracy": corr_mean,
            "corr_selection_share_vs_median_accuracy": corr_median,
            "num_values_mean": num_values_mean,
            "num_values_median": num_values_median,
        })
    return pd.DataFrame(rows).sort_values("search_epoch").reset_index(drop=True)


def _plot_dimension_over_time_alignment(df: pd.DataFrame, metric_column: str, title: str, output_path: Path) -> None:
    if df.empty:
        return
    fig, ax = _save_single_axes_figure(output_path, width=12, height=max(6, 0.45 * df["dimension"].nunique() + 3))
    for dimension, frame in df.groupby("dimension", dropna=False):
        frame = frame.sort_values("search_epoch")
        ax.plot(frame["search_epoch"], frame[metric_column], marker="o", linewidth=2, label=str(dimension))
    ax.set_xlabel("search_epoch")
    ax.set_ylabel("correlation")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _generate_correlation_outputs(dimension: str, cumulative_profile_df: pd.DataFrame, local_profile_df: pd.DataFrame, cumulative_summary_df: pd.DataFrame, local_summary_df: pd.DataFrame, correlations_dir: Path, global_alignment_rows: List[Dict[str, Any]], cumulative_over_time_rows: List[Dict[str, Any]], local_over_time_rows: List[Dict[str, Any]]) -> None:
    if cumulative_summary_df.empty and local_summary_df.empty:
        return

    by_value_cumulative_dir = correlations_dir / "by_value_cumulative"
    by_value_local_dir = correlations_dir / "by_value_local"
    by_value_cumulative_dir.mkdir(parents=True, exist_ok=True)
    by_value_local_dir.mkdir(parents=True, exist_ok=True)

    safe_dimension = _safe_name(dimension)

    if not cumulative_summary_df.empty:
        cumulative_csv = by_value_cumulative_dir / (safe_dimension + "_value_correlations_cumulative.csv")
        cumulative_png = by_value_cumulative_dir / (safe_dimension + "_value_correlations_cumulative.png")
        cumulative_summary_df.to_csv(cumulative_csv, index=False)
        _plot_corr_bars(
            cumulative_summary_df,
            [
                "corr_selection_share_vs_cum_mean_accuracy",
                "corr_selection_share_vs_cum_median_accuracy",
                "corr_selection_count_vs_cum_mean_accuracy",
                "corr_selection_count_vs_cum_median_accuracy",
            ],
            "{}: correlación by value (cumulative)".format(dimension),
            cumulative_png,
        )
        cumulative_alignment_df = _build_dimension_over_time_alignment(
            cumulative_profile_df,
            cumulative_summary_df,
            dimension,
            selection_column="cumulative_selection_share",
            mean_column="cumulative_mean_accuracy",
            median_column="cumulative_median_accuracy",
        )
        if not cumulative_alignment_df.empty:
            cumulative_over_time_rows.extend(cumulative_alignment_df.to_dict(orient="records"))
            last_row = cumulative_alignment_df.sort_values("search_epoch").iloc[-1]
            global_alignment_rows.append({
                "dimension": dimension,
                "corr_final_selection_share_vs_final_cum_median_accuracy": float(last_row["corr_selection_share_vs_median_accuracy"]),
                "corr_final_selection_share_vs_final_cum_mean_accuracy": float(last_row["corr_selection_share_vs_mean_accuracy"]),
                "num_values": int(max(last_row.get("num_values_mean", 0), last_row.get("num_values_median", 0))),
                "best_value_by_accuracy": str(cumulative_summary_df.sort_values("final_cumulative_median_accuracy", ascending=False).iloc[0]["value"]),
                "most_selected_value": str(cumulative_summary_df.sort_values("final_selection_share", ascending=False).iloc[0]["value"]),
                "source": "cumulative_last_epoch",
                "final_epoch": int(last_row["search_epoch"]),
            })

    if not local_summary_df.empty:
        local_csv = by_value_local_dir / (safe_dimension + "_value_correlations_local.csv")
        local_png = by_value_local_dir / (safe_dimension + "_value_correlations_local.png")
        local_summary_df.to_csv(local_csv, index=False)
        _plot_corr_bars(
            local_summary_df,
            [
                "corr_selection_share_vs_cum_mean_accuracy",
                "corr_selection_share_vs_cum_median_accuracy",
                "corr_selection_count_vs_cum_mean_accuracy",
                "corr_selection_count_vs_cum_median_accuracy",
            ],
            "{}: correlación by value (local per epoch)".format(dimension),
            local_png,
        )
        local_alignment_df = _build_dimension_over_time_alignment(
            local_profile_df,
            local_summary_df,
            dimension,
            selection_column="epoch_selection_share",
            mean_column="epoch_mean_accuracy",
            median_column="epoch_median_accuracy",
        )
        if not local_alignment_df.empty:
            local_over_time_rows.extend(local_alignment_df.to_dict(orient="records"))


def _plot_dimension_profiles(sample_df: pd.DataFrame, value_stats_df: pd.DataFrame, output_dir: Path, correlations_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    correlations_dir.mkdir(parents=True, exist_ok=True)
    if value_stats_df.empty or sample_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    combinations = [
        ("cumulative_mean_accuracy", "cumulative_selection_count"),
        ("cumulative_mean_accuracy", "cumulative_selection_share"),
        ("cumulative_median_accuracy", "cumulative_selection_count"),
        ("cumulative_median_accuracy", "cumulative_selection_share"),
    ]

    global_alignment_rows: List[Dict[str, Any]] = []
    cumulative_over_time_rows: List[Dict[str, Any]] = []
    local_over_time_rows: List[Dict[str, Any]] = []

    for dimension, frame in value_stats_df.groupby("dimension"):
        profile_df = _build_dimension_epoch_profile_frame(sample_df, dimension)
        local_profile_df = _build_dimension_local_epoch_profile_frame(sample_df, dimension)
        if profile_df.empty:
            continue
        dimension_dir = output_dir / _safe_name(dimension)
        dimension_dir.mkdir(parents=True, exist_ok=True)
        ordered_values = frame.sort_values(["mean_accuracy", "count"], ascending=[False, False])["value"].tolist()
        trajectory_summary_df = _build_value_trajectory_summary(profile_df)
        local_trajectory_summary_df = _build_value_trajectory_summary(local_profile_df)
        trajectory_summary_path = dimension_dir / "value_trajectory_summary.csv"
        local_trajectory_summary_path = dimension_dir / "value_trajectory_summary_local.csv"
        trajectory_summary_df.to_csv(trajectory_summary_path, index=False)
        local_trajectory_summary_df.to_csv(local_trajectory_summary_path, index=False)
        for accuracy_column, selection_column in combinations:
            output_path = dimension_dir / ("{}_vs_{}.png".format(accuracy_column, selection_column))
            _plot_dimension_profile(profile_df, dimension, ordered_values, accuracy_column, selection_column, output_path)
        _generate_correlation_outputs(dimension, profile_df, local_profile_df, trajectory_summary_df, local_trajectory_summary_df, correlations_dir, global_alignment_rows, cumulative_over_time_rows, local_over_time_rows)

    final_df = pd.DataFrame(global_alignment_rows)
    if not final_df.empty:
        final_df = final_df.sort_values("corr_final_selection_share_vs_final_cum_median_accuracy", ascending=False).reset_index(drop=True)
    cumulative_over_time_df = pd.DataFrame(cumulative_over_time_rows)
    if not cumulative_over_time_df.empty:
        cumulative_over_time_df = cumulative_over_time_df.sort_values(["dimension", "search_epoch"]).reset_index(drop=True)
    local_over_time_df = pd.DataFrame(local_over_time_rows)
    if not local_over_time_df.empty:
        local_over_time_df = local_over_time_df.sort_values(["dimension", "search_epoch"]).reset_index(drop=True)
    return final_df, cumulative_over_time_df, local_over_time_df


def generate_search_analysis(summary_json: Union[str, Path], output_dir: Union[str, Path, None] = None) -> SearchAnalysisArtifacts:
    files = resolve_search_experiment(summary_json)
    summary_payload = _load_summary_payload(files)
    output_root = Path(output_dir).resolve() if output_dir else _default_output_dir(files)
    output_root.mkdir(parents=True, exist_ok=True)
    tables_dir = output_root / "tables"
    overview_dir = output_root / "overview"
    search_space_dir = output_root / "search_space"
    dimensions_dir = output_root / "dimensions"
    correlations_dir = output_root / "correlations"
    for directory in (tables_dir, overview_dir, search_space_dir, dimensions_dir, correlations_dir):
        directory.mkdir(parents=True, exist_ok=True)

    controller_df = _build_controller_step_df(files)
    sample_df = load_sample_metrics(files, summary_payload)
    epoch_df = load_epoch_metrics(files, summary_payload, sample_df, controller_df)
    search_space_options = _search_space_options(summary_payload)

    metrics_by_epoch_csv = tables_dir / "metrics_by_epoch.csv"
    metrics_by_sample_csv = tables_dir / "metrics_by_sample.csv"
    dimension_value_stats_csv = tables_dir / "dimension_value_stats.csv"
    dimension_importance_csv = tables_dir / "dimension_importance.csv"
    pairwise_interactions_csv = tables_dir / "pairwise_interactions.csv"
    manifest_json = output_root / "analysis_manifest.json"
    epoch_cumulative_plot = overview_dir / "epoch_accuracy_cumulative.png"
    epoch_rolling_plot = overview_dir / "epoch_accuracy_rolling.png"
    controller_loss_plot = overview_dir / "controller_loss.png"
    layer_distribution_plot = overview_dir / "layer_distribution_by_epoch.png"
    cumulative_layer_distribution_plot = overview_dir / "layer_distribution_cumulative.png"
    dimension_importance_plot = search_space_dir / "dimension_importance.png"
    pairwise_interactions_plot = search_space_dir / "pairwise_interactions.png"
    dimension_profiles_dir = dimensions_dir

    dimension_value_stats_df = compute_dimension_value_stats(sample_df, search_space_options)
    dimension_importance_df = compute_dimension_importance(sample_df, search_space_options)
    pairwise_interactions_df = compute_pairwise_interactions(sample_df, search_space_options)

    sample_df.to_csv(metrics_by_sample_csv, index=False)
    epoch_df.to_csv(metrics_by_epoch_csv, index=False)
    dimension_value_stats_df.to_csv(dimension_value_stats_csv, index=False)
    dimension_importance_df.to_csv(dimension_importance_csv, index=False)
    pairwise_interactions_df.to_csv(pairwise_interactions_csv, index=False)

    _plot_epoch_cumulative(epoch_df, epoch_cumulative_plot)
    _plot_epoch_rolling(epoch_df, epoch_rolling_plot)
    _plot_controller_loss(controller_df, controller_loss_plot)
    _plot_dimension_importance(dimension_importance_df, dimension_importance_plot)
    _plot_pairwise_interactions(pairwise_interactions_df, pairwise_interactions_plot)
    _plot_layer_distribution(epoch_df, layer_distribution_plot)
    _plot_cumulative_layer_distribution(epoch_df, cumulative_layer_distribution_plot)
    dimension_alignment_df, cumulative_dimension_over_time_df, local_dimension_over_time_df = _plot_dimension_profiles(sample_df, dimension_value_stats_df, dimension_profiles_dir, correlations_dir)
    by_dimension_over_time_dir = correlations_dir / "by_dimension_over_time"
    final_dimension_alignment_dir = correlations_dir / "final_dimension_alignment"
    by_dimension_over_time_dir.mkdir(parents=True, exist_ok=True)
    final_dimension_alignment_dir.mkdir(parents=True, exist_ok=True)
    cumulative_dimension_over_time_csv = by_dimension_over_time_dir / "dimension_alignment_over_time_cumulative.csv"
    local_dimension_over_time_csv = by_dimension_over_time_dir / "dimension_alignment_over_time_local.csv"
    cumulative_dimension_over_time_mean_plot = by_dimension_over_time_dir / "dimension_alignment_over_time_cumulative_mean.png"
    cumulative_dimension_over_time_median_plot = by_dimension_over_time_dir / "dimension_alignment_over_time_cumulative_median.png"
    local_dimension_over_time_mean_plot = by_dimension_over_time_dir / "dimension_alignment_over_time_local_mean.png"
    local_dimension_over_time_median_plot = by_dimension_over_time_dir / "dimension_alignment_over_time_local_median.png"
    dimension_alignment_csv = final_dimension_alignment_dir / "dimension_selection_accuracy_alignment.csv"
    dimension_alignment_plot = final_dimension_alignment_dir / "dimension_selection_accuracy_alignment.png"
    if not cumulative_dimension_over_time_df.empty:
        cumulative_dimension_over_time_df.to_csv(cumulative_dimension_over_time_csv, index=False)
        _plot_dimension_over_time_alignment(cumulative_dimension_over_time_df, "corr_selection_share_vs_mean_accuracy", "Correlación por dimensión y epoch (cumulative, mean)", cumulative_dimension_over_time_mean_plot)
        _plot_dimension_over_time_alignment(cumulative_dimension_over_time_df, "corr_selection_share_vs_median_accuracy", "Correlación por dimensión y epoch (cumulative, median)", cumulative_dimension_over_time_median_plot)
    if not local_dimension_over_time_df.empty:
        local_dimension_over_time_df.to_csv(local_dimension_over_time_csv, index=False)
        _plot_dimension_over_time_alignment(local_dimension_over_time_df, "corr_selection_share_vs_mean_accuracy", "Correlación por dimensión y epoch (local, mean)", local_dimension_over_time_mean_plot)
        _plot_dimension_over_time_alignment(local_dimension_over_time_df, "corr_selection_share_vs_median_accuracy", "Correlación por dimensión y epoch (local, median)", local_dimension_over_time_median_plot)
    if not dimension_alignment_df.empty:
        dimension_alignment_df.to_csv(dimension_alignment_csv, index=False)
        _plot_dimension_alignment(dimension_alignment_df, dimension_alignment_plot)

    manifest = {
        "experiment_id": files.experiment_id,
        "source_artifacts": {
            "summary_json": str(files.summary_json),
            "architectures_csv": str(files.architectures_csv),
            "controller_history_csv": str(files.controller_csv),
        },
        "search_space": {
            "dimensions": list(search_space_options.keys()),
            "configured_options_per_dimension": {key: len(values) for key, values in search_space_options.items()},
        },
        "generated_artifacts": {
            "metrics_by_epoch_csv": str(metrics_by_epoch_csv),
            "metrics_by_sample_csv": str(metrics_by_sample_csv),
            "dimension_value_stats_csv": str(dimension_value_stats_csv),
            "dimension_importance_csv": str(dimension_importance_csv),
            "pairwise_interactions_csv": str(pairwise_interactions_csv),
            "epoch_cumulative_png": str(epoch_cumulative_plot),
            "epoch_rolling_png": str(epoch_rolling_plot),
            "controller_loss_png": str(controller_loss_plot),
            "dimension_importance_png": str(dimension_importance_plot),
            "pairwise_interactions_png": str(pairwise_interactions_plot),
            "layer_distribution_png": str(layer_distribution_plot),
            "cumulative_layer_distribution_png": str(cumulative_layer_distribution_plot),
            "tables_dir": str(tables_dir),
            "overview_dir": str(overview_dir),
            "search_space_dir": str(search_space_dir),
            "dimensions_dir": str(dimensions_dir),
            "correlations_dir": str(correlations_dir),
            "correlations_by_value_cumulative_dir": str(correlations_dir / "by_value_cumulative"),
            "correlations_by_value_local_dir": str(correlations_dir / "by_value_local"),
            "correlations_by_dimension_over_time_dir": str(by_dimension_over_time_dir),
            "correlations_final_dimension_alignment_dir": str(final_dimension_alignment_dir),
            "dimension_alignment_over_time_cumulative_csv": str(cumulative_dimension_over_time_csv),
            "dimension_alignment_over_time_local_csv": str(local_dimension_over_time_csv),
            "dimension_alignment_over_time_cumulative_mean_png": str(cumulative_dimension_over_time_mean_plot),
            "dimension_alignment_over_time_cumulative_median_png": str(cumulative_dimension_over_time_median_plot),
            "dimension_alignment_over_time_local_mean_png": str(local_dimension_over_time_mean_plot),
            "dimension_alignment_over_time_local_median_png": str(local_dimension_over_time_median_plot),
            "dimension_selection_accuracy_alignment_csv": str(dimension_alignment_csv),
            "dimension_selection_accuracy_alignment_png": str(dimension_alignment_plot),
        },
        "metric_semantics": {
            "cumulative_mean_accuracy / cumulative_median_accuracy": "Media y mediana acumuladas sobre las métricas por epoch o por candidata.",
            "cumulative_max_accuracy / cumulative_min_accuracy": "Media acumulada de los máximos y mínimos por epoch.",
            "cumulative_best_max_accuracy / cumulative_best_min_accuracy": "Mejor techo y mejor suelo alcanzados históricamente al recorrer los epochs.",
            "rolling_mean_accuracy / rolling_median_accuracy": "Media y mediana sobre una ventana móvil definida por rolling_window.",
            "rolling_max_accuracy / rolling_min_accuracy": "Media móvil de los máximos y mínimos por epoch en la ventana rolling_window.",
            "rolling_best_max_accuracy / rolling_best_min_accuracy": "Mejor techo y mejor suelo dentro de cada ventana móvil definida por rolling_window.",
            "eta_squared_accuracy": "Fracción de varianza de accuracy explicada por una dimensión o por una interacción de pares dentro de las arquitecturas sampleadas.",
            "relative_importance": "Normalización de eta_squared_accuracy sobre la suma de todas las dimensiones analizadas.",
            "sample_coverage": "Proporción de valores configurados para una dimensión que realmente aparecieron en el muestreo.",
            "sampling_entropy": "Diversidad relativa del muestreo dentro de una dimensión; 1.0 implica distribución muy repartida entre valores sampleados.",
            "by_value_cumulative correlations": "Correlaciones por valor usando series acumuladas de selección y accuracy, calculadas solo cuando el valor aparece en al menos MIN_VALID_EPOCHS_FOR_CORRELATION epochs válidos.",
            "by_value_local correlations": "Correlaciones por valor usando métricas locales por epoch de selección y accuracy, calculadas solo cuando el valor aparece en al menos MIN_VALID_EPOCHS_FOR_CORRELATION epochs válidos.",
            "by_dimension_over_time correlations": "Para cada dimensión y cada epoch, correlación entre selection_share y accuracy entre los valores válidos de la dimensión; se ofrece en variantes cumulative y local.",
            "value_correlations_cumulative": "Correlaciones by value calculadas sobre métricas acumuladas por epoch (selection count/share acumuladas frente a accuracy mean/median acumuladas).",
            "value_correlations_local": "Correlaciones by value calculadas sobre métricas locales por epoch (selection count/share del epoch frente a accuracy mean/median del epoch). Solo se calculan si el valor aparece en al menos MIN_VALID_EPOCHS_FOR_CORRELATION epochs válidos.",
        },
    }
    manifest_json.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    return SearchAnalysisArtifacts(
        output_dir=output_root,
        tables_dir=tables_dir,
        overview_dir=overview_dir,
        search_space_dir=search_space_dir,
        dimensions_dir=dimensions_dir,
        correlations_dir=correlations_dir,
        metrics_by_epoch_csv=metrics_by_epoch_csv,
        metrics_by_sample_csv=metrics_by_sample_csv,
        dimension_value_stats_csv=dimension_value_stats_csv,
        dimension_importance_csv=dimension_importance_csv,
        pairwise_interactions_csv=pairwise_interactions_csv,
        manifest_json=manifest_json,
        epoch_cumulative_plot=epoch_cumulative_plot,
        epoch_rolling_plot=epoch_rolling_plot,
        controller_loss_plot=controller_loss_plot,
        dimension_importance_plot=dimension_importance_plot,
        pairwise_interactions_plot=pairwise_interactions_plot,
        layer_distribution_plot=layer_distribution_plot,
        cumulative_layer_distribution_plot=cumulative_layer_distribution_plot,
        dimension_profiles_dir=dimension_profiles_dir,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Genera análisis estadísticos y gráficas de un experimento NAS ya ejecutado.")
    parser.add_argument(
        "--summary-json",
        required=True,
        help="Ruta a artifacts/nas/searches/<nas_search_signature>/runs/<run_id>/search_summary.json o equivalente.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Carpeta de salida para el análisis. Si no se indica, se crea una carpeta analysis junto al summary.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    artifacts = generate_search_analysis(summary_json=args.summary_json, output_dir=args.output_dir)
    print(json.dumps({
        "output_dir": str(artifacts.output_dir),
        "tables_dir": str(artifacts.tables_dir),
        "overview_dir": str(artifacts.overview_dir),
        "search_space_dir": str(artifacts.search_space_dir),
        "dimensions_dir": str(artifacts.dimensions_dir),
        "correlations_dir": str(artifacts.correlations_dir),
        "metrics_by_epoch_csv": str(artifacts.metrics_by_epoch_csv),
        "metrics_by_sample_csv": str(artifacts.metrics_by_sample_csv),
        "dimension_value_stats_csv": str(artifacts.dimension_value_stats_csv),
        "dimension_importance_csv": str(artifacts.dimension_importance_csv),
        "pairwise_interactions_csv": str(artifacts.pairwise_interactions_csv),
        "manifest_json": str(artifacts.manifest_json),
        "epoch_cumulative_plot": str(artifacts.epoch_cumulative_plot),
        "epoch_rolling_plot": str(artifacts.epoch_rolling_plot),
        "controller_loss_plot": str(artifacts.controller_loss_plot),
        "dimension_importance_plot": str(artifacts.dimension_importance_plot),
        "pairwise_interactions_plot": str(artifacts.pairwise_interactions_plot),
        "layer_distribution_plot": str(artifacts.layer_distribution_plot),
        "cumulative_layer_distribution_plot": str(artifacts.cumulative_layer_distribution_plot),
        "dimension_profiles_dir": str(artifacts.dimension_profiles_dir),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
