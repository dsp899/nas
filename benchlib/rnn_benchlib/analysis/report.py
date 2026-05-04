from __future__ import annotations

import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from rnn_benchlib.storage.jsonl import read_json, read_jsonl, write_json


def load_results_from_paths(results_jsonl_path: str, experiment_meta_path: Optional[str] = None) -> tuple[list[dict], dict]:
    rows = read_jsonl(results_jsonl_path)
    meta = read_json(experiment_meta_path, default={}) if experiment_meta_path else {}
    return rows, meta or {}


def flatten_records(rows: List[Dict]) -> pd.DataFrame:
    flat: List[Dict] = []
    for row in rows:
        experiment = row.get("experiment", {})
        runtime_summary = row.get("runtime_summary", {})
        numeric_check = row.get("numeric_check") or {}
        video_timing = row.get("video_timing", {})
        extra = row.get("extra", {})
        clip_timings = row.get("clip_timings", []) or []
        flat.append({
            "experiment_id": experiment.get("experiment_id"),
            "experiment_name": experiment.get("experiment_name"),
            "model_id": row.get("model_id"),
            "runtime_kind": row.get("runtime_kind"),
            "memory_mode": row.get("memory_mode"),
            "seq": row.get("seq"),
            "video_index": row.get("video_index"),
            "clips_per_video": row.get("clips_per_video"),
            "threads": row.get("threads"),
            "batch_size": row.get("batch_size"),
            "direction": extra.get("direction"),
            "rnn": extra.get("rnn"),
            "head_units": extra.get("head_units"),
            "num_classes": extra.get("num_classes"),
            "video_decision": extra.get("video_decision"),
            "video_decision_input": extra.get("video_decision_input"),
            "feature_dim": extra.get("feature_dim"),
            "video_steps": extra.get("video_steps"),
            "init_ms": runtime_summary.get("init_ms"),
                        "steady_clip_encoder_mean_ms": runtime_summary.get("steady_clip_encoder_mean_ms"),
            "steady_clip_head_mean_ms": runtime_summary.get("steady_clip_head_mean_ms"),
            "steady_clip_e2e_wall_mean_ms": runtime_summary.get("steady_clip_e2e_wall_mean_ms"),
            "steady_clip_e2e_sum_mean_ms": runtime_summary.get("steady_clip_e2e_sum_mean_ms"),
            "steady_clip_bridge_mean_ms": runtime_summary.get("steady_clip_bridge_mean_ms"),
            "steady_video_aggregation_mean_ms": runtime_summary.get("steady_video_aggregation_mean_ms"),
            "steady_video_head_mean_ms": runtime_summary.get("steady_video_head_mean_ms"),
            "steady_video_e2e_wall_mean_ms": runtime_summary.get("steady_video_e2e_wall_mean_ms"),
            "steady_video_e2e_sum_mean_ms": runtime_summary.get("steady_video_e2e_sum_mean_ms"),
            "video_aggregation_ms": video_timing.get("video_aggregation_ms"),
            "video_encoder_sum_ms": video_timing.get("video_encoder_sum_ms"),
            "video_bridge_sum_ms": video_timing.get("video_bridge_sum_ms"),
            "video_head_clip_sum_ms": video_timing.get("video_head_clip_sum_ms"),
            "video_head_ms": video_timing.get("video_head_ms"),
            "video_e2e_sum_ms": video_timing.get("video_e2e_sum_ms"),
            "video_e2e_wall_ms": video_timing.get("video_e2e_wall_ms"),
            "numeric_max_abs_diff": numeric_check.get("max_abs_diff"),
            "numeric_mean_abs_diff": numeric_check.get("mean_abs_diff"),
            "numeric_allclose": numeric_check.get("allclose_atol_1e5_rtol_1e5"),
            "mean_clip_encoder_ms": _mean_clip_field(clip_timings, "clip_encoder_ms"),
            "mean_clip_bridge_ms": _mean_clip_field(clip_timings, "clip_bridge_ms"),
            "mean_clip_head_ms": _mean_clip_field(clip_timings, "clip_head_ms"),
            "mean_clip_e2e_sum_ms": _mean_clip_field(clip_timings, "clip_e2e_sum_ms"),
            "mean_clip_e2e_wall_ms": _mean_clip_field(clip_timings, "clip_e2e_wall_ms"),
            "total_units": _total_units_from_row(row),
        })
    df = pd.DataFrame(flat)
    return df


def _mean_clip_field(clip_timings: List[Dict], field: str) -> Optional[float]:
    if not clip_timings:
        return None
    values = [float(item.get(field, 0.0)) for item in clip_timings]
    return sum(values) / len(values)


def _total_units_from_row(row: Dict) -> Optional[int]:
    extra = row.get("extra") or {}
    units = [extra.get("units_0"), extra.get("units_1"), extra.get("units_2")]
    vals = [int(u) for u in units if isinstance(u, int) and int(u) > 0]
    return sum(vals) if vals else None


def build_model_runtime_summary(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "model_id", "runtime_kind", "direction", "memory_mode", "rnn", "seq",
        "head_units", "num_classes", "video_decision", "video_decision_input", "feature_dim", "video_steps",
    ]
    agg = df.groupby(group_cols, dropna=False).agg(
        videos=("video_index", "count"),
        init_ms=("init_ms", "first"),
        first_clip_total_ms=("first_clip_total_ms", "mean"),
        steady_clip_encoder_mean_ms=("steady_clip_encoder_mean_ms", "mean"),
        steady_clip_bridge_mean_ms=("steady_clip_bridge_mean_ms", "mean"),
        steady_clip_head_mean_ms=("steady_clip_head_mean_ms", "mean"),
        steady_clip_e2e_wall_mean_ms=("steady_clip_e2e_wall_mean_ms", "mean"),
        steady_clip_e2e_sum_mean_ms=("steady_clip_e2e_sum_mean_ms", "mean"),
        steady_video_aggregation_mean_ms=("steady_video_aggregation_mean_ms", "mean"),
        steady_video_head_mean_ms=("steady_video_head_mean_ms", "mean"),
        steady_video_e2e_wall_mean_ms=("steady_video_e2e_wall_mean_ms", "mean"),
        steady_video_e2e_sum_mean_ms=("steady_video_e2e_sum_mean_ms", "mean"),
        numeric_max_abs_diff=("numeric_max_abs_diff", "max"),
        numeric_mean_abs_diff=("numeric_mean_abs_diff", "mean"),
    ).reset_index()
    return agg


def build_paired_float_vs_tflite_summary(model_runtime_df: pd.DataFrame) -> pd.DataFrame:
    subset = model_runtime_df[[
        "model_id", "runtime_kind", "direction", "memory_mode", "rnn", "seq", "head_units", "num_classes",
        "video_decision", "video_decision_input", "feature_dim", "video_steps",
        "init_ms", "steady_clip_e2e_wall_mean_ms", "steady_video_e2e_wall_mean_ms", "steady_clip_e2e_sum_mean_ms", "steady_video_e2e_sum_mean_ms",
        "steady_clip_encoder_mean_ms", "steady_clip_bridge_mean_ms", "steady_clip_head_mean_ms", "steady_video_aggregation_mean_ms", "steady_video_head_mean_ms",
        "numeric_max_abs_diff", "numeric_mean_abs_diff",
    ]].copy()
    pivot = subset.pivot_table(
        index=["model_id", "direction", "memory_mode", "rnn", "seq", "head_units", "num_classes", "video_decision", "video_decision_input", "feature_dim", "video_steps"],
        columns="runtime_kind",
        values=[
            "init_ms", "steady_clip_e2e_wall_mean_ms", "steady_video_e2e_wall_mean_ms", "steady_clip_e2e_sum_mean_ms", "steady_video_e2e_sum_mean_ms",
            "steady_clip_encoder_mean_ms", "steady_clip_bridge_mean_ms", "steady_clip_head_mean_ms", "steady_video_aggregation_mean_ms", "steady_video_head_mean_ms",
            "numeric_max_abs_diff", "numeric_mean_abs_diff",
        ],
        aggfunc="first",
    )
    if pivot.empty:
        return pd.DataFrame()
    pivot.columns = [f"{metric}_{runtime}" for metric, runtime in pivot.columns]
    out = pivot.reset_index()
    if "steady_clip_e2e_wall_mean_ms_float" in out.columns and "steady_clip_e2e_wall_mean_ms_tflite" in out.columns:
        out["speedup_clip_e2e_wall"] = out["steady_clip_e2e_wall_mean_ms_float"] / out["steady_clip_e2e_wall_mean_ms_tflite"]
    if "steady_video_e2e_wall_mean_ms_float" in out.columns and "steady_video_e2e_wall_mean_ms_tflite" in out.columns:
        out["speedup_video_e2e_wall"] = out["steady_video_e2e_wall_mean_ms_float"] / out["steady_video_e2e_wall_mean_ms_tflite"]
    return out


def build_field_runtime_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for field in ["direction", "memory_mode", "rnn", "video_decision", "video_decision_input", "seq", "head_units", "num_classes"]:
        group = df.groupby([field, "runtime_kind"], dropna=False).agg(
            models=("model_id", "nunique"),
            rows=("video_index", "count"),
            steady_clip_e2e_wall_mean_ms=("steady_clip_e2e_wall_mean_ms", "mean"),
        steady_clip_e2e_sum_mean_ms=("steady_clip_e2e_sum_mean_ms", "mean"),
            steady_video_e2e_wall_mean_ms=("steady_video_e2e_wall_mean_ms", "mean"),
        steady_video_e2e_sum_mean_ms=("steady_video_e2e_sum_mean_ms", "mean"),
            init_ms=("init_ms", "mean"),
        ).reset_index()
        group.insert(0, "field", field)
        group = group.rename(columns={field: "field_value"})
        rows.extend(group.to_dict(orient="records"))
    return pd.DataFrame(rows)


def write_analysis_outputs(output_dir: str, df: pd.DataFrame, meta: Dict) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    raw_csv = os.path.join(output_dir, "records_flat.csv")
    model_csv = os.path.join(output_dir, "model_runtime_summary.csv")
    paired_csv = os.path.join(output_dir, "paired_float_vs_tflite.csv")
    field_csv = os.path.join(output_dir, "field_runtime_summary.csv")
    summary_json = os.path.join(output_dir, "analysis_summary.json")
    report_md = os.path.join(output_dir, "benchmark_report.md")

    model_runtime_df = build_model_runtime_summary(df)
    paired_df = build_paired_float_vs_tflite_summary(model_runtime_df)
    field_df = build_field_runtime_summary(df)

    df.to_csv(raw_csv, index=False)
    model_runtime_df.to_csv(model_csv, index=False)
    paired_df.to_csv(paired_csv, index=False)
    field_df.to_csv(field_csv, index=False)

    summary_payload = build_summary_payload(df, model_runtime_df, paired_df, meta)
    write_json(summary_json, summary_payload, indent=2)
    _write_markdown_report(report_md, summary_payload)

    _plot_float_vs_tflite_scatter(paired_df, os.path.join(plots_dir, "float_vs_tflite_video_total_scatter.png"))
    _plot_speedup_hist(paired_df, os.path.join(plots_dir, "speedup_video_total_hist.png"))
    _plot_component_breakdown(df, os.path.join(plots_dir, "component_breakdown_by_runtime.png"))
    _plot_box_by_field(df, "direction", os.path.join(plots_dir, "latency_by_direction.png"))
    _plot_box_by_field(df, "memory_mode", os.path.join(plots_dir, "latency_by_memory_mode.png"))
    _plot_units_vs_latency(df, os.path.join(plots_dir, "latency_vs_total_units.png"))

    return {
        "raw_csv": raw_csv,
        "model_runtime_csv": model_csv,
        "paired_csv": paired_csv,
        "field_csv": field_csv,
        "summary_json": summary_json,
        "report_md": report_md,
        "plots_dir": plots_dir,
    }


def build_summary_payload(df: pd.DataFrame, model_runtime_df: pd.DataFrame, paired_df: pd.DataFrame, meta: Dict) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "experiment": meta,
        "rows": int(len(df)),
        "models": int(df["model_id"].nunique()) if not df.empty else 0,
        "runtimes": sorted(str(v) for v in df["runtime_kind"].dropna().unique().tolist()) if not df.empty else [],
    }
    if not paired_df.empty:
        payload["speedup_video_e2e_wall_median"] = float(paired_df["speedup_video_e2e_wall"].median()) if "speedup_video_e2e_wall" in paired_df.columns else None
        payload["speedup_video_e2e_wall_mean"] = float(paired_df["speedup_video_e2e_wall"].mean()) if "speedup_video_e2e_wall" in paired_df.columns else None
        top = paired_df.sort_values("speedup_video_e2e_wall", ascending=False).head(10) if "speedup_video_e2e_wall" in paired_df.columns else pd.DataFrame()
        payload["top_speedups"] = top[[c for c in ["model_id", "speedup_video_e2e_wall", "speedup_clip_e2e_wall", "direction", "memory_mode", "seq", "video_decision_input"] if c in top.columns]].to_dict(orient="records")
    if not model_runtime_df.empty:
        fastest = model_runtime_df.sort_values("steady_video_e2e_wall_mean_ms", ascending=True).head(10)
        payload["fastest_model_runtime_rows"] = fastest[[c for c in ["model_id", "runtime_kind", "steady_video_e2e_wall_mean_ms", "steady_clip_e2e_wall_mean_ms", "direction", "memory_mode", "seq"] if c in fastest.columns]].to_dict(orient="records")
    return payload


def _write_markdown_report(path: str, summary: Dict[str, object]) -> None:
    lines = [
        "# Benchmark analysis report",
        "",
        f"- Rows: {summary.get('rows')}",
        f"- Models: {summary.get('models')}",
        f"- Runtimes: {', '.join(summary.get('runtimes', []))}",
        f"- Median speedup video e2e wall (float/tflite): {summary.get('speedup_video_e2e_wall_median')}",
        f"- Mean speedup video e2e wall (float/tflite): {summary.get('speedup_video_e2e_wall_mean')}",
        "",
        "## Top speedups",
        "",
    ]
    for item in summary.get("top_speedups", []):
        lines.append(f"- {item}")
    lines += ["", "## Fastest rows", ""]
    for item in summary.get("fastest_model_runtime_rows", []):
        lines.append(f"- {item}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _plot_float_vs_tflite_scatter(df: pd.DataFrame, path: str) -> None:
    if df.empty or "steady_video_e2e_wall_mean_ms_float" not in df.columns or "steady_video_e2e_wall_mean_ms_tflite" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(df["steady_video_e2e_wall_mean_ms_float"], df["steady_video_e2e_wall_mean_ms_tflite"], alpha=0.8)
    max_val = float(max(df["steady_video_e2e_wall_mean_ms_float"].max(), df["steady_video_e2e_wall_mean_ms_tflite"].max()))
    ax.plot([0, max_val], [0, max_val])
    ax.set_xlabel("float steady_video_total_mean_ms")
    ax.set_ylabel("tflite steady_video_total_mean_ms")
    ax.set_title("Float vs TFLite video latency")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_speedup_hist(df: pd.DataFrame, path: str) -> None:
    if df.empty or "speedup_video_e2e_wall" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df["speedup_video_e2e_wall"].dropna(), bins=20)
    ax.set_xlabel("float / tflite speedup (video e2e wall)")
    ax.set_ylabel("count")
    ax.set_title("Distribution of video speedup")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_component_breakdown(df: pd.DataFrame, path: str) -> None:
    if df.empty:
        return
    agg = df.groupby("runtime_kind", dropna=False)[[
        "steady_encoder_mean_ms", "steady_head_mean_ms", "steady_aggregation_mean_ms", "steady_video_head_mean_ms"
    ]].mean(numeric_only=True)
    if agg.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    bottom = None
    x = range(len(agg.index))
    components = [
        ("steady_encoder_mean_ms", "encoder"),
        ("steady_head_mean_ms", "clip_head"),
        ("steady_aggregation_mean_ms", "aggregation"),
        ("steady_video_head_mean_ms", "video_head"),
    ]
    for col, label in components:
        values = agg[col].fillna(0.0).tolist()
        ax.bar(x, values, bottom=bottom, label=label)
        if bottom is None:
            bottom = values
        else:
            bottom = [b + v for b, v in zip(bottom, values)]
    ax.set_xticks(list(x))
    ax.set_xticklabels(list(agg.index))
    ax.set_ylabel("mean ms")
    ax.set_title("Component latency breakdown by runtime")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_box_by_field(df: pd.DataFrame, field: str, path: str) -> None:
    if df.empty or field not in df.columns:
        return
    subset = df[[field, "runtime_kind", "steady_video_e2e_wall_mean_ms"]].dropna()
    if subset.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    labels: List[str] = []
    data: List[List[float]] = []
    for runtime in sorted(subset["runtime_kind"].unique().tolist()):
        for field_value in sorted(str(v) for v in subset[field].unique().tolist()):
            values = subset[(subset["runtime_kind"] == runtime) & (subset[field].astype(str) == field_value)]["steady_video_e2e_wall_mean_ms"].tolist()
            if values:
                labels.append(f"{runtime}\n{field_value}")
                data.append(values)
    if not data:
        return
    ax.boxplot(data, tick_labels=labels)
    ax.set_ylabel("steady_video_e2e_wall_mean_ms")
    ax.set_title(f"Latency by {field}")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_units_vs_latency(df: pd.DataFrame, path: str) -> None:
    subset = df[["total_units", "runtime_kind", "steady_video_e2e_wall_mean_ms"]].dropna()
    if subset.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for runtime in sorted(subset["runtime_kind"].unique().tolist()):
        part = subset[subset["runtime_kind"] == runtime]
        ax.scatter(part["total_units"], part["steady_video_e2e_wall_mean_ms"], alpha=0.8, label=str(runtime))
    ax.set_xlabel("total units")
    ax.set_ylabel("steady_video_e2e_wall_mean_ms")
    ax.set_title("Latency vs total units")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
