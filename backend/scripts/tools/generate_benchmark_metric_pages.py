from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


REPO_ROOT = Path(__file__).resolve().parents[3]
LOG_DIR = REPO_ROOT / "log"


MODELS: List[str] = [
    "Baseline (CV)",
    "Camera-only Transformer",
    "Fusion Transformer"
]


PRESETS: Dict[str, Dict[str, object]] = {
    "measured": {
        "display_name": "Measured benchmark",
        "source_primary": "Source: provided benchmark values",
        "source_runtime": "Source: provided runtime benchmark",
        "optional_note": "Estimated supporting CV metric chart (replace with measured evaluation values when available)",
        "primary_metrics": {
            "minADE@3 (m)": [0.65, 0.55, 0.54],
            "minFDE@3 (m)": [1.35, 1.09, 1.07],
            "Miss Rate >2.0m (%)": [19.9, 13.0, 12.4],
        },
        "runtime_metrics": {
            "Detection latency": (76.0, "ms/frame"),
            "Transformer predict latency": (13.6, "ms"),
            "End-to-end live cycle": (89.6, "ms"),
            "End-to-end throughput": (11.6, "FPS"),
        },
        "runtime_targets": {
            "Detection latency": (60.0, True),
            "Transformer predict latency": (20.0, True),
            "End-to-end live cycle": (66.7, True),
            "End-to-end throughput": (15.0, False),
        },
        "optional_metrics": {
            "Precision (%)": ([74.0, 85.0, 88.0], 85.0),
            "Recall (%)": ([68.0, 80.0, 83.0], 80.0),
            "F1 (%)": ([71.0, 82.0, 85.0], 82.0),
            "mAP@0.5 (%)": ([62.0, 76.0, 79.0], 75.0),
            "mAP@[0.5:0.95] (%)": ([34.0, 46.0, 49.0], 45.0),
            "IoU (%)": ([52.0, 62.0, 65.0], 60.0),
        },
    },
    "best": {
        "display_name": "Best benchmark (analyzed target)",
        "source_primary": "Source: analyst-optimized trajectory target",
        "source_runtime": "Source: analyst-optimized runtime target",
        "optional_note": "Analyzed best-case CV metric chart (target values)",
        "primary_metrics": {
            "minADE@3 (m)": [0.65, 0.50, 0.42],
            "minFDE@3 (m)": [1.35, 0.95, 0.78],
            "Miss Rate >2.0m (%)": [19.9, 9.8, 7.1],
        },
        "runtime_metrics": {
            "Detection latency": (42.0, "ms/frame"),
            "Transformer predict latency": (8.5, "ms"),
            "End-to-end live cycle": (55.0, "ms"),
            "End-to-end throughput": (18.2, "FPS"),
        },
        "runtime_targets": {
            "Detection latency": (45.0, True),
            "Transformer predict latency": (10.0, True),
            "End-to-end live cycle": (60.0, True),
            "End-to-end throughput": (16.0, False),
        },
        "optional_metrics": {
            "Precision (%)": ([74.0, 89.0, 92.0], 90.0),
            "Recall (%)": ([68.0, 86.0, 90.0], 88.0),
            "F1 (%)": ([71.0, 87.0, 91.0], 89.0),
            "mAP@0.5 (%)": ([62.0, 82.0, 86.0], 85.0),
            "mAP@[0.5:0.95] (%)": ([34.0, 54.0, 60.0], 58.0),
            "IoU (%)": ([52.0, 66.0, 72.0], 70.0),
        },
    },
}


COLOR_BG = "#030712"
COLOR_BAND = "#0B152A"
COLOR_PANEL = "#09121F"
COLOR_GRID = "#314258"
MODEL_COLORS = ["#5E6B7E", "#69B3FF", "#7BE5A7"]
COLOR_LINE = "#B5E6FF"
COLOR_GOOD = "#7BE5A7"
COLOR_WARN = "#FFC47A"
COLOR_BAD = "#FF7F96"
COLOR_TARGET = "#8CBFFF"


def setup_theme() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": COLOR_BG,
            "axes.facecolor": COLOR_PANEL,
            "savefig.facecolor": COLOR_BG,
            "text.color": "#FFFFFF",
            "axes.labelcolor": "#FFFFFF",
            "xtick.color": "#FFFFFF",
            "ytick.color": "#FFFFFF",
            "axes.edgecolor": "#C5D4EA",
            "font.family": "Calibri",
            "font.size": 22,
        }
    )


def clean_name(name: str) -> str:
    out = "".join(ch if ch.isalnum() else "_" for ch in name)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_").lower()


def pct_improvement(old: float, new: float) -> float:
    if abs(old) < 1e-12:
        return 0.0
    return 100.0 * (old - new) / old


def style_figure(fig: plt.Figure) -> None:
    fig.patch.set_facecolor(COLOR_BG)
    fig.add_artist(
        Rectangle(
            (0, 0.92),
            1,
            0.08,
            transform=fig.transFigure,
            facecolor=COLOR_BAND,
            alpha=0.95,
            linewidth=0,
            zorder=0,
        )
    )


def style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(COLOR_PANEL)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.8, color=COLOR_GRID, alpha=0.55)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color("#C5D4EA")


def draw_model_metric_page(
    title: str,
    values: Sequence[float],
    out_path: Path,
    lower_is_better: bool = True,
    is_percent: bool = False,
    footnote: str = "Source: provided benchmark values",
) -> None:
    fig, ax = plt.subplots(figsize=(13.333, 7.5), dpi=150)

    style_figure(fig)
    style_axes(ax)

    x = np.arange(len(MODELS))
    bars = ax.bar(x, values, color=MODEL_COLORS, edgecolor="#DCE8F6", linewidth=1.2, zorder=2)
    ax.plot(x, values, color=COLOR_LINE, linewidth=2.8, marker="o", markersize=7, zorder=3)

    value_suffix = "%" if is_percent or ("%" in title) else ""
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.2f}{value_suffix}",
            ha="center",
            va="bottom",
            fontsize=18,
            color="#FFFFFF",
            zorder=4,
        )

    baseline = values[0]
    cam = values[1]
    fusion = values[2]

    if lower_is_better:
        cam_delta = pct_improvement(baseline, cam)
        fusion_delta = pct_improvement(baseline, fusion)
        subtitle = f"Improvement vs Baseline: Camera {cam_delta:.1f}% | Fusion {fusion_delta:.1f}%"
    else:
        cam_delta = 100.0 * (cam - baseline) / max(1e-12, baseline)
        fusion_delta = 100.0 * (fusion - baseline) / max(1e-12, baseline)
        subtitle = f"Gain vs Baseline: Camera {cam_delta:.1f}% | Fusion {fusion_delta:.1f}%"

    ax.set_title(title, fontsize=40, weight="bold", pad=18)
    ax.set_ylabel("Value", fontsize=24)
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS)
    ax.tick_params(axis="x", labelrotation=0, labelsize=15)
    ax.tick_params(axis="y", labelsize=18)
    ax.margins(x=0.05)

    fig.text(0.5, 0.06, subtitle, ha="center", va="center", fontsize=22, color="#FFFFFF")
    fig.text(0.01, 0.01, footnote, ha="left", va="bottom", fontsize=12, color="#D0D0D0")
    fig.tight_layout(rect=(0.02, 0.10, 0.98, 0.95))
    fig.savefig(out_path)
    plt.close(fig)


def draw_runtime_metric_page(
    title: str,
    value: float,
    unit: str,
    target: float,
    lower_is_better: bool,
    out_path: Path,
    footnote: str,
) -> None:
    fig, ax = plt.subplots(figsize=(13.333, 7.5), dpi=150)

    style_figure(fig)
    style_axes(ax)

    labels = ["Measured", "Target"]
    vals = [value, target]

    if lower_is_better:
        measured_color = COLOR_GOOD if value <= target else COLOR_BAD
        status = f"Gap vs target: {value - target:+.1f} {unit}"
        hint = "Lower is better"
    else:
        measured_color = COLOR_GOOD if value >= target else COLOR_WARN
        status = f"Gap vs target: {value - target:+.1f} {unit}"
        hint = "Higher is better"

    bars = ax.barh(labels, vals, color=[measured_color, COLOR_TARGET], edgecolor="#DCE8F6", linewidth=1.2, zorder=2)

    for bar, val in zip(bars, vals):
        ax.text(
            val + max(vals) * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f} {unit}",
            ha="left",
            va="center",
            fontsize=20,
            color="#FFFFFF",
        )

    ax.set_xlim(0, max(vals) * 1.45)
    ax.set_title(f"{title} ({unit})", fontsize=40, weight="bold", pad=18)
    ax.set_xlabel("Value", fontsize=22)
    ax.tick_params(axis="x", labelsize=16)
    ax.tick_params(axis="y", labelsize=20)

    fig.text(0.5, 0.06, f"{hint} | {status}", ha="center", va="center", fontsize=22, color="#FFFFFF")
    fig.text(0.01, 0.01, footnote, ha="left", va="bottom", fontsize=12, color="#D0D0D0")

    fig.tight_layout(rect=(0.03, 0.10, 0.98, 0.95))
    fig.savefig(out_path)
    plt.close(fig)


def draw_optional_metric_page(
    metric_name: str,
    values: Sequence[float],
    target: float,
    out_path: Path,
    footnote: str,
) -> None:
    fig, ax = plt.subplots(figsize=(13.333, 7.5), dpi=150)

    style_figure(fig)
    style_axes(ax)

    x = np.arange(len(MODELS))
    bars = ax.bar(x, values, color=MODEL_COLORS, edgecolor="#DCE8F6", linewidth=1.2, zorder=2)
    ax.plot(x, values, color=COLOR_LINE, linewidth=2.8, marker="o", markersize=7, zorder=3)
    ax.axhline(target, color=COLOR_TARGET, linestyle="--", linewidth=2.0, zorder=1)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=18,
            color="#FFFFFF",
        )

    ax.text(
        x[-1] + 0.35,
        target,
        f"Target {target:.1f}%",
        ha="left",
        va="center",
        fontsize=16,
        color=COLOR_TARGET,
    )

    baseline = values[0]
    cam = values[1]
    fusion = values[2]
    cam_gain = 100.0 * (cam - baseline) / max(1e-12, baseline)
    fusion_gain = 100.0 * (fusion - baseline) / max(1e-12, baseline)

    ax.set_title(metric_name, fontsize=40, weight="bold", pad=18)
    ax.set_ylabel("Percent", fontsize=24)
    ax.set_ylim(0, max(max(values), target) * 1.25)
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS)
    ax.tick_params(axis="x", labelrotation=0, labelsize=15)
    ax.tick_params(axis="y", labelsize=18)

    fig.text(
        0.5,
        0.06,
        f"Estimated gain vs Baseline: Camera {cam_gain:.1f}% | Fusion {fusion_gain:.1f}%",
        ha="center",
        va="center",
        fontsize=20,
        color="#FFFFFF",
    )
    fig.text(
        0.01,
        0.01,
        footnote,
        ha="left",
        va="bottom",
        fontsize=11,
        color="#D0D0D0",
    )

    fig.tight_layout(rect=(0.03, 0.10, 0.98, 0.95))
    fig.savefig(out_path)
    plt.close(fig)


def draw_latency_share_chart(
    runtime_metrics: Dict[str, Tuple[float, str]],
    out_path: Path,
    footnote: str,
) -> None:
    det = runtime_metrics["Detection latency"][0]
    pred = runtime_metrics["Transformer predict latency"][0]
    e2e = runtime_metrics["End-to-end live cycle"][0]
    other = max(0.0, e2e - det - pred)

    values = [det, pred]
    labels = ["Detection", "Transformer"]
    colors = [COLOR_BAD, COLOR_GOOD]
    if other > 1e-6:
        values.append(other)
        labels.append("Other")
        colors.append("#7991B0")

    fig, ax = plt.subplots(figsize=(13.333, 7.5), dpi=150)
    style_figure(fig)
    ax.set_facecolor(COLOR_PANEL)

    wedges, _, _ = ax.pie(
        values,
        labels=labels,
        autopct=lambda pct: f"{pct:.1f}%",
        startangle=90,
        colors=colors,
        textprops={"color": "#FFFFFF", "fontsize": 16},
        wedgeprops={"width": 0.38, "edgecolor": COLOR_BG, "linewidth": 2.0},
    )
    for w in wedges:
        w.set_alpha(0.92)

    ax.text(0, 0.06, f"{e2e:.1f} ms", ha="center", va="center", fontsize=34, color="#FFFFFF", weight="bold")
    ax.text(0, -0.12, "total cycle", ha="center", va="center", fontsize=16, color="#FFFFFF")
    ax.set_title("End-to-end latency share", fontsize=40, weight="bold", pad=20)
    ax.axis("equal")

    fig.text(0.5, 0.06, "Detection dominates runtime cost; optimize detector stage first", ha="center", va="center", fontsize=22, color="#FFFFFF")
    fig.text(0.01, 0.01, footnote, ha="left", va="bottom", fontsize=12, color="#D0D0D0")
    fig.savefig(out_path)
    plt.close(fig)


def write_analysis(
    out_dir: Path,
    preset_name: str,
    display_name: str,
    primary_metrics: Dict[str, List[float]],
    runtime_metrics: Dict[str, Tuple[float, str]],
    optional_note: str,
) -> None:
    ade = primary_metrics["minADE@3 (m)"]
    fde = primary_metrics["minFDE@3 (m)"]
    miss = primary_metrics["Miss Rate >2.0m (%)"]

    det = runtime_metrics["Detection latency"][0]
    pred = runtime_metrics["Transformer predict latency"][0]
    e2e = runtime_metrics["End-to-end live cycle"][0]
    fps = runtime_metrics["End-to-end throughput"][0]

    lines: List[str] = []
    lines.append(f"Preset: {preset_name} ({display_name})")
    lines.append("")
    lines.append("Trajectory metric interpretation")
    lines.append("--------------------------------")
    lines.append(f"Baseline -> Fusion ADE improvement: {pct_improvement(ade[0], ade[2]):.2f}%")
    lines.append(f"Baseline -> Fusion FDE improvement: {pct_improvement(fde[0], fde[2]):.2f}%")
    lines.append(f"Baseline -> Fusion Miss Rate improvement: {pct_improvement(miss[0], miss[2]):.2f}%")
    lines.append("")
    lines.append(f"Camera -> Fusion ADE improvement: {pct_improvement(ade[1], ade[2]):.2f}%")
    lines.append(f"Camera -> Fusion FDE improvement: {pct_improvement(fde[1], fde[2]):.2f}%")
    lines.append(f"Camera -> Fusion Miss Rate improvement: {pct_improvement(miss[1], miss[2]):.2f}%")
    lines.append("")
    lines.append("Runtime interpretation")
    lines.append("----------------------")
    lines.append(f"Detection share of end-to-end latency: {100.0 * det / e2e:.2f}%")
    lines.append(f"Transformer share of end-to-end latency: {100.0 * pred / e2e:.2f}%")
    lines.append(f"Current cycle: {e2e:.1f} ms ({fps:.1f} FPS)")
    miss_fusion = miss[2]
    approx_one_in = 100.0 / max(1e-12, miss_fusion)
    lines.append(
        f"Miss Rate {miss_fusion:.1f}% means about 1 in {approx_one_in:.1f} trajectories still exceed 2.0m final error."
    )
    lines.append("")
    lines.append("Supporting CV metrics")
    lines.append("---------------------")
    lines.append(optional_note)

    analysis_file = out_dir / "benchmark_analysis.txt"
    analysis_file.write_text("\n".join(lines), encoding="utf-8")


def write_manifest(
    out_dir: Path,
    preset_name: str,
    display_name: str,
    generated_files: Sequence[Path],
) -> None:
    manifest_file = out_dir / "benchmark_manifest.txt"
    lines = [
        f"Preset: {preset_name}",
        f"Preset display: {display_name}",
        f"Output directory: {out_dir}",
        "",
        "Generated pages:",
    ]
    for item in generated_files:
        lines.append(f"- {item.name}")
    manifest_file.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PPT-ready benchmark metric pages from provided values.")
    parser.add_argument(
        "--preset",
        type=str,
        default="measured",
        choices=sorted(PRESETS.keys()),
        help="Metric preset to render.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Output directory (default: log/ppt_metric_pages/trajectory_benchmark_pack_<preset>)",
    )
    args = parser.parse_args()

    setup_theme()

    preset_cfg = PRESETS[args.preset]
    display_name = str(preset_cfg["display_name"])
    source_primary = str(preset_cfg["source_primary"])
    source_runtime = str(preset_cfg["source_runtime"])
    optional_note = str(preset_cfg["optional_note"])
    primary_metrics = dict(preset_cfg["primary_metrics"])
    runtime_metrics = dict(preset_cfg["runtime_metrics"])
    runtime_targets = dict(preset_cfg["runtime_targets"])
    optional_metrics = dict(preset_cfg["optional_metrics"])

    default_out = LOG_DIR / "ppt_metric_pages" / f"trajectory_benchmark_pack_{args.preset}"
    out_dir = Path(args.output_dir) if args.output_dir else default_out
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for old_png in out_dir.glob("*.png"):
        old_png.unlink()

    generated: List[Path] = []

    primary_defs = [
        ("01_minade_at3", "minADE@3 (m)", True),
        ("02_minfde_at3", "minFDE@3 (m)", True),
        ("03_miss_rate_gt_2m", "Miss Rate >2.0m (%)", True),
    ]
    for file_stem, metric_name, lower_better in primary_defs:
        out_path = out_dir / f"{file_stem}.png"
        draw_model_metric_page(
            metric_name,
            primary_metrics[metric_name],
            out_path,
            lower_is_better=lower_better,
            is_percent=("%" in metric_name),
            footnote=f"{source_primary} | {display_name}",
        )
        generated.append(out_path)

    runtime_defs = [
        ("04_detection_latency", "Detection latency"),
        ("05_transformer_predict_latency", "Transformer predict latency"),
        ("06_end_to_end_cycle", "End-to-end live cycle"),
        ("07_end_to_end_fps", "End-to-end throughput"),
    ]
    for file_stem, metric_key in runtime_defs:
        val, unit = runtime_metrics[metric_key]
        target, lower_better = runtime_targets[metric_key]
        out_path = out_dir / f"{file_stem}.png"
        draw_runtime_metric_page(
            metric_key,
            val,
            unit,
            target,
            lower_better,
            out_path,
            footnote=f"{source_runtime} | {display_name}",
        )
        generated.append(out_path)

    for idx, (metric_name, metric_payload) in enumerate(optional_metrics.items(), start=8):
        values, target = metric_payload
        out_path = out_dir / f"{idx:02d}_{clean_name(metric_name)}_estimated_chart.png"
        draw_optional_metric_page(
            metric_name,
            values,
            target,
            out_path,
            footnote=f"{optional_note} | {display_name}",
        )
        generated.append(out_path)

    latency_share_path = out_dir / "14_latency_share_breakdown.png"
    draw_latency_share_chart(
        runtime_metrics,
        latency_share_path,
        footnote=f"{source_runtime} | {display_name}",
    )
    generated.append(latency_share_path)

    write_analysis(
        out_dir,
        args.preset,
        display_name,
        primary_metrics,
        runtime_metrics,
        optional_note,
    )
    write_manifest(out_dir, args.preset, display_name, generated)

    print(f"Generated {len(generated)} benchmark pages in: {out_dir}")
    print(f"Manifest: {out_dir / 'benchmark_manifest.txt'}")
    print(f"Analysis: {out_dir / 'benchmark_analysis.txt'}")


if __name__ == "__main__":
    main()
