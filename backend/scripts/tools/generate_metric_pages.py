from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[3]
LOG_DIR = REPO_ROOT / "log"


@dataclass
class ParsedMetrics:
    series: Dict[str, List[Tuple[int, float]]] = field(default_factory=dict)
    paired: Dict[str, Tuple[float, float, bool]] = field(default_factory=dict)
    paired_labels: Tuple[str, str] = ("Baseline", "Model")


def canonical_metric(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def sanitize_filename(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", name.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "metric"


def parse_number(token: str) -> Optional[Tuple[float, bool]]:
    s = token.strip()
    is_percent = s.endswith("%")
    s = s.replace("%", "")

    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    if not match:
        return None
    return float(match.group(0)), is_percent


def append_series(series: Dict[str, List[Tuple[int, float]]], metric: str, epoch: Optional[int], value: float) -> None:
    points = series.setdefault(metric, [])
    x = epoch
    if x is None:
        x = points[-1][0] + 1 if points else 1
    points.append((x, value))


def parse_metrics_from_log(log_path: Path) -> ParsedMetrics:
    parsed = ParsedMetrics()
    current_epoch: Optional[int] = None

    lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        epoch_match = re.search(r"^Epoch\s+(\d+)(?:/\d+)?$", line, flags=re.IGNORECASE)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
            continue

        header_match = re.search(r"^METRIC\s*\|\s*(.+?)\s*\|\s*(.+?)\s*$", line, flags=re.IGNORECASE)
        if header_match:
            parsed.paired_labels = (header_match.group(1).strip(), header_match.group(2).strip())
            continue

        train_loss_match = re.search(r"Train Loss:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", line, flags=re.IGNORECASE)
        if train_loss_match:
            append_series(parsed.series, "Train Loss", current_epoch, float(train_loss_match.group(1)))
            continue

        ade_fde_match = re.search(
            r"^ADE:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*FDE:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
            line,
            flags=re.IGNORECASE,
        )
        if ade_fde_match:
            append_series(parsed.series, "ADE", current_epoch, float(ade_fde_match.group(1)))
            append_series(parsed.series, "FDE", current_epoch, float(ade_fde_match.group(2)))
            continue

        val_ade_fde_match = re.search(
            r"^Val\s+ADE:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\|\s*Val\s+FDE:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
            line,
            flags=re.IGNORECASE,
        )
        if val_ade_fde_match:
            append_series(parsed.series, "Val ADE", current_epoch, float(val_ade_fde_match.group(1)))
            append_series(parsed.series, "Val FDE", current_epoch, float(val_ade_fde_match.group(2)))
            continue

        lr_match = re.search(r"Current Learning Rate:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", line, flags=re.IGNORECASE)
        if lr_match:
            append_series(parsed.series, "Learning Rate", current_epoch, float(lr_match.group(1)))
            continue

        lr_pair_match = re.search(
            r"LR\s+base=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\|\s*fusion=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
            line,
            flags=re.IGNORECASE,
        )
        if lr_pair_match:
            append_series(parsed.series, "LR base", current_epoch, float(lr_pair_match.group(1)))
            append_series(parsed.series, "LR fusion", current_epoch, float(lr_pair_match.group(2)))
            continue

        table_row_match = re.search(r"^(.+?)\|\s*([^|]+)\|\s*([^|]+)$", line)
        if table_row_match and "----" not in line and not line.upper().startswith("METRIC"):
            metric_name = table_row_match.group(1).strip()
            left_token = table_row_match.group(2).strip()
            right_token = table_row_match.group(3).strip()

            left_parsed = parse_number(left_token)
            right_parsed = parse_number(right_token)
            if left_parsed and right_parsed:
                left_val, left_is_pct = left_parsed
                right_val, right_is_pct = right_parsed
                parsed.paired[metric_name] = (left_val, right_val, left_is_pct or right_is_pct)

    # Alias validation trajectory metrics to generic names when only validation labels are present.
    if "ADE" not in parsed.series and "Val ADE" in parsed.series:
        parsed.series["ADE"] = list(parsed.series["Val ADE"])
    if "FDE" not in parsed.series and "Val FDE" in parsed.series:
        parsed.series["FDE"] = list(parsed.series["Val FDE"])

    return parsed


def setup_theme() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "#000000",
            "axes.facecolor": "#000000",
            "savefig.facecolor": "#000000",
            "text.color": "#FFFFFF",
            "axes.labelcolor": "#FFFFFF",
            "xtick.color": "#FFFFFF",
            "ytick.color": "#FFFFFF",
            "axes.edgecolor": "#FFFFFF",
            "font.family": "Calibri",
            "font.size": 20,
        }
    )


def create_series_page(metric_name: str, points: List[Tuple[int, float]], source_name: str, out_path: Path) -> None:
    points = sorted(points, key=lambda x: x[0])
    x_vals = [p[0] for p in points]
    y_vals = [p[1] for p in points]

    fig, ax = plt.subplots(figsize=(13.333, 7.5), dpi=150)
    ax.plot(x_vals, y_vals, color="#FFFFFF", linewidth=3.0, marker="o", markersize=5)

    ax.set_title(metric_name, fontsize=42, weight="bold", pad=20)
    ax.set_xlabel("Epoch / Step", fontsize=24, labelpad=12)
    ax.set_ylabel(metric_name, fontsize=24, labelpad=12)
    ax.grid(True, linestyle="--", linewidth=0.8, color="#5E5E5E", alpha=0.6)

    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    min_v = min(y_vals)
    max_v = max(y_vals)
    last_v = y_vals[-1]

    summary = f"Min: {min_v:.4f}    Max: {max_v:.4f}    Last: {last_v:.4f}"
    fig.text(0.5, 0.05, summary, ha="center", va="center", fontsize=22, color="#FFFFFF")
    fig.text(0.01, 0.01, f"Source: {source_name}", ha="left", va="bottom", fontsize=12, color="#D8D8D8")

    fig.tight_layout(rect=(0.02, 0.08, 0.98, 0.96))
    fig.savefig(out_path)
    plt.close(fig)


def create_paired_page(
    metric_name: str,
    left_value: float,
    right_value: float,
    is_percent: bool,
    left_label: str,
    right_label: str,
    source_name: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(13.333, 7.5), dpi=150)

    labels = [left_label, right_label]
    vals = [left_value, right_value]
    bars = ax.bar(labels, vals, color=["#B8B8B8", "#FFFFFF"], width=0.55)

    suffix = "%" if is_percent else ""
    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.2f}{suffix}",
            ha="center",
            va="bottom",
            fontsize=20,
            color="#FFFFFF",
        )

    ax.set_title(metric_name, fontsize=42, weight="bold", pad=20)
    ax.set_ylabel(metric_name + (" (%)" if is_percent else ""), fontsize=24)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.8, color="#5E5E5E", alpha=0.6)

    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    fig.text(0.01, 0.01, f"Source: {source_name}", ha="left", va="bottom", fontsize=12, color="#D8D8D8")
    fig.tight_layout(rect=(0.02, 0.06, 0.98, 0.96))
    fig.savefig(out_path)
    plt.close(fig)


def create_unavailable_page(metric_name: str, source_name: str, out_path: Path) -> None:
    fig = plt.figure(figsize=(13.333, 7.5), dpi=150)
    fig.patch.set_facecolor("#000000")

    fig.text(0.5, 0.62, metric_name, ha="center", va="center", fontsize=48, color="#FFFFFF", weight="bold")
    fig.text(0.5, 0.44, "Not available in selected log", ha="center", va="center", fontsize=26, color="#FFFFFF")
    fig.text(0.01, 0.01, f"Source: {source_name}", ha="left", va="bottom", fontsize=12, color="#D8D8D8")

    fig.savefig(out_path)
    plt.close(fig)


def pick_default_log() -> Path:
    candidates = list(LOG_DIR.glob("phase2_fusion_train_*.txt")) + list(LOG_DIR.glob("train_log_*.txt"))
    if not candidates:
        candidates = list(LOG_DIR.glob("*.txt"))
    if not candidates:
        raise FileNotFoundError("No .txt logs found in log folder.")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate one PPT-ready page per metric from training/evaluation logs.")
    parser.add_argument("--log-file", type=str, default="", help="Path to source log file. Default: latest train/eval log.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Directory to save generated metric pages. Default: log/ppt_metric_pages/<log_name>/",
    )
    parser.add_argument(
        "--requested",
        type=str,
        default="ADE,FDE,Val ADE,Val FDE,Train Loss,MSE,F1,Precision,Recall,Accuracy",
        help="Comma-separated metrics to include as missing pages if absent.",
    )
    parser.add_argument(
        "--include-missing-pages",
        action="store_true",
        help="Create a separate page for requested metrics that are not found in the log.",
    )
    args = parser.parse_args()

    setup_theme()

    log_path = Path(args.log_file) if args.log_file else pick_default_log()
    if not log_path.is_absolute():
        log_path = REPO_ROOT / log_path
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    output_dir = Path(args.output_dir) if args.output_dir else (LOG_DIR / "ppt_metric_pages" / log_path.stem)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Keep output deterministic for presentation export by removing old pages from previous runs.
    for old_png in output_dir.glob("*.png"):
        old_png.unlink()

    parsed = parse_metrics_from_log(log_path)
    generated: List[str] = []

    for metric_name in sorted(parsed.series.keys()):
        filename = f"{sanitize_filename(metric_name)}.png"
        out_path = output_dir / filename
        create_series_page(metric_name, parsed.series[metric_name], log_path.name, out_path)
        generated.append(metric_name)

    left_label, right_label = parsed.paired_labels
    for metric_name in sorted(parsed.paired.keys()):
        left_value, right_value, is_percent = parsed.paired[metric_name]
        filename = f"{sanitize_filename(metric_name)}_comparison.png"
        out_path = output_dir / filename
        create_paired_page(
            metric_name=metric_name,
            left_value=left_value,
            right_value=right_value,
            is_percent=is_percent,
            left_label=left_label,
            right_label=right_label,
            source_name=log_path.name,
            out_path=out_path,
        )
        generated.append(metric_name)

    requested = [m.strip() for m in args.requested.split(",") if m.strip()]
    generated_canonical = {canonical_metric(m) for m in generated}
    missing = [m for m in requested if canonical_metric(m) not in generated_canonical]

    if args.include_missing_pages:
        for metric_name in missing:
            filename = f"{sanitize_filename(metric_name)}_not_available.png"
            out_path = output_dir / filename
            create_unavailable_page(metric_name, log_path.name, out_path)

    manifest_path = output_dir / "metrics_manifest.txt"
    manifest_lines: List[str] = [
        f"Source log: {log_path}",
        f"Output directory: {output_dir}",
        "",
        "Detected metrics:",
    ]
    for m in sorted(set(generated)):
        manifest_lines.append(f"- {m}")

    manifest_lines.append("")
    manifest_lines.append("Requested but missing:")
    if missing:
        for m in missing:
            manifest_lines.append(f"- {m}")
    else:
        manifest_lines.append("- None")

    manifest_path.write_text("\n".join(manifest_lines), encoding="utf-8")

    print(f"Generated {len(list(output_dir.glob('*.png')))} metric pages in: {output_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
