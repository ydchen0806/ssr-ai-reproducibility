#!/usr/bin/env python3
"""Generate meeting-revision evidence figures from aggregated CSV tables only.

The default mode is a submission gate: every prespecified contrast and its
locked paired-seed count must be present.  ``--staged``/``--allow-incomplete``
is intended only for internal review and adds an explicit incomplete-evidence
banner to both figures.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TABLE_FILES = {
    "matched_kd": "kd_matched_cl.csv",
    "segmentation": "cub_segmentation.csv",
    "editing": "editing_ablation.csv",
    "mapping": "editing_mapping.csv",
}
REQUIRED_COLUMNS = {
    "dataset",
    "contrast",
    "mapping",
    "metric",
    "n",
    "control_mean",
    "control_sd",
    "treatment_mean",
    "treatment_sd",
    "difference_mean",
    "difference_sd",
    "ci95_low",
    "ci95_high",
    "favorable_pairs",
}
NUMERIC_COLUMNS = {
    "control_mean",
    "control_sd",
    "treatment_mean",
    "treatment_sd",
    "difference_mean",
    "difference_sd",
    "ci95_low",
    "ci95_high",
}

DATASET_LABELS = {
    "split_cifar100": "CIFAR-100",
    "split_tiny_imagenet": "Tiny ImageNet",
    "cub200_masks": "CUB-200 masks",
    "zsre": "ZsRE",
    "cf": "WikiCounterFact",
    "recent": "WikiRecent",
}
CONTRAST_LABELS = {
    "kd_ewc_minus_kd": "KD + EWC",
    "kd_mas_minus_kd": "KD + MAS",
    "kd_si_minus_kd": "KD + SI",
    "kd_ssr_minus_kd": "KD + SSR",
    "ssr_kd_minus_kd": "SSR + KD - KD",
    "ssr_only_minus_plain": "SSR-only - Plain",
    "full_minus_stabilized": "Full - Stabilized",
    "full_minus_plain": "Full - Plain",
}
METRIC_LABELS = {
    "avg_accuracy": "Average accuracy",
    "avg_forgetting": "Forgetting reduction",
    "mean_iou": "mIoU",
    "mean_dice": "Dice",
    "avg_forgetting_iou": "IoU forgetting",
    "efficacy": "Efficacy",
    "locality": "Locality",
}
MAPPING_LABELS = {
    "cosine": "cos",
    "projective": "proj",
    "projective_vs_cosine": "proj - cos",
}

COLORS = {
    "ink": "#171A1F",
    "grid": "#D8DCE1",
    "zero": "#717780",
    "ssr": "#087F5B",
    "blue": "#2F6690",
    "gold": "#C38B2C",
    "rose": "#A65353",
    "violet": "#6B5B95",
    "control": "#A7ADB5",
    "staged": "#B5473C",
}


class EvidenceError(ValueError):
    """Raised when a publication figure would rely on incomplete evidence."""


@dataclass(frozen=True)
class RequiredRow:
    table: str
    dataset: str
    contrast: str
    mapping: str
    metric: str
    expected_n: int
    recipe: str = ""

    def describe(self) -> str:
        suffix = f", recipe={self.recipe}" if self.recipe else ""
        return (
            f"{TABLE_FILES[self.table]}: dataset={self.dataset}, "
            f"contrast={self.contrast}, mapping={self.mapping}, "
            f"metric={self.metric}{suffix}, required n={self.expected_n}"
        )


@dataclass
class EvidenceBundle:
    rows: dict[str, list[dict[str, Any]]]
    selected: dict[RequiredRow, dict[str, Any]]
    issues: list[str]
    source_hashes: dict[str, str]
    expected_counts: dict[str, int]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_table(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - fields
        if missing:
            raise EvidenceError(f"{path} is missing columns: {sorted(missing)}")
        rows = []
        for line_number, raw in enumerate(reader, start=2):
            if not any(str(value).strip() for value in raw.values()):
                continue
            row: dict[str, Any] = dict(raw)
            try:
                row["n"] = int(raw["n"])
                row["favorable_pairs"] = int(raw["favorable_pairs"])
                for column in NUMERIC_COLUMNS:
                    row[column] = float(raw[column])
            except (TypeError, ValueError) as error:
                raise EvidenceError(
                    f"{path}:{line_number} has a non-numeric summary field: {error}"
                ) from error
            if row["n"] < 0 or not all(math.isfinite(row[name]) for name in NUMERIC_COLUMNS):
                raise EvidenceError(f"{path}:{line_number} has invalid numeric values")
            rows.append(row)
    return rows


def required_specs(
    *,
    expected_cl_n: int = 5,
    expected_seg_n: int = 10,
    expected_editing_n: int = 10,
) -> tuple[list[RequiredRow], list[RequiredRow]]:
    """Return the prespecified Figure 4 and Figure 5 evidence keys."""
    figure4: list[RequiredRow] = []
    for dataset in ("split_cifar100", "split_tiny_imagenet"):
        for contrast in (
            "kd_ewc_minus_kd",
            "kd_mas_minus_kd",
            "kd_si_minus_kd",
            "kd_ssr_minus_kd",
        ):
            mapping = "cosine" if contrast == "kd_ssr_minus_kd" else "none"
            for metric in ("avg_accuracy", "avg_forgetting"):
                figure4.append(
                    RequiredRow(
                        "matched_kd", dataset, contrast, mapping, metric, expected_cl_n
                    )
                )
    for metric in ("mean_iou", "mean_dice", "avg_forgetting_iou"):
        figure4.append(
            RequiredRow(
                "segmentation",
                "cub200_masks",
                "ssr_kd_minus_kd",
                "cosine",
                metric,
                expected_seg_n,
            )
        )

    figure5: list[RequiredRow] = []
    for dataset in ("zsre", "cf", "recent"):
        for mapping in ("cosine", "projective"):
            for metric in ("efficacy", "locality"):
                for contrast in ("ssr_only_minus_plain", "full_minus_stabilized"):
                    figure5.append(
                        RequiredRow(
                            "editing",
                            dataset,
                            contrast,
                            mapping,
                            metric,
                            expected_editing_n,
                        )
                    )
                figure5.append(
                    RequiredRow(
                        "editing",
                        dataset,
                        "full_minus_plain",
                        mapping,
                        metric,
                        expected_editing_n,
                    )
                )
        for recipe in ("ssr_only", "full"):
            for metric in ("efficacy", "locality"):
                figure5.append(
                    RequiredRow(
                        "mapping",
                        dataset,
                        "projective_minus_cosine",
                        "projective_vs_cosine",
                        metric,
                        expected_editing_n,
                        recipe=recipe,
                    )
                )
    return figure4, figure5


def _matches(row: dict[str, Any], spec: RequiredRow) -> bool:
    return (
        row.get("dataset") == spec.dataset
        and row.get("contrast") == spec.contrast
        and row.get("mapping") == spec.mapping
        and row.get("metric") == spec.metric
        and (not spec.recipe or row.get("recipe") == spec.recipe)
    )


def load_evidence(
    table_root: Path,
    *,
    allow_incomplete: bool = False,
    expected_cl_n: int = 5,
    expected_seg_n: int = 10,
    expected_editing_n: int = 10,
) -> EvidenceBundle:
    """Load generated CSVs and enforce the submission evidence matrix."""
    rows: dict[str, list[dict[str, Any]]] = {}
    issues: list[str] = []
    source_hashes: dict[str, str] = {}
    for key, filename in TABLE_FILES.items():
        path = table_root / filename
        if not path.is_file():
            rows[key] = []
            issues.append(f"missing required generated table: {path}")
            continue
        rows[key] = _read_table(path)
        source_hashes[filename] = _sha256(path)

    figure4, figure5 = required_specs(
        expected_cl_n=expected_cl_n,
        expected_seg_n=expected_seg_n,
        expected_editing_n=expected_editing_n,
    )
    selected: dict[RequiredRow, dict[str, Any]] = {}
    for spec in (*figure4, *figure5):
        candidates = [row for row in rows[spec.table] if _matches(row, spec)]
        if not candidates:
            issues.append(f"missing contrast: {spec.describe()}")
            continue
        if len(candidates) > 1:
            raise EvidenceError(f"ambiguous duplicate contrast: {spec.describe()}")
        row = candidates[0]
        selected[spec] = row
        if row["n"] < spec.expected_n:
            issues.append(
                f"incomplete contrast: {spec.describe()}, observed n={row['n']}"
            )

    if issues and not allow_incomplete:
        preview = "\n  - ".join(issues[:20])
        remainder = len(issues) - min(len(issues), 20)
        tail = f"\n  - ... and {remainder} more" if remainder else ""
        raise EvidenceError(
            "Meeting-revision figures are blocked by incomplete generated evidence:\n"
            f"  - {preview}{tail}\n"
            "Use --staged/--allow-incomplete only for an explicitly marked internal draft."
        )
    return EvidenceBundle(
        rows=rows,
        selected=selected,
        issues=issues,
        source_hashes=source_hashes,
        expected_counts={
            "matched_kd": expected_cl_n,
            "segmentation": expected_seg_n,
            "editing": expected_editing_n,
        },
    )


def _set_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 10,
            "axes.labelsize": 8.5,
            "axes.linewidth": 0.7,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.3,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "text.color": COLORS["ink"],
            "axes.labelcolor": COLORS["ink"],
            "axes.edgecolor": COLORS["ink"],
            "xtick.color": COLORS["ink"],
            "ytick.color": COLORS["ink"],
        }
    )


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.14,
        1.05,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="bottom",
        color=COLORS["ink"],
    )


def _clean_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.55, alpha=0.7)
    ax.set_axisbelow(True)


def _missing_panel(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, loc="left", fontweight="bold")
    ax.text(
        0.5,
        0.5,
        "Evidence pending\n(staged draft)",
        transform=ax.transAxes,
        ha="center",
        va="center",
        color=COLORS["staged"],
        fontweight="bold",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _forest(
    ax: plt.Axes,
    entries: list[tuple[str, dict[str, Any], str, str]],
    *,
    title: str,
    xlabel: str = "Favorable paired change (percentage points)",
) -> None:
    if not entries:
        _missing_panel(ax, title)
        return
    y = np.arange(len(entries))[::-1]
    ax.axvline(0.0, color=COLORS["zero"], linewidth=0.8, linestyle="--", zorder=0)
    for position, (label, row, color, marker) in zip(y, entries):
        mean = row["difference_mean"]
        low = row["ci95_low"]
        high = row["ci95_high"]
        ax.plot([low, high], [position, position], color=color, linewidth=1.35, zorder=2)
        ax.scatter(
            [mean],
            [position],
            color=color,
            marker=marker,
            s=29,
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
        )
    ax.set_yticks(y, [entry[0] for entry in entries])
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.margins(x=0.12, y=0.08)
    _clean_axis(ax)


def _status_banner(fig: plt.Figure, issues: list[str]) -> None:
    if not issues:
        return
    fig.text(
        0.5,
        1.015,
        f"STAGED / INCOMPLETE EVIDENCE - {len(issues)} required entries unresolved",
        ha="center",
        va="top",
        color="white",
        fontsize=8.5,
        fontweight="bold",
        bbox={"boxstyle": "square,pad=0.28", "facecolor": COLORS["staged"], "edgecolor": "none"},
    )


def _select_rows(
    bundle: EvidenceBundle,
    specs: Iterable[RequiredRow],
) -> list[tuple[RequiredRow, dict[str, Any]]]:
    return [(spec, bundle.selected[spec]) for spec in specs if spec in bundle.selected]


def build_figure4(bundle: EvidenceBundle) -> plt.Figure:
    """Build the matched-KD and segmentation evidence summary."""
    figure4_specs, _ = required_specs(
        expected_cl_n=bundle.expected_counts["matched_kd"],
        expected_seg_n=bundle.expected_counts["segmentation"],
        expected_editing_n=bundle.expected_counts["editing"],
    )
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.2), constrained_layout=True)
    contrast_colors = {
        "kd_ewc_minus_kd": COLORS["gold"],
        "kd_mas_minus_kd": COLORS["blue"],
        "kd_si_minus_kd": COLORS["rose"],
        "kd_ssr_minus_kd": COLORS["ssr"],
    }
    markers = {
        "kd_ewc_minus_kd": "o",
        "kd_mas_minus_kd": "s",
        "kd_si_minus_kd": "^",
        "kd_ssr_minus_kd": "D",
    }
    for panel_index, metric in enumerate(("avg_accuracy", "avg_forgetting")):
        specs = [
            spec
            for spec in figure4_specs
            if spec.table == "matched_kd" and spec.metric == metric
        ]
        entries = []
        for spec, row in _select_rows(bundle, specs):
            label = (
                f"{DATASET_LABELS[spec.dataset]} | "
                f"{CONTRAST_LABELS[spec.contrast]}"
            )
            entries.append(
                (label, row, contrast_colors[spec.contrast], markers[spec.contrast])
            )
        _forest(
            axes[0, panel_index],
            entries,
            title=(
                "Vision classification | fixed-KD accuracy"
                if metric == "avg_accuracy"
                else "Vision classification | fixed-KD forgetting"
            ),
        )
        _panel_label(axes[0, panel_index], chr(ord("a") + panel_index))

    segmentation_specs = [spec for spec in figure4_specs if spec.table == "segmentation"]
    segmentation = _select_rows(bundle, segmentation_specs)
    endpoint_ax = axes[1, 0]
    if segmentation:
        positions = np.arange(len(segmentation))
        width = 0.34
        control = [row["control_mean"] for _, row in segmentation]
        treatment = [row["treatment_mean"] for _, row in segmentation]
        control_sd = [row["control_sd"] for _, row in segmentation]
        treatment_sd = [row["treatment_sd"] for _, row in segmentation]
        endpoint_ax.bar(
            positions - width / 2,
            control,
            width,
            yerr=control_sd,
            capsize=2,
            color=COLORS["control"],
            label="KD",
            linewidth=0,
        )
        endpoint_ax.bar(
            positions + width / 2,
            treatment,
            width,
            yerr=treatment_sd,
            capsize=2,
            color=COLORS["ssr"],
            label="SSR + KD",
            linewidth=0,
        )
        endpoint_ax.set_xticks(
            positions,
            [METRIC_LABELS[spec.metric] for spec, _ in segmentation],
        )
        endpoint_ax.set_ylabel("Endpoint (%)")
        endpoint_ax.set_title(
            "Vision segmentation | matched endpoints", loc="left", fontweight="bold"
        )
        endpoint_ax.legend(frameon=False, ncol=2, loc="upper right")
        _clean_axis(endpoint_ax)
    else:
        _missing_panel(endpoint_ax, "Vision segmentation | matched endpoints")
    _panel_label(endpoint_ax, "c")

    gain_entries = [
        (
            (
                "IoU forgetting reduction"
                if spec.metric == "avg_forgetting_iou"
                else METRIC_LABELS[spec.metric]
            ),
            row,
            COLORS["ssr"],
            "D",
        )
        for spec, row in segmentation
    ]
    _forest(
        axes[1, 1],
        gain_entries,
        title="Vision segmentation | direct SSR attribution",
    )
    _panel_label(axes[1, 1], "d")
    fig.suptitle(
        "Figure 4 evidence summary: matched continual learning and structured output",
        fontsize=12,
        fontweight="bold",
        y=1.06 if bundle.issues else 1.01,
    )
    fig.text(
        0.5,
        -0.025,
        "Points show paired mean changes; horizontal intervals are paired 95% t intervals.",
        ha="center",
        va="bottom",
        fontsize=7.4,
        color=COLORS["ink"],
    )
    _status_banner(fig, bundle.issues)
    return fig


def _editing_entries(
    bundle: EvidenceBundle,
    specs: list[RequiredRow],
) -> list[tuple[str, dict[str, Any], str, str]]:
    entries = []
    for spec, row in _select_rows(bundle, specs):
        metric = METRIC_LABELS[spec.metric]
        mapping = MAPPING_LABELS[spec.mapping]
        label = f"{DATASET_LABELS[spec.dataset]} | {mapping} | {metric}"
        color = COLORS["blue"] if spec.metric == "efficacy" else COLORS["ssr"]
        marker = "o" if spec.metric == "efficacy" else "D"
        entries.append((label, row, color, marker))
    return entries


def build_figure5(bundle: EvidenceBundle) -> plt.Figure:
    """Build editing direct-attribution, mapping, and transfer summaries."""
    _, figure5_specs = required_specs(
        expected_cl_n=bundle.expected_counts["matched_kd"],
        expected_seg_n=bundle.expected_counts["segmentation"],
        expected_editing_n=bundle.expected_counts["editing"],
    )
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 9.8), constrained_layout=True)
    direct_specs = {
        contrast: [
            spec
            for spec in figure5_specs
            if spec.table == "editing" and spec.contrast == contrast
        ]
        for contrast in ("ssr_only_minus_plain", "full_minus_stabilized")
    }
    _forest(
        axes[0, 0],
        _editing_entries(bundle, direct_specs["ssr_only_minus_plain"]),
        title="Direct SSR attribution | SSR-only - Plain",
    )
    _panel_label(axes[0, 0], "a")
    _forest(
        axes[0, 1],
        _editing_entries(bundle, direct_specs["full_minus_stabilized"]),
        title="Direct SSR attribution | Full - Stabilized",
    )
    _panel_label(axes[0, 1], "b")

    mapping_specs = [spec for spec in figure5_specs if spec.table == "mapping"]
    mapping_entries = []
    for spec, row in _select_rows(bundle, mapping_specs):
        label = (
            f"{DATASET_LABELS[spec.dataset]} | {spec.recipe} | "
            f"{METRIC_LABELS[spec.metric]}"
        )
        color = COLORS["violet"] if spec.recipe == "full" else COLORS["ssr"]
        marker = "s" if spec.metric == "efficacy" else "D"
        mapping_entries.append((label, row, color, marker))
    _forest(
        axes[1, 0],
        mapping_entries,
        title="Distance mapping sensitivity | Projective - Cosine",
    )
    _panel_label(axes[1, 0], "c")

    transfer_specs = [
        spec
        for spec in figure5_specs
        if spec.table == "editing" and spec.contrast == "full_minus_plain"
    ]
    transfer_entries = []
    for label, row, _, marker in _editing_entries(bundle, transfer_specs):
        transfer_entries.append((label, row, COLORS["gold"], marker))
    _forest(
        axes[1, 1],
        transfer_entries,
        title="Recipe transfer | Full - Plain (not isolated SSR)",
    )
    _panel_label(axes[1, 1], "d")
    fig.suptitle(
        "Figure 5 evidence summary: sequential editing objectives and distance mappings",
        fontsize=12,
        fontweight="bold",
        y=1.06 if bundle.issues else 1.01,
    )
    fig.text(
        0.5,
        -0.035,
        "Direct attribution holds the non-SSR objective fixed; recipe transfer changes multiple components.",
        ha="center",
        va="bottom",
        fontsize=7.4,
        color=COLORS["ink"],
    )
    _status_banner(fig, bundle.issues)
    return fig


def generate_figures(
    table_root: Path,
    output_root: Path,
    *,
    allow_incomplete: bool = False,
    expected_cl_n: int = 5,
    expected_seg_n: int = 10,
    expected_editing_n: int = 10,
    dpi: int = 300,
) -> dict[str, Any]:
    _set_publication_style()
    bundle = load_evidence(
        table_root,
        allow_incomplete=allow_incomplete,
        expected_cl_n=expected_cl_n,
        expected_seg_n=expected_seg_n,
        expected_editing_n=expected_editing_n,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    figures = {
        "figure4_evidence_summary": build_figure4(bundle),
        "figure5_evidence_summary": build_figure5(bundle),
    }
    output_paths: dict[str, list[str]] = {}
    for stem, figure in figures.items():
        pdf = output_root / f"{stem}.pdf"
        png = output_root / f"{stem}.png"
        figure.savefig(pdf, bbox_inches="tight", pad_inches=0.04)
        figure.savefig(png, dpi=dpi, bbox_inches="tight", pad_inches=0.04)
        plt.close(figure)
        output_paths[stem] = [str(pdf), str(png)]

    report = {
        "status": "staged_incomplete" if bundle.issues else "complete",
        "tables": str(table_root),
        "source_sha256": bundle.source_hashes,
        "expected_paired_counts": bundle.expected_counts,
        "issues": bundle.issues,
        "claim_labels": {
            "ssr_only_minus_plain": "direct attribution",
            "full_minus_stabilized": "direct attribution",
            "ssr_kd_minus_kd": "direct attribution",
            "full_minus_plain": "recipe transfer",
        },
        "outputs": output_paths,
    }
    (output_root / "figure_evidence_provenance.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-incomplete",
        "--staged",
        dest="allow_incomplete",
        action="store_true",
        help="Generate a visibly marked internal draft when required evidence is incomplete.",
    )
    parser.add_argument("--expected-cl-n", type=int, default=5)
    parser.add_argument("--expected-seg-n", type=int, default=10)
    parser.add_argument("--expected-editing-n", type=int, default=10)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    for name in ("expected_cl_n", "expected_seg_n", "expected_editing_n", "dpi"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    try:
        report = generate_figures(
            args.tables.resolve(),
            args.output.resolve(),
            allow_incomplete=args.allow_incomplete,
            expected_cl_n=args.expected_cl_n,
            expected_seg_n=args.expected_seg_n,
            expected_editing_n=args.expected_editing_n,
            dpi=args.dpi,
        )
    except EvidenceError as error:
        parser.exit(2, f"error: {error}\n")
    print(
        f"Generated Figure 4/5 evidence summaries: status={report['status']}, "
        f"output={args.output}"
    )


if __name__ == "__main__":
    main()
