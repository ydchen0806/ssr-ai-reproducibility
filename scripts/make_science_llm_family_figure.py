#!/usr/bin/env python3
"""Generate a composite figure for open-family LLM transfer results and protocol coverage."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path
from typing import Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager, patches
from matplotlib.lines import Line2D

SCI_FIG_DIR = Path(os.environ.get("SSR_FIGURE_DIR", PROJECT_ROOT / "figures"))
ICLR_FIG_DIR = Path(os.environ.get("SSR_LEGACY_FIGURE_DIR", PROJECT_ROOT / "figures" / "legacy"))

DEFAULT_CONCEPT_TITLE = "(a) SSR-FT mechanism and design logic"
DEFAULT_FIGURE_TITLE = "Open-family LLM transfer results with protocol coverage"
DEFAULT_FIGURE_NAME = "fig_llm_family_transfer"

TARGET_MODELS = [
    ("GPT-2 XL", ("gpt-2-xl", "gpt2-xl", "gpt2_xl", "gpt2xl")),
    ("Qwen/Qwen2.5-0.5B", ("qwen/qwen2.5-0.5b", "qwen2.5-0.5b", "qwen2.5_0.5b")),
    ("Qwen/Qwen2.5-1.5B", ("qwen/qwen2.5-1.5b", "qwen2.5-1.5b", "qwen2.5_1.5b")),
    ("meta-llama/Meta-Llama-3-8B", ("meta-llama/meta-llama-3-8b", "nousresearch/meta-llama-3-8b", "meta-llama-3-8b", "llama-3-8b", "meta-llama3-8b")),
    ("Qwen/Qwen2.5-VL-3B-Instruct", ("qwen/qwen2.5-vl-3b-instruct", "qwen/qwen2.5-vl-3b", "qwen2.5-vl-3b-instruct", "qwen2.5-vl-3b", "qwen2.5-vl")),
]

TARGET_DATASETS = ("zsre", "cf", "recent")
TARGET_EDIT_COUNTS = (50, 100)

TARGET_METHODS = [
    ("biocs", "SSR", ("biocs",)),
    ("ft", "Fine-tune", ("ft",)),
]
TARGET_METHOD_KEYS = [key for key, _, _ in TARGET_METHODS]
METHOD_LABELS = {key: label for key, label, _ in TARGET_METHODS}
METHOD_ALIASES = {key: aliases for key, _, aliases in TARGET_METHODS}

FOCUS_MODELS = [
    "meta-llama/Meta-Llama-3-8B",
    "Qwen/Qwen2.5-VL-3B-Instruct",
]
FOCUS_MODEL_LABELS = {
    "meta-llama/Meta-Llama-3-8B": "Llama-3-8B",
    "Qwen/Qwen2.5-VL-3B-Instruct": "Qwen2.5-VL-3B",
}
FOCUS_SLICES = [
    ("zsre", 50, "ZsRE\n50 edits"),
    ("cf", 100, "CF\n100 edits"),
    ("recent", 100, "Recent\n100 edits"),
]

GRID_X = [f"{dataset}-{edits}" for dataset in TARGET_DATASETS for edits in TARGET_EDIT_COUNTS]
MODEL_LABELS = [name for name, _ in TARGET_MODELS]

PALETTE = {
    "ink": "#2D3436",
    "paper": "#FFFFFF",
    "paper_muted": "#F4F6F8",
    "paper_soft": "#F8FAFB",
    "completed_face": "#FFFFFF",
    "completed_edge": "#2F4F56",
    "pending_face": "#F9FAFC",
    "pending_edge": "#6C727A",
    "skipped_face": "#F2ECE3",
    "skipped_edge": "#8C7D64",
    "failed_face": "#F7F0EF",
    "failed_edge": "#A15858",
    "placeholder_face": "#FFF2E0",
    "placeholder_edge": "#926737",
    "missing_face": "#F7F9FB",
    "missing_edge": "#CFD5DE",
    "grid": "#D7DCE2",
    "line": "#2D3436",
}

STATUS_STYLES: dict[str, dict[str, str]] = {
    "completed": {
        "face": PALETTE["completed_face"],
        "edge": PALETTE["completed_edge"],
        "hatch": "",
        "linestyle": "-",
        "linewidth": 1.8,
        "marker": "o",
        "marker_face": PALETTE["paper"],
        "marker_edge": PALETTE["completed_edge"],
        "label": "completed",
    },
    "pending": {
        "face": PALETTE["pending_face"],
        "edge": PALETTE["pending_edge"],
        "hatch": "///",
        "linestyle": "--",
        "linewidth": 1.5,
        "marker": "s",
        "marker_face": "#FFFFFF",
        "marker_edge": "#6E7884",
        "label": "pending / running",
    },
    "skipped": {
        "face": PALETTE["skipped_face"],
        "edge": PALETTE["skipped_edge"],
        "hatch": "--",
        "linestyle": "-.",
        "linewidth": 1.5,
        "marker": "D",
        "marker_face": "#F0E3C9",
        "marker_edge": "#8B785D",
        "label": "skipped",
    },
    "failed": {
        "face": PALETTE["failed_face"],
        "edge": PALETTE["failed_edge"],
        "hatch": "xx",
        "linestyle": ":",
        "linewidth": 1.8,
        "marker": "X",
        "marker_face": "#F2D9D7",
        "marker_edge": PALETTE["failed_edge"],
        "label": "failed",
    },
    "placeholder": {
        "face": PALETTE["placeholder_face"],
        "edge": PALETTE["placeholder_edge"],
        "hatch": "++",
        "linestyle": "-",
        "linewidth": 1.6,
        "marker": "P",
        "marker_face": "#FFF4D4",
        "marker_edge": PALETTE["placeholder_edge"],
        "label": "placeholder (vl branch)",
    },
    "missing": {
        "face": PALETTE["missing_face"],
        "edge": PALETTE["missing_edge"],
        "hatch": "..",
        "linestyle": "dashed",
        "linewidth": 1.2,
        "marker": ".",
        "marker_face": PALETTE["paper"],
        "marker_edge": PALETTE["missing_edge"],
        "label": "not launched",
    },
}

PLOT_STYLE = {
    "figure_width": 15.4,
    "figure_height": 6.35,
    "left_panel_ratio": 1.28,
    "right_panel_ratio": 1.42,
    "font_family": [
        "Times New Roman",
        "Times",
        "TeX Gyre Termes",
        "Nimbus Roman",
        "Liberation Serif",
        "STIXGeneral",
        "DejaVu Serif",
        "serif",
    ],
    "title_size": 15,
    "panel_title_size": 13,
    "label_size": 10,
    "body_size": 8.4,
    "legend_size": 7.1,
    "tiny_size": 6.5,
    "panel_divider": "#5D6670",
}

CONCEPT_NODE_STYLES = [
    ("#EAEFF4", "s", "//"),
    ("#EAF2EE", "^", "xx"),
    ("#F5EEE4", "v", ".."),
    ("#EAF1E6", "P", "--"),
]

STATUS_LEGEND_ORDER = ("completed", "placeholder", "skipped", "failed", "pending", "missing")

METHOD_STYLES = {
    "biocs": {
        "row_face": "#FFFFFF",
        "row_marker": "B",
        "line": "-",
        "label_offset": -0.88,
        "label_ha": "right",
    },
    "ft": {
        "row_face": "#F6F8FC",
        "row_marker": "F",
        "line": ":",
        "label_offset": -0.88,
        "label_ha": "right",
    },
}

STATUS_PRIORITY = {"missing": 0, "pending": 1, "failed": 2, "skipped": 3, "placeholder": 4, "completed": 5}
RUN_SOURCE_PRIORITY = {"default": 0, "text": 1, "vl": 2}
RUN_SOURCE_LABEL = {"default": "D", "text": "T", "vl": "V"}
BLOCKER_LABELS = {
    "missing_local_model": "missing local checkpoint",
    "placeholder_multimodal_model": "VL placeholder",
    "model_download_failed": "model download failed",
}
RunStatus = tuple[str, float | None, float | None, str, str, str, str]


def _infer_run_source(path: Path) -> str:
    low = path.name.lower()
    if "vl" in low:
        return "vl"
    if "text" in low:
        return "text"
    return "default"


def _normalize_text(text: str) -> str:
    return (text or "").lower().replace("_", "-")


def _match_model(model_name: str) -> str | None:
    low = _normalize_text(model_name)
    for display_name, tokens in TARGET_MODELS:
        for token in tokens:
            if token and token in low:
                return display_name
    return None


def _match_method(method_name: str) -> str | None:
    low = _normalize_text(method_name)
    for method_key, aliases in METHOD_ALIASES.items():
        for alias in aliases:
            if alias == low:
                return method_key
            if f"-{alias}" in f"-{low}-" or f"_{alias}" in f"_{low}_":
                return method_key
    return None


def _parse_status(raw: str) -> str:
    status = (raw or "").strip().lower()
    if not status:
        return "missing"
    if status in {"ok", "finished", "success", "succeeded"}:
        return "completed"
    if status in {"queued", "running", "pending", "prepared"}:
        return "pending"
    if status in {"skipped", "skip"}:
        return "skipped"
    if status in {"failed", "error", "timeout", "crash", "killed", "crashed"}:
        return "failed"
    return "missing"


def _is_placeholder_skip(skip: str, reason: str, status: str) -> bool:
    merged = f"{skip} {reason} {status}".lower()
    return "placeholder" in merged or "multimodal" in merged


def _safe_int(value: str) -> int | None:
    try:
        return int(float(value))
    except Exception:
        return None


def _safe_float(value: str) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [{k: (v or "") for k, v in row.items()} for row in csv.DictReader(handle, delimiter="\t")]


def _resolve_font_family() -> str:
    available = {entry.name for entry in font_manager.fontManager.ttflist}
    for candidate in PLOT_STYLE["font_family"]:
        if candidate == "serif":
            continue
        if candidate in available:
            return candidate
    return "serif"


def _normalize_blocker_reason(reason: str) -> str:
    key = (reason or "").strip().lower()
    if not key:
        return ""
    return BLOCKER_LABELS.get(key, key.replace("_", " "))


def _short_model_name(model_name: str) -> str:
    if "/" in model_name:
        return model_name.split("/")[-1]
    return model_name


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _list_run_dirs(results_root: Path) -> list[Path]:
    """
    Resolve a root as either:
      - a concrete run directory (contains manifest.tsv), or
      - a parent directory that contains multiple run directories.
    """
    if not results_root.exists():
        return []
    if (results_root / "manifest.tsv").exists():
        return [results_root]
    return [p for p in sorted(results_root.glob("*")) if p.is_dir()]


def collect_run_statuses(results_roots: list[Path]) -> dict[tuple[str, str, str, int], RunStatus]:
    """
    Returns a dict keyed by (method, model_name, dataset, edits):
      (status, efficacy, locality, tag, reason, log_path, run_source)
    """
    state: dict[tuple[str, str, str, int], RunStatus] = {}
    if not results_roots:
        return state

    for results_root in results_roots:
        for run_dir in _list_run_dirs(results_root):
            if not run_dir.exists():
                continue

            manifest_rows = _read_tsv(run_dir / "manifest.tsv")
            if not manifest_rows:
                continue

            summary_rows = _read_csv(run_dir / "summary.csv")
            summary_by_tag = {row.get("tag", ""): row for row in summary_rows if row.get("tag")}
            manifest_by_tag = {row.get("tag", ""): row for row in manifest_rows if row.get("tag")}
            status_dir = run_dir / "status"
            run_source = _infer_run_source(run_dir)

            def classify_status(row: Mapping[str, str], fallback: str) -> str:
                method = _match_method(row.get("method", "") or "")
                if method not in TARGET_METHOD_KEYS:
                    return "missing"
                skip = (row.get("skip", "") or "").strip().lower()
                skip_reason = (row.get("skip_reason", "") or "").strip()
                status_raw = (row.get("status", fallback) or "").strip()
                status = _parse_status(status_raw)
                status_file = status_dir / f"{row['tag']}.status" if row.get("tag") else None

                if skip in {"1", "true", "yes", "y"}:
                    return "placeholder" if _is_placeholder_skip(skip, skip_reason, status_raw) else "skipped"
                if status == "skipped" and _is_placeholder_skip(skip, skip_reason, status_raw):
                    return "placeholder"
                if status_file is not None and status_file.exists():
                    live_status = _parse_status(status_file.read_text(encoding="utf-8").strip())
                    if live_status != "missing" and (
                        status in {"missing", "pending"} or STATUS_PRIORITY[live_status] >= STATUS_PRIORITY[status]
                    ):
                        status = live_status
                if status == "missing" and skip_reason:
                    status = "skipped"
                return status

            def maybe_update(row: Mapping[str, str], fallback_status: str) -> None:
                method = _match_method(row.get("method", "") or "")
                if method is None:
                    return

                dataset = (row.get("dataset", "") or "").strip().lower()
                if dataset not in TARGET_DATASETS:
                    return

                edits = _safe_int(row.get("n_edits", ""))
                if edits is None or edits not in TARGET_EDIT_COUNTS:
                    return

                model_name = _match_model(
                    row.get("model", "") or row.get("model_hparams", "") or row.get("model_name", "") or ""
                )
                if model_name is None:
                    return

                status = classify_status(row, fallback_status)
                efficacy = _safe_float(row.get("efficacy", ""))
                locality = _safe_float(row.get("locality", ""))
                if efficacy is None or locality is None:
                    output_dir = Path((row.get("output", "") or "").strip())
                    results_json = output_dir / "results.json" if output_dir else Path()
                    if results_json.exists():
                        try:
                            payload = json.loads(results_json.read_text(encoding="utf-8"))
                            efficacy = _safe_float(str(payload.get("efficacy", "")))
                            locality = _safe_float(str(payload.get("locality", "")))
                        except Exception:
                            pass
                if status == "completed" and (efficacy is None or locality is None):
                    status = "failed"

                key = (method, model_name, dataset, edits)
                tag = row.get("tag", "")
                reason = (row.get("skip_reason", "") or row.get("status", "")).strip()
                log = (row.get("log", "") or "").strip()

                prev = state.get(key)
                if prev is None:
                    state[key] = (status, efficacy, locality, tag, reason, log, run_source)
                    return

                prev_status = prev[0]
                prev_source = prev[6]
                if prev_status == "placeholder" and prev_source == "text" and run_source == "vl" and status != "missing":
                    state[key] = (status, efficacy, locality, tag, reason, log, run_source)
                    return
                if STATUS_PRIORITY[status] > STATUS_PRIORITY[prev_status]:
                    state[key] = (status, efficacy, locality, tag, reason, log, run_source)
                    return
                if status == "completed" and prev_status == "completed":
                    prev_loc = prev[2] if prev[2] is not None else -1.0
                    if locality is not None and locality > prev_loc:
                        state[key] = (status, efficacy, locality, tag, reason, log, run_source)
                        return
                if status == prev_status and RUN_SOURCE_PRIORITY.get(run_source, 0) > RUN_SOURCE_PRIORITY.get(prev_source, 0):
                    state[key] = (status, efficacy, locality, tag, reason, log, run_source)

            for row in manifest_rows:
                row_tag = row.get("tag", "")
                if not row_tag:
                    continue
                merged = dict(row)
                merged.update(summary_by_tag.get(row_tag, {}))
                maybe_update(merged, "pending")

            for row in summary_rows:
                row_tag = row.get("tag", "")
                if not row_tag or row_tag in manifest_by_tag:
                    continue
                maybe_update(row, "failed")

    return state


def _resolve_concept_image(path: Path | None) -> Path | None:
    if path is None:
        return None
    if path.exists():
        return path
    return None


def draw_concept_panel(ax: plt.Axes, concept_image: Path | None = None, title: str = DEFAULT_CONCEPT_TITLE) -> None:
    ax.set_title(
        title,
        fontsize=PLOT_STYLE["panel_title_size"],
        fontweight="bold",
        pad=10,
        color=PALETTE["ink"],
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_facecolor(PALETTE["paper_muted"])
    ax.axis("off")

    if concept_image is not None:
        try:
            concept = plt.imread(concept_image).astype(np.float32)
            if concept.max() > 1.5:
                concept = concept / 255.0
            if concept.ndim == 3 and concept.shape[2] == 4:
                alpha = concept[..., 3:4]
                concept = concept[..., :3] * alpha + (1.0 - alpha)
            rgb = concept[..., :3]
            mask = rgb.mean(axis=2) < 0.992
            if mask.any():
                rows = np.where(mask.any(axis=1))[0]
                cols = np.where(mask.any(axis=0))[0]
                pad_r = max(4, int(0.01 * concept.shape[0]))
                pad_c = max(4, int(0.01 * concept.shape[1]))
                r0 = max(0, rows[0] - pad_r)
                r1 = min(concept.shape[0], rows[-1] + pad_r + 1)
                c0 = max(0, cols[0] - pad_c)
                c1 = min(concept.shape[1], cols[-1] + pad_c + 1)
                concept = concept[r0:r1, c0:c1]
            h, w = concept.shape[:2]
            trim_r = max(2, int(0.04 * h))
            trim_c = max(2, int(0.025 * w))
            if h > 2 * trim_r and w > 2 * trim_c:
                concept = concept[trim_r:h - trim_r, trim_c:w - trim_c]
            concept[..., :3] = np.clip((concept[..., :3] - 0.5) * 1.10 + 0.5, 0.0, 1.0)
            ax.add_patch(
                patches.FancyBboxPatch(
                    (0.02, 0.07),
                    0.96,
                    0.86,
                    boxstyle="round,pad=0.012",
                    linewidth=1.55,
                    edgecolor=PALETTE["line"],
                    facecolor=PALETTE["paper"],
                    alpha=1.0,
                    zorder=1,
                )
            )
            ax.imshow(concept, extent=(0.04, 0.96, 0.09, 0.91), aspect="auto", zorder=2)
            return
        except Exception:
            ax.text(
                0.50,
                0.97,
                "Concept image unavailable",
                ha="center",
                va="top",
                fontsize=PLOT_STYLE["tiny_size"],
                color=PALETTE["ink"],
                fontstyle="italic",
            )
    else:
        ax.add_patch(
            patches.FancyBboxPatch(
                (0.04, 0.60),
                0.92,
                0.30,
                boxstyle="round,pad=0.015",
                linewidth=1.35,
                edgecolor=PALETTE["line"],
                facecolor=PALETTE["paper"],
                alpha=0.98,
            )
        )
        concept_cards = [
            (0.08, 0.66, 0.22, 0.16, "#EEF3F8", "//", "Edited fact", "new target relation"),
            (0.39, 0.66, 0.22, 0.16, "#F3EEE5", "xx", "SSR prior", "local topology anchor"),
            (0.70, 0.66, 0.22, 0.16, "#EEF5ED", "..", "Retained neighbors", "reduced collateral drift"),
        ]
        for x, y, w, h, face, hatch, headline, subtitle in concept_cards:
            ax.add_patch(
                patches.FancyBboxPatch(
                    (x, y),
                    w,
                    h,
                    boxstyle="round,pad=0.015",
                    linewidth=1.0,
                    edgecolor=PALETTE["ink"],
                    facecolor=face,
                    hatch=hatch,
                    alpha=0.92,
                )
            )
            ax.text(
                x + w / 2,
                y + h * 0.64,
                headline,
                ha="center",
                va="center",
                fontsize=PLOT_STYLE["body_size"],
                fontweight="bold",
                color=PALETTE["ink"],
            )
            ax.text(
                x + w / 2,
                y + h * 0.28,
                subtitle,
                ha="center",
                va="center",
                fontsize=PLOT_STYLE["tiny_size"],
                color=PALETTE["ink"],
            )
        for x0, x1 in ((0.30, 0.39), (0.61, 0.70)):
            ax.annotate(
                "",
                xy=(x1, 0.74),
                xytext=(x0, 0.74),
                arrowprops=dict(arrowstyle="->", lw=2.1, color=PALETTE["ink"]),
            )
        ax.text(
            0.50,
            0.885,
            "Localized weight edits preserve the requested fact while suppressing off-target drift.",
            ha="center",
            va="center",
            fontsize=PLOT_STYLE["tiny_size"] + 0.1,
            color=PALETTE["ink"],
            fontweight="bold",
        )

    ax.add_patch(
        patches.FancyBboxPatch(
            (0.04, 0.47),
            0.92,
            0.10,
            boxstyle="round,pad=0.02",
            linewidth=1.2,
            edgecolor=PALETTE["ink"],
            facecolor=PALETTE["paper"],
            alpha=0.95,
        )
    )
    ax.text(
        0.50,
        0.52,
        "Mechanism: SSR constrains local structure while preserving edit intent",
        ha="center",
        va="center",
        fontsize=PLOT_STYLE["body_size"],
        fontweight="bold",
        color=PALETTE["ink"],
    )

    box_nodes = [
        (0.07, 0.26, 0.22, "KnowEdit edit", "fact + prompt batch"),
        (0.39, 0.26, 0.22, "SSR update", "localized constrained step"),
        (0.71, 0.26, 0.22, "Post-edit model", "better retention"),
    ]
    for (x, y, width, text, node_label), (color, marker, hatch) in zip(
        box_nodes,
        CONCEPT_NODE_STYLES,
    ):
        patch = patches.FancyBboxPatch(
            (x, y),
            width,
            0.15,
            boxstyle="round,pad=0.02",
            linewidth=1.1,
            edgecolor=PALETTE["ink"],
            facecolor=color,
            hatch=hatch,
            alpha=0.80,
        )
        ax.add_patch(patch)
        ax.text(
            x + width / 2,
            y + 0.083,
            text,
            ha="center",
            va="center",
            fontsize=PLOT_STYLE["tiny_size"],
            fontweight="bold",
            color=PALETTE["ink"],
        )
        ax.text(
            x + width / 2,
            y - 0.01,
            node_label,
            ha="center",
            va="top",
            fontsize=PLOT_STYLE["tiny_size"] - 1.2,
            color=PALETTE["ink"],
        )
        ax.scatter(
            x + width / 2,
            y + 0.155,
            marker=marker,
            s=46 if marker != "s" else 58,
            c=PALETTE["ink"],
            edgecolors=PALETTE["ink"],
            linewidths=0.7,
            alpha=0.9,
        )

    ax.annotate(
        "",
        xy=(0.29, 0.34),
        xytext=(0.39, 0.34),
        arrowprops=dict(arrowstyle="->", lw=2.0, color=PALETTE["ink"]),
    )
    ax.annotate(
        "",
        xy=(0.61, 0.34),
        xytext=(0.71, 0.34),
        arrowprops=dict(arrowstyle="->", lw=2.0, color=PALETTE["ink"]),
    )

    callouts = [
        (0.10, 0.17, "1. factual utility enters through the edit batch"),
        (0.10, 0.12, "2. SSR keeps the update local in representation space"),
        (0.10, 0.07, "3. retained neighbors anchor the post-edit model"),
    ]
    for x, y, text in callouts:
        ax.text(
            x,
            y,
            text,
            fontsize=PLOT_STYLE["body_size"] - 0.2,
            color=PALETTE["ink"],
            fontweight="bold" if y > 0.15 else "normal",
        )

    ax.text(
        0.5,
        0.03,
        "Concept target: improve locality while retaining edited fact utility.",
        ha="center",
        va="center",
        color=PALETTE["ink"],
        fontsize=8.6,
        fontweight="bold",
    )

    ax.text(
        0.04,
        0.455,
        "Protocol semantics: each cell is a {method, model, dataset, edit-count} state.",
        ha="left",
        va="top",
        fontsize=6.3,
        color=PALETTE["ink"],
        fontstyle="italic",
    )


def _method_row_text(method_key: str) -> tuple[str, str]:
    style = METHOD_STYLES.get(method_key, METHOD_STYLES["biocs"])
    return METHOD_LABELS[method_key], style["row_marker"]


def _row_status_counts(
    row_pairs: list[tuple[str, str]],
    run_status: dict[tuple[str, str, str, int], RunStatus],
) -> list[Counter[str]]:
    counts = []
    for method_key, model_name in row_pairs:
        counter: Counter[str] = Counter()
        for col_name in GRID_X:
            dataset, edits = col_name.split("-")
            key = (method_key, model_name, dataset, int(edits))
            status = run_status.get(key, ("missing", None, None, "", "", "", "default"))[0]
            counter[status] += 1
        counts.append(counter)
    return counts


def _focus_pair_records(
    run_status: dict[tuple[str, str, str, int], RunStatus],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model_name in FOCUS_MODELS:
        for dataset, edits, label in FOCUS_SLICES:
            pair = {}
            for method_key in TARGET_METHOD_KEYS:
                pair[method_key] = run_status.get(
                    (method_key, model_name, dataset, edits),
                    ("missing", None, None, "", "", "", "default"),
                )
            rows.append(
                {
                    "model": model_name,
                    "dataset": dataset,
                    "edits": edits,
                    "label": label,
                    "pair": pair,
                }
            )
    return rows


def _focus_pair_label(model_name: str, dataset: str, edits: int) -> str:
    model_short = "Llama" if "Llama" in model_name else "Qwen-VL"
    ds = {"zsre": "Z50", "cf": "CF100", "recent": "REC100"}[dataset]
    return f"{model_short} {ds}"


def draw_result_matrix(
    ax: plt.Axes,
    run_status: dict[tuple[str, str, str, int], RunStatus],
) -> None:
    ax.set_title(
        "(b) Completed paired open-family transfer results",
        fontsize=PLOT_STYLE["panel_title_size"],
        fontweight="bold",
        pad=8,
        color=PALETTE["ink"],
    )
    row_pairs = [(model_name, method_key) for model_name in FOCUS_MODELS for method_key in TARGET_METHOD_KEYS]
    ax.set_xlim(-0.98, len(FOCUS_SLICES) - 0.5)
    ax.set_ylim(len(row_pairs) - 0.5, -0.5)
    ax.set_facecolor(PALETTE["paper"])
    ax.set_xticks(range(len(FOCUS_SLICES)))
    ax.set_xticklabels([label for _, _, label in FOCUS_SLICES], fontsize=PLOT_STYLE["body_size"])
    ax.set_yticks(range(len(row_pairs)))
    ax.set_yticklabels([""] * len(row_pairs))
    ax.tick_params(axis="both", which="both", length=0)
    ax.grid(visible=True, which="both", axis="both", color=PALETTE["grid"], linewidth=0.75)

    for model_block, model_name in enumerate(FOCUS_MODELS):
        row_start = model_block * len(TARGET_METHOD_KEYS)
        row_end = row_start + len(TARGET_METHOD_KEYS)
        ax.add_patch(
            patches.Rectangle(
                (-1.0, row_start - 0.5),
                len(FOCUS_SLICES) + 0.55,
                len(TARGET_METHOD_KEYS),
                facecolor=PALETTE["paper_soft"] if model_block % 2 else PALETTE["paper"],
                edgecolor="none",
                alpha=0.95,
                zorder=0,
            )
        )
        ax.text(
            -0.90,
            (row_start + row_end - 1) / 2,
            FOCUS_MODEL_LABELS[model_name],
            ha="left",
            va="center",
            fontsize=PLOT_STYLE["label_size"] - 0.2,
            fontweight="bold",
            color=PALETTE["ink"],
        )
        ax.axhline(row_start - 0.5, color=PALETTE["ink"], linewidth=1.2, alpha=0.55)
        ax.axhline(row_end - 0.5, color=PALETTE["ink"], linewidth=1.2, alpha=0.55)

    for row_idx, (model_name, method_key) in enumerate(row_pairs):
        style = METHOD_STYLES[method_key]
        ax.text(
            -0.72,
            row_idx,
            METHOD_LABELS[method_key],
            ha="right",
            va="center",
            fontsize=PLOT_STYLE["body_size"],
            color=PALETTE["ink"],
            fontweight="bold",
        )
        ax.text(
            -0.74,
            row_idx - 0.24,
            style["row_marker"],
            ha="right",
            va="center",
            fontsize=7.6,
            color=PALETTE["ink"],
        )
        for col_idx, (dataset, edits, _) in enumerate(FOCUS_SLICES):
            status, eff, loc, _, _, _, source = run_status.get(
                (method_key, model_name, dataset, edits),
                ("missing", None, None, "", "", "", "default"),
            )
            style_status = STATUS_STYLES[status]
            rect = patches.FancyBboxPatch(
                (col_idx - 0.48, row_idx - 0.42),
                0.96,
                0.84,
                boxstyle="round,pad=0.02",
                facecolor=style_status["face"],
                edgecolor=style_status["edge"],
                hatch=style_status["hatch"],
                linewidth=1.7 if method_key == "biocs" else 1.45,
                linestyle="-" if method_key == "biocs" else "--",
                alpha=0.98,
            )
            ax.add_patch(rect)
            ax.scatter(
                col_idx - 0.31,
                row_idx - 0.22,
                marker="o" if method_key == "biocs" else "s",
                s=38,
                facecolors=PALETTE["paper"],
                edgecolors=style_status["edge"],
                linewidths=1.0,
                zorder=4,
            )
            ax.text(
                col_idx - 0.05,
                row_idx - 0.24,
                f"[{RUN_SOURCE_LABEL.get(source, 'D')}]",
                ha="left",
                va="center",
                fontsize=5.8,
                color=style_status["edge"],
                fontweight="bold",
            )
            if status == "completed" and eff is not None and loc is not None:
                text = f"{eff:.1f}\n{loc:.1f}"
                weight = "bold"
            else:
                text = status
                weight = "normal"
            ax.text(
                col_idx,
                row_idx + 0.08,
                text,
                ha="center",
                va="center",
                fontsize=7.4 if status == "completed" else 6.8,
                fontweight=weight,
                color=PALETTE["ink"],
                zorder=5,
            )

    ax.text(
        0.01,
        0.98,
        "top=e, bottom=l; [T]/[V]/[D] = text, VL, default/live source",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.9,
        color=PALETTE["ink"],
        bbox=dict(boxstyle="round,pad=0.18", facecolor=PALETTE["paper"], edgecolor=PALETTE["grid"], linewidth=0.8),
    )


def draw_locality_gain_panel(
    ax: plt.Axes,
    run_status: dict[tuple[str, str, str, int], RunStatus],
) -> None:
    records = _focus_pair_records(run_status)
    ax.set_title(
        "(c) Locality margins in completed slices",
        fontsize=8.2,
        fontweight="bold",
        pad=7,
        color=PALETTE["ink"],
    )
    y_positions = np.arange(len(records))
    max_loc = 0.0
    for rec in records:
        bio = rec["pair"]["biocs"][2]
        ft = rec["pair"]["ft"][2]
        for val in (bio, ft):
            if val is not None:
                max_loc = max(max_loc, float(val))
    ax.set_xlim(0, max(46.0, max_loc + 7.0))
    ax.set_ylim(len(records) - 0.5, -0.5)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(
        [_focus_pair_label(rec["model"], rec["dataset"], rec["edits"]) for rec in records],
        fontsize=6.4,
    )
    ax.set_xlabel("Locality (%)", fontsize=7.5, fontweight="bold")
    ax.grid(axis="x", alpha=0.22, linewidth=0.9)
    ax.set_facecolor(PALETTE["paper"])

    for yi, rec in enumerate(records):
        bio_status, bio_eff, bio_loc, _, _, _, _ = rec["pair"]["biocs"]
        ft_status, ft_eff, ft_loc, _, _, _, _ = rec["pair"]["ft"]
        if bio_status == "completed" and ft_status == "completed" and bio_loc is not None and ft_loc is not None:
            ax.hlines(yi, ft_loc, bio_loc, color="#C7CED6", linewidth=2.0, zorder=1)
            ax.scatter(ft_loc, yi, marker="s", s=48, facecolors=PALETTE["paper"], edgecolors=PALETTE["completed_edge"], linewidths=1.2, zorder=3)
            ax.scatter(bio_loc, yi, marker="o", s=52, facecolors=PALETTE["completed_edge"], edgecolors=PALETTE["completed_edge"], linewidths=1.0, zorder=4)
            ax.text(
                max(ft_loc, bio_loc) + 0.8,
                yi,
                f"+{bio_loc - ft_loc:.1f} loc"
                + (f", +{bio_eff - ft_eff:.1f} eff" if bio_eff is not None and ft_eff is not None else ""),
                ha="left",
                va="center",
                fontsize=5.9,
                color=PALETTE["ink"],
            )

    ax.scatter([], [], marker="s", s=42, facecolors=PALETTE["paper"], edgecolors=PALETTE["completed_edge"], linewidths=1.1, label="FT")
    ax.scatter([], [], marker="o", s=46, facecolors=PALETTE["completed_edge"], edgecolors=PALETTE["completed_edge"], linewidths=1.0, label="SSR")
    ax.legend(frameon=False, fontsize=6.0, loc="lower right")


def draw_protocol_coverage_panel(
    ax: plt.Axes,
    run_status: dict[tuple[str, str, str, int], RunStatus],
) -> None:
    model_counts = _model_summary_counts(run_status)
    per_model_total = len(TARGET_METHOD_KEYS) * len(TARGET_DATASETS) * len(TARGET_EDIT_COUNTS)
    order = MODEL_LABELS
    ax.set_title(
        "(d) Protocol coverage",
        fontsize=8.2,
        fontweight="bold",
        pad=7,
        color=PALETTE["ink"],
    )
    y_positions = np.arange(len(order))
    ax.set_xlim(0, per_model_total)
    ax.set_ylim(len(order) - 0.5, -0.5)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([_short_model_name(model) for model in order], fontsize=6.2)
    ax.set_xticks([0, 3, 6, 9, 12])
    ax.set_xlabel("Completed cells per model (max 12)", fontsize=7.3, fontweight="bold")
    ax.grid(axis="x", alpha=0.22, linewidth=0.9)
    ax.set_facecolor(PALETTE["paper"])

    for yi, model_name in enumerate(order):
        counts = model_counts[model_name]
        done = counts.get("completed", 0)
        pending = counts.get("pending", 0)
        placeholder = counts.get("placeholder", 0)
        ax.add_patch(
            patches.FancyBboxPatch(
                (0, yi - 0.20),
                per_model_total,
                0.40,
                boxstyle="round,pad=0.01",
                linewidth=0.6,
                facecolor=PALETTE["paper"],
                edgecolor=PALETTE["grid"],
            )
        )
        if done:
            ax.add_patch(
                patches.Rectangle(
                    (0, yi - 0.20),
                    done,
                    0.40,
                    linewidth=0.0,
                    facecolor=PALETTE["completed_face"],
                    edgecolor=PALETTE["completed_edge"],
                )
            )
            ax.add_patch(
                patches.Rectangle(
                    (0, yi - 0.20),
                    done,
                    0.40,
                    linewidth=1.0,
                    facecolor="none",
                    edgecolor=PALETTE["completed_edge"],
                )
            )
        remaining = per_model_total - done
        if remaining > 0:
            ax.add_patch(
                patches.Rectangle(
                    (done, yi - 0.20),
                    remaining,
                    0.40,
                    linewidth=0.0,
                    facecolor=PALETTE["pending_face"],
                    edgecolor=PALETTE["pending_edge"],
                    hatch="///" if pending else "..",
                    alpha=0.7,
                )
            )
        ax.text(
            per_model_total + 0.20,
            yi,
            f"{done}/{per_model_total}"
            + (f", {pending} queued" if pending else "")
            + (f", {placeholder} placeholder" if placeholder else ""),
            ha="left",
            va="center",
            fontsize=5.8,
            color=PALETTE["ink"],
        )



def _total_status_counts(
    run_status: dict[tuple[str, str, str, int], RunStatus],
) -> Counter[str]:
    row_pairs = [(method_key, model_name) for method_key in TARGET_METHOD_KEYS for model_name in MODEL_LABELS]
    counter: Counter[str] = Counter()
    for method_key in TARGET_METHOD_KEYS:
        for model_name in MODEL_LABELS:
            for col_name in GRID_X:
                dataset, edits = col_name.split("-")
                key = (method_key, model_name, dataset, int(edits))
                status = run_status.get(key, ("missing", None, None, "", "", "", "default"))[0]
                counter[status] += 1
    return counter


def _model_summary_counts(
    run_status: dict[tuple[str, str, str, int], RunStatus],
) -> dict[str, Counter[str]]:
    model_counter: dict[str, Counter[str]] = {model: Counter() for model in MODEL_LABELS}
    for method_key in TARGET_METHOD_KEYS:
        for model_name in MODEL_LABELS:
            for col_name in GRID_X:
                dataset, edits = col_name.split("-")
                key = (method_key, model_name, dataset, int(edits))
                status = run_status.get(key, ("missing", None, None, "", "", "", "default"))[0]
                model_counter[model_name][status] += 1
    return model_counter


def _blocker_reason_counts(
    run_status: dict[tuple[str, str, str, int], RunStatus],
) -> tuple[Counter[str], dict[str, set[str]]]:
    counter: Counter[str] = Counter()
    models_by_reason: dict[str, set[str]] = {}
    for (_, model_name, _, _), (status, _, _, _, reason, _, _) in run_status.items():
        if status not in {"skipped", "failed", "placeholder"}:
            continue
        label = _normalize_blocker_reason(reason)
        if not label:
            continue
        counter[label] += 1
        models_by_reason.setdefault(label, set()).add(model_name)
    return counter, models_by_reason


def draw_status_matrix(
    ax: plt.Axes,
    run_status: dict[tuple[str, str, str, int], RunStatus],
    include_footer: bool = True,
) -> None:
    ax.set_title(
        "(b) Open-family run-status matrix",
        fontsize=PLOT_STYLE["panel_title_size"],
        fontweight="bold",
        pad=8,
        color=PALETTE["ink"],
    )
    row_pairs = [(method_key, model_name) for method_key in TARGET_METHOD_KEYS for model_name in MODEL_LABELS]
    ax.set_xticks(range(len(GRID_X)))
    ax.set_xticklabels(
        [str(edits) for _ in TARGET_DATASETS for edits in TARGET_EDIT_COUNTS],
        rotation=0,
        fontsize=PLOT_STYLE["body_size"],
    )
    ax.set_yticks(range(len(row_pairs)))
    ax.set_yticklabels([_short_model_name(model) for _, model in row_pairs], fontsize=PLOT_STYLE["body_size"])
    ax.set_xlabel("Edit budget", fontsize=PLOT_STYLE["label_size"], fontweight="bold")
    ax.set_ylabel("Model family", fontsize=PLOT_STYLE["label_size"], fontweight="bold")
    ax.set_xlim(-1.08, len(GRID_X) - 0.5)
    ax.set_ylim(len(row_pairs) - 0.5, -0.5)
    ax.set_facecolor(PALETTE["paper"])
    ax.tick_params(axis="both", which="both", length=0)
    ax.grid(visible=True, which="both", axis="both", color=PALETTE["grid"], linewidth=0.7)

    # Method blocks and row striping.
    for row_start in range(0, len(row_pairs), len(MODEL_LABELS)):
        row_end = row_start + len(MODEL_LABELS)
        method_key = row_pairs[row_start][0]
        style = METHOD_STYLES.get(method_key, METHOD_STYLES["biocs"])
        ax.add_patch(
            patches.Rectangle(
                (-1.1, row_start - 0.5),
                len(GRID_X) + 0.6,
                len(MODEL_LABELS),
                facecolor=style["row_face"],
                edgecolor="none",
                alpha=0.7,
                zorder=0,
            )
        )
        ax.text(
            style["label_offset"],
            (row_start + row_end - 1) / 2,
            METHOD_LABELS[method_key],
            va="center",
            ha="right",
            rotation=90,
            fontsize=PLOT_STYLE["label_size"],
            fontweight="bold",
            color=PALETTE["ink"],
            zorder=5,
        )
        ax.text(
            style["label_offset"] - 0.08,
            (row_start + row_end - 1) / 2,
            style["row_marker"],
            va="center",
            ha="right",
            fontsize=10.8,
            color=PALETTE["ink"],
            zorder=5,
        )
        ax.axhline(
            row_start - 0.5,
            color=PALETTE["ink"],
            linewidth=1.2,
            linestyle=style["line"],
            alpha=0.6,
        )
        ax.axhline(
            row_end - 0.5,
            color=PALETTE["ink"],
            linewidth=1.2,
            linestyle=style["line"],
            alpha=0.6,
        )

    # Dataset blocks and subheader: edit budgets under each dataset column pair.
    for idx in range(0, len(GRID_X), len(TARGET_EDIT_COUNTS)):
        if idx > 0:
            ax.axvline(idx - 0.5, color=PALETTE["ink"], linewidth=1.2, linestyle=":", alpha=0.5)
        x_center = idx + (len(TARGET_EDIT_COUNTS) - 1) / 2
        ax.text(
            x_center,
            -0.72,
            TARGET_DATASETS[idx // len(TARGET_EDIT_COUNTS)],
            ha="center",
            va="center",
            fontsize=PLOT_STYLE["label_size"],
            fontweight="bold",
            color=PALETTE["ink"],
        )

    # Main status lattice.
    for row_idx, (method_key, model_name) in enumerate(row_pairs):
        for col_idx, col_name in enumerate(GRID_X):
            dataset, edits = col_name.split("-")
            key = (method_key, model_name, dataset, int(edits))
            status, eff, loc, _, _, _, source = run_status.get(
                key,
                ("missing", None, None, "", "", "", "default"),
            )
            style = STATUS_STYLES[status]
            rect = patches.FancyBboxPatch(
                (col_idx - 0.5, row_idx - 0.5),
                1.0,
                1.0,
                boxstyle="round,pad=0.01",
                facecolor=style["face"],
                edgecolor=style["edge"],
                hatch=style["hatch"],
                linewidth=style["linewidth"],
                linestyle=style["linestyle"],
                alpha=0.97,
            )
            ax.add_patch(rect)

            marker = STATUS_STYLES[status]
            ax.scatter(
                col_idx,
                row_idx,
                marker=marker["marker"],
                s=98,
                c=[marker["marker_face"]],
                edgecolors=marker["marker_edge"],
                linewidths=1.1,
                zorder=4,
            )

            source_tag = f"[{RUN_SOURCE_LABEL.get(source, 'D')}]"
            if status != "missing":
                ax.text(
                    col_idx - 0.39,
                    row_idx - 0.34,
                    source_tag,
                    ha="left",
                    va="top",
                    fontsize=5.3,
                    color=style["edge"],
                    fontweight="bold",
                    zorder=6,
                )

            if status == "completed" and eff is not None and loc is not None:
                text = f"{eff:.1f}/{loc:.1f}"
                text_weight = "bold"
                text_size = PLOT_STYLE["tiny_size"] + 0.4
            elif status == "placeholder":
                text = "VL"
                text_weight = "bold"
                text_size = PLOT_STYLE["tiny_size"] + 0.2
            else:
                text = ""
                text_weight = "normal"
                text_size = PLOT_STYLE["tiny_size"]
            if text:
                ax.text(
                    col_idx,
                    row_idx + 0.16,
                    text,
                    ha="center",
                    va="center",
                    fontsize=text_size,
                    fontweight=text_weight,
                    color=PALETTE["ink"],
                    zorder=5,
                )
    ax.text(
        len(GRID_X) - 0.36,
        -0.98,
        "[T]/[V]/[D] denote text, VL, and default manifest provenance; marker + hatch encode run state.",
        ha="right",
        va="center",
        fontsize=6.1,
        color=PALETTE["ink"],
        transform=ax.get_xaxis_transform(),
    )

    if include_footer:
        total_counts = _total_status_counts(run_status)
        total_cells = len(TARGET_METHOD_KEYS) * len(MODEL_LABELS) * len(TARGET_DATASETS) * len(TARGET_EDIT_COUNTS)
        status_summary = " | ".join(
            f"{status}: {total_counts.get(status, 0)}"
            for status in STATUS_LEGEND_ORDER
        )
        ax.text(
            0.0,
            -1.22,
            f"Panel encodes: status cell = method × model × dataset × edits ({total_cells} cells). {status_summary}.",
            ha="left",
            va="center",
            fontsize=6.7,
            color=PALETTE["ink"],
            transform=ax.get_xaxis_transform(),
        )

        completed_ratio = total_counts.get("completed", 0)
        if total_cells:
            completed_percent = 100 * completed_ratio / total_cells
        else:
            completed_percent = 0.0
        ax.text(
            0.0,
            -1.10,
            f"Completed-cell ratio: {completed_ratio}/{total_cells} ({completed_percent:.1f}%).",
            ha="left",
            va="center",
            fontsize=6.7,
            color=PALETTE["ink"],
            transform=ax.get_xaxis_transform(),
        )


def draw_status_digest(
    ax: plt.Axes,
    run_status: dict[tuple[str, str, str, int], RunStatus],
) -> None:
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_facecolor(PALETTE["paper_muted"])
    ax.axis("off")

    total_counts = _total_status_counts(run_status)
    model_counts = _model_summary_counts(run_status)
    total_cells = len(TARGET_METHOD_KEYS) * len(MODEL_LABELS) * len(TARGET_DATASETS) * len(TARGET_EDIT_COUNTS)

    completed = total_counts.get("completed", 0)
    completed_ratio = completed / total_cells if total_cells else 0.0
    source_total = completed if completed else 0
    blocker_counts, blocker_models = _blocker_reason_counts(run_status)
    source_completed = {
        "text": len([1 for v in run_status.values() if v[0] == "completed" and v[6] == "text"]),
        "vl": len([1 for v in run_status.values() if v[0] == "completed" and v[6] == "vl"]),
        "default": len([1 for v in run_status.values() if v[0] == "completed" and v[6] == "default"]),
    }
    if source_total:
        source_summary = (
            "Completed provenance: "
            f"text={source_completed['text']} ({100 * source_completed['text'] / source_total:.0f}%), "
            f"vl={source_completed['vl']} ({100 * source_completed['vl'] / source_total:.0f}%), "
            f"manifest={source_completed['default']} ({100 * source_completed['default'] / source_total:.0f}%)"
        )
    else:
        source_summary = "Completed provenance: no completed runs yet."
    if blocker_counts:
        top_lines = []
        for label, count in blocker_counts.most_common(2):
            model_list = ", ".join(sorted(_short_model_name(name) for name in blocker_models.get(label, set())))
            top_lines.append(f"{label} ({count} cells; {model_list})")
        blocker_summary = "Primary blockers: " + " | ".join(top_lines)
    else:
        blocker_summary = "Primary blockers: none."

    ax.text(
        0.02,
        0.96,
        "(c) Queue digest and model coverage",
        ha="left",
        va="top",
        fontsize=8.7,
        fontweight="bold",
        color=PALETTE["ink"],
    )

    ax.text(
        0.02,
        0.90,
        f"Overall: {completed}/{total_cells} completed ({100 * completed_ratio:.1f}%), "
        f"{total_counts.get('pending', 0)} queued, {total_counts.get('placeholder', 0)} text-only VL placeholders.",
        ha="left",
        va="top",
        fontsize=6.8,
        fontweight="bold",
        color=PALETTE["ink"],
    )

    ax.text(
        0.02,
        0.84,
        source_summary,
        ha="left",
        va="top",
        fontsize=6.0,
        color=PALETTE["ink"],
    )

    ax.text(
        0.02,
        0.78,
        blocker_summary,
        ha="left",
        va="top",
        fontsize=5.7,
        color=PALETTE["ink"],
    )

    bar_x = 0.02
    bar_y = 0.63
    bar_w = 0.63
    bar_h = 0.09
    ax.add_patch(
        patches.FancyBboxPatch(
            (bar_x, bar_y),
            bar_w,
            bar_h,
            boxstyle="round,pad=0.01",
            linewidth=0.9,
            facecolor=PALETTE["paper"],
            edgecolor=PALETTE["grid"],
        )
    )
    cursor = bar_x
    for status_key in ("completed", "pending", "placeholder", "failed", "skipped", "missing"):
        count = total_counts.get(status_key, 0)
        if not count:
            continue
        frac = count / total_cells if total_cells else 0.0
        width = bar_w * frac
        style = STATUS_STYLES[status_key]
        ax.add_patch(
            patches.FancyBboxPatch(
                (cursor, bar_y),
                width,
                bar_h,
                boxstyle="round,pad=0.004",
                linewidth=0.8,
                facecolor=style["face"],
                edgecolor=style["edge"],
                hatch=style["hatch"],
                alpha=0.95,
            )
        )
        if width > 0.06:
            ax.text(
                cursor + width / 2,
                bar_y + bar_h / 2,
                f"{status_key} {count}",
                ha="center",
                va="center",
                fontsize=5.8,
                color=PALETTE["ink"],
                fontweight="bold",
            )
        cursor += width

    legend_points = [
        (0.74, 0.67, "completed"),
        (0.74, 0.58, "pending"),
        (0.74, 0.49, "placeholder"),
        (0.88, 0.67, "failed"),
        (0.88, 0.58, "skipped"),
        (0.88, 0.49, "missing"),
    ]
    for x, y, status_key in legend_points:
        style = STATUS_STYLES[status_key]
        ax.scatter(
            x,
            y,
            marker=style["marker"],
            s=34,
            c=style["marker_face"],
            edgecolors=style["marker_edge"],
            linewidths=0.8,
            zorder=3,
        )
        ax.text(
            x + 0.03,
            y,
            style["label"],
            ha="left",
            va="center",
            fontsize=5.8,
            color=PALETTE["ink"],
        )

    ax.text(
        0.02,
        0.47,
        "Model-level coverage (both methods):",
        ha="left",
        va="top",
        fontsize=6.9,
        fontweight="bold",
        color=PALETTE["ink"],
    )

    per_model_total = len(TARGET_METHOD_KEYS) * len(TARGET_DATASETS) * len(TARGET_EDIT_COUNTS)
    y_cursor = 0.37
    for model_name in MODEL_LABELS:
        counts = model_counts[model_name]
        model_done = counts.get("completed", 0)
        model_pending = counts.get("pending", 0)
        model_placeholder = counts.get("placeholder", 0)
        ax.text(
            0.03,
            y_cursor,
            _short_model_name(model_name),
            ha="left",
            va="center",
            fontsize=6.2,
            fontweight="bold",
            color=PALETTE["ink"],
        )
        ax.add_patch(
            patches.FancyBboxPatch(
                (0.30, y_cursor - 0.015),
                0.26,
                0.03,
                boxstyle="round,pad=0.003",
                linewidth=0.4,
                facecolor=PALETTE["paper"],
                edgecolor=PALETTE["grid"],
            )
        )
        if per_model_total:
            cursor = 0.30
            for status_key in ("completed", "pending", "placeholder", "missing"):
                count = counts.get(status_key, 0)
                if not count:
                    continue
                width = 0.26 * (count / per_model_total)
                style = STATUS_STYLES[status_key]
                ax.add_patch(
                    patches.Rectangle(
                        (cursor, y_cursor - 0.015),
                        width,
                        0.03,
                        linewidth=0.0,
                        facecolor=style["face"],
                        edgecolor=style["edge"],
                        hatch=style["hatch"],
                    )
                )
                cursor += width
        ax.text(
            0.59,
            y_cursor,
            f"{model_done}/{per_model_total} done | {model_pending} queued"
            + (f" | {model_placeholder} placeholder" if model_placeholder else ""),
            ha="left",
            va="center",
            fontsize=6.0,
            color=PALETTE["ink"],
        )
        y_cursor += -0.08


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--concept-title",
        type=str,
        default=DEFAULT_CONCEPT_TITLE,
        help="Title text for the concept panel.",
    )
    parser.add_argument(
        "--concept-image",
        type=Path,
        default=None,
        help="Optional path to a concept image inserted into the left panel.",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        action="append",
        default=None,
        help=(
            "Run directory or parent directory containing run folders. "
            "Repeat this flag to merge multiple directories (for example text+VL phases)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        action="append",
        default=None,
        help="Optional extra output directory in addition to Science and ICLR figure folders.",
    )
    parser.add_argument(
        "--figure-title",
        type=str,
        default=DEFAULT_FIGURE_TITLE,
        help="Top-level title for the composite figure.",
    )
    parser.add_argument(
        "--figure-name",
        type=str,
        default=DEFAULT_FIGURE_NAME,
        help="Output filename basename (without extension).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="Output image DPI.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_roots = args.runs_root if args.runs_root else [PROJECT_ROOT / "results" / "llm_ke"]
    concept_image = _resolve_concept_image(args.concept_image)
    output_dirs = [SCI_FIG_DIR, ICLR_FIG_DIR]
    if args.output_dir:
        output_dirs.extend(args.output_dir)

    run_status = collect_run_statuses(results_roots)

    plt.rcParams.update(
        {
            "font.family": _resolve_font_family(),
            "axes.titlesize": PLOT_STYLE["panel_title_size"],
            "axes.labelsize": PLOT_STYLE["label_size"],
            "xtick.labelsize": PLOT_STYLE["body_size"],
            "ytick.labelsize": PLOT_STYLE["body_size"],
            "legend.fontsize": PLOT_STYLE["legend_size"],
            "font.size": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    fig = plt.figure(
        figsize=(
            PLOT_STYLE["figure_width"],
            PLOT_STYLE["figure_height"],
        ),
        dpi=args.dpi,
    )
    gs = fig.add_gridspec(
        1,
        2,
        width_ratios=[
            PLOT_STYLE["left_panel_ratio"],
            PLOT_STYLE["right_panel_ratio"],
        ],
        wspace=0.16,
    )
    ax_left = fig.add_subplot(gs[0, 0])
    right_spec = gs[0, 1]
    gs_right = right_spec.subgridspec(2, 1, height_ratios=[2.8, 1.75], hspace=0.18)
    ax_right = fig.add_subplot(gs_right[0, 0])
    gs_bottom = gs_right[1, 0].subgridspec(1, 2, width_ratios=[1.18, 1.0], wspace=0.30)
    ax_digest_left = fig.add_subplot(gs_bottom[0, 0])
    ax_digest_right = fig.add_subplot(gs_bottom[0, 1])

    draw_concept_panel(ax_left, concept_image=concept_image, title=args.concept_title)
    draw_result_matrix(ax_right, run_status)
    draw_locality_gain_panel(ax_digest_left, run_status)
    draw_protocol_coverage_panel(ax_digest_right, run_status)

    fig.suptitle(
        args.figure_title,
        fontsize=PLOT_STYLE["title_size"],
        fontweight="bold",
        color=PALETTE["ink"],
    )

    for ax in (ax_left, ax_right, ax_digest_left, ax_digest_right):
        for side in ("top", "right", "bottom", "left"):
            ax.spines[side].set_visible(False)

    for out_dir in output_dirs:
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"{args.figure_name}.pdf")
        fig.savefig(out_dir / f"{args.figure_name}.png", dpi=args.dpi)

    plt.close(fig)


if __name__ == "__main__":
    main()
