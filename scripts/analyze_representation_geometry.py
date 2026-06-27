"""Representation geometry diagnostics for SSR experiments.

The script intentionally separates geometry evidence from functional claims:
classifier rank/cosine can support the SSR mechanism, but capacity or mask
quality still requires matched functional metrics.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PAPER_FIGURES = Path(os.environ.get("SSR_LEGACY_FIGURE_DIR", ROOT / "figures" / "legacy"))
SCIENCE_FIGURES = Path(os.environ.get("SSR_FIGURE_DIR", ROOT / "figures"))
ANALYSIS_DIR = ROOT / "results" / "representation_geometry_20260430"
PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
SCIENCE_FIGURES.mkdir(parents=True, exist_ok=True)
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "baseline": "#8A9296",
    "biocs": "#5C9275",
    "biocs_kd": "#2F6F73",
    "negative": "#9B4F48",
    "grid": "#E8E6DF",
    "ink": "#2D3436",
}

LABELS = {
    "baseline": "Baseline",
    "biocs": "SSR",
    "biocs_kd": "SSR+KD",
}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 12,
        "font.weight": "bold",
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
        "axes.linewidth": 1.8,
        "axes.edgecolor": COLORS["ink"],
        "xtick.major.width": 1.5,
        "ytick.major.width": 1.5,
        "legend.frameon": True,
        "legend.edgecolor": COLORS["ink"],
        "legend.framealpha": 0.95,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.12,
    }
)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def stat(summary: dict[str, Any], method: str, metric: str) -> float:
    return float(summary["classification"][method][metric]["mean"])


def thick_spines(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_linewidth(1.8)
        spine.set_color(COLORS["ink"])
    ax.tick_params(width=1.5, length=5, colors=COLORS["ink"])


def effective_rank(matrix: np.ndarray) -> float:
    s = np.linalg.svd(matrix.astype(np.float64), compute_uv=False)
    total = float(s.sum())
    if total <= 0:
        return 0.0
    p = s / total
    return float(np.exp(-(p * np.log(p + 1e-12)).sum()))


def mean_abs_offdiag_cosine(matrix: np.ndarray) -> float:
    x = matrix.reshape(matrix.shape[0], -1).astype(np.float64)
    norm = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    x = x / norm
    gram = x @ x.T
    mask = ~np.eye(gram.shape[0], dtype=bool)
    return float(np.abs(gram[mask]).mean())


def cod_checkpoint_geometry() -> dict[str, dict[str, float]]:
    """Measure decoder/query geometry from completed COD checkpoints.

    These checkpoints are from the main COD ablation where the combined
    query+channel regularizer was too strong. We use them as a boundary
    diagnostic, not as the positive COD performance claim.
    """
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on local env
        return {"error": {"message": str(exc)}}  # type: ignore[dict-item]

    paths = {
        "baseline": RESULTS
        / "cod_main_gpu2_20260429_r3"
        / "checkpoints"
        / "baseline_seed0_best.pt",
        "biocs_both": RESULTS
        / "cod_main_gpu2_20260429_r3"
        / "checkpoints"
        / "biocs_both_seed0_best.pt",
    }
    out: dict[str, dict[str, float]] = {}
    for name, path in paths.items():
        if not path.exists():
            continue
        state = torch.load(path, map_location="cpu")
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        queries = state["mask_queries"].detach().cpu().numpy()
        smooth = np.concatenate(
            [
                state["smooth.0.weight"].detach().cpu().numpy().reshape(128, -1),
                state["smooth.3.weight"].detach().cpu().numpy().reshape(128, -1),
            ],
            axis=1,
        )
        out[name] = {
            "query_effective_rank": effective_rank(queries),
            "query_offdiag_cosine": mean_abs_offdiag_cosine(queries),
            "channel_effective_rank": effective_rank(smooth),
            "channel_offdiag_cosine": mean_abs_offdiag_cosine(smooth),
        }
    return out


def build_summary() -> dict[str, Any]:
    synthetic = load_json(RESULTS / "capacity_mechanism_probe_20260429" / "summary_aggregated.json")
    cub_semantic = load_json(RESULTS / "cub200_capacity_classification_20260429" / "summary.json")
    cub_random = load_json(RESULTS / "cub200_capacity_random_order_20260429" / "summary.json")
    cod_320_baseline = load_json(RESULTS / "cod_320_baseline_gpu2_20260429" / "summary.json")
    cod_320_channel = load_json(RESULTS / "cod_320_channel_gpu3_20260429" / "summary.json")
    cod_main = load_json(RESULTS / "cod_main_gpu2_20260429_r3" / "summary.json")

    return {
        "synthetic": synthetic,
        "cub_semantic": cub_semantic,
        "cub_random": cub_random,
        "cod_320_macro_score": {
            "baseline": cod_320_baseline["baseline"]["macro_score"]["mean"],
            "biocs_channel": cod_320_channel["biocs_channel"]["macro_score"]["mean"],
        },
        "cod_boundary_macro_score": {
            "baseline": cod_main["baseline"]["macro_score"]["mean"],
            "biocs_both": cod_main["biocs_both"]["macro_score"]["mean"],
        },
        "cod_boundary_geometry": cod_checkpoint_geometry(),
    }


def plot(summary: dict[str, Any]) -> None:
    fig = plt.figure(figsize=(13.6, 7.2))
    gs = fig.add_gridspec(
        2, 2, width_ratios=[1.18, 0.96], height_ratios=[1.0, 0.80], wspace=0.28, hspace=0.32
    )

    methods = ["baseline", "biocs", "biocs_kd"]
    semantic_rank = [stat(summary["cub_semantic"], m, "effective_rank") for m in methods]
    random_rank = [stat(summary["cub_random"], m, "effective_rank") for m in methods]
    semantic_cos = [stat(summary["cub_semantic"], m, "mean_abs_offdiag_cosine") for m in methods]
    random_cos = [stat(summary["cub_random"], m, "mean_abs_offdiag_cosine") for m in methods]

    # (a) Hero panel: geometry phase portrait under semantic/random orders
    ax = fig.add_subplot(gs[:, 0])
    marker_map = {"baseline": "o", "biocs": "s", "biocs_kd": "^"}
    offsets = {"baseline": (0.004, -1.5), "biocs": (0.004, -1.8), "biocs_kd": (0.004, 1.8)}
    for m in methods:
        c = COLORS[m]
        i = methods.index(m)
        ax.plot(
            [semantic_cos[i], random_cos[i]],
            [semantic_rank[i], random_rank[i]],
            color=c,
            linestyle=(0, (4, 3)),
            linewidth=2.0,
            alpha=0.9,
            zorder=2,
        )
        ax.scatter(
            semantic_cos[i], semantic_rank[i],
            s=150, marker=marker_map[m], facecolors=c, edgecolors=COLORS["ink"],
            linewidths=1.6, zorder=4,
        )
        ax.scatter(
            random_cos[i], random_rank[i],
            s=150, marker=marker_map[m], facecolors="white", edgecolors=COLORS["ink"],
            linewidths=1.8, zorder=5,
        )
        dx, dy = offsets[m]
        ax.text(
            semantic_cos[i] + dx, semantic_rank[i] + dy, LABELS[m],
            fontsize=9.8, fontweight="bold", color=COLORS["ink"], ha="left", va="center"
        )

    ax.scatter([], [], s=95, marker="o", facecolors=COLORS["ink"], edgecolors=COLORS["ink"], label="Semantic order")
    ax.scatter([], [], s=95, marker="o", facecolors="white", edgecolors=COLORS["ink"], label="Random order")
    ax.legend(loc="lower left", fontsize=8.3, frameon=True, framealpha=0.96)
    ax.annotate(
        "Method separation is larger\nthan order-specific drift",
        xy=(0.108, 158.8),
        xytext=(0.182, 166.8),
        arrowprops=dict(arrowstyle="->", color="#666666", lw=1.3),
        bbox=dict(boxstyle="round,pad=0.32", facecolor="white", edgecolor="#B8C0CA", linewidth=1.2),
        fontsize=8.8,
        fontweight="bold",
        color=COLORS["ink"],
    )
    ax.set_xlabel("Mean |prototype cosine| ↓", fontsize=12.2, fontweight="bold")
    ax.set_ylabel("Effective rank ↑", fontsize=12.2, fontweight="bold")
    ax.set_title("(a) Order-robust geometry shift", fontsize=12.8, fontweight="bold", pad=8)
    ax.set_xlim(0.36, 0.03)
    ax.set_ylim(112, 174)
    ax.grid(True, color=COLORS["grid"], linewidth=1.0)
    thick_spines(ax)

    # (b) Geometry-to-capacity conversion panel
    ax = fig.add_subplot(gs[0, 1])
    domain_cfg = {
        "Synthetic": {"marker": "o", "filled": False},
        "CUB200": {"marker": "^", "filled": True},
    }
    domain_data = {
        "Synthetic": [(summary["synthetic"][m]["effective_rank"]["mean"], summary["synthetic"][m]["capacity_at_50"]["mean"]) for m in methods],
        "CUB200": [(stat(summary["cub_semantic"], m, "effective_rank"), stat(summary["cub_semantic"], m, "cil_capacity_at_20")) for m in methods],
    }
    for domain, pts in domain_data.items():
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, color="#A8AFB7", linewidth=1.8, zorder=1)
        for (xv, yv), m in zip(pts, methods):
            c = COLORS[m]
            ax.scatter(
                xv, yv,
                s=128 if domain == "CUB200" else 118,
                marker=domain_cfg[domain]["marker"],
                facecolors=c if domain_cfg[domain]["filled"] else "white",
                edgecolors=COLORS["ink"],
                linewidths=1.7,
                color=c,
                zorder=3,
            )
    ax.scatter([], [], s=96, marker="o", facecolors="white", edgecolors=COLORS["ink"], label="Synthetic capacity probe")
    ax.scatter([], [], s=100, marker="^", facecolors=COLORS["ink"], edgecolors=COLORS["ink"], label="CUB200 seen-class CIL")
    ax.legend(loc="upper left", fontsize=7.8, frameon=True, framealpha=0.96)
    ax.text(
        0.98, 0.06,
        "SSR shifts geometry;\nKD converts that shift into retention",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.6,
        fontweight="bold",
        color=COLORS["ink"],
        bbox=dict(boxstyle="round,pad=0.30", facecolor="white", edgecolor="#B8C0CA", linewidth=1.2),
    )
    ax.set_xlabel("Effective rank ↑", fontsize=12.0, fontweight="bold")
    ax.set_ylabel("Functional proxy ↑", fontsize=12.0, fontweight="bold")
    ax.set_title("(b) KD links geometry to retention", fontsize=12.8, fontweight="bold", pad=8)
    ax.set_xlim(40, 175)
    ax.set_ylim(0, 135)
    ax.grid(True, color=COLORS["grid"], linewidth=1.0)
    thick_spines(ax)

    # (c) COD matched-boundary panel
    ax = fig.add_subplot(gs[1, 1])
    cod_delta = (summary["cod_320_macro_score"]["biocs_channel"] - summary["cod_320_macro_score"]["baseline"]) * 100.0
    cod_boundary = (summary["cod_boundary_macro_score"]["biocs_both"] - summary["cod_boundary_macro_score"]["baseline"]) * 100.0
    labels = ["Channel-only\nSSR", "Query+channel\nSSR"]
    vals = [cod_delta, cod_boundary]
    colors = [COLORS["biocs_kd"], COLORS["negative"]]
    markers = ["s", "X"]
    ypos = np.arange(len(labels))
    for yi, val, c, mk in zip(ypos, vals, colors, markers):
        ax.hlines(yi, 0, val, color="#A8AFB7", linewidth=2.4, zorder=1)
        ax.scatter(0, yi, s=80, marker="o", facecolors="white", edgecolors=COLORS["ink"], linewidths=1.5, zorder=3)
        ax.scatter(val, yi, s=135, marker=mk, facecolors=c if mk != "X" else "none",
                   edgecolors=COLORS["ink"] if mk != "X" else c, color=c, linewidths=2.0, zorder=4)
        ax.text(
            val + (0.9 if val >= 0 else -0.9), yi, f"{val:+.1f}",
            ha="left" if val >= 0 else "right", va="center",
            fontsize=9.8, fontweight="bold", color=c
        )
    ax.axvline(0, color=COLORS["ink"], linewidth=1.5)
    ax.text(
        0.98, 0.08,
        "Positive channel evidence,\nbut a bounded dose window",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8.6,
        fontweight="bold", color=COLORS["ink"],
        bbox=dict(boxstyle="round,pad=0.30", facecolor="white", edgecolor="#B8C0CA", linewidth=1.2),
    )
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=9.8, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlabel("COD change vs baseline (pts)", fontsize=11.6, fontweight="bold")
    ax.set_title("(c) COD target and dose are bounded", fontsize=12.8, fontweight="bold", pad=8)
    ax.set_xlim(min(-8.0, cod_boundary - 2.0), max(8.0, cod_delta + 2.0))
    ax.grid(axis="x", color=COLORS["grid"], linewidth=1.0)
    thick_spines(ax)
    for out_dir in (PAPER_FIGURES, SCIENCE_FIGURES):
        fig.savefig(out_dir / "fig_representation_geometry.pdf")
        fig.savefig(out_dir / "fig_representation_geometry.png", dpi=300)
    plt.close(fig)


def main() -> None:
    summary = build_summary()
    with (ANALYSIS_DIR / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    plot(summary)
    print(f"Wrote {ANALYSIS_DIR / 'summary.json'}")
    print(f"Wrote {PAPER_FIGURES / 'fig_representation_geometry.pdf'}")


if __name__ == "__main__":
    main()
