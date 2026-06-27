#!/usr/bin/env python3
"""Fourier diagnostics for task-wise SSR classifier updates.

The early project analysis used one scalar: high-frequency power after
flattening task-to-task classifier updates. That scalar is fragile because the
flattened ordering of a weight matrix is arbitrary. This script keeps that
legacy number for comparability, but adds three less ambiguous diagnostics:

1. class-axis FFT: do updates vary abruptly across class prototypes?
2. feature-axis FFT: do updates vary abruptly within each prototype vector?
3. two-dimensional FFT: where is update energy concentrated in the matrix?

It also reports a permutation null for the flattened signal. If a method only
looks different after arbitrary flattening but not on axis-aware spectra, the
paper should not use it as mechanistic evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SNAPSHOT = (
    ROOT / "results" / "deep_representation_case_study_20260430" / "weights_and_snapshots.npz"
)
DEFAULT_OUT = ROOT / "results" / "fourier_update_analysis_20260514"
FIG_DIR = Path(os.environ.get("SSR_LEGACY_FIGURE_DIR", ROOT / "figures" / "legacy"))
SCI_FIG_DIR = Path(os.environ.get("SSR_FIGURE_DIR", ROOT / "figures"))
SCI_FIG_DIR.mkdir(parents=True, exist_ok=True)

METHODS = ["baseline", "biocs", "biocs_kd"]
LABELS = {"baseline": "Baseline", "biocs": "SSR", "biocs_kd": "SSR+KD"}
COLORS = {"baseline": "#8A9296", "biocs": "#5C9275", "biocs_kd": "#2F6F73"}


def band_stats(power: np.ndarray, low_cut: float = 0.2, high_cut: float = 0.6) -> dict[str, float]:
    """Return normalized spectral band statistics for a non-negative spectrum."""

    p = np.asarray(power, dtype=np.float64).reshape(-1)
    if p.size <= 1:
        return {
            "low_ratio": 0.0,
            "mid_ratio": 0.0,
            "high_ratio": 0.0,
            "spectral_centroid": 0.0,
            "spectral_entropy": 0.0,
            "total_power": float(p.sum()) if p.size else 0.0,
        }
    total = float(p.sum() + 1e-12)
    freqs = np.linspace(0.0, 1.0, p.size)
    pn = p / total
    return {
        "low_ratio": float(p[freqs <= low_cut].sum() / total),
        "mid_ratio": float(p[(freqs > low_cut) & (freqs <= high_cut)].sum() / total),
        "high_ratio": float(p[freqs > high_cut].sum() / total),
        "spectral_centroid": float((freqs * pn).sum()),
        "spectral_entropy": float(-(pn * np.log(pn + 1e-12)).sum() / math.log(p.size)),
        "total_power": float(total),
    }


def radial_power_2d(delta: np.ndarray) -> np.ndarray:
    """Radially average a centered 2D FFT power spectrum."""

    fft = np.fft.fftshift(np.fft.fft2(delta.astype(np.float64)))
    power = np.abs(fft) ** 2
    rows, cols = power.shape
    yy, xx = np.indices(power.shape)
    cy, cx = (rows - 1) / 2.0, (cols - 1) / 2.0
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    radius = radius / (radius.max() + 1e-12)
    bins = np.linspace(0.0, 1.0, 65)
    out = np.zeros(len(bins) - 1, dtype=np.float64)
    for i in range(len(out)):
        mask = (radius >= bins[i]) & (radius < bins[i + 1])
        out[i] = power[mask].sum()
    return out


def one_dim_power(signal: np.ndarray) -> np.ndarray:
    spec = np.fft.rfft(signal.astype(np.float64))
    return np.abs(spec) ** 2


def axis_power(delta: np.ndarray, axis: int) -> np.ndarray:
    """Average rFFT power over rows or columns."""

    spec = np.fft.rfft(delta.astype(np.float64), axis=axis)
    power = np.abs(spec) ** 2
    reduce_axes = tuple(i for i in range(power.ndim) if i != axis)
    return power.mean(axis=reduce_axes)


def permutation_null(delta: np.ndarray, rng: np.random.Generator, n_perm: int) -> dict[str, float]:
    flat = delta.reshape(-1).copy()
    ratios = []
    centroids = []
    for _ in range(n_perm):
        rng.shuffle(flat)
        stats = band_stats(one_dim_power(flat))
        ratios.append(stats["high_ratio"])
        centroids.append(stats["spectral_centroid"])
    return {
        "flat_high_ratio_perm_mean": float(np.mean(ratios)),
        "flat_high_ratio_perm_std": float(np.std(ratios)),
        "flat_centroid_perm_mean": float(np.mean(centroids)),
        "flat_centroid_perm_std": float(np.std(centroids)),
    }


def summarize_method(snapshots: np.ndarray, rng: np.random.Generator, n_perm: int) -> dict:
    deltas = np.diff(snapshots.astype(np.float64), axis=0)
    per_task = []
    spectra = {"flat": [], "class_axis": [], "feature_axis": [], "radial_2d": []}

    for task_id, delta in enumerate(deltas, start=2):
        flat = band_stats(one_dim_power(delta.reshape(-1)))
        class_axis = band_stats(axis_power(delta, axis=0))
        feature_axis = band_stats(axis_power(delta, axis=1))
        radial_2d = band_stats(radial_power_2d(delta))
        null = permutation_null(delta, rng, n_perm)
        smoothness = float(np.mean(np.abs(np.diff(delta, axis=0))) + np.mean(np.abs(np.diff(delta, axis=1))))
        row_norm_cv = float(np.std(np.linalg.norm(delta, axis=1)) / (np.mean(np.linalg.norm(delta, axis=1)) + 1e-12))
        item = {
            "task_after": task_id,
            "delta_l2": float(np.linalg.norm(delta)),
            "delta_l1": float(np.abs(delta).sum()),
            "matrix_total_variation": smoothness,
            "row_update_norm_cv": row_norm_cv,
            "flat": flat,
            "class_axis": class_axis,
            "feature_axis": feature_axis,
            "radial_2d": radial_2d,
            "permutation_null": null,
        }
        per_task.append(item)
        spectra["flat"].append(one_dim_power(delta.reshape(-1)))
        spectra["class_axis"].append(axis_power(delta, axis=0))
        spectra["feature_axis"].append(axis_power(delta, axis=1))
        spectra["radial_2d"].append(radial_power_2d(delta))

    aggregate = {}
    for key in ["delta_l2", "delta_l1", "matrix_total_variation", "row_update_norm_cv"]:
        vals = np.asarray([x[key] for x in per_task], dtype=float)
        aggregate[key] = {"mean": float(vals.mean()), "std": float(vals.std())}
    for view in ["flat", "class_axis", "feature_axis", "radial_2d"]:
        for metric in ["low_ratio", "mid_ratio", "high_ratio", "spectral_centroid", "spectral_entropy", "total_power"]:
            vals = np.asarray([x[view][metric] for x in per_task], dtype=float)
            aggregate[f"{view}_{metric}"] = {"mean": float(vals.mean()), "std": float(vals.std())}
    for metric in [
        "flat_high_ratio_perm_mean",
        "flat_high_ratio_perm_std",
        "flat_centroid_perm_mean",
        "flat_centroid_perm_std",
    ]:
        vals = np.asarray([x["permutation_null"][metric] for x in per_task], dtype=float)
        aggregate[metric] = {"mean": float(vals.mean()), "std": float(vals.std())}

    mean_spectra = {}
    for view, arrs in spectra.items():
        max_len = max(len(a) for a in arrs)
        padded = []
        for a in arrs:
            x_old = np.linspace(0.0, 1.0, len(a))
            x_new = np.linspace(0.0, 1.0, max_len)
            y = np.interp(x_new, x_old, a)
            y = y / (y.sum() + 1e-12)
            padded.append(y)
        mean_spectra[view] = np.mean(np.stack(padded), axis=0)

    return {"per_task": per_task, "aggregate": aggregate, "mean_spectra": mean_spectra}


def fmt(stat: dict[str, float], digits: int = 3) -> str:
    return f"{stat['mean']:.{digits}f} +/- {stat['std']:.{digits}f}"


def write_markdown(results: dict, output: Path) -> None:
    lines = [
        "# Fourier Update Analysis",
        "",
        "This analysis uses task-wise CUB200 classifier snapshots from `deep_representation_case_study_20260430`.",
        "It keeps the legacy flattened FFT metric, but treats it as a sensitivity check because flattened ordering is arbitrary.",
        "",
        "## Aggregate Metrics",
        "",
        "| Method | Delta L2 | Total variation | Flat HF | Class-axis HF | Feature-axis HF | 2D radial HF | 2D centroid | Flat perm-null HF |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        agg = results[method]["aggregate"]
        lines.append(
            "| "
            + " | ".join(
                [
                    LABELS[method],
                    fmt(agg["delta_l2"]),
                    fmt(agg["matrix_total_variation"]),
                    fmt(agg["flat_high_ratio"]),
                    fmt(agg["class_axis_high_ratio"]),
                    fmt(agg["feature_axis_high_ratio"]),
                    fmt(agg["radial_2d_high_ratio"]),
                    fmt(agg["radial_2d_spectral_centroid"]),
                    fmt(agg["flat_high_ratio_perm_mean"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The flattened FFT should not be used alone because shuffling the same update matrix gives a similar high-frequency ratio.",
            "- In this cached CUB snapshot analysis, SSR+KD reduces the task-to-task update norm, but the FFT high-frequency ratios are not reduced. They are higher on the class-axis, feature-axis, and 2D radial spectra.",
            "- Therefore Fourier spectra should be reported as a boundary condition: SSR reorganizes update geometry and can reduce update magnitude, but current data do not support a universal claim of high-frequency suppression.",
            "- If a future checkpoint set shows lower high-frequency power, report it as setting-specific and keep the axis-aware/permutation-null controls.",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_results(results: dict, figure_path: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 10.5,
            "axes.linewidth": 1.8,
            "figure.dpi": 180,
            "savefig.dpi": 320,
        }
    )
    styles = {
        "baseline": {"marker": "o", "ls": "-", "lw": 2.3},
        "biocs": {"marker": "D", "ls": "--", "lw": 2.4},
        "biocs_kd": {"marker": "s", "ls": "-", "lw": 3.0},
    }

    def _style_axes(ax):
        for spine in ax.spines.values():
            spine.set_linewidth(1.75)
            spine.set_color("#222222")
        ax.tick_params(width=1.35, length=4.5, colors="#222222")

    fig = plt.figure(figsize=(14.0, 7.7))
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.22, 1.0, 1.0],
        height_ratios=[1.0, 0.92],
        wspace=0.34,
        hspace=0.42,
    )

    ax = fig.add_subplot(gs[:, 0])
    view_specs = [
        ("flat_high_ratio", "Flat 1D", 3),
        ("class_axis_high_ratio", "Class-axis", 2),
        ("feature_axis_high_ratio", "Feature-axis", 1),
        ("radial_2d_high_ratio", "2D radial", 0),
    ]
    flat_null_lo = min(
        results[m]["aggregate"]["flat_high_ratio_perm_mean"]["mean"]
        - results[m]["aggregate"]["flat_high_ratio_perm_std"]["mean"]
        for m in METHODS
    )
    flat_null_hi = max(
        results[m]["aggregate"]["flat_high_ratio_perm_mean"]["mean"]
        + results[m]["aggregate"]["flat_high_ratio_perm_std"]["mean"]
        for m in METHODS
    )
    for metric, label, yi in view_specs:
        vals = [results[m]["aggregate"][metric]["mean"] for m in METHODS]
        ax.hlines(yi, min(vals), max(vals), color="#CAD2D9", linewidth=2.2, zorder=1)
        if metric == "flat_high_ratio":
            ax.fill_betweenx(
                [yi - 0.30, yi + 0.30],
                flat_null_lo,
                flat_null_hi,
                color="#CBD5E1",
                alpha=0.45,
                zorder=0,
            )
        for method in METHODS:
            val = results[method]["aggregate"][metric]["mean"]
            cfg = styles[method]
            ax.scatter(
                val,
                yi,
                s=88 if method == "biocs_kd" else 72,
                marker=cfg["marker"],
                facecolors="white" if method != "biocs_kd" else COLORS[method],
                edgecolors=COLORS[method],
                linewidths=2.0,
                zorder=4,
            )
        ax.text(0.414, yi, label, va="center", ha="left", fontsize=10.6, color="#333333")

    ax.text(
        0.03,
        0.95,
        "High-frequency ratio above 0.6",
        transform=ax.transAxes,
        va="top",
        fontsize=10.4,
        color="#333333",
        fontweight="bold",
    )
    ax.text(
        0.03,
        0.87,
        "Flat grey band = permutation-null\nrange under random flattening.",
        transform=ax.transAxes,
        va="top",
        fontsize=9.7,
        color="#333333",
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#D5DAE0", linewidth=1.0),
    )
    ax.text(
        0.03,
        0.73,
        "Boundary result: lower update norm does not imply\nlower high-frequency mass in axis-aware views.",
        transform=ax.transAxes,
        va="top",
        fontsize=9.5,
        color="#333333",
        bbox=dict(boxstyle="round,pad=0.22", facecolor="#F8F9FB", edgecolor="#D5DAE0", linewidth=1.0),
    )
    ax.scatter([], [], marker="o", s=68, facecolors="white", edgecolors=COLORS["baseline"], linewidths=1.9, label="Baseline")
    ax.scatter([], [], marker="D", s=68, facecolors="white", edgecolors=COLORS["biocs"], linewidths=1.9, label="SSR")
    ax.scatter([], [], marker="s", s=80, color=COLORS["biocs_kd"], edgecolors=COLORS["biocs_kd"], linewidths=1.2, label="SSR+KD")
    ax.legend(frameon=False, loc="lower left")
    ax.set_xlim(0.14, 0.43)
    ax.set_ylim(-0.6, 3.6)
    ax.set_yticks([])
    ax.set_xlabel("Power ratio above 0.6")
    ax.set_title("(a) High-frequency ratios rise", loc="left", fontweight="bold", fontsize=13.2)
    ax.grid(axis="x", alpha=0.22, linewidth=0.9)
    _style_axes(ax)

    for ax, view, title in [
        (fig.add_subplot(gs[0, 1]), "class_axis", "(b) Class-axis spectrum"),
        (fig.add_subplot(gs[0, 2]), "radial_2d", "(c) 2D radial spectrum"),
    ]:
        for method in METHODS:
            y = np.asarray(results[method]["mean_spectra"][view], dtype=float)
            x = np.linspace(0.0, 1.0, len(y))
            cfg = styles[method]
            ax.plot(
                x,
                y,
                color=COLORS[method],
                linestyle=cfg["ls"],
                lw=cfg["lw"],
                label=LABELS[method],
            )
        ax.axvspan(0.6, 1.0, color="#9B4F48", alpha=0.08, lw=0)
        ax.set_title(title.replace("spectrum", "power"), loc="left", fontweight="bold", fontsize=13.0)
        ax.set_xlabel("Normalized frequency")
        ax.set_ylabel("Mean normalized power")
        ax.grid(alpha=0.24, linewidth=0.9)
        _style_axes(ax)

    ax = fig.add_subplot(gs[1, 1:])
    metric_specs = [
        ("delta_l2", r"$\Delta L_2$", 2),
        ("matrix_total_variation", "Total variation", 1),
        ("radial_2d_spectral_centroid", "2D centroid", 0),
    ]
    ax.axvline(1.0, color="#7B8791", lw=1.1, ls="--", alpha=0.8)
    y_offsets = {"biocs": 0.12, "biocs_kd": -0.12}
    for metric, label, yi in metric_specs:
        ax.hlines(yi, 0.82, 1.58, color="#E1E6EB", linewidth=1.5, zorder=0)
        ax.scatter(1.0, yi, s=68, marker="o", facecolors="white", edgecolors="#6B7780", linewidths=1.6, zorder=2)
        ax.text(0.835, yi, label, va="center", ha="left", fontsize=10.6, color="#333333")
        for method in ["biocs", "biocs_kd"]:
            val = results[method]["aggregate"][metric]["mean"] / (
                results["baseline"]["aggregate"][metric]["mean"] + 1e-12
            )
            marker = styles[method]["marker"]
            y_plot = yi + y_offsets[method]
            ax.plot([1.0, val], [y_plot, y_plot], color=COLORS[method], linewidth=2.0 if method == "biocs" else 2.4)
            ax.scatter(
                val,
                y_plot,
                s=74 if method == "biocs_kd" else 68,
                marker=marker,
                facecolors="white" if method == "biocs" else COLORS[method],
                edgecolors=COLORS[method],
                linewidths=1.8,
                zorder=3,
            )
            ax.text(val + 0.018, y_plot, f"{val:.2f}x", va="center", fontsize=9.4, color="#333333")
    ax.text(
        0.02,
        0.96,
        "Baseline = 1.0  |  SSR lowers update norm only after KD, but spectral centroid and total variation rise.",
        transform=ax.transAxes,
        va="top",
        fontsize=9.2,
        color="#333333",
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#D5DAE0", linewidth=1.0),
    )
    ax.set_xlim(0.82, 1.58)
    ax.set_ylim(-0.5, 2.5)
    ax.set_yticks([])
    ax.set_xlabel("Relative to baseline")
    ax.set_title("(d) Update norm falls, but concentration rises", loc="left", fontweight="bold", fontsize=13.0)
    ax.grid(axis="x", alpha=0.22, linewidth=0.9)
    _style_axes(ax)

    fig.subplots_adjust(left=0.06, right=0.985, bottom=0.08, top=0.965, wspace=0.34, hspace=0.38)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path)
    fig.savefig(figure_path.with_suffix(".png"))
    if figure_path.parent != SCI_FIG_DIR:
        fig.savefig(SCI_FIG_DIR / figure_path.name)
        fig.savefig(SCI_FIG_DIR / f"{figure_path.stem}.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", default=str(DEFAULT_SNAPSHOT))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUT))
    parser.add_argument("--n_perm", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    snapshot = Path(args.snapshot)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    payload = np.load(snapshot)
    rng = np.random.default_rng(args.seed)
    results = {}
    for method in METHODS:
        key = f"{method}_snapshots"
        if key not in payload:
            raise KeyError(f"Missing {key} in {snapshot}")
        results[method] = summarize_method(payload[key], rng, args.n_perm)

    serializable = {
        method: {
            "per_task": data["per_task"],
            "aggregate": data["aggregate"],
            "mean_spectra": {k: v.tolist() for k, v in data["mean_spectra"].items()},
        }
        for method, data in results.items()
    }
    (output_dir / "fourier_update_summary.json").write_text(
        json.dumps(serializable, indent=2), encoding="utf-8"
    )
    write_markdown(serializable, output_dir / "summary.md")
    plot_results(serializable, FIG_DIR / "fig_fourier_update_analysis.pdf")
    print(f"Wrote Fourier update analysis to {output_dir}")
    print(f"Wrote figure to {FIG_DIR / 'fig_fourier_update_analysis.pdf'}")


if __name__ == "__main__":
    main()
