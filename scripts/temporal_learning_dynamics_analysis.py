#!/usr/bin/env python3
"""Temporal learning-dynamics diagnostics for SSR.

The axis-aware Fourier audit in ``fourier_update_analysis.py`` asks whether a
single task-to-task update matrix is internally fragmented along class or
feature axes. This script asks the complementary temporal question: as tasks
advance, are the updates themselves smoother, more autocorrelated, and more
low-frequency?

Using cached CUB200 classifier snapshots, it constructs task-indexed series
from W_t and Delta W_t, estimates autocorrelation functions, and obtains power
spectra from the autocorrelations in the Wiener-Khinchin sense. Because the
available snapshots are task-level only, conclusions should be stated as a
task-level temporal audit; epoch/step-level claims require denser checkpoints.
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
DEFAULT_OUT = ROOT / "results" / "temporal_learning_dynamics_20260514"
FIG_DIR = Path(os.environ.get("SSR_LEGACY_FIGURE_DIR", ROOT / "figures" / "legacy"))
SCI_FIG_DIR = Path(os.environ.get("SSR_FIGURE_DIR", ROOT / "figures"))
SCI_FIG_DIR.mkdir(parents=True, exist_ok=True)

METHODS = ["baseline", "kd", "biocs", "biocs_kd"]
LABELS = {"baseline": "Baseline", "kd": "KD", "biocs": "SSR", "biocs_kd": "SSR+KD"}
COLORS = {"baseline": "#8A9296", "kd": "#B58A4A", "biocs": "#5C9275", "biocs_kd": "#2F6F73"}


def safe_cosine(a: np.ndarray, b: np.ndarray) -> float:
    av = np.asarray(a, dtype=np.float64).reshape(-1)
    bv = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv) + 1e-12)
    return float(np.dot(av, bv) / denom)


def autocorr_1d(series: np.ndarray) -> np.ndarray:
    """Return normalized non-negative-lag autocorrelation for a scalar series."""

    x = np.asarray(series, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return np.zeros(0, dtype=np.float64)
    x = x - x.mean()
    var = float(np.dot(x, x))
    if var <= 1e-12:
        out = np.zeros(x.size, dtype=np.float64)
        out[0] = 1.0
        return out
    return np.asarray([np.dot(x[: x.size - lag], x[lag:]) / var for lag in range(x.size)])


def vector_lagged_cosine(deltas: np.ndarray) -> np.ndarray:
    """Mean cosine similarity between update vectors separated by each lag."""

    x = np.asarray(deltas, dtype=np.float64)
    if x.ndim < 2:
        x = x.reshape(-1, 1)
    flat = x.reshape(x.shape[0], -1)
    out = []
    for lag in range(flat.shape[0]):
        vals = []
        for i in range(flat.shape[0] - lag):
            vals.append(safe_cosine(flat[i], flat[i + lag]))
        out.append(float(np.mean(vals)) if vals else 0.0)
    return np.asarray(out, dtype=np.float64)


def psd_from_autocorr(acf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Estimate a one-sided power spectrum from a non-negative-lag ACF."""

    r = np.asarray(acf, dtype=np.float64).reshape(-1)
    if r.size == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
    if r.size == 1:
        return np.asarray([0.0]), np.asarray([max(float(r[0]), 0.0)])
    # Mirror the ACF to approximate the even autocorrelation sequence required
    # by Wiener-Khinchin, then keep the one-sided real FFT power.
    even_r = np.concatenate([r, r[-2:0:-1]])
    spec = np.real(np.fft.rfft(even_r))
    power = np.maximum(spec, 0.0)
    freqs = np.fft.rfftfreq(even_r.size, d=1.0)
    if freqs.max() > 0:
        freqs = freqs / freqs.max()
    return freqs, power


def band_metrics(power: np.ndarray, freqs: np.ndarray, low_cut: float = 0.25, high_cut: float = 0.50) -> dict:
    p = np.asarray(power, dtype=np.float64).reshape(-1)
    f = np.asarray(freqs, dtype=np.float64).reshape(-1)
    if p.size == 0:
        return {
            "low_ratio": 0.0,
            "high_ratio": 0.0,
            "centroid": 0.0,
            "total_power": 0.0,
            "absolute_high_power": 0.0,
        }
    total = float(p.sum() + 1e-12)
    pn = p / total
    return {
        "low_ratio": float(p[f <= low_cut].sum() / total),
        "high_ratio": float(p[f >= high_cut].sum() / total),
        "centroid": float((f * pn).sum()),
        "total_power": float(total),
        "absolute_high_power": float(p[f >= high_cut].sum()),
    }


def scalar_series_metrics(series: np.ndarray) -> dict:
    x = np.asarray(series, dtype=np.float64).reshape(-1)
    acf = autocorr_1d(x)
    freqs, power = psd_from_autocorr(acf)
    tv = float(np.abs(np.diff(x)).sum() / (np.abs(x).sum() + 1e-12)) if x.size > 1 else 0.0
    lag1 = float(acf[1]) if acf.size > 1 else 0.0
    decay_lag = next((int(i) for i, v in enumerate(acf) if i > 0 and v < math.exp(-1)), int(acf.size - 1))
    out = {
        "mean": float(x.mean()) if x.size else 0.0,
        "std": float(x.std()) if x.size else 0.0,
        "cv": float(x.std() / (abs(x.mean()) + 1e-12)) if x.size else 0.0,
        "temporal_total_variation": tv,
        "lag1_autocorr": lag1,
        "decay_lag_below_1_over_e": decay_lag,
        "acf": acf.tolist(),
        "psd_freqs": freqs.tolist(),
        "psd_power": power.tolist(),
    }
    out.update({f"psd_{k}": v for k, v in band_metrics(power, freqs).items()})
    return out


def vector_autocorr_metrics(deltas: np.ndarray) -> dict:
    acf = vector_lagged_cosine(deltas)
    freqs, power = psd_from_autocorr(acf)
    out = {
        "lag1_update_cosine": float(acf[1]) if acf.size > 1 else 0.0,
        "lag2_update_cosine": float(acf[2]) if acf.size > 2 else 0.0,
        "mean_positive_lag_cosine": float(acf[1:].mean()) if acf.size > 1 else 0.0,
        "acf": acf.tolist(),
        "psd_freqs": freqs.tolist(),
        "psd_power": power.tolist(),
    }
    out.update({f"psd_{k}": v for k, v in band_metrics(power, freqs).items()})
    return out


def summarize_method(snapshots: np.ndarray) -> dict:
    w = np.asarray(snapshots, dtype=np.float64)
    deltas = np.diff(w, axis=0)
    update_norm = np.linalg.norm(deltas.reshape(deltas.shape[0], -1), axis=1)
    weight_cosine_distance = np.asarray(
        [1.0 - safe_cosine(w[i], w[i - 1]) for i in range(1, w.shape[0])], dtype=np.float64
    )
    update_direction_cosine = np.asarray(
        [safe_cosine(deltas[i], deltas[i - 1]) for i in range(1, deltas.shape[0])], dtype=np.float64
    )
    update_direction_change = 1.0 - update_direction_cosine
    return {
        "series": {
            "update_norm": update_norm.tolist(),
            "weight_cosine_distance": weight_cosine_distance.tolist(),
            "update_direction_cosine": update_direction_cosine.tolist(),
            "update_direction_change": update_direction_change.tolist(),
        },
        "update_norm": scalar_series_metrics(update_norm),
        "weight_cosine_distance": scalar_series_metrics(weight_cosine_distance),
        "update_direction_change": scalar_series_metrics(update_direction_change),
        "update_vector_autocorr": vector_autocorr_metrics(deltas),
    }


def fmt(x: float, digits: int = 3) -> str:
    return f"{x:.{digits}f}"


def write_markdown(results: dict, output: Path) -> None:
    lines = [
        "# Temporal Learning-Dynamics Wiener-Khinchin Audit",
        "",
        "This audit answers a different question from the matrix-internal FFT analysis.",
        "The previous class-axis, feature-axis, and 2D FFT diagnostics analyze whether one task-transition update matrix is internally high-frequency.",
        "Here we construct task-indexed time series from cached CUB200 classifier snapshots, estimate autocorrelation functions, and obtain PSDs from those autocorrelations following the Wiener-Khinchin relation.",
        "",
        "Available checkpoints are task-level snapshots only, so the conclusion is task-level. Epoch-level or step-level smoothness requires denser training checkpoints.",
        "",
        "## Main temporal metrics",
        "",
        "| Method | mean ||Delta W|| | norm TV | norm lag-1 ACF | norm PSD low | norm PSD high | update lag-1 cosine | update ACF PSD low | update ACF PSD high |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        r = results[method]
        lines.append(
            "| "
            + " | ".join(
                [
                    LABELS[method],
                    fmt(r["update_norm"]["mean"]),
                    fmt(r["update_norm"]["temporal_total_variation"]),
                    fmt(r["update_norm"]["lag1_autocorr"]),
                    fmt(r["update_norm"]["psd_low_ratio"]),
                    fmt(r["update_norm"]["psd_high_ratio"]),
                    fmt(r["update_vector_autocorr"]["lag1_update_cosine"]),
                    fmt(r["update_vector_autocorr"]["psd_low_ratio"]),
                    fmt(r["update_vector_autocorr"]["psd_high_ratio"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The time-series audit is the correct analysis for claims about learning dynamics being smoother or lower-frequency over task time.",
            "- The KD-only control tests whether temporal smoothing is explained by distillation alone.",
            "- On the cached CUB200 task snapshots, SSR+KD has the smallest mean task-to-task update norm.",
            "- SSR+KD also has the highest lag-1 update-vector autocorrelation, indicating more directionally persistent task-to-task updates.",
            "- The scalar update-norm PSD should be interpreted cautiously because only 19 task transitions are available; denser epoch/step checkpoints would make this test stronger.",
            "- The matrix-internal FFT remains useful as a boundary analysis, but it should not be used as the primary test of temporal smoothness.",
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
        "baseline": {"marker": "o", "ls": "-", "lw": 2.5},
        "kd": {"marker": "s", "ls": "--", "lw": 2.4},
        "biocs": {"marker": "^", "ls": "-.", "lw": 2.4},
        "biocs_kd": {"marker": "D", "ls": "-", "lw": 3.2},
    }
    summary = {}
    for method in METHODS:
        summary[method] = {
            "mean_norm": float(np.mean(results[method]["series"]["update_norm"])),
            "lag1": float(results[method]["update_vector_autocorr"]["acf"][1]),
            "low": float(results[method]["update_vector_autocorr"]["psd_low_ratio"]),
            "high": float(results[method]["update_vector_autocorr"]["psd_high_ratio"]),
        }

    def _style_axes(ax):
        for spine in ax.spines.values():
            spine.set_linewidth(1.75)
            spine.set_color("#222222")
        ax.tick_params(width=1.35, length=4.5, colors="#222222")

    fig = plt.figure(figsize=(14.2, 7.6))
    gs = fig.add_gridspec(
        3,
        2,
        width_ratios=[1.58, 1.0],
        height_ratios=[1.0, 1.0, 0.92],
        wspace=0.30,
        hspace=0.42,
    )

    ax = fig.add_subplot(gs[:, 0])
    end_offsets = {"baseline": 0.10, "kd": -0.06, "biocs": 0.22, "biocs_kd": -0.16}
    for method in METHODS:
        y = np.asarray(results[method]["series"]["update_norm"], dtype=float)
        x = np.arange(1, len(y) + 1)
        cfg = styles[method]
        ax.plot(
            x,
            y,
            marker=cfg["marker"],
            linestyle=cfg["ls"],
            lw=cfg["lw"],
            ms=5.5 if method == "biocs_kd" else 4.8,
            color=COLORS[method],
            markeredgecolor="#2D3436",
            markeredgewidth=1.0,
            zorder=12 if method == "biocs_kd" else 5,
            label=LABELS[method],
        )
        ax.text(
            x[-1] + 0.18,
            y[-1] + end_offsets[method],
            f"{LABELS[method]}",
            color=COLORS[method],
            fontsize=10.5,
            va="center",
            fontweight="bold" if method == "biocs_kd" else None,
        )

    ax.text(
        0.03,
        0.96,
        (
            "Mean $||\\Delta W_t||_F$:\n"
            f"Baseline {summary['baseline']['mean_norm']:.2f}  |  "
            f"KD {summary['kd']['mean_norm']:.2f}  |  "
            f"SSR+KD {summary['biocs_kd']['mean_norm']:.2f}"
        ),
        transform=ax.transAxes,
        va="top",
        fontsize=10.0,
        color="#333333",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#D5DAE0", linewidth=1.2),
    )
    ax.text(
        0.03,
        0.84,
        "KD already contracts task-to-task updates;\nSSR+KD trims them further without changing the task schedule.",
        transform=ax.transAxes,
        va="top",
        fontsize=9.7,
        color="#333333",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#F8F9FB", edgecolor="#D5DAE0", linewidth=1.0),
    )
    ax.set_title("(a) Update magnitudes tighten", loc="left", fontweight="bold", fontsize=13.4)
    ax.set_xlabel("Task transition")
    ax.set_ylabel(r"$||\Delta W_t||_F$")
    ax.set_xlim(0.8, 20.3)
    ax.grid(alpha=0.24, linewidth=0.9)
    ax.legend(frameon=False, loc="lower right")
    _style_axes(ax)

    ax = fig.add_subplot(gs[0, 1])
    for method in METHODS:
        y = np.asarray(results[method]["update_vector_autocorr"]["acf"], dtype=float)[1:]
        x = np.arange(1, len(y) + 1)
        cfg = styles[method]
        ax.plot(
            x,
            y,
            marker=cfg["marker"],
            linestyle=cfg["ls"],
            lw=cfg["lw"],
            ms=4.6 if method == "biocs_kd" else 4.1,
            color=COLORS[method],
            markeredgecolor="#2D3436",
            markeredgewidth=0.9,
            zorder=12 if method == "biocs_kd" else 5,
        )
    ax.axhline(0.0, color="#2D3436", lw=1.0, alpha=0.42)
    ax.axvspan(0.8, 1.2, color="#CBD5E1", alpha=0.22, lw=0)
    ax.text(
        0.04,
        0.94,
        f"Lag-1 persistence: KD {summary['kd']['lag1']:.3f} | SSR+KD {summary['biocs_kd']['lag1']:.3f}",
        transform=ax.transAxes,
        va="top",
        fontsize=9.5,
        color="#333333",
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#D5DAE0", linewidth=1.0),
    )
    ax.set_title("(b) Direction persistence", loc="left", fontweight="bold", fontsize=13.2)
    ax.set_xlabel("Lag in task transitions")
    ax.set_ylabel(r"mean cosine $R_\Delta(d)$")
    ax.set_xlim(0.9, len(y) + 0.35)
    ax.grid(alpha=0.24, linewidth=0.9)
    _style_axes(ax)

    ax = fig.add_subplot(gs[1, 1])
    for method in METHODS:
        freqs = np.asarray(results[method]["update_vector_autocorr"]["psd_freqs"], dtype=float)
        power = np.asarray(results[method]["update_vector_autocorr"]["psd_power"], dtype=float)
        power = power / (power.sum() + 1e-12)
        cfg = styles[method]
        ax.plot(
            freqs,
            power,
            lw=cfg["lw"],
            linestyle=cfg["ls"],
            color=COLORS[method],
            zorder=12 if method == "biocs_kd" else 5,
        )
    ax.axvspan(0.0, 0.25, color="#5C9275", alpha=0.10, lw=0)
    ax.axvspan(0.5, 1.0, color="#9B4F48", alpha=0.08, lw=0)
    ax.text(
        0.04,
        0.94,
        f"High-band mass: KD {summary['kd']['high']:.3f} | SSR+KD {summary['biocs_kd']['high']:.3f}",
        transform=ax.transAxes,
        va="top",
        fontsize=9.5,
        color="#333333",
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#D5DAE0", linewidth=1.0),
    )
    ax.set_title("(c) PSD moves away from the high band", loc="left", fontweight="bold", fontsize=13.0)
    ax.set_xlabel("Normalized temporal frequency")
    ax.set_ylabel("Normalized power")
    ax.grid(alpha=0.24, linewidth=0.9)
    _style_axes(ax)

    ax = fig.add_subplot(gs[2, 1])
    order = ["baseline", "kd", "biocs", "biocs_kd"]
    y_pos = np.arange(len(order))
    for yi, method in enumerate(order):
        low = summary[method]["low"]
        high = summary[method]["high"]
        color = COLORS[method]
        ax.hlines(yi, low, high, color=color, linewidth=2.6 if method == "biocs_kd" else 2.0, zorder=2)
        ax.scatter(low, yi, s=68, marker="o", facecolors="white", edgecolors=color, linewidths=1.8, zorder=3)
        ax.scatter(high, yi, s=72, marker="s", color=color, edgecolors="#2D3436", linewidths=0.8, zorder=4)
        ax.text(high + 0.012, yi, f"{high:.3f}", va="center", fontsize=9.5, color="#333333")
    ax.text(
        0.02,
        0.95,
        "○ low band 0–0.25   ■ high band 0.5–1.0",
        transform=ax.transAxes,
        va="top",
        fontsize=8.9,
        color="#333333",
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels([LABELS[m] for m in order])
    ax.invert_yaxis()
    ax.set_xlim(0.18, 0.64)
    ax.set_xlabel("PSD band ratio")
    ax.set_title("(d) Low/high PSD bands", loc="left", fontweight="bold", fontsize=13.0)
    ax.grid(axis="x", alpha=0.24, linewidth=0.9)
    _style_axes(ax)

    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.08, top=0.965, wspace=0.30, hspace=0.36)
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
    args = parser.parse_args()

    snapshot = Path(args.snapshot)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    payload = np.load(snapshot)
    results = {}
    for method in METHODS:
        key = f"{method}_snapshots"
        if key not in payload:
            raise KeyError(f"Missing {key} in {snapshot}")
        results[method] = summarize_method(payload[key])

    (output_dir / "temporal_learning_dynamics_summary.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    write_markdown(results, output_dir / "summary.md")
    plot_results(results, FIG_DIR / "fig_temporal_learning_dynamics.pdf")
    print(f"Wrote temporal learning-dynamics audit to {output_dir}")
    print(f"Wrote figure to {FIG_DIR / 'fig_temporal_learning_dynamics.pdf'}")


if __name__ == "__main__":
    main()
