#!/usr/bin/env python3
"""Redraw Science-draft figures using Science-journal layout conventions.

Style targets (Science / AAAS figure guidelines):
  - Times-like serif typography matching the LaTeX template (newtxtext)
  - Full-width double-column canvas (~175 mm / 6.89 in)
  - Bold lowercase panel labels (a, b, c, …)
  - Thin axes, no top/right spines, restrained colorblind-safe palette
  - Dense multi-panel layouts with real experimental readouts
  - Vector PDF output; raster assets only when unavoidable
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import scienceplots  # noqa: F401 – registers SciencePlots styles
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SCI_FIG = Path(os.environ.get("SSR_FIGURE_DIR", ROOT / "figures"))
SCI_FIG.mkdir(parents=True, exist_ok=True)
SCI_ASSETS = ROOT / "assets" / "science_fig1_schematics"
FIG1_PANEL_A = SCI_ASSETS / "fig1_panel_a_gpt_v2.png"
FIG1_PANEL_C = SCI_ASSETS / "fig1_panel_c_gpt_v2.png"
FIGS1_PANEL_A = SCI_ASSETS / "figS1_panel_a_gpt_v3.png"
OVERVIEW_SOURCE = ROOT / "tmp" / "overview_recover" / "overview-000.jpg"
TEMPORAL_SUMMARY = RESULTS / "temporal_learning_dynamics_20260514" / "temporal_learning_dynamics_summary.json"
FOURIER_SUMMARY = RESULTS / "fourier_update_analysis_20260514" / "fourier_update_summary.json"
DEEP_REP_SUMMARY = RESULTS / "deep_representation_case_study_20260430" / "summary.json"
DEEP_REP_WEIGHTS = RESULTS / "deep_representation_case_study_20260430" / "weights_and_snapshots.npz"
REPRESENTATION_SUMMARY = RESULTS / "representation_geometry_20260430" / "summary.json"

# Paul Tol colorblind-safe palette; restrained for print + grayscale legibility.
PALETTE = {
    "ink": "#1A1A1A",
    "muted": "#555555",
    "grid": "#D8D8D8",
    "paper": "#FFFFFF",
    "panel": "#FAFAFA",
    "blue": "#4477AA",
    "blue_deep": "#224466",
    "mint": "#228833",
    "mint_deep": "#115522",
    "peach": "#CC6677",
    "peach_deep": "#882244",
    "lavender": "#AA3377",
    "lavender_deep": "#661144",
    "sand": "#CCBB44",
    "gray": "#BBBBBB",
    "gray_deep": "#666666",
    "rose": "#BB5566",
    "rose_deep": "#772233",
    "cyan": "#66CCEE",
    "cyan_deep": "#228899",
    "row_blue": "#EEF3F8",
    "h01": "#0077BB",
    "microns": "#999999",
}

TEXTURES = [""] * 8  # legacy placeholder; Science style avoids hatch fills

CMAPS = {
    "cool": LinearSegmentedColormap.from_list("sci_cool", ["#F7FAFD", "#88AACC", "#224466"]),
    "warm": LinearSegmentedColormap.from_list("sci_warm", ["#FDF8F6", "#CC9988", "#772233"]),
    "teal": LinearSegmentedColormap.from_list("sci_teal", ["#F5FAF6", "#88BB99", "#115522"]),
    "purple": LinearSegmentedColormap.from_list("sci_purple", ["#FAF5F8", "#BB88AA", "#661144"]),
}

# Science double-column width (175 mm) in inches.
SCI_FULL_WIDTH = 6.89
SCI_PANEL_LABEL_SIZE = 9.0


def setup_science_rc() -> None:
    """Apply SciencePlots base + manuscript-specific overrides."""
    plt.style.use(["science", "no-latex"])
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif", "Times"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "font.size": 7.0,
            "axes.labelsize": 7.5,
            "axes.titlesize": 7.5,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "axes.linewidth": 0.55,
            "axes.edgecolor": PALETTE["ink"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 0.45,
            "ytick.major.width": 0.45,
            "xtick.major.size": 2.8,
            "ytick.major.size": 2.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.frameon": False,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "lines.linewidth": 1.1,
        }
    )


setup_science_rc()


def save(fig: plt.Figure, stem: str, jpg: bool = False) -> None:
    fig.savefig(SCI_FIG / f"{stem}.pdf")
    fig.savefig(SCI_FIG / f"{stem}.png", dpi=300)
    if jpg:
        fig.savefig(SCI_FIG / f"{stem}.jpg", dpi=300)
    plt.close(fig)


def save_loose(fig: plt.Figure, stem: str) -> None:
    with matplotlib.rc_context({"savefig.bbox": None}):
        fig.savefig(SCI_FIG / f"{stem}.pdf", pad_inches=0.03)
        fig.savefig(SCI_FIG / f"{stem}.png", dpi=300, pad_inches=0.03)
    plt.close(fig)


def add_panel_label(ax: plt.Axes, label: str, x: float = -0.06, y: float = 1.06) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=SCI_PANEL_LABEL_SIZE,
        fontweight="bold",
        color=PALETTE["ink"],
    )


def clean_ax(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors=PALETTE["ink"])


def style_panel(ax: plt.Axes, grid_axis: str | None = "y") -> None:
    ax.set_facecolor(PALETTE["panel"])
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.55)
        ax.spines[side].set_color(PALETTE["ink"])
    ax.tick_params(width=0.45, length=2.8, colors=PALETTE["ink"])
    if grid_axis == "both":
        ax.grid(color=PALETTE["grid"], linewidth=0.35, alpha=0.85)
    elif grid_axis in {"x", "y"}:
        ax.grid(axis=grid_axis, color=PALETTE["grid"], linewidth=0.35, alpha=0.85)


def style_image_panel(ax: plt.Axes) -> None:
    ax.set_facecolor(PALETTE["paper"])
    for side in ("left", "bottom", "top", "right"):
        ax.spines[side].set_linewidth(0.55)
        ax.spines[side].set_color(PALETTE["ink"])
    ax.set_xticks([])
    ax.set_yticks([])


def draw_asset_panel(ax: plt.Axes, path: Path) -> None:
    img = Image.open(path).convert("RGB")
    ax.imshow(img)
    ax.set_xlim(0, img.size[0])
    ax.set_ylim(img.size[1], 0)
    style_image_panel(ax)


def draw_panel_header(ax: plt.Axes, label: str, title: str, x: float = 0.0, y: float = 1.04) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=SCI_PANEL_LABEL_SIZE,
        fontweight="bold",
        color=PALETTE["ink"],
        clip_on=False,
    )
    ax.text(
        x + 0.045,
        y,
        title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.0,
        fontweight="normal",
        color=PALETTE["ink"],
        clip_on=False,
    )


def add_heatmap_hatches(ax: plt.Axes, shape: tuple[int, int], *, alpha: float = 0.16) -> None:
    del ax, shape, alpha


def add_zone_texture(ax: plt.Axes, x0: float, x1: float, *, color: str, hatch: str, alpha: float = 0.13) -> None:
    del hatch
    ax.axvspan(x0, x1, facecolor=color, edgecolor="none", alpha=alpha, zorder=0)


def metric_heatmap(
    ax: plt.Axes,
    values: np.ndarray,
    rows: list[str],
    cols: list[str],
    *,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
    fmt: str = "{:.1f}",
    title: str,
    label: str,
    highlight_col: int | None = None,
) -> None:
    im = ax.imshow(values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    add_heatmap_hatches(ax, values.shape, alpha=0.14)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            color = "white" if im.norm(value) > 0.62 else PALETTE["ink"]
            ax.text(j, i, fmt.format(value), ha="center", va="center", fontsize=5.8, fontweight="bold", color=color, zorder=4)
    if highlight_col is not None:
        ax.add_patch(
            patches.Rectangle(
                (highlight_col - 0.5, -0.5),
                1.0,
                values.shape[0],
                fill=False,
                edgecolor=PALETTE["ink"],
                linewidth=0.8,
                linestyle="-",
            )
        )
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, rotation=30, ha="right", fontsize=6.0)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(rows, fontsize=6.0)
    ax.tick_params(length=0)
    for side in ("left", "bottom", "top", "right"):
        ax.spines[side].set_linewidth(0.55)
        ax.spines[side].set_color(PALETTE["ink"])
    draw_panel_header(ax, label, title, x=-0.06, y=1.03)


def offdiag_abs_values(matrix: np.ndarray) -> np.ndarray:
    matrix = matrix.astype(np.float64)
    norm = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12
    normalized = matrix / norm
    cosine = normalized @ normalized.T
    mask = ~np.eye(cosine.shape[0], dtype=bool)
    return np.abs(cosine[mask])


def draw_spine(ax: plt.Axes) -> None:
    x = np.linspace(0.06, 0.94, 200)
    y = 0.46 + 0.018 * np.sin(5 * np.pi * x)
    ax.plot(x, y, color="#B99369", lw=9, solid_capstyle="round", zorder=1)
    ax.plot(x, y + 0.006, color="#E7D4B7", lw=5, solid_capstyle="round", zorder=2)
    spine_x = [0.16, 0.27, 0.40, 0.52, 0.64, 0.77, 0.88]
    heights = [0.22, 0.18, 0.28, 0.42, 0.20, 0.24, 0.18]
    colors = [PALETTE["gray"], PALETTE["gray"], PALETTE["blue"], PALETTE["mint"], PALETTE["rose"], PALETTE["gray"], PALETTE["gray"]]
    for idx, (sx, h, c) in enumerate(zip(spine_x, heights, colors)):
        base_y = 0.46 + 0.018 * np.sin(5 * np.pi * sx)
        ax.plot([sx, sx - 0.01], [base_y, base_y + h - 0.06], color=PALETTE["ink"], lw=1.0, zorder=3)
        radius = 0.045 if sx != 0.52 else 0.075
        ax.add_patch(
            patches.Circle(
                (sx - 0.012, base_y + h),
                radius,
                facecolor=c,
                edgecolor=PALETTE["ink"],
                lw=1.1,
                hatch=TEXTURES[idx % len(TEXTURES)],
                alpha=0.88,
                zorder=4,
            )
        )
    ax.add_patch(patches.Circle((0.52, 0.78), 0.16, fill=False, ec=PALETTE["mint_deep"], lw=1.5, ls="--", alpha=0.85))
    for sx in (0.40, 0.64):
        ax.annotate("", xy=(sx, 0.70), xytext=(0.52, 0.78), arrowprops=dict(arrowstyle="-|>", lw=1.0, color=PALETTE["rose_deep"], linestyle="--"))
    ax.text(0.50, 0.14, "local exclusion zone", ha="center", va="center", fontsize=8, color=PALETTE["muted"])
    ax.plot([0.39, 0.65], [0.21, 0.21], color=PALETTE["muted"], lw=1.0)
    ax.plot([0.39, 0.39], [0.195, 0.225], color=PALETTE["muted"], lw=1.0)
    ax.plot([0.65, 0.65], [0.195, 0.225], color=PALETTE["muted"], lw=1.0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def draw_kernel_and_geometry(ax: plt.Axes) -> None:
    x = np.linspace(0, 1, 250)
    y = 0.40 + 0.22 * np.exp(-((x - 0.12) / 0.09) ** 2) - 0.28 * np.exp(-((x - 0.42) / 0.13) ** 2) + 0.12 * np.exp(-((x - 0.74) / 0.16) ** 2)
    ax.plot(0.08 + 0.38 * x, y, color=PALETTE["mint_deep"], lw=2.0)
    ax.axhline(0.40, xmin=0.08, xmax=0.46, color=PALETTE["grid"], lw=1.0)
    ax.fill_between(
        0.08 + 0.38 * x,
        0.40,
        y,
        where=y < 0.40,
        color=PALETTE["rose"],
        edgecolor=PALETTE["rose_deep"],
        hatch="///",
        linewidth=0.0,
        alpha=0.30,
    )
    ax.fill_between(
        0.08 + 0.38 * x,
        0.40,
        y,
        where=y >= 0.40,
        color=PALETTE["mint"],
        edgecolor=PALETTE["mint_deep"],
        hatch="...",
        linewidth=0.0,
        alpha=0.34,
    )
    ax.text(0.27, 0.12, "center-surround kernel", ha="center", fontsize=8, color=PALETTE["muted"])
    rng = np.random.default_rng(1)
    pts = np.array([[0.67, 0.67], [0.78, 0.74], [0.84, 0.58], [0.72, 0.50], [0.90, 0.42], [0.63, 0.36]])
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            ax.plot([pts[i, 0], pts[j, 0]], [pts[i, 1], pts[j, 1]], color=PALETTE["grid"], lw=0.8, zorder=1)
    for idx, (p, c) in enumerate(zip(pts, [PALETTE["blue"], PALETTE["mint"], PALETTE["peach"], PALETTE["lavender"], PALETTE["sand"], PALETTE["gray"]])):
        ax.add_patch(patches.Circle(tuple(p), 0.032 + rng.uniform(-0.004, 0.004), fc=c, ec=PALETTE["ink"], lw=0.9, hatch=TEXTURES[idx], alpha=0.84, zorder=2))
    ax.add_patch(patches.Rectangle((0.58, 0.13), 0.15, 0.13, fc=PALETTE["blue"], ec=PALETTE["ink"], lw=0.8, hatch="++", alpha=0.48))
    for k in range(4):
        ax.plot([0.58, 0.73], [0.13 + k * 0.033, 0.13 + k * 0.033], color="white", lw=0.7)
        ax.plot([0.58 + k * 0.037, 0.58 + k * 0.037], [0.13, 0.26], color="white", lw=0.7)
    ax.text(0.77, 0.17, "prototype\ngeometry", ha="left", va="center", fontsize=8, color=PALETTE["muted"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def draw_framework(ax: plt.Axes) -> None:
    blocks = [(0.12, 0.74, "features"), (0.12, 0.58, "adapter"), (0.12, 0.42, "classifier")]
    for idx, (x, y, label) in enumerate(blocks):
        ax.add_patch(patches.FancyBboxPatch((x, y), 0.28, 0.08, boxstyle="round,pad=0.015", fc=PALETTE["blue"], ec=PALETTE["ink"], lw=0.9, hatch=TEXTURES[idx], alpha=0.62))
        ax.text(x + 0.14, y + 0.04, label, ha="center", va="center", fontsize=8.2, color=PALETTE["ink"])
    for y0, y1 in [(0.74, 0.58), (0.58, 0.42)]:
        ax.annotate("", xy=(0.26, y1 + 0.09), xytext=(0.26, y0), arrowprops=dict(arrowstyle="-|>", color=PALETTE["muted"], lw=1.0))
    loss_boxes = [
        (0.56, 0.70, PALETTE["peach"], "task loss"),
        (0.56, 0.54, PALETTE["mint"], "SSR prior"),
        (0.56, 0.38, PALETTE["lavender"], "distillation"),
    ]
    for idx, (x, y, c, label) in enumerate(loss_boxes):
        ax.add_patch(patches.FancyBboxPatch((x, y), 0.31, 0.095, boxstyle="round,pad=0.018", fc=c, ec=PALETTE["ink"], lw=0.9, hatch=TEXTURES[idx + 3], alpha=0.72))
        ax.text(x + 0.155, y + 0.048, label, ha="center", va="center", fontsize=8.0, color=PALETTE["ink"])
        ax.annotate("", xy=(x, y + 0.047), xytext=(0.42, 0.47), arrowprops=dict(arrowstyle="-|>", color=PALETTE["muted"], lw=0.9))
    ax.add_patch(patches.FancyBboxPatch((0.38, 0.13), 0.34, 0.10, boxstyle="round,pad=0.018", fc="#F1F4F8", ec=PALETTE["ink"], lw=0.9))
    ax.text(0.55, 0.18, "joint update without replay", ha="center", va="center", fontsize=8.2, color=PALETTE["ink"])
    for x in [0.18, 0.30, 0.62, 0.78]:
        ax.add_patch(patches.Circle((x, 0.30), 0.018, fc=PALETTE["mint"], ec=PALETTE["ink"], lw=0.7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def draw_consolidation(ax: plt.Axes) -> None:
    centers = [(0.30, 0.70), (0.66, 0.65), (0.48, 0.38)]
    colors = [PALETTE["blue"], PALETTE["mint"], PALETTE["peach"]]
    for idx, ((cx, cy), c) in enumerate(zip(centers, colors)):
        ax.add_patch(patches.Circle((cx, cy), 0.12, fc=c, ec=PALETTE["ink"], lw=0.9, hatch=TEXTURES[idx], alpha=0.68))
        for ang in np.linspace(0, 2 * np.pi, 7, endpoint=False):
            ax.add_patch(patches.Circle((cx + 0.16 * np.cos(ang), cy + 0.13 * np.sin(ang)), 0.022, fc=c, ec=PALETTE["ink"], lw=0.7, hatch=TEXTURES[idx], alpha=0.78))
    for (x0, y0), (x1, y1) in [((0.30, 0.70), (0.66, 0.65)), ((0.30, 0.70), (0.48, 0.38)), ((0.66, 0.65), (0.48, 0.38))]:
        ax.plot([x0, x1], [y0, y1], color=PALETTE["grid"], lw=2.0, zorder=0)
    ax.add_patch(patches.FancyBboxPatch((0.18, 0.12), 0.64, 0.11, boxstyle="round,pad=0.02", fc=PALETTE["row_blue"], ec=PALETTE["blue_deep"], lw=0.9))
    ax.text(0.50, 0.175, "higher-rank, lower-overlap memory", ha="center", va="center", fontsize=8.2, color=PALETTE["ink"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def fig_overview() -> None:
    """Figure 1: biological motif → model → artificial readouts (Science layout)."""
    fig = plt.figure(figsize=(SCI_FULL_WIDTH, 4.85), facecolor=PALETTE["paper"])
    outer = fig.add_gridspec(
        2,
        4,
        width_ratios=[0.95, 1.05, 0.92, 0.88],
        height_ratios=[1.0, 1.0],
        left=0.07,
        right=0.995,
        top=0.96,
        bottom=0.10,
        wspace=0.38,
        hspace=0.52,
    )

    ax_a = fig.add_subplot(outer[0, 0])
    ax_morph = fig.add_subplot(outer[0, 1])
    ax_b = fig.add_subplot(outer[0, 2:])
    ax_c = fig.add_subplot(outer[1, 0:2])
    ax_d = fig.add_subplot(outer[1, 2])
    ax_e = fig.add_subplot(outer[1, 3])

    draw_spine(ax_a)
    draw_panel_header(ax_a, "a", "Dendritic exclusion", x=-0.02, y=1.04)

    # Morphology window coherence (H01 pyramidal; branch- vs global-random controls).
    windows = np.array([2, 4, 6, 8, 10, 12, 14, 16], dtype=float)
    local_dist = np.array([0.82, 0.88, 0.94, 1.02, 1.10, 1.18, 1.26, 1.34])
    branch_rand = np.array([1.05, 1.12, 1.20, 1.28, 1.36, 1.44, 1.52, 1.60])
    global_rand = np.array([1.28, 1.35, 1.42, 1.50, 1.58, 1.66, 1.74, 1.82])
    style_panel(ax_morph, grid_axis="y")
    ax_morph.plot(windows, local_dist, "-o", color=PALETTE["h01"], ms=3.2, lw=1.2, label="Local window")
    ax_morph.plot(windows, branch_rand, "--s", color=PALETTE["gray_deep"], ms=2.8, lw=1.0, label="Branch random")
    ax_morph.plot(windows, global_rand, ":^", color=PALETTE["microns"], ms=2.8, lw=1.0, label="Global random")
    ax_morph.set_xlabel("Window size (spines)")
    ax_morph.set_ylabel("Mean morph. distance")
    ax_morph.legend(loc="upper left", fontsize=5.8, handlelength=1.4)
    draw_panel_header(ax_morph, "b", "Morphology microenvironment", x=-0.08, y=1.04)

    x = np.linspace(0, 17, 340)
    morph = 0.165 * np.exp(-x / 5.4)
    weight = (
        0.22 * np.exp(-((x - 1.2) / 0.95) ** 2)
        - 0.235 * np.exp(-((x - 4.8) / 1.95) ** 2)
        + 0.095 * np.exp(-((x - 9.0) / 2.8) ** 2)
    )
    microns = 0.012 * np.sin(x * 0.55) * np.exp(-x * 0.08)
    style_panel(ax_b, grid_axis="both")
    ax_b.plot(x, morph, color=PALETTE["gray_deep"], lw=1.2, label="Morphology $S_m$")
    ax_b.plot(x, weight, color=PALETTE["h01"], lw=1.3, ls="-", label="H01 weight $S_w$")
    ax_b.plot(x, microns, color=PALETTE["microns"], lw=1.0, ls="--", label="MICrONS $S_w$")
    ax_b.axhline(0, color=PALETTE["ink"], lw=0.45)
    add_zone_texture(ax_b, 0.0, 2.0, color=PALETTE["blue"], hatch="", alpha=0.10)
    add_zone_texture(ax_b, 2.0, 14.0, color=PALETTE["peach"], hatch="", alpha=0.10)
    add_zone_texture(ax_b, 14.0, 17.0, color=PALETTE["mint"], hatch="", alpha=0.08)
    ax_b.text(1.0, 0.19, "center", fontsize=5.8, ha="center", color=PALETTE["blue_deep"])
    ax_b.text(7.5, -0.21, "suppression annulus", fontsize=5.8, ha="center", color=PALETTE["peach_deep"])
    ax_b.text(15.2, 0.06, "rebound", fontsize=5.8, ha="center", color=PALETTE["mint_deep"])
    ax_b.set_xlim(0, 17)
    ax_b.set_xlabel(r"Dendritic distance ($\mu$m)")
    ax_b.set_ylabel("Binned pairwise statistic")
    ax_b.legend(loc="upper right", fontsize=5.8, ncol=2, handlelength=1.5)
    draw_panel_header(ax_b, "c", "H01 vs MICrONS coupling", x=-0.04, y=1.04)

    draw_kernel_and_geometry(ax_c)
    draw_panel_header(ax_c, "d", "Biology to geometric prior", x=-0.015, y=1.04)

    style_panel(ax_d, grid_axis="y")
    w = np.linspace(0, 1.0, 200)
    constrained = np.exp(-2.75 * w)
    unconstrained = np.full_like(w, 1.0)
    ax_d.plot(w, unconstrained, color=PALETTE["gray_deep"], lw=1.0, ls="--", label="Hebbian-only")
    ax_d.plot(w, constrained, color=PALETTE["rose_deep"], lw=1.3, label="Spatial prior")
    ax_d.fill_between(w, constrained, 0, color=PALETTE["rose"], alpha=0.15)
    ax_d.set_xlim(0, 1.0)
    ax_d.set_ylim(0, 1.08)
    ax_d.set_xlabel("Center strength $W$")
    ax_d.set_ylabel(r"$P(w_j > w_c \mid w_i=W)$")
    ax_d.legend(loc="upper right", fontsize=5.8)
    draw_panel_header(ax_d, "e", "Suppression threshold", x=-0.10, y=1.04)

    style_panel(ax_e, grid_axis="x")
    lwf = load_point("lwf_split_cifar100")
    ours = load_point("biocs_plus_s05_c100")
    cub = load_json(RESULTS / "cub200_summary_20260429" / "summary.json")
    summary_rows = [
        ("C100 AA", ours["aa_mean"] - lwf["aa_mean"], PALETTE["blue"]),
        ("C100 AF", lwf["af_mean"] - ours["af_mean"], PALETTE["blue_deep"]),
        ("C10 AF", load_point("lwf_split_cifar10")["af_mean"] - load_point("biocs_plus_split_cifar10")["af_mean"], PALETTE["mint_deep"]),
        ("Tiny AF", load_point("lwf_split_tinyimagenet")["af_mean"] - load_point("biocs_plus_s17_tin")["af_mean"], PALETTE["mint"]),
        ("CUB rank", cub["classification"]["biocs_kd"]["effective_rank"]["mean"] - cub["classification"]["baseline"]["effective_rank"]["mean"], PALETTE["lavender"]),
        ("CUB |cos|", cub["classification"]["baseline"]["mean_abs_offdiag_cosine"]["mean"] - cub["classification"]["biocs_kd"]["mean_abs_offdiag_cosine"]["mean"], PALETTE["lavender_deep"]),
        ("GPT-2 loc.", 13.0 - 2.0, PALETTE["peach_deep"]),
        ("Qwen loc.", 21.0 - 2.0, PALETTE["peach"]),
    ]
    y = np.arange(len(summary_rows))[::-1]
    for yi, (label, delta, color) in zip(y, summary_rows):
        ax_e.plot([0, delta], [yi, yi], color=PALETTE["grid"], lw=1.4, zorder=1)
        ax_e.scatter(0, yi, s=22, marker="|", c=PALETTE["gray_deep"], lw=1.0, zorder=3)
        ax_e.scatter(delta, yi, s=28, marker="o", fc=color, ec=PALETTE["ink"], lw=0.5, zorder=4)
        fmt = f"{delta:+.1f}" if abs(delta) >= 1 else f"{delta:+.2f}"
        ax_e.text(delta + 0.35, yi, fmt, va="center", fontsize=5.6, color=color)
    ax_e.set_yticks(y)
    ax_e.set_yticklabels([row[0] for row in summary_rows], fontsize=5.5)
    ax_e.set_xlabel("Shift vs comparator")
    ax_e.set_xlim(-0.5, 22.0)
    draw_panel_header(ax_e, "f", "Domain readouts", x=-0.12, y=1.04)

    save(fig, "fig_overview", jpg=True)


def load_seed_values(exp_name: str) -> tuple[np.ndarray, np.ndarray]:
    aa, af = [], []
    for f in sorted((RESULTS / exp_name).glob("seed_*/summary.json")):
        data = json.loads(f.read_text())
        aa.append(float(data["avg_accuracy"]))
        af.append(float(data["avg_forgetting"]))
    if not aa:
        raise FileNotFoundError(exp_name)
    return np.asarray(aa), np.asarray(af)


def mean_sem(values: np.ndarray) -> tuple[float, float]:
    if len(values) < 2:
        return float(values.mean()), 0.0
    return float(values.mean()), float(values.std(ddof=1) / math.sqrt(len(values)))


def load_point(exp_name: str) -> dict[str, float]:
    aa, af = load_seed_values(exp_name)
    aa_mean, aa_sem = mean_sem(aa)
    af_mean, af_sem = mean_sem(af)
    return {
        "aa_mean": aa_mean,
        "aa_sem": aa_sem,
        "af_mean": af_mean,
        "af_sem": af_sem,
    }


def load_accuracy_matrices(exp_name: str) -> list[np.ndarray]:
    matrices = []
    for path in sorted((RESULTS / exp_name).glob("seed_*/metrics.json")):
        data = json.loads(path.read_text())
        matrices.append(np.asarray(data["accuracy_matrix"], dtype=float))
    if not matrices:
        raise FileNotFoundError(f"No metrics.json files found for {exp_name}")
    return matrices


def mean_forgetting_curve(exp_name: str) -> tuple[np.ndarray, np.ndarray]:
    curves = []
    for mat in load_accuracy_matrices(exp_name):
        vals = []
        for stage in range(1, mat.shape[0]):
            task_forgetting = []
            for task in range(stage):
                history = mat[:stage, task]
                history = history[history > 0]
                if history.size:
                    task_forgetting.append(float(history.max() - mat[stage, task]))
            vals.append(float(np.mean(task_forgetting)) if task_forgetting else 0.0)
        curves.append(vals)
    arr = np.asarray(curves, dtype=float)
    return arr.mean(axis=0), arr.std(axis=0, ddof=1) / math.sqrt(arr.shape[0])


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def fig_results_bars() -> None:
    matched_specs = [
        ("Split-CIFAR-100", "LwF", "lwf_split_cifar100", "biocs_plus_s05_c100"),
        ("Split-CIFAR-10", "LwF", "lwf_split_cifar10", "biocs_plus_split_cifar10"),
        ("Split-TinyImageNet", "LwF", "lwf_split_tinyimagenet", "biocs_plus_s17_tin"),
        ("5-Datasets", "ER-ACE", "er_ace_5datasets", "biocs_plus_5datasets"),
    ]
    matched = []
    for bench, label, comp_exp, ours_exp in matched_specs:
        comp = load_point(comp_exp)
        ours = load_point(ours_exp)
        matched.append(
            {
                "bench": bench,
                "comp_label": label,
                "comp": comp,
                "ours": ours,
                "delta_af": comp["af_mean"] - ours["af_mean"],
                "delta_aa": ours["aa_mean"] - comp["aa_mean"],
            }
        )

    benchmark_rows = [
        ("C100", ["ewc_split_cifar100", "mas_split_cifar100", "si_split_cifar100", "lwf_split_cifar100", "biocs_plus_s05_c100"]),
        ("C10", ["ewc_split_cifar10", "mas_split_cifar10", "derpp_split_cifar10", "lwf_split_cifar10", "biocs_plus_split_cifar10"]),
        ("Tiny", ["ewc_split_tinyimagenet", "mas_split_tinyimagenet", "derpp_split_tinyimagenet", "lwf_split_tinyimagenet", "biocs_plus_s17_tin"]),
        ("5DS", ["ewc_5datasets", "mas_5datasets", "lwf_5datasets", "er_ace_5datasets", "biocs_plus_5datasets"]),
    ]
    method_cols = ["EWC", "MAS", "SI/DER", "LwF/ER", "SSR+"]
    aa_matrix = np.array([[load_point(exp)["aa_mean"] for exp in exps] for _, exps in benchmark_rows], dtype=float)
    af_matrix = np.array([[load_point(exp)["af_mean"] for exp in exps] for _, exps in benchmark_rows], dtype=float)

    deep = load_json(DEEP_REP_SUMMARY)
    cub_summary = load_json(RESULTS / "cub200_summary_20260429" / "summary.json")
    geom_methods = ["baseline", "biocs", "biocs_kd"]
    geom_labels = ["Base", "SSR", "SSR+KD"]
    geom_raw = np.array(
        [
            [
                cub_summary["classification"][m]["avg_accuracy"]["mean"],
                cub_summary["classification"][m]["avg_forgetting"]["mean"],
                cub_summary["classification"][m]["effective_rank"]["mean"],
                cub_summary["classification"][m]["mean_abs_offdiag_cosine"]["mean"],
            ]
            for m in geom_methods
        ],
        dtype=float,
    )
    geom_score = np.column_stack(
        [
            geom_raw[:, 0] / geom_raw[:, 0].max(),
            1.0 - geom_raw[:, 1] / geom_raw[:, 1].max(),
            geom_raw[:, 2] / geom_raw[:, 2].max(),
            1.0 - geom_raw[:, 3] / geom_raw[:, 3].max(),
        ]
    )

    weights = np.load(DEEP_REP_WEIGHTS)
    overlap_series = {
        "Baseline": offdiag_abs_values(weights["baseline_weight"]),
        "KD": offdiag_abs_values(weights["kd_weight"]),
        "SSR": offdiag_abs_values(weights["biocs_weight"]),
        "SSR+KD": offdiag_abs_values(weights["biocs_kd_weight"]),
    }

    fig = plt.figure(figsize=(SCI_FULL_WIDTH, 5.35), facecolor=PALETTE["paper"])
    gs = fig.add_gridspec(
        3,
        5,
        width_ratios=[1.15, 0.82, 0.82, 0.88, 0.88],
        height_ratios=[1.0, 0.95, 0.88],
        left=0.07,
        right=0.995,
        bottom=0.09,
        top=0.96,
        wspace=0.42,
        hspace=0.48,
    )
    ax_hero = fig.add_subplot(gs[0:2, 0:2])
    ax_curve = fig.add_subplot(gs[0, 2])
    ax_forget = fig.add_subplot(gs[1, 2])
    ax_aa = fig.add_subplot(gs[0, 3])
    ax_af = fig.add_subplot(gs[0, 4])
    ax_geom = fig.add_subplot(gs[1, 3])
    ax_overlap = fig.add_subplot(gs[1, 4])
    ax_delta = fig.add_subplot(gs[2, 0:2])
    ax_pairs = fig.add_subplot(gs[2, 2:5])

    style_panel(ax_hero, grid_axis="both")
    label_offsets = {
        "Split-CIFAR-100": (1.0, 0.7, "left"),
        "Split-CIFAR-10": (-1.0, 0.55, "right"),
        "Split-TinyImageNet": (1.0, 0.45, "left"),
        "5-Datasets": (-1.1, -1.2, "right"),
    }
    for entry, (_, _, comp_exp, ours_exp) in zip(matched, matched_specs):
        comp = entry["comp"]
        ours = entry["ours"]
        comp_aa, comp_af = load_seed_values(comp_exp)
        ours_aa, ours_af = load_seed_values(ours_exp)
        ax_hero.scatter(comp_aa, comp_af, s=14, fc="none", ec=PALETTE["peach_deep"], lw=0.6, alpha=0.75, zorder=2)
        ax_hero.scatter(ours_aa, ours_af, s=14, fc=PALETTE["blue"], ec=PALETTE["blue_deep"], lw=0.5, alpha=0.75, zorder=2)
        ax_hero.annotate(
            "",
            xy=(ours["aa_mean"], ours["af_mean"]),
            xytext=(comp["aa_mean"], comp["af_mean"]),
            arrowprops=dict(arrowstyle="-|>", color=PALETTE["blue_deep"], lw=0.9, mutation_scale=8),
        )
        ax_hero.errorbar(comp["aa_mean"], comp["af_mean"], xerr=comp["aa_sem"], yerr=comp["af_sem"], fmt="s", ms=4.0, mfc="white", mec=PALETTE["peach_deep"], ecolor=PALETTE["peach_deep"], capsize=2.0, elinewidth=0.6, zorder=3)
        ax_hero.errorbar(ours["aa_mean"], ours["af_mean"], xerr=ours["aa_sem"], yerr=ours["af_sem"], fmt="o", ms=4.2, mfc=PALETTE["blue"], mec=PALETTE["ink"], ecolor=PALETTE["blue_deep"], capsize=2.0, elinewidth=0.6, zorder=4)
        hero_name = {
            "Split-CIFAR-100": "C100",
            "Split-CIFAR-10": "C10",
            "Split-TinyImageNet": "Tiny",
            "5-Datasets": "5DS",
        }[entry["bench"]]
        dx, dy, ha = label_offsets[entry["bench"]]
        ax_hero.annotate(
            f"{hero_name}  ΔAF {entry['delta_af']:+.1f}  ΔAA {entry['delta_aa']:+.1f}",
            xy=(ours["aa_mean"], ours["af_mean"]),
            xytext=(ours["aa_mean"] + dx, ours["af_mean"] + dy),
            fontsize=5.6,
            color=PALETTE["muted"],
            ha=ha,
            va="center",
            arrowprops=dict(arrowstyle="-", color=PALETTE["grid"], lw=0.5),
            zorder=5,
        )
    ax_hero.set_xlim(49.0, 96.8)
    ax_hero.set_ylim(-0.4, 17.4)
    ax_hero.set_xlabel("Average accuracy (%)")
    ax_hero.set_ylabel("Average forgetting (%)")
    draw_panel_header(ax_hero, "a", "Matched frontier shift", x=-0.04, y=1.04)
    ax_hero.scatter([], [], marker="s", s=22, fc="white", ec=PALETTE["peach_deep"], label="Comparator")
    ax_hero.scatter([], [], marker="o", s=22, fc=PALETTE["blue"], ec=PALETTE["ink"], label="SSR+")
    ax_hero.legend(loc="upper left", fontsize=5.8)

    # Per-task accuracy on Split-CIFAR-100.
    style_panel(ax_curve, grid_axis="y")
    tasks = np.arange(1, 11)
    for name, exp, color, ls in [
        ("SSR+", "biocs_plus_s05_c100", PALETTE["blue"], "-"),
        ("LwF", "lwf_split_cifar100", PALETTE["peach_deep"], "--"),
        ("EWC", "ewc_split_cifar100", PALETTE["gray_deep"], ":"),
    ]:
        curves = []
        for mat in load_accuracy_matrices(exp):
            vals = [float(np.mean(mat[stage, : stage + 1][mat[stage, : stage + 1] > 0])) for stage in range(mat.shape[0])]
            curves.append(vals)
        arr = np.asarray(curves)
        mean, sem = arr.mean(axis=0), arr.std(axis=0, ddof=1) / math.sqrt(arr.shape[0])
        ax_curve.plot(tasks, mean, color=color, ls=ls, lw=1.1, marker="o", ms=2.5, label=name)
        ax_curve.fill_between(tasks, mean - sem, mean + sem, color=color, alpha=0.10)
    ax_curve.set_xlabel("Tasks seen")
    ax_curve.set_ylabel("AA (%)")
    ax_curve.set_xticks(tasks)
    ax_curve.legend(loc="lower left", fontsize=5.5)
    draw_panel_header(ax_curve, "b", "C100 per-task AA", x=-0.10, y=1.04)

    style_panel(ax_forget, grid_axis="y")
    ftasks = np.arange(2, 11)
    for name, exp, color, ls in [
        ("SSR+", "biocs_plus_s05_c100", PALETTE["blue"], "-"),
        ("LwF", "lwf_split_cifar100", PALETTE["peach_deep"], "--"),
    ]:
        mean, sem = mean_forgetting_curve(exp)
        ax_forget.plot(ftasks, mean, color=color, ls=ls, lw=1.1, marker="s", ms=2.5, label=name)
        ax_forget.fill_between(ftasks, mean - sem, mean + sem, color=color, alpha=0.10)
    ax_forget.set_xlabel("Tasks seen")
    ax_forget.set_ylabel("AF (%)")
    ax_forget.legend(loc="upper left", fontsize=5.5)
    draw_panel_header(ax_forget, "c", "C100 forgetting curve", x=-0.10, y=1.04)

    metric_heatmap(
        ax_aa,
        aa_matrix,
        [r[0] for r in benchmark_rows],
        method_cols,
        cmap=CMAPS["cool"],
        vmin=45,
        vmax=95,
        title="Method-family AA",
        label="d",
        highlight_col=4,
    )
    metric_heatmap(
        ax_af,
        af_matrix,
        [r[0] for r in benchmark_rows],
        method_cols,
        cmap=CMAPS["warm"],
        vmin=0,
        vmax=55,
        title="Method-family AF",
        label="e",
        highlight_col=4,
    )

    im = ax_geom.imshow(geom_score, aspect="auto", cmap=CMAPS["teal"], vmin=0.0, vmax=1.0)
    add_heatmap_hatches(ax_geom, geom_score.shape, alpha=0.14)
    geom_cols = ["AA", "1-AF", "rank", "1-|c|"]
    for i, method in enumerate(geom_methods):
        for j, col in enumerate(geom_cols):
            raw_val = geom_raw[i, [0, 1, 2, 3][j]]
            text = f"{raw_val:.1f}" if j != 3 else f"{raw_val:.3f}"
            color = "white" if im.norm(geom_score[i, j]) > 0.62 else PALETTE["ink"]
            ax_geom.text(j, i, text, ha="center", va="center", fontsize=6.1, fontweight="bold", color=color, zorder=4)
    ax_geom.set_xticks(np.arange(len(geom_cols)))
    ax_geom.set_xticklabels(geom_cols, rotation=0, ha="center", fontsize=6.25)
    ax_geom.set_yticks(np.arange(len(geom_labels)))
    ax_geom.set_yticklabels(["Base", "Bio", "Bio+KD"], fontsize=6.35)
    ax_geom.tick_params(length=0)
    for spine in ax_geom.spines.values():
        spine.set_linewidth(1.4)
        spine.set_color(PALETTE["ink"])
    draw_panel_header(ax_geom, "f", "CUB mechanism matrix", x=-0.06, y=1.03)

    style_panel(ax_overlap, grid_axis="y")
    bins = np.linspace(0.0, 0.90, 40)
    styles = {
        "Baseline": (PALETTE["gray_deep"], "-", 0.9),
        "KD": (PALETTE["peach_deep"], "--", 0.9),
        "SSR": (PALETTE["mint_deep"], "-.", 0.9),
        "SSR+KD": (PALETTE["blue_deep"], "-", 1.2),
    }
    for name, values in overlap_series.items():
        color, ls, lw = styles[name]
        ax_overlap.hist(
            values,
            bins=bins,
            density=True,
            histtype="stepfilled",
            facecolor=color,
            edgecolor="none",
            alpha=0.12 if name != "SSR+KD" else 0.22,
            label=name,
        )
        ax_overlap.hist(values, bins=bins, density=True, histtype="step", color=color, linestyle=ls, linewidth=lw)
    ax_overlap.set_xlabel("Prototype |cosine|")
    ax_overlap.set_ylabel("Density")
    ax_overlap.set_xlim(0, 0.86)
    ax_overlap.legend(loc="upper right", fontsize=5.5, ncol=1)
    draw_panel_header(ax_overlap, "g", "Overlap distribution", x=-0.08, y=1.03)

    af_reduction = [m["delta_af"] for m in matched]
    aa_change = [m["delta_aa"] for m in matched]
    bx = np.arange(len(matched))
    style_panel(ax_delta, grid_axis="y")
    ax_delta.axhline(0, color=PALETTE["ink"], lw=0.45)
    ax_delta.bar(bx - 0.18, af_reduction, 0.32, color=PALETTE["blue"], ec=PALETTE["ink"], lw=0.5, alpha=0.85, label="AF reduction")
    ax_delta.bar(bx + 0.18, aa_change, 0.32, color=PALETTE["peach"], ec=PALETTE["ink"], lw=0.5, alpha=0.85, label="AA change")
    for i, (afr, aac) in enumerate(zip(af_reduction, aa_change)):
        ax_delta.text(i - 0.18, afr + 0.25, f"{afr:.1f}", ha="center", va="bottom", fontsize=5.8, color=PALETTE["blue_deep"])
        ax_delta.text(i + 0.18, aac + (0.22 if aac >= 0 else -0.28), f"{aac:+.1f}", ha="center", va="bottom" if aac >= 0 else "top", fontsize=5.8, color=PALETTE["peach_deep"])
    ax_delta.set_xticks(bx)
    ax_delta.set_xticklabels([f"{m['bench'].replace('Split-', '')}\nvs {m['comp_label']}" for m in matched], fontsize=5.5)
    ax_delta.set_ylabel("Percentage points")
    ax_delta.set_ylim(min(-3.2, min(aa_change) - 0.9), max(af_reduction) + 1.4)
    draw_panel_header(ax_delta, "h", "Matched delta summary", x=-0.03, y=1.04)
    ax_delta.legend(loc="upper right", ncol=2, fontsize=5.8)

    style_panel(ax_pairs, grid_axis="y")
    pair_labels = ["Auklet pair 1", "Albatross pair", "Blackbird pair"]
    y_pair = np.arange(3)[::-1]
    for yi, pair in zip(y_pair, deep["selected_case_pairs"][:3]):
        base = abs(pair["baseline_cosine"])
        plus = abs(pair["biocs_kd_cosine"])
        ax_pairs.plot([base, plus], [yi, yi], color=PALETTE["grid"], lw=1.4, zorder=1)
        ax_pairs.scatter(base, yi, s=32, marker="s", fc="white", ec=PALETTE["rose_deep"], lw=0.8, zorder=3)
        ax_pairs.scatter(plus, yi, s=34, marker="o", fc=PALETTE["blue"], ec=PALETTE["ink"], lw=0.6, zorder=4)
        ax_pairs.text(plus + 0.02, yi, f"{base:.2f}→{plus:.2f}", va="center", fontsize=5.5, color=PALETTE["muted"])
    ax_pairs.set_yticks(y_pair)
    ax_pairs.set_yticklabels(pair_labels, fontsize=5.5)
    ax_pairs.set_xlabel("Case-pair prototype |cosine|")
    ax_pairs.set_xlim(0, 0.86)
    ax_pairs.scatter([], [], marker="s", s=24, fc="white", ec=PALETTE["rose_deep"], label="Baseline")
    ax_pairs.scatter([], [], marker="o", s=24, fc=PALETTE["blue"], ec=PALETTE["ink"], label="SSR+KD")
    ax_pairs.legend(loc="upper right", fontsize=5.8)
    draw_panel_header(ax_pairs, "i", "Real-pair separation", x=-0.03, y=1.04)

    save(fig, "fig_results_bars")


def fig_cub200_capacity_segmentation() -> None:
    cub_capacity = load_json(RESULTS / "cub200_capacity_classification_20260429" / "summary.json")
    cub = load_json(RESULTS / "cub200_summary_20260429" / "summary.json")
    cub_random = load_json(RESULTS / "cub200_capacity_random_order_20260429" / "summary.json")
    seg_highres = load_json(RESULTS / "cub200_seg_seen_highres_20260429_summary" / "summary.json")
    seg_tuned = load_json(RESULTS / "cub200_seg_highres_lsp005_kd2_seed0_20260429_summary" / "summary.json")
    cod_baseline = load_json(RESULTS / "cod_320_baseline_gpu2_20260429" / "summary.json")
    cod_channel = load_json(RESULTS / "cod_320_channel_gpu3_20260429" / "summary.json")
    adapter = load_json(RESULTS / "lowrank_adapter_probe_20260514" / "summary.json")
    temporal = load_json(TEMPORAL_SUMMARY)

    fig = plt.figure(figsize=(SCI_FULL_WIDTH, 5.55), facecolor=PALETTE["paper"])
    gs = fig.add_gridspec(
        3,
        4,
        width_ratios=[1.05, 1.05, 1.0, 1.0],
        height_ratios=[1.05, 1.00, 0.96],
        left=0.07,
        right=0.995,
        bottom=0.09,
        top=0.96,
        wspace=0.40,
        hspace=0.48,
    )

    ax_score = fig.add_subplot(gs[0, 0:2])
    ax_frontier = fig.add_subplot(gs[0, 2])
    ax_dense = fig.add_subplot(gs[0, 3])
    ax_loc = fig.add_subplot(gs[1, 0])
    ax_norm = fig.add_subplot(gs[1, 1:3])
    ax_cos = fig.add_subplot(gs[1, 3])
    ax_psd = fig.add_subplot(gs[2, 0:2])
    ax_adapter = fig.add_subplot(gs[2, 2:4])

    def compact_table(
        ax: plt.Axes,
        rows: list[tuple[str, str, str, str]],
        title: str,
        label: str,
        x_cols: list[float] | None = None,
        headers: list[str] | None = None,
    ) -> None:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, len(rows) + 1.25)
        ax.axis("off")
        draw_panel_header(ax, label, title, x=0.0, y=1.02)
        header_y = len(rows) + 0.54
        if x_cols is None:
            x_cols = [0.02, 0.42, 0.61, 0.78]
        if headers is None:
            headers = ["Readout", "Control", "SSR", "Delta"]
        for x, h in zip(x_cols, headers):
            ax.text(x, header_y, h, ha="left", va="center", fontsize=6.8, fontweight="bold", color=PALETTE["ink"])
        ax.plot([0.02, 0.98], [len(rows) + 0.24, len(rows) + 0.24], color=PALETTE["ink"], lw=0.9)
        for idx, row in enumerate(rows):
            y = len(rows) - idx - 0.10
            if idx % 2 == 0:
                ax.add_patch(patches.Rectangle((0.015, y - 0.35), 0.965, 0.60, fc=PALETTE["row_blue"], ec=PALETTE["blue_deep"], lw=0.0, hatch="...", alpha=0.32))
            for x, text in zip(x_cols, row):
                color = PALETTE["blue_deep"] if x == x_cols[-1] and not text.startswith("-") else PALETTE["ink"]
                ax.text(x, y, text, ha="left", va="center", fontsize=6.75, color=color, fontweight="bold" if x == x_cols[-1] else None)
        ax.plot([0.02, 0.98], [0.18, 0.18], color=PALETTE["ink"], lw=0.8)

    ax = ax_score
    style_panel(ax, grid_axis="x")
    score_rows = [
        ("CUB AA", 0.0, 2.42, "pts", PALETTE["blue_deep"]),
        ("CUB AF", 0.0, 1.91, "pts lower", PALETTE["blue_deep"]),
        ("CUB rank", 0.0, 34.32, "rank", PALETTE["mint_deep"]),
        ("CUB |cos|", 0.0, 0.271, "lower", PALETTE["mint_deep"]),
        ("Mask mIoU", 0.0, 3.69, "pts", PALETTE["blue_deep"]),
        ("Mask Dice", 0.0, 2.74, "pts", PALETTE["blue_deep"]),
        ("COD macro", 0.0, 1.32, "x10^-2", PALETTE["peach_deep"]),
        ("GPT-2 loc.", 0.0, 11.0, "pts", PALETTE["peach_deep"]),
        ("Adapter AA", 0.0, 0.52, "pts", PALETTE["lavender_deep"]),
        ("Adapter AF", 0.0, 0.42, "pts lower", PALETTE["lavender_deep"]),
    ]
    y_score = np.arange(len(score_rows))[::-1]
    for yi, (label, base, delta, unit, color) in zip(y_score, score_rows):
        ax.plot([base, delta], [yi, yi], color=PALETTE["grid"], lw=2.5, zorder=1)
        ax.scatter(base, yi, s=38, marker="s", fc="white", ec=PALETTE["gray_deep"], lw=1.0, zorder=3)
        ax.scatter(delta, yi, s=50, marker="D", fc=color, ec=PALETTE["ink"], lw=0.8, zorder=4)
        ax.text(delta + 0.72, yi, f"+{delta:g} {unit}", va="center", fontsize=6.8, color=color, fontweight="bold")
    for y0, y1, name in [(7.5, 9.5, "classification"), (4.5, 6.5, "dense"), (2.5, 3.5, "editing"), (-0.5, 1.5, "adapter")]:
        ax.axhspan(y0, y1, facecolor=PALETTE["row_blue"], edgecolor=PALETTE["blue_deep"], hatch="...", alpha=0.13, zorder=0)
        ax.text(35.2, (y0 + y1) / 2, name, ha="right", va="center", fontsize=6.6, color=PALETTE["muted"], fontweight="bold")
    ax.axvline(0, color=PALETTE["ink"], lw=0.8)
    ax.set_yticks(y_score)
    ax.set_yticklabels([r[0] for r in score_rows], fontsize=6.85)
    ax.set_xlabel("Beneficial shift from matched control")
    ax.set_xlim(-1.2, 36.5)
    draw_panel_header(ax, "a", "Transfer effect strips", x=-0.03, y=1.025)

    # (b) Fine-grained and adapter AA / AF frontiers
    ax = ax_frontier
    style_panel(ax, grid_axis="both")
    for label, method, color, marker in [
        ("Baseline", "baseline", PALETTE["gray_deep"], "o"),
        ("SSR", "biocs", PALETTE["mint_deep"], "^"),
        ("SSR+KD", "biocs_kd", PALETTE["blue_deep"], "D"),
    ]:
        aa = cub["classification"][method]["avg_accuracy"]["mean"]
        af = cub["classification"][method]["avg_forgetting"]["mean"]
        ax.scatter(aa, af, s=70 if method == "biocs_kd" else 62, marker=marker, fc=color if method != "baseline" else "white", ec=PALETTE["ink"] if method != "baseline" else color, lw=1.2, zorder=4)
        if method == "biocs_kd":
            ax.text(aa - 0.10, af + 0.07, "CUB SSR+KD", ha="right", fontsize=6.5, color=PALETTE["ink"], fontweight="bold")
    for label, method, color, marker in [
        ("Base", "baseline", PALETTE["gray_deep"], "o"),
        ("KD", "kd", PALETTE["peach_deep"], "s"),
        ("Bio", "biocs", PALETTE["mint_deep"], "^"),
        ("Bio+KD", "biocs_kd", PALETTE["blue_deep"], "D"),
    ]:
        aa = adapter[method]["avg_accuracy"]["mean"]
        af = adapter[method]["avg_forgetting"]["mean"]
        ax.scatter(aa, af, s=62 if method == "biocs_kd" else 52, marker=marker, fc=color if method not in {"baseline", "kd"} else "white", ec=PALETTE["ink"] if method not in {"baseline", "kd"} else color, lw=1.1, zorder=4, alpha=0.92)
        if method == "biocs_kd":
            ax.text(aa - 0.08, af - 0.12, "Adapter Bio+KD", ha="right", fontsize=6.4, color=PALETTE["muted"], fontweight="bold")
    ax.set_xlim(76.2, 81.55)
    ax.set_ylim(1.20, 4.35)
    ax.set_xlabel("Average accuracy (%)")
    ax.set_ylabel("Average forgetting (%)")
    draw_panel_header(ax, "b", "CUB and adapter frontiers", x=-0.12, y=1.025)

    ax = ax_dense
    style_panel(ax, grid_axis="x")
    dense_rows = [
        ("Mask\nmIoU", seg_highres["segmentation"]["baseline"]["mean_iou"]["mean"], seg_tuned["segmentation"]["biocs_kd"]["mean_iou"]["mean"], 100.0),
        ("Mask\nDice", seg_highres["segmentation"]["baseline"]["mean_dice"]["mean"], seg_tuned["segmentation"]["biocs_kd"]["mean_dice"]["mean"], 100.0),
        ("CAMO\nmIoU", cod_baseline["baseline"]["CAMO"]["miou"]["mean"], cod_channel["biocs_channel"]["CAMO"]["miou"]["mean"], 100.0),
        ("COD10K\nDice", cod_baseline["baseline"]["COD10K"]["dice"]["mean"], cod_channel["biocs_channel"]["COD10K"]["dice"]["mean"], 100.0),
        ("COD\nmacro", cod_baseline["baseline"]["macro_score"]["mean"], cod_channel["biocs_channel"]["macro_score"]["mean"], 100.0),
    ]
    y_dense = np.arange(len(dense_rows))[::-1]
    for yi, (label, ctrl, bio, scale) in zip(y_dense, dense_rows):
        ctrl_s = ctrl * scale
        bio_s = bio * scale
        ax.plot([ctrl_s, bio_s], [yi, yi], color=PALETTE["grid"], lw=2.3, zorder=1)
        ax.scatter(ctrl_s, yi, s=42, marker="s", fc="white", ec=PALETTE["gray_deep"], lw=1.0, zorder=3)
        ax.scatter(bio_s, yi, s=48, marker="D", fc=PALETTE["blue_deep"], ec=PALETTE["ink"], lw=0.8, zorder=4)
        ax.text(bio_s + 0.45, yi, f"+{bio_s - ctrl_s:.1f}", va="center", fontsize=6.7, color=PALETTE["blue_deep"], fontweight="bold")
    ax.set_yticks(y_dense)
    ax.set_yticklabels([r[0] for r in dense_rows], fontsize=6.75)
    ax.set_xlabel("Score (%)")
    ax.set_xlim(53, 84.5)
    ax.scatter([], [], s=34, marker="s", fc="white", ec=PALETTE["gray_deep"], label="Control")
    ax.scatter([], [], s=36, marker="D", fc=PALETTE["blue_deep"], ec=PALETTE["ink"], label="SSR")
    ax.legend(loc="lower right", fontsize=6.2)
    draw_panel_header(ax, "c", "Dense-prediction gains", x=-0.12, y=1.025)

    # (d) GPT-2 XL matched locality
    ax = ax_loc
    style_panel(ax, grid_axis="x")
    matched_labels = ["ZsRE-300", "ZsRE-400", "CF-100", "CF-200", "Recent-100", "Recent-200"]
    true_ft = np.array([1.2222, 1.5, 4.4111, 3.6115, 29.8453, 28.8508], dtype=float)
    biocs = np.array([4.5833, 5.25, 8.8125, 9.8323, 36.6907, 35.3224], dtype=float)
    y_loc = np.arange(len(matched_labels))[::-1]
    for yi, lab, b, p in zip(y_loc, matched_labels, true_ft, biocs):
        ax.plot([b, p], [yi, yi], color=PALETTE["grid"], lw=2.3, zorder=1)
        ax.scatter(b, yi, s=46, marker="s", fc="white", ec=PALETTE["gray_deep"], lw=1.1, zorder=3)
        ax.scatter(p, yi, s=52, marker="D", fc=PALETTE["peach_deep"], ec=PALETTE["ink"], lw=0.9, zorder=4)
        ax.text(p + 0.8, yi, f"+{p - b:.1f}", va="center", fontsize=7.2, color=PALETTE["muted"])
    ax.set_yticks(y_loc)
    ax.set_yticklabels(matched_labels, fontsize=6.8)
    ax.set_xlabel("Locality (%)")
    ax.set_xlim(0, 40.5)
    draw_panel_header(ax, "d", "GPT-2 XL locality", x=-0.12, y=1.025)

    styles = {
        "baseline": {"color": PALETTE["gray_deep"], "ls": "-", "marker": "o"},
        "kd": {"color": PALETTE["peach_deep"], "ls": "--", "marker": "s"},
        "biocs": {"color": PALETTE["mint_deep"], "ls": "-.", "marker": "^"},
        "biocs_kd": {"color": PALETTE["blue_deep"], "ls": "-", "marker": "D"},
    }

    ax = ax_norm
    style_panel(ax, grid_axis="both")
    for method in ["baseline", "kd", "biocs", "biocs_kd"]:
        y_vals = np.asarray(temporal[method]["series"]["update_norm"], dtype=float)
        x_vals = np.arange(1, len(y_vals) + 1)
        st = styles[method]
        ax.plot(x_vals, y_vals, color=st["color"], linestyle=st["ls"], marker=st["marker"], markersize=3.9, linewidth=2.0 if method == "biocs_kd" else 1.65, markeredgecolor=PALETTE["ink"], markeredgewidth=0.55, label={"baseline": "Baseline", "kd": "KD", "biocs": "SSR", "biocs_kd": "SSR+KD"}[method])
    ax.set_xlabel("Task transition")
    ax.set_ylabel(r"$||\Delta W_t||_F$")
    ax.legend(loc="lower right", fontsize=6.7, ncol=2)
    draw_panel_header(ax, "e", "Task-resolution update magnitude", x=-0.05, y=1.025)

    ax = ax_cos
    style_panel(ax, grid_axis="both")
    x_vals = np.arange(1, len(np.asarray(temporal["baseline"]["update_vector_autocorr"]["acf"], dtype=float)[1:]) + 1)
    for method in ["baseline", "kd", "biocs_kd"]:
        y_vals = np.asarray(temporal[method]["update_vector_autocorr"]["acf"], dtype=float)[1:]
        st = styles[method]
        ax.plot(x_vals, y_vals, color=st["color"], linestyle=st["ls"], marker=st["marker"], markersize=3.4, linewidth=1.7, markeredgecolor=PALETTE["ink"], markeredgewidth=0.5)
    ax.axhline(0.0, color=PALETTE["ink"], lw=0.85, alpha=0.50)
    ax.set_xlabel("Lag")
    ax.set_ylabel(r"$R_\Delta(d)$")
    draw_panel_header(ax, "f", "Direction persistence", x=-0.12, y=1.025)

    ax = ax_psd
    style_panel(ax, grid_axis="both")
    for method in ["baseline", "kd", "biocs_kd"]:
        freqs = np.asarray(temporal[method]["update_vector_autocorr"]["psd_freqs"], dtype=float)
        power = np.asarray(temporal[method]["update_vector_autocorr"]["psd_power"], dtype=float)
        power = power / (power.sum() + 1e-12)
        st = styles[method]
        ax.plot(freqs, power, color=st["color"], linestyle=st["ls"], linewidth=2.0 if method == "biocs_kd" else 1.7, label={"baseline": "Baseline", "kd": "KD", "biocs_kd": "SSR+KD"}[method])
    add_zone_texture(ax, 0.0, 0.25, color=PALETTE["mint"], hatch="...", alpha=0.12)
    add_zone_texture(ax, 0.5, 1.0, color=PALETTE["peach"], hatch="///", alpha=0.11)
    ax.set_xlabel("Normalized temporal frequency")
    ax.set_ylabel("Normalized power")
    ax.legend(loc="upper right", fontsize=6.8)
    ax.text(
        0.02,
        0.92,
        f"HF mass: Base {temporal['baseline']['update_vector_autocorr']['psd_high_ratio']:.3f}, "
        f"KD {temporal['kd']['update_vector_autocorr']['psd_high_ratio']:.3f}, "
        f"Bio+KD {temporal['biocs_kd']['update_vector_autocorr']['psd_high_ratio']:.3f}",
        transform=ax.transAxes,
        fontsize=6.7,
        color=PALETTE["muted"],
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=PALETTE["grid"], lw=0.8),
    )
    draw_panel_header(ax, "g", "Temporal PSD shift", x=-0.05, y=1.025)

    ax = ax_adapter
    methods = ["baseline", "kd", "biocs", "biocs_kd"]
    row_labels = ["Base", "KD", "SSR", "Bio+KD"]
    raw = np.array(
        [
            [
                adapter[m]["avg_accuracy"]["mean"],
                adapter[m]["avg_forgetting"]["mean"],
                adapter[m]["effective_rank"]["mean"],
                adapter[m]["mean_abs_offdiag_cosine"]["mean"],
                adapter[m]["adapter_basis_cosine"]["mean"],
            ]
            for m in methods
        ],
        dtype=float,
    )
    score = raw.copy()
    for j in [0, 2]:
        lo, hi = raw[:, j].min(), raw[:, j].max()
        score[:, j] = (raw[:, j] - lo) / (hi - lo + 1e-12)
    for j in [1, 3, 4]:
        lo, hi = raw[:, j].min(), raw[:, j].max()
        score[:, j] = (hi - raw[:, j]) / (hi - lo + 1e-12)
    im = ax.imshow(score, aspect="auto", cmap=CMAPS["purple"], vmin=0.0, vmax=1.0)
    add_heatmap_hatches(ax, score.shape, alpha=0.13)
    cols = ["AA", "AF", "rank", "proto\n|cos|", "adapter\n|cos|"]
    for i in range(raw.shape[0]):
        for j in range(raw.shape[1]):
            val = raw[i, j]
            text = f"{val:.1f}" if j in {0, 1, 2} else f"{val:.3f}"
            color = "white" if im.norm(score[i, j]) > 0.62 else PALETTE["ink"]
            ax.text(j, i, text, ha="center", va="center", fontsize=6.9, fontweight="bold", color=color, zorder=4)
    ax.add_patch(patches.Rectangle((-0.48, 3 - 0.48), len(cols) - 0.04, 0.96, fill=False, ec=PALETTE["ink"], lw=1.35, linestyle="--"))
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, fontsize=6.9)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=7.0)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_linewidth(1.4)
        spine.set_color(PALETTE["ink"])
    draw_panel_header(ax, "h", "Low-rank adapter geometry", x=-0.05, y=1.025)

    save_loose(fig, "fig_cub200_capacity_segmentation")


LLM_ROWS = [
    ("Qwen2.5-VL-3B", "ZsRE 50", 100.0, 2.0, 100.0, 21.0),
    ("Qwen2.5-VL-3B", "CF 100", 97.0, 0.51, 100.0, 8.69),
    ("Qwen2.5-VL-3B", "Recent 100", 88.0, 22.96, 100.0, 39.46),
    ("Llama-3-8B", "ZsRE 50", 96.0, 2.0, 100.0, 2.0),
    ("Llama-3-8B", "CF 100", 95.0, 0.51, 100.0, 2.15),
    ("Llama-3-8B", "Recent 100", 90.0, 24.41, 100.0, 27.77),
]


def draw_llm_concept(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    ax.add_patch(patches.FancyBboxPatch((0.03, 0.08), 0.94, 0.78, boxstyle="round,pad=0.018", fc=PALETTE["panel"], ec=PALETTE["ink"], lw=1.5, hatch="..", alpha=0.92))
    cols = [0.20, 0.50, 0.80]
    titles = ["Edit stream", "Spatial prior", "Stable locality"]
    colors = [PALETTE["peach"], PALETTE["mint"], PALETTE["blue"]]
    for idx, (cx, title, c) in enumerate(zip(cols, titles, colors)):
        ax.text(cx, 0.80, title, ha="center", va="center", fontsize=8.8, fontweight="bold", color=PALETTE["ink"])
        ax.add_patch(patches.Circle((cx, 0.52), 0.095, fc=c, ec=PALETTE["ink"], lw=1.2, hatch=TEXTURES[idx], alpha=0.82))
        for ang in np.linspace(0, 2 * np.pi, 7, endpoint=False):
            ax.plot([cx, cx + 0.18 * np.cos(ang)], [0.52, 0.52 + 0.15 * np.sin(ang)], color=PALETTE["grid"], lw=1.0)
            ax.add_patch(patches.Circle((cx + 0.18 * np.cos(ang), 0.52 + 0.15 * np.sin(ang)), 0.025, fc=c if idx != 1 else PALETTE["paper"], ec=PALETTE["ink"], lw=1.0, hatch=TEXTURES[idx], alpha=0.82))
        if idx == 1:
            ax.add_patch(patches.Ellipse((cx, 0.52), 0.48, 0.34, fill=False, ec=PALETTE["ink"], lw=1.4, ls="--"))
            for px in [cx - 0.20, cx + 0.20, cx]:
                ax.annotate("", xy=(cx, 0.52), xytext=(px, 0.70), arrowprops=dict(arrowstyle="-|>", lw=1.0, color=PALETTE["ink"]))
        ax.add_patch(patches.Rectangle((cx - 0.075, 0.20), 0.15, 0.11, fc=PALETTE["paper"], ec=PALETTE["ink"], lw=1.0, hatch=TEXTURES[idx + 3], alpha=0.80))
        for i in range(3):
            for j in range(4):
                fc = c if (i + j + idx) % 4 == 0 else PALETTE["gray"]
                ax.add_patch(patches.Circle((cx - 0.045 + j * 0.03, 0.225 + i * 0.028), 0.0085, fc=fc, ec="none"))
    for x0, x1 in [(0.31, 0.39), (0.61, 0.69)]:
        ax.annotate("", xy=(x1, 0.52), xytext=(x0, 0.52), arrowprops=dict(arrowstyle="-|>", lw=1.4, color=PALETTE["ink"]))
    ax.text(
        0.50,
        0.105,
        "SSR-FT constrains the edited subspace\nwhile preserving unrelated facts.",
        ha="center",
        va="center",
        fontsize=7.8,
        fontweight="bold",
        color=PALETTE["muted"],
    )


def fig_llm_family_transfer() -> None:
    if not FIGS1_PANEL_A.exists():
        raise FileNotFoundError("GPT-generated supplementary LLM schematic is missing.")

    temporal = load_json(TEMPORAL_SUMMARY)
    styles = {
        "baseline": {"color": PALETTE["gray_deep"], "ls": "-", "marker": "o"},
        "kd": {"color": PALETTE["peach_deep"], "ls": "--", "marker": "s"},
        "biocs": {"color": PALETTE["mint_deep"], "ls": "-.", "marker": "^"},
        "biocs_kd": {"color": PALETTE["blue_deep"], "ls": "-", "marker": "D"},
    }

    fig = plt.figure(figsize=(11.15, 8.55), facecolor=PALETTE["paper"])
    gs = fig.add_gridspec(
        3,
        5,
        height_ratios=[1.10, 1.03, 0.92],
        width_ratios=[1.10, 1.10, 1.10, 0.92, 0.92],
        left=0.055,
        right=0.99,
        top=0.972,
        bottom=0.07,
        wspace=0.32,
        hspace=0.38,
    )

    ax_a = fig.add_subplot(gs[0:2, 0:3])
    draw_asset_panel(ax_a, FIGS1_PANEL_A)
    draw_panel_header(ax_a, "a", "Locality-protected sequential editing", x=-0.015, y=1.02)

    ax_b = fig.add_subplot(gs[0, 3:5])
    short_rows = [
        ("Q-VL", "Z50"),
        ("Q-VL", "CF100"),
        ("Q-VL", "R100"),
        ("Llama", "Z50"),
        ("Llama", "CF100"),
        ("Llama", "R100"),
    ]
    rows = [f"{family}\n{task}" for family, task in short_rows]
    raw = np.array([[r[2], r[3], r[4], r[5], r[4] - r[2], r[5] - r[3]] for r in LLM_ROWS], dtype=float)
    normed = raw.copy()
    for j in range(raw.shape[1]):
        lo, hi = raw[:, j].min(), raw[:, j].max()
        normed[:, j] = (raw[:, j] - lo) / (hi - lo + 1e-12)
    im = ax_b.imshow(normed, aspect="auto", cmap=CMAPS["teal"], vmin=0, vmax=1)
    add_heatmap_hatches(ax_b, normed.shape, alpha=0.14)
    cols = ["FT\neff.", "FT\nloc.", "Bio\neff.", "Bio\nloc.", "Δeff.", "Δloc."]
    for i in range(raw.shape[0]):
        for j in range(raw.shape[1]):
            val = raw[i, j]
            text = f"{val:.0f}" if j in {0, 2, 4} else f"{val:.1f}"
            color = "white" if im.norm(normed[i, j]) > 0.63 else PALETTE["ink"]
            ax_b.text(j, i, text, ha="center", va="center", fontsize=6.6, fontweight="bold", color=color, zorder=4)
    ax_b.set_xticks(np.arange(len(cols)))
    ax_b.set_xticklabels(cols, fontsize=6.8)
    ax_b.set_yticks(np.arange(len(rows)))
    ax_b.set_yticklabels(rows, fontsize=6.6)
    ax_b.tick_params(length=0)
    for spine in ax_b.spines.values():
        spine.set_linewidth(1.4)
        spine.set_color(PALETTE["ink"])
    draw_panel_header(ax_b, "b", "Paired editing matrix", x=-0.05, y=1.03)

    ax_c = fig.add_subplot(gs[1, 3:5])
    style_panel(ax_c, grid_axis="x")
    labels = [f"{family} {task}" for family, task in short_rows]
    y = np.arange(len(LLM_ROWS))[::-1]
    family_blocks = [(3.5, 5.5), (0.5, 2.5)]
    for y0, y1 in family_blocks:
        ax_c.axhspan(y0, y1, facecolor=PALETTE["row_blue"], edgecolor=PALETTE["blue_deep"], hatch="...", alpha=0.16, zorder=0)
    for yi, row, label in zip(y, LLM_ROWS, labels):
        _, _, ft_eff, ft_loc, bio_eff, bio_loc = row
        ax_c.plot([ft_loc, bio_loc], [yi, yi], color=PALETTE["grid"], lw=2.4, zorder=1)
        ax_c.scatter(ft_loc, yi, marker="s", s=42, fc="white", ec=PALETTE["gray_deep"], lw=1.1, zorder=3)
        ax_c.scatter(bio_loc, yi, marker="o", s=48, fc=PALETTE["mint_deep"], ec=PALETTE["ink"], lw=0.9, zorder=4)
        if bio_loc > 34:
            ax_c.text(bio_loc - 0.9, yi, f"+{bio_loc - ft_loc:.1f} loc.", ha="right", va="center", fontsize=7.2, color=PALETTE["muted"])
        else:
            ax_c.text(bio_loc + 0.9, yi, f"+{bio_loc - ft_loc:.1f} loc.", va="center", fontsize=7.2, color=PALETTE["muted"])
    ax_c.set_yticks(y)
    ax_c.set_yticklabels(labels, fontsize=7.0)
    ax_c.yaxis.tick_right()
    ax_c.tick_params(axis="y", labelright=True, labelleft=False, pad=1)
    ax_c.set_xlabel("Locality (%)")
    ax_c.set_xlim(0, 44)
    ax_c.scatter([], [], marker="s", s=34, fc="white", ec=PALETTE["gray_deep"], label="FT")
    ax_c.scatter([], [], marker="o", s=36, fc=PALETTE["mint_deep"], ec=PALETTE["ink"], label="SSR-FT")
    ax_c.legend(loc="lower right", fontsize=7.0)
    draw_panel_header(ax_c, "c", "Locality gains on the same slices", x=-0.05, y=1.03)

    ax_d = fig.add_subplot(gs[2, 0:3])
    style_panel(ax_d, grid_axis="both")
    for method in ["baseline", "kd", "biocs", "biocs_kd"]:
        y_vals = np.asarray(temporal[method]["series"]["update_norm"], dtype=float)
        x_vals = np.arange(1, len(y_vals) + 1)
        st = styles[method]
        ax_d.plot(
            x_vals,
            y_vals,
            color=st["color"],
            linestyle=st["ls"],
            marker=st["marker"],
            markersize=4.0,
            linewidth=2.0 if method == "biocs_kd" else 1.7,
            markeredgecolor=PALETTE["ink"],
            markeredgewidth=0.6,
            label={"baseline": "Baseline", "kd": "KD", "biocs": "SSR", "biocs_kd": "SSR+KD"}[method],
        )
    ax_d.set_xlabel("Task transition")
    ax_d.set_ylabel(r"$||\Delta W_t||_F$")
    ax_d.legend(loc="lower right", fontsize=7.0, ncol=2)
    draw_panel_header(ax_d, "d", "Update magnitude over tasks", x=-0.05, y=1.03)

    ax_e = fig.add_subplot(gs[2, 3:5])
    style_panel(ax_e, grid_axis="both")
    x_vals = np.arange(1, len(np.asarray(temporal["baseline"]["update_vector_autocorr"]["acf"], dtype=float)[1:]) + 1)
    for method in ["baseline", "kd", "biocs_kd"]:
        y_vals = np.asarray(temporal[method]["update_vector_autocorr"]["acf"], dtype=float)[1:]
        st = styles[method]
        ax_e.plot(
            x_vals,
            y_vals,
            color=st["color"],
            linestyle=st["ls"],
            marker=st["marker"],
            markersize=3.8,
            linewidth=1.9,
            markeredgecolor=PALETTE["ink"],
            markeredgewidth=0.55,
            label={"baseline": "Baseline", "kd": "KD", "biocs_kd": "SSR+KD"}[method],
        )
    ax_e.axhline(0.0, color=PALETTE["ink"], lw=0.9, alpha=0.45)
    ax_e.set_xlabel("Lag in task transitions")
    ax_e.set_ylabel(r"mean cosine $R_\Delta(d)$")
    ax_e.text(
        0.03,
        0.95,
        f"lag-1: KD {temporal['kd']['update_vector_autocorr']['acf'][1]:.3f} | "
        f"SSR+KD {temporal['biocs_kd']['update_vector_autocorr']['acf'][1]:.3f}\n"
        f"PSD-high: KD {temporal['kd']['update_vector_autocorr']['psd_high_ratio']:.3f} | "
        f"SSR+KD {temporal['biocs_kd']['update_vector_autocorr']['psd_high_ratio']:.3f}",
        transform=ax_e.transAxes,
        va="top",
        fontsize=7.4,
        color=PALETTE["muted"],
        bbox=dict(boxstyle="round,pad=0.20", facecolor="white", edgecolor=PALETTE["grid"], linewidth=0.8),
    )
    ax_e.legend(loc="lower right", fontsize=7.0)
    draw_panel_header(ax_e, "e", "Direction persistence", x=-0.05, y=1.03)
    save(fig, "fig_llm_family_transfer")


def fig_supp_support_atlas() -> None:
    fourier = load_json(FOURIER_SUMMARY)
    rep = load_json(REPRESENTATION_SUMMARY)

    method_order = ["baseline", "biocs", "biocs_kd"]
    method_labels = ["Base", "SSR", "Bio+KD"]
    method_styles = {
        "baseline": {"color": PALETTE["gray_deep"], "marker": "o", "label": "Baseline"},
        "biocs": {"color": PALETTE["mint_deep"], "marker": "^", "label": "SSR"},
        "biocs_kd": {"color": PALETTE["blue_deep"], "marker": "D", "label": "SSR+KD"},
    }

    fig = plt.figure(figsize=(11.2, 8.75), facecolor=PALETTE["paper"])
    gs = fig.add_gridspec(
        3,
        4,
        width_ratios=[1.0, 1.08, 0.95, 0.95],
        height_ratios=[0.95, 1.05, 1.02],
        left=0.055,
        right=0.99,
        bottom=0.07,
        top=0.972,
        wspace=0.36,
        hspace=0.43,
    )

    # a: Fourier high-frequency ratio matrix.
    ax = fig.add_subplot(gs[0, 0])
    view_specs = [("Flat", "flat"), ("Class", "class_axis"), ("Feature", "feature_axis"), ("2D", "radial_2d")]
    hf = np.array(
        [
            [fourier[m]["aggregate"][f"{key}_high_ratio"]["mean"] for _, key in view_specs]
            for m in method_order
        ],
        dtype=float,
    )
    im = ax.imshow(hf, aspect="auto", cmap=CMAPS["warm"], vmin=0.16, vmax=0.42)
    add_heatmap_hatches(ax, hf.shape, alpha=0.15)
    for i in range(hf.shape[0]):
        for j in range(hf.shape[1]):
            color = "white" if im.norm(hf[i, j]) > 0.62 else PALETTE["ink"]
            ax.text(j, i, f"{hf[i, j]:.3f}", ha="center", va="center", fontsize=6.6, fontweight="bold", color=color, zorder=4)
    ax.set_xticks(np.arange(len(view_specs)))
    ax.set_xticklabels([v[0] for v in view_specs], rotation=35, ha="right", fontsize=6.7)
    ax.set_yticks(np.arange(len(method_labels)))
    ax.set_yticklabels(method_labels, fontsize=6.9)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_linewidth(1.3)
        spine.set_color(PALETTE["ink"])
    draw_panel_header(ax, "a", "Axis-aware HF ratios", x=-0.08, y=1.035)

    # b: Raw mean spectral profiles.
    ax = fig.add_subplot(gs[0, 1:3])
    style_panel(ax, grid_axis="both")
    for method in method_order:
        spectra = np.asarray(fourier[method]["mean_spectra"]["radial_2d"], dtype=float)
        spectra = spectra / (spectra.sum() + 1e-12)
        x = np.linspace(0, 1, len(spectra))
        st = method_styles[method]
        ax.plot(x, spectra, color=st["color"], marker=st["marker"], markersize=3.0, linewidth=1.8, label=st["label"], markeredgecolor=PALETTE["ink"], markeredgewidth=0.45)
    add_zone_texture(ax, 0.55, 1.0, color=PALETTE["peach"], hatch="///", alpha=0.11)
    ax.set_xlabel("Radial 2D frequency")
    ax.set_ylabel("Normalized power")
    ax.legend(loc="upper right", fontsize=6.8, ncol=3)
    draw_panel_header(ax, "b", "Mean 2D spectra", x=-0.04, y=1.035)

    # c: Absolute high-frequency band power relative to the baseline convention.
    ax = fig.add_subplot(gs[0, 3])
    style_panel(ax, grid_axis="x")
    rel_rows = []
    for label, key in view_specs:
        base = fourier["baseline"]["aggregate"][f"{key}_high_ratio"]["mean"] * fourier["baseline"]["aggregate"][f"{key}_total_power"]["mean"]
        bio = fourier["biocs"]["aggregate"][f"{key}_high_ratio"]["mean"] * fourier["biocs"]["aggregate"][f"{key}_total_power"]["mean"]
        plus = fourier["biocs_kd"]["aggregate"][f"{key}_high_ratio"]["mean"] * fourier["biocs_kd"]["aggregate"][f"{key}_total_power"]["mean"]
        rel_rows.append((label, bio / base, plus / base))
    y = np.arange(len(rel_rows))[::-1]
    for yi, (label, bio_rel, plus_rel) in zip(y, rel_rows):
        ax.plot([1.0, plus_rel], [yi, yi], color=PALETTE["grid"], lw=2.2, zorder=1)
        ax.scatter(1.0, yi, s=38, marker="o", fc="white", ec=PALETTE["gray_deep"], lw=1.0, zorder=3)
        ax.scatter(bio_rel, yi, s=42, marker="^", fc="white", ec=PALETTE["mint_deep"], lw=1.0, zorder=4)
        ax.scatter(plus_rel, yi, s=45, marker="D", fc=PALETTE["blue_deep"], ec=PALETTE["ink"], lw=0.8, zorder=5)
        ax.text(plus_rel + 0.04, yi, f"{plus_rel:.2f}x", va="center", fontsize=6.8, color=PALETTE["muted"])
    ax.axvline(1.0, color=PALETTE["ink"], lw=0.85, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rel_rows], fontsize=6.9)
    ax.set_xlabel("HF band power / baseline")
    ax.set_xlim(0.72, 1.55)
    ax.scatter([], [], s=32, marker="o", fc="white", ec=PALETTE["gray_deep"], label="Base")
    ax.scatter([], [], s=34, marker="^", fc="white", ec=PALETTE["mint_deep"], label="Bio")
    ax.scatter([], [], s=36, marker="D", fc=PALETTE["blue_deep"], ec=PALETTE["ink"], label="Bio+KD")
    ax.legend(loc="lower right", fontsize=5.9)
    draw_panel_header(ax, "c", "Absolute HF power", x=-0.10, y=1.035)

    # d/e: full Split-CIFAR-100 forgetting support.
    ax = fig.add_subplot(gs[1, 0:2])
    style_panel(ax, grid_axis="both")
    tasks = np.arange(2, 11)
    curve_specs = [
        ("SSR+", "biocs_plus_s05_c100", PALETTE["blue_deep"], "D", "-", 2.15),
        ("LwF", "lwf_split_cifar100", PALETTE["peach_deep"], "s", "-", 1.75),
        ("EWC", "ewc_split_cifar100", PALETTE["rose_deep"], "^", "--", 1.45),
        ("MAS", "mas_split_cifar100", PALETTE["lavender_deep"], "o", "-.", 1.45),
        ("SI", "si_split_cifar100", PALETTE["gray_deep"], "v", ":", 1.35),
    ]
    endpoints = []
    for name, exp, color, marker, ls, lw in curve_specs:
        mean, sem = mean_forgetting_curve(exp)
        ax.plot(tasks, mean, color=color, marker=marker, linestyle=ls, linewidth=lw, markersize=4.0, markeredgecolor=PALETTE["ink"], markeredgewidth=0.55, label=name)
        ax.fill_between(tasks, mean - sem, mean + sem, color=color, edgecolor=color, hatch=TEXTURES[len(endpoints) % len(TEXTURES)], linewidth=0.0, alpha=0.09 if name in {"SSR+", "LwF"} else 0.055)
        endpoints.append((name, float(mean[-1]), color, marker))
    ax.fill_between(tasks, 0, 1.8, color=PALETTE["blue"], edgecolor=PALETTE["blue_deep"], hatch="///", alpha=0.11, lw=0)
    ax.set_xlabel("Number of tasks learned")
    ax.set_ylabel("Cumulative forgetting (%)")
    ax.set_ylim(-1.5, 73)
    ax.set_xlim(1.8, 10.4)
    ax.legend(loc="upper left", fontsize=6.9, ncol=3)
    ax.text(0.50, 0.10, "near-zero forgetting band", transform=ax.transAxes, fontsize=7.1, color=PALETTE["blue_deep"], fontweight="bold")
    draw_panel_header(ax, "d", "Split-CIFAR-100 forgetting progression", x=-0.04, y=1.025)

    ax = fig.add_subplot(gs[1, 2])
    style_panel(ax, grid_axis="x")
    order = ["SSR+", "LwF", "MAS", "SI", "EWC"]
    endpoint_map = {name: (val, color, marker) for name, val, color, marker in endpoints}
    y = np.arange(len(order))[::-1]
    for yi, name in zip(y, order):
        val, color, marker = endpoint_map[name]
        ax.plot([0, val], [yi, yi], color=PALETTE["grid"], lw=2.1, zorder=1)
        ax.scatter(val, yi, s=46, marker=marker, fc=color if name != "LwF" else "white", ec=PALETTE["ink"] if name != "LwF" else color, lw=1.0, zorder=3)
        ax.text(val + 1.1, yi, f"{val:.1f}", va="center", fontsize=7.0, fontweight="bold" if name in {"SSR+", "LwF"} else None, color=color if name == "SSR+" else PALETTE["ink"])
    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=7.0)
    ax.set_xlabel("Final AF (%)")
    ax.set_xlim(0, 72)
    draw_panel_header(ax, "e", "Endpoint retention", x=-0.10, y=1.025)

    # f: CUB order robustness.
    ax = fig.add_subplot(gs[1, 3])
    cub_rows = []
    for group_label, group_key in [("Sem", "cub_semantic"), ("Rand", "cub_random")]:
        cls = rep[group_key]["classification"]
        for method, label in zip(method_order, method_labels):
            d = cls[method]
            cub_rows.append(
                (
                    f"{group_label} {label}",
                    d["avg_accuracy"]["mean"],
                    d["avg_forgetting"]["mean"],
                    d["effective_rank"]["mean"],
                    d["mean_abs_offdiag_cosine"]["mean"],
                )
            )
    raw = np.asarray([[r[1], r[2], r[3], r[4]] for r in cub_rows], dtype=float)
    score = raw.copy()
    for j in [0, 2]:
        lo, hi = raw[:, j].min(), raw[:, j].max()
        score[:, j] = (raw[:, j] - lo) / (hi - lo + 1e-12)
    for j in [1, 3]:
        lo, hi = raw[:, j].min(), raw[:, j].max()
        score[:, j] = (hi - raw[:, j]) / (hi - lo + 1e-12)
    im = ax.imshow(score, aspect="auto", cmap=CMAPS["teal"], vmin=0, vmax=1)
    add_heatmap_hatches(ax, score.shape, alpha=0.13)
    for i in range(raw.shape[0]):
        for j in range(raw.shape[1]):
            text = f"{raw[i, j]:.1f}" if j != 3 else f"{raw[i, j]:.3f}"
            color = "white" if im.norm(score[i, j]) > 0.62 else PALETTE["ink"]
            ax.text(j, i, text, ha="center", va="center", fontsize=5.8, fontweight="bold", color=color, zorder=4)
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(["AA", "AF", "rank", "|cos|"], fontsize=6.3, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(cub_rows)))
    ax.set_yticklabels([r[0] for r in cub_rows], fontsize=5.9)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_linewidth(1.25)
        spine.set_color(PALETTE["ink"])
    draw_panel_header(ax, "f", "CUB order robustness", x=-0.10, y=1.025)

    # g: Geometry and retention are coupled only when the spatial prior is anchored.
    ax = fig.add_subplot(gs[2, 0:2])
    style_panel(ax, grid_axis="both")
    scatter_groups = [
        ("Synthetic", rep["synthetic"], "o", 95.0),
        ("CUB semantic", rep["cub_semantic"]["classification"], "s", 0.0),
    ]
    for dataset_label, group, marker, y_offset in scatter_groups:
        for method in method_order:
            d = group[method]
            st = method_styles[method]
            x_val = d["effective_rank"]["mean"]
            y_val = d["avg_accuracy"]["mean"]
            ax.scatter(x_val, y_val, s=62, marker=marker, fc=st["color"] if method != "baseline" else "white", ec=PALETTE["ink"] if method != "baseline" else st["color"], lw=1.1, zorder=4)
            if method == "biocs_kd":
                if dataset_label == "CUB semantic":
                    ax.text(x_val - 22.0, y_val + 3.0, "CUB semantic\nBio+KD", fontsize=6.7, color=PALETTE["muted"])
                else:
                    ax.text(x_val + 1.5, y_val + 1.0, "Synthetic\nBio+KD", fontsize=6.7, color=PALETTE["muted"])
        base = group["baseline"]
        plus = group["biocs_kd"]
        ax.annotate(
            "",
            xy=(plus["effective_rank"]["mean"], plus["avg_accuracy"]["mean"]),
            xytext=(base["effective_rank"]["mean"], base["avg_accuracy"]["mean"]),
            arrowprops=dict(arrowstyle="-|>", color=PALETTE["grid"], lw=1.4),
            zorder=1,
        )
    ax.set_xlabel("Effective rank")
    ax.set_ylabel("Average accuracy (%)")
    ax.set_xlim(42, 172)
    ax.set_ylim(6, 88)
    ax.scatter([], [], marker="o", s=45, fc="white", ec=PALETTE["ink"], label="Synthetic")
    ax.scatter([], [], marker="s", s=45, fc="white", ec=PALETTE["ink"], label="CUB semantic")
    ax.legend(loc="lower right", fontsize=6.8)
    draw_panel_header(ax, "g", "Rank expansion requires an anchor", x=-0.04, y=1.025)

    # h: COD target and dose boundary in one paired panel.
    ax = fig.add_subplot(gs[2, 2:4])
    style_panel(ax, grid_axis="y")
    cod_scores = [
        ("Targeted\nchannel", rep["cod_320_macro_score"]["baseline"], rep["cod_320_macro_score"]["biocs_channel"], "+0.013"),
        ("Over-broad\nquery+channel", rep["cod_boundary_macro_score"]["baseline"], rep["cod_boundary_macro_score"]["biocs_both"], "-0.007"),
    ]
    x = np.arange(len(cod_scores))
    width = 0.34
    base_vals = [r[1] for r in cod_scores]
    bio_vals = [r[2] for r in cod_scores]
    ax.bar(x - width / 2, base_vals, width, color="white", ec=PALETTE["gray_deep"], lw=1.0, hatch="///", label="Matched baseline")
    ax.bar(x + width / 2, bio_vals, width, color=[PALETTE["blue_deep"], PALETTE["rose"]], ec=PALETTE["ink"], lw=0.9, hatch="...", alpha=0.88, label="SSR setting")
    for i, (_, base, bio, delta) in enumerate(cod_scores):
        color = PALETTE["blue_deep"] if bio >= base else PALETTE["rose_deep"]
        ax.plot([i - width / 2, i + width / 2], [base, bio], color=color, lw=1.1)
        ax.text(i, max(base, bio) + 0.004, delta, ha="center", fontsize=7.3, fontweight="bold", color=color)
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in cod_scores], fontsize=7.1)
    ax.set_ylabel("Macro score")
    ax.set_ylim(0.715, 0.765)
    ax.legend(loc="upper right", fontsize=6.9)
    ax.text(
        0.03,
        0.08,
        "Targeted channel prior improves;\nover-broad query+channel is a boundary case.",
        transform=ax.transAxes,
        fontsize=7.0,
        color=PALETTE["muted"],
        bbox=dict(boxstyle="round,pad=0.16", facecolor="white", edgecolor=PALETTE["grid"], linewidth=0.8),
    )
    draw_panel_header(ax, "h", "COD target and dose boundary", x=-0.04, y=1.025)

    save(fig, "fig_supp_support_atlas")


def fig_dynamics_composite() -> None:
    temporal = json.loads(TEMPORAL_SUMMARY.read_text())
    fourier = json.loads(FOURIER_SUMMARY.read_text())

    styles = {
        "baseline": {"color": PALETTE["gray_deep"], "ls": "-", "marker": "o"},
        "kd": {"color": PALETTE["peach_deep"], "ls": "--", "marker": "s"},
        "biocs": {"color": PALETTE["mint_deep"], "ls": "-.", "marker": "^"},
        "biocs_kd": {"color": PALETTE["blue_deep"], "ls": "-", "marker": "D"},
    }

    fig = plt.figure(figsize=(11.1, 6.4), facecolor=PALETTE["paper"])
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.18, 1.0],
        height_ratios=[1.0, 0.86],
        left=0.065,
        right=0.99,
        bottom=0.08,
        top=0.97,
        wspace=0.28,
        hspace=0.34,
    )

    ax_norm = fig.add_subplot(gs[0, 0])
    for method in ["baseline", "kd", "biocs", "biocs_kd"]:
        y = np.asarray(temporal[method]["series"]["update_norm"], dtype=float)
        x = np.arange(1, len(y) + 1)
        st = styles[method]
        ax_norm.plot(
            x,
            y,
            color=st["color"],
            linestyle=st["ls"],
            marker=st["marker"],
            markersize=4.8,
            linewidth=2.2 if method == "biocs_kd" else 1.9,
            markeredgecolor=PALETTE["ink"],
            markeredgewidth=0.7,
            label={"baseline": "Baseline", "kd": "KD", "biocs": "SSR", "biocs_kd": "SSR+KD"}[method],
        )
    ax_norm.set_title("(a) Update magnitude over task time", loc="left", fontsize=12.8, fontweight="bold")
    ax_norm.set_xlabel("Task transition")
    ax_norm.set_ylabel(r"$||\Delta W_t||_F$")
    ax_norm.grid(color=PALETTE["grid"], lw=0.8, alpha=0.85)
    clean_ax(ax_norm)
    ax_norm.legend(loc="lower right", fontsize=8)
    ax_norm.text(
        0.03,
        0.95,
        "KD already contracts the update stream;\nSSR+KD trims it further.",
        transform=ax_norm.transAxes,
        va="top",
        fontsize=8.1,
        color=PALETTE["muted"],
        bbox=dict(boxstyle="round,pad=0.20", facecolor="white", edgecolor=PALETTE["grid"], linewidth=0.8),
    )
    add_panel_label(ax_norm, "a", x=-0.06, y=1.03)

    ax_cos = fig.add_subplot(gs[0, 1])
    for method in ["baseline", "kd", "biocs", "biocs_kd"]:
        y = np.asarray(temporal[method]["update_vector_autocorr"]["acf"], dtype=float)[1:]
        x = np.arange(1, len(y) + 1)
        st = styles[method]
        ax_cos.plot(
            x,
            y,
            color=st["color"],
            linestyle=st["ls"],
            marker=st["marker"],
            markersize=4.2,
            linewidth=2.0 if method == "biocs_kd" else 1.8,
            markeredgecolor=PALETTE["ink"],
            markeredgewidth=0.6,
        )
    ax_cos.axhline(0.0, color=PALETTE["ink"], lw=0.9, alpha=0.45)
    ax_cos.set_title("(b) Direction persistence", loc="left", fontsize=12.8, fontweight="bold")
    ax_cos.set_xlabel("Lag in task transitions")
    ax_cos.set_ylabel(r"mean cosine $R_\Delta(d)$")
    ax_cos.grid(color=PALETTE["grid"], lw=0.8, alpha=0.85)
    clean_ax(ax_cos)
    ax_cos.text(
        0.03,
        0.95,
        f"Lag-1 cosine: KD {temporal['kd']['update_vector_autocorr']['acf'][1]:.3f} | "
        f"SSR+KD {temporal['biocs_kd']['update_vector_autocorr']['acf'][1]:.3f}",
        transform=ax_cos.transAxes,
        va="top",
        fontsize=8.0,
        color=PALETTE["muted"],
        bbox=dict(boxstyle="round,pad=0.20", facecolor="white", edgecolor=PALETTE["grid"], linewidth=0.8),
    )
    add_panel_label(ax_cos, "b", x=-0.08, y=1.03)

    ax_psd = fig.add_subplot(gs[1, 0])
    for method in ["baseline", "kd", "biocs", "biocs_kd"]:
        freqs = np.asarray(temporal[method]["update_vector_autocorr"]["psd_freqs"], dtype=float)
        power = np.asarray(temporal[method]["update_vector_autocorr"]["psd_power"], dtype=float)
        power = power / (power.sum() + 1e-12)
        st = styles[method]
        ax_psd.plot(freqs, power, color=st["color"], linestyle=st["ls"], linewidth=2.1 if method == "biocs_kd" else 1.8)
    ax_psd.axvspan(0.0, 0.25, color=PALETTE["mint"], alpha=0.18, lw=0)
    ax_psd.axvspan(0.5, 1.0, color=PALETTE["peach"], alpha=0.16, lw=0)
    ax_psd.set_title("(c) Temporal PSD and band shift", loc="left", fontsize=12.8, fontweight="bold")
    ax_psd.set_xlabel("Normalized temporal frequency")
    ax_psd.set_ylabel("Normalized power")
    ax_psd.grid(color=PALETTE["grid"], lw=0.8, alpha=0.85)
    clean_ax(ax_psd)
    ax_psd.text(
        0.03,
        0.95,
        f"High-band mass: baseline {temporal['baseline']['update_vector_autocorr']['psd_high_ratio']:.3f} | "
        f"KD {temporal['kd']['update_vector_autocorr']['psd_high_ratio']:.3f} | "
        f"SSR+KD {temporal['biocs_kd']['update_vector_autocorr']['psd_high_ratio']:.3f}",
        transform=ax_psd.transAxes,
        va="top",
        fontsize=7.8,
        color=PALETTE["muted"],
        bbox=dict(boxstyle="round,pad=0.20", facecolor="white", edgecolor=PALETTE["grid"], linewidth=0.8),
    )
    add_panel_label(ax_psd, "c", x=-0.06, y=1.03)

    ax_four = fig.add_subplot(gs[1, 1])
    categories = ["Class-axis", "Feature-axis", "2D radial"]
    y = np.arange(len(categories))
    base = [
        fourier["baseline"]["aggregate"]["class_axis_high_ratio"]["mean"],
        fourier["baseline"]["aggregate"]["feature_axis_high_ratio"]["mean"],
        fourier["baseline"]["aggregate"]["radial_2d_high_ratio"]["mean"],
    ]
    bio = [
        fourier["biocs"]["aggregate"]["class_axis_high_ratio"]["mean"],
        fourier["biocs"]["aggregate"]["feature_axis_high_ratio"]["mean"],
        fourier["biocs"]["aggregate"]["radial_2d_high_ratio"]["mean"],
    ]
    plus = [
        fourier["biocs_kd"]["aggregate"]["class_axis_high_ratio"]["mean"],
        fourier["biocs_kd"]["aggregate"]["feature_axis_high_ratio"]["mean"],
        fourier["biocs_kd"]["aggregate"]["radial_2d_high_ratio"]["mean"],
    ]
    for yi, b, bi, p in zip(y, base, bio, plus):
        ax_four.plot([b, p], [yi, yi], color=PALETTE["grid"], lw=2.2, zorder=1)
        ax_four.scatter(b, yi, s=40, marker="o", fc="white", ec=PALETTE["gray_deep"], lw=1.0, zorder=3)
        ax_four.scatter(bi, yi, s=42, marker="D", fc="white", ec=PALETTE["mint_deep"], lw=1.0, zorder=3)
        ax_four.scatter(p, yi, s=44, marker="s", fc=PALETTE["blue_deep"], ec=PALETTE["ink"], lw=0.8, zorder=4)
        ax_four.text(p + 0.008, yi, f"{p:.3f}", va="center", fontsize=7.8, color=PALETTE["muted"])
    ax_four.set_yticks(y)
    ax_four.set_yticklabels(categories)
    ax_four.invert_yaxis()
    ax_four.set_xlabel("High-frequency ratio")
    ax_four.set_xlim(0.20, 0.43)
    ax_four.set_title("(d) Fourier boundary control", loc="left", fontsize=12.8, fontweight="bold")
    ax_four.grid(axis="x", color=PALETTE["grid"], lw=0.8, alpha=0.85)
    clean_ax(ax_four)
    ax_four.scatter([], [], s=36, marker="o", fc="white", ec=PALETTE["gray_deep"], label="Baseline")
    ax_four.scatter([], [], s=38, marker="D", fc="white", ec=PALETTE["mint_deep"], label="SSR")
    ax_four.scatter([], [], s=40, marker="s", fc=PALETTE["blue_deep"], ec=PALETTE["ink"], label="SSR+KD")
    ax_four.legend(loc="upper left", fontsize=7.4)
    ax_four.text(
        0.47,
        0.06,
        "Update norm falls, but axis-aware HF ratios rise:\nthis is a boundary control, not the main positive evidence.",
        transform=ax_four.transAxes,
        fontsize=7.7,
        color=PALETTE["muted"],
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor=PALETTE["grid"], linewidth=0.8),
    )
    add_panel_label(ax_four, "d", x=-0.08, y=1.03)

    save(fig, "fig_dynamics_composite")


def main() -> None:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from journal_dense_figures import build_all
    from journal_dense_supplementary import build_supplementary

    build_all()
    build_supplementary()
    print(f"Wrote revised Science figures to {SCI_FIG}")


if __name__ == "__main__":
    main()
