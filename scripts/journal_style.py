"""Unified Nature/Cell-style helpers (SciencePlots + dense-annotation patterns).

Inspired by open-source journal figure toolkits:
  - SciencePlots: https://github.com/garrettj403/SciencePlots  (science+nature mplstyle)
  - cnsplots:       https://github.com/faridrashidi/cnsplots    (CNS palette / multipanel)
  - sciplotlib:     https://github.com/Timothysit/sciplotlib     (nature-reviews polish)
  - plotstyle:      https://pypi.org/project/plotstyle/          (journal size presets)
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

# Nature Methods / Nature Communications default qualitative cycle
NAT = {
    "navy": "#3C5488",
    "blue": "#4DBBD5",
    "teal": "#00A087",
    "red": "#E64B35",
    "comp": "#E64B35",
    "orange": "#F39B7F",
    "purple": "#8491B4",
    "rose": "#DC0000",
    "gold": "#E18727",
    "gray": "#8491B4",
    "gray_light": "#B0B8CA",
    "ink": "#1F1F1F",
    "muted": "#666666",
    "grid": "#E6E9EF",
    "paper": "#FFFFFF",
    "band": "#F4F7FB",
}

METHOD_STYLE: dict[str, dict] = {
    "SSR+": {"color": NAT["teal"], "marker": "o", "ls": "-", "lw": 1.5, "zorder": 10},
    "SSR+KD": {"color": NAT["teal"], "marker": "o", "ls": "-", "lw": 1.5, "zorder": 10},
    "Bio+KD": {"color": NAT["teal"], "marker": "o", "ls": "-", "lw": 1.5, "zorder": 10},
    "SSR": {"color": NAT["blue"], "marker": "^", "ls": "-.", "lw": 1.2, "zorder": 6},
    "KD-only": {"color": NAT["orange"], "marker": "s", "ls": "--", "lw": 1.2, "zorder": 7},
    "KD": {"color": NAT["orange"], "marker": "s", "ls": "--", "lw": 1.2, "zorder": 7},
    "LwF": {"color": NAT["orange"], "marker": "s", "ls": "--", "lw": 1.1, "zorder": 6},
    "EWC": {"color": NAT["gray"], "marker": "^", "ls": ":", "lw": 1.0, "zorder": 4},
    "MAS": {"color": NAT["gold"], "marker": "D", "ls": "-.", "lw": 1.0, "zorder": 4},
    "SI": {"color": NAT["purple"], "marker": "v", "ls": ":", "lw": 1.0, "zorder": 4},
    "DER++": {"color": NAT["blue"], "marker": "P", "ls": "-.", "lw": 1.0, "zorder": 4},
    "ER-ACE": {"color": NAT["rose"], "marker": "P", "ls": "--", "lw": 1.1, "zorder": 6},
    "Center": {"color": NAT["purple"], "marker": "h", "ls": "-.", "lw": 1.0, "zorder": 5},
    "SupCon": {"color": NAT["gold"], "marker": "X", "ls": ":", "lw": 1.0, "zorder": 5},
    "CosOrth": {"color": NAT["navy"], "marker": "v", "ls": ":", "lw": 1.0, "zorder": 5},
    "Baseline": {"color": NAT["gray_light"], "marker": "o", "ls": "-", "lw": 1.0, "zorder": 3},
    "Base": {"color": NAT["gray_light"], "marker": "o", "ls": "-", "lw": 1.0, "zorder": 3},
}

CMAP_DIV = LinearSegmentedColormap.from_list("div", ["#B2182B", "#F7F7F7", "#2166AC"])
CMAP_SEQ = LinearSegmentedColormap.from_list("seq", ["#F7FBFF", "#6BAED6", "#08306B"])

W_DOUBLE = 7.08  # inches ≈ 180 mm Nature double column
FIG_HEIGHT = 8.2  # match main composite figures
GRID_MAIN = dict(left=0.06, right=0.98, top=0.90, bottom=0.05, hspace=0.78, wspace=0.52)
ANNO_PT = 5.4  # inline numeric labels (matches main figures)
CELL_PT = 5.4  # heatmap cell values
LABEL_PT = 9.5
# Offset points above axes top-left; keeps letter/title in figure margin, not on data.
PANEL_LETTER_DX = -12
PANEL_LETTER_DY = 16
PANEL_TITLE_DX = 4
PANEL_TITLE_DY = 16


def setup() -> None:
    plt.style.use(["science", "nature", "no-latex"])
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 6.8,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 5.8,
            "ytick.labelsize": 5.8,
            "legend.fontsize": 5.6,
            "axes.linewidth": 0.5,
            "axes.edgecolor": NAT["ink"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "savefig.dpi": 300,
        }
    )


def panel_label(
    ax,
    letter: str,
    title: str = "",
    *,
    letter_dx: float = PANEL_LETTER_DX,
    letter_dy: float = PANEL_LETTER_DY,
    title_dx: float = PANEL_TITLE_DX,
    title_dy: float | None = None,
) -> None:
    """Place bold panel letter and optional subtitle in the margin above the axes."""
    tdy = PANEL_LETTER_DY if title_dy is None else title_dy
    ax.annotate(
        letter,
        xy=(0.0, 1.0),
        xycoords="axes fraction",
        xytext=(letter_dx, letter_dy),
        textcoords="offset points",
        fontsize=LABEL_PT,
        fontweight="bold",
        va="bottom",
        ha="right",
        annotation_clip=False,
        zorder=100,
    )
    if title:
        ax.annotate(
            title,
            xy=(0.0, 1.0),
            xycoords="axes fraction",
            xytext=(title_dx, tdy),
            textcoords="offset points",
            fontsize=6.0,
            va="bottom",
            ha="left",
            annotation_clip=False,
            color=NAT["muted"],
            zorder=100,
        )


def style_ax(ax, grid: str | None = "y") -> None:
    ax.set_facecolor(NAT["band"])
    if grid == "both":
        ax.grid(True, color=NAT["grid"], lw=0.3, alpha=0.95)
    elif grid in {"x", "y"}:
        ax.grid(axis=grid, color=NAT["grid"], lw=0.3, alpha=0.95)
    ax.tick_params(width=0.4, length=2.2)


def load_json(p: Path) -> dict:
    return json.loads(p.read_text())


def load_seeds(exp: str) -> tuple[np.ndarray, np.ndarray]:
    aa, af = [], []
    for f in sorted((RESULTS / exp).glob("seed_*/summary.json")):
        d = load_json(f)
        aa.append(float(d["avg_accuracy"]))
        af.append(float(d["avg_forgetting"]))
    if not aa:
        raise FileNotFoundError(exp)
    return np.asarray(aa), np.asarray(af)


def load_point(exp: str) -> dict[str, float]:
    aa, af = load_seeds(exp)
    sem = lambda v: float(v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1 else 0.0
    return {"aa_mean": float(aa.mean()), "aa_sem": sem(aa), "af_mean": float(af.mean()), "af_sem": sem(af), "aa": aa, "af": af}


def load_mats(exp: str) -> list[np.ndarray]:
    mats = []
    for p in sorted((RESULTS / exp).glob("seed_*/metrics.json")):
        mats.append(np.asarray(load_json(p)["accuracy_matrix"], dtype=float))
    if not mats:
        raise FileNotFoundError(exp)
    return mats


def mean_acc_curve(exp: str) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    curves = []
    for mat in load_mats(exp):
        curves.append([float(np.mean(mat[s, : s + 1][mat[s, : s + 1] > 0])) for s in range(mat.shape[0])])
    arr = np.asarray(curves)
    return arr.mean(0), arr.std(0, ddof=1) / math.sqrt(arr.shape[0]), curves


def mean_forget_curve(exp: str) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    curves = []
    for mat in load_mats(exp):
        vals = []
        for stage in range(1, mat.shape[0]):
            fs = []
            for task in range(stage):
                h = mat[:stage, task]
                h = h[h > 0]
                if h.size:
                    fs.append(float(h.max() - mat[stage, task]))
            vals.append(float(np.mean(fs)) if fs else 0.0)
        curves.append(vals)
    arr = np.asarray(curves)
    return arr.mean(0), arr.std(0, ddof=1) / math.sqrt(arr.shape[0]), curves


def mean_acc_matrix(exp: str) -> np.ndarray:
    return np.mean(np.stack(load_mats(exp), 0), 0)


def plot_seed_curves(ax, exp: str, label: str, seed_alpha: float = 0.35) -> None:
    st = METHOD_STYLE.get(label, {"color": NAT["gray"], "marker": "o", "ls": "-", "lw": 1.0, "zorder": 3})
    mean, sem, curves = mean_acc_curve(exp)
    x = np.arange(1, len(mean) + 1)
    for i, c in enumerate(curves):
        ax.plot(x, c, color=st["color"], alpha=seed_alpha, lw=0.7, zorder=st["zorder"] - 2)
    ax.plot(x, mean, color=st["color"], ls=st["ls"], lw=st["lw"], marker=st["marker"], ms=2.5, mec=NAT["ink"], mew=0.3, label=f"{label} (n={len(curves)})", zorder=st["zorder"])
    ax.fill_between(x, mean - sem, mean + sem, color=st["color"], alpha=0.15, lw=0)


def heatmap_dense(
    ax,
    data: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    *,
    cmap=CMAP_SEQ,
    vmin=None,
    vmax=None,
    fmt: str = "{:.1f}",
    fontsize: float = 5.2,
    highlight: tuple[int, int] | None = None,
) -> None:
    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if val == 0 and (i > j or j > i):
                continue
            color = "white" if im.norm(val) > 0.65 else NAT["ink"]
            ax.text(j, i, fmt.format(val), ha="center", va="center", fontsize=fontsize, color=color, fontweight="bold")
    if highlight:
        ax.add_patch(plt.Rectangle((highlight[1] - 0.5, highlight[0] - 0.5), 1, 1, fill=False, ec=NAT["red"], lw=1.0))
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=5.5, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=5.5)
    ax.tick_params(length=0)
    return im


def bar_sem(ax, x, means, sems, labels, colors, width=0.65, vertical=True) -> None:
    err_kw = dict(capsize=1.5, capthick=0.4, elinewidth=0.5, ecolor=NAT["ink"])
    if vertical:
        bars = ax.bar(x, means, width, yerr=sems, color=colors, ec=NAT["ink"], lw=0.35, error_kw=err_kw, alpha=0.92)
        for b, m in zip(bars, means):
            ax.text(b.get_x() + b.get_width() / 2, m + 0.02 * max(means), f"{m:.1f}", ha="center", va="bottom", fontsize=5.4)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=5.5, rotation=25, ha="right")
    else:
        bars = ax.barh(x, means, width, xerr=sems, color=colors, ec=NAT["ink"], lw=0.35, error_kw=err_kw, alpha=0.92)
        for b, m in zip(bars, means):
            ax.text(m + 0.01 * max(means), b.get_y() + b.get_height() / 2, f"{m:.1f}", va="center", fontsize=5.4)


def effective_rank(W: np.ndarray) -> float:
    s = np.linalg.svd(W, compute_uv=False)
    p = s / (s.sum() + 1e-12)
    return float(np.exp(-np.sum(p * np.log(p + 1e-12))))


def svd_spectrum(W: np.ndarray) -> np.ndarray:
    s = np.linalg.svd(W, compute_uv=False)
    e = s ** 2
    return e / (e.sum() + 1e-12)


def offdiag_cos(W: np.ndarray) -> np.ndarray:
    n = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-12)
    c = n @ n.T
    m = ~np.eye(c.shape[0], dtype=bool)
    return np.abs(c[m])
