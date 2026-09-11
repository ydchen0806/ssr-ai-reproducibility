#!/usr/bin/env python3
"""Build the strict, evidence-dense Figures 4--6 requested on 29 August."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import fitz
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
import numpy as np
from PIL import Image
from scipy.stats import t, ttest_1samp


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "artifacts" / "final_strict_20260829"
FIGURES = ROOT / "figures"
PREVIEWS = DATA / "previews"

WIDTH = 183 / 25.4
HEIGHT4 = 168 / 25.4
HEIGHT5 = 170 / 25.4
HEIGHT6 = 166 / 25.4

# The manuscript figures share the restrained atlas palette used in the
# anatomical results: neutral scaffolding, teal SSR, coral contrasts, and
# sparing purple/green accents.
INK = "#252B30"
GRAY = "#6F767B"
LIGHT = "#DDE1E3"
CONTROL = "#8B9195"
SSR = "#168A99"
ORANGE = "#CF565D"
GREEN = "#6A9C63"
PURPLE = "#7657A6"
RED = "#B94844"
PALE_BLUE = "#E6F1F2"
PALE_ORANGE = "#F7E9E9"
PALE_GREEN = "#EAF1E8"
PALE_GRAY = "#F0F2F3"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": 5.9,
    "axes.titlesize": 6.5,
    "axes.labelsize": 5.9,
    "xtick.labelsize": 5.2,
    "ytick.labelsize": 5.2,
    "legend.fontsize": 5.1,
    "axes.linewidth": 0.56,
    "lines.linewidth": 0.85,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.titleweight": "bold",
})


def read_rows(name: str) -> list[dict[str, str]]:
    path = DATA / name
    if not path.exists() and name.startswith("Fig5"):
        path = ROOT / "figure_source_data" / "figure5_20260905" / name
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def number(row: dict[str, str], key: str) -> float:
    return float(row[key])


def p_text(values: np.ndarray) -> str:
    """Format a two-sided paired t test on treatment-minus-control values."""
    p_value = float(ttest_1samp(np.asarray(values, dtype=float), 0.0).pvalue)
    return r"$P<0.001$" if p_value < 0.001 else rf"$P={p_value:.3f}$"


def panel_label(ax, label: str, title: str | None = None, x: float = -0.075) -> None:
    ax.text(x, 1.028, label, transform=ax.transAxes, fontsize=8.4, fontweight="bold",
            ha="left", va="bottom", clip_on=False)
    if title:
        ax.set_title(title, loc="left", pad=4.0)


def source_line(ax, text: str, y: float = -0.30, va: str = "top") -> None:
    """Place dataset/model provenance in a dedicated white footer strip."""
    ax.text(0.50, y, text, transform=ax.transAxes, ha="center", va=va,
            fontsize=4.55, color="#596166", clip_on=False, zorder=20,
            bbox=dict(fc="white", ec="none", alpha=0.98, pad=0.65))


def clean(ax, keep: str = "both") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if keep == "x":
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    elif keep == "y":
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(axis="x", length=0)
    ax.grid(False)
    ax.tick_params(width=0.5, length=2.0, pad=1.6)


def arrow(ax, start, end, color=GRAY, lw=0.85, mutation=6.5) -> None:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=mutation,
                                linewidth=lw, color=color, shrinkA=0, shrinkB=0))


def error_point(ax, row, y, color=SSR, marker="o", ms=4.2) -> None:
    mean = number(row, "paired_change")
    low = number(row, "ci_low")
    high = number(row, "ci_high")
    ax.errorbar(mean, y, xerr=[[mean - low], [high - mean]], fmt=marker, ms=ms,
                mfc=color, mec=INK, mew=0.35, color=color, ecolor=color,
                capsize=1.7, capthick=0.65, lw=0.8, zorder=3)


def save(fig, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    PREVIEWS.mkdir(parents=True, exist_ok=True)
    # Keep vector line art and text editable while embedding qualitative tiles
    # at publication resolution.
    fig.savefig(FIGURES / f"{stem}.pdf", dpi=450, facecolor="white")
    fig.savefig(FIGURES / f"{stem}.svg", facecolor="white")
    fig.savefig(FIGURES / f"{stem}.png", dpi=300, facecolor="white")
    fig.savefig(PREVIEWS / f"{stem}.png", dpi=190, facecolor="white")
    plt.close(fig)


def metric_row(rows, panel, group, metric, rank=None):
    matches = [row for row in rows
               if row["panel"] == panel and row["plot_group"] == group
               and row["primary_metric"] == metric]
    if rank is not None:
        matches = [row for row in matches if int(row["rank"]) == rank]
    if len(matches) != 1:
        raise ValueError((panel, group, metric, rank, len(matches)))
    return matches[0]


def draw_notre_dame(ax, x, y, scale=1.0, color=GREEN):
    """Compact vector landmark used only in the editable mechanism candidate."""
    tower_w = 0.027 * scale
    for offset in (-0.024, 0.024):
        ax.add_patch(Rectangle((x + offset - tower_w / 2, y), tower_w,
                               0.090 * scale, fc=PALE_GREEN, ec=INK, lw=0.45))
        ax.add_patch(Polygon([[x + offset - tower_w / 2, y + 0.090 * scale],
                              [x + offset + tower_w / 2, y + 0.090 * scale],
                              [x + offset, y + 0.116 * scale]],
                             closed=True, fc=color, ec=INK, lw=0.45))
    ax.add_patch(Rectangle((x - 0.018 * scale, y), 0.036 * scale,
                           0.065 * scale, fc="#F5F7F5", ec=INK, lw=0.45))
    ax.add_patch(Circle((x, y + 0.047 * scale), 0.009 * scale,
                        fc="white", ec=color, lw=0.45))


def draw_tokyo_tower(ax, x, y, scale=1.0, color=RED):
    ax.plot([x - 0.032 * scale, x, x + 0.032 * scale],
            [y, y + 0.156 * scale, y], color=color, lw=1.25)
    for height, width in ((0.045, 0.048), (0.093, 0.030)):
        ax.plot([x - width * scale / 2, x + width * scale / 2],
                [y + height * scale, y + height * scale], color=INK, lw=0.45)


def draw_oriental_pearl(ax, x, y, scale=1.0, color=SSR):
    ax.plot([x, x], [y, y + 0.150 * scale], color=INK, lw=0.65)
    ax.add_patch(Circle((x, y + 0.055 * scale), 0.024 * scale,
                        fc=PALE_BLUE, ec=color, lw=0.75))
    ax.add_patch(Circle((x, y + 0.112 * scale), 0.014 * scale,
                        fc=PALE_BLUE, ec=color, lw=0.75))
    ax.plot([x - 0.026 * scale, x + 0.026 * scale], [y, y], color=INK, lw=0.65)


def draw_mechanism(ax) -> None:
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    panel_label(ax, "a", "A center-surround prior for editable directions", x=-0.018)
    ax.add_patch(Rectangle((0.012, 0.12), 0.976, 0.70, fc="#FBFCFC", ec="none"))
    xs = [0.105, 0.305, 0.505, 0.705, 0.905]
    zone_titles = ["DENDRITE", "REPRESENTATION", "KNOWLEDGE EDIT",
                   "LOW-RANK EDIT BASIS", "SSR GEOMETRY"]
    for x, title in zip(xs, zone_titles):
        ax.text(x, 0.765, title, ha="center", va="center", fontsize=5.0,
                color=GRAY, fontweight="bold")
    for divider in (0.205, 0.405, 0.605, 0.805):
        ax.plot([divider, divider], [0.18, 0.72], color="#E3E6E7", lw=0.55)
    for start, end in zip(xs[:-1], xs[1:]):
        arrow(ax, (start + 0.070, 0.49), (end - 0.070, 0.49), color="#B3B8BB",
              lw=0.65, mutation=5.8)

    # Organic branch and spatially ordered synapses.
    branch_x = np.linspace(xs[0] - 0.070, xs[0] + 0.066, 90)
    branch_y = 0.47 + 0.018 * np.sin((branch_x - xs[0]) * 28)
    ax.plot(branch_x, branch_y, color=INK, lw=1.25, solid_capstyle="round")
    spine_specs = [(-0.060, 0.014, GREEN, 0.080), (-0.028, 0.018, SSR, 0.105),
                   (0.008, 0.024, SSR, 0.125), (0.043, 0.017, ORANGE, 0.090),
                   (0.070, 0.012, GREEN, 0.070)]
    for dx, radius, color, height in spine_specs:
        base_y = 0.47 + 0.018 * np.sin(dx * 28)
        ax.plot([xs[0] + dx, xs[0] + dx + 0.006], [base_y, base_y + height],
                color=INK, lw=0.55)
        ax.add_patch(Circle((xs[0] + dx + 0.006, base_y + height + radius * 0.35),
                            radius, fc=color, ec="white", lw=0.45, alpha=0.95))

    # Recognizable objects make neighbourhood distance concrete.
    ax.add_patch(Circle((xs[1], 0.50), 0.082, fc=PALE_BLUE, ec="none", alpha=0.85))
    draw_notre_dame(ax, xs[1] - 0.055, 0.43, scale=0.58, color=GREEN)
    draw_tokyo_tower(ax, xs[1] + 0.014, 0.43, scale=0.58, color=RED)
    draw_oriental_pearl(ax, xs[1] + 0.071, 0.43, scale=0.58, color=SSR)

    # Dense update matrix: the learned plastic state before factorization.
    matrix = np.array([[0.15, 0.72, 0.25, 0.48, 0.18],
                       [0.38, 0.20, 0.82, 0.31, 0.60],
                       [0.76, 0.34, 0.12, 0.69, 0.27],
                       [0.22, 0.63, 0.44, 0.17, 0.79]])
    x0, y0, cell = xs[2] - 0.065, 0.405, 0.026
    for iy in range(matrix.shape[0]):
        for ix in range(matrix.shape[1]):
            ax.add_patch(Rectangle((x0 + ix * cell, y0 + iy * cell),
                                   cell * 0.86, cell * 0.86,
                                   fc=mpl.colors.to_rgba(SSR, 0.18 + 0.72 * matrix[iy, ix]),
                                   ec="white", lw=0.25))
    ax.text(xs[2], 0.545, r"$\Delta W$", ha="center", va="bottom",
            fontsize=7.0, color=INK, fontweight="bold")

    # Factorized LoRA update: columns of B are the directions organized by SSR.
    basis_x = np.linspace(xs[3] - 0.056, xs[3] + 0.056, 4)
    angles = (-16, -4, 9, 20)
    colors = (GREEN, SSR, ORANGE, PURPLE)
    for bx, angle, color in zip(basis_x, angles, colors):
        block = Rectangle((bx - 0.012, 0.415), 0.024, 0.125,
                          fc=color, ec=INK, lw=0.42, alpha=0.85)
        block.set_transform(mpl.transforms.Affine2D().rotate_deg_around(
            bx, 0.475, angle) + ax.transData)
        ax.add_patch(block)
    ax.text(xs[3], 0.570, r"$\Delta W=AB^{\mathrm{T}}$", ha="center",
            fontsize=6.2, color=INK, fontweight="bold")
    ax.text(xs[3], 0.395, r"$r\ll d$", ha="center", fontsize=5.2, color=GRAY)
    # Radial widths mirror the observed short elevation, broad suppression,
    # and narrower recovery regimes.
    center = (xs[4], 0.50)
    ax.add_patch(Circle(center, 0.100, fc=PALE_BLUE, ec=SSR, lw=0.70, alpha=0.82))
    ax.add_patch(Circle(center, 0.080, fc=PALE_ORANGE, ec=RED, lw=0.65, alpha=0.94))
    ax.add_patch(Circle(center, 0.028, fc=PALE_GREEN, ec=GREEN, lw=0.70, alpha=1.0))
    ax.add_patch(Circle(center, 0.008, fc=INK, ec="white", lw=0.35))
    ax.text(xs[4], 0.535, "elevation", fontsize=4.6, color=GREEN, ha="center")
    ax.text(xs[4] + 0.055, 0.455, "suppression", fontsize=4.6, color=RED, ha="center")
    ax.text(xs[4] + 0.073, 0.580, "recovery", fontsize=4.6, color=SSR, ha="center")

    labels = ["spatial neighborhood", "semantic neighborhood", "editable update",
              "restricted edit directions", "bounded center-surround coupling"]
    for x, label in zip(xs, labels):
        ax.text(x, 0.265, label, ha="center", va="center", fontsize=5.0,
                color=INK, fontweight="bold")
    ax.text(0.405, 0.125, r"cosine: $d_{ij}=\sqrt{\max(1-q_i^{\mathrm{T}}q_j,\epsilon)}$",
            ha="center", va="center", fontsize=4.8, color=GRAY)
    ax.text(0.665, 0.125,
            r"projective: $d_{ij}=\sqrt{\max(1-(q_i^{\mathrm{T}}q_j)^2,\epsilon)}$  ($q\equiv -q$)",
            ha="center", va="center", fontsize=4.8, color=GRAY)


def build_figure4() -> None:
    rows = read_rows("figure4_source_data.csv")
    fig = plt.figure(figsize=(WIDTH, HEIGHT4), facecolor="white")
    gs = fig.add_gridspec(4, 12, height_ratios=[0.88, 1.17, 1.12, 1.14],
                          hspace=1.10, wspace=1.82)

    def mean_ci(values):
        values = np.asarray(values, dtype=float)
        mean = float(values.mean())
        half = float(t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
        return mean, mean - half, mean + half

    def direct_arm(arm):
        records = []
        source = ROOT / "source_data/raw/knowledge_editing/meeting_extension_cf" / arm
        for path in sorted(source.glob("seed_*/results.json")):
            record = read_json(path)
            records.append((int(record["seed"]), record["historical_retention"]["final"]))
        if len(records) != 10:
            raise ValueError(f"Expected ten direct KE records for {arm}")
        return records

    axa = fig.add_subplot(gs[0, :])
    axa.axis("off")

    # Direct, paired evidence: no interpolation between the 50- and 100-edit checkpoints.
    axb = fig.add_subplot(gs[1, :6])
    panel_label(axb, "b", "Direct SSR addition after 100 edits", x=-0.07)
    axb.axis("off")
    plain = dict(direct_arm("plain"))
    ssr = dict(direct_arm("ssr_only_cosine"))
    direct_seeds = sorted(set(plain) & set(ssr))
    for idx, (metric, title, color) in enumerate([
            ("efficacy", "History efficacy", SSR),
            ("locality", "History locality", GREEN)]):
        sub = axb.inset_axes([0.045 + 0.515 * idx, 0.18, 0.405, 0.67])
        control_values = np.asarray([plain[seed][metric] for seed in direct_seeds])
        ssr_values = np.asarray([ssr[seed][metric] for seed in direct_seeds])
        offsets = np.linspace(-0.035, 0.035, len(direct_seeds))
        for offset, control_value, ssr_value in zip(offsets, control_values, ssr_values):
            sub.plot([offset, 1 + offset], [control_value, ssr_value], color="#D4D9DB",
                     lw=0.48, alpha=0.72, zorder=1)
            sub.scatter(offset, control_value, s=10, facecolor="white",
                        edgecolor=CONTROL, linewidth=0.48, alpha=0.78, zorder=2)
            sub.scatter(1 + offset, ssr_value, s=10, facecolor="white",
                        edgecolor=color, linewidth=0.48, alpha=0.78, zorder=2)
        for xpos, values, point_color in [(0, control_values, CONTROL), (1, ssr_values, color)]:
            mean, low, high = mean_ci(values)
            sub.errorbar(xpos, mean, yerr=[[mean - low], [high - mean]], fmt="o",
                         ms=4.1, mfc=point_color, mec=INK, mew=0.4,
                         ecolor=INK, capsize=2.0, lw=0.8, zorder=4)
        delta, low, high = mean_ci(ssr_values - control_values)
        sub.text(0.50, 0.98,
                 f"$\\Delta$ {delta:+.2f} pp\n95% CI [{low:.2f}, {high:.2f}] | {p_text(ssr_values - control_values)}",
                 transform=sub.transAxes, ha="center", va="top", fontsize=4.7,
                 color=color, fontweight="bold")
        sub.set_xticks([0, 1], ["Plain FT", "+SSR"])
        sub.set_ylabel(f"{title} (%)", labelpad=0.8)
        sub.set_xlim(-0.16, 1.16)
        clean(sub)
    source_line(axb, "Dataset: CounterFact | base model: GPT-2 XL | n=10 paired seeds",
                y=-0.16)

    # Controlled mapping confirmation plus the complete development-only radial screen.
    axc = fig.add_subplot(gs[1, 6:])
    panel_label(axc, "c", "Mapping confirmation and radial-family screen", x=-0.07)
    axc.axis("off")
    confirmation = read_json(ROOT / "source_data/cosmos_ke_completion_20260901/confirmation_summary.json")
    mapping = confirmation["comparisons"]["counterfact_projective_minus_cosine"]["metrics"]
    # Leave a dedicated gutter between the confirmation and radial-family
    # axes so the right-hand y label cannot intrude into the left plot.
    map_ax = axc.inset_axes([0.035, 0.18, 0.32, 0.66])
    mapping_metrics = [
        ("final_history_efficacy", "Efficacy", SSR),
        ("final_history_locality", "Locality", PURPLE),
    ]
    for xpos, (metric, _label, color) in enumerate(mapping_metrics):
        deltas = np.asarray([item["delta"] for item in mapping[metric]["seed_deltas"]], dtype=float)
        jitter = np.linspace(-0.11, 0.11, len(deltas))
        map_ax.scatter(xpos + jitter, deltas, s=9, fc="white", ec=color,
                       lw=0.45, alpha=0.75, zorder=2)
        map_ax.scatter(xpos, deltas.mean(), s=32, fc=color, ec=INK,
                       lw=0.4, zorder=3)
    map_ax.axhline(0, color=INK, lw=0.55)
    map_ax.text(0.50, 0.96, "all 10 paired deltas = 0.00 pp",
                transform=map_ax.transAxes, ha="center", va="top", fontsize=4.5,
                color=INK, fontweight="bold")
    map_ax.set_xticks([0, 1], ["Efficacy", "Locality"])
    map_ax.set_ylabel("projective - cosine (pp)", labelpad=0.7)
    map_ax.set_xlim(-0.35, 1.35)
    map_ax.set_ylim(-0.10, 0.10)
    map_ax.set_yticks([-0.10, 0, 0.10])
    map_ax.set_title("Mapping confirmation | n=10", fontsize=5.1, pad=2)
    clean(map_ax)

    kernel = read_json(ROOT / "source_data/cosmos_ke_completion_20260901/development_selection.json")
    kernel_ax = axc.inset_axes([0.59, 0.18, 0.39, 0.66])
    kernel_colors = {"gaussian": CONTROL, "laplace": PURPLE,
                     "cauchy": SSR, "inverse": ORANGE}
    kernel_labels = {"gaussian": "Gaussian", "laplace": "Laplace",
                     "cauchy": "Cauchy", "inverse": "Inverse multiquadric"}
    for candidate in kernel["all_candidates"]:
        family = candidate["family"]
        color = kernel_colors[family]
        raw_x = [point["final_history_locality"] for point in candidate["seed_delta_pp"]]
        raw_y = [point["final_history_efficacy"] for point in candidate["seed_delta_pp"]]
        kernel_ax.scatter(raw_x, raw_y, s=11, fc=color, ec="white", lw=0.25,
                          alpha=0.50, zorder=2)
        mean_x = candidate["mean_delta_pp"]["final_history_locality"]
        mean_y = candidate["mean_delta_pp"]["final_history_efficacy"]
        kernel_ax.scatter(mean_x, mean_y, s=29, marker="o", fc=color, ec=INK,
                          lw=0.45, zorder=3)
    kernel_ax.axvline(0, color=LIGHT, lw=0.55)
    kernel_ax.axhline(0, color=LIGHT, lw=0.55)
    kernel_ax.set_xlabel("locality change (pp)")
    kernel_ax.set_ylabel("efficacy change (pp)", labelpad=0.4)
    kernel_ax.set_title("Radial families | development, n=3", fontsize=5.1, pad=2)
    clean(kernel_ax)
    key_positions = {
        "gaussian": (0.405, 0.70),
        "laplace": (0.405, 0.56),
        "cauchy": (0.405, 0.42),
        "inverse": (0.405, 0.28),
    }
    for family, (xpos, ypos) in key_positions.items():
        axc.scatter([xpos], [ypos], s=12, fc=kernel_colors[family], ec=INK, lw=0.3,
                    transform=axc.transAxes, clip_on=False)
        label = kernel_labels[family].replace("Inverse multiquadric", "Inverse MQ")
        axc.text(xpos + 0.014, ypos, label, transform=axc.transAxes,
                 va="center", ha="left", fontsize=3.75, color=INK)
    source_line(axc,
                "Dataset: CounterFact | base model: GPT-2 XL | mapping n=10; radial development n=3",
                y=-0.16)

    # Coefficients were chosen within the two-order development pairs; the panel is descriptive.
    axd = fig.add_subplot(gs[2, :4])
    panel_label(axd, "d", "Compatibility with established editors", x=-0.12)
    editor_values = {
        "AlphaEdit": (0.10, 5.078125, "o", SSR),
        "MEMIT": (1.00, 5.00, "D", PURPLE),
    }
    axd.scatter(0, 0, s=25, fc="white", ec=CONTROL, lw=0.8, zorder=3)
    for editor, (locality, efficacy, marker, color) in editor_values.items():
        axd.add_patch(FancyArrowPatch((0, 0), (locality, efficacy), arrowstyle="-|>",
                                     mutation_scale=7.0, linewidth=0.95, color=color,
                                     shrinkA=5.5, shrinkB=7.0, zorder=1))
        axd.scatter(locality, efficacy, s=40, marker=marker, fc=color, ec=INK,
                    lw=0.4, zorder=3)
        label = f"{editor}+SSR vs {editor}"
        axd.annotate(label, (locality, efficacy), xytext=(4, 2),
                     textcoords="offset points", fontsize=4.8, color=color,
                     fontweight="bold")
    axd.set_xlabel("locality change from editor baseline (pp)")
    axd.set_ylabel("efficacy change (pp)", labelpad=0.8)
    axd.set_xlim(-0.12, 1.32); axd.set_ylim(-0.35, 5.75)
    source_line(axd, "Dataset: ZsRE | base model: Qwen2.5-7B | development n=2/editor",
                y=-0.34)
    clean(axd)

    # Absolute locked factorial means; locality is shown even when it does not improve.
    axe = fig.add_subplot(gs[2, 4:8])
    panel_label(axe, "e", "Direct SSR across datasets", x=-0.10)
    axe.axis("off")
    cross_dataset = [
        ("CounterFact", 10.0, 12.4, 6.94, 7.39),
        ("WikiRecent", 56.4, 57.7, 32.66, 33.72),
        ("ZsRE", 51.1, 54.2, 12.20, 11.40),
    ]
    with (ROOT / "figure_source_data/figure4_20260905/Fig4e_cross_dataset_seed_level.csv").open(
            newline="", encoding="utf-8") as handle:
        cross_seed_rows = list(csv.DictReader(handle))
    source_dataset = {
        "CounterFact": "WikiData CounterFact", "WikiRecent": "WikiRecent", "ZsRE": "ZsRE"
    }
    for idx, (value_keys, title) in enumerate([
            ((1, 2), "History efficacy"), ((3, 4), "History locality")]):
        sub = axe.inset_axes([0.04 + idx * 0.50, 0.20, 0.42, 0.65])
        for ypos, record in enumerate(cross_dataset[::-1]):
            control_value, treatment_value = record[value_keys[0]], record[value_keys[1]]
            line_color = SSR if treatment_value >= control_value else ORANGE
            sub.plot([control_value, treatment_value], [ypos, ypos], color=line_color,
                     lw=1.1, zorder=1)
            sub.scatter(control_value, ypos, s=18, fc="white", ec=CONTROL, lw=0.75, zorder=2)
            sub.scatter(treatment_value, ypos, s=22, fc=line_color, ec=INK, lw=0.35, zorder=3)
            endpoint = "efficacy_delta_pp" if idx == 0 else "locality_delta_pp"
            paired = np.asarray([
                float(row[endpoint]) for row in cross_seed_rows
                if row["dataset"] == source_dataset[record[0]]
            ])
            sub.text(0.98, ypos, p_text(paired), transform=sub.get_yaxis_transform(),
                     ha="right", va="bottom", fontsize=3.9, color=line_color)
        sub.set_yticks(range(3), [row[0] for row in cross_dataset[::-1]] if idx == 0 else [])
        sub.set_xlabel(f"{title} (%)")
        sub.set_title(title, fontsize=5.1, pad=2)
        sub.set_ylim(-0.42, 2.42)
        sub.tick_params(axis="y", pad=1.0)
        clean(sub, "x")
    source_line(axe,
                "Datasets: CounterFact, WikiRecent, ZsRE | base model: GPT-2 XL | n=10/dataset",
                y=-0.34)

    axf = fig.add_subplot(gs[2, 8:])
    panel_label(axf, "f", "Component and recipe ablation", x=-0.10)
    ablation = [
        ("Plain FT", 10.0, 6.94, "o", CONTROL, "none"),
        ("Plain FT+Anchor", 11.2, 8.23, "^", PURPLE, "none"),
        ("Plain FT+Spectral", 9.6, 5.15, "s", ORANGE, "none"),
        ("Stabilized", 10.9, 6.17, "D", INK, "none"),
        ("Plain FT+SSR", 12.4, 7.39, "P", SSR, "cosine"),
        ("SSR+FT+Anchor+Spectral", 10.4, 9.33, "X", GREEN, "projective"),
    ]
    label_offsets = {
        "Plain FT": (3, -8), "Plain FT+Anchor": (3, 3),
        "Plain FT+Spectral": (3, -1), "Stabilized": (3, -8),
        "Plain FT+SSR": (5, 2),
        "SSR+FT+Anchor+Spectral": (-3, 2),
    }
    for name, efficacy, locality, marker, color, _mapping_name in ablation:
        axf.scatter(locality, efficacy, s=34, marker=marker, fc=color, ec=INK,
                    lw=0.4, zorder=3)
        axf.annotate(name, (locality, efficacy), xytext=label_offsets[name],
                     textcoords="offset points", fontsize=4.25, color=color,
                     fontweight="bold" if "SSR" in name else "normal",
                     ha="right" if name == "SSR+FT+Anchor+Spectral" else "left")
    axf.set_xlabel("history locality (%)")
    axf.set_ylabel("history efficacy (%)", labelpad=0.8)
    axf.set_xlim(4.65, 9.85); axf.set_ylim(9.25, 13.20)
    source_line(axf, "CounterFact | GPT-2 XL | n=10; mapping identities retained in Source Data", y=-0.34)
    clean(axf)

    axg = fig.add_subplot(gs[3, :7])
    panel_label(axg, "g", "SSR+FT+Anchor+Spectral transfer", x=-0.075)
    transfer_rows = read_rows("figure4_source_data.csv")
    transfer = [row for row in transfer_rows if row["panel"] == "g"]
    model_alias = {"GPT-2 XL": "GPT-2 XL", "Qwen-VL": "Qwen2.5-VL",
                   "Qwen2.5-VL": "Qwen2.5-VL", "Llama-3": "Llama-3"}
    pair_values = {}
    for row in transfer:
        key = (model_alias.get(row["model"], row["model"]), row["dataset"])
        metric = "efficacy" if row["primary_metric"].startswith("efficacy") else "locality"
        pair_values.setdefault(key, {})[metric] = number(row, "paired_change")
    model_markers = {"GPT-2 XL": "o", "Qwen2.5-VL": "s", "Llama-3": "^"}
    dataset_colors = {"CounterFact": ORANGE, "WikiRecent": GREEN, "ZsRE": PURPLE}
    for (model, dataset), values in pair_values.items():
        axg.scatter(values["efficacy"], values["locality"], s=36,
                    marker=model_markers[model], fc=dataset_colors[dataset], ec=INK,
                    lw=0.4, zorder=3)
    axg.axvline(0, color=LIGHT, lw=0.55); axg.axhline(0, color=LIGHT, lw=0.55)
    axg.set_xlabel("efficacy gain (pp)")
    axg.set_ylabel("locality gain (pp)", labelpad=0.8)
    for model, dataset in (("GPT-2 XL", "ZsRE"), ("Qwen2.5-VL", "ZsRE")):
        x, y = pair_values[(model, dataset)]["efficacy"], pair_values[(model, dataset)]["locality"]
        axg.scatter(x, y, s=58, marker=model_markers[model], fc="none", ec=INK,
                    lw=0.7, zorder=2)
        axg.annotate("efficacy ceiling", (x, y), xytext=(5, 0),
                     textcoords="offset points", fontsize=4.2, color=GRAY, va="center")
    model_handles = [axg.scatter([], [], s=23, marker=marker, fc="white", ec=INK,
                                 lw=0.5, label=model) for model, marker in model_markers.items()]
    dataset_handles = [axg.scatter([], [], s=23, marker="o", fc=color, ec=INK,
                                   lw=0.35, label=dataset) for dataset, color in dataset_colors.items()]
    axg.text(0.28, 0.965, "Model:", transform=axg.transAxes, ha="right",
             va="center", fontsize=4.4, color=GRAY)
    model_legend = axg.legend(handles=model_handles, frameon=False,
                             loc="upper center", bbox_to_anchor=(0.62, 1.005), ncol=3,
                             handletextpad=0.2, columnspacing=0.65, fontsize=4.5)
    axg.add_artist(model_legend)
    axg.text(0.28, 0.855, "Dataset:", transform=axg.transAxes, ha="right",
             va="center", fontsize=4.4, color=GRAY)
    axg.legend(handles=dataset_handles, frameon=False,
               loc="upper center", bbox_to_anchor=(0.62, 0.895), ncol=3,
               handletextpad=0.2, columnspacing=0.65, fontsize=4.5)
    source_line(
        axg,
        "Datasets: CounterFact, WikiRecent, ZsRE | base models: GPT-2 XL, Qwen2.5-VL, Llama-3 | n=1/pair",
        y=-0.34,
    )
    clean(axg)

    # Dense Qwen mechanism checkpoints use update-row geometry. Keep the native
    # scales separate rather than rescaling these diagnostics as percentages.
    axh = fig.add_subplot(gs[3, 7:])
    panel_label(axh, "h", "Dense-edit update geometry", x=-0.075)
    axh.axis("off")
    mechanism = read_json(ROOT / "source_data/ke_geometry_cosmos/summary.json")
    geometry_specs = [
        ("update_effective_rank", "Effective-rank gain", 1.0, "gain (rank units)"),
        ("update_offdiag_cosine_reduction", "Off-diagonal cosine reduction",
         1000.0, r"reduction ($\times 10^{-3}$)"),
    ]
    datasets = [("zsre", "ZsRE", PURPLE), ("cf", "CounterFact", SSR)]
    for index, (metric, title, scale, xlabel) in enumerate(geometry_specs):
        sub = axh.inset_axes([0.02 + index * 0.50, 0.18, 0.43, 0.67])
        for ypos, (dataset_key, dataset_label, color) in enumerate(datasets[::-1]):
            summary = mechanism["datasets"][dataset_key]["paired_metrics"][metric]
            values = np.asarray(summary["all_seed_favorable_deltas"], dtype=float) * scale
            mean = float(summary["mean_favorable_delta"]) * scale
            low, high = np.asarray(summary["bootstrap_95_ci"], dtype=float) * scale
            jitter = np.linspace(-0.075, 0.075, len(values))
            sub.scatter(values, ypos + jitter, s=10, fc="white", ec=color,
                        lw=0.45, alpha=0.75, zorder=2)
            sub.errorbar(mean, ypos, xerr=[[mean - low], [high - mean]], fmt="o",
                         ms=4.7, mfc=color, mec=INK, mew=0.4, ecolor=color,
                         capsize=1.8, lw=0.8, zorder=3)
        sub.axvline(0, color=LIGHT, lw=0.55)
        sub.set_yticks([0, 1], ["CounterFact", "ZsRE"] if index == 0 else [])
        sub.set_xlabel(xlabel)
        sub.set_title(title, fontsize=5.0, pad=2)
        clean(sub)
    source_line(
        axh,
        "ZsRE and CounterFact | Qwen2.5-7B dense updates | 100 edits | n=4/dataset",
        y=-0.34,
    )
    clean(axh)

    fig.subplots_adjust(left=0.070, right=0.992, top=0.978, bottom=0.055)
    save(fig, "Fig4_final_strict")


def build_figure5_legacy() -> None:
    rows = read_rows("figure5_source_data.csv")
    voc_rows = rows
    stage_rows = read_rows("figure5_stage_data.csv")
    fig = plt.figure(figsize=(WIDTH, HEIGHT5), facecolor="white")
    gs = fig.add_gridspec(4, 12, height_ratios=[0.98, 0.94, 0.94, 0.72],
                          hspace=0.68, wspace=1.28)

    # Low-rank KE leads the figure and uses the same endpoint definitions as Fig. 4.
    axa = fig.add_subplot(gs[0, :7])
    panel_label(axa, "a", "LoRA knowledge editing across rank")
    axa.axis("off")
    rank_colors = {8: SSR, 16: ORANGE, 32: PURPLE, 64: CONTROL}
    cosmos = json.loads(
        (ROOT / "source_data/ke_lora_cosmos/summary.json").read_text(encoding="utf-8")
    )
    v36 = json.loads(
        (ROOT / "source_data/ke_lora_v36/confirmation_summary.json").read_text(encoding="utf-8")
    )
    metric_specs = [
        ("immediate_efficacy", "Immediate efficacy", (-8.0, 21.5)),
        ("immediate_rephrase", "Rephrase", (-7.0, 16.0)),
    ]
    for idx, (metric, title, ylim) in enumerate(metric_specs):
        sub = axa.inset_axes([0.015 + idx * 0.505, 0.13, 0.455, 0.72])
        means = []
        for xpos, rank in enumerate((8, 16, 32, 64)):
            summary = cosmos["cohorts"]["independent_100"]["rank_summaries"][str(rank)]
            endpoint = summary["curves"][-1]["metrics"][metric]
            mean = float(endpoint["delta_mean"])
            low, high = (float(value) for value in endpoint["delta_ci95"])
            means.append(mean)
            sub.errorbar(
                xpos, mean, yerr=[[mean - low], [high - mean]], fmt="o", ms=4.2,
                mfc=rank_colors[rank], mec=INK, mew=0.35, color=rank_colors[rank],
                ecolor=rank_colors[rank], capsize=1.6, lw=0.75, zorder=3,
            )
        sub.plot(range(4), means, color=GRAY, lw=0.65, zorder=0)
        endpoint_250 = cosmos["cohorts"]["rank32_shared_250"]["rank_summaries"]["32"]
        endpoint_250 = endpoint_250["curves"][-1]["metrics"][metric]
        mean_250 = float(endpoint_250["delta_mean"])
        low_250, high_250 = (float(value) for value in endpoint_250["delta_ci95"])
        sub.errorbar(
            2.18, mean_250,
            yerr=[[mean_250 - low_250], [high_250 - mean_250]],
            fmt="D", ms=4.0, mfc="white", mec=PURPLE, mew=0.8, color=PURPLE,
            ecolor=PURPLE, capsize=1.6, lw=0.75, zorder=4,
        )
        endpoint_250_r8 = v36["rank_summaries"]["8"]["curves"][-1]["metrics"][metric]
        mean_250_r8 = float(endpoint_250_r8["delta_mean"])
        low_250_r8, high_250_r8 = (float(value) for value in endpoint_250_r8["delta_ci95"])
        sub.errorbar(
            0.18, mean_250_r8,
            yerr=[[mean_250_r8 - low_250_r8], [high_250_r8 - mean_250_r8]],
            fmt="D", ms=4.0, mfc="white", mec=SSR, mew=0.8, color=SSR,
            ecolor=SSR, capsize=1.6, lw=0.75, zorder=4,
        )
        sub.axhline(0, color=INK, lw=0.55)
        sub.set_xticks(range(4), ["8", "16", "32", "64"])
        sub.set_xlabel("LoRA rank")
        if idx == 0:
            sub.set_ylabel("gain over LoRA (pp)")
        sub.set_ylim(*ylim)
        sub.set_title(title, fontsize=5.2, pad=2.0)
        clean(sub)
    axa.text(0.98, 0.96, "circles: 100 edits | diamonds: 250 edits | n=10",
             transform=axa.transAxes, ha="right", va="top", fontsize=5.0,
             color=GRAY)

    axb = fig.add_subplot(gs[0, 7:])
    panel_label(axb, "b", "CUB200 rank response")
    rank_markers = {8: "o", 16: "s", 32: "^"}
    rank_colors = {8: GREEN, 16: ORANGE, 32: SSR}
    trajectory = []
    for rank in (8, 16, 32):
        aa = metric_row(rows, "c", "shared cosine", "aa_gain_pp", rank)
        af = metric_row(rows, "c", "shared cosine", "af_reduction_pp", rank)
        x, y = number(aa, "paired_change"), number(af, "paired_change")
        trajectory.append((x, y))
        axb.errorbar(x, y,
                     xerr=[[x-number(aa, "ci_low")], [number(aa, "ci_high")-x]],
                     yerr=[[y-number(af, "ci_low")], [number(af, "ci_high")-y]],
                     fmt=rank_markers[rank], ms=5.0, mfc=rank_colors[rank],
                     mec=INK, mew=0.4, ecolor=rank_colors[rank], capsize=1.7, lw=0.75)
        axb.annotate(str(rank), (x, y), xytext=(5, 6 if rank != 16 else -10),
                     textcoords="offset points", fontsize=5.0, color=rank_colors[rank],
                     fontweight="bold")
    for start, end in zip(trajectory[:-1], trajectory[1:]):
        axb.annotate("", xy=end, xytext=start,
                     arrowprops=dict(arrowstyle="->", lw=0.8, color=GRAY, alpha=0.75))
    axb.axhline(0, color=LIGHT, lw=0.6); axb.axvline(0, color=LIGHT, lw=0.6)
    axb.set_xlabel("AA gain vs KD+LowRank (pp)")
    axb.set_ylabel("AF reduction vs KD+LowRank (pp)")
    clean(axb)

    axc = fig.add_subplot(gs[1, :4])
    panel_label(axc, "c", "Rank-16 target factorization")
    groups = ["prototype only", "basis only", "joint"]
    markers = {"prototype only": "o", "basis only": "s", "joint": "D"}
    colors = {"prototype only": CONTROL, "basis only": ORANGE, "joint": SSR}
    for group in groups:
        aa = metric_row(rows, "b", group, "aa_gain_pp")
        af = metric_row(rows, "b", group, "af_reduction_pp")
        x, y = number(aa, "paired_change"), number(af, "paired_change")
        axc.errorbar(x, y,
                     xerr=[[x-number(aa, "ci_low")], [number(aa, "ci_high")-x]],
                     yerr=[[y-number(af, "ci_low")], [number(af, "ci_high")-y]],
                     fmt=markers[group], ms=4.8, mfc=colors[group], mec=INK, mew=0.35,
                     ecolor=colors[group], capsize=1.7, lw=0.75)
        short = {"prototype only": "P", "basis only": "B", "joint": "P+B"}[group]
        offsets = {"prototype only": (5, -9), "basis only": (5, 7), "joint": (6, 7)}
        axc.annotate(short, (x, y), xytext=offsets[group], textcoords="offset points",
                     fontsize=5.0, color=colors[group],
                     fontweight="bold" if group == "joint" else "normal")
    axc.axhline(0, color=LIGHT, lw=0.6); axc.axvline(0, color=LIGHT, lw=0.6)
    axc.set_xlabel("AA gain vs KD+LowRank (pp)")
    axc.set_ylabel("AF reduction vs KD+LowRank (pp)")
    clean(axc)

    axd = fig.add_subplot(gs[1, 4:8])
    panel_label(axd, "d", "Rank-32 operating point")
    coordinates = []
    for group, color, marker in [("KD+LowRank", CONTROL, "o"), ("KD+LowRank+SSR", SSR, "D")]:
        aa = metric_row(rows, "d", group, "AA (%)", 32)
        af = metric_row(rows, "d", group, "AF (%)", 32)
        x, y = number(aa, "paired_change"), number(af, "paired_change")
        coordinates.append((x, y))
        axd.errorbar(x, y,
                     xerr=[[x-number(aa, "ci_low")], [number(aa, "ci_high")-x]],
                     yerr=[[y-number(af, "ci_low")], [number(af, "ci_high")-y]],
                     fmt=marker, ms=5.2, mfc=color, mec=INK, mew=0.4, ecolor=color,
                     capsize=1.7, lw=0.75)
        label = "KD+LowRank" if group == "KD+LowRank" else "+SSR"
        offset = (6, -12) if group == "KD+LowRank" else (-6, 8)
        axd.annotate(label, (x, y), xytext=offset, textcoords="offset points",
                     ha="left" if group == "KD+LowRank" else "right",
                     fontsize=5.0, color=color, fontweight="bold" if "SSR" in group else "normal")
    axd.annotate("", xy=coordinates[1], xytext=coordinates[0],
                 arrowprops=dict(arrowstyle="->", lw=1.0, color=SSR))
    axd.invert_yaxis()
    axd.set_xlabel("average accuracy, AA (%)")
    axd.set_ylabel("average forgetting, AF (%) ↓")
    clean(axd)

    axe = fig.add_subplot(gs[1, 8:])
    panel_label(axe, "e", "Seed-level rank effects")
    ranks = [8, 16, 32]
    with (ROOT / "source_data/figure_panels/fig6_adapter_seed_level.csv").open(
            newline="", encoding="utf-8") as handle:
        seed_rows = list(csv.DictReader(handle))
    jitter = np.array([-0.055, 0.030, -0.020, 0.050, -0.040,
                       0.010, -0.010, 0.040, -0.030, 0.055])
    for xpos, rank in enumerate(ranks):
        for offset, metric, color, marker in [
            (-0.16, "aa_gain_pp", SSR, "o"),
            (0.16, "af_reduction_pp", GREEN, "s"),
        ]:
            values = np.array([
                float(row[metric]) for row in seed_rows
                if int(row["rank"]) == rank and row["condition"] == "gaussian_all_cosine"
            ])
            if len(values) != 10:
                raise ValueError(f"Expected ten rank-{rank} seed values for {metric}")
            summary = metric_row(rows, "e", "rank budget", metric, rank)
            mean = number(summary, "paired_change")
            q1, median, q3 = np.percentile(values, [25, 50, 75])
            center = xpos + offset
            axe.scatter(center + jitter, values, s=12, marker=marker, fc=color,
                        ec="white", lw=0.25, alpha=0.58, zorder=2)
            axe.add_patch(Rectangle((center - 0.065, q1), 0.13, q3 - q1,
                                    fc=color, ec=INK, lw=0.45, alpha=0.18, zorder=1))
            axe.plot([center - 0.065, center + 0.065], [median, median],
                     color=color, lw=0.85, zorder=3)
            axe.errorbar(center, mean,
                         yerr=[[mean-number(summary, "ci_low")],
                               [number(summary, "ci_high")-mean]],
                         fmt=marker, ms=4.8, mfc=color, mec=INK, mew=0.4,
                         ecolor=color, capsize=1.7, lw=0.78, zorder=4)
    axe.axhline(0, color=INK, lw=0.55)
    axe.set_xticks(np.arange(len(ranks)), [str(rank) for rank in ranks])
    axe.set_xlabel("adapter rank")
    axe.set_ylabel("favourable paired change (pp)")
    axe.set_xlim(-0.48, 2.48); axe.set_ylim(-0.08, 2.05)
    axe.text(0.98, 0.96, "circle: AA | square: AF reduction",
             transform=axe.transAxes, ha="right", va="top", fontsize=4.6, color=GRAY)
    clean(axe)

    axf = fig.add_subplot(gs[2, :5])
    panel_label(axf, "f", "Low-rank PASCAL VOC retention")
    endpoint_specs = [
        ("all-mIoU trajectory AUC gain (pp)", "all-mIoU AUC", "o"),
        ("old-mIoU trajectory AUC gain (pp)", "old-mIoU AUC", "o"),
        ("final all-class mIoU gain (pp)", "final all mIoU", "s"),
        ("final old-class mIoU gain (pp)", "final old mIoU", "s"),
        ("final old-class Boundary IoU gain (pp)", "final old BIoU", "D"),
    ]
    y_positions = np.arange(len(endpoint_specs))[::-1]
    for y, (metric, label, marker) in zip(y_positions, endpoint_specs):
        row = next(record for record in voc_rows
                   if record["panel"] == "f" and record["primary_metric"] == metric)
        error_point(axf, row, y, SSR, marker, ms=4.3)
        axf.text(number(row, "ci_high") + 0.035, y,
                 f"{number(row, 'paired_change'):+.2f}", va="center",
                 fontsize=4.8, color=SSR, fontweight="bold")
    axf.axvline(0, color=INK, lw=0.60)
    axf.set_yticks(y_positions, [label for _, label, _ in endpoint_specs])
    axf.set_xlim(-0.12, 1.55); axf.set_ylim(-0.55, 4.60)
    axf.set_xlabel("LowRank+SSR minus LowRank (pp)")
    clean(axf, "x")

    axg = fig.add_subplot(gs[2, 5:])
    panel_label(axg, "g", "Stage-wise direct SSR effect")
    axg.axis("off")
    for idx, (metric, title, auc_text) in enumerate([
        ("all_miou", "All classes", "AUC +1.07 pp"),
        ("old_miou", "Old classes", "AUC +0.94 pp"),
    ]):
        sub = axg.inset_axes([0.02 + idx * 0.51, 0.13, 0.45, 0.72])
        selected = sorted([row for row in stage_rows if row["metric"] == metric],
                          key=lambda row: int(row["stage"]))
        stages = np.array([int(row["stage"]) for row in selected])
        means = np.array([float(row["mean"]) for row in selected])
        lows = np.array([float(row["ci95_low"]) for row in selected])
        highs = np.array([float(row["ci95_high"]) for row in selected])
        sub.fill_between(stages, lows, highs, color=SSR, alpha=0.12, linewidth=0)
        sub.plot(stages, means, color=SSR, marker="o", ms=2.8, lw=0.90)
        sub.axhline(0, color=INK, lw=0.55)
        sub.set_xlim(0.8, 10.2); sub.set_ylim(-1.8, 3.25)
        sub.set_xticks([1, 3, 5, 7, 10])
        sub.set_xlabel("incremental stage")
        sub.set_title(title, fontsize=5.2, pad=2.0)
        sub.text(0.98, 0.96, auc_text, transform=sub.transAxes, ha="right", va="top",
                 fontsize=4.8, color=SSR, fontweight="bold")
        if idx == 0:
            sub.set_ylabel("SSR effect (pp)")
        else:
            sub.set_yticklabels([])
        clean(sub)

    axh = fig.add_subplot(gs[3, :])
    panel_label(axh, "h", "Qualitative PASCAL VOC example", x=-0.035)
    axh.axis("off")
    qrow = next(row for row in voc_rows
                if row["panel"] == "h" and row["primary_metric"] == "pixel-level outcome")
    sample = ROOT / qrow["source_file"]
    panels = [
        np.asarray(Image.open(sample / "input.png").convert("RGB")),
        np.asarray(Image.open(sample / "gt_color.png").convert("RGB")),
        np.asarray(Image.open(sample / "lowrank_error.png").convert("RGB")),
        np.asarray(Image.open(sample / "lowrank_ssr_error.png").convert("RGB")),
        np.asarray(Image.open(sample / "change_map.png").convert("RGB")),
    ]
    titles = ["Input", "Ground truth", "LowRank error", "+SSR error", "Corrected / harmed"]
    for idx, (array, title) in enumerate(zip(panels, titles)):
        inset = axh.inset_axes([0.004 + idx * 0.199, 0.01, 0.190, 0.84])
        inset.imshow(array)
        inset.set_title(title, fontsize=5.2, pad=1.4, fontweight="bold")
        inset.axis("off")
        inset.add_patch(Rectangle((0, 0), 1, 1, transform=inset.transAxes,
                                  fill=False, ec=SSR if idx == 4 else "#C8CCCF",
                                  lw=0.85 if idx == 4 else 0.55))

    fig.subplots_adjust(left=0.095, right=0.988, top=0.972, bottom=0.035)
    save(fig, "Fig5_final_strict")


def colorize(mask: np.ndarray) -> np.ndarray:
    palette = np.array([
        [247, 247, 247], [63, 104, 145], [179, 112, 66], [92, 134, 99],
        [128, 104, 146], [188, 76, 73], [78, 145, 150], [196, 163, 74],
        [115, 115, 115], [98, 121, 160], [159, 92, 139], [121, 153, 78],
        [188, 128, 101], [89, 151, 128], [141, 115, 91], [85, 94, 104],
        [154, 132, 187], [216, 150, 70], [99, 158, 173], [69, 112, 170],
        [184, 105, 105],
    ], dtype=np.uint8)
    return palette[np.asarray(mask, dtype=np.int64) % len(palette)]


def change_map(gt: np.ndarray, control: np.ndarray, treatment: np.ndarray) -> np.ndarray:
    corrected = (control != gt) & (treatment == gt)
    harmed = (control == gt) & (treatment != gt)
    unchanged = ~(corrected | harmed)
    output = np.full((*gt.shape, 3), 242, dtype=np.uint8)
    output[unchanged & (gt != 0)] = [205, 210, 214]
    foreground = gt != 0
    interior = np.zeros_like(foreground)
    interior[1:-1, 1:-1] = (foreground[1:-1, 1:-1]
                             & foreground[:-2, 1:-1] & foreground[2:, 1:-1]
                             & foreground[1:-1, :-2] & foreground[1:-1, 2:])
    output[foreground & ~interior & unchanged] = [145, 151, 155]
    output[corrected] = [35, 105, 176]
    output[harmed] = [205, 91, 48]
    return output


def target_class_overlay(image: np.ndarray, color_mask: np.ndarray,
                         target_rgb: tuple[int, int, int],
                         fill_rgb: tuple[int, int, int],
                         alpha: float = 0.58) -> np.ndarray:
    """Overlay one semantic class while retaining the underlying photograph."""
    image = np.asarray(image, dtype=np.uint8)
    color_mask = np.asarray(color_mask, dtype=np.uint8)
    selected = np.all(color_mask == np.asarray(target_rgb, dtype=np.uint8), axis=-1)
    output = image.astype(float)
    fill = np.asarray(fill_rgb, dtype=float)
    output[selected] = (1.0 - alpha) * output[selected] + alpha * fill

    # A one-pixel contour keeps the prediction legible over textured imagery.
    interior = np.zeros_like(selected)
    interior[1:-1, 1:-1] = (selected[1:-1, 1:-1]
                             & selected[:-2, 1:-1] & selected[2:, 1:-1]
                             & selected[1:-1, :-2] & selected[1:-1, 2:])
    boundary = selected & ~interior
    output[boundary] = 0.32 * output[boundary] + 0.68 * np.array([35, 43, 48])
    return np.clip(output, 0, 255).astype(np.uint8)


def max_window_box(score: np.ndarray, fraction: float) -> tuple[int, int, int, int]:
    """Return the fixed-size crop with the largest summed evidence score."""
    height, width = score.shape
    window_h = max(8, int(round(height * fraction)))
    window_w = max(8, int(round(width * fraction)))
    integral = np.pad(np.asarray(score, dtype=float).cumsum(0).cumsum(1),
                      ((1, 0), (1, 0)))
    sums = (integral[window_h:, window_w:] - integral[:-window_h, window_w:]
            - integral[window_h:, :-window_w] + integral[:-window_h, :-window_w])
    y0, x0 = np.unravel_index(np.argmax(sums), sums.shape)
    return int(x0), int(y0), int(x0 + window_w), int(y0 + window_h)


def render_pdf(path: Path, scale: float = 2.0) -> np.ndarray:
    document = fitz.open(path)
    pixmap = document[0].get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    array = np.frombuffer(pixmap.samples, dtype=np.uint8)
    return array.reshape(pixmap.height, pixmap.width, pixmap.n)[..., :3]


def render_pdf_image_row(path: Path, row_index: int) -> list[np.ndarray]:
    """Extract the five original image objects without rasterizing the PDF page."""
    document = fitz.open(path)
    page = document[0]
    images = page.get_images(full=True)
    selected = images[row_index * 5:(row_index + 1) * 5]
    if len(selected) != 5:
        raise ValueError(f"Expected five gallery tiles in row {row_index}: {path}")
    tiles = []
    for image in selected:
        pixmap = fitz.Pixmap(document, image[0])
        if pixmap.colorspace is None or pixmap.colorspace.n != 3 or pixmap.alpha:
            pixmap = fitz.Pixmap(fitz.csRGB, pixmap)
        array = np.frombuffer(pixmap.samples, dtype=np.uint8)
        tiles.append(array.reshape(pixmap.height, pixmap.width, pixmap.n)[..., :3])
    return tiles


def build_figure6_legacy() -> None:
    rows = read_rows("figure6_source_data.csv")
    fig = plt.figure(figsize=(WIDTH, HEIGHT6), facecolor="white")
    gs = fig.add_gridspec(4, 12, height_ratios=[1.08, 0.94, 0.88, 0.88],
                          hspace=0.56, wspace=1.25)

    axa = fig.add_subplot(gs[0, :7])
    panel_label(axa, "a", "Full-rank geometry regularizers")
    taxonomy = [row for row in rows if row["panel"] == "a"]
    family_order = ["SSR", "center-only", "surround-only", "cosine orthogonality",
                    "prototype decorrelation", "spectral flattening"]
    family_style = {
        "SSR": (SSR, "D", 5.8), "center-only": (GREEN, "o", 4.6),
        "surround-only": (ORANGE, "s", 4.6),
        "cosine orthogonality": ("#AEB3B6", "^", 4.2),
        "prototype decorrelation": ("#BFC3C5", "v", 4.2),
        "spectral flattening": (PURPLE, "P", 4.4),
    }
    labels = {
        "SSR": "SSR", "center-only": "center", "surround-only": "surround",
        "cosine orthogonality": "orthogonal", "prototype decorrelation": "prototype",
        "spectral flattening": "spectral",
    }
    offsets = {
        "SSR": (6, 7), "center-only": (6, 8), "surround-only": (-24, 7),
        "cosine orthogonality": (-5, -12), "prototype decorrelation": (6, 8),
        "spectral flattening": (6, -11),
    }
    axa.axvline(0, color=INK, lw=0.60); axa.axhline(0, color=INK, lw=0.60)
    for family in family_order:
        aa = metric_row(taxonomy, "a", family, "delta_aa")
        af = metric_row(taxonomy, "a", family, "forgetting_reduction")
        x, y = number(aa, "paired_change"), number(af, "paired_change")
        color, marker, ms = family_style[family]
        axa.errorbar(x, y,
                     xerr=[[x-number(aa, "ci_low")], [number(aa, "ci_high")-x]],
                     yerr=[[y-number(af, "ci_low")], [number(af, "ci_high")-y]],
                     fmt=marker, ms=ms, mfc=color, mec=INK, mew=0.38,
                     ecolor=color, capsize=1.6, lw=0.72,
                     alpha=1.0 if family == "SSR" else 0.86)
        if family == "SSR":
            axa.scatter(x, y, s=112, fc="none", ec=SSR, lw=1.0, zorder=2)
        axa.annotate(labels[family], (x, y), xytext=offsets[family],
                     textcoords="offset points", fontsize=5.0, color=color,
                     fontweight="bold" if family == "SSR" else "normal")
    axa.set_xlim(-0.13, 0.105); axa.set_ylim(-0.105, 0.122)
    axa.set_xlabel("AA gain versus KD (pp)")
    axa.set_ylabel("AF reduction versus KD (pp)")
    clean(axa)

    axb = fig.add_subplot(gs[0, 7:])
    panel_label(axb, "b", "SSR seed-level effects")
    with (ROOT / "source_data/figure_panels/fig4_cub_taxonomy_seed_level.csv").open(
            newline="", encoding="utf-8") as handle:
        seed_rows = [row for row in csv.DictReader(handle) if row["family"] == "ssr"]
    jitter = np.linspace(-0.14, 0.14, len(seed_rows))
    for y, (field, metric, label, color, marker) in enumerate([
        ("delta_aa", "delta_aa", "AA gain", SSR, "o"),
        ("forgetting_reduction", "forgetting_reduction", "AF reduction", GREEN, "s"),
    ][::-1]):
        values = np.asarray([float(row[field]) for row in seed_rows])
        summary = metric_row(taxonomy, "a", "SSR", metric)
        mean = number(summary, "paired_change")
        q1, median, q3 = np.percentile(values, [25, 50, 75])
        axb.scatter(values, y + jitter, s=12, marker=marker, fc=color,
                    ec="white", lw=0.25, alpha=0.58, zorder=2)
        axb.plot([q1, q3], [y, y], color=color, lw=4.0, alpha=0.20,
                 solid_capstyle="round", zorder=1)
        axb.plot([median, median], [y - 0.13, y + 0.13], color=color, lw=0.8)
        axb.errorbar(mean, y,
                     xerr=[[mean-number(summary, "ci_low")],
                           [number(summary, "ci_high")-mean]],
                     fmt=marker, ms=4.7, mfc=color, mec=INK, mew=0.4,
                     ecolor=color, capsize=1.7, lw=0.78, zorder=4)
    axb.axvline(0, color=INK, lw=0.60)
    axb.set_yticks([0, 1], ["AF reduction", "AA gain"])
    axb.set_xlim(-0.13, 0.23); axb.set_ylim(-0.43, 1.43)
    axb.set_xlabel("favourable paired change (pp)")
    axb.text(0.98, 0.96, "n=20 independent seeds", transform=axb.transAxes,
             ha="right", va="top", fontsize=4.8, color=GRAY)
    clean(axb, "x")

    axc = fig.add_subplot(gs[1, :6])
    panel_label(axc, "c", "Accuracy advantage on fixed KD")
    regularizers = [row for row in rows if row["panel"] == "c"]
    method_colors = {"EWC": CONTROL, "MAS": GREEN, "SI": PURPLE}
    methods = ["EWC", "MAS", "SI"]
    datasets = ["Split-CIFAR-100", "TinyImageNet"]
    base_positions = {"Split-CIFAR-100": np.array([5.2, 4.2, 3.2]),
                      "TinyImageNet": np.array([2.0, 1.0, 0.0])}
    for dataset in datasets:
        for y, method in zip(base_positions[dataset], methods):
            row = next(record for record in regularizers
                       if record["dataset"] == dataset and record["external_method"] == method)
            error_point(axc, row, y, method_colors[method], "o", ms=4.5)
    axc.axvline(0, color=INK, lw=0.60)
    axc.axhline(2.62, color=LIGHT, lw=0.65)
    axc.set_yticks(np.r_[base_positions["Split-CIFAR-100"], base_positions["TinyImageNet"]],
                   methods + methods)
    axc.text(13.8, 5.55, "Split-CIFAR-100", ha="right", va="top",
             fontsize=5.2, fontweight="bold")
    axc.text(13.8, 2.36, "TinyImageNet", ha="right", va="top",
             fontsize=5.2, fontweight="bold")
    axc.set_xlim(-0.35, 14.0); axc.set_ylim(-0.55, 5.75)
    axc.set_xlabel("paired AA advantage of SSR (pp)")
    clean(axc, "x")

    axd = fig.add_subplot(gs[1, 6:])
    panel_label(axd, "d", "Complete-objective mask transfer")
    structured = [row for row in rows if row["panel"] == "d"]
    seed_rows = read_rows("figure6_structured_seed_data.csv")
    structured_specs = [
        ("mIoU gain (pp)", "mIoU gain", SSR),
        ("Dice gain (pp)", "Dice gain", PURPLE),
        ("IoU forgetting reduction (pp)", "forgetting reduction", GREEN),
    ]
    for y, (metric, label, color) in zip([2, 1, 0], structured_specs):
        row = next(record for record in structured if record["primary_metric"] == metric)
        raw = [float(record["favorable_change_pp"]) for record in seed_rows
               if record["metric"] == metric]
        axd.scatter(raw, np.full(len(raw), y), s=13, color=color, alpha=0.45,
                    edgecolor="white", linewidth=0.25, zorder=2)
        error_point(axd, row, y, color, "D", ms=4.6)
        axd.text(number(row, "ci_high") + 0.12, y,
                 f"{number(row, 'paired_change'):+.2f}", va="center",
                 fontsize=5.0, color=color, fontweight="bold")
    axd.axvline(0, color=INK, lw=0.60)
    axd.set_yticks([2, 1, 0], ["mIoU", "Dice", "IoU forgetting"])
    axd.set_xlim(-0.2, 5.8); axd.set_ylim(-0.55, 2.70)
    axd.set_xlabel("favourable paired change (pp)")
    axd.text(0.98, 0.98, "CUB200 masks · n=3", transform=axd.transAxes,
             ha="right", va="top", fontsize=5.0, color=GRAY)
    clean(axd, "x")

    bird_row = next(row for row in rows if row["point_id"] == "cub_response_gallery_qualitative")
    gallery_path = ROOT / bird_row["source_file"]
    axe = fig.add_subplot(gs[2, :])
    panel_label(axe, "e", "Localized response reorganization I", x=-0.035)
    axe.set_xlim(0, 1); axe.set_ylim(0, 1); axe.axis("off")
    tiles = render_pdf_image_row(gallery_path, row_index=0)
    titles = ["Input", "GT mask", "KD", "SSR+KD", "Difference"]
    for idx, (tile, title) in enumerate(zip(tiles, titles)):
        x0 = 0.006 + idx * 0.199
        inset = axe.inset_axes([x0, 0.035, 0.190, 0.84])
        inset.imshow(tile)
        inset.set_title(title, fontsize=5.2, pad=1.4, fontweight="bold")
        inset.axis("off")
        inset.add_patch(Rectangle((0, 0), 1, 1, transform=inset.transAxes,
                                  fill=False, ec=SSR if idx == 4 else "#C8CCCF",
                                  lw=0.85 if idx == 4 else 0.55))

    axf = fig.add_subplot(gs[3, :])
    panel_label(axf, "f", "Localized response reorganization II", x=-0.035)
    axf.set_xlim(0, 1); axf.set_ylim(0, 1)
    axf.axis("off")
    tiles = render_pdf_image_row(gallery_path, row_index=1)
    for idx, (array, title) in enumerate(zip(tiles, titles)):
        inset = axf.inset_axes([0.004 + idx * 0.199, 0.025, 0.190, 0.85])
        inset.imshow(array)
        inset.set_title(title, fontsize=5.2, pad=1.4, fontweight="bold")
        inset.axis("off")
        inset.add_patch(Rectangle((0, 0), 1, 1, transform=inset.transAxes,
                                  fill=False, ec=SSR if idx == 4 else "#C8CCCF",
                                  lw=0.85 if idx == 4 else 0.55))

    fig.subplots_adjust(left=0.098, right=0.988, top=0.970, bottom=0.040)
    save(fig, "Fig6_final_strict")


def build_figure5() -> None:
    """Teacher-aligned Fig. 5: geometry family, low-rank CL, then confirmed LoRA KE."""
    rows = read_rows("figure5_source_data.csv")
    geometry_rows = read_rows("figure6_source_data.csv")
    fig = plt.figure(figsize=(WIDTH, HEIGHT5), facecolor="white")
    gs = fig.add_gridspec(3, 12, height_ratios=[1.05, 1.00, 0.92],
                          hspace=0.62, wspace=1.15)

    axa = fig.add_subplot(gs[0, :6])
    panel_label(axa, "a", "Full-rank geometry regularizers", x=-0.13)
    taxonomy = [row for row in geometry_rows if row["panel"] == "a"]
    family_order = ["SSR", "center-only", "surround-only", "cosine orthogonality",
                    "prototype decorrelation", "spectral flattening"]
    family_style = {
        "SSR": (SSR, "D", 5.8), "center-only": (GREEN, "o", 4.5),
        "surround-only": (ORANGE, "s", 4.5),
        "cosine orthogonality": ("#737B80", "^", 4.1),
        "prototype decorrelation": ("#8A9296", "v", 4.1),
        "spectral flattening": (PURPLE, "P", 4.3),
    }
    short_labels = {
        "SSR": "SSR", "center-only": "center", "surround-only": "surround",
        "cosine orthogonality": "orthogonal", "prototype decorrelation": "prototype",
        "spectral flattening": "spectral",
    }
    offsets = {
        "SSR": (6, 7), "center-only": (5, 8), "surround-only": (-24, 7),
        "cosine orthogonality": (-5, -12), "prototype decorrelation": (5, 8),
        "spectral flattening": (5, -11),
    }
    axa.axvline(0, color=INK, lw=0.60); axa.axhline(0, color=INK, lw=0.60)
    for family in family_order:
        aa = metric_row(taxonomy, "a", family, "delta_aa")
        af = metric_row(taxonomy, "a", family, "forgetting_reduction")
        x, y = number(aa, "paired_change"), number(af, "paired_change")
        color, marker, ms = family_style[family]
        axa.errorbar(
            x, y,
            xerr=[[x-number(aa, "ci_low")], [number(aa, "ci_high")-x]],
            yerr=[[y-number(af, "ci_low")], [number(af, "ci_high")-y]],
            fmt=marker, ms=ms, mfc=color, mec=INK, mew=0.38,
            ecolor=color, capsize=1.6, lw=0.72,
            alpha=1.0 if family == "SSR" else 0.92,
        )
        if family == "SSR":
            axa.scatter(x, y, s=112, fc="none", ec=SSR, lw=1.0, zorder=2)
        axa.annotate(short_labels[family], (x, y), xytext=offsets[family],
                     textcoords="offset points", fontsize=5.0, color=color,
                     fontweight="bold" if family == "SSR" else "normal")
    axa.set_xlim(-0.13, 0.105); axa.set_ylim(-0.105, 0.122)
    axa.set_xlabel("AA gain versus KD (pp)")
    axa.set_ylabel("AF reduction versus KD (pp)")
    clean(axa)

    axb = fig.add_subplot(gs[0, 6:])
    panel_label(axb, "b", "Rank-16 target factorization", x=-0.13)
    target_style = {
        "prototype only": (CONTROL, "o", "P"),
        "basis only": (ORANGE, "s", "B"),
        "joint": (SSR, "D", "P+B"),
    }
    axb.axvline(0, color=LIGHT, lw=0.6); axb.axhline(0, color=LIGHT, lw=0.6)
    for group in ("prototype only", "basis only", "joint"):
        aa = metric_row(rows, "b", group, "aa_gain_pp")
        af = metric_row(rows, "b", group, "af_reduction_pp")
        x, y = number(aa, "paired_change"), number(af, "paired_change")
        color, marker, label = target_style[group]
        axb.errorbar(
            x, y,
            xerr=[[x-number(aa, "ci_low")], [number(aa, "ci_high")-x]],
            yerr=[[y-number(af, "ci_low")], [number(af, "ci_high")-y]],
            fmt=marker, ms=5.0, mfc=color, mec=INK, mew=0.4,
            ecolor=color, capsize=1.7, lw=0.75,
        )
        axb.annotate(label, (x, y), xytext=(5, 6), textcoords="offset points",
                     fontsize=5.0, color=color,
                     fontweight="bold" if group == "joint" else "normal")
    axb.set_xlim(-0.05, 1.08); axb.set_ylim(-0.22, 1.12)
    axb.set_xlabel("AA gain vs KD+LowRank (pp)")
    axb.set_ylabel("AF reduction vs KD+LowRank (pp)")
    clean(axb)

    axc = fig.add_subplot(gs[1, :4])
    panel_label(axc, "c", "Rank response", x=-0.16)
    rank_markers = {8: "o", 16: "s", 32: "^"}
    rank_colors = {8: GREEN, 16: ORANGE, 32: SSR}
    trajectory = []
    for rank in (8, 16, 32):
        aa = metric_row(rows, "c", "shared cosine", "aa_gain_pp", rank)
        af = metric_row(rows, "c", "shared cosine", "af_reduction_pp", rank)
        x, y = number(aa, "paired_change"), number(af, "paired_change")
        trajectory.append((x, y))
        axc.errorbar(
            x, y,
            xerr=[[x-number(aa, "ci_low")], [number(aa, "ci_high")-x]],
            yerr=[[y-number(af, "ci_low")], [number(af, "ci_high")-y]],
            fmt=rank_markers[rank], ms=5.0, mfc=rank_colors[rank],
            mec=INK, mew=0.4, ecolor=rank_colors[rank], capsize=1.7, lw=0.75,
        )
        axc.annotate(str(rank), (x, y), xytext=(5, 6 if rank != 16 else -10),
                     textcoords="offset points", fontsize=5.0, color=rank_colors[rank],
                     fontweight="bold")
    for start, end in zip(trajectory[:-1], trajectory[1:]):
        axc.annotate("", xy=end, xytext=start,
                     arrowprops=dict(arrowstyle="->", lw=0.8, color=GRAY, alpha=0.75))
    axc.axvline(0, color=LIGHT, lw=0.6); axc.axhline(0, color=LIGHT, lw=0.6)
    axc.set_xlim(-0.05, 1.75); axc.set_ylim(-0.08, 1.38)
    axc.set_xlabel("AA gain (pp)"); axc.set_ylabel("AF reduction (pp)")
    clean(axc)

    axd = fig.add_subplot(gs[1, 4:8])
    panel_label(axd, "d", "Rank-32 operating point", x=-0.16)
    control_aa = metric_row(rows, "d", "KD+LowRank", "AA (%)", 32)
    control_af = metric_row(rows, "d", "KD+LowRank", "AF (%)", 32)
    ssr_aa = metric_row(rows, "d", "KD+LowRank+SSR", "AA (%)", 32)
    ssr_af = metric_row(rows, "d", "KD+LowRank+SSR", "AF (%)", 32)
    x0, y0 = number(control_aa, "paired_change"), number(control_af, "paired_change")
    x1, y1 = number(ssr_aa, "paired_change"), number(ssr_af, "paired_change")
    axd.errorbar(x0, y0,
                 xerr=[[x0-number(control_aa, "ci_low")], [number(control_aa, "ci_high")-x0]],
                 yerr=[[y0-number(control_af, "ci_low")], [number(control_af, "ci_high")-y0]],
                 fmt="o", ms=5.0, mfc=CONTROL, mec=INK, mew=0.4,
                 ecolor=CONTROL, capsize=1.7, lw=0.75)
    axd.errorbar(x1, y1,
                 xerr=[[x1-number(ssr_aa, "ci_low")], [number(ssr_aa, "ci_high")-x1]],
                 yerr=[[y1-number(ssr_af, "ci_low")], [number(ssr_af, "ci_high")-y1]],
                 fmt="D", ms=5.3, mfc=SSR, mec=INK, mew=0.4,
                 ecolor=SSR, capsize=1.7, lw=0.75)
    axd.annotate("", xy=(x1, y1), xytext=(x0, y0),
                 arrowprops=dict(arrowstyle="->", lw=1.0, color=SSR))
    axd.annotate("LowRank", (x0, y0), xytext=(5, -11), textcoords="offset points",
                 fontsize=4.8, color=CONTROL)
    axd.annotate("+SSR", (x1, y1), xytext=(-4, 8), textcoords="offset points",
                 ha="right", fontsize=5.0, color=SSR, fontweight="bold")
    axd.set_xlim(78.55, 80.55); axd.set_ylim(3.78, 2.25)
    axd.set_xlabel("average accuracy, AA (%)")
    axd.set_ylabel("average forgetting, AF (%)")
    clean(axd)

    axe = fig.add_subplot(gs[1, 8:])
    panel_label(axe, "e", "Seed-level rank effects", x=-0.16)
    with (ROOT / "source_data/figure_panels/fig6_adapter_seed_level.csv").open(
            newline="", encoding="utf-8") as handle:
        seed_rows = list(csv.DictReader(handle))
    ranks = (8, 16, 32)
    jitter = np.array([-0.055, 0.030, -0.020, 0.050, -0.040,
                       0.010, -0.010, 0.040, -0.030, 0.055])
    for xpos, rank in enumerate(ranks):
        for offset, metric, color, marker in [
            (-0.16, "aa_gain_pp", SSR, "o"),
            (0.16, "af_reduction_pp", GREEN, "s"),
        ]:
            values = np.array([
                float(row[metric]) for row in seed_rows
                if int(row["rank"]) == rank and row["condition"] == "gaussian_all_cosine"
            ])
            summary = metric_row(rows, "e", "rank budget", metric, rank)
            mean = number(summary, "paired_change")
            q1, median, q3 = np.percentile(values, [25, 50, 75])
            center = xpos + offset
            axe.scatter(center + jitter, values, s=11, marker=marker, fc=color,
                        ec="white", lw=0.25, alpha=0.55, zorder=2)
            axe.add_patch(Rectangle((center - 0.065, q1), 0.13, q3 - q1,
                                    fc=color, ec=INK, lw=0.4, alpha=0.18, zorder=1))
            axe.plot([center - 0.065, center + 0.065], [median, median],
                     color=color, lw=0.8, zorder=3)
            axe.errorbar(center, mean,
                         yerr=[[mean-number(summary, "ci_low")],
                               [number(summary, "ci_high")-mean]],
                         fmt=marker, ms=4.6, mfc=color, mec=INK, mew=0.4,
                         ecolor=color, capsize=1.6, lw=0.75, zorder=4)
    axe.axhline(0, color=INK, lw=0.55)
    axe.set_xticks(np.arange(3), ["8", "16", "32"])
    axe.set_xlabel("adapter rank")
    axe.set_ylabel("favourable change (pp)")
    axe.set_xlim(-0.48, 2.48); axe.set_ylim(-0.08, 2.05)
    axe.text(0.98, 0.96, "circle  AA gain   |   square  AF reduction",
             transform=axe.transAxes, ha="right", va="top", fontsize=4.9, color=INK)
    clean(axe)

    axf = fig.add_subplot(gs[2, :])
    panel_label(axf, "f", "Confirmed low-rank knowledge editing", x=-0.035)
    axf.axis("off")
    point_specs = [
        ("rank 8, 100 edits", 0, "o", SSR, "rank 8 | 100 edits"),
        ("rank 8, 250 edits", 1, "D", SSR, "rank 8 | 250 edits"),
        ("rank 32, 250 edits", 2, "D", PURPLE, "rank 32 | 250 edits"),
    ]
    for idx, (metric, title, xlim) in enumerate([
        ("immediate efficacy gain (pp)", "Immediate efficacy", (0, 22.5)),
        ("rephrase gain (pp)", "Rephrase", (0, 14.5)),
    ]):
        sub = axf.inset_axes([0.025 + idx * 0.505, 0.10, 0.455, 0.76])
        for group, ypos, marker, color, label in point_specs:
            row = metric_row(rows, "a", group, metric, int(group.split()[1].rstrip(",")))
            error_point(sub, row, ypos, color, marker, ms=4.8)
            sub.text(number(row, "ci_high") + 0.20, ypos,
                     f"{number(row, 'paired_change'):+.2f}", va="center",
                     fontsize=4.9, color=color, fontweight="bold")
        sub.axvline(0, color=INK, lw=0.60)
        sub.set_yticks([0, 1, 2], [spec[4] for spec in point_specs] if idx == 0 else [])
        sub.set_xlim(*xlim); sub.set_ylim(-0.55, 2.55)
        sub.set_xlabel("LoRA+SSR - LoRA (pp)")
        sub.set_title(title, fontsize=5.2, pad=2.0)
        clean(sub, "x")

    fig.subplots_adjust(left=0.090, right=0.988, top=0.972, bottom=0.045)
    save(fig, "Fig5_final_strict")


def build_figure5_lowrank_ke() -> None:
    """Low-rank KE acquisition, preservation, and geometry evidence."""
    # Figure 5 uses a warmer SSR identity so the KE result does not read as a
    # continuation of the blue-green segmentation palette. Rank and cohort
    # accents remain color-blind distinguishable and are reinforced by shape.
    f5_control = "#7B858C"
    f5_ssr = "#B64E6F"
    f5_blue = "#3B75AF"
    f5_gold = "#D3942F"
    f5_purple = "#775AA6"
    v36 = read_json(ROOT / "source_data/ke_lora_v36/confirmation_summary.json")
    v37 = read_json(ROOT / "source_data/ke_lora_v37/confirmation_summary.json")
    cosmos = read_json(ROOT / "source_data/ke_lora_cosmos/summary.json")
    primary = cosmos["cohorts"]["independent_100"]["rank_summaries"]["8"]
    curves100 = {int(item["after_edits"]): item for item in primary["curves"]}
    curves37 = {int(item["after_edits"]): item for item in v37["curves"]}
    with (ROOT / "source_data/ke_lora_v37/seed_level_metrics.csv").open(
            newline="", encoding="utf-8") as handle:
        seed_rows37 = list(csv.DictReader(handle))
    with (ROOT / "source_data/ke_lora_cosmos/independent_100_seed_level.csv").open(
            newline="", encoding="utf-8") as handle:
        seed_rows100 = list(csv.DictReader(handle))

    def v36_metric(rank: int, metric: str = "immediate_efficacy",
                   edits: int = 250) -> dict:
        curve = next(item for item in v36["rank_summaries"][str(rank)]["curves"]
                     if int(item["after_edits"]) == edits)
        return curve["metrics"][metric]

    def rank32_metric(metric: str, edits: int = 250) -> dict:
        curves = cosmos["cohorts"]["rank32_shared_250"]["rank_summaries"]["32"]["curves"]
        curve = next(item for item in curves if int(item["after_edits"]) == edits)
        return curve["metrics"][metric]

    def seed_values37(arm: str, edits: int, metric: str) -> np.ndarray:
        selected = sorted(
            (row for row in seed_rows37
             if row["arm"] == arm and int(row["edits"]) == edits),
            key=lambda row: int(row["seed"]),
        )
        if len(selected) != 10:
            raise ValueError(f"Expected ten v37 seed rows for {arm}, {edits}, {metric}")
        return np.asarray([float(row[metric]) for row in selected], dtype=float)

    def seed_values100(arm: str, metric: str) -> np.ndarray:
        selected = sorted((row for row in seed_rows100 if row["arm"] == arm),
                          key=lambda row: int(row["seed"]))
        if len(selected) != 10:
            raise ValueError(f"Expected ten independent 100-edit rows for {arm}, {metric}")
        return np.asarray([float(row[metric]) for row in selected], dtype=float)

    def paired_absolute(ax, control_values: np.ndarray, ssr_values: np.ndarray,
                        summary: dict, ylabel: str, ylim: tuple[float, float],
                        delta_label: str) -> None:
        offsets = np.linspace(-0.035, 0.035, len(control_values))
        for offset, control_value, ssr_value in zip(offsets, control_values, ssr_values):
            line_color = f5_ssr if ssr_value >= control_value else f5_gold
            ax.plot([offset, 1 + offset], [control_value, ssr_value],
                    color=mpl.colors.to_rgba(line_color, 0.28), lw=0.62, zorder=1)
            ax.scatter([offset, 1 + offset], [control_value, ssr_value], s=10,
                       c=[f5_control, f5_ssr], edgecolor=INK, linewidth=0.22, zorder=2)
        for xpos, prefix, color, marker in ((0, "baseline", f5_control, "o"),
                                             (1, "ssr", f5_ssr, "D")):
            mean = summary[f"{prefix}_mean"]
            low, high = summary[f"{prefix}_ci95"]
            ax.errorbar(xpos, mean, yerr=[[mean - low], [high - mean]], fmt=marker,
                        ms=4.6, mfc=color, mec=INK, mew=0.40, ecolor=INK,
                        capsize=2.0, lw=0.78, zorder=4)
        ax.text(0.50, 0.98, delta_label, transform=ax.transAxes, ha="center",
                va="top", fontsize=5.0, color=f5_ssr if summary["delta_mean"] >= 0
                else "#9B724B", fontweight="bold")
        ax.set_xticks([0, 1], ["LoRA", "LoRA+SSR"])
        ax.set_xlim(-0.16, 1.16); ax.set_ylim(*ylim)
        ax.set_ylabel(ylabel, labelpad=0.8)
        clean(ax)

    def trajectory(ax, curve_map: dict[int, dict], family: str, metric: str, ylabel: str,
                   ylim: tuple[float, float], show_x: bool) -> None:
        edits = np.asarray(sorted(curve_map), dtype=float)
        for prefix, label, color, marker in (("baseline", "LoRA", f5_control, "o"),
                                               ("ssr", "LoRA+SSR", f5_ssr, "D")):
            values = np.asarray([
                curve_map[int(edit)][family][metric][f"{prefix}_mean"] for edit in edits
            ])
            lows = np.asarray([
                curve_map[int(edit)][family][metric][f"{prefix}_ci95"][0] for edit in edits
            ])
            highs = np.asarray([
                curve_map[int(edit)][family][metric][f"{prefix}_ci95"][1] for edit in edits
            ])
            ax.plot(edits, values, color=color, marker=marker, ms=2.5,
                    label=label, zorder=3)
            ax.fill_between(edits, lows, highs, color=color, alpha=0.09,
                            linewidth=0, zorder=1)
        ax.set_xscale("log")
        ax.set_xlim(0.8, max(edits) * 1.22); ax.set_ylim(*ylim)
        ax.set_xticks(edits, [str(int(edit)) for edit in edits] if show_x else [])
        if show_x:
            ax.set_xlabel("edits applied")
        ax.set_ylabel(ylabel, labelpad=0.8)
        clean(ax)

    fig = plt.figure(figsize=(WIDTH, HEIGHT5), facecolor="white")
    gs = fig.add_gridspec(3, 12, height_ratios=[1.03, 0.98, 1.03],
                          hspace=0.78, wspace=1.35)
    final100 = curves100[100]["metrics"]
    final37 = curves37[250]["performance"]["immediate_efficacy"]

    axa = fig.add_subplot(gs[0, :7])
    panel_label(axa, "a", "Independent acquisition-preservation confirmation", x=-0.08)
    axa.axis("off")
    eff_ax = axa.inset_axes([0.03, 0.15, 0.43, 0.73])
    paired_absolute(eff_ax,
                    seed_values100("LoRA", "immediate_efficacy_pct"),
                    seed_values100("LoRA+SSR", "immediate_efficacy_pct"),
                    final100["immediate_efficacy"],
                    "immediate efficacy (%)", (50, 94),
                    r"$\Delta$ efficacy +12.10 pp")
    loc100 = final100["immediate_locality_target_consistency"]
    loc_ax = axa.inset_axes([0.56, 0.15, 0.41, 0.73])
    paired_absolute(loc_ax,
                    seed_values100("LoRA", "immediate_locality_pct"),
                    seed_values100("LoRA+SSR", "immediate_locality_pct"),
                    loc100, "immediate locality (%)", (-0.12, 2.25),
                    r"$\Delta$ locality +0.05 pp (CI includes 0)")
    source_line(axa, "Qwen2.5-7B | ZsRE | rank 8 | 100 edits | n=10 paired seeds",
                y=-0.01)

    axb = fig.add_subplot(gs[0, 7:])
    panel_label(axb, "b", "Acquisition gain emerges with editing", x=-0.13)
    trajectory(axb, curves100, "metrics", "immediate_efficacy",
               "immediate efficacy (%)", (48, 104), True)
    axb.legend(frameon=False, loc="lower left", ncol=2, handletextpad=0.30,
               columnspacing=0.75, borderaxespad=0.0)
    source_line(axb, "Qwen2.5-7B | ZsRE | rank 8 | n=10", y=-0.30)

    axc = fig.add_subplot(gs[1, :7])
    panel_label(axc, "c", "Acquisition remains favorable across rank budgets", x=-0.08)
    ranks = (8, 16, 32, 64)
    rank_colors = {8: f5_ssr, 16: f5_blue, 32: f5_gold, 64: f5_purple}
    jitter = np.array([-0.095, -0.072, -0.050, -0.028, -0.006,
                       0.016, 0.038, 0.060, 0.082, 0.104])
    for xpos, rank in enumerate(ranks):
        metric = rank32_metric("immediate_efficacy") if rank == 32 else v36_metric(rank)
        values = np.asarray(metric["paired_deltas"], dtype=float)
        face = "white" if rank == 32 else rank_colors[rank]
        axc.scatter(xpos + jitter, values, s=12, fc=face, ec=rank_colors[rank],
                    lw=0.38, alpha=0.58, zorder=2)
        mean, (low, high) = metric["delta_mean"], metric["delta_ci95"]
        axc.errorbar(xpos, mean, yerr=[[mean-low], [high-mean]], fmt="D", ms=4.8,
                     mfc=face, mec=rank_colors[rank], mew=0.78,
                     ecolor=rank_colors[rank], capsize=1.8, lw=0.82, zorder=4)
        axc.text(xpos, high + 1.0, f"{mean:+.2f}", ha="center", va="bottom",
                 fontsize=4.9, color=rank_colors[rank], fontweight="bold")
    axc.axhline(0, color=LIGHT, lw=0.62)
    axc.set_xticks(range(4), ["8", "16", "32*", "64"])
    axc.set_xlabel("LoRA rank")
    axc.set_ylabel(r"$\Delta$ immediate efficacy (pp)", labelpad=0.8)
    axc.set_ylim(-21, 35)
    clean(axc)
    source_line(axc,
                "Qwen2.5-7B | ZsRE | 250 edits | n=10/rank | 32*: shared coefficient",
                y=-0.25)

    axd = fig.add_subplot(gs[1, 7:])
    panel_label(axd, "d", "Independent rank-8 acquisition confirmations", x=-0.13)
    replicate_metrics = [final100["immediate_efficacy"], v36_metric(8), final37]
    labels = ["100-edit cohort", "250-edit rank cohort", "250-edit mechanism cohort"]
    colors = [f5_ssr, f5_blue, f5_gold]
    for ypos, (metric, label, color) in enumerate(zip(replicate_metrics[::-1],
                                                       labels[::-1], colors[::-1])):
        values = np.asarray(metric["paired_deltas"], dtype=float)
        yjitter = ypos + np.linspace(-0.075, 0.075, len(values))
        axd.scatter(values, yjitter, s=11, fc=color, ec="white", lw=0.25,
                    alpha=0.42, zorder=2)
        mean, (low, high) = metric["delta_mean"], metric["delta_ci95"]
        axd.errorbar(mean, ypos, xerr=[[mean-low], [high-mean]], fmt="D", ms=5.0,
                     mfc=color, mec=INK, mew=0.4, ecolor=color, capsize=1.8,
                     lw=0.85, zorder=4)
        axd.text(high + 0.7, ypos, f"{mean:+.2f}", va="center", color=color,
                 fontsize=5.0, fontweight="bold")
    axd.axvline(0, color=LIGHT, lw=0.62)
    axd.set_yticks([0, 1, 2], labels[::-1])
    axd.set_xlabel(r"$\Delta$ immediate efficacy (pp)")
    axd.set_xlim(-10, 31); axd.set_ylim(-0.42, 2.42)
    clean(axd)
    source_line(axd, "Qwen2.5-7B | ZsRE | rank 8 | n=10/cohort",
                y=-0.26)

    axe = fig.add_subplot(gs[2, :7])
    panel_label(axe, "e", "Geometry forms by 100 edits and persists", x=-0.08)
    axe.axis("off")

    def geometry_endpoint(ax, edits: int, show_labels: bool) -> None:
        specs = [
            ("center_surround_contrast", "center-surround\ncontrast", 1.0),
            ("kernel_alignment", "kernel\nalignment", 0.0),
        ]
        for metric, label, ypos in specs:
            baseline = seed_values37("LoRA", edits, metric)
            ssr = seed_values37("LoRA+SSR", edits, metric)
            summary = curves37[edits]["geometry"][metric]
            jitter = np.linspace(-0.11, 0.11, len(baseline))
            for left, right, offset in zip(baseline, ssr, jitter):
                ax.plot([left, right], [ypos + offset, ypos + offset],
                        color=mpl.colors.to_rgba(f5_ssr, 0.20), lw=0.48, zorder=1)
                ax.scatter(left, ypos + offset, s=7.0, fc="white", ec=f5_control,
                           lw=0.36, zorder=2)
                ax.scatter(right, ypos + offset, s=7.5, fc=f5_ssr, ec="white",
                           lw=0.22, alpha=0.72, zorder=2)
            for prefix, marker, face in (("baseline", "o", "white"),
                                          ("ssr", "D", f5_ssr)):
                mean = summary[f"{prefix}_mean"]
                low, high = summary[f"{prefix}_ci95"]
                ax.errorbar(mean, ypos, xerr=[[mean - low], [high - mean]],
                            fmt=marker, ms=4.5, mfc=face, mec=INK, mew=0.42,
                            ecolor=INK, capsize=1.7, lw=0.72, zorder=4)
            ax.text(0.98, ypos + 0.23, f"$\\Delta$ {summary['delta_mean']:+.3f}",
                    transform=ax.get_yaxis_transform(), ha="right", va="center",
                    fontsize=4.9, color=f5_ssr, fontweight="bold")
        ax.axvline(0, color=LIGHT, lw=0.58, zorder=0)
        ax.set_xlim(-0.055, 0.80); ax.set_ylim(-0.38, 1.42)
        row_labels = {item[2]: item[1] for item in specs}
        ax.set_yticks([0, 1], [row_labels[0.0], row_labels[1.0]]
                      if show_labels else [])
        ax.set_xlabel("geometry score")
        ax.set_title(f"{edits} edits", fontsize=5.7, pad=2.2)
        clean(ax)

    geom100_ax = axe.inset_axes([0.08, 0.20, 0.40, 0.62])
    geometry_endpoint(geom100_ax, 100, True)
    geom250_ax = axe.inset_axes([0.57, 0.20, 0.40, 0.62])
    geometry_endpoint(geom250_ax, 250, False)
    axe.scatter([0.61], [0.94], s=12, marker="o", fc="white", ec=INK,
                lw=0.42, transform=axe.transAxes, clip_on=False)
    axe.scatter([0.78], [0.94], s=14, marker="D", fc=f5_ssr, ec=INK,
                lw=0.42, transform=axe.transAxes, clip_on=False)
    axe.text(0.63, 0.94, "LoRA", transform=axe.transAxes, va="center",
             fontsize=4.8, color=GRAY)
    axe.text(0.80, 0.94, "LoRA+SSR", transform=axe.transAxes, va="center",
             fontsize=4.8, color=f5_ssr)
    source_line(axe, "Independent mechanism cohort | Qwen2.5-7B | ZsRE | rank 8 | n=10",
                y=-0.01)

    axf = fig.add_subplot(gs[2, 7:])
    panel_label(axf, "f", "Geometry develops across editing", x=-0.13)
    axf.axis("off")
    f_cs = axf.inset_axes([0.17, 0.54, 0.80, 0.36])
    trajectory(f_cs, curves37, "geometry", "center_surround_contrast",
               "contrast", (-0.045, 0.42), False)
    f_cs.legend(frameon=False, loc="upper left", ncol=2, handletextpad=0.30,
                columnspacing=0.75, borderaxespad=0.0)
    f_ka = axf.inset_axes([0.17, 0.09, 0.80, 0.32])
    trajectory(f_ka, curves37, "geometry", "kernel_alignment",
               "alignment", (-0.07, 0.82), True)
    source_line(axf, "Independent 250-edit mechanism cohort | Qwen2.5-7B | ZsRE | rank 8 | n=10", y=-0.11)

    fig.subplots_adjust(left=0.075, right=0.990, top=0.975, bottom=0.060)
    save(fig, "Fig5_lowrank_ke_locked_20260904_v3")


def build_figure5_lowrank_ke_20260905() -> None:
    """Audited low-rank KE figure with separated acquisition and preservation."""
    source = ROOT / "figure_source_data" / "figure5_20260905"
    control = "#7C858A"
    ssr = "#D27645"
    acquisition = "#C7643C"
    preservation = "#4E8C75"
    retention = "#4E719E"
    gold = "#D1A23F"
    pale = "#D9DDDF"

    def csv_rows(name: str) -> list[dict[str, str]]:
        with (source / name).open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    primary_bundle = read_json(ROOT / "source_data/ke_lora_cosmos/summary.json")
    primary = primary_bundle["cohorts"]["independent_100"]["rank_summaries"]["8"]
    rank250_bundle = read_json(ROOT / "source_data/ke_lora_v36/confirmation_summary.json")
    rank250 = rank250_bundle["rank_summaries"]["8"]
    final_primary = primary["curves"][-1]["metrics"]
    rows_a = csv_rows("Fig5a_primary_seed_level.csv")
    rows_b = csv_rows("Fig5b_primary_trajectories.csv")
    rows_c = csv_rows("Fig5c_rank_stress_endpoints.csv")
    rows_e = csv_rows("Fig5e_mechanism_seed_level.csv")
    rows_f = csv_rows("Fig5f_mechanism_trajectories.csv")

    def paired_panel(ax, endpoint: str, summary: dict, ylabel: str,
                     ylim: tuple[float, float], color: str) -> None:
        selected = [row for row in rows_a if row["endpoint"] == endpoint]
        selected.sort(key=lambda row: int(row["seed"]))
        offsets = np.linspace(-0.038, 0.038, len(selected))
        for offset, row in zip(offsets, selected):
            left = float(row["LoRA_percent"])
            right = float(row["LoRA_plus_SSR_percent"])
            ax.plot([offset, 1 + offset], [left, right], color=pale, lw=0.62, zorder=1)
            ax.scatter(offset, left, s=10, fc="white", ec=control, lw=0.48, zorder=2)
            ax.scatter(1 + offset, right, s=10, fc="white", ec=color, lw=0.48, zorder=2)
        for xpos, prefix, face in [(0, "baseline", control), (1, "ssr", color)]:
            mean = summary[f"{prefix}_mean"]
            low, high = summary[f"{prefix}_ci95"]
            ax.errorbar(xpos, mean, yerr=[[mean - low], [high - mean]], fmt="o",
                        ms=5.0, mfc=face, mec=INK, mew=0.42, color=face,
                        capsize=2.0, lw=0.82, zorder=4)
        delta = summary["delta_mean"]
        low, high = summary["delta_ci95"]
        delta_label = (rf"mean $\Delta$ {delta:+.2f} pp; CI includes 0"
                       if low < 0 < high else
                       rf"$\Delta$ {delta:+.2f} pp [{low:.2f}, {high:.2f}]")
        ax.text(0.50, 0.98, delta_label,
                transform=ax.transAxes, ha="center", va="top", fontsize=4.6,
                color=color, fontweight="bold")
        ax.set_xticks([0, 1], ["LoRA", "+SSR"])
        ax.set_xlim(-0.17, 1.17); ax.set_ylim(*ylim)
        ax.set_ylabel(ylabel, labelpad=1.0)
        clean(ax)

    fig = plt.figure(figsize=(WIDTH, HEIGHT5), facecolor="white")
    gs = fig.add_gridspec(3, 12, height_ratios=[1.00, 1.08, 1.00],
                          hspace=0.47, wspace=0.82)

    # a: absolute paired values, not a gain-only summary.
    axa = fig.add_subplot(gs[0, :7])
    panel_label(axa, "a", "Efficacy and locality", x=-0.075)
    axa.axis("off")
    eff = axa.inset_axes([0.03, 0.15, 0.43, 0.74])
    paired_panel(eff, "immediate efficacy", final_primary["immediate_efficacy"],
                 "efficacy (%)", (50, 94), acquisition)
    pre = axa.inset_axes([0.57, 0.15, 0.40, 0.74])
    paired_panel(pre, "pre-edit output consistency",
                 final_primary["pre_edit_output_consistency"],
                 "locality: pre-edit consistency (%)", (12.5, 27.2), preservation)
    source_line(axa, "Qwen2.5-7B-Instruct | ZsRE | rank 8 | 100 edits | n=10 paired", y=-0.01)

    # b: observed editing checkpoints only, with aligned efficacy and locality axes.
    axb = fig.add_subplot(gs[0, 7:])
    panel_label(axb, "b", "Matched editing trajectory", x=-0.12)
    axb.axis("off")
    for idx, (endpoint, ylabel, color, ylim) in enumerate([
        ("immediate efficacy", "efficacy (%)", acquisition, (48, 104)),
        ("pre-edit output consistency", "locality (%)", preservation, (14, 25.8)),
    ]):
        sub = axb.inset_axes([0.14, 0.57 - 0.42 * idx, 0.83, 0.32])
        selected = sorted((row for row in rows_b if row["endpoint"] == endpoint),
                          key=lambda row: int(row["checkpoint_edits"]))
        edits = np.array([int(row["checkpoint_edits"]) for row in selected])
        for key, label, line_color, marker in [
            ("LoRA_percent", "LoRA", control, "o"),
            ("LoRA_plus_SSR_percent", "LoRA+SSR", color, "D"),
        ]:
            values = np.array([float(row[key]) for row in selected])
            sub.plot(edits, values, color=line_color, marker=marker, ms=2.6,
                     lw=0.90, label=label, zorder=3)
        sub.set_xscale("log")
        sub.set_xlim(0.8, 125); sub.set_ylim(*ylim)
        sub.set_xticks(edits, [str(value) for value in edits] if idx else [])
        if idx:
            sub.set_xlabel("edits applied")
        sub.set_ylabel(ylabel, labelpad=1.0)
        clean(sub)
        if idx == 0:
            sub.legend(frameon=False, loc="lower left", ncol=2, handletextpad=0.25,
                       columnspacing=0.65, borderaxespad=0.15)
        final = selected[-1]
        sub.text(0.98, 0.92, rf"final $\Delta$ {float(final['paired_delta_pp']):+.2f} pp",
                 transform=sub.transAxes, ha="right", va="top", fontsize=4.5,
                 color=color, fontweight="bold")
    source_line(axb, "same paired ZsRE queue as a | observed checkpoints", y=-0.14)

    # c: matched 250-edit acquisition stress test. Complete endpoint vectors stay in ED.
    axc = fig.add_subplot(gs[1, :7])
    panel_label(axc, "c", "Acquisition gain across low-rank budgets", x=-0.075)
    ranks = [8, 16, 32, 64]
    selected = {int(row["rank"]): row for row in rows_c
                if row["endpoint"] == "immediate efficacy"}
    means = np.array([float(selected[rank]["delta_pp"]) for rank in ranks])
    lows = np.array([float(selected[rank]["ci_low_pp"]) for rank in ranks])
    highs = np.array([float(selected[rank]["ci_high_pp"]) for rank in ranks])
    xpos = np.arange(len(ranks))
    axc.fill_between(xpos, lows, highs, color=acquisition, alpha=0.10, zorder=1)
    axc.plot(xpos, means, color=acquisition, lw=0.95, zorder=2)
    for xvalue, rank, mean, low, high in zip(xpos, ranks, means, lows, highs):
        face = "white" if rank == 32 else acquisition
        axc.errorbar(xvalue, mean, yerr=[[mean - low], [high - mean]], fmt="o",
                     ms=5.0, mfc=face, mec=acquisition, mew=0.80,
                     ecolor=acquisition, capsize=2.0, lw=0.85, zorder=3)
        axc.text(xvalue, high + 1.15, f"{mean:+.2f}", ha="center", va="bottom",
                 fontsize=4.8, color=acquisition, fontweight="bold")
    axc.axhline(0, color=INK, lw=0.55)
    axc.set_xticks(xpos, ["8", "16", "32*", "64"])
    axc.set_xlim(-0.35, 3.35); axc.set_ylim(0, 23.8)
    axc.set_xlabel("LoRA rank   |   *shared frozen coefficient")
    axc.set_ylabel("efficacy gain (pp)", labelpad=1.0)
    axc.text(0.99, 0.97, "all paired 95% CIs > 0", transform=axc.transAxes,
             ha="right", va="top", fontsize=4.6, color=acquisition,
             fontweight="bold")
    clean(axc)
    source_line(axc, "Qwen2.5-7B-Instruct | ZsRE | 250 edits | n=10/rank", y=-0.17)

    # d: paraphrase generalization at two fixed rank-8 edit horizons.  The
    # primary metric is kept consistent with the acquisition panels, while
    # the secondary prompt form tests transfer beyond the exact edit query.
    axd = fig.add_subplot(gs[1, 7:])
    panel_label(axd, "d", "Paraphrase generalization", x=-0.12)
    horizons = [
        ("100 edits", primary["curves"][-1]["metrics"]["immediate_rephrase"]),
        ("250 edits", rank250["curves"][-1]["metrics"]["immediate_rephrase"]),
    ]
    for xpos, (label, summary) in enumerate(horizons):
        deltas = np.asarray(summary["paired_deltas"], dtype=float)
        jitter = np.linspace(-0.075, 0.075, len(deltas))
        axd.scatter(xpos + jitter, deltas, s=10.0, facecolors="white",
                    edgecolors=acquisition, linewidths=0.48, alpha=0.80, zorder=2)
        mean = float(summary["delta_mean"])
        low, high = (float(value) for value in summary["delta_ci95"])
        axd.errorbar(xpos, mean, yerr=[[mean - low], [high - mean]], fmt="o",
                     ms=5.2, mfc=acquisition, mec=INK, mew=0.42,
                     ecolor=acquisition, capsize=2.0, lw=0.85, zorder=4)
        axd.text(xpos, high + 1.0, rf"$\Delta$ {mean:+.2f} pp",
                 ha="center", va="bottom", fontsize=4.7,
                 color=acquisition, fontweight="bold")
    axd.axhline(0, color=INK, lw=0.55, zorder=1)
    axd.set_xticks([0, 1], ["100 edits", "250 edits"])
    axd.set_xlim(-0.35, 1.35); axd.set_ylim(-14.5, 28.5)
    axd.set_xlabel("independent rank-8 confirmation")
    axd.set_ylabel("paraphrase efficacy gain (pp)", labelpad=1.0)
    axd.text(0.98, 0.98, "both paired 95% CIs > 0", transform=axd.transAxes,
             ha="right", va="top", fontsize=4.5, color=acquisition,
             fontweight="bold")
    clean(axd)
    source_line(axd, "Qwen2.5-7B-Instruct | ZsRE | rank 8 | n=10 paired/cohort", y=-0.17)

    # e: absolute geometry at 100 and 250 edits in the same mechanism cohort.
    axe = fig.add_subplot(gs[2, :7])
    panel_label(axe, "e", "Restricted-update geometry", x=-0.075)
    axe.axis("off")
    geometry_specs = [
        ("center_surround_contrast", "Row contrast", ssr, (-0.05, 0.39)),
        ("kernel_alignment", "Kernel alignment", preservation, (-0.05, 0.79)),
    ]
    for idx, (metric, title, color, ylim) in enumerate(geometry_specs):
        sub = axe.inset_axes([0.04 + idx * 0.49, 0.20, 0.43, 0.66])
        for xpos, edits in enumerate([100, 250]):
            for offset, arm, face, edge in [(-0.13, "LoRA", "white", control),
                                             (0.13, "LoRA+SSR", color, color)]:
                values = np.array([float(row[metric]) for row in rows_e
                                   if row["arm"] == arm and int(row["edits"]) == edits])
                jitter = np.linspace(-0.055, 0.055, len(values))
                sub.scatter(xpos + offset + jitter, values, s=8.5, fc=face, ec=edge,
                            lw=0.38, alpha=0.70, zorder=2)
                mean = float(values.mean())
                half = float(t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
                sub.errorbar(xpos + offset, mean, yerr=half, fmt="o", ms=4.3,
                             mfc=edge, mec=INK, mew=0.35, ecolor=INK,
                             capsize=1.7, lw=0.72, zorder=4)
        sub.axhline(0, color=pale, lw=0.55)
        sub.set_xticks([0, 1], ["100", "250"])
        sub.set_xlabel("edits")
        sub.set_ylabel("dimensionless score", labelpad=1.0)
        sub.set_title(title, fontsize=5.5, color=color, pad=2.2)
        sub.set_ylim(*ylim)
        clean(sub)
    axe.scatter([0.56], [0.96], s=11, fc="white", ec=control, lw=0.45,
                transform=axe.transAxes, clip_on=False)
    axe.text(0.58, 0.96, "LoRA", transform=axe.transAxes, va="center", fontsize=4.7)
    axe.scatter([0.72], [0.96], s=12, fc=ssr, ec=INK, lw=0.35,
                transform=axe.transAxes, clip_on=False)
    axe.text(0.74, 0.96, "LoRA+SSR", transform=axe.transAxes, va="center", fontsize=4.7)
    source_line(axe, "Qwen2.5-7B-Instruct | ZsRE | rank 8 mechanism queue | n=10 paired", y=-0.065)

    # f: actual checkpoints and endpoint decline are retained; no smoothing.
    axf = fig.add_subplot(gs[2, 7:])
    panel_label(axf, "f", "Geometry forms early and is maintained", x=-0.12)
    for endpoint, label, color, marker in [
        ("row contrast", "row contrast", ssr, "o"),
        ("kernel alignment", "kernel alignment", preservation, "D"),
    ]:
        selected = sorted((row for row in rows_f if row["endpoint"] == endpoint),
                          key=lambda row: int(row["checkpoint_edits"]))
        edits = np.array([int(row["checkpoint_edits"]) for row in selected])
        means = np.array([float(row["paired_delta"]) for row in selected])
        lows = np.array([float(row["ci_low"]) for row in selected])
        highs = np.array([float(row["ci_high"]) for row in selected])
        axf.fill_between(edits, lows, highs, color=color, alpha=0.11, linewidth=0)
        axf.plot(edits, means, color=color, marker=marker, ms=3.0, lw=1.0, label=label)
    axf.axhline(0, color=INK, lw=0.55)
    axf.set_xscale("log")
    axf.set_xlim(0.8, 310); axf.set_ylim(-0.03, 0.79)
    axf.set_xticks([1, 2, 5, 10, 25, 50, 100, 250],
                   ["1", "2", "5", "10", "25", "50", "100", "250"])
    axf.set_xlabel("edits applied")
    axf.set_ylabel("LoRA+SSR - LoRA (dimensionless)", labelpad=1.0)
    axf.legend(frameon=False, loc="upper left", handletextpad=0.35, borderaxespad=0.2)
    axf.annotate("0.369", (100, 0.369), xytext=(-12, 8), textcoords="offset points",
                 fontsize=4.5, color=ssr)
    axf.annotate("0.338", (250, 0.338), xytext=(-2, -10), textcoords="offset points",
                 fontsize=4.5, color=ssr, ha="right")
    clean(axf)
    source_line(axf, "same mechanism queue as e | observed checkpoints; no smoothing", y=-0.21)

    fig.subplots_adjust(left=0.072, right=0.992, top=0.976, bottom=0.060)
    save(fig, "Fig5_lowrank_ke_audited_20260905")


def build_supp_lowrank_ke_boundaries_20260905() -> None:
    """Complete endpoint vectors for the cohorts used in Figure 5."""
    source = ROOT / "figure_source_data" / "figure5_20260905"
    with (source / "FigS_lowrank_ke_boundaries.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    acquisition = "#C7643C"
    preservation = "#4E8C75"
    retention = "#4E719E"

    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 76 / 25.4), facecolor="white",
                             gridspec_kw={"width_ratios": [1.28, 1.0]})

    cohorts = [
        "rank 8 | 100 edits | primary",
        "rank 8 | 250 edits | rank cohort",
        "rank 8 | 250 edits | mechanism cohort",
        "rank 40 | 100 edits | selected confirmation",
    ]
    short = ["r8 / 100 / primary", "r8 / 250 / rank", "r8 / 250 / mechanism",
             "r40 / 100 / selected"]
    ax = axes[0]
    panel_label(ax, "a", "ZsRE endpoint validation", x=-0.18)
    offsets = {"immediate efficacy": 0.18, "pre-edit output consistency": 0.0,
               "normalized history AUC": -0.18}
    specs = [("immediate efficacy", acquisition, "o"),
             ("pre-edit output consistency", preservation, "s"),
             ("normalized history AUC", retention, "D")]
    lookup = {(row["cohort"], row["endpoint"]): row for row in rows}
    for ypos, cohort in enumerate(cohorts[::-1]):
        for endpoint, color, marker in specs:
            row = lookup[(cohort, endpoint)]
            mean = float(row["delta_pp"]); low = float(row["ci_low_pp"]); high = float(row["ci_high_pp"])
            ax.errorbar(mean, ypos + offsets[endpoint], xerr=[[mean-low], [high-mean]],
                        fmt=marker, ms=3.4, mfc=color, mec=INK, mew=0.3,
                        ecolor=color, capsize=1.2, lw=0.65)
    ax.axvline(0, color=INK, lw=0.55)
    ax.set_yticks(range(4), short[::-1]); ax.set_ylim(-0.48, 3.48); ax.set_xlim(-7, 25)
    ax.set_xlabel("LoRA+SSR - LoRA (pp)")
    clean(ax, "x")

    ax = axes[1]
    panel_label(ax, "b", "250-edit rank endpoints", x=-0.18)
    for endpoint, color, marker, offset in [
        ("pre-edit output consistency", preservation, "s", -0.07),
        ("normalized history AUC", retention, "D", 0.07),
    ]:
        for xpos, rank in enumerate([8, 16, 32, 64]):
            cohort = f"ZsRE rank {rank}{'*' if rank == 32 else ''} | 250 edits | rank stress"
            row = lookup[(cohort, endpoint)]
            mean = float(row["delta_pp"]); low = float(row["ci_low_pp"]); high = float(row["ci_high_pp"])
            ax.errorbar(xpos + offset, mean, yerr=[[mean-low], [high-mean]], fmt=marker,
                        ms=3.7, mfc=color, mec=INK, mew=0.32, ecolor=color,
                        capsize=1.3, lw=0.68, label=endpoint if xpos == 0 else None)
    ax.axhline(0, color=INK, lw=0.55)
    ax.set_xticks(range(4), ["8", "16", "32*", "64"])
    ax.set_xlabel("LoRA rank")
    ax.set_ylabel("change (pp)")
    ax.set_ylim(-3.3, 3.2)
    ax.legend(frameon=False, loc="upper left", fontsize=4.4, handletextpad=0.25)
    clean(ax)

    fig.subplots_adjust(left=0.105, right=0.987, top=0.84, bottom=0.22, wspace=0.42)
    save(fig, "fig_supp_lowrank_ke_boundaries_20260905")


def build_figure6() -> None:
    """Transfer figure ordered by task and by strength of causal attribution."""
    voc_rows = read_rows("figure5_source_data.csv")
    stage_rows = read_rows("figure5_stage_data.csv")
    rows = read_rows("figure6_source_data.csv")
    voc = read_json(ROOT / "data" / "lowrank_confirmations" / "v24_summary.json")
    fig = plt.figure(figsize=(WIDTH, HEIGHT6), facecolor="white")
    gs = fig.add_gridspec(4, 12, height_ratios=[1.20, 0.83, 1.08, 0.83],
                          hspace=0.64, wspace=1.18)

    # a, Direct SSR attribution: absolute paired endpoints plus the prespecified
    # paired-difference interval. AUC is the arithmetic mean over stages 1--10,
    # matching the archived confirmation summary exactly.
    axa = fig.add_subplot(gs[0, :7])
    panel_label(axa, "a", "Direct low-rank VOC retention", x=-0.11)
    endpoint_specs = [
        ("auc", "all_miou", "all-mIoU AUC", "all-mIoU trajectory AUC gain (pp)"),
        ("auc", "old_miou", "old-mIoU AUC", "old-mIoU trajectory AUC gain (pp)"),
        ("final", "all_miou", "final all mIoU", "final all-class mIoU gain (pp)"),
        ("final", "old_miou", "final old mIoU", "final old-class mIoU gain (pp)"),
        ("final", "old_boundary_iou", "final old BIoU", "final old-class Boundary IoU gain (pp)"),
    ]
    y_positions = np.arange(len(endpoint_specs))[::-1]
    for ypos, (kind, metric, label, delta_label) in zip(y_positions, endpoint_specs):
        if kind == "auc":
            control = np.array([
                100 * np.mean([record["lowrank"]["trajectory"][str(stage)][metric]
                               for stage in range(1, 11)])
                for record in voc["per_seed"]
            ])
            treatment = np.array([
                100 * np.mean([record["lowrank_ssr"]["trajectory"][str(stage)][metric]
                               for stage in range(1, 11)])
                for record in voc["per_seed"]
            ])
            comparison = voc["trajectory_auc_comparisons"]["lowrank_ssr_vs_lowrank"][metric]
        else:
            control = np.array([100 * record["lowrank"]["final"][metric]
                                for record in voc["per_seed"]])
            treatment = np.array([100 * record["lowrank_ssr"]["final"][metric]
                                  for record in voc["per_seed"]])
            comparison = voc["final_comparisons"]["lowrank_ssr_vs_lowrank"][metric]
        jitter = np.linspace(-0.15, 0.15, len(control))
        for base, ssr, offset in zip(control, treatment, jitter):
            axa.plot([base, ssr], [ypos + offset, ypos + offset], color=LIGHT,
                     lw=0.55, zorder=1)
            axa.scatter(base, ypos + offset, s=7.0, facecolor="white",
                        edgecolor=CONTROL, linewidth=0.45, zorder=2)
            axa.scatter(ssr, ypos + offset, s=7.0, facecolor=SSR,
                        edgecolor="white", linewidth=0.25, alpha=0.72, zorder=2)
        base_mean, ssr_mean = float(control.mean()), float(treatment.mean())
        axa.annotate("", xy=(ssr_mean, ypos), xytext=(base_mean, ypos),
                     arrowprops=dict(arrowstyle="-|>", color=SSR, lw=1.35,
                                     mutation_scale=6.0), zorder=4)
        axa.scatter(base_mean, ypos, s=26, facecolor="white", edgecolor=INK,
                    linewidth=0.75, zorder=5)
        axa.scatter(ssr_mean, ypos, s=28, facecolor=SSR, edgecolor=INK,
                    linewidth=0.45, zorder=5)
        delta = 100 * float(comparison["paired_delta_mean"])
        low, high = [100 * float(value) for value in comparison["bootstrap_95_ci"]]
        annotation_x = (30.5 if kind == "auc"
                        else max(float(control.max()), float(treatment.max())) + 0.55)
        axa.text(annotation_x, ypos,
                 rf"$\Delta$ {delta:+.2f} [{low:.2f}, {high:.2f}]",
                 va="center", fontsize=4.55, color=SSR, fontweight="bold")
    axa.set_yticks(y_positions, [spec[2] for spec in endpoint_specs])
    axa.set_xlim(3.0, 49.0); axa.set_ylim(-0.50, 4.52)
    axa.set_xlabel("absolute endpoint (%)")
    axa.scatter([], [], s=18, facecolor="white", edgecolor=CONTROL,
                linewidth=0.6, label="LowRank")
    axa.scatter([], [], s=18, facecolor=SSR, edgecolor=INK,
                linewidth=0.35, label="LowRank+SSR")
    axa.legend(frameon=False, loc="lower right", ncol=2, handletextpad=0.25,
               columnspacing=0.75, borderaxespad=0.15)
    clean(axa, "x")
    source_line(axa, "Direct SSR attribution | PASCAL VOC 10-1 | rank-8 history decoder | n=10",
                y=-0.28)

    axb = fig.add_subplot(gs[0, 7:])
    panel_label(axb, "b", "Stage-wise direct effect", x=-0.13)
    axb.axis("off")
    for idx, (metric, title, auc_text) in enumerate([
        ("all_miou", "All classes", "AUC +1.07 pp"),
        ("old_miou", "Old classes", "AUC +0.94 pp"),
    ]):
        sub = axb.inset_axes([0.02 + idx * 0.51, 0.15, 0.45, 0.70])
        selected = sorted([row for row in stage_rows if row["metric"] == metric],
                          key=lambda row: int(row["stage"]))
        stages = np.array([int(row["stage"]) for row in selected])
        means = np.array([float(row["mean"]) for row in selected])
        lows = np.array([float(row["ci95_low"]) for row in selected])
        highs = np.array([float(row["ci95_high"]) for row in selected])
        sub.fill_between(stages, lows, highs, color=SSR, alpha=0.12, linewidth=0)
        sub.plot(stages, means, color=SSR, marker="o", ms=2.8, lw=0.90)
        sub.axhline(0, color=INK, lw=0.55)
        sub.set_xlim(0.8, 10.2); sub.set_ylim(-1.8, 3.25)
        sub.set_xticks([1, 3, 5, 7, 10])
        sub.set_xlabel("incremental stage")
        sub.set_title(title, fontsize=5.2, pad=2.0)
        sub.text(0.98, 0.96, auc_text, transform=sub.transAxes, ha="right", va="top",
                 fontsize=4.8, color=SSR, fontweight="bold")
        if idx == 0:
            sub.set_ylabel("SSR effect (pp)")
        else:
            sub.set_yticklabels([])
        clean(sub)
    source_line(axb, "LowRank+SSR - LowRank | PASCAL VOC 10-1 | rank 8 | n=10",
                y=-0.06)

    # c, Use the same overlay grammar as the Supplementary segmentation gallery.
    # The case and predictions are unchanged; only the rendering is different.
    axc = fig.add_subplot(gs[1, :])
    panel_label(axc, "c", "PASCAL VOC segmentation example", x=-0.035)
    axc.axis("off")
    qrow = next(row for row in voc_rows
                if row["panel"] == "h" and row["primary_metric"] == "pixel-level outcome")
    sample = ROOT / qrow["source_file"]
    input_image = np.asarray(Image.open(sample / "input.png").convert("RGB"))
    gt_color = np.asarray(Image.open(sample / "gt_color.png").convert("RGB"))
    lowrank_color = np.asarray(Image.open(sample / "lowrank_color.png").convert("RGB"))
    lowrank_ssr_color = np.asarray(Image.open(sample / "lowrank_ssr_color.png").convert("RGB"))
    target_rgb = (92, 134, 99)  # VOC class 3 (bird) in the archived palette.
    panels = [
        input_image,
        target_class_overlay(input_image, gt_color, target_rgb, (22, 138, 153)),
        target_class_overlay(input_image, lowrank_color, target_rgb, (207, 86, 93)),
        target_class_overlay(input_image, lowrank_ssr_color, target_rgb, (22, 138, 153)),
        np.asarray(Image.open(sample / "change_map.png").convert("RGB")),
    ]
    voc_titles = ["Input", "Ground truth", "LowRank", "LowRank + SSR",
                  r"Change ($\Delta$mIoU +13.93 pp)"]
    for idx, (array, title) in enumerate(zip(panels, voc_titles)):
        inset = axc.inset_axes([0.004 + idx * 0.199, 0.00, 0.190, 0.88])
        inset.imshow(array)
        inset.set_title(title, fontsize=5.2, pad=1.3, fontweight="bold")
        inset.axis("off")
        inset.add_patch(Rectangle((0, 0), 1, 1, transform=inset.transAxes,
                                  fill=False, ec=SSR if idx == 4 else "#C8CCCF",
                                  lw=0.85 if idx == 4 else 0.55))
    axc.text(0.004, -0.02,
             "PASCAL VOC 10-1 | bird-class overlays | rank-8 history decoder | one display case",
             transform=axc.transAxes, ha="left", va="top", fontsize=4.45,
             color=GRAY)
    axc.text(0.992, -0.02, "blue: corrected  |  warm: harmed",
             transform=axc.transAxes, ha="right", va="top", fontsize=4.6,
             color=GRAY)

    # d, Nested SSR attribution on a fixed KD+LowRank reference.
    axd = fig.add_subplot(gs[2, :6])
    panel_label(axd, "d", "Low-rank CUB classification", x=-0.13)
    rank_style = {8: (GREEN, "o"), 16: (ORANGE, "s"), 32: (SSR, "D")}
    rank_path = []
    for rank in (8, 16, 32):
        aa = metric_row(voc_rows, "c", "shared cosine", "aa_gain_pp", rank)
        af = metric_row(voc_rows, "c", "shared cosine", "af_reduction_pp", rank)
        x, y = number(aa, "paired_change"), number(af, "paired_change")
        rank_path.append((x, y))
        color, marker = rank_style[rank]
        axd.errorbar(x, y,
                     xerr=[[x-number(aa, "ci_low")], [number(aa, "ci_high")-x]],
                     yerr=[[y-number(af, "ci_low")], [number(af, "ci_high")-y]],
                     fmt=marker, ms=5.0, mfc=color, mec=INK, mew=0.4,
                     ecolor=color, capsize=1.7, lw=0.75)
        axd.annotate(f"rank {rank}", (x, y),
                     xytext=(5, 6 if rank != 16 else -11), textcoords="offset points",
                     fontsize=5.0, color=color, fontweight="bold")
    for start, end in zip(rank_path[:-1], rank_path[1:]):
        axd.annotate("", xy=end, xytext=start,
                     arrowprops=dict(arrowstyle="->", lw=0.75, color=GRAY, alpha=0.70))
    axd.axvline(0, color=LIGHT, lw=0.60); axd.axhline(0, color=LIGHT, lw=0.60)
    axd.set_xlim(-0.05, 1.75); axd.set_ylim(-0.08, 1.38)
    axd.set_xlabel(r"$\Delta$AA (pp)")
    axd.set_ylabel("AF reduction (pp)")
    axd.text(0.98, 0.04, "reference: KD+LowRank", transform=axd.transAxes,
             ha="right", va="bottom", fontsize=4.65, color=GRAY)
    clean(axd)
    source_line(axd, "Nested SSR attribution | CUB-200-2011 | ResNet-18 + GELU adapter | n=10/rank",
                y=-0.31)

    # e, Combined-objective transfer. This panel deliberately shows absolute
    # paired values because the control does not share the KD term.
    axe = fig.add_subplot(gs[2, 6:])
    panel_label(axe, "e", "Combined-objective mask transfer", x=-0.13)
    axe.axis("off")
    structured_records = [json.loads(line) for line in
                          (ROOT / "source_data" / "raw" / "structured_output" /
                           "runs.jsonl").read_text(encoding="utf-8").splitlines()
                          if line.strip()]
    by_method = {method: sorted([record for record in structured_records
                                 if record["method"] == method], key=lambda record: record["seed"])
                 for method in ("baseline", "biocs_kd")}
    structured_specs = [
        ("mean_iou", "mIoU", "higher", SSR),
        ("mean_dice", "Dice", "higher", PURPLE),
        ("avg_forgetting_iou", "IoU forgetting", "lower", ORANGE),
    ]
    for idx, (key, label, direction, color) in enumerate(structured_specs):
        sub = axe.inset_axes([0.015 + idx * 0.327, 0.17, 0.285, 0.66])
        control = np.array([record[key] for record in by_method["baseline"]], dtype=float)
        treatment = np.array([record[key] for record in by_method["biocs_kd"]], dtype=float)
        for seed_index, (base, target) in enumerate(zip(control, treatment)):
            xj = np.array([0.0, 1.0]) + (seed_index - 1) * 0.018
            sub.plot(xj, [base, target], color=LIGHT, lw=0.80, zorder=1)
            sub.scatter(xj[0], base, s=16, facecolor="white", edgecolor=CONTROL,
                        linewidth=0.55, zorder=2)
            sub.scatter(xj[1], target, s=16, facecolor=color, edgecolor="white",
                        linewidth=0.25, alpha=0.78, zorder=2)
        sub.plot([0, 1], [control.mean(), treatment.mean()], color=color, lw=1.35,
                 zorder=3)
        sub.scatter(0, control.mean(), s=28, facecolor="white", edgecolor=INK,
                    linewidth=0.75, zorder=4)
        sub.scatter(1, treatment.mean(), s=30, facecolor=color, edgecolor=INK,
                    linewidth=0.45, zorder=4)
        favorable = (treatment - control) if direction == "higher" else (control - treatment)
        sub.text(0.50, 0.98, f"favorable {favorable.mean():+.2f} pp",
                 transform=sub.transAxes, ha="center", va="top", fontsize=4.5,
                 color=color, fontweight="bold")
        sub.set_xlim(-0.25, 1.25)
        padding = max(0.35, 0.18 * (max(control.max(), treatment.max()) -
                                   min(control.min(), treatment.min())))
        sub.set_ylim(min(control.min(), treatment.min()) - padding,
                     max(control.max(), treatment.max()) + padding)
        sub.set_xticks([0, 1], ["Unreg.", "KD+SSR"])
        sub.set_ylabel(f"{label} (%)")
        clean(sub)
    axe.text(0.50, 0.02, "CUB-200-2011 masks | frozen ResNet-18 | n=3 paired seeds",
             transform=axe.transAxes, ha="center", va="bottom", fontsize=4.55,
             color="#596166")
    axe.text(0.50, -0.10, "combined objective; not an isolated SSR attribution",
             transform=axe.transAxes, ha="center", va="top", fontsize=4.55,
             color=ORANGE, fontweight="bold", clip_on=False)

    # f, Outcome-aware CUB visualization is placed last and is not used for
    # population inference. Row 1 is the archived main/supplement-linked case.
    bird_row = next(row for row in rows if row["point_id"] == "cub_response_gallery_qualitative")
    gallery_path = ROOT / bird_row["source_file"]
    axf = fig.add_subplot(gs[3, :])
    panel_label(axf, "f", "CUB response-map example", x=-0.035)
    axf.set_xlim(0, 1); axf.set_ylim(0, 1); axf.axis("off")
    tiles = render_pdf_image_row(gallery_path, row_index=1)
    titles = ["Input", "GT mask", "KD response", "KD+SSR response", "KD+SSR - KD"]
    for idx, (tile, title) in enumerate(zip(tiles, titles)):
        inset = axf.inset_axes([0.006 + idx * 0.199, 0.00, 0.190, 0.88])
        inset.imshow(tile)
        inset.set_title(title, fontsize=5.2, pad=1.4, fontweight="bold")
        inset.axis("off")
        inset.add_patch(Rectangle((0, 0), 1, 1, transform=inset.transAxes,
                                  fill=False, ec=SSR if idx == 4 else "#C8CCCF",
                                  lw=0.85 if idx == 4 else 0.55))
    axf.text(0.50, -0.02,
             "CUB-200-2011 | response map | frozen ResNet-18 | one illustrative case",
             transform=axf.transAxes, ha="center", va="top", fontsize=4.45,
             color=GRAY)

    fig.subplots_adjust(left=0.102, right=0.988, top=0.973, bottom=0.035)
    save(fig, "Fig6_transfer_ordered_20260903_v2")


def build_supp_voc_efficiency() -> None:
    rows = read_rows("figure6_efficiency.csv")
    fig, ax = plt.subplots(figsize=(112 / 25.4, 54 / 25.4), facecolor="white")
    colors = {"cohort 1": GREEN, "replication": ORANGE}
    markers = {"cohort 1": "o", "replication": "s"}
    metrics = [("wall time", "wall time"), ("peak GPU memory", "peak GPU memory")]
    ybase = np.array([1, 0], dtype=float)
    for cohort, offset in [("cohort 1", 0.10), ("replication", -0.10)]:
        for y, (metric, _) in zip(ybase + offset, metrics):
            row = next(record for record in rows
                       if record["cohort"] == cohort and record["metric"] == metric)
            value = float(row["percent_overhead"])
            ax.plot([0, value], [y, y], color=LIGHT, lw=1.15, zorder=1)
            ax.scatter(value, y, s=30, marker=markers[cohort], fc=colors[cohort],
                       ec=INK, lw=0.35, zorder=3)
            ax.text(value + 0.08, y, f"{value:.2f}", va="center", fontsize=5.2,
                    color=colors[cohort], fontweight="bold")
    ax.axvline(0, color=INK, lw=0.55)
    ax.set_yticks(ybase, [label for _, label in metrics])
    ax.set_xlim(-0.15, 3.65); ax.set_ylim(-0.42, 1.42)
    ax.set_xlabel("SSR overhead relative to LowRank (%)")
    ax.scatter([], [], marker="o", s=24, fc=GREEN, ec=INK, lw=0.35,
               label="cohort 1")
    ax.scatter([], [], marker="s", s=24, fc=ORANGE, ec=INK, lw=0.35,
               label="replication")
    ax.legend(frameon=False, loc="lower right", ncol=2, handletextpad=0.35,
              columnspacing=0.8)
    clean(ax, "x")
    fig.subplots_adjust(left=0.22, right=0.97, top=0.94, bottom=0.22)
    fig.savefig(FIGURES / "fig_supp_voc_efficiency.pdf", facecolor="white")
    fig.savefig(FIGURES / "fig_supp_voc_efficiency.svg", facecolor="white")
    plt.close(fig)


def build_supp_plop_tradeoff() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(150 / 25.4, 72 / 25.4), facecolor="white")
    method_colors = {"PLOP": CONTROL, "LowRank": ORANGE, "LowRank+SSR": SSR}
    method_keys = [("plop", "PLOP"), ("lowrank", "LowRank"),
                   ("lowrank_ssr", "LowRank+SSR")]
    specs = [("cohort 1", "data/lowrank_confirmations/v24_summary.json"),
             ("independent replication", "data/lowrank_confirmations/v31_summary.json")]
    for ax, (title, relative) in zip(axes, specs):
        import json
        data = json.loads((ROOT / relative).read_text(encoding="utf-8"))
        coordinates = {}
        for method_key, method in method_keys:
            all_values = np.array([100.0 * float(record[method_key]["final"]["all_miou"])
                                   for record in data["per_seed"]])
            margin_values = np.array([float(record[method_key]["final"]["old_boundary_margin"])
                                      for record in data["per_seed"]])
            x, y = float(all_values.mean()), float(margin_values.mean())
            xhalf = float(t.ppf(0.975, len(all_values) - 1)
                          * all_values.std(ddof=1) / np.sqrt(len(all_values)))
            yhalf = float(t.ppf(0.975, len(margin_values) - 1)
                          * margin_values.std(ddof=1) / np.sqrt(len(margin_values)))
            coordinates[method] = (x, y)
            ax.errorbar(x, y, xerr=xhalf, yerr=yhalf, fmt="o",
                        ms=4.7 if method == "LowRank+SSR" else 4.1,
                        mfc=method_colors[method], mec=INK, mew=0.4,
                        ecolor=method_colors[method], capsize=1.5, lw=0.65)
            offset = {"PLOP": (-3, 7), "LowRank": (-4, 7),
                      "LowRank+SSR": (5, -10)}[method]
            ax.annotate(method, (x, y), xytext=offset, textcoords="offset points",
                        ha="right" if method != "LowRank+SSR" else "left",
                        fontsize=5.0, color=method_colors[method],
                        fontweight="bold" if method == "LowRank+SSR" else "normal")
        ax.annotate("", xy=coordinates["LowRank+SSR"], xytext=coordinates["LowRank"],
                    arrowprops=dict(arrowstyle="->", lw=0.85, color=SSR))
        ax.set_xlim(11.5, 25.5); ax.set_ylim(1.8, 6.15)
        ax.set_xlabel("final all-class mIoU (%)")
        ax.set_title(title, fontsize=6.2, pad=3)
        clean(ax)
    axes[0].set_ylabel("old-class boundary-logit margin")
    fig.suptitle("External accuracy-retention trade-off", fontsize=7.0,
                 fontweight="bold", x=0.07, ha="left", y=0.99)
    fig.subplots_adjust(left=0.105, right=0.985, top=0.83, bottom=0.20, wspace=0.30)
    fig.savefig(FIGURES / "fig_supp_plop_tradeoff.pdf", facecolor="white")
    fig.savefig(FIGURES / "fig_supp_plop_tradeoff.svg", facecolor="white")
    plt.close(fig)


def build_contact_sheet() -> None:
    stems = [
        "Fig4_submission_20260905",
        "Fig5_lowrank_ke_audited_20260905",
        "Fig6_transfer_ordered_20260903_v2",
    ]
    fig, axes = plt.subplots(3, 1, figsize=(183 / 25.4, 270 / 25.4), facecolor="white")
    for ax, stem in zip(axes, stems):
        ax.imshow(Image.open(FIGURES / f"{stem}.png"))
        ax.set_title(stem.split("_")[0], loc="left", fontsize=8.5,
                     fontweight="bold", pad=2)
        ax.axis("off")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.985, bottom=0.015, hspace=0.09)
    fig.savefig(DATA / "Fig4_6_final_strict_contact_sheet.pdf", facecolor="white")
    plt.close(fig)


def main() -> None:
    build_figure4()
    build_figure5_lowrank_ke_20260905()
    build_figure6()
    build_supp_lowrank_ke_boundaries_20260905()
    build_supp_plop_tradeoff()
    build_supp_voc_efficiency()
    build_contact_sheet()
    print("Built strict Figures 4--6 and 183-mm contact sheet")


if __name__ == "__main__":
    main()
