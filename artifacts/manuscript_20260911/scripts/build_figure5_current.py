#!/usr/bin/env python3
"""Build the revised Figure 5 and its endpoint-only Extended Data audit."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, NullLocator
import numpy as np
from scipy.stats import t, ttest_1samp


ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "figure5_current"
FIGURES = ROOT / "figures"
SUMMARY_CSV = HERE / "efficacy_locality_only_source_data.csv"
SEED_CSV = HERE / "efficacy_locality_seed_source_data.csv"
WIKIRECENT_CURVES = HERE / "wikirecent_rank8_100.json"

WIDTH = 183 / 25.4
HEIGHT = 170 / 25.4
INK = "#23282C"
GRAY = "#788187"
PALE = "#D9DDDF"
EFFICACY = "#C86643"
LOCALITY = "#348299"
SSR = "#C86643"
GEOMETRY = "#4D8A74"


def configure_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 5.5,
        "axes.labelsize": 5.8,
        "axes.titlesize": 6.8,
        "xtick.labelsize": 5.1,
        "ytick.labelsize": 5.1,
        "legend.fontsize": 5.0,
        "axes.linewidth": 0.55,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def clean(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(width=0.5, length=2.0, pad=1.5)
    ax.grid(axis="y", color="#E1E4E6", linewidth=0.42, zorder=0)


def panel_label(ax: plt.Axes, label: str, title: str, x: float = -0.10) -> None:
    ax.text(x, 1.07, label, transform=ax.transAxes, fontsize=8.5,
            fontweight="bold", ha="left", va="bottom")
    ax.text(0.0, 1.07, title, transform=ax.transAxes, fontsize=6.8,
            fontweight="bold", ha="left", va="bottom")


def source_line(ax: plt.Axes, text: str, y: float = -0.20) -> None:
    ax.text(0.50, y, text, transform=ax.transAxes, ha="center", va="top",
            fontsize=4.6, color="#596166", clip_on=False)


def dataset_label(path: str) -> str:
    if "wiki_recent" in path.lower():
        return "WikiRecent"
    if "wiki_counterfact" in path.lower():
        return "WikiCounterFact"
    return path


def p_text(values: np.ndarray) -> str:
    p_value = float(ttest_1samp(np.asarray(values, dtype=float), 0.0).pvalue)
    return r"$P<0.001$" if p_value < 0.001 else rf"$P={p_value:.3f}$"


def load_seed_pairs() -> dict[tuple[str, int], list[dict[str, float]]]:
    arms: dict[tuple[str, int], dict[float, dict[int, dict[str, str]]]] = defaultdict(lambda: defaultdict(dict))
    for row in read_csv(SEED_CSV):
        key = (dataset_label(row["dataset"]), int(row["rank"]))
        arms[key][float(row["lambda"])][int(row["seed"])] = row
    output: dict[tuple[str, int], list[dict[str, float]]] = {}
    for key, by_coefficient in arms.items():
        positives = sorted(value for value in by_coefficient if value > 0)
        if 0.0 not in by_coefficient or not positives:
            continue
        coefficient = positives[0]
        pairs = []
        for seed in sorted(set(by_coefficient[0.0]) & set(by_coefficient[coefficient])):
            baseline = by_coefficient[0.0][seed]
            ssr = by_coefficient[coefficient][seed]
            pairs.append({
                "seed": seed,
                "coefficient": coefficient,
                "efficacy_baseline": float(baseline["immediate_efficacy"]),
                "efficacy_ssr": float(ssr["immediate_efficacy"]),
                "locality_baseline": float(baseline["immediate_locality"]),
                "locality_ssr": float(ssr["immediate_locality"]),
            })
        output[key] = pairs
    return output


def load_summary() -> dict[tuple[str, str, int], dict[str, str]]:
    return {
        (row["dataset"], row["cohort"], int(row["rank"])): row
        for row in read_csv(SUMMARY_CSV)
    }


def paired_absolute(ax: plt.Axes, pairs: list[dict[str, float]], endpoint: str,
                    color: str, ylabel: str, summary: dict[str, str]) -> None:
    baseline = np.asarray([row[f"{endpoint}_baseline"] for row in pairs])
    ssr = np.asarray([row[f"{endpoint}_ssr"] for row in pairs])
    offsets = np.linspace(-0.045, 0.045, len(pairs))
    for offset, left, right in zip(offsets, baseline, ssr):
        ax.plot([offset, 1 + offset], [left, right], color=PALE, lw=0.60, zorder=1)
        ax.scatter(offset, left, s=11, fc="white", ec=GRAY, lw=0.45, zorder=2)
        ax.scatter(1 + offset, right, s=11, fc="white", ec=color, lw=0.50, zorder=2)
    ax.scatter([0, 1], [baseline.mean(), ssr.mean()], s=32,
               fc=[GRAY, color], ec=INK, lw=0.45, zorder=4)
    delta = float(summary[f"{endpoint}_delta_mean"])
    low = float(summary[f"{endpoint}_delta_ci_low"])
    high = float(summary[f"{endpoint}_delta_ci_high"])
    ax.text(0.5, 0.99, rf"$\Delta$ {delta:+.2f} pp [{low:+.2f}, {high:+.2f}]",
            transform=ax.transAxes, ha="center", va="top", color=color,
            fontsize=5.0, fontweight="bold")
    ax.text(0.5, 0.86, p_text(ssr - baseline), transform=ax.transAxes,
            ha="center", va="top", color=color, fontsize=4.6)
    margin = max((baseline.max() - baseline.min()), (ssr.max() - ssr.min()), 1.0) * 0.16
    ax.set_ylim(min(baseline.min(), ssr.min()) - margin,
                max(baseline.max(), ssr.max()) + margin)
    ax.set_xticks([0, 1], ["LoRA", "+SSR"])
    ax.set_xlim(-0.18, 1.18)
    ax.set_ylabel(ylabel)
    clean(ax)


def rank_panel(ax: plt.Axes, pairs_by_rank: dict[tuple[str, int], list[dict[str, float]]],
               summaries: dict[tuple[str, str, int], dict[str, str]], endpoint: str,
               color: str, label: str, title: str) -> None:
    ranks = [8, 16, 32, 64]
    means, lows, highs = [], [], []
    for rank in ranks:
        row = summaries[("WikiRecent", "new 100-edit rank sweep", rank)]
        means.append(float(row[f"{endpoint}_delta_mean"]))
        lows.append(float(row[f"{endpoint}_delta_ci_low"]))
        highs.append(float(row[f"{endpoint}_delta_ci_high"]))
    x = np.arange(len(ranks))
    ax.plot(x, means, color=color, lw=1.0, zorder=2)
    jitter = np.linspace(-0.10, 0.10, 10)
    for xpos, rank, mean, low, high in zip(x, ranks, means, lows, highs):
        values = np.asarray([
            row[f"{endpoint}_ssr"] - row[f"{endpoint}_baseline"]
            for row in pairs_by_rank[("WikiRecent", rank)]
        ])
        ax.scatter(xpos + jitter, values, s=8.5, fc="white", ec=color,
                   lw=0.38, alpha=0.50, zorder=1)
        ax.errorbar(xpos, mean, yerr=[[mean - low], [high - mean]], fmt="o",
                    ms=4.7, mfc=color, mec=INK, mew=0.38, ecolor=color,
                    capsize=1.8, lw=0.78, zorder=4)
        ax.annotate(f"{mean:+.2f}", (xpos, high), xytext=(0, 5),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=4.8, color=color, fontweight="bold")
        ax.annotate(p_text(values), (xpos, low), xytext=(0, -6),
                    textcoords="offset points", ha="center", va="top",
                    fontsize=4.0, color=GRAY)
    ax.axhline(0, color=INK, lw=0.55, zorder=0)
    ax.set_xticks(x, [str(rank) for rank in ranks])
    ax.set_xlabel("LoRA rank")
    ax.set_ylabel(label)
    clean(ax)
    panel_label(ax, title[0], title[1:], x=-0.09)
    source_line(ax, "WikiRecent | 100 edits | n=10 paired/rank | frozen rank-specific coefficients", y=-0.20)


def build_main_figure() -> None:
    summaries = load_summary()
    pairs_by_rank = load_seed_pairs()
    rank32 = pairs_by_rank[("WikiRecent", 32)]
    rank32_summary = summaries[("WikiRecent", "new 100-edit rank sweep", 32)]
    curve_summary = json.loads(WIKIRECENT_CURVES.read_text(encoding="utf-8"))
    curve_by_edit = {int(row["after_edits"]): row for row in curve_summary["curves"]}

    fig = plt.figure(figsize=(WIDTH, HEIGHT), facecolor="white")
    gs = fig.add_gridspec(3, 12, height_ratios=[1.0, 1.0, 1.0],
                          hspace=0.70, wspace=1.15)

    axa = fig.add_subplot(gs[0, :7])
    panel_label(axa, "a", "Matched efficacy and locality")
    axa.axis("off")
    eff = axa.inset_axes([0.03, 0.12, 0.43, 0.76])
    loc = axa.inset_axes([0.57, 0.12, 0.40, 0.76])
    paired_absolute(eff, rank32, "efficacy", EFFICACY, "efficacy (%)", rank32_summary)
    paired_absolute(loc, rank32, "locality", LOCALITY, "locality (%)", rank32_summary)
    source_line(axa, "Qwen2.5-7B-Instruct | WikiRecent | rank 32 | 100 edits | n=10 paired", y=-0.02)

    axb = fig.add_subplot(gs[0, 7:])
    panel_label(axb, "b", "SSR gain over edits", x=-0.13)
    axb.axis("off")
    displayed_edits = np.asarray([1, 25, 50, 100])
    trajectory_specs = [
        ("immediate_efficacy", "efficacy gain (pp)", EFFICACY, (-0.8, 13.0)),
        ("immediate_locality_target_consistency", "locality gain (pp)", LOCALITY, (-2.4, 4.9)),
    ]
    for index, (metric, ylabel, color, ylim) in enumerate(trajectory_specs):
        sub = axb.inset_axes([0.12, 0.56 - index * 0.46, 0.84, 0.34])
        means = np.asarray([curve_by_edit[int(ed)]["performance"][metric]["delta_mean"]
                            for ed in displayed_edits])
        lows = np.asarray([curve_by_edit[int(ed)]["performance"][metric]["delta_ci95"][0]
                           for ed in displayed_edits])
        highs = np.asarray([curve_by_edit[int(ed)]["performance"][metric]["delta_ci95"][1]
                            for ed in displayed_edits])
        sub.fill_between(displayed_edits, lows, highs, color=color, alpha=0.12, linewidth=0)
        sub.errorbar(displayed_edits, means, yerr=[means - lows, highs - means],
                     color=color, marker="D", ms=3.0, lw=1.0, capsize=1.6,
                     elinewidth=0.7, label="LoRA+SSR - LoRA")
        sub.axhline(0, color=INK, lw=0.52, zorder=0)
        final_deltas = np.asarray(curve_by_edit[100]["performance"][metric]["paired_deltas"])
        sub.text(0.98, 0.94, "100 edits: " + p_text(final_deltas), transform=sub.transAxes,
                 ha="right", va="top", color=color, fontsize=4.2)
        sub.set_xscale("log")
        sub.set_xticks(displayed_edits, [str(value) for value in displayed_edits])
        sub.xaxis.set_minor_locator(NullLocator())
        sub.xaxis.set_minor_formatter(NullFormatter())
        sub.set_ylabel(ylabel)
        sub.set_ylim(*ylim)
        sub.set_yticks([0, 5, 10] if index == 0 else [-2, 0, 2, 4])
        if index == 0:
            sub.tick_params(labelbottom=False)
        else:
            sub.set_xlabel("edits applied")
        clean(sub)
    source_line(axb, "WikiRecent | rank 8 | observed accumulation checkpoints | n=10 paired", y=-0.18)

    axc = fig.add_subplot(gs[1, :6])
    rank_panel(axc, pairs_by_rank, summaries, "efficacy", EFFICACY,
               "efficacy gain (pp)", "cEfficacy across rank budgets")
    axd = fig.add_subplot(gs[1, 6:])
    rank_panel(axd, pairs_by_rank, summaries, "locality", LOCALITY,
               "locality gain (pp)", "dLocality across rank budgets")

    axe = fig.add_subplot(gs[2, :7])
    panel_label(axe, "e", "Geometric endpoints after 100 edits", x=-0.09)
    axe.axis("off")
    geometry_specs = [
        ("center_surround_contrast", "row contrast", SSR),
        ("kernel_alignment", "kernel alignment", GEOMETRY),
    ]
    for index, (metric, label, color) in enumerate(geometry_specs):
        summary = curve_by_edit[100]["geometry"][metric]
        values = np.asarray(summary["paired_deltas"], dtype=float)
        sub = axe.inset_axes([0.08 + index * 0.48, 0.12, 0.38, 0.75])
        boxplot = sub.boxplot(
            [values], positions=[0], widths=0.48, patch_artist=True,
            showfliers=False, whis=(0, 100),
            medianprops={"color": INK, "linewidth": 0.95},
            whiskerprops={"color": GRAY, "linewidth": 0.70},
            capprops={"color": GRAY, "linewidth": 0.70},
        )
        boxplot["boxes"][0].set(
            facecolor=mpl.colors.to_rgba(color, 0.18), edgecolor=color,
            linewidth=0.85,
        )
        jitter = np.linspace(-0.10, 0.10, len(values))
        sub.scatter(jitter, values, s=12, fc="white", ec=color,
                    lw=0.45, alpha=0.72, zorder=2)
        mean = float(values.mean())
        half = float(t.ppf(0.975, len(values)-1) * values.std(ddof=1) / np.sqrt(len(values)))
        sub.errorbar(0, mean, yerr=half, fmt="D", ms=4.7, mfc=color,
                     mec=INK, mew=0.4, ecolor=color, capsize=1.8, lw=0.8, zorder=4)
        span = max(float(values.max() - values.min()), half * 2, 0.005)
        sub.set_ylim(float(values.min()) - 0.38 * span,
                     float(values.max()) + 0.72 * span)
        sub.set_xlim(-0.34, 0.34)
        sub.set_xticks([0], [label])
        sub.set_ylabel("SSR - LoRA" if index == 0 else "")
        sub.text(0.50, 0.98, f"mean {mean:+.3f}\n{p_text(values)}",
                 transform=sub.transAxes, ha="center", va="top",
                 fontsize=4.5, color=color, fontweight="bold")
        clean(sub)
    source_line(axe, "WikiRecent | rank 8 | 100 edits | n=10 paired", y=-0.22)

    axf = fig.add_subplot(gs[2, 7:])
    panel_label(axf, "f", "Geometry forms during editing", x=-0.13)
    for endpoint, label, color, marker in [
        ("center_surround_contrast", "row contrast", SSR, "o"),
        ("kernel_alignment", "kernel alignment", GEOMETRY, "D"),
    ]:
        edits = np.asarray(sorted(curve_by_edit))
        means = np.asarray([curve_by_edit[int(ed)]["geometry"][endpoint]["delta_mean"]
                            for ed in edits])
        lows = np.asarray([curve_by_edit[int(ed)]["geometry"][endpoint]["delta_ci95"][0]
                           for ed in edits])
        highs = np.asarray([curve_by_edit[int(ed)]["geometry"][endpoint]["delta_ci95"][1]
                            for ed in edits])
        axf.fill_between(edits, lows, highs, color=color, alpha=0.10, linewidth=0)
        axf.plot(edits, means, color=color, marker=marker, ms=3.0,
                 lw=1.0, label=label)
    axf.axhline(0, color=INK, lw=0.55)
    axf.set_xscale("log")
    axf.set_xlim(0.8, 125); axf.set_ylim(-0.03, 0.79)
    checkpoints = [1, 2, 5, 10, 25, 50, 100]
    axf.set_xticks(checkpoints, [str(value) for value in checkpoints])
    axf.set_xlabel("edits applied")
    axf.set_ylabel("LoRA+SSR - LoRA (dimensionless)")
    axf.legend(frameon=False, loc="upper left", handletextpad=0.35)
    clean(axf)
    axf.text(0.98, 0.08, r"both endpoints: $P<0.001$ at 100 edits",
             transform=axf.transAxes, ha="right", va="bottom", fontsize=4.4, color=GRAY)
    source_line(axf, "same WikiRecent rank-8 cohort as b,e | observed checkpoints", y=-0.20)

    fig.subplots_adjust(left=0.075, right=0.992, top=0.965, bottom=0.065)
    FIGURES.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "svg", "png"):
        fig.savefig(FIGURES / f"Figure5.{extension}", dpi=400, facecolor="white")
    plt.close(fig)


def build_extended_data() -> None:
    summaries = load_summary()
    cohorts = [
        ("WikiRecent", "new 100-edit rank sweep", "WikiRecent | 100 edits"),
        ("WikiCounterFact", "new 100-edit rank sweep", "WikiCounterFact | 100 edits"),
        ("ZsRE", "ZsRE main 100-edit", "ZsRE primary | 100 edits"),
        ("ZsRE", "ZsRE breadth 100-edit", "ZsRE breadth | 100 edits"),
        ("ZsRE", "ZsRE rank stress 250-edit", "ZsRE rank stress | 250 edits"),
    ]
    fig, axes = plt.subplots(len(cohorts), 2, figsize=(WIDTH, 205/25.4),
                             facecolor="white")
    for row_index, (dataset, cohort, title) in enumerate(cohorts):
        selected = sorted(
            (value for (ds, ch, _), value in summaries.items()
             if ds == dataset and ch == cohort and value["status"] != "partial"),
            key=lambda value: int(value["rank"]),
        )
        ranks = [int(row["rank"]) for row in selected]
        for col, (endpoint, color, ylabel) in enumerate([
            ("efficacy", EFFICACY, "efficacy gain (pp)"),
            ("locality", LOCALITY, "locality gain (pp)"),
        ]):
            ax = axes[row_index, col]
            means = np.asarray([float(row[f"{endpoint}_delta_mean"]) for row in selected])
            lows = np.asarray([float(row[f"{endpoint}_delta_ci_low"]) for row in selected])
            highs = np.asarray([float(row[f"{endpoint}_delta_ci_high"]) for row in selected])
            x = np.arange(len(ranks))
            ax.plot(x, means, color=color, lw=0.9)
            ax.errorbar(x, means, yerr=[means-lows, highs-means], fmt="o", ms=4.0,
                        mfc=color, mec=INK, mew=0.35, ecolor=color,
                        capsize=1.6, lw=0.7)
            ax.axhline(0, color=INK, lw=0.5)
            ax.set_xticks(x, [str(rank) for rank in ranks])
            ax.set_xlabel("LoRA rank")
            ax.set_ylabel(ylabel)
            clean(ax)
            if col == 0:
                ax.text(-0.18, 1.08, chr(ord("a") + row_index), transform=ax.transAxes,
                        fontsize=8, fontweight="bold", ha="left", va="bottom")
            ax.set_title((title + " | " + ("efficacy" if col == 0 else "locality")),
                         loc="left", fontweight="bold", pad=4)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.965, bottom=0.06,
                        hspace=0.80, wspace=0.33)
    for extension in ("pdf", "svg", "png"):
        fig.savefig(FIGURES / f"ExtendedData_lowrank_knowledge_editing.{extension}",
                    dpi=400, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    configure_style()
    build_main_figure()
    build_extended_data()
    print(f"Wrote revised Figure 5 and endpoint audit to {FIGURES}")
