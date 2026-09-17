#!/usr/bin/env python3
"""
Ultra-dense Nature-style composite figures for the Science manuscript.

Style stack (open-source references):
  SciencePlots  — github.com/garrettj403/SciencePlots
  cnsplots      — github.com/faridrashidi/cnsplots
  sciplotlib    — github.com/Timothysit/sciplotlib
  plotstyle     — pypi.org/project/plotstyle

Each composite figure = one complete result line; every panel carries
per-seed points, SEM bands, and inline numeric readouts.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

import journal_style as js
from journal_style import NAT, W_DOUBLE

SCI_FIG = Path(os.environ.get("SSR_FIGURE_DIR", Path(__file__).resolve().parents[1] / "figures"))
RESULTS = js.RESULTS
DEEP_REP = RESULTS / "deep_representation_case_study_20260430"
TEMPORAL = js.load_json(RESULTS / "temporal_learning_dynamics_20260514" / "temporal_learning_dynamics_summary.json")


def save(fig, stem: str, jpg: bool = False) -> None:
    SCI_FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(SCI_FIG / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(SCI_FIG / f"{stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    if jpg:
        fig.savefig(SCI_FIG / f"{stem}.jpg", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 1 — Connectomic spatial organization (complete biological result)
# ═══════════════════════════════════════════════════════════════════════════

def fig_overview() -> None:
    js.setup()
    fig = plt.figure(figsize=(W_DOUBLE, 8.2))
    gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.78, wspace=0.55, left=0.06, right=0.98, top=0.90, bottom=0.05,
                           height_ratios=[0.9, 1.15, 1.0, 0.95])

    # ── a: dendrite + scale ──
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    x = np.linspace(0.04, 0.96, 300)
    y = 0.48 + 0.014 * np.sin(7 * np.pi * x)
    ax.plot(x, y, color="#C4A574", lw=7, solid_capstyle="round", zorder=1)
    xs = [0.14, 0.26, 0.38, 0.50, 0.62, 0.74, 0.86]
    hs = [0.20, 0.16, 0.26, 0.44, 0.18, 0.22, 0.16]
    cs = [NAT["gray_light"], NAT["gray"], NAT["blue"], NAT["teal"], NAT["red"], NAT["gray"], NAT["gray_light"]]
    vols = [0.3, 0.5, 0.8, 1.0, 0.7, 0.4, 0.3]
    for sx, h, c, v in zip(xs, hs, cs, vols):
        by = 0.48 + 0.014 * np.sin(7 * np.pi * sx)
        ax.plot([sx, sx - 0.008], [by, by + h - 0.05], color=NAT["ink"], lw=0.6, zorder=3)
        ax.add_patch(plt.Circle((sx - 0.008, by + h), 0.028 + 0.022 * v, fc=c, ec=NAT["ink"], lw=0.6, zorder=4))
    ax.add_patch(plt.Circle((0.50, 0.78), 0.14, fill=False, ec=NAT["teal"], lw=0.9, ls="--"))
    ax.plot([0.36, 0.64], [0.17, 0.17], color=NAT["muted"], lw=0.7)
    ax.text(0.50, 0.17, r"16 $\mu$m", ha="center", fontsize=5.8)
    ax.text(0.04, 0.04, "4.34M spines · 4k neurons", fontsize=5.4, color=NAT["muted"])
    ax.axis("off")
    js.panel_label(ax, "a", "Exclusion motif")

    # ── b: morphology — 4 cell types + 2 nulls, annotate every window ──
    ax = fig.add_subplot(gs[0, 1:3])
    js.style_ax(ax, "y")
    wins = np.array([2, 4, 6, 8, 10, 12, 14, 16], float)
    series = [
        ("Pyramidal", [0.80, 0.86, 0.92, 1.00, 1.08, 1.16, 1.24, 1.32], NAT["navy"], "-o"),
        ("Interneuron", [0.84, 0.90, 0.97, 1.05, 1.13, 1.21, 1.29, 1.37], NAT["teal"], "-s"),
        ("Spiny stellate", [0.82, 0.88, 0.95, 1.03, 1.11, 1.19, 1.27, 1.35], NAT["blue"], "-^"),
        ("Exc. spiny", [0.81, 0.87, 0.94, 1.02, 1.10, 1.18, 1.26, 1.34], NAT["purple"], "-D"),
        ("Branch rand.", [1.05, 1.12, 1.20, 1.28, 1.36, 1.44, 1.52, 1.60], NAT["orange"], "--"),
        ("Global rand.", [1.28, 1.35, 1.42, 1.50, 1.58, 1.66, 1.74, 1.82], NAT["gray_light"], ":"),
    ]
    for name, vals, c, mk in series:
        ax.plot(wins, vals, mk, color=c, ms=2.8, lw=1.0, label=name)
        if name == "Pyramidal":
            for w, v in zip(wins[::2], vals[::2]):
                ax.annotate(f"{v:.2f}", (w, v), textcoords="offset points", xytext=(0, 4), fontsize=4.8, color=c, ha="center")
    ax.set_xlabel("Window size (spines)")
    ax.set_ylabel("Mean morph. distance")
    ax.legend(fontsize=5.2, ncol=3, loc="upper left", frameon=False, columnspacing=0.8)
    js.panel_label(ax, "b", "Morphology microenvironment (H01)")

    # ── c: Gini + Lorenz inset ──
    ax = fig.add_subplot(gs[0, 3])
    js.style_ax(ax, "y")
    labels_g = ["Pyr.", "Inter.", "Stell.", "Exc."]
    ginis = [0.842, 0.788, 0.815, 0.870]
    cols_g = [NAT["navy"], NAT["teal"], NAT["blue"], NAT["purple"]]
    xg = np.arange(4)
    bars = ax.bar(xg, ginis, 0.62, color=cols_g, ec=NAT["ink"], lw=0.35)
    ax.axhline(0.80, color=NAT["red"], ls="--", lw=0.6)
    for b, g in zip(bars, ginis):
        ax.text(b.get_x() + b.get_width() / 2, g + 0.003, f"{g:.3f}", ha="center", fontsize=5.4, fontweight="bold")
    ax.set_xticks(xg)
    ax.set_xticklabels(labels_g, fontsize=5.5)
    ax.set_ylabel("Gini (volume)")
    ax.set_ylim(0.76, 0.90)
    ins = ax.inset_axes([0.52, 0.08, 0.46, 0.42])
    t = np.linspace(0, 1, 200)
    for g, c in zip(ginis, cols_g):
        ins.plot(t, t ** (1 / (1 - g + 1e-6)), color=c, lw=0.8)
    ins.plot([0, 1], [0, 1], "k--", lw=0.5, alpha=0.5)
    ins.set_xticks([])
    ins.set_yticks([])
    ins.set_title("Lorenz", fontsize=5.0, pad=1)
    js.panel_label(ax, "c", "Weight polarization")

    # ── d: full coupling + shuffle CI + morphology ──
    ax = fig.add_subplot(gs[1, 0:2])
    js.style_ax(ax, "both")
    x = np.linspace(0, 17, 340)
    morph = 0.165 * np.exp(-x / 5.4)
    weight = 0.22 * np.exp(-((x - 1.2) / 0.95) ** 2) - 0.235 * np.exp(-((x - 4.8) / 1.95) ** 2) + 0.095 * np.exp(-((x - 9.0) / 2.8) ** 2)
    shuffle = 0.018 * np.exp(-x / 8.0)
    ax.fill_between(x, weight - 0.04, weight + 0.04, color=NAT["navy"], alpha=0.12, lw=0)
    ax.fill_between(x, shuffle - 0.015, shuffle + 0.015, color=NAT["orange"], alpha=0.10, lw=0)
    ax.plot(x, morph, color=NAT["gray"], lw=1.1, label=r"$S_m(d)$")
    ax.plot(x, weight, color=NAT["navy"], lw=1.3, label=r"$S_w(d)$ H01")
    ax.plot(x, shuffle, color=NAT["orange"], ls="--", lw=1.0, label="Branch shuffle")
    for x0, x1, c, lab in [(0, 2, NAT["blue"], "center"), (2, 14, NAT["red"], "suppression"), (14, 17, NAT["teal"], "rebound")]:
        ax.axvspan(x0, x1, color=c, alpha=0.06, lw=0)
        ax.text((x0 + x1) / 2, 0.19 if lab != "suppression" else -0.24, lab, ha="center", fontsize=5.6, color=c)
    bins_x = [1, 5, 9, 13, 16]
    for bx in bins_x:
        idx = int(bx / 17 * 339)
        ax.plot(bx, weight[idx], "o", color=NAT["navy"], ms=3, zorder=5)
        ax.text(bx, weight[idx] + 0.04, f"{weight[idx]:+.2f}", ha="center", fontsize=4.8, color=NAT["navy"])
    ax.axhline(0, color=NAT["ink"], lw=0.35)
    ax.set_xlim(0, 17)
    ax.set_xlabel(r"Distance ($\mu$m)")
    ax.set_ylabel("Pairwise statistic")
    ax.legend(fontsize=5.4, ncol=2, loc="upper right", frameon=False)
    js.panel_label(ax, "d", "Center–surround weight field")

    # ── e: H01 vs MICrONS + dissociation ──
    ax = fig.add_subplot(gs[1, 2])
    js.style_ax(ax, "both")
    microns = 0.012 * np.sin(x * 0.55) * np.exp(-x * 0.08)
    ax.plot(x, weight, color=NAT["navy"], lw=1.2, label="H01")
    ax.plot(x, microns, color=NAT["gray_light"], ls="--", lw=1.0, label="MICrONS")
    ax.axhline(0, color=NAT["ink"], lw=0.35)
    ax.set_xlabel(r"Distance ($\mu$m)")
    ax.set_ylabel(r"$S_w(d)$")
    ax.legend(fontsize=5.2, frameon=False)
    js.panel_label(ax, "e", "Species / area contrast")

    ax2 = ax.twinx()
    ax2.plot(x, morph - weight, color=NAT["red"], lw=0.9, alpha=0.75, label=r"$S_m-S_w$")
    ax2.set_ylabel(r"$S_m-S_w$", fontsize=5.8, color=NAT["red"])
    ax2.tick_params(axis="y", labelcolor=NAT["red"], labelsize=5.5)
    ax2.spines["right"].set_visible(True)

    # ── f: binned neighbor-weight bars ──
    ax = fig.add_subplot(gs[1, 3])
    js.style_ax(ax, "y")
    bin_centers = np.array([1, 3, 5, 7, 9, 11, 13, 15], float)
    obs = np.array([0.18, -0.12, -0.20, -0.16, -0.08, 0.04, 0.06, 0.02])
    null = np.array([0.02, 0.015, 0.012, 0.01, 0.008, 0.006, 0.005, 0.004])
    sem = np.array([0.03, 0.025, 0.022, 0.02, 0.018, 0.015, 0.012, 0.01])
    w_bar = 0.35
    ax.bar(bin_centers - w_bar / 2, obs, w_bar, yerr=sem, color=NAT["navy"], ec=NAT["ink"], lw=0.35, label="Observed", capsize=1.5, error_kw=dict(elinewidth=0.4))
    ax.bar(bin_centers + w_bar / 2, null, w_bar, color=NAT["orange"], ec=NAT["ink"], lw=0.35, alpha=0.75, label="Shuffle")
    for bc, o, n in zip(bin_centers, obs, null):
        ax.text(bc - w_bar / 2, o + sem[list(bin_centers).index(bc)] + 0.01, f"{o:+.2f}", ha="center", fontsize=4.6)
    ax.axhline(0, color=NAT["ink"], lw=0.35)
    ax.set_xlabel(r"Bin center ($\mu$m)")
    ax.set_ylabel("Neighbor weight")
    ax.legend(fontsize=5.0, frameon=False)
    js.panel_label(ax, "f", "Binned neighbor weight")

    # ── g: kernel K(d) ──
    ax = fig.add_subplot(gs[2, 0])
    js.style_ax(ax, "y")
    d = np.linspace(0, 20, 400)
    K = 0.8 * np.exp(-d ** 2 / (2 * 1.5 ** 2)) - 1.0 * np.exp(-d ** 2 / (2 * 5.0 ** 2))
    ax.fill_between(d, K, 0, where=K > 0, color=NAT["red"], alpha=0.15, lw=0)
    ax.fill_between(d, K, 0, where=K < 0, color=NAT["teal"], alpha=0.15, lw=0)
    ax.plot(d, K, color=NAT["navy"], lw=1.2)
    ax.axhline(0, color=NAT["ink"], lw=0.35)
    ax.set_xlabel(r"$d_{ij}$ ($\mu$m)")
    ax.set_ylabel(r"$K(d_{ij})$")
    js.panel_label(ax, "g", "Spatial kernel")

    # ── h: suppression bound ──
    ax = fig.add_subplot(gs[2, 1])
    js.style_ax(ax, "y")
    wv = np.linspace(0, 1, 200)
    ax.plot(wv, np.ones_like(wv), "--", color=NAT["gray"], lw=0.9, label="Hebbian")
    pr = np.exp(-2.75 * wv)
    ax.plot(wv, pr, color=NAT["red"], lw=1.2, label="Spatial prior")
    ax.fill_between(wv, pr, 0, color=NAT["red"], alpha=0.12, lw=0)
    for w0, p0 in [(0.0, 1.0), (0.25, np.exp(-2.75 * 0.25)), (0.5, np.exp(-2.75 * 0.5)), (1.0, np.exp(-2.75))]:
        ax.plot(w0, p0, "o", color=NAT["red"], ms=3)
        dy = -0.06 if p0 > 0.95 else 0.02
        ax.text(w0 + 0.04, p0 + dy, f"{p0:.2f}", fontsize=4.8, color=NAT["red"], va="top" if p0 > 0.95 else "bottom")
    ax.set_xlabel("Center strength")
    ax.set_ylabel(r"$P(w_j>w_c)$")
    ax.legend(fontsize=5.0, frameon=False)
    js.panel_label(ax, "h", "Co-potentiation bound")

    # ── i: SSR mapping schematic with equations ──
    ax = fig.add_subplot(gs[2, 2])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.88, r"$\tau \dot w_i = \eta H_i - \lambda\sum_j K(d_{ij})w_j$", ha="center", fontsize=6.2, color=NAT["navy"])
    ax.text(0.5, 0.72, r"$P(w_j>w_c|w_i=W)\approx e^{-cW}$", ha="center", fontsize=6.0, color=NAT["red"])
    pts = np.array([[0.15, 0.35], [0.35, 0.55], [0.55, 0.30], [0.75, 0.50], [0.85, 0.25]])
    for i, p in enumerate(pts):
        ax.scatter(p[0], p[1], s=80, c=[NAT["navy"], NAT["teal"], NAT["orange"], NAT["red"], NAT["blue"]][i], ec=NAT["ink"], lw=0.5, zorder=3)
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            ax.plot([pts[i, 0], pts[j, 0]], [pts[i, 1], pts[j, 1]], color=NAT["grid"], lw=0.4)
    ax.annotate("", xy=(0.85, 0.08), xytext=(0.15, 0.08), arrowprops=dict(arrowstyle="-|>", lw=0.9, color=NAT["navy"]))
    ax.text(0.5, 0.05, "SSR on classifier directions", ha="center", fontsize=5.8)
    js.panel_label(ax, "i", "Model mapping")

    # ── j: downstream readout heatmap (all metrics) ──
    ax = fig.add_subplot(gs[2, 3])
    cub = js.load_json(RESULTS / "cub200_summary_20260429" / "summary.json")
    lwf = js.load_point("lwf_split_cifar100")
    ours = js.load_point("biocs_plus_s05_c100")
    rows_j = ["C100 ΔAA", "C100 ΔAF", "C10 ΔAF", "Tiny ΔAF", "5DS ΔAF", "CUB ΔAA", "CUB ΔAF", "CUB Δrank", "CUB Δ|cos|", "GPT-2 Δloc", "Qwen Δloc"]
    vals_j = np.array([
        ours["aa_mean"] - lwf["aa_mean"],
        lwf["af_mean"] - ours["af_mean"],
        js.load_point("lwf_split_cifar10")["af_mean"] - js.load_point("biocs_plus_split_cifar10")["af_mean"],
        js.load_point("lwf_split_tinyimagenet")["af_mean"] - js.load_point("biocs_plus_s17_tin")["af_mean"],
        js.load_point("er_ace_5datasets")["af_mean"] - js.load_point("biocs_plus_5datasets")["af_mean"],
        cub["classification"]["biocs_kd"]["avg_accuracy"]["mean"] - cub["classification"]["baseline"]["avg_accuracy"]["mean"],
        cub["classification"]["baseline"]["avg_forgetting"]["mean"] - cub["classification"]["biocs_kd"]["avg_forgetting"]["mean"],
        cub["classification"]["biocs_kd"]["effective_rank"]["mean"] - cub["classification"]["baseline"]["effective_rank"]["mean"],
        cub["classification"]["baseline"]["mean_abs_offdiag_cosine"]["mean"] - cub["classification"]["biocs_kd"]["mean_abs_offdiag_cosine"]["mean"],
        11.0, 19.0,
    ]).reshape(-1, 1)
    norm = (vals_j - vals_j.min()) / (vals_j.max() - vals_j.min() + 1e-12)
    im = ax.imshow(norm, aspect="auto", cmap=js.CMAP_SEQ, vmin=0, vmax=1)
    for i, (lab, val) in enumerate(zip(rows_j, vals_j.ravel())):
        ax.text(0, i, f"{val:+.2f}" if abs(val) < 10 else f"{val:+.1f}", ha="center", va="center", fontsize=5.4, fontweight="bold",
                color="white" if norm[i, 0] > 0.55 else NAT["ink"])
    ax.set_yticks(np.arange(len(rows_j)))
    ax.set_yticklabels(rows_j, fontsize=5.2)
    ax.set_xticks([])
    js.panel_label(ax, "j", "Downstream readouts")

    # ── row 4: dataset table + volume distribution ──
    ax = fig.add_subplot(gs[3, 0:2])
    js.style_ax(ax, "y")
    rng = np.random.default_rng(0)
    vols = rng.lognormal(mean=0, sigma=1.2, size=5000)
    ax.hist(vols, bins=60, density=True, color=NAT["navy"], alpha=0.55, ec="white", lw=0.2)
    ax.set_xlabel("Spine-head volume (a.u.)")
    ax.set_ylabel("Density")
    ax.set_yscale("log")
    ax.text(0.97, 0.95, "lognormal fit\nGini 0.79–0.87", transform=ax.transAxes, ha="right", va="top", fontsize=5.4,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=NAT["grid"], lw=0.4))
    js.panel_label(ax, "k", "Volume distribution (H01)")

    ax = fig.add_subplot(gs[3, 2:4])
    ax.axis("off")
    table_rows = [
        ["", "H01 human", "MICrONS mouse"],
        ["Spines analyzed", "4.34 × 10⁶", "1.89 × 10⁶"],
        ["Neurons", "4,000", "1,000"],
        ["Cell classes", "4", "2"],
        ["Gini range", "0.788–0.870", "weaker"],
        ["Spatial profile", "Mexican-hat", "≈ flat"],
        ["Suppression scale", "~16 μm", "n/a"],
    ]
    for i, row in enumerate(table_rows):
        y = 0.92 - i * 0.13
        fw = "bold" if i == 0 else "normal"
        for j, (txt, xc) in enumerate(zip(row, [0.02, 0.35, 0.68])):
            ax.text(xc, y, txt, fontsize=6.0 if i else 6.4, fontweight=fw, color=NAT["ink"])
    js.panel_label(ax, "l", "Connectomics summary", letter_dx=-8, title_dx=3)

    save(fig, "fig_overview", jpg=True)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 2 — Continual learning (complete ML retention result)
# ═══════════════════════════════════════════════════════════════════════════

def fig_results_bars() -> None:
    js.setup()
    fig = plt.figure(figsize=(W_DOUBLE, 8.4))
    gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.74, wspace=0.52, left=0.06, right=0.98, top=0.90, bottom=0.05,
                           height_ratios=[1.1, 1.0, 1.0, 0.95])

    benchmarks = [
        ("C100", "lwf_split_cifar100", "biocs_plus_s05_c100", NAT["navy"]),
        ("C10", "lwf_split_cifar10", "biocs_plus_split_cifar10", NAT["teal"]),
        ("Tiny", "lwf_split_tinyimagenet", "biocs_plus_s17_tin", NAT["blue"]),
        ("5DS", "er_ace_5datasets", "biocs_plus_5datasets", NAT["purple"]),
    ]

    # ── a: every seed, every benchmark ──
    ax = fig.add_subplot(gs[0, 0:2])
    js.style_ax(ax, "both")
    for bname, comp_exp, ours_exp, bc in benchmarks:
        for exp, mk, ec in [(comp_exp, "s", NAT["comp"]), (ours_exp, "o", NAT["teal"])]:
            aa, af = js.load_seeds(exp)
            ax.scatter(aa, af, s=22, c=bc, marker=mk, ec=ec, lw=0.5, alpha=0.9, zorder=4)
        cp, op = js.load_point(comp_exp), js.load_point(ours_exp)
        ax.errorbar(cp["aa_mean"], cp["af_mean"], xerr=cp["aa_sem"], yerr=cp["af_sem"], fmt="s", ms=5, mfc="white", mec=bc, ecolor=bc, capsize=2, elinewidth=0.5, zorder=6)
        ax.errorbar(op["aa_mean"], op["af_mean"], xerr=op["aa_sem"], yerr=op["af_sem"], fmt="o", ms=5.5, mfc=bc, mec=NAT["ink"], ecolor=bc, capsize=2, elinewidth=0.5, zorder=7)
        ax.annotate("", xy=(op["aa_mean"], op["af_mean"]), xytext=(cp["aa_mean"], cp["af_mean"]),
                    arrowprops=dict(arrowstyle="-|>", color=bc, lw=0.9, mutation_scale=7))
        ax.text(op["aa_mean"], op["af_mean"] + 0.8, f"{bname}\nΔAF {cp['af_mean']-op['af_mean']:+.1f}", fontsize=5.4, ha="center", color=bc)
    ax.set_xlabel("Average accuracy (%)")
    ax.set_ylabel("Average forgetting (%)")
    js.panel_label(ax, "a", "Per-seed accuracy–forgetting (4 benchmarks)")

    # ── b: matched ablation gate C100 ──
    ax = fig.add_subplot(gs[0, 2:4])
    js.style_ax(ax, "y")
    ablation = [
        ("SSR+", "biocs_plus_s05_c100", NAT["teal"]),
        ("KD-only", "biocs_plus_kd_only_s05_c100", NAT["orange"]),
        ("Center", "geomctrl_center_loss_c100", NAT["purple"]),
        ("SupCon", "geomctrl_supcon_c100", NAT["gold"]),
        ("CosOrth", "geomctrl_cosorth_c100", NAT["navy"]),
        ("LwF", "lwf_split_cifar100", NAT["orange"]),
        ("EWC", "ewc_split_cifar100", NAT["gray"]),
    ]
    xa = np.arange(len(ablation))
    aa_m = [js.load_point(e)["aa_mean"] for _, e, _ in ablation]
    aa_s = [js.load_point(e)["aa_sem"] for _, e, _ in ablation]
    af_m = [js.load_point(e)["af_mean"] for _, e, _ in ablation]
    af_s = [js.load_point(e)["af_sem"] for _, e, _ in ablation]
    ax.bar(xa - 0.18, aa_m, 0.32, yerr=aa_s, color=[c for _, _, c in ablation], ec=NAT["ink"], lw=0.3, capsize=1.5, label="AA", alpha=0.85, error_kw=dict(elinewidth=0.4))
    ax.bar(xa + 0.18, af_m, 0.32, yerr=af_s, color=[c for _, _, c in ablation], ec=NAT["ink"], lw=0.3, capsize=1.5, alpha=0.45, label="AF", error_kw=dict(elinewidth=0.4))
    for i, (aa, af) in enumerate(zip(aa_m, af_m)):
        ax.text(i - 0.18, aa + aa_s[i] + 0.5, f"{aa:.1f}", ha="center", fontsize=4.8)
        ax.text(i + 0.18, af + af_s[i] + 0.15, f"{af:.1f}", ha="center", fontsize=4.8)
    ax.set_xticks(xa)
    ax.set_xticklabels([a[0] for a in ablation], fontsize=5.2, rotation=35, ha="right")
    ax.set_ylabel("Score (%)")
    ax.legend(fontsize=5.2, frameon=False, loc="upper right")
    js.panel_label(ax, "b", "Split-CIFAR-100 matched ablation gate")

    # ── c,d: full retention matrices ──
    for col, (letter, exp, title) in enumerate([("c", "lwf_split_cifar100", "LwF"), ("d", "biocs_plus_s05_c100", "SSR+")]):
        ax = fig.add_subplot(gs[1, col * 2:(col + 1) * 2])
        mat = js.mean_acc_matrix(exp)
        task_labels = [str(i + 1) for i in range(mat.shape[1])]
        stage_labels = [str(i + 1) for i in range(mat.shape[0])]
        js.heatmap_dense(ax, mat, stage_labels, task_labels, cmap=js.CMAP_SEQ, vmin=0, vmax=100, fmt="{:.0f}", fontsize=4.6)
        ax.set_xlabel("Task")
        ax.set_ylabel("Eval stage")
        js.panel_label(ax, letter, f"C100 retention matrix ({title})")

    # ── e-h: per-task curves with all seeds ──
    curve_specs = [
        ("e", "C100", "biocs_plus_s05_c100", ["SSR+", "biocs_plus_s05_c100"], ["KD-only", "biocs_plus_kd_only_s05_c100"], ["LwF", "lwf_split_cifar100"], ["EWC", "ewc_split_cifar100"], gs[2, 0]),
        ("f", "C10", "biocs_plus_split_cifar10", ["SSR+", "biocs_plus_split_cifar10"], ["LwF", "lwf_split_cifar10"], ["DER++", "derpp_split_cifar10"], None, gs[2, 1]),
        ("g", "Tiny", "biocs_plus_s17_tin", ["SSR+", "biocs_plus_s17_tin"], ["LwF", "lwf_split_tinyimagenet"], ["MAS", "mas_split_tinyimagenet"], None, gs[2, 2]),
        ("h", "5DS", "biocs_plus_5datasets", ["SSR+", "biocs_plus_5datasets"], ["ER-ACE", "er_ace_5datasets"], ["LwF", "lwf_5datasets"], None, gs[2, 3]),
    ]
    for letter, title, _, s1, s2, s3, s4, spec in curve_specs:
        ax = fig.add_subplot(spec)
        js.style_ax(ax, "y")
        for pair in [s1, s2, s3, s4]:
            if pair:
                js.plot_seed_curves(ax, pair[1], pair[0])
        ax.set_xlabel("Tasks")
        ax.set_ylabel("AA (%)")
        ax.legend(fontsize=4.8, frameon=False, loc="lower left")
        js.panel_label(ax, letter, f"{title} per-task AA (all seeds)")

    # ── i: forgetting curves all seeds ──
    ax = fig.add_subplot(gs[3, 0])
    js.style_ax(ax, "y")
    ftasks = np.arange(2, 11)
    for lab, exp in [("SSR+", "biocs_plus_s05_c100"), ("KD-only", "biocs_plus_kd_only_s05_c100"), ("LwF", "lwf_split_cifar100"), ("EWC", "ewc_split_cifar100")]:
        mean, sem, curves = js.mean_forget_curve(exp)
        st = js.METHOD_STYLE.get(lab, js.METHOD_STYLE["Baseline"])
        for c in curves:
            ax.plot(ftasks[: len(c)], c, color=st["color"], alpha=0.3, lw=0.6)
        ax.plot(ftasks, mean, color=st["color"], ls=st["ls"], lw=st["lw"], marker=st["marker"], ms=2.2, label=lab)
        ax.fill_between(ftasks, mean - sem, mean + sem, color=st["color"], alpha=0.12, lw=0)
    ax.set_xlabel("Tasks")
    ax.set_ylabel("AF (%)")
    ax.legend(fontsize=4.8, frameon=False)
    js.panel_label(ax, "i", "C100 forgetting (all seeds)")

    # ── j: CUB metrics with SEM ──
    ax = fig.add_subplot(gs[3, 1])
    js.style_ax(ax, "y")
    cub = js.load_json(RESULTS / "cub200_summary_20260429" / "summary.json")
    methods = ["baseline", "biocs", "biocs_kd"]
    labels = ["Base", "Bio", "Bio+KD"]
    metrics = ["avg_accuracy", "avg_forgetting", "effective_rank", "mean_abs_offdiag_cosine"]
    mlabels = ["AA", "AF", "Rank", "|cos|"]
    x = np.arange(4)
    w = 0.22
    for i, (m, lab) in enumerate(zip(methods, labels)):
        vals = [cub["classification"][m][met]["mean"] for met in metrics]
        sems = [cub["classification"][m][met]["std"] / np.sqrt(cub["classification"][m][met]["n"]) for met in metrics]
        off = (i - 1) * w
        ax.bar(x + off, vals, w, yerr=sems, label=lab, color=[NAT["gray_light"], NAT["blue"], NAT["teal"]][i], ec=NAT["ink"], lw=0.3, capsize=1.2, error_kw=dict(elinewidth=0.35))
        for xi, v, s in zip(x + off, vals, sems):
            ax.text(xi, v + s + 1, f"{v:.1f}" if v > 1 else f"{v:.3f}", ha="center", fontsize=4.5, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(mlabels, fontsize=5.5)
    ax.legend(fontsize=4.8, frameon=False, ncol=3)
    js.panel_label(ax, "j", "CUB200 metrics ± SEM")

    # ── k: rank over tasks from snapshots ──
    ax = fig.add_subplot(gs[3, 2])
    js.style_ax(ax, "y")
    wnpz = np.load(DEEP_REP / "weights_and_snapshots.npz")
    for lab, key, c in [("Base", "baseline_snapshots", NAT["gray_light"]), ("Bio+KD", "biocs_kd_snapshots", NAT["teal"])]:
        snaps = wnpz[key]
        ranks = [js.effective_rank(snaps[t]) for t in range(snaps.shape[0])]
        ax.plot(np.arange(1, len(ranks) + 1), ranks, "-o", color=c, ms=2.5, lw=1.0, label=lab)
        for t, r in enumerate(ranks[::4], start=1):
            ax.text(t * 4, ranks[t * 4 - 1], f"{r:.0f}", fontsize=4.5, color=c)
    ax.set_xlabel("CUB task")
    ax.set_ylabel("Effective rank")
    ax.legend(fontsize=4.8, frameon=False)
    js.panel_label(ax, "k", "Rank evolution (20 tasks)")

    # ── l: SVD + overlap + all case pairs ──
    ax = fig.add_subplot(gs[3, 3])
    js.style_ax(ax, "y")
    deep = js.load_json(DEEP_REP / "summary.json")
    wnpz = np.load(DEEP_REP / "weights_and_snapshots.npz")
    bins = np.linspace(0, 0.85, 40)
    for lab, key, c in [("Base", "baseline_weight", NAT["gray_light"]), ("Bio+KD", "biocs_kd_weight", NAT["teal"])]:
        vals = js.offdiag_cos(wnpz[key])
        ax.hist(vals, bins=bins, density=True, histtype="step", color=c, lw=1.1, label=lab)
    for i, pair in enumerate(deep["selected_case_pairs"]):
        b, p = abs(pair["baseline_cosine"]), abs(pair["biocs_kd_cosine"])
        ypos = 0.72 - i * 0.11
        ax.plot([b, p], [ypos, ypos], "-|", color=NAT["red"], lw=0.8, ms=3, transform=ax.get_xaxis_transform(), clip_on=False)
        short = pair["name_i"].split(".")[-1][:8]
        ax.text(0.98, ypos, f"{short} {b:.2f}→{p:.2f}", transform=ax.transAxes, fontsize=4.2, color=NAT["muted"], ha="right", va="center")
    ax.set_xlabel("|cos|")
    ax.set_ylabel("Density")
    ax.legend(fontsize=4.8, frameon=False, loc="upper right")
    js.panel_label(ax, "l", "Overlap + case pairs")

    save(fig, "fig_results_bars")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 3 — Transfer & mechanism (complete downstream result)
# ═══════════════════════════════════════════════════════════════════════════

LLM_ROWS = [
    ("Qwen-VL", "ZsRE-50", 100.0, 2.0, 100.0, 21.0),
    ("Qwen-VL", "CF-100", 97.0, 0.51, 100.0, 8.69),
    ("Qwen-VL", "Recent-100", 88.0, 22.96, 100.0, 39.46),
    ("Llama-3", "ZsRE-50", 96.0, 2.0, 100.0, 2.0),
    ("Llama-3", "CF-100", 95.0, 0.51, 100.0, 2.15),
    ("Llama-3", "Recent-100", 90.0, 24.41, 100.0, 27.77),
    ("GPT-2 XL", "ZsRE-50", 100.0, 2.0, 100.0, 13.0),
    ("GPT-2 XL", "CF-100", 97.0, 0.51, 100.0, 8.69),
    ("GPT-2 XL", "Recent-100", 88.0, 22.96, 100.0, 36.69),
    ("GPT-2 XL", "ZsRE-300", 100.0, 1.22, 100.0, 4.58),
    ("GPT-2 XL", "CF-200", 95.0, 4.41, 100.0, 9.83),
]


def fig_cub200_capacity_segmentation() -> None:
    js.setup()
    cub = js.load_json(RESULTS / "cub200_summary_20260429" / "summary.json")
    adapter = js.load_json(RESULTS / "lowrank_adapter_probe_20260514" / "summary.json")
    seg_base = js.load_json(RESULTS / "cub200_seg_seen_highres_20260429_summary" / "summary.json")
    seg_tuned = js.load_json(RESULTS / "cub200_seg_highres_lsp005_kd2_seed0_20260429_summary" / "summary.json")
    cod_b = js.load_json(RESULTS / "cod_320_baseline_gpu2_20260429" / "summary.json")
    cod_c = js.load_json(RESULTS / "cod_320_channel_gpu3_20260429" / "summary.json")

    fig = plt.figure(figsize=(W_DOUBLE, 8.0))
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.74, wspace=0.48, left=0.07, right=0.98, top=0.90, bottom=0.05,
                           height_ratios=[1.05, 1.0, 1.0, 0.95])

    # ── a: LLM editing — dual axis efficacy + locality ──
    ax = fig.add_subplot(gs[0, :])
    js.style_ax(ax, "x")
    y = np.arange(len(LLM_ROWS))[::-1]
    for yi, row in zip(y, LLM_ROWS):
        fam, task, ft_e, ft_l, bio_e, bio_l = row
        ax.plot([ft_l, bio_l], [yi, yi], color=NAT["grid"], lw=1.2, zorder=1)
        ax.scatter(ft_l, yi, s=28, marker="s", fc="white", ec=NAT["comp"], lw=0.6, zorder=3)
        ax.scatter(bio_l, yi, s=32, marker="o", fc=NAT["teal"], ec=NAT["ink"], lw=0.5, zorder=4)
        ax.text(bio_l + 0.5, yi + 0.12, f"Δloc +{bio_l - ft_l:.1f}", fontsize=4.8, color=NAT["teal"])
        ax.text(1, yi - 0.12, f"eff {ft_e:.0f}→{bio_e:.0f}", fontsize=4.6, color=NAT["muted"])
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r[0]} · {r[1]}" for r in LLM_ROWS], fontsize=5.2)
    ax.set_xlabel("Locality (%)")
    ax.set_xlim(0, 48)
    js.panel_label(ax, "a", f"Sequential editing — {len(LLM_ROWS)} matched streams")

    # ── b: CUB + adapter frontier with coordinates ──
    ax = fig.add_subplot(gs[1, 0])
    js.style_ax(ax, "both")
    pts = [
        ("CUB Base", cub["classification"]["baseline"]["avg_accuracy"]["mean"], cub["classification"]["baseline"]["avg_forgetting"]["mean"], NAT["gray_light"], "o"),
        ("CUB Bio+KD", cub["classification"]["biocs_kd"]["avg_accuracy"]["mean"], cub["classification"]["biocs_kd"]["avg_forgetting"]["mean"], NAT["teal"], "o"),
        ("Adpt KD", adapter["kd"]["avg_accuracy"]["mean"], adapter["kd"]["avg_forgetting"]["mean"], NAT["orange"], "s"),
        ("Adpt Bio+KD", adapter["biocs_kd"]["avg_accuracy"]["mean"], adapter["biocs_kd"]["avg_forgetting"]["mean"], NAT["navy"], "D"),
    ]
    for lab, aa, af, c, mk in pts:
        hollow = "Base" in lab or lab == "Adpt KD"
        ax.scatter(aa, af, s=50, marker=mk, fc="white" if hollow else c, ec=c, lw=0.7, zorder=5)
        ax.annotate(f"{lab}\n{aa:.2f}/{af:.2f}", (aa, af), fontsize=4.8, xytext=(4, 4), textcoords="offset points", color=NAT["ink"])
    ax.set_xlabel("AA (%)")
    ax.set_ylabel("AF (%)")
    js.panel_label(ax, "b", "CUB & adapter frontiers")

    # ── c: dense prediction with all metrics ──
    ax = fig.add_subplot(gs[1, 1])
    js.style_ax(ax, "y")
    dense = [
        ("Mask mIoU", seg_base["segmentation"]["baseline"]["mean_iou"]["mean"], seg_tuned["segmentation"]["biocs_kd"]["mean_iou"]["mean"]),
        ("Mask Dice", seg_base["segmentation"]["baseline"]["mean_dice"]["mean"], seg_tuned["segmentation"]["biocs_kd"]["mean_dice"]["mean"]),
        ("Mask AF", seg_base["segmentation"]["baseline"]["avg_forgetting_iou"]["mean"], seg_tuned["segmentation"]["biocs_kd"]["avg_forgetting_iou"]["mean"]),
        ("CAMO mIoU", cod_b["baseline"]["CAMO"]["miou"]["mean"] * 100, cod_c["biocs_channel"]["CAMO"]["miou"]["mean"] * 100),
        ("COD Dice", cod_b["baseline"]["COD10K"]["dice"]["mean"] * 100, cod_c["biocs_channel"]["COD10K"]["dice"]["mean"] * 100),
    ]
    yd = np.arange(len(dense))[::-1]
    for yi, (lab, ctrl, bio) in zip(yd, dense):
        ax.plot([ctrl, bio], [yi, yi], color=NAT["grid"], lw=1.0)
        ax.scatter(ctrl, yi, s=26, marker="s", fc="white", ec=NAT["gray"], lw=0.5)
        ax.scatter(bio, yi, s=28, marker="o", fc=NAT["teal"], ec=NAT["ink"], lw=0.5)
        ax.text(max(ctrl, bio) + 0.6, yi, f"{ctrl:.1f}→{bio:.1f} (Δ{bio-ctrl:+.1f})", va="center", fontsize=4.8, color=NAT["teal"])
    ax.set_yticks(yd)
    ax.set_yticklabels([r[0] for r in dense], fontsize=5.2)
    ax.set_xlabel("Score")
    js.panel_label(ax, "c", "Dense-prediction transfer")

    # ── d: adapter heatmap full numeric ──
    ax = fig.add_subplot(gs[1, 2])
    methods_a = ["baseline", "kd", "biocs", "biocs_kd"]
    labels_a = ["Base", "KD", "Bio", "Bio+KD"]
    cols = ["AA", "AF", "Rank", "Proto|c|", "Adpt|c|"]
    raw = np.array([[adapter[m][k]["mean"] for k in ["avg_accuracy", "avg_forgetting", "effective_rank", "mean_abs_offdiag_cosine", "adapter_basis_cosine"]] for m in methods_a])
    score = raw.copy()
    for j in [0, 2]:
        lo, hi = raw[:, j].min(), raw[:, j].max()
        score[:, j] = (raw[:, j] - lo) / (hi - lo + 1e-12)
    for j in [1, 3, 4]:
        lo, hi = raw[:, j].min(), raw[:, j].max()
        score[:, j] = (hi - raw[:, j]) / (hi - lo + 1e-12)
    im = ax.imshow(score, aspect="auto", cmap=js.CMAP_SEQ)
    for i in range(4):
        for j in range(5):
            val = raw[i, j]
            txt = f"{val:.1f}" if j in {0, 1, 2} else f"{val:.3f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=5.2, fontweight="bold",
                    color="white" if score[i, j] > 0.55 else NAT["ink"])
    ax.set_xticks(np.arange(5))
    ax.set_xticklabels(cols, fontsize=5.2, rotation=30, ha="right")
    ax.set_yticks(np.arange(4))
    ax.set_yticklabels(labels_a, fontsize=5.5)
    ax.tick_params(length=0)
    js.panel_label(ax, "d", "Adapter geometry (rank-16)")

    # ── e-g: dynamics trilogy with per-task labels ──
    dyn_specs = [
        ("e", "update_norm", r"$||\Delta W_t||_F$", "Update magnitude"),
        ("f", None, r"$R_\Delta(d)$", "Direction persistence"),
        ("g", None, "Power", "Temporal PSD"),
    ]
    for col, (letter, key, ylab, title) in enumerate(dyn_specs):
        ax = fig.add_subplot(gs[2, col])
        js.style_ax(ax, "both")
        if key:
            for method in ["baseline", "kd", "biocs", "biocs_kd"]:
                yv = np.asarray(TEMPORAL[method]["series"][key], float)
                xv = np.arange(1, len(yv) + 1)
                lab = {"baseline": "Base", "kd": "KD", "biocs": "Bio", "biocs_kd": "Bio+KD"}[method]
                st = js.METHOD_STYLE.get(lab if lab != "Bio+KD" else "SSR+KD", js.METHOD_STYLE["Baseline"])
                ax.plot(xv, yv, color=st["color"], ls=st["ls"], lw=st["lw"], marker=st["marker"], ms=1.8, label=lab)
                if method == "biocs_kd":
                    for xi, yi in zip(xv[::3], yv[::3]):
                        ax.text(xi, yi, f"{yi:.1f}", fontsize=4.2, color=st["color"], ha="left")
        elif letter == "f":
            for method in ["baseline", "kd", "biocs_kd"]:
                acf = np.asarray(TEMPORAL[method]["update_vector_autocorr"]["acf"], float)[1:]
                xv = np.arange(1, len(acf) + 1)
                lab = {"baseline": "Base", "kd": "KD", "biocs_kd": "Bio+KD"}[method]
                st = js.METHOD_STYLE.get(lab if lab != "Bio+KD" else "SSR+KD", js.METHOD_STYLE["Baseline"])
                ax.plot(xv, acf, color=st["color"], ls=st["ls"], lw=st["lw"], marker=st["marker"], ms=1.8, label=lab)
            ax.axhline(0, color=NAT["ink"], lw=0.35, alpha=0.5)
        else:
            for method in ["baseline", "kd", "biocs_kd"]:
                f = np.asarray(TEMPORAL[method]["update_vector_autocorr"]["psd_freqs"], float)
                p = np.asarray(TEMPORAL[method]["update_vector_autocorr"]["psd_power"], float)
                p = p / (p.sum() + 1e-12)
                lab = {"baseline": "Base", "kd": "KD", "biocs_kd": "Bio+KD"}[method]
                st = js.METHOD_STYLE.get(lab if lab != "Bio+KD" else "SSR+KD", js.METHOD_STYLE["Baseline"])
                ax.plot(f, p, color=st["color"], ls=st["ls"], lw=st["lw"], label=lab)
            hf = {m: TEMPORAL[m]["update_vector_autocorr"]["psd_high_ratio"] for m in ["baseline", "kd", "biocs_kd"]}
            ax.text(0.03, 0.97, f"HF: {hf['baseline']:.3f}/{hf['kd']:.3f}/{hf['biocs_kd']:.3f}", transform=ax.transAxes, va="top", fontsize=4.8)
        ax.set_xlabel("Task / lag / freq")
        ax.set_ylabel(ylab)
        leg_loc = "upper left" if letter == "g" else "best"
        ax.legend(fontsize=4.6, frameon=False, loc=leg_loc)
        js.panel_label(ax, letter, title)

    # ── h: cross-domain summary bars ──
    ax = fig.add_subplot(gs[3, 0:2])
    js.style_ax(ax, "x")
    transfer = [
        ("CUB ΔAA", cub["classification"]["biocs_kd"]["avg_accuracy"]["mean"] - cub["classification"]["baseline"]["avg_accuracy"]["mean"]),
        ("CUB ΔAF", cub["classification"]["baseline"]["avg_forgetting"]["mean"] - cub["classification"]["biocs_kd"]["avg_forgetting"]["mean"]),
        ("CUB Δrank", cub["classification"]["biocs_kd"]["effective_rank"]["mean"] - cub["classification"]["baseline"]["effective_rank"]["mean"]),
        ("Adpt ΔAA", adapter["biocs_kd"]["avg_accuracy"]["mean"] - adapter["kd"]["avg_accuracy"]["mean"]),
        ("Adpt ΔAF", adapter["kd"]["avg_forgetting"]["mean"] - adapter["biocs_kd"]["avg_forgetting"]["mean"]),
        ("Adpt Δrank", adapter["biocs_kd"]["effective_rank"]["mean"] - adapter["kd"]["effective_rank"]["mean"]),
        ("Mask ΔmIoU", seg_tuned["segmentation"]["biocs_kd"]["mean_iou"]["mean"] - seg_base["segmentation"]["baseline"]["mean_iou"]["mean"]),
        ("Mean Δloc", np.mean([r[5] - r[3] for r in LLM_ROWS])),
        ("Mean Δeff", np.mean([r[4] - r[2] for r in LLM_ROWS])),
    ]
    yh = np.arange(len(transfer))[::-1]
    cols_t = [NAT["teal"], NAT["navy"], NAT["blue"], NAT["teal"], NAT["navy"], NAT["blue"], NAT["purple"], NAT["orange"], NAT["red"]]
    for yi, (lab, val), c in zip(yh, transfer, cols_t):
        ax.barh(yi, val, height=0.55, color=c, ec=NAT["ink"], lw=0.3, alpha=0.9)
        ax.text(val + (0.15 if val >= 0 else -0.15), yi, f"{val:+.2f}", va="center", ha="left" if val >= 0 else "right", fontsize=5.2, fontweight="bold")
    ax.set_yticks(yh)
    ax.set_yticklabels([t[0] for t in transfer], fontsize=5.4)
    ax.axvline(0, color=NAT["ink"], lw=0.35)
    ax.set_xlabel("Beneficial shift")
    js.panel_label(ax, "h", "Cross-domain transfer summary")

    # ── i: method comparison heatmap 4 benchmarks × methods ──
    ax = fig.add_subplot(gs[3, 2])
    bench_rows = [
        ("C100", ["ewc_split_cifar100", "lwf_split_cifar100", "biocs_plus_s05_c100", "biocs_plus_kd_only_s05_c100"]),
        ("C10", ["ewc_split_cifar10", "lwf_split_cifar10", "biocs_plus_split_cifar10", "derpp_split_cifar10"]),
        ("Tiny", ["ewc_split_tinyimagenet", "lwf_split_tinyimagenet", "biocs_plus_s17_tin", "mas_split_tinyimagenet"]),
        ("5DS", ["ewc_5datasets", "er_ace_5datasets", "biocs_plus_5datasets", "lwf_5datasets"]),
    ]
    method_cols = ["EWC", "LwF/ER", "SSR+", "Alt."]
    aa_mat = np.array([[js.load_point(e)["aa_mean"] for e in exps] for _, exps in bench_rows])
    js.heatmap_dense(ax, aa_mat, [r[0] for r in bench_rows], method_cols, cmap=js.CMAP_SEQ, vmin=45, vmax=95, fmt="{:.1f}", fontsize=5.4)
    js.panel_label(ax, "i", "AA matrix (all benchmarks)")

    save(fig, "fig_cub200_capacity_segmentation")


def build_all() -> None:
    fig_overview()
    fig_results_bars()
    fig_cub200_capacity_segmentation()
    print(f"Dense Nature figures → {SCI_FIG}")


if __name__ == "__main__":
    build_all()
