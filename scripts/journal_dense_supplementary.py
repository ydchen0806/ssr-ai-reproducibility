#!/usr/bin/env python3
"""Dense Nature-style supplementary figures, matched to main-text layout/fonts."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

import journal_style as js
from journal_style import ANNO_PT, CELL_PT, FIG_HEIGHT, GRID_MAIN, NAT, W_DOUBLE

SCI_FIG = Path(os.environ.get("SSR_FIGURE_DIR", Path(__file__).resolve().parents[1] / "figures"))
RESULTS = js.RESULTS
TEMPORAL = js.load_json(RESULTS / "temporal_learning_dynamics_20260514" / "temporal_learning_dynamics_summary.json")
FOURIER = js.load_json(RESULTS / "fourier_update_analysis_20260514" / "fourier_update_summary.json")
REPRESENTATION = js.load_json(RESULTS / "representation_geometry_20260430" / "summary.json")

# Supplementary caption: Qwen-VL + Llama-3 paired slices (Table S / Fig S1)
SUPP_LLM_ROWS = [
    ("Qwen2.5-VL-3B", "ZsRE-50", 100.0, 2.0, 100.0, 21.0),
    ("Qwen2.5-VL-3B", "CF-100", 97.0, 0.51, 100.0, 8.69),
    ("Qwen2.5-VL-3B", "Recent-100", 88.0, 22.96, 100.0, 39.46),
    ("Llama-3-8B", "ZsRE-50", 96.0, 2.0, 100.0, 2.0),
    ("Llama-3-8B", "CF-100", 95.0, 0.51, 100.0, 2.15),
    ("Llama-3-8B", "Recent-100", 90.0, 24.41, 100.0, 27.77),
]

METHOD_ORDER = ["baseline", "biocs", "biocs_kd"]
METHOD_LABELS = ["Base", "SSR", "SSR+KD"]
STYLE_LABELS = {
    "SSR": "SSR",
    "SSR+": "SSR+",
    "SSR+KD": "SSR+KD",
    "SSR-FT": "SSR",
    "Base": "Baseline",
}


def save(fig, stem: str) -> None:
    SCI_FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(SCI_FIG / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(SCI_FIG / f"{stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _method_style(lab: str) -> dict:
    key = STYLE_LABELS.get(lab, lab)
    return js.METHOD_STYLE.get(key, js.METHOD_STYLE["Baseline"])


def draw_llm_schematic(ax) -> None:
    """Vector schematic aligned with main-text figure quality."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(patches.FancyBboxPatch((0.03, 0.08), 0.94, 0.82, boxstyle="round,pad=0.02", fc=NAT["band"], ec=NAT["ink"], lw=0.5))
    stages = [
        (0.17, "Sequential\nedit stream", NAT["orange"], "Facts $f_1…f_T$ enter\none at a time"),
        (0.50, "SSR-FT\nspatial prior", NAT["teal"], r"Center–surround on $\Delta W$\nreduces cross-edit leakage"),
        (0.83, "Stable\nlocality", NAT["navy"], "Unrelated queries\nstay unchanged"),
    ]
    for cx, title, color, note in stages:
        ax.text(cx, 0.88, title, ha="center", fontsize=6.8, fontweight="bold", color=NAT["ink"])
        ax.add_patch(patches.Circle((cx, 0.55), 0.09, fc=color, ec=NAT["ink"], lw=0.5, alpha=0.88))
        for ang in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            rx, ry = cx + 0.17 * np.cos(ang), 0.55 + 0.14 * np.sin(ang)
            ax.plot([cx, rx], [0.55, ry], color=NAT["grid"], lw=0.5)
            ax.add_patch(patches.Circle((rx, ry), 0.024, fc=color, ec=NAT["ink"], lw=0.35, alpha=0.9))
        if cx == 0.50:
            ax.add_patch(patches.Ellipse((cx, 0.55), 0.44, 0.32, fill=False, ec=NAT["ink"], lw=0.7, ls="--"))
        ax.text(cx, 0.16, note, ha="center", fontsize=5.6, color=NAT["muted"], linespacing=1.15)
    for x0, x1 in [(0.28, 0.36), (0.64, 0.72)]:
        ax.annotate("", xy=(x1, 0.55), xytext=(x0, 0.55), arrowprops=dict(arrowstyle="-|>", lw=0.9, color=NAT["ink"]))
    ax.text(0.50, 0.04, "Locality = fraction of unrelated prompts unchanged after editing", ha="center", fontsize=5.6, color=NAT["muted"])


def _plot_temporal_update(ax, *, annotate_bio: bool = True) -> None:
    js.style_ax(ax, "both")
    for method in ["baseline", "kd", "biocs", "biocs_kd"]:
        yv = np.asarray(TEMPORAL[method]["series"]["update_norm"], float)
        xv = np.arange(1, len(yv) + 1)
        lab = {"baseline": "Base", "kd": "KD", "biocs": "SSR", "biocs_kd": "SSR+KD"}[method]
        st = _method_style(lab)
        ax.plot(xv, yv, color=st["color"], ls=st["ls"], lw=st["lw"], marker=st["marker"], ms=2.2, label=lab)
        if annotate_bio and method == "biocs_kd":
            for xi, yi in zip(xv[::3], yv[::3]):
                ax.text(xi, yi, f"{yi:.1f}", fontsize=ANNO_PT, color=st["color"], ha="left", va="bottom")
    ax.set_xlabel("Task transition")
    ax.set_ylabel(r"$||\Delta W_t||_F$")


def _plot_temporal_acf(ax) -> None:
    js.style_ax(ax, "both")
    x_vals = np.arange(1, len(np.asarray(TEMPORAL["baseline"]["update_vector_autocorr"]["acf"], float)[1:]) + 1)
    for method in ["baseline", "kd", "biocs_kd"]:
        yv = np.asarray(TEMPORAL[method]["update_vector_autocorr"]["acf"], float)[1:]
        lab = {"baseline": "Base", "kd": "KD", "biocs_kd": "SSR+KD"}[method]
        st = _method_style(lab)
        ax.plot(x_vals, yv, color=st["color"], ls=st["ls"], lw=st["lw"], marker=st["marker"], ms=2.0, label=lab)
    ax.axhline(0, color=NAT["ink"], lw=0.35, alpha=0.5)
    lag1_kd = TEMPORAL["kd"]["update_vector_autocorr"]["acf"][1]
    lag1_bio = TEMPORAL["biocs_kd"]["update_vector_autocorr"]["acf"][1]
    psd_kd = TEMPORAL["kd"]["update_vector_autocorr"]["psd_high_ratio"]
    psd_bio = TEMPORAL["biocs_kd"]["update_vector_autocorr"]["psd_high_ratio"]
    ax.text(
        0.03, 0.97,
        f"lag-1: {lag1_kd:.3f} / {lag1_bio:.3f}\nPSD-high: {psd_kd:.3f} / {psd_bio:.3f}",
        transform=ax.transAxes, va="top", fontsize=ANNO_PT, color=NAT["muted"],
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=NAT["grid"], lw=0.4),
    )
    ax.set_xlabel("Lag (task transitions)")
    ax.set_ylabel(r"mean cosine $R_\Delta(d)$")


def _plot_temporal_psd(ax) -> None:
    js.style_ax(ax, "both")
    for method in ["baseline", "kd", "biocs_kd"]:
        f = np.asarray(TEMPORAL[method]["update_vector_autocorr"]["psd_freqs"], float)
        p = np.asarray(TEMPORAL[method]["update_vector_autocorr"]["psd_power"], float)
        p = p / (p.sum() + 1e-12)
        lab = {"baseline": "Base", "kd": "KD", "biocs_kd": "SSR+KD"}[method]
        st = _method_style(lab)
        ax.plot(f, p, color=st["color"], ls=st["ls"], lw=st["lw"], label=lab)
    hf = {m: TEMPORAL[m]["update_vector_autocorr"]["psd_high_ratio"] for m in ["baseline", "kd", "biocs_kd"]}
    ax.text(0.03, 0.97, f"HF ratio: {hf['baseline']:.3f} / {hf['kd']:.3f} / {hf['biocs_kd']:.3f}",
            transform=ax.transAxes, va="top", fontsize=ANNO_PT, color=NAT["muted"])
    ax.set_xlabel("Normalized frequency")
    ax.set_ylabel("Power")


def fig_llm_family_transfer() -> None:
    js.setup()
    fig = plt.figure(figsize=(W_DOUBLE, FIG_HEIGHT))
    gs = gridspec.GridSpec(
        4, 4, figure=fig,
        height_ratios=[0.95, 1.05, 1.0, 0.95],
        **GRID_MAIN,
    )

    # ── a: schematic ──
    ax = fig.add_subplot(gs[0, 0:2])
    draw_llm_schematic(ax)
    js.panel_label(ax, "a", "Locality-protected sequential editing")

    # ── b: paired editing matrix ──
    ax = fig.add_subplot(gs[0, 2:4])
    short = [("Q-VL", "Z50"), ("Q-VL", "CF100"), ("Q-VL", "R100"), ("Llama", "Z50"), ("Llama", "CF100"), ("Llama", "R100")]
    rows = [f"{f}\n{t}" for f, t in short]
    raw = np.array([[r[2], r[3], r[4], r[5], r[4] - r[2], r[5] - r[3]] for r in SUPP_LLM_ROWS], dtype=float)
    score = raw.copy()
    for j in range(raw.shape[1]):
        lo, hi = raw[:, j].min(), raw[:, j].max()
        score[:, j] = (raw[:, j] - lo) / (hi - lo + 1e-12)
    cols = ["FT\neff.", "FT\nloc.", "SSR\neff.", "SSR\nloc.", "Δeff.", "Δloc."]
    js.heatmap_dense(ax, score, rows, cols, cmap=js.CMAP_SEQ, vmin=0, vmax=1, fmt="", fontsize=CELL_PT)
    for i in range(raw.shape[0]):
        for j in range(raw.shape[1]):
            val = raw[i, j]
            txt = f"{val:.0f}" if j in {0, 2, 4} else f"{val:.1f}"
            color = "white" if score[i, j] > 0.63 else NAT["ink"]
            ax.text(j, i, txt, ha="center", va="center", fontsize=CELL_PT, fontweight="bold", color=color)
    js.panel_label(ax, "b", "Paired efficacy / locality matrix")

    # ── c: locality gains (same style as main Fig. 3a) ──
    ax = fig.add_subplot(gs[1, 0:2])
    js.style_ax(ax, "x")
    y = np.arange(len(SUPP_LLM_ROWS))[::-1]
    for yi, row in zip(y, SUPP_LLM_ROWS):
        fam, task, ft_e, ft_l, bio_e, bio_l = row
        ax.plot([ft_l, bio_l], [yi, yi], color=NAT["grid"], lw=1.2, zorder=1)
        ax.scatter(ft_l, yi, s=28, marker="s", fc="white", ec=NAT["comp"], lw=0.6, zorder=3)
        ax.scatter(bio_l, yi, s=32, marker="o", fc=NAT["teal"], ec=NAT["ink"], lw=0.5, zorder=4)
        ax.text(bio_l + 0.6, yi, f"Δloc +{bio_l - ft_l:.1f}", va="center", fontsize=ANNO_PT, color=NAT["teal"])
        ax.text(1.5, yi, f"eff {ft_e:.0f}→{bio_e:.0f}", va="center", fontsize=ANNO_PT, color=NAT["muted"])
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r[0].replace('2.5-VL-3B', 'VL')} · {r[1]}" for r in SUPP_LLM_ROWS], fontsize=5.2)
    ax.set_xlabel("Locality (%)")
    ax.set_xlim(0, 46)
    ax.scatter([], [], marker="s", s=24, fc="white", ec=NAT["comp"], label="FT")
    ax.scatter([], [], marker="o", s=26, fc=NAT["teal"], ec=NAT["ink"], label="SSR-FT")
    ax.legend(fontsize=5.6, frameon=False, loc="lower right")
    js.panel_label(ax, "c", "Locality gains on paired slices")

    # ── c companion: Δ summary bars ──
    ax = fig.add_subplot(gs[1, 2:4])
    js.style_ax(ax, "y")
    labels_d = [f"{s[0].split('-')[0]}\n{s[1]}" for s in short]
    deff = raw[:, 4]
    dloc = raw[:, 5]
    x = np.arange(len(labels_d))
    ax.bar(x - 0.18, deff, 0.32, color=NAT["blue"], ec=NAT["ink"], lw=0.3, label="Δeff.", alpha=0.85)
    ax.bar(x + 0.18, dloc, 0.32, color=NAT["teal"], ec=NAT["ink"], lw=0.3, label="Δloc.", alpha=0.85)
    for i, (e, l) in enumerate(zip(deff, dloc)):
        ax.text(i - 0.18, e + 0.4, f"{e:+.0f}", ha="center", fontsize=ANNO_PT)
        ax.text(i + 0.18, l + 0.4, f"{l:+.1f}", ha="center", fontsize=ANNO_PT)
    ax.axhline(0, color=NAT["ink"], lw=0.35)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_d, fontsize=5.2, rotation=30, ha="right")
    ax.set_ylabel("Gain (pp)")
    ax.legend(fontsize=5.6, frameon=False, ncol=2, loc="upper left")
    ax.text(0.02, 0.98, "Paired gain summary", transform=ax.transAxes, fontsize=5.8, color=NAT["muted"], va="top")

    # ── d: update magnitude ──
    ax = fig.add_subplot(gs[2, 0:2])
    _plot_temporal_update(ax)
    ax.legend(fontsize=5.6, frameon=False, ncol=4, loc="upper right")
    js.panel_label(ax, "d", "Update magnitude (CUB temporal audit)")

    # ── e: direction persistence + PSD (twin panel via nested spec) ──
    inner = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[2, 2:4], wspace=0.38)
    ax_e = fig.add_subplot(inner[0])
    _plot_temporal_acf(ax_e)
    ax_e.legend(fontsize=5.6, frameon=False, loc="lower right")
    js.panel_label(ax_e, "e", "Direction persistence")

    ax_psd = fig.add_subplot(inner[1])
    _plot_temporal_psd(ax_psd)
    ax_psd.legend(fontsize=5.6, frameon=False, loc="upper right")
    ax_psd.text(0.03, 0.03, "PSD readout", transform=ax_psd.transAxes, fontsize=5.8, color=NAT["muted"])

    # ── row 4: temporal metric readout table ──
    ax = fig.add_subplot(gs[3, :])
    ax.axis("off")
    metrics = [
        ("Update norm (mean)", [TEMPORAL[m]["update_norm"]["mean"] for m in METHOD_ORDER]),
        ("Update norm (final)", [float(np.asarray(TEMPORAL[m]["series"]["update_norm"], float)[-1]) for m in METHOD_ORDER]),
        ("Lag-1 ACF", [TEMPORAL[m]["update_vector_autocorr"]["acf"][1] for m in METHOD_ORDER]),
        ("PSD high-ratio", [TEMPORAL[m]["update_vector_autocorr"]["psd_high_ratio"] for m in METHOD_ORDER]),
    ]
    headers = ["Metric", *METHOD_LABELS]
    for j, h in enumerate(headers):
        ax.text(0.02 + j * 0.22, 0.82, h, fontsize=6.4, fontweight="bold", color=NAT["ink"])
    for i, (name, vals) in enumerate(metrics):
        y = 0.58 - i * 0.18
        ax.text(0.02, y, name, fontsize=6.0, color=NAT["muted"])
        for j, v in enumerate(vals):
            ax.text(0.02 + (j + 1) * 0.22, y, f"{v:.3f}", fontsize=6.2, fontweight="bold", color=NAT["navy"])

    save(fig, "fig_llm_family_transfer")


def fig_supp_support_atlas() -> None:
    js.setup()
    fig = plt.figure(figsize=(W_DOUBLE, FIG_HEIGHT))
    gs = gridspec.GridSpec(
        4, 4, figure=fig,
        height_ratios=[0.95, 1.05, 1.0, 0.95],
        **GRID_MAIN,
    )
    view_specs = [("Flat", "flat"), ("Class", "class_axis"), ("Feature", "feature_axis"), ("2D", "radial_2d")]

    # ── a: HF ratio heatmap ──
    ax = fig.add_subplot(gs[0, 0])
    hf = np.array([[FOURIER[m]["aggregate"][f"{key}_high_ratio"]["mean"] for _, key in view_specs] for m in METHOD_ORDER])
    js.heatmap_dense(ax, hf, METHOD_LABELS, [v[0] for v in view_specs], cmap=js.CMAP_DIV, vmin=0.16, vmax=0.42, fmt="{:.3f}", fontsize=CELL_PT)
    js.panel_label(ax, "a", "Axis-aware HF ratios")

    # ── b: radial spectra ──
    ax = fig.add_subplot(gs[0, 1:3])
    js.style_ax(ax, "both")
    for method, lab in zip(METHOD_ORDER, METHOD_LABELS):
        spectra = np.asarray(FOURIER[method]["mean_spectra"]["radial_2d"], float)
        spectra = spectra / (spectra.sum() + 1e-12)
        x = np.linspace(0, 1, len(spectra))
        st = _method_style(lab)
        ax.plot(x, spectra, color=st["color"], marker=st["marker"], ms=2.2, lw=st["lw"], label=lab)
    ax.axvspan(0.55, 1.0, color=NAT["orange"], alpha=0.10, lw=0)
    ax.text(0.78, 0.92, "HF band", transform=ax.transAxes, fontsize=5.8, color=NAT["orange"], ha="center")
    ax.set_xlabel("Radial 2D frequency")
    ax.set_ylabel("Normalized power")
    ax.legend(fontsize=5.6, frameon=False, ncol=3, loc="upper right")
    js.panel_label(ax, "b", "Mean 2D radial spectra")

    # ── c: absolute HF power ──
    ax = fig.add_subplot(gs[0, 3])
    js.style_ax(ax, "x")
    rel_rows = []
    for label, key in view_specs:
        base = FOURIER["baseline"]["aggregate"][f"{key}_high_ratio"]["mean"] * FOURIER["baseline"]["aggregate"][f"{key}_total_power"]["mean"]
        bio = FOURIER["biocs"]["aggregate"][f"{key}_high_ratio"]["mean"] * FOURIER["biocs"]["aggregate"][f"{key}_total_power"]["mean"]
        plus = FOURIER["biocs_kd"]["aggregate"][f"{key}_high_ratio"]["mean"] * FOURIER["biocs_kd"]["aggregate"][f"{key}_total_power"]["mean"]
        rel_rows.append((label, bio / base, plus / base))
    y = np.arange(len(rel_rows))[::-1]
    for yi, (label, bio_rel, plus_rel) in zip(y, rel_rows):
        ax.plot([1.0, plus_rel], [yi, yi], color=NAT["grid"], lw=1.0, zorder=1)
        ax.scatter(1.0, yi, s=26, marker="o", fc="white", ec=NAT["gray_light"], lw=0.5, zorder=3)
        ax.scatter(bio_rel, yi, s=28, marker="^", fc="white", ec=NAT["blue"], lw=0.5, zorder=4)
        ax.scatter(plus_rel, yi, s=30, marker="D", fc=NAT["teal"], ec=NAT["ink"], lw=0.5, zorder=5)
        ax.text(plus_rel + 0.025, yi, f"{plus_rel:.2f}×", va="center", fontsize=ANNO_PT, color=NAT["muted"])
    ax.axvline(1.0, color=NAT["ink"], lw=0.5, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rel_rows], fontsize=5.2)
    ax.set_xlabel("HF band power / baseline")
    ax.set_xlim(0.72, 1.55)
    js.panel_label(ax, "c", "Absolute HF band power")

    # ── d: forgetting progression (match main Fig. 2i style) ──
    ax = fig.add_subplot(gs[1, 0:2])
    js.style_ax(ax, "both")
    tasks = np.arange(2, 11)
    curve_specs = [
        ("SSR+", "biocs_plus_s05_c100"),
        ("KD-only", "biocs_plus_kd_only_s05_c100"),
        ("LwF", "lwf_split_cifar100"),
        ("EWC", "ewc_split_cifar100"),
    ]
    endpoints = []
    for name, exp in curve_specs:
        mean, sem, curves = js.mean_forget_curve(exp)
        st = _method_style(name)
        for c in curves:
            ax.plot(tasks[: len(c)], c, color=st["color"], alpha=0.30, lw=0.6)
        ax.plot(tasks, mean, color=st["color"], ls=st["ls"], lw=st["lw"], marker=st["marker"], ms=2.2, label=name)
        ax.fill_between(tasks, mean - sem, mean + sem, color=st["color"], alpha=0.12, lw=0)
        endpoints.append((name, float(mean[-1])))
    ax.set_xlabel("Tasks learned")
    ax.set_ylabel("Cumulative forgetting (%)")
    ax.legend(fontsize=5.6, frameon=False, ncol=2, loc="upper left")
    js.panel_label(ax, "d", "Split-CIFAR-100 forgetting progression")

    # ── e: endpoint retention ──
    ax = fig.add_subplot(gs[1, 2])
    js.style_ax(ax, "x")
    extra = [("MAS", "mas_split_cifar100"), ("SI", "si_split_cifar100")]
    all_end = list(endpoints)
    for name, exp in extra:
        mean, _, _ = js.mean_forget_curve(exp)
        all_end.append((name, float(mean[-1])))
    order = ["SSR+", "LwF", "MAS", "SI", "EWC"]
    emap = {n: v for n, v in all_end}
    y = np.arange(len(order))[::-1]
    for yi, name in zip(y, order):
        val = emap.get(name, 0)
        st = _method_style(name)
        ax.plot([0, val], [yi, yi], color=NAT["grid"], lw=1.0, zorder=1)
        ax.scatter(val, yi, s=32, marker=st["marker"], fc=st["color"] if name != "LwF" else "white",
                   ec=st["color"] if name == "LwF" else NAT["ink"], lw=0.6, zorder=3)
        ax.text(val + 1.0, yi, f"{val:.1f}", va="center", fontsize=ANNO_PT, fontweight="bold" if name == "SSR+" else "normal")
    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=5.2)
    ax.set_xlabel("Final AF (%)")
    js.panel_label(ax, "e", "Endpoint retention")

    # ── f: CUB order robustness ──
    ax = fig.add_subplot(gs[1, 3])
    cub_rows = []
    for group_label, group_key in [("Sem", "cub_semantic"), ("Rand", "cub_random")]:
        cls = REPRESENTATION[group_key]["classification"]
        for method, label in zip(METHOD_ORDER, METHOD_LABELS):
            d = cls[method]
            cub_rows.append((f"{group_label} {label}", d["avg_accuracy"]["mean"], d["avg_forgetting"]["mean"],
                             d["effective_rank"]["mean"], d["mean_abs_offdiag_cosine"]["mean"]))
    raw = np.asarray([[r[1], r[2], r[3], r[4]] for r in cub_rows])
    score = raw.copy()
    for j in [0, 2]:
        lo, hi = raw[:, j].min(), raw[:, j].max()
        score[:, j] = (raw[:, j] - lo) / (hi - lo + 1e-12)
    for j in [1, 3]:
        lo, hi = raw[:, j].min(), raw[:, j].max()
        score[:, j] = (hi - raw[:, j]) / (hi - lo + 1e-12)
    js.heatmap_dense(ax, score, [r[0] for r in cub_rows], ["AA", "AF", "rank", "|cos|"], cmap=js.CMAP_SEQ, vmin=0, vmax=1, fmt="", fontsize=5.0)
    for i in range(raw.shape[0]):
        for j in range(raw.shape[1]):
            txt = f"{raw[i, j]:.1f}" if j != 3 else f"{raw[i, j]:.3f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=5.0, fontweight="bold",
                    color="white" if score[i, j] > 0.62 else NAT["ink"])
    js.panel_label(ax, "f", "CUB order robustness")

    # ── g: rank vs accuracy ──
    ax = fig.add_subplot(gs[2, 0:2])
    js.style_ax(ax, "both")
    for dataset_label, group_key, marker in [("Synthetic", "synthetic", "o"), ("CUB sem.", "cub_semantic", "s")]:
        group = REPRESENTATION[group_key] if group_key == "synthetic" else REPRESENTATION[group_key]["classification"]
        for method in METHOD_ORDER:
            d = group[method]
            lab = METHOD_LABELS[METHOD_ORDER.index(method)]
            st = _method_style(lab)
            hollow = method == "baseline"
            x_val, y_val = d["effective_rank"]["mean"], d["avg_accuracy"]["mean"]
            ax.scatter(x_val, y_val, s=48, marker=marker, fc="white" if hollow else st["color"], ec=st["color"], lw=0.6, zorder=4)
            if method == "biocs_kd":
                ax.annotate(
                    f"{dataset_label}\n{y_val:.1f}/{d['avg_forgetting']['mean']:.2f}",
                    (x_val, y_val), fontsize=ANNO_PT, xytext=(6, 4), textcoords="offset points", color=NAT["ink"],
                )
        base, plus = group["baseline"], group["biocs_kd"]
        ax.annotate("", xy=(plus["effective_rank"]["mean"], plus["avg_accuracy"]["mean"]),
                    xytext=(base["effective_rank"]["mean"], base["avg_accuracy"]["mean"]),
                    arrowprops=dict(arrowstyle="-|>", color=NAT["grid"], lw=0.8))
    ax.set_xlabel("Effective rank")
    ax.set_ylabel("Average accuracy (%)")
    js.panel_label(ax, "g", "Rank expansion requires KD anchor")

    # ── h: COD boundary ──
    ax = fig.add_subplot(gs[2, 2:4])
    js.style_ax(ax, "y")
    cod_scores = [
        ("Targeted channel", REPRESENTATION["cod_320_macro_score"]["baseline"], REPRESENTATION["cod_320_macro_score"]["biocs_channel"], "+0.013"),
        ("Over-broad query+ch.", REPRESENTATION["cod_boundary_macro_score"]["baseline"], REPRESENTATION["cod_boundary_macro_score"]["biocs_both"], "-0.007"),
    ]
    x = np.arange(len(cod_scores))
    w = 0.30
    for i, (lab, base, bio, delta) in enumerate(cod_scores):
        ax.bar(i - w / 2, base, w, color="white", ec=NAT["gray_light"], lw=0.5, label="Baseline" if i == 0 else "")
        ax.bar(i + w / 2, bio, w, color=NAT["teal"] if bio >= base else NAT["red"], ec=NAT["ink"], lw=0.4, alpha=0.88, label="SSR" if i == 0 else "")
        ax.plot([i - w / 2, i + w / 2], [base, bio], color=NAT["teal"] if bio >= base else NAT["red"], lw=0.9)
        ax.text(i, max(base, bio) + 0.005, delta, ha="center", fontsize=ANNO_PT, fontweight="bold")
        ax.text(i - w / 2, base - 0.005, f"{base:.3f}", ha="center", va="top", fontsize=ANNO_PT)
        ax.text(i + w / 2, bio + 0.001, f"{bio:.3f}", ha="center", va="bottom", fontsize=ANNO_PT)
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in cod_scores], fontsize=5.5)
    ax.set_ylabel("Macro score")
    ax.legend(fontsize=5.6, frameon=False, loc="upper right")
    js.panel_label(ax, "h", "COD target and dose boundary")

    # ── row 4: Fourier + retention summary strip ──
    ax = fig.add_subplot(gs[3, :])
    ax.axis("off")
    fourier_note = (
        "Fourier boundary: SSR+KD does not reduce axis-aware HF ratios vs baseline; "
        "absolute HF band power is ≥ baseline (panels a–c). "
        "Retention: SSR+ maintains near-zero AF through task 10 (panel d). "
        "Geometry: semantic vs random CUB order preserves rank/|cos| trend (panel f)."
    )
    ax.text(0.02, 0.55, fourier_note, fontsize=6.0, color=NAT["muted"], va="center", wrap=True)
    summary_vals = [
        ("C100 final AF", f"{emap.get('SSR+', 0):.1f}% vs LwF {emap.get('LwF', 0):.1f}%"),
        ("Flat HF ratio", f"{hf[2, 0]:.3f} vs base {hf[0, 0]:.3f}"),
        ("COD targeted Δ", "+0.013 macro"),
    ]
    for i, (k, v) in enumerate(summary_vals):
        ax.text(0.02 + i * 0.32, 0.15, f"{k}: {v}", fontsize=6.2, fontweight="bold", color=NAT["navy"])

    save(fig, "fig_supp_support_atlas")


def build_supplementary() -> None:
    fig_llm_family_transfer()
    fig_supp_support_atlas()
    print(f"Dense supplementary figures → {SCI_FIG}")


if __name__ == "__main__":
    build_supplementary()
