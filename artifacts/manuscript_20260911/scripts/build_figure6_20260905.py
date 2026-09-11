#!/usr/bin/env python3
"""Build the audited Figure 6 and the moved mask-transfer panel."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import stats

from build_final_strict_figures import (
    CONTROL,
    GRAY,
    GREEN,
    INK,
    LIGHT,
    ORANGE,
    PURPLE,
    SSR,
    clean,
    panel_label,
    save,
    source_line,
    target_class_overlay,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "figure_source_data" / "figure6_20260905"
ROBUST_SOURCE = ROOT / "figure_source_data" / "figure6_20260907"
FIGURES = ROOT / "figures"
V24 = ROOT / "data" / "lowrank_confirmations" / "v24_summary.json"
VOC_CASE = ROOT / "artifacts" / "fig6f_stronger_case_2010_003362"
RESPONSE = ROOT / "source_data" / "figure6_20260905" / "response_maps_rank32_seed8411"
WIDTH = 183 / 25.4
HEIGHT = 168 / 25.4


def rows(name: str) -> list[dict[str, str]]:
    with (SOURCE / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def t_ci(values: np.ndarray) -> tuple[float, float, float]:
    mean = float(values.mean())
    half = float(stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / math.sqrt(len(values)))
    return mean, mean - half, mean + half


def p_text(values: np.ndarray) -> str:
    p_value = float(stats.ttest_1samp(np.asarray(values, dtype=float), 0.0).pvalue)
    return r"$P<0.001$" if p_value < 0.001 else rf"$P={p_value:.3f}$"


def paired_horizontal(ax, data: list[dict[str, str]], metric_order: list[str],
                      comparison: dict, endpoint_key: str, xlim: tuple[float, float]) -> None:
    y_positions = np.arange(len(metric_order))[::-1]
    labels = []
    for y, metric in zip(y_positions, metric_order):
        selected = [item for item in data if item["metric_key"] == metric]
        selected.sort(key=lambda item: int(item["seed"]))
        labels.append(selected[0]["metric_label"])
        control = np.asarray([float(item["lowrank"]) for item in selected])
        treatment = np.asarray([float(item["lowrank_ssr"]) for item in selected])
        jitter = np.linspace(-0.14, 0.14, len(selected))
        for base, target, shift in zip(control, treatment, jitter):
            ax.plot([base, target], [y + shift, y + shift], color=LIGHT, lw=0.48, zorder=1)
            ax.scatter(base, y + shift, s=6.5, facecolor="white", edgecolor=CONTROL,
                       linewidth=0.40, zorder=2)
            ax.scatter(target, y + shift, s=6.5, facecolor="white", edgecolor=SSR,
                       linewidth=0.48, zorder=2)
        ax.plot([control.mean(), treatment.mean()], [y, y], color=SSR, lw=1.2, zorder=3)
        ax.scatter(control.mean(), y, s=23, facecolor="white", edgecolor=INK,
                   linewidth=0.70, zorder=4)
        ax.scatter(treatment.mean(), y, s=24, facecolor=SSR, edgecolor=INK,
                   linewidth=0.42, zorder=4)
        record = comparison[endpoint_key]["lowrank_ssr_vs_lowrank"][metric]
        delta = 100 * float(record["paired_delta_mean"])
        low, high = [100 * float(value) for value in record["bootstrap_95_ci"]]
        ax.text(xlim[1] - 0.10, y + 0.27,
                f"{delta:+.2f} [{low:+.2f}, {high:+.2f}]\n{p_text(treatment - control)}",
                ha="right", va="center", fontsize=4.15, color=SSR, fontweight="bold")
    ax.set_yticks(y_positions, labels)
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.48, len(metric_order) - 0.52)
    clean(ax, "x")


def build_main() -> None:
    voc_endpoint_rows = rows("Fig6a_voc_primary_seed_endpoints.csv")
    voc_stage_rows = rows("Fig6b_voc_primary_stagewise.csv")
    adapter_rows = rows("Fig6d_cub_adapter_absolute_seed_level.csv")
    geometry_rows = rows("Fig6e_lowrank_fixedkd_geometry_rank32_seed_level.csv")
    robustness_rows = list(csv.DictReader(
        (ROBUST_SOURCE / "Fig6e_cifar100_rank4_robustness_seed_level.csv").open(
            newline="", encoding="utf-8"
        )
    ))
    robustness_summary = json.loads(
        (ROBUST_SOURCE / "Fig6e_cifar100_rank4_robustness_summary.json").read_text(
            encoding="utf-8"
        )
    )
    v24 = json.loads(V24.read_text(encoding="utf-8"))

    fig = plt.figure(figsize=(WIDTH, HEIGHT), facecolor="white")
    gs = fig.add_gridspec(4, 14, height_ratios=[1.18, 0.72, 1.06, 0.77],
                          hspace=0.78, wspace=0.92)

    # a | normalized trajectory summaries and final endpoints are separated.
    axa = fig.add_subplot(gs[0, :8])
    panel_label(axa, "a", "Direct low-rank segmentation retention", x=-0.105)
    axa.axis("off")
    auc_ax = axa.inset_axes([0.00, 0.12, 0.46, 0.76])
    final_ax = axa.inset_axes([0.54, 0.12, 0.46, 0.76])
    auc_rows = [item for item in voc_endpoint_rows if item["endpoint_type"] == "normalized trajectory AUC"]
    final_rows = [item for item in voc_endpoint_rows if item["endpoint_type"] == "final endpoint"]
    paired_horizontal(auc_ax, auc_rows, ["all_miou", "old_miou"], v24,
                      "trajectory_auc_comparisons", (38.0, 49.0))
    paired_horizontal(final_ax, final_rows, ["all_miou", "old_miou", "old_boundary_iou"],
                      v24, "final_comparisons", (4.0, 27.0))
    auc_ax.set_title("Normalized trajectory AUC (%)", fontsize=5.2, pad=2.0)
    final_ax.set_title("Final score (%)", fontsize=5.2, pad=2.0)
    auc_ax.scatter([], [], s=16, facecolor="white", edgecolor=CONTROL, label="LowRank")
    auc_ax.scatter([], [], s=16, facecolor=SSR, edgecolor=INK, linewidth=0.35,
                   label="LowRank + SSR")
    auc_ax.legend(frameon=False, loc="lower left", ncol=2, bbox_to_anchor=(0.0, -0.34),
                  handletextpad=0.20, columnspacing=0.55)
    source_line(axa, "PASCAL VOC 2012 10-1 | rank-8 history decoder | primary seeds 12401-12410",
                y=-0.13)

    # b | retain every stage, including the stage-2 decrease.
    axb = fig.add_subplot(gs[0, 9:])
    panel_label(axb, "b", "Stage-wise direct effect", x=-0.12)
    for metric, label, color, marker, style in (
        ("all_miou", "all-class mIoU", SSR, "o", "-"),
        ("old_miou", "old-class mIoU", PURPLE, "s", "--"),
    ):
        selected = sorted([item for item in voc_stage_rows if item["metric"] == metric],
                          key=lambda item: int(item["stage"]))
        x = np.asarray([int(item["stage"]) for item in selected])
        mean = np.asarray([float(item["mean_lowrank_ssr_minus_lowrank"]) for item in selected])
        low = np.asarray([float(item["ci95_low"]) for item in selected])
        high = np.asarray([float(item["ci95_high"]) for item in selected])
        axb.fill_between(x, low, high, color=color, alpha=0.10, linewidth=0)
        axb.plot(x, mean, linestyle=style, color=color, marker=marker, ms=3.0, lw=1.0,
                 label=label)
    axb.axhline(0, color=INK, lw=0.55)
    axb.set_xlim(0.8, 10.2)
    axb.set_ylim(-1.65, 3.0)
    axb.set_xticks([1, 2, 4, 6, 8, 10])
    axb.set_xlabel("incremental stage")
    axb.set_ylabel("SSR effect (pp)", labelpad=1.0)
    axb.legend(frameon=False, loc="upper right", handlelength=1.7)
    clean(axb)
    source_line(axb, "same primary queue | mean and paired 95% bootstrap CI | n=10",
                y=-0.29)

    # c | same image, crop, stage and checkpoints; change is correctness vs GT.
    axc = fig.add_subplot(gs[1, :])
    panel_label(axc, "c", "Pixel-level correction at the final increment", x=-0.035)
    axc.axis("off")
    input_image = np.asarray(Image.open(VOC_CASE / "input.png").convert("RGB"))
    gt_color = np.asarray(Image.open(VOC_CASE / "gt_color.png").convert("RGB"))
    lowrank_color = np.asarray(Image.open(VOC_CASE / "lowrank_color.png").convert("RGB"))
    ssr_color = np.asarray(Image.open(VOC_CASE / "lowrank_ssr_color.png").convert("RGB"))
    target_rgb = (92, 134, 99)
    images = [
        input_image,
        target_class_overlay(input_image, gt_color, target_rgb, (22, 138, 153)),
        target_class_overlay(input_image, lowrank_color, target_rgb, (207, 86, 93)),
        target_class_overlay(input_image, ssr_color, target_rgb, (22, 138, 153)),
        np.asarray(Image.open(VOC_CASE / "change_map.png").convert("RGB")),
    ]
    titles = ["Input", "GT mask", "LowRank", "LowRank + SSR", "Corrected / harmed"]
    for index, (image, title) in enumerate(zip(images, titles)):
        tile = axc.inset_axes([0.004 + 0.199 * index, 0.02, 0.190, 0.88])
        tile.imshow(image)
        tile.set_title(title, fontsize=5.2, pad=1.3, fontweight="bold")
        tile.axis("off")
    axc.text(0.004, -0.02,
             "PASCAL VOC 2012 10-1 | seed 13401, increment 10, image 2010_003362 | rank 8",
             transform=axc.transAxes, ha="left", va="top", fontsize=4.35, color=GRAY)
    axc.text(0.995, -0.02, "blue: corrected  |  warm: harmed  |  present-class mIoU +13.93 pp",
             transform=axc.transAxes, ha="right", va="top", fontsize=4.35, color=GRAY)

    # d | absolute AA/AF from one 40-epoch queue, not deltas reconstructed elsewhere.
    axd = fig.add_subplot(gs[2, :6])
    panel_label(axd, "d", "Fixed-KD low-rank classification", x=-0.105)
    axd.axis("off")
    d_axes = [axd.inset_axes([0.00, 0.19, 0.47, 0.71]), axd.inset_axes([0.54, 0.19, 0.46, 0.71])]
    for axis, metric in zip(d_axes, ("AA", "AF")):
        for index, rank in enumerate((8, 16, 32)):
            selected = sorted([item for item in adapter_rows if int(item["rank"]) == rank],
                              key=lambda item: int(item["seed"]))
            if metric == "AA":
                control = np.asarray([float(item["control_aa_percent"]) for item in selected])
                treatment = np.asarray([float(item["treatment_aa_percent"]) for item in selected])
            else:
                control = np.asarray([float(item["control_af_percent"]) for item in selected])
                treatment = np.asarray([float(item["treatment_af_percent"]) for item in selected])
            jitter = np.linspace(-0.055, 0.055, len(selected))
            for base, target, shift in zip(control, treatment, jitter):
                axis.plot([index - 0.12 + shift, index + 0.12 + shift], [base, target],
                          color=LIGHT, lw=0.45, zorder=1)
                axis.scatter(index - 0.12 + shift, base, s=6.3, facecolor="white",
                             edgecolor=CONTROL, linewidth=0.40, zorder=2)
                axis.scatter(index + 0.12 + shift, target, s=6.3, facecolor="white",
                             edgecolor=SSR, linewidth=0.45, zorder=2)
            for xpos, values, color, filled in (
                (index - 0.12, control, CONTROL, False),
                (index + 0.12, treatment, SSR, True),
            ):
                mean, low, high = t_ci(values)
                axis.errorbar(xpos, mean, yerr=[[mean - low], [high - mean]], fmt="o", ms=4.4,
                              mfc=color if filled else "white", mec=INK, mew=0.45,
                              ecolor=color, capsize=1.3, lw=0.75, zorder=4)
            favorable = treatment - control if metric == "AA" else control - treatment
            delta, low, high = t_ci(favorable)
            axis.text(index, 0.985,
                      f"{delta:+.2f}\n[{low:+.2f}, {high:+.2f}]\n{p_text(favorable)}",
                      transform=axis.get_xaxis_transform(), ha="center", va="top",
                      fontsize=3.75, color=SSR, fontweight="bold")
        axis.set_xticks(range(3), ["8", "16", "32"])
        axis.set_xlabel("adapter rank")
        axis.set_ylabel(f"{metric} (%)")
        if metric == "AA":
            axis.set_ylim(77.7, 82.0)
        else:
            axis.set_ylim(1.3, 4.5)
        clean(axis)
    d_axes[0].scatter([], [], s=18, facecolor="white", edgecolor=CONTROL, label="KD + LowRank")
    d_axes[0].scatter([], [], s=18, facecolor=SSR, edgecolor=INK, linewidth=0.35,
                      label="+ SSR")
    d_axes[0].legend(frameon=False, loc="lower left", ncol=2,
                     handletextpad=0.20, columnspacing=0.50)
    source_line(axd, "CUB-200-2011 | frozen ResNet-18 + GELU bottleneck | 40 epochs/task | n=10/rank",
                y=-0.17)

    # e | matched geometry controls and a disjoint frozen-strength confirmation.
    axe = fig.add_subplot(gs[2, 6:])
    panel_label(axe, "e", "Matched controls and frozen SSR strength", x=-0.11)
    axe.axis("off")
    control_ax = axe.inset_axes([0.00, 0.19, 0.47, 0.66])
    strength_ax = axe.inset_axes([0.55, 0.19, 0.45, 0.66])
    method_order = ["kd_orthogonal", "kd_protodecor", "kd_spectral", "kd_ssr"]
    labels = {
        "kd_orthogonal": "KD + orthogonal",
        "kd_protodecor": "KD + prototype decor.",
        "kd_spectral": "KD + spectral",
        "kd_ssr": "KD + SSR",
    }
    colors = {method: CONTROL for method in method_order}
    colors["kd_ssr"] = SSR
    markers = {
        "kd_orthogonal": "o",
        "kd_protodecor": "s",
        "kd_spectral": "^",
        "kd_ssr": "D",
    }
    indexed = {
        (item["method"], int(item["seed"])): item
        for item in geometry_rows
    }
    seeds = sorted({int(item["seed"]) for item in geometry_rows})
    for method in method_order:
        aa = np.asarray([
            float(indexed[(method, seed)]["avg_accuracy"])
            - float(indexed[("kd", seed)]["avg_accuracy"])
            for seed in seeds
        ])
        af = np.asarray([
            float(indexed[("kd", seed)]["avg_forgetting"])
            - float(indexed[(method, seed)]["avg_forgetting"])
            for seed in seeds
        ])
        is_ssr = method == "kd_ssr"
        control_ax.scatter(aa, af, s=6.5 if is_ssr else 5.2, facecolor="white",
                           edgecolor=colors[method], linewidth=0.40,
                           alpha=0.34 if is_ssr else 0.20, zorder=1)
        aa_mean, aa_low, aa_high = t_ci(aa)
        af_mean, af_low, af_high = t_ci(af)
        control_ax.errorbar(
            aa_mean, af_mean,
            xerr=[[aa_mean-aa_low], [aa_high-aa_mean]],
            yerr=[[af_mean-af_low], [af_high-af_mean]],
            fmt=markers[method], ms=5.4 if is_ssr else 4.1,
            mfc=colors[method] if is_ssr else "white", mec=INK, mew=0.45,
            ecolor=colors[method], capsize=1.35,
            lw=0.95 if is_ssr else 0.62, zorder=3, label=labels[method]
        )
        if is_ssr:
            control_ax.text(0.98, 0.06,
                            "SSR: " + p_text(aa) + " AA; " + p_text(af) + " AF",
                            transform=control_ax.transAxes, ha="right", va="bottom",
                            fontsize=3.7, color=SSR, fontweight="bold")
    control_ax.axvline(0, color=LIGHT, lw=0.50)
    control_ax.axhline(0, color=LIGHT, lw=0.50)
    control_ax.set_xlim(-1.10, 2.15)
    control_ax.set_ylim(-1.00, 2.25)
    control_ax.set_xlabel("AA gain (pp)", labelpad=1.0)
    control_ax.set_ylabel("AF reduction (pp)", labelpad=1.0)
    control_ax.set_title("CUB matched controls", fontsize=5.15, pad=1.8)
    control_ax.legend(frameon=False, loc="upper left", ncol=2, handletextpad=0.15,
                      columnspacing=0.38, fontsize=3.55, borderaxespad=0.1)
    clean(control_ax)

    # Only the three prespecified low-strength SSR settings are emphasized here.
    # The complete nine-setting sweep, including over-regularization, is drawn in
    # Extended Data and retained in Source Data.
    strength_specs = [
        ("scale_m0p0625", r"$1/16\times$", "o", "#78B8BE"),
        ("scale_m0p125", r"$1/8\times$", "s", "#3D9BA5"),
        ("scale_m0p25", r"$1/4\times$", "D", SSR),
    ]
    ssr_candidates = robustness_summary["families"]["kd_ssr"]["candidates"]
    label_offsets = {
        "scale_m0p0625": (0.02, -0.15),
        "scale_m0p125": (-0.03, 0.13),
        "scale_m0p25": (0.03, -0.16),
    }
    for key, label, marker, color in strength_specs:
        seed_rows = [
            item for item in robustness_rows
            if item["method"] == "kd_ssr" and item["candidate_id"] == key
        ]
        aa = np.asarray([float(item["aa_gain_vs_kd_pp"]) for item in seed_rows])
        af = np.asarray([float(item["af_reduction_vs_kd_pp"]) for item in seed_rows])
        strength_ax.scatter(aa, af, s=5.6, facecolor="white", edgecolor=color,
                            linewidth=0.38, alpha=0.28, zorder=1)
        record = ssr_candidates[key]
        aa_mean = float(record["aa_gain_pp"]["mean"])
        aa_low, aa_high = [float(value) for value in record["aa_gain_pp"]["ci95"]]
        af_mean = float(record["af_reduction_pp"]["mean"])
        af_low, af_high = [float(value) for value in record["af_reduction_pp"]["ci95"]]
        strength_ax.errorbar(
            aa_mean, af_mean,
            xerr=[[aa_mean-aa_low], [aa_high-aa_mean]],
            yerr=[[af_mean-af_low], [af_high-af_mean]],
            fmt=marker, ms=4.8, mfc=color, mec=INK, mew=0.42,
            ecolor=color, capsize=1.3, lw=0.82, zorder=3
        )
        dx, dy = label_offsets[key]
        strength_ax.text(aa_mean + dx, af_mean + dy, label, fontsize=4.0,
                         color=color, fontweight="bold", ha="center", va="center")
    strength_ax.axvline(0, color=LIGHT, lw=0.50)
    strength_ax.axhline(0, color=LIGHT, lw=0.50)
    strength_ax.set_xlim(-0.05, 1.80)
    strength_ax.set_ylim(-0.05, 1.70)
    strength_ax.set_xlabel("AA gain (pp)", labelpad=1.0)
    strength_ax.set_ylabel("AF reduction (pp)", labelpad=1.0)
    strength_ax.set_title("Split-CIFAR-100 low-strength sweep", fontsize=5.15, pad=1.8)
    strength_ax.text(0.98, 0.06, "all three paired CIs > 0",
                     transform=strength_ax.transAxes, ha="right", fontsize=3.85,
                     color=SSR, fontweight="bold")
    clean(strength_ax)
    source_line(
        axe,
        "CUB: rank 32, common KD, n=10 | Split-CIFAR-100: rank 4, 20 tasks, common KD, n=10",
        y=-0.17,
    )

    # f | true scalar maps from reconstructed low-rank checkpoints.
    axf = fig.add_subplot(gs[3, :])
    panel_label(axf, "f", "Low-rank response-map illustration", x=-0.035)
    axf.axis("off")
    manifest = json.loads((RESPONSE / "manifest.json").read_text(encoding="utf-8"))
    input_image = np.asarray(Image.open(RESPONSE / "input.png").convert("RGB"))
    gt_mask = np.asarray(Image.open(RESPONSE / "gt_mask.png").convert("L")) > 127
    gt_overlay = input_image.astype(float)
    gt_color = np.asarray([38, 151, 166], dtype=float)
    gt_overlay[gt_mask] = 0.42 * gt_overlay[gt_mask] + 0.58 * gt_color
    images = [
        input_image,
        np.clip(gt_overlay, 0, 255).astype(np.uint8),
        np.load(RESPONSE / "kd_response.npy"),
        np.load(RESPONSE / "kd_ssr_response.npy"),
        np.load(RESPONSE / "signed_difference.npy"),
    ]
    titles = ["Input", "GT mask", "KD + LowRank", "KD + LowRank + SSR", "Signed difference"]
    for index, (image, title) in enumerate(zip(images, titles)):
        tile = axf.inset_axes([0.004 + 0.199 * index, 0.00, 0.190, 0.88])
        tile.imshow(input_image)
        if index in (0, 1):
            tile.images[0].set_data(image)
        elif index in (2, 3):
            tile.imshow(
                image,
                cmap="coolwarm",
                vmin=-float(manifest["response_shared_abs_bound"]),
                vmax=float(manifest["response_shared_abs_bound"]),
                alpha=0.62,
                interpolation="bilinear",
            )
        else:
            tile.imshow(
                image,
                cmap="PiYG",
                vmin=-float(manifest["difference_zero_center_abs_bound"]),
                vmax=float(manifest["difference_zero_center_abs_bound"]),
                alpha=0.70,
                interpolation="bilinear",
            )
        tile.set_title(title, fontsize=5.15, pad=1.2, fontweight="bold")
        tile.axis("off")
    axf.text(0.004, -0.03,
             f"CUB-200-2011 | seed {manifest['seed']} | rank {manifest['rank']} GELU adapter | "
             f"image {manifest['image_id']}",
             transform=axf.transAxes, ha="left", va="top", fontsize=4.25, color=GRAY)
    axf.text(0.995, -0.03, "target-minus-hard-confuser margin | outcome-aware illustration",
             transform=axf.transAxes, ha="right", va="top", fontsize=4.25, color=GRAY)

    fig.subplots_adjust(left=0.102, right=0.988, top=0.972, bottom=0.040)
    save(fig, "Fig6_transfer_audited_20260905")


def build_strength_robustness_supp() -> None:
    """Draw every frozen rank-4 SSR strength without filtering failed settings."""
    summary = json.loads(
        (ROBUST_SOURCE / "Fig6e_cifar100_rank4_robustness_summary.json").read_text(
            encoding="utf-8"
        )
    )
    candidates = sorted(
        summary["families"]["kd_ssr"]["candidates"].values(),
        key=lambda item: float(item["scale_multiplier"]),
    )
    scales = np.asarray([float(item["scale_multiplier"]) for item in candidates])
    log_scales = np.log2(scales)
    aa = np.asarray([float(item["aa_gain_pp"]["mean"]) for item in candidates])
    aa_low = np.asarray([float(item["aa_gain_pp"]["ci95"][0]) for item in candidates])
    aa_high = np.asarray([float(item["aa_gain_pp"]["ci95"][1]) for item in candidates])
    af = np.asarray([float(item["af_reduction_pp"]["mean"]) for item in candidates])
    af_low = np.asarray([float(item["af_reduction_pp"]["ci95"][0]) for item in candidates])
    af_high = np.asarray([float(item["af_reduction_pp"]["ci95"][1]) for item in candidates])
    new = np.asarray([float(item["final_new_accuracy_gain_pp"]["mean"]) for item in candidates])
    new_low = np.asarray([float(item["final_new_accuracy_gain_pp"]["ci95"][0]) for item in candidates])
    new_high = np.asarray([float(item["final_new_accuracy_gain_pp"]["ci95"][1]) for item in candidates])
    eligible = np.asarray([bool(item["acquisition_eligible"]) for item in candidates])

    fig, (axa, axb) = plt.subplots(
        1, 2, figsize=(183 / 25.4, 70 / 25.4), facecolor="white",
        gridspec_kw={"width_ratios": [1.08, 1.32], "wspace": 0.36},
    )
    panel_label(axa, "a", "Complete frozen SSR strength sweep", x=-0.14)
    color_values = plt.get_cmap("viridis")(np.linspace(0.15, 0.85, len(candidates)))
    axa.plot(aa, af, color=LIGHT, lw=0.85, zorder=0)
    for index, (item, color) in enumerate(zip(candidates, color_values)):
        axa.errorbar(
            aa[index], af[index],
            xerr=[[aa[index]-aa_low[index]], [aa_high[index]-aa[index]]],
            yerr=[[af[index]-af_low[index]], [af_high[index]-af[index]]],
            fmt="o", ms=4.8, mfc=color if eligible[index] else "white",
            mec=color, mew=0.70, ecolor=color, capsize=1.4, lw=0.75, zorder=3,
        )
        if index in (0, 4, 8):
            offsets = {0: (4, 4), 4: (5, 5), 8: (5, 5)}
            axa.annotate(
                f"{item['scale_multiplier']:g}x", (aa[index], af[index]),
                xytext=offsets[index], textcoords="offset points",
                fontsize=4.0, color=color,
            )
    axa.axvline(0, color=LIGHT, lw=0.55)
    axa.axhline(0, color=LIGHT, lw=0.55)
    axa.set_xlabel("AA gain from paired KD (pp)")
    axa.set_ylabel("AF reduction from paired KD (pp)")
    axa.set_xlim(-1.75, 1.95)
    axa.set_ylim(0.25, 1.95)
    axa.text(0.03, 0.05, "filled: mean acquisition-eligible",
             transform=axa.transAxes, fontsize=4.3, color=GRAY)
    clean(axa)

    panel_label(axb, "b", "Endpoint and acquisition effects", x=-0.11)
    for values, low, high, color, marker, label in (
        (aa, aa_low, aa_high, SSR, "o", "AA gain"),
        (af, af_low, af_high, PURPLE, "s", "AF reduction"),
        (new, new_low, new_high, ORANGE, "D", "final-new gain"),
    ):
        axb.errorbar(
            log_scales, values, yerr=[values-low, high-values],
            color=color, marker=marker, ms=3.4, lw=0.85, capsize=1.4,
            label=label,
        )
    axb.axhspan(-0.5, 2.1, color="#EEF3F1", alpha=0.65, zorder=-2)
    axb.axhline(0, color=INK, lw=0.55)
    axb.axhline(-0.5, color=GRAY, lw=0.55, ls="--")
    axb.set_xticks(log_scales, ["1/16", "1/8", "1/4", "1/2", "1", "2", "4", "8", "16"])
    axb.set_xlabel("relative strength around frozen center")
    axb.set_ylabel("change from paired KD (pp)")
    axb.set_ylim(-5.25, 2.15)
    axb.legend(frameon=False, loc="lower left", ncol=3, handletextpad=0.25,
               columnspacing=0.60)
    axb.text(0.99, 0.27, "mean acquisition floor: -0.5 pp",
             transform=axb.transAxes, ha="right", fontsize=4.15, color=GRAY)
    clean(axb)
    fig.text(
        0.50, 0.01,
        "Split-CIFAR-100 | frozen ResNet-18 features + rank-4 GELU adapter | 20 tasks x 5 classes | common KD | n=10 paired",
        ha="center", va="bottom", fontsize=4.55, color=GRAY,
    )
    fig.subplots_adjust(left=0.09, right=0.985, top=0.86, bottom=0.22)
    save(fig, "fig_supp_lowrank_cifar100_strength_20260907")


def build_fullrank_regularizer_supp() -> None:
    """Historical non-low-rank fixed-KD context with AA and AF shown together."""
    source = ROBUST_SOURCE / "FigS_fullrank_fixedkd_regularizers.csv"
    with source.open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    datasets = ["Split-CIFAR-100", "TinyImageNet"]
    methods = ["EWC", "MAS", "SI", "SSR"]
    colors = {"EWC": PURPLE, "MAS": "#D1A23F", "SI": ORANGE, "SSR": SSR}
    markers = {"EWC": "o", "MAS": "s", "SI": "^", "SSR": "D"}

    fig, axes = plt.subplots(
        2, 2, figsize=(183 / 25.4, 96 / 25.4), facecolor="white",
        gridspec_kw={"width_ratios": [1.02, 1.15], "hspace": 0.62, "wspace": 0.46},
    )
    panel_ids = [("a", "b"), ("c", "d")]
    limits = {
        "Split-CIFAR-100": ((58.0, 73.3), (0.0, 14.2)),
        "TinyImageNet": ((41.0, 55.0), (0.0, 7.0)),
    }
    label_offsets = {
        ("Split-CIFAR-100", "EWC"): (4, 5),
        ("Split-CIFAR-100", "MAS"): (4, -8),
        ("Split-CIFAR-100", "SI"): (4, 4),
        ("Split-CIFAR-100", "SSR"): (-4, 5),
        ("TinyImageNet", "EWC"): (4, 5),
        ("TinyImageNet", "MAS"): (4, -8),
        ("TinyImageNet", "SI"): (4, 4),
        ("TinyImageNet", "SSR"): (-4, 5),
    }
    for row_index, dataset in enumerate(datasets):
        selected = {item["method"]: item for item in records if item["dataset"] == dataset}
        trade_ax, advantage_ax = axes[row_index]
        panel_label(trade_ax, panel_ids[row_index][0], f"{dataset}: absolute endpoints", x=-0.13)
        for method in methods:
            item = selected[method]
            aa = float(item["aa_mean_percent"])
            af = float(item["af_mean_percent"])
            aa_sd = float(item["aa_sd_percent"])
            af_sd = float(item["af_sd_percent"])
            is_ssr = method == "SSR"
            trade_ax.errorbar(
                aa, af, xerr=aa_sd, yerr=af_sd, fmt=markers[method],
                ms=5.6 if is_ssr else 4.7,
                mfc=colors[method] if is_ssr else "white",
                mec=INK, mew=0.48, ecolor=colors[method], capsize=1.5,
                lw=0.82, zorder=3,
            )
            dx, dy = label_offsets[(dataset, method)]
            trade_ax.annotate(
                method, (aa, af), xytext=(dx, dy), textcoords="offset points",
                ha="right" if dx < 0 else "left", fontsize=4.45,
                color=colors[method], fontweight="bold" if is_ssr else "normal",
            )
        trade_ax.set_xlim(*limits[dataset][0])
        trade_ax.set_ylim(*limits[dataset][1])
        trade_ax.set_xlabel("average accuracy, AA (%)")
        trade_ax.set_ylabel("average forgetting, AF (%)")
        trade_ax.text(0.98, 0.94, "higher AA  |  lower AF",
                      transform=trade_ax.transAxes, ha="right", fontsize=4.2,
                      color=GRAY)
        clean(trade_ax)

        panel_label(advantage_ax, panel_ids[row_index][1], "Paired AA advantage of SSR", x=-0.13)
        comparison_methods = ["EWC", "MAS", "SI"]
        y_positions = np.arange(len(comparison_methods))[::-1]
        for ypos, method in zip(y_positions, comparison_methods):
            item = selected[method]
            mean = float(item["ssr_aa_advantage_pp"])
            low = float(item["ssr_aa_advantage_ci_low_pp"])
            high = float(item["ssr_aa_advantage_ci_high_pp"])
            advantage_ax.errorbar(
                mean, ypos, xerr=[[mean-low], [high-mean]], fmt=markers[method],
                ms=4.8, mfc=colors[method], mec=INK, mew=0.42,
                ecolor=colors[method], capsize=1.6, lw=0.85, zorder=3,
            )
            advantage_ax.text(high + 0.25, ypos, f"{mean:+.2f} [{low:.2f}, {high:.2f}]",
                              va="center", fontsize=4.35, color=INK,
                              fontweight="bold")
        advantage_ax.axvline(0, color=INK, lw=0.55)
        advantage_ax.set_yticks(y_positions, [f"SSR - {method}" for method in comparison_methods])
        advantage_ax.set_ylim(-0.55, 2.55)
        max_high = max(float(selected[method]["ssr_aa_advantage_ci_high_pp"])
                       for method in comparison_methods)
        advantage_ax.set_xlim(-0.45, max_high + 3.0)
        advantage_ax.set_xlabel("paired AA difference (pp)")
        advantage_ax.text(0.98, 0.05, "all paired 95% CIs > 0",
                          transform=advantage_ax.transAxes, ha="right",
                          fontsize=4.25, color=SSR, fontweight="bold")
        clean(advantage_ax)

    fig.text(
        0.50, 0.015,
        "common fixed-KD scaffold | non-low-rank ResNet-18 protocol | n=5 paired seeds per dataset",
        ha="center", va="bottom", fontsize=4.55, color=GRAY,
    )
    fig.subplots_adjust(left=0.095, right=0.985, top=0.91, bottom=0.14)
    save(fig, "fig_supp_kd_regularizer_comparison")


def build_moved_mask_panel() -> None:
    records = [json.loads(line) for line in
               (ROOT / "source_data" / "raw" / "structured_output" / "runs.jsonl").read_text(
                   encoding="utf-8").splitlines() if line.strip()]
    by_method = {
        method: sorted([item for item in records if item["method"] == method],
                       key=lambda item: item["seed"])
        for method in ("baseline", "biocs_kd")
    }
    fig, axes = plt.subplots(1, 3, figsize=(150 / 25.4, 52 / 25.4), facecolor="white")
    specs = [
        ("mean_iou", "mIoU (%)", True, SSR),
        ("mean_dice", "Dice (%)", True, PURPLE),
        ("avg_forgetting_iou", "IoU forgetting (%)", False, ORANGE),
    ]
    for axis, (key, label, higher, color) in zip(axes, specs):
        control = np.asarray([item[key] for item in by_method["baseline"]], dtype=float)
        treatment = np.asarray([item[key] for item in by_method["biocs_kd"]], dtype=float)
        for index, (base, target) in enumerate(zip(control, treatment)):
            shift = (index - 1) * 0.025
            axis.plot([shift, 1 + shift], [base, target], color=LIGHT, lw=0.8)
            axis.scatter(shift, base, s=18, facecolor="white", edgecolor=CONTROL, linewidth=0.55)
            axis.scatter(1 + shift, target, s=18, facecolor=color, edgecolor=INK, linewidth=0.35)
        axis.plot([0, 1], [control.mean(), treatment.mean()], color=color, lw=1.3)
        axis.scatter(0, control.mean(), s=30, facecolor="white", edgecolor=INK, linewidth=0.65)
        axis.scatter(1, treatment.mean(), s=30, facecolor=color, edgecolor=INK, linewidth=0.40)
        favorable = treatment - control if higher else control - treatment
        axis.set_title(f"favorable {favorable.mean():+.2f} pp", fontsize=6.0, color=color,
                       fontweight="bold", pad=3)
        axis.set_xticks([0, 1], ["Unreg.", "complete\nKD + SSR"])
        axis.set_ylabel(label)
        span = max(control.max(), treatment.max()) - min(control.min(), treatment.min())
        pad = max(0.25, 0.18 * span)
        axis.set_ylim(min(control.min(), treatment.min()) - pad,
                      max(control.max(), treatment.max()) + pad)
        clean(axis)
    fig.suptitle("Complete-objective CUB mask transfer", x=0.06, y=0.995,
                 ha="left", fontsize=7.4, fontweight="bold")
    for axis in axes:
        axis.set_xlim(-0.08, 1.12)
        axis.tick_params(axis="x", labelsize=5.3)
    fig.subplots_adjust(left=0.09, right=0.965, top=0.82, bottom=0.25, wspace=0.52)
    for suffix, kwargs in (("pdf", {"dpi": 450}), ("svg", {}), ("png", {"dpi": 300})):
        fig.savefig(FIGURES / f"fig_supp_cub_mask_complete_objective_20260905.{suffix}",
                    facecolor="white", **kwargs)
    plt.close(fig)


if __name__ == "__main__":
    build_main()
    build_moved_mask_panel()
    build_strength_robustness_supp()
    build_fullrank_regularizer_supp()
    print("Built audited Figure 6 and its mask, strength, and full-rank context panels")
