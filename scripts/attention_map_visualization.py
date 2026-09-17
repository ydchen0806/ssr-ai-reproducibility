"""Attention / class-activation visualizations for SSR+ classifiers.

This script renders compound figures that combine:

* CUB200 test images with class-activation maps produced by projecting
  pre-pool spatial features through each method's classifier prototype.
* Method-vs-method attention deltas that highlight where SSR+KD
  re-allocates attention compared with the baseline.
* Pairwise prototype attention overlap for confusing fine-grained pairs
  to expose where representational interference originates.

The cached prototype matrices come from
``results/deep_representation_case_study_20260430/weights_and_snapshots.npz``.
The backbone is ImageNet-pretrained ResNet-18 (matching the cached feature
extractor). The script does not retrain anything; it forward-passes each
selected CUB test image through the backbone, harvests the 7x7 spatial map
from ``layer4``, and computes a CAM per class prototype as ``relu(<feat,
proto>)`` followed by min-max normalization.

Outputs::

    ICLR_paper/figures/fig_compound_attention_maps.{pdf,png}
    ICLR_paper/figures/fig_compound_attention_overlap.{pdf,png}

Reproduce::

    cd ssr-ai-reproducibility
    CUDA_VISIBLE_DEVICES=0 python scripts/attention_map_visualization.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18


ROOT = Path(__file__).resolve().parents[1]
CUB_ROOT = ROOT / "data" / "cub200" / "CUB_200_2011"
SEG_ROOT = ROOT / "data" / "cub200" / "segmentations"
SNAPSHOT_FILE = (
    ROOT / "results" / "deep_representation_case_study_20260430" / "weights_and_snapshots.npz"
)
SUMMARY_FILE = (
    ROOT / "results" / "deep_representation_case_study_20260430" / "summary.json"
)
OUT_DIR = ROOT / "results" / "comprehensive_visualization_20260506"
FIG_DIR = Path(os.environ.get("SSR_FIGURE_DIR", ROOT / "figures"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


METHODS = ["baseline", "biocs", "biocs_kd"]
LABELS = {"baseline": "Baseline", "biocs": "SSR", "biocs_kd": "SSR+KD"}
COLORS = {
    "baseline": "#8A9296",
    "biocs": "#5C9275",
    "biocs_kd": "#2F6F73",
    "accent": "#9B4F48",
    "ink": "#2D3436",
    "panel": "#F8F6F0",
    "grid": "#E8E6DF",
}


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 10.5,
        "font.weight": "bold",
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
        "axes.linewidth": 1.3,
        "axes.edgecolor": COLORS["ink"],
        "figure.dpi": 220,
        "savefig.dpi": 320,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.10,
    }
)


def read_class_names_with_id() -> list[str]:
    names = []
    with (CUB_ROOT / "classes.txt").open("r", encoding="utf-8") as f:
        for line in f:
            _, name = line.strip().split(" ", 1)
            names.append(name)
    return names


def read_metadata() -> list[dict]:
    images, labels, splits = {}, {}, {}
    with (CUB_ROOT / "images.txt").open("r", encoding="utf-8") as f:
        for line in f:
            idx, rel = line.strip().split()
            images[int(idx)] = rel
    with (CUB_ROOT / "image_class_labels.txt").open("r", encoding="utf-8") as f:
        for line in f:
            idx, cls = line.strip().split()
            labels[int(idx)] = int(cls) - 1
    with (CUB_ROOT / "train_test_split.txt").open("r", encoding="utf-8") as f:
        for line in f:
            idx, is_train = line.strip().split()
            splits[int(idx)] = int(is_train)
    rows = []
    for idx in sorted(images):
        rows.append(
            {
                "id": idx,
                "relpath": images[idx],
                "label": labels[idx],
                "is_train": bool(splits[idx]),
            }
        )
    return rows


def first_test_image_for_class(rows: list[dict], cls: int) -> Path:
    for row in rows:
        if (not row["is_train"]) and row["label"] == cls:
            return CUB_ROOT / "images" / row["relpath"]
    raise FileNotFoundError(f"no test image for class {cls}")


def load_prototypes() -> dict[str, np.ndarray]:
    payload = np.load(SNAPSHOT_FILE)
    return {m: payload[f"{m}_weight"] for m in METHODS}


def build_backbone(device: torch.device) -> tuple[nn.Module, transforms.Compose]:
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)
    model.fc = nn.Identity()
    model.avgpool = nn.Identity()
    feat_extractor = nn.Sequential(
        model.conv1, model.bn1, model.relu, model.maxpool,
        model.layer1, model.layer2, model.layer3, model.layer4,
    )
    feat_extractor.eval().to(device)
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return feat_extractor, preprocess


@torch.no_grad()
def spatial_features(model: nn.Module, image: Image.Image, preprocess, device: torch.device) -> np.ndarray:
    x = preprocess(image).unsqueeze(0).to(device)
    feats = model(x).detach().cpu().numpy()[0]
    return feats


def class_activation_map(feats: np.ndarray, prototype: np.ndarray) -> np.ndarray:
    """Compute a normalized class activation map.

    feats: (C, H, W) float array of pre-pool features.
    prototype: (C,) classifier prototype vector.

    Returns a (H, W) heatmap normalized to [0, 1]. Uses a signed projection
    rescaled to [0, 1] so that informative attention is always visible even
    when the dot product is mostly negative (which happens for some
    fine-grained CUB prototypes against ImageNet-pretrained spatial features).
    """

    proto = prototype.astype(np.float32)
    proto = proto / (np.linalg.norm(proto) + 1e-8)
    norm_feats = feats / (np.linalg.norm(feats, axis=0, keepdims=True) + 1e-8)
    cam = np.einsum("c,chw->hw", proto, norm_feats)
    finite = cam[np.isfinite(cam)]
    if finite.size == 0:
        return np.zeros_like(cam)
    lo = float(np.percentile(finite, 5))
    hi = float(np.percentile(finite, 95))
    if hi - lo < 1e-9:
        return np.zeros_like(cam)
    cam = np.clip((cam - lo) / (hi - lo), 0.0, 1.0)
    return cam


def overlay_cam(image: Image.Image, cam: np.ndarray, size: int = 224) -> np.ndarray:
    img = image.convert("RGB").resize((size, size), Image.BICUBIC)
    cam_img = Image.fromarray((cam * 255).astype(np.uint8))
    cam_img = cam_img.resize((size, size), Image.BICUBIC)
    cam_arr = np.asarray(cam_img).astype(np.float32) / 255.0
    img_arr = np.asarray(img).astype(np.float32) / 255.0
    cmap = plt.cm.inferno(cam_arr)[..., :3]
    return np.clip(0.42 * cmap + 0.58 * img_arr, 0, 1)


def select_visual_classes(summary: dict, fallback: list[int]) -> list[int]:
    seen: list[int] = []
    for pair in summary.get("selected_case_pairs", []):
        for key in ("class_i", "class_j"):
            cls = pair[key]
            if cls not in seen:
                seen.append(cls)
    while len(seen) < 6 and fallback:
        cls = fallback.pop(0)
        if cls not in seen:
            seen.append(cls)
    return seen[:6]


def plot_attention_grid(
    feats_per_image: list[np.ndarray],
    images: list[Image.Image],
    class_names: list[str],
    selected: list[int],
    prototypes: dict[str, np.ndarray],
) -> None:
    n_rows = len(selected)
    n_cols = 1 + len(METHODS) + 1  # original + 3 methods + delta
    fig = plt.figure(figsize=(2.4 * n_cols, 2.45 * n_rows + 0.6))
    gs = fig.add_gridspec(n_rows, n_cols, hspace=0.20, wspace=0.10)

    for row, cls in enumerate(selected):
        feats = feats_per_image[row]
        image = images[row]
        cams: list[np.ndarray] = []
        for col, method in enumerate(METHODS):
            cam = class_activation_map(feats, prototypes[method][cls])
            cams.append(cam)

        ax = fig.add_subplot(gs[row, 0])
        ax.imshow(image.resize((224, 224), Image.BICUBIC))
        if row == 0:
            ax.set_title("Test image", fontsize=11.5)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_color(COLORS["ink"])
        short_name = class_names[cls].split(".", 1)[-1].replace("_", " ")
        ax.set_ylabel(short_name, fontsize=10, rotation=0, ha="right", va="center", labelpad=10)

        for col_idx, (method, cam) in enumerate(zip(METHODS, cams)):
            ax = fig.add_subplot(gs[row, 1 + col_idx])
            ax.imshow(overlay_cam(image, cam))
            if row == 0:
                ax.set_title(LABELS[method], fontsize=11.5, color=COLORS[method])
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(1.2)
                spine.set_color(COLORS[method])

        delta = cams[METHODS.index("biocs_kd")] - cams[METHODS.index("baseline")]
        ax = fig.add_subplot(gs[row, n_cols - 1])
        ax.imshow(image.resize((224, 224), Image.BICUBIC))
        upsample = np.asarray(
            Image.fromarray(((delta + 1) * 127.5).astype(np.uint8)).resize((224, 224), Image.BICUBIC)
        ).astype(np.float32) / 127.5 - 1.0
        rgba = plt.cm.coolwarm((upsample + 1.0) / 2.0)
        rgba[..., 3] = 0.45 * np.abs(upsample)
        ax.imshow(rgba)
        if row == 0:
            ax.set_title("SSR+KD − Baseline", fontsize=11.5, color=COLORS["accent"])
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_color(COLORS["accent"])

    fig.suptitle(
        "SSR+KD reorganizes spatial attention towards diagnostic CUB200 parts",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"fig_compound_attention_maps.{ext}")
    plt.close(fig)


def plot_attention_overlap(
    feats_per_image: list[np.ndarray],
    images: list[Image.Image],
    class_names: list[str],
    selected: list[int],
    prototypes: dict[str, np.ndarray],
    summary: dict,
) -> None:
    pairs = summary.get("selected_case_pairs", [])[:3]
    if not pairs:
        return
    n_rows = len(pairs)
    fig = plt.figure(figsize=(13.5, 2.8 * n_rows + 0.7))
    gs = fig.add_gridspec(
        n_rows,
        7,
        width_ratios=[0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 1.05],
        hspace=0.32,
        wspace=0.10,
    )

    cls_to_idx = {cls: idx for idx, cls in enumerate(selected)}

    for row, pair in enumerate(pairs):
        cls_i, cls_j = pair["class_i"], pair["class_j"]
        if cls_i not in cls_to_idx or cls_j not in cls_to_idx:
            continue
        feats_i = feats_per_image[cls_to_idx[cls_i]]
        feats_j = feats_per_image[cls_to_idx[cls_j]]
        img_i = images[cls_to_idx[cls_i]]
        img_j = images[cls_to_idx[cls_j]]

        for class_id, feats, img, base_col in (
            (cls_i, feats_i, img_i, 0),
            (cls_j, feats_j, img_j, 3),
        ):
            ax = fig.add_subplot(gs[row, base_col])
            ax.imshow(img.resize((224, 224), Image.BICUBIC))
            short = class_names[class_id].split(".", 1)[-1].replace("_", " ")
            if row == 0 and base_col == 0:
                ax.set_title("Test image\n(class A)", fontsize=10)
            elif row == 0 and base_col == 3:
                ax.set_title("Test image\n(class B)", fontsize=10)
            ax.set_xlabel(short, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(1.0)
                spine.set_color(COLORS["ink"])

        for offset, method in enumerate(["baseline", "biocs_kd"]):
            ax_i = fig.add_subplot(gs[row, 1 + offset])
            ax_j = fig.add_subplot(gs[row, 4 + offset])
            cam_i = class_activation_map(feats_i, prototypes[method][cls_i])
            cam_j_other = class_activation_map(feats_i, prototypes[method][cls_j])
            ax_i.imshow(overlay_cam(img_i, cam_i))
            if row == 0:
                ax_i.set_title(f"{LABELS[method]}\nA-CAM on A", fontsize=9.5, color=COLORS[method])
            ax_i.set_xticks([])
            ax_i.set_yticks([])
            for spine in ax_i.spines.values():
                spine.set_linewidth(1.0)
                spine.set_color(COLORS[method])

            ax_j.imshow(overlay_cam(img_i, cam_j_other))
            if row == 0:
                ax_j.set_title(f"{LABELS[method]}\nB-CAM on A", fontsize=9.5, color=COLORS[method])
            ax_j.set_xticks([])
            ax_j.set_yticks([])
            for spine in ax_j.spines.values():
                spine.set_linewidth(1.0)
                spine.set_color(COLORS[method])

        ax = fig.add_subplot(gs[row, 6])
        overlap_baseline = []
        overlap_biocs = []
        for feats in (feats_i, feats_j):
            cam_a_base = class_activation_map(feats, prototypes["baseline"][cls_i])
            cam_b_base = class_activation_map(feats, prototypes["baseline"][cls_j])
            cam_a_plus = class_activation_map(feats, prototypes["biocs_kd"][cls_i])
            cam_b_plus = class_activation_map(feats, prototypes["biocs_kd"][cls_j])
            num = (cam_a_base * cam_b_base).sum()
            den = np.sqrt((cam_a_base**2).sum() * (cam_b_base**2).sum() + 1e-9)
            overlap_baseline.append(float(num / den))
            num = (cam_a_plus * cam_b_plus).sum()
            den = np.sqrt((cam_a_plus**2).sum() * (cam_b_plus**2).sum() + 1e-9)
            overlap_biocs.append(float(num / den))
        bar_x = np.arange(2)
        ax.bar(bar_x - 0.2, overlap_baseline, width=0.36, color=COLORS["baseline"], edgecolor=COLORS["ink"], label="Baseline")
        ax.bar(bar_x + 0.2, overlap_biocs, width=0.36, color=COLORS["biocs_kd"], edgecolor=COLORS["ink"], label="SSR+KD")
        ax.set_xticks(bar_x)
        ax.set_xticklabels(["on A", "on B"])
        ax.set_ylabel("CAM overlap")
        if row == 0:
            ax.set_title("CAM overlap A↔B", fontsize=10)
            ax.legend(fontsize=8, loc="upper right")
        ax.grid(axis="y", alpha=0.30)
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_color(COLORS["ink"])

    fig.suptitle(
        "Pairwise CAM overlap: SSR+KD reduces cross-class attention bleeding on CUB200 fine-grained pairs",
        fontsize=14.0,
        fontweight="bold",
        y=0.995,
    )
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"fig_compound_attention_overlap.{ext}")
    plt.close(fig)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prototypes = load_prototypes()
    class_names = read_class_names_with_id()
    rows = read_metadata()
    with SUMMARY_FILE.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    selected = select_visual_classes(summary, fallback=[0, 8, 14, 22, 30, 70, 110])
    print(f"[attn] selected classes: {selected}")
    backbone, preprocess = build_backbone(device)
    feats_per_image: list[np.ndarray] = []
    images: list[Image.Image] = []
    for cls in selected:
        path = first_test_image_for_class(rows, cls)
        image = Image.open(path).convert("RGB")
        images.append(image)
        feats_pool_input = backbone(preprocess(image).unsqueeze(0).to(device))
        feats = feats_pool_input.detach().cpu().numpy()[0]
        feats_per_image.append(feats)
    plot_attention_grid(feats_per_image, images, class_names, selected, prototypes)
    plot_attention_overlap(feats_per_image, images, class_names, selected, prototypes, summary)
    print(f"[attn] wrote {FIG_DIR / 'fig_compound_attention_maps.pdf'}")
    print(f"[attn] wrote {FIG_DIR / 'fig_compound_attention_overlap.pdf'}")


if __name__ == "__main__":
    main()
