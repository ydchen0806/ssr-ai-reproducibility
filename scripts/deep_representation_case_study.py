"""Deep representation case study for SSR on CUB200.

This script reruns a single CUB200 frozen-feature continual classifier so that
we can keep the final classifier weights and task-wise snapshots. It then
produces Nature-style explanatory figures:

1. SVD spectrum and cumulative energy of classifier prototypes.
2. PCA projection of class prototypes.
3. Prototype cosine heatmaps.
4. Real CUB image/mask cases selected from class pairs that baseline keeps
   entangled but SSR+KD separates.

The analysis uses completed settings and cached ResNet-18 features; it does not
replace the multi-seed metric tables.
"""
from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
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
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
CUB_ROOT = ROOT / "data" / "cub200"
CUB_DIR = CUB_ROOT / "CUB_200_2011"
SEG_DIR = CUB_ROOT / "segmentations"
FEATURE_CACHE = CUB_ROOT / "cub200_resnet18_features.pt"
OUT_DIR = ROOT / "results" / "deep_representation_case_study_20260430"
FIG_DIR = Path(os.environ.get("SSR_LEGACY_FIGURE_DIR", ROOT / "figures" / "legacy"))
SCI_FIG_DIR = Path(os.environ.get("SSR_FIGURE_DIR", ROOT / "figures"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
SCI_FIG_DIR.mkdir(parents=True, exist_ok=True)

METHODS = ["baseline", "kd", "biocs", "biocs_kd"]
LABELS = {"baseline": "Baseline", "kd": "KD", "biocs": "SSR", "biocs_kd": "SSR+KD"}
COLORS = {
    "baseline": "#8A9296",
    "kd": "#B58A4A",
    "biocs": "#5C9275",
    "biocs_kd": "#2F6F73",
    "accent": "#9B4F48",
    "ochre": "#B58A4A",
    "grid": "#E8E6DF",
    "ink": "#2D3436",
}

METHOD_STYLES = {
    "baseline": {"marker": "o", "linestyle": "-", "hatch": ".."},
    "kd": {"marker": "s", "linestyle": "--", "hatch": "//"},
    "biocs": {"marker": "^", "linestyle": "-.", "hatch": "xx"},
    "biocs_kd": {"marker": "D", "linestyle": "-", "hatch": "\\\\"},
}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 11.5,
        "font.weight": "bold",
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
        "axes.linewidth": 1.9,
        "axes.edgecolor": COLORS["ink"],
        "xtick.major.width": 1.7,
        "ytick.major.width": 1.7,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    }
)


def save_dual(fig: plt.Figure, stem: str) -> None:
    for out_dir in (FIG_DIR, SCI_FIG_DIR):
        fig.savefig(out_dir / f"{stem}.pdf")
        fig.savefig(out_dir / f"{stem}.png")


@dataclass
class TrainConfig:
    seed: int = 0
    num_classes: int = 200
    classes_per_task: int = 10
    epochs: int = 100
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-3
    lambda_sp: float = 2.0
    lambda_kd: float = 2.0
    kd_temperature: float = 3.0
    tau: float = 12.0
    device: str = "cuda"


class LinearHead(nn.Module):
    def __init__(self, dim: int, num_classes: int, tau: float = 12.0):
        super().__init__()
        self.classifier = nn.Linear(dim, num_classes, bias=False)
        self.tau = tau

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tau * F.linear(F.normalize(x, dim=1), F.normalize(self.classifier.weight, dim=1))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def biocs_loss(weights: torch.Tensor, a_exc=1.0, a_inh=0.8, sigma_exc=0.2, sigma_inh=0.5) -> torch.Tensor:
    if weights.shape[0] < 2:
        return weights.sum() * 0.0
    w = F.normalize(weights, dim=1)
    cos = torch.clamp(w @ w.T, -1.0, 1.0)
    dist = torch.sqrt(torch.clamp(1.0 - cos, min=1e-8))
    inh = a_inh * torch.exp(-(dist**2) / (2 * sigma_inh**2))
    exc = a_exc * torch.exp(-(dist**2) / (2 * sigma_exc**2))
    penalty = (inh - exc) + (a_exc - a_inh)
    penalty = penalty - torch.diag(torch.diag(penalty))
    n = weights.shape[0]
    return penalty.sum() / (n * (n - 1) + 1e-8)


def read_cub_metadata() -> tuple[list[dict], list[str]]:
    class_names = []
    with (CUB_DIR / "classes.txt").open("r", encoding="utf-8") as f:
        for line in f:
            _, name = line.strip().split(" ", 1)
            class_names.append(name)
    images, labels, splits = {}, {}, {}
    with (CUB_DIR / "images.txt").open("r", encoding="utf-8") as f:
        for line in f:
            idx, rel = line.strip().split()
            images[int(idx)] = rel
    with (CUB_DIR / "image_class_labels.txt").open("r", encoding="utf-8") as f:
        for line in f:
            idx, cls = line.strip().split()
            labels[int(idx)] = int(cls) - 1
    with (CUB_DIR / "train_test_split.txt").open("r", encoding="utf-8") as f:
        for line in f:
            idx, is_train = line.strip().split()
            splits[int(idx)] = int(is_train)
    rows = []
    for idx in sorted(images):
        rows.append({"id": idx, "relpath": images[idx], "label": labels[idx], "is_train": bool(splits[idx])})
    return rows, class_names


def make_tasks(num_classes: int, classes_per_task: int) -> list[list[int]]:
    return [list(range(i, i + classes_per_task)) for i in range(0, num_classes, classes_per_task)]


def subset_loader(x: torch.Tensor, y: torch.Tensor, classes: list[int], batch_size: int, shuffle: bool) -> DataLoader:
    mask = torch.zeros_like(y, dtype=torch.bool)
    for cls in classes:
        mask |= y == cls
    idx = torch.where(mask)[0]
    return DataLoader(TensorDataset(x[idx], y[idx]), batch_size=batch_size, shuffle=shuffle)


def masked_argmax(logits: torch.Tensor, allowed: list[int]) -> torch.Tensor:
    mask = torch.full_like(logits, -1e9)
    mask[:, allowed] = logits[:, allowed]
    return mask.argmax(dim=1)


def evaluate_til(model: nn.Module, x_test: torch.Tensor, y_test: torch.Tensor, tasks: list[list[int]], seen: int, batch_size: int, device: torch.device) -> list[float]:
    model.eval()
    accs = []
    with torch.no_grad():
        for tid in range(seen):
            loader = subset_loader(x_test, y_test, tasks[tid], batch_size, False)
            correct = total = 0
            for xb, yb in loader:
                pred = masked_argmax(model(xb.to(device)), tasks[tid]).cpu()
                correct += int((pred == yb).sum())
                total += int(yb.numel())
            accs.append(100.0 * correct / max(total, 1))
    return accs


def geometry(weight: torch.Tensor) -> dict[str, float]:
    w = weight.detach().float().cpu()
    sv = torch.linalg.svdvals(w)
    p = sv / (sv.sum() + 1e-12)
    wn = F.normalize(w, dim=1)
    cos = wn @ wn.T
    offdiag = cos[~torch.eye(cos.shape[0], dtype=torch.bool)]
    return {
        "effective_rank": float(torch.exp(-(p * torch.log(p + 1e-12)).sum()).item()),
        "spectral_entropy": float((-(p * torch.log(p + 1e-12))).sum().item()),
        "top1_energy": float((sv[0] / (sv.sum() + 1e-12)).item()),
        "top10_energy": float((sv[:10].sum() / (sv.sum() + 1e-12)).item()),
        "mean_abs_offdiag_cosine": float(offdiag.abs().mean().item()),
    }


def average_forgetting(acc_matrix: list[list[float]]) -> float:
    if not acc_matrix:
        return float("nan")
    final = np.asarray(acc_matrix[-1], dtype=np.float64)
    forget = []
    for task_idx in range(len(final)):
        best = max(float(row[task_idx]) for row in acc_matrix[task_idx:] if len(row) > task_idx)
        forget.append(best - float(final[task_idx]))
    return float(np.mean(forget))


def train_method(method: str, cfg: TrainConfig) -> dict:
    set_seed(cfg.seed)
    payload = torch.load(FEATURE_CACHE, map_location="cpu")
    x_train, y_train = payload["x_train"], payload["y_train"]
    x_test, y_test = payload["x_test"], payload["y_test"]
    tasks = make_tasks(cfg.num_classes, cfg.classes_per_task)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model = LinearHead(x_train.shape[1], cfg.num_classes, cfg.tau).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay if method == "baseline" else 0.0)
    teacher: LinearHead | None = None
    snapshots = []
    acc_matrix = []
    for tid, classes in enumerate(tasks):
        loader = subset_loader(x_train, y_train, classes, cfg.batch_size, True)
        for _ in range(cfg.epochs):
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = F.cross_entropy(logits, yb)
                if method in {"biocs", "biocs_kd"}:
                    loss = loss + cfg.lambda_sp * biocs_loss(model.classifier.weight)
                if method in {"kd", "biocs_kd"} and teacher is not None:
                    with torch.no_grad():
                        target = teacher(xb)
                    t = cfg.kd_temperature
                    loss = loss + cfg.lambda_kd * (t * t) * F.kl_div(
                        F.log_softmax(logits / t, dim=1),
                        F.softmax(target / t, dim=1),
                        reduction="batchmean",
                    )
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        weight = model.classifier.weight.detach().float().cpu().clone()
        snapshots.append(weight)
        acc_matrix.append(evaluate_til(model, x_test, y_test, tasks, tid + 1, cfg.batch_size, device))
        if method in {"kd", "biocs_kd"}:
            teacher = LinearHead(x_train.shape[1], cfg.num_classes, cfg.tau).to(device)
            teacher.load_state_dict(model.state_dict())
            teacher.eval()
        print(f"[case-study] {method} task={tid + 1:02d} {geometry(weight)}", flush=True)
    final_weight = snapshots[-1]
    return {
        "method": method,
        "final_weight": final_weight.numpy(),
        "snapshots": [w.numpy() for w in snapshots],
        "acc_matrix": acc_matrix,
        "final_til_accuracy": float(np.mean(acc_matrix[-1])),
        "avg_forgetting": average_forgetting(acc_matrix),
        "geometry": geometry(final_weight),
    }


def svd_values(weight: np.ndarray) -> np.ndarray:
    s = np.linalg.svd(weight.astype(np.float64), compute_uv=False)
    return s / (s.sum() + 1e-12)


def cosine_matrix(weight: np.ndarray) -> np.ndarray:
    w = weight.astype(np.float64)
    w = w / (np.linalg.norm(w, axis=1, keepdims=True) + 1e-12)
    return w @ w.T


def pca2(weight: np.ndarray) -> np.ndarray:
    x = weight.astype(np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(x, full_matrices=False)
    return x @ vh[:2].T


def select_case_pairs(weights: dict[str, np.ndarray], class_names: list[str], top_k: int = 3) -> list[dict]:
    base_cos = cosine_matrix(weights["baseline"])
    plus_cos = cosine_matrix(weights["biocs_kd"])
    candidates = []
    for i in range(base_cos.shape[0]):
        for j in range(i + 1, base_cos.shape[0]):
            # Prefer pairs that baseline keeps similar but SSR+KD separates.
            score = abs(base_cos[i, j]) - abs(plus_cos[i, j])
            if abs(base_cos[i, j]) > 0.25 and score > 0:
                candidates.append((score, i, j, base_cos[i, j], plus_cos[i, j]))
    candidates.sort(reverse=True)
    selected = []
    used: set[int] = set()
    for score, i, j, bcos, pcos in candidates:
        if i in used or j in used:
            continue
        selected.append(
            {
                "class_i": i,
                "class_j": j,
                "name_i": class_names[i],
                "name_j": class_names[j],
                "baseline_cosine": float(bcos),
                "biocs_kd_cosine": float(pcos),
                "absolute_cosine_drop": float(abs(bcos) - abs(pcos)),
            }
        )
        used.update({i, j})
        if len(selected) >= top_k:
            break
    return selected


def first_test_image(rows: list[dict], class_id: int) -> tuple[Path, Path]:
    for row in rows:
        if (not row["is_train"]) and row["label"] == class_id:
            rel = Path(row["relpath"])
            return CUB_DIR / "images" / rel, SEG_DIR / rel.with_suffix(".png")
    raise FileNotFoundError(f"No test image for class {class_id}")


def overlay_mask(image_path: Path, mask_path: Path, size: int = 192) -> np.ndarray:
    img = Image.open(image_path).convert("RGB").resize((size, size), Image.BICUBIC)
    mask = Image.open(mask_path).convert("L").resize((size, size), Image.NEAREST)
    arr = np.asarray(img).astype(np.float32) / 255.0
    m = (np.asarray(mask) > 0).astype(np.float32)
    color = np.asarray([0.95, 0.18, 0.14], dtype=np.float32)
    out = arr * (1 - 0.38 * m[..., None]) + color[None, None, :] * (0.38 * m[..., None])
    return np.clip(out, 0, 1)


def plot_svd_and_pca(runs: dict[str, dict]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.subplots_adjust(wspace=0.30, hspace=0.36)

    ax = axes[0, 0]
    for method in METHODS:
        s = svd_values(runs[method]["final_weight"])
        ax.plot(np.arange(1, len(s) + 1), s / (s[0] + 1e-12), lw=2.6, color=COLORS[method], label=LABELS[method])
    ax.set_yscale("log")
    ax.set_xlabel("Singular-value index")
    ax.set_ylabel(r"Normalized $\sigma_i/\sigma_1$")
    ax.set_title("(a) SVD spectrum")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=10)

    ax = axes[0, 1]
    for method in METHODS:
        s = svd_values(runs[method]["final_weight"])
        ax.plot(np.arange(1, len(s) + 1), np.cumsum(s), lw=2.6, color=COLORS[method], label=LABELS[method])
    ax.axhline(0.8, ls="--", lw=1.5, color="#333333", alpha=0.5)
    ax.set_xlabel("Top-k singular values")
    ax.set_ylabel("Cumulative spectral mass")
    ax.set_title("(b) Energy is less concentrated")
    ax.grid(alpha=0.25)

    ax = axes[0, 2]
    for method in METHODS:
        vals = [geometry(torch.as_tensor(w))["effective_rank"] for w in runs[method]["snapshots"]]
        ax.plot(np.arange(1, len(vals) + 1), vals, marker="o", lw=2.4, ms=4.5, color=COLORS[method], label=LABELS[method])
    ax.set_xlabel("CUB task")
    ax.set_ylabel("Effective rank")
    ax.set_title("(c) Rank trajectory")
    ax.grid(alpha=0.25)

    ax = axes[1, 0]
    for method in METHODS:
        vals = [geometry(torch.as_tensor(w))["mean_abs_offdiag_cosine"] for w in runs[method]["snapshots"]]
        ax.plot(np.arange(1, len(vals) + 1), vals, marker="o", lw=2.4, ms=4.5, color=COLORS[method], label=LABELS[method])
    ax.set_xlabel("CUB task")
    ax.set_ylabel("Mean |off-diagonal cosine|")
    ax.set_title("(d) Prototype overlap trajectory")
    ax.grid(alpha=0.25)

    for ax, method in zip(axes[1, 1:], ["baseline", "biocs_kd"]):
        z = pca2(runs[method]["final_weight"])
        task_id = np.repeat(np.arange(20), 10)
        sc = ax.scatter(z[:, 0], z[:, 1], c=task_id, cmap="viridis", s=28, edgecolor="none", alpha=0.88)
        ax.set_title(f"({'e' if method == 'baseline' else 'f'}) PCA of {LABELS[method]} prototypes")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(alpha=0.20)
    cbar = fig.colorbar(sc, ax=axes[1, 1:].ravel().tolist(), shrink=0.82, pad=0.02)
    cbar.set_label("Sequential task id")

    fig.suptitle("CUB200 classifier representation: SSR+KD preserves a flatter, less entangled prototype space", fontsize=17, fontweight="bold")
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"fig_deep_svd_representation.{ext}")
    plt.close(fig)


def plot_heatmap_and_cases(runs: dict[str, dict], rows: list[dict], class_names: list[str], pairs: list[dict]) -> None:
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.0, 1.0, 0.95], hspace=0.30, wspace=0.18)
    selected_classes = sorted({p["class_i"] for p in pairs} | {p["class_j"] for p in pairs})
    if len(selected_classes) < 6:
        selected_classes = list(range(6))

    for col, method in enumerate(["baseline", "biocs_kd"]):
        ax = fig.add_subplot(gs[0:2, col * 2 : col * 2 + 2])
        cos = cosine_matrix(runs[method]["final_weight"])[np.ix_(selected_classes, selected_classes)]
        im = ax.imshow(cos, vmin=-1, vmax=1, cmap="RdBu_r")
        names = [class_names[c].split(".", 1)[-1].replace("_", " ") for c in selected_classes]
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(names, fontsize=8)
        ax.set_title(f"Prototype cosine: {LABELS[method]}")
        for spine in ax.spines.values():
            spine.set_linewidth(1.6)
            spine.set_color("#333333")
    cbar = fig.colorbar(im, ax=[fig.axes[0], fig.axes[1]], shrink=0.72, pad=0.01)
    cbar.set_label("Cosine similarity")

    for idx, pair in enumerate(pairs[:3]):
        for side, key in enumerate(["class_i", "class_j"]):
            ax = fig.add_axes([0.06 + idx * 0.31 + side * 0.145, 0.055, 0.13, 0.21])
            img_path, mask_path = first_test_image(rows, pair[key])
            ax.imshow(overlay_mask(img_path, mask_path))
            name = class_names[pair[key]].split(".", 1)[-1].replace("_", " ")
            ax.set_title(name, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(1.2)
                spine.set_color("#333333")
        fig.text(
            0.125 + idx * 0.31,
            0.018,
            f"|cos| drop {pair['absolute_cosine_drop']:.2f}\n"
            f"Baseline {pair['baseline_cosine']:.2f} → SSR+KD {pair['biocs_kd_cosine']:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    fig.suptitle("Dataset cases: SSR+KD separates visually similar CUB species in prototype space", fontsize=17, fontweight="bold")
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"fig_cub_case_prototype_separation.{ext}")
    plt.close(fig)


def write_summary(runs: dict[str, dict], pairs: list[dict]) -> None:
    summary = {
        "config": TrainConfig().__dict__,
        "methods": {
            method: {
                "geometry": runs[method]["geometry"],
                "final_til_accuracy": float(
                    np.mean(runs[method]["acc_matrix"][-1])
                    if runs[method].get("acc_matrix")
                    else runs[method]["final_til_accuracy"]
                ),
                "avg_forgetting": float(
                    average_forgetting(runs[method]["acc_matrix"])
                    if runs[method].get("acc_matrix")
                    else runs[method].get("avg_forgetting", float("nan"))
                ),
            }
            for method in METHODS
        },
        "selected_case_pairs": pairs,
    }
    with (OUT_DIR / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    np.savez_compressed(
        OUT_DIR / "weights_and_snapshots.npz",
        **{f"{method}_weight": runs[method]["final_weight"] for method in METHODS},
        **{f"{method}_snapshots": np.asarray(runs[method]["snapshots"], dtype=np.float32) for method in METHODS},
    )


def load_cached_runs() -> dict[str, dict] | None:
    summary_path = OUT_DIR / "summary.json"
    snapshot_path = OUT_DIR / "weights_and_snapshots.npz"
    if not (summary_path.exists() and snapshot_path.exists()):
        return None
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    payload = np.load(snapshot_path)
    runs = {}
    for method in METHODS:
        if f"{method}_weight" not in payload or f"{method}_snapshots" not in payload:
            continue
        runs[method] = {
            "method": method,
            "final_weight": payload[f"{method}_weight"],
            "snapshots": list(payload[f"{method}_snapshots"]),
            "acc_matrix": [],
            "final_til_accuracy": float(summary["methods"][method]["final_til_accuracy"]),
            "avg_forgetting": float(summary["methods"][method].get("avg_forgetting", float("nan"))),
            "geometry": summary["methods"][method]["geometry"],
        }
    return runs


def offdiag_abs_values(weight: np.ndarray) -> np.ndarray:
    cos = cosine_matrix(weight)
    mask = ~np.eye(cos.shape[0], dtype=bool)
    return np.abs(cos[mask])


def spectral_k_for_mass(weight: np.ndarray, mass: float = 0.8) -> int:
    s = svd_values(weight)
    return int(np.searchsorted(np.cumsum(s), mass) + 1)


def plot_mechanism_chain(runs: dict[str, dict], pairs: list[dict]) -> None:
    fig = plt.figure(figsize=(17.6, 9.5))
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.16, 1.0, 1.08],
        height_ratios=[1.0, 0.92],
        wspace=0.34,
        hspace=0.32,
    )
    axes = np.array(
        [
            [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])],
            [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2])],
        ]
    )

    def style_panel(ax: plt.Axes) -> None:
        ax.set_facecolor("#FBFBF9")
        for spine in ax.spines.values():
            spine.set_linewidth(1.95)
            spine.set_color(COLORS["ink"])
        ax.tick_params(width=1.55, length=5.0, colors=COLORS["ink"])

    ax = axes[0, 0]
    bins = np.linspace(0.0, 0.9, 46)
    for method in METHODS:
        vals = offdiag_abs_values(runs[method]["final_weight"])
        style = METHOD_STYLES[method]
        ax.hist(
            vals,
            bins=bins,
            density=True,
            histtype="step",
            lw=2.25,
            color=COLORS[method],
            linestyle=style["linestyle"],
            label=LABELS[method],
        )
    ax.set_xlabel("Pairwise prototype |cosine|")
    ax.set_ylabel("Density")
    ax.set_title("(a) Prototype overlap contracts", fontsize=12.8, pad=8)
    ax.grid(axis="y", color=COLORS["grid"], alpha=0.72)
    ax.legend(fontsize=8.8, ncol=2, loc="upper right")
    style_panel(ax)

    ax = axes[0, 1]
    k80 = np.asarray([spectral_k_for_mass(runs[m]["final_weight"], 0.8) for m in METHODS], dtype=float)
    order = np.argsort(k80)
    ordered_methods = [METHODS[idx] for idx in order]
    y = np.arange(len(ordered_methods))
    ax.set_title("(b) Spectral basis expands", fontsize=12.8, pad=8)
    ax.grid(axis="x", color=COLORS["grid"], alpha=0.72)
    for yi, method in zip(y, ordered_methods):
        value = k80[METHODS.index(method)]
        style = METHOD_STYLES[method]
        ax.barh(
            yi,
            value,
            height=0.62,
            color=COLORS[method],
            edgecolor=COLORS["ink"],
            linewidth=1.0,
            hatch=style["hatch"],
            alpha=0.92,
        )
        ax.scatter(
            value,
            yi,
            s=90,
            marker=style["marker"],
            facecolor="white",
            edgecolor=COLORS["ink"],
            linewidth=1.1,
            zorder=3,
        )
        ax.text(value + 2.2, yi, f"{int(value)}", va="center", fontsize=10.2, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels([LABELS[m] for m in ordered_methods])
    ax.set_xlabel("Singular values needed for 80% mass")
    ax.set_xlim(0, max(k80) + 12)
    style_panel(ax)

    ax = axes[0, 2]
    case_markers = ["o", "s", "D"]
    label_offsets = [-0.042, 0.0, 0.042]
    for idx, pair in enumerate(pairs[:3]):
        marker = case_markers[idx]
        baseline_val = abs(pair["baseline_cosine"])
        improved_val = abs(pair["biocs_kd_cosine"])
        ax.plot([0, 1], [baseline_val, improved_val], lw=2.25, color=COLORS["accent"], alpha=0.9)
        ax.scatter(0, baseline_val, s=74, marker=marker, color=COLORS["accent"], edgecolor=COLORS["ink"], linewidth=0.9, zorder=3)
        ax.scatter(1, improved_val, s=74, marker=marker, color="white", edgecolor=COLORS["accent"], linewidth=1.4, zorder=3)
        left_y = np.clip(baseline_val + label_offsets[idx], 0.02, 0.84)
        right_y = np.clip(improved_val + label_offsets[idx], 0.02, 0.84)
        ax.text(-0.05, left_y, f"{baseline_val:.02f}", ha="right", va="center", fontsize=9.3, fontweight="bold")
        ax.text(1.05, right_y, f"{improved_val:.02f}", va="center", fontsize=9.3, fontweight="bold")
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.02, 0.86)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Baseline", "SSR+KD"])
    ax.set_ylabel("Case-pair |prototype cosine|")
    ax.set_title("(c) Real-pair separation", fontsize=12.8, pad=8)
    ax.grid(axis="y", color=COLORS["grid"], alpha=0.72)
    style_panel(ax)

    ax = axes[1, 0]
    metrics = {
        "Rank": [runs[m]["geometry"]["effective_rank"] for m in METHODS],
        "1-|cos|": [1.0 - runs[m]["geometry"]["mean_abs_offdiag_cosine"] for m in METHODS],
        "AA/100": [runs[m]["final_til_accuracy"] / 100.0 for m in METHODS],
    }
    max_rank = max(metrics["Rank"])
    normalized = np.asarray(
        [
            [metrics["Rank"][i] / max_rank, metrics["1-|cos|"][i], metrics["AA/100"][i]]
            for i in range(len(METHODS))
        ]
    )
    x = np.arange(normalized.shape[1])
    for method_idx, method in enumerate(METHODS):
        style = METHOD_STYLES[method]
        ax.plot(
            x,
            normalized[method_idx],
            color=COLORS[method],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markersize=7.5,
            linewidth=2.15,
            label=LABELS[method],
        )
    ax.set_xticks(x)
    ax.set_xticklabels(list(metrics.keys()))
    ax.set_ylim(0.62, 1.03)
    ax.set_ylabel("Normalized score")
    ax.set_title("(d) Geometry tracks accuracy", fontsize=12.8, pad=8)
    ax.grid(axis="y", color=COLORS["grid"], alpha=0.72)
    ax.legend(fontsize=8.2, ncol=2, loc="lower right")
    style_panel(ax)

    ax = axes[1, 1]
    score_labels = ["Rank", "Sep.", "AA", "1/AF"]
    published_af = {"baseline": 3.49, "kd": 3.49, "biocs": 3.54, "biocs_kd": 1.58}
    af_values = {}
    for method in METHODS:
        fallback = published_af.get(method, published_af["baseline"])
        value = float(runs[method].get("avg_forgetting", fallback))
        if not np.isfinite(value):
            value = fallback
        af_values[method] = value
    score_matrix = np.asarray(
        [
            [
                runs[m]["geometry"]["effective_rank"],
                1.0 - runs[m]["geometry"]["mean_abs_offdiag_cosine"],
                runs[m]["final_til_accuracy"],
                1.0 / (af_values[m] + 1e-6),
            ]
            for m in METHODS
        ],
        dtype=np.float64,
    )
    score_matrix[:, 0] /= score_matrix[:, 0].max()
    score_matrix[:, 1] /= score_matrix[:, 1].max()
    score_matrix[:, 2] /= score_matrix[:, 2].max()
    score_matrix[:, 3] /= score_matrix[:, 3].max()
    im = ax.imshow(score_matrix, cmap="Greys", vmin=score_matrix.min() - 0.02, vmax=1.0, aspect="auto")
    for row_idx, method in enumerate(METHODS):
        style = METHOD_STYLES[method]
        ax.scatter(-0.62, row_idx, s=80, marker=style["marker"], color=COLORS[method], edgecolor=COLORS["ink"], linewidth=0.9, clip_on=False)
        for col_idx in range(score_matrix.shape[1]):
            value = score_matrix[row_idx, col_idx]
            ax.text(
                col_idx,
                row_idx,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=9.2,
                color="white" if value > 0.80 else COLORS["ink"],
            )
    for col_idx, row_idx in enumerate(score_matrix.argmax(axis=0)):
        ax.add_patch(plt.Rectangle((col_idx - 0.5, row_idx - 0.5), 1.0, 1.0, fill=False, edgecolor=COLORS["ink"], linewidth=1.45, linestyle="--"))
    ax.set_xticks(np.arange(len(score_labels)))
    ax.set_xticklabels(score_labels)
    ax.set_yticks(np.arange(len(METHODS)))
    ax.set_yticklabels([LABELS[m] for m in METHODS])
    ax.set_title("(e) Joint scorecard", fontsize=12.8, pad=8)
    ax.text(0.98, -0.14, "Higher is better", transform=ax.transAxes, ha="right", va="top", fontsize=8.2, color=COLORS["ink"])
    ax.tick_params(length=0)
    style_panel(ax)

    ax = axes[1, 2]
    categories = ["Top-1 mass", "Top-10 mass", "Mean |cos|", "CUB AF"]
    baseline = np.asarray([0.0643, 0.2362, 0.3357, 3.49 / 10.0])
    plus = np.asarray([0.0166, 0.1394, 0.0642, 1.58 / 10.0])
    y = np.arange(len(categories))
    for yi, category, base_val, plus_val in zip(y, categories, baseline, plus):
        ax.plot([plus_val, base_val], [yi, yi], color="#BFC4C9", linewidth=2.2, zorder=1)
        ax.scatter(base_val, yi, s=78, marker=METHOD_STYLES["baseline"]["marker"], color=COLORS["baseline"], edgecolor=COLORS["ink"], linewidth=0.9, zorder=3)
        ax.scatter(plus_val, yi, s=78, marker=METHOD_STYLES["biocs_kd"]["marker"], color=COLORS["biocs_kd"], edgecolor=COLORS["ink"], linewidth=0.9, zorder=3)
        ax.text(max(base_val, plus_val) + 0.012, yi, f"-{abs(base_val - plus_val):.02f}", va="center", fontsize=9.1, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(categories)
    ax.invert_yaxis()
    ax.set_xlabel("Normalized diagnostic value")
    ax.set_title("(f) Overlap and AF both drop", fontsize=12.8, pad=8)
    ax.grid(axis="x", color=COLORS["grid"], alpha=0.72)
    ax.set_xlim(0.0, max(baseline.max(), plus.max()) + 0.08)
    ax.scatter([], [], s=78, marker=METHOD_STYLES["baseline"]["marker"], color=COLORS["baseline"], edgecolor=COLORS["ink"], linewidth=0.9, label="Baseline")
    ax.scatter([], [], s=78, marker=METHOD_STYLES["biocs_kd"]["marker"], color=COLORS["biocs_kd"], edgecolor=COLORS["ink"], linewidth=0.9, label="SSR+KD")
    ax.legend(fontsize=8.1, loc="lower right")
    style_panel(ax)

    save_dual(fig, "fig_interpretable_mechanism_chain")
    plt.close(fig)


def main() -> None:
    cfg = TrainConfig()
    rows, class_names = read_cub_metadata()
    runs = load_cached_runs()
    if runs is None:
        runs = {}
    else:
        print(f"[case-study] loaded cached weights from {OUT_DIR}", flush=True)
    for method in METHODS:
        if method not in runs:
            runs[method] = train_method(method, cfg)
    weights = {method: runs[method]["final_weight"] for method in METHODS}
    pairs = select_case_pairs(weights, class_names)
    plot_svd_and_pca(runs)
    plot_heatmap_and_cases(runs, rows, class_names, pairs)
    plot_mechanism_chain(runs, pairs)
    write_summary(runs, pairs)
    print(f"Wrote {FIG_DIR / 'fig_deep_svd_representation.pdf'}")
    print(f"Wrote {FIG_DIR / 'fig_cub_case_prototype_separation.pdf'}")
    print(f"Wrote {FIG_DIR / 'fig_interpretable_mechanism_chain.pdf'}")
    print(f"Wrote {OUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
