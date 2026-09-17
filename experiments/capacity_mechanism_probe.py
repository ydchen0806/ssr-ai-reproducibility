"""Capacity and mechanism probe for SSR.

This experiment is intentionally lightweight: it uses synthetic fine-grained
classes arranged around shared latent bases, then adds classes sequentially.
It reports a capacity-style curve plus representation diagnostics:
effective rank, classifier orthogonality, total weight movement, and Fourier
high-frequency ratio of classifier updates.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class RunSummary:
    method: str
    seed: int
    avg_accuracy: float
    final_cil_accuracy: float
    capacity_at_50: int
    capacity_at_40: int
    avg_forgetting: float
    effective_rank: float
    spectral_entropy: float
    mean_abs_offdiag_cosine: float
    total_classifier_movement: float
    fourier_high_freq_ratio: float


class ProbeMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, tau: float = 12.0):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes, bias=False)
        self.tau = tau

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        return self.tau * F.linear(F.normalize(h, dim=1), F.normalize(self.classifier.weight, dim=1))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_latent_classes(
    seed: int,
    num_classes: int,
    input_dim: int,
    num_bases: int,
    bases_per_class: int,
    samples_per_class: int,
    noise_std: float,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    bases = []
    for _ in range(num_bases):
        v = rng.normal(size=input_dim)
        for b in bases:
            v = v - np.dot(v, b) * b
        v = v / (np.linalg.norm(v) + 1e-12)
        bases.append(v)
    bases = np.asarray(bases)

    x_all, y_all = [], []
    for cls in range(num_classes):
        # Neighboring classes share many bases, giving a fine-grained
        # discrimination problem instead of easy orthogonal clusters.
        start = cls % num_bases
        chosen = [(start + k * 3) % num_bases for k in range(bases_per_class)]
        center = bases[chosen].sum(axis=0)
        center = center / (np.linalg.norm(center) + 1e-12)
        jitter = 0.08 * rng.normal(size=input_dim)
        center = center + jitter
        center = center / (np.linalg.norm(center) + 1e-12)
        for _ in range(samples_per_class):
            sample = center + noise_std * rng.normal(size=input_dim)
            sample = sample / (np.linalg.norm(sample) + 1e-12)
            x_all.append(sample.astype(np.float32))
            y_all.append(cls)
    return np.asarray(x_all, dtype=np.float32), np.asarray(y_all, dtype=np.int64)


def make_loaders(
    x: np.ndarray,
    y: np.ndarray,
    classes: list[int],
    samples_per_class: int,
    train_per_class: int,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    train_idx, test_idx = [], []
    for cls in classes:
        start = cls * samples_per_class
        train_idx.extend(range(start, start + train_per_class))
        test_idx.extend(range(start + train_per_class, start + samples_per_class))
    train = TensorDataset(torch.from_numpy(x[train_idx]), torch.from_numpy(y[train_idx]))
    test = TensorDataset(torch.from_numpy(x[test_idx]), torch.from_numpy(y[test_idx]))
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=False),
        DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=False),
    )


def biocs_loss(weights: torch.Tensor, a_exc=1.0, a_inh=0.8, sigma_exc=0.2, sigma_inh=0.5) -> torch.Tensor:
    w = F.normalize(weights, dim=1)
    cos = torch.clamp(w @ w.T, -1.0, 1.0)
    dist = torch.sqrt(torch.clamp(1.0 - cos, min=1e-8))
    inh = a_inh * torch.exp(-(dist**2) / (2 * sigma_inh**2))
    exc = a_exc * torch.exp(-(dist**2) / (2 * sigma_exc**2))
    penalty = (inh - exc) + (a_exc - a_inh)
    penalty = penalty - torch.diag(torch.diag(penalty))
    n = weights.shape[0]
    return penalty.sum() / (n * (n - 1) + 1e-8)


def evaluate_cil(model: nn.Module, loaders: list[DataLoader], seen_tasks: int, device: torch.device) -> list[float]:
    model.eval()
    accs = []
    with torch.no_grad():
        for tid in range(seen_tasks):
            correct = 0
            total = 0
            for xb, yb in loaders[tid]:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb).argmax(dim=1)
                correct += (pred == yb).sum().item()
                total += yb.numel()
            accs.append(100.0 * correct / max(total, 1))
    return accs


def diagnostics(
    final_weight: torch.Tensor,
    weight_snapshots: list[torch.Tensor],
) -> dict[str, float]:
    w = final_weight.detach().float().cpu()
    sv = torch.linalg.svdvals(w)
    p = sv / (sv.sum() + 1e-12)
    entropy = -(p * torch.log(p + 1e-12)).sum().item()
    effective_rank = math.exp(entropy)

    wn = F.normalize(w, dim=1)
    cos = wn @ wn.T
    offdiag = cos[~torch.eye(cos.shape[0], dtype=torch.bool)]
    mean_abs_offdiag = offdiag.abs().mean().item()

    movement = 0.0
    high_freq = []
    for prev, cur in zip(weight_snapshots[:-1], weight_snapshots[1:]):
        delta = (cur - prev).detach().float().cpu()
        movement += delta.abs().sum().item()
        spectrum = torch.fft.rfft(delta.flatten())
        power = spectrum.abs().pow(2)
        cutoff = max(1, int(0.6 * power.numel()))
        high_freq.append((power[cutoff:].sum() / (power.sum() + 1e-12)).item())

    return {
        "effective_rank": effective_rank,
        "spectral_entropy": entropy,
        "mean_abs_offdiag_cosine": mean_abs_offdiag,
        "total_classifier_movement": movement,
        "fourier_high_freq_ratio": float(np.mean(high_freq)) if high_freq else 0.0,
    }


def run_one(args: argparse.Namespace, method: str, seed: int, device: torch.device) -> RunSummary:
    set_seed(seed)
    x, y = make_latent_classes(
        seed=seed,
        num_classes=args.num_classes,
        input_dim=args.input_dim,
        num_bases=args.num_bases,
        bases_per_class=args.bases_per_class,
        samples_per_class=args.samples_per_class,
        noise_std=args.noise_std,
    )

    classes = list(range(args.num_classes))
    tasks = [classes[i : i + args.classes_per_task] for i in range(0, args.num_classes, args.classes_per_task)]
    tasks = [t for t in tasks if len(t) == args.classes_per_task]
    train_loaders, test_loaders = [], []
    for task_classes in tasks:
        train, test = make_loaders(
            x, y, task_classes, args.samples_per_class, args.train_per_class, args.batch_size
        )
        train_loaders.append(train)
        test_loaders.append(test)

    model = ProbeMLP(args.input_dim, args.hidden_dim, args.num_classes).to(device)
    weight_decay = args.l2_weight_decay if method == "l2" else 0.0
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=weight_decay)
    teacher: ProbeMLP | None = None
    acc_matrix = []
    snapshots = [model.classifier.weight.detach().clone().cpu()]

    for tid, loader in enumerate(train_loaders):
        model.train()
        for _ in range(args.epochs):
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = F.cross_entropy(logits, yb)
                if method in {"biocs", "biocs_kd"}:
                    loss = loss + args.lambda_sp * biocs_loss(model.classifier.weight)
                if method == "biocs_kd" and teacher is not None:
                    with torch.no_grad():
                        target = teacher(xb)
                    t = args.kd_temperature
                    loss = loss + args.lambda_kd * (t * t) * F.kl_div(
                        F.log_softmax(logits / t, dim=1),
                        F.softmax(target / t, dim=1),
                        reduction="batchmean",
                    )
                opt.zero_grad()
                loss.backward()
                opt.step()

        acc_matrix.append(evaluate_cil(model, test_loaders, tid + 1, device))
        snapshots.append(model.classifier.weight.detach().clone().cpu())
        if method == "biocs_kd":
            teacher = ProbeMLP(args.input_dim, args.hidden_dim, args.num_classes).to(device)
            teacher.load_state_dict(model.state_dict())
            teacher.eval()

    final_accs = acc_matrix[-1]
    avg_accuracy = float(np.mean(final_accs))
    final_cil_accuracy = float(final_accs[-1])
    thresholds = {}
    running_means = [float(np.mean(row)) for row in acc_matrix]
    for thr in (50, 40):
        cap = 0
        for i, val in enumerate(running_means, start=1):
            if val >= thr:
                cap = i * args.classes_per_task
        thresholds[thr] = cap

    forgetting = []
    for task_idx in range(len(tasks) - 1):
        best = max(row[task_idx] for row in acc_matrix[task_idx:])
        forgetting.append(best - acc_matrix[-1][task_idx])

    diag = diagnostics(model.classifier.weight, snapshots)
    return RunSummary(
        method=method,
        seed=seed,
        avg_accuracy=avg_accuracy,
        final_cil_accuracy=final_cil_accuracy,
        capacity_at_50=thresholds[50],
        capacity_at_40=thresholds[40],
        avg_forgetting=float(np.mean(forgetting)) if forgetting else 0.0,
        **diag,
    )


def summarize(rows: list[RunSummary], outdir: Path) -> None:
    by_method: dict[str, list[RunSummary]] = {}
    for row in rows:
        by_method.setdefault(row.method, []).append(row)

    summary = {}
    keys = [k for k in asdict(rows[0]).keys() if k not in {"method", "seed"}]
    for method, method_rows in by_method.items():
        summary[method] = {}
        for key in keys:
            vals = np.asarray([getattr(r, key) for r in method_rows], dtype=float)
            summary[method][key] = {"mean": float(vals.mean()), "std": float(vals.std())}

    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    methods = list(by_method)
    metrics = [
        ("avg_accuracy", "Avg CIL Accuracy (%)"),
        ("capacity_at_40", "Capacity@40 (%)"),
        ("effective_rank", "Effective Rank"),
        ("mean_abs_offdiag_cosine", "Mean |Offdiag Cosine|"),
        ("total_classifier_movement", "Classifier Movement"),
        ("fourier_high_freq_ratio", "High-Frequency Ratio"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for ax, (key, label) in zip(axes, metrics):
        means = [summary[m][key]["mean"] for m in methods]
        stds = [summary[m][key]["std"] for m in methods]
        ax.bar(methods, means, yerr=stds, color=["#6B7280", "#4B9CD3", "#0D7C66", "#C27C0E"][: len(methods)])
        ax.set_title(label)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "capacity_mechanism_summary.png", dpi=180)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default="results/capacity_mechanism_probe")
    p.add_argument("--methods", nargs="+", default=["baseline", "l2", "biocs", "biocs_kd"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--num_classes", type=int, default=60)
    p.add_argument("--classes_per_task", type=int, default=5)
    p.add_argument("--input_dim", type=int, default=128)
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--num_bases", type=int, default=18)
    p.add_argument("--bases_per_class", type=int, default=4)
    p.add_argument("--samples_per_class", type=int, default=180)
    p.add_argument("--train_per_class", type=int, default=140)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--noise_std", type=float, default=0.06)
    p.add_argument("--lambda_sp", type=float, default=5.0)
    p.add_argument("--lambda_kd", type=float, default=2.0)
    p.add_argument("--kd_temperature", type=float, default=3.0)
    p.add_argument("--l2_weight_decay", type=float, default=1e-3)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rows: list[RunSummary] = []
    for seed in args.seeds:
        for method in args.methods:
            print(f"[capacity] method={method} seed={seed}", flush=True)
            row = run_one(args, method, seed, device)
            rows.append(row)
            with (outdir / "runs.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")
            print(json.dumps(asdict(row), ensure_ascii=False), flush=True)
    summarize(rows, outdir)
    print(f"Saved to {outdir}", flush=True)


if __name__ == "__main__":
    main()
