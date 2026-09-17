#!/usr/bin/env python3
"""Continual low-rank adapter probe for Spatial Synaptic Regularization (SSR).

This is a lightweight PEFT-style experiment inspired by LoRA/adapters. It uses
cached CUB200 ResNet-18 features as a frozen representation, then trains only a
small bottleneck adapter plus a classifier over a sequential class stream.

The purpose is not to claim LoRA SOTA; it is to test whether the biological
center-surround prior remains useful when the plastic parameters are low-rank
adapter directions rather than only classifier prototypes.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
FEATURE_CACHE = ROOT / "data" / "cub200" / "cub200_resnet18_features.pt"


@dataclass
class AdapterRun:
    method: str
    seed: int
    avg_accuracy: float
    avg_forgetting: float
    effective_rank: float
    mean_abs_offdiag_cosine: float
    adapter_basis_cosine: float
    adapter_delta_norm: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


KERNEL_FAMILIES = ("gaussian", "laplace", "cauchy", "inverse")
METHODS = ("baseline", "kd", "biocs", "biocs_kd")


def radial_kernel(dist: torch.Tensor, sigma: float, family: str) -> torch.Tensor:
    """Evaluate the adapter-study radial response.

    The historical ``inverse`` key denotes
    ``sigma / sqrt(distance**2 + sigma**2)`` in this experiment.
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    if family == "gaussian":
        return torch.exp(-(dist ** 2) / (2 * sigma ** 2))
    if family == "laplace":
        return torch.exp(-dist / sigma)
    if family == "cauchy":
        return 1.0 / (1.0 + (dist / sigma) ** 2)
    if family == "inverse":
        return sigma / torch.sqrt(dist ** 2 + sigma ** 2)
    raise ValueError(f"Unknown kernel_family={family!r}; choose from {KERNEL_FAMILIES}")


def biocs_loss(
    weights: torch.Tensor,
    a_exc=1.0,
    a_inh=0.8,
    sigma_exc=0.2,
    sigma_inh=0.5,
    kernel_family="gaussian",
    distance_metric="cosine",
) -> torch.Tensor:
    if weights.shape[0] < 2:
        return weights.sum() * 0.0
    w = F.normalize(weights.float(), dim=1)
    cos = torch.clamp(w @ w.T, -1.0, 1.0)
    if distance_metric == "projective":
        dist = torch.sqrt(torch.clamp(1.0 - cos.square(), min=1e-8))
    elif distance_metric == "cosine":
        dist = torch.sqrt(torch.clamp(1.0 - cos, min=1e-8))
    else:
        raise ValueError(
            f"Unknown distance_metric={distance_metric!r}; "
            "choose 'cosine' or 'projective'"
        )
    inh = a_inh * radial_kernel(dist, sigma_inh, kernel_family)
    exc = a_exc * radial_kernel(dist, sigma_exc, kernel_family)
    penalty = (inh - exc) + (a_exc - a_inh)
    penalty = penalty - torch.diag(torch.diag(penalty))
    n = weights.shape[0]
    return penalty.sum() / (n * (n - 1) + 1e-8)


class AdapterHead(nn.Module):
    def __init__(self, dim: int, num_classes: int, rank: int, tau: float = 12.0, adapter_scale: float = 0.5):
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        self.classifier = nn.Linear(dim, num_classes, bias=False)
        self.tau = tau
        self.adapter_scale = adapter_scale
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        delta = self.up(F.gelu(self.down(x)))
        return x + self.adapter_scale * delta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encode(x)
        return self.tau * F.linear(F.normalize(h, dim=1), F.normalize(self.classifier.weight, dim=1))

    def adapter_basis(self) -> torch.Tensor:
        return self.up.weight.T


def make_tasks(num_classes: int, classes_per_task: int, seed: int, order: str) -> list[list[int]]:
    classes = list(range(num_classes))
    if order == "random":
        random.Random(seed).shuffle(classes)
    return [classes[i : i + classes_per_task] for i in range(0, num_classes, classes_per_task)]


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


def evaluate_til(
    model: nn.Module,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    tasks: list[list[int]],
    seen: int,
    batch_size: int,
    device: torch.device,
) -> list[float]:
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


def geometry(model: AdapterHead, initial_up: torch.Tensor) -> dict[str, float]:
    w = model.classifier.weight.detach().float().cpu()
    sv = torch.linalg.svdvals(w)
    p = sv / (sv.sum() + 1e-12)
    wn = F.normalize(w, dim=1)
    cos = wn @ wn.T
    offdiag = cos[~torch.eye(cos.shape[0], dtype=torch.bool)]

    basis = model.adapter_basis().detach().float().cpu()
    bn = F.normalize(basis, dim=1)
    bcos = bn @ bn.T
    boff = bcos[~torch.eye(bcos.shape[0], dtype=torch.bool)]
    return {
        "effective_rank": float(torch.exp(-(p * torch.log(p + 1e-12)).sum()).item()),
        "mean_abs_offdiag_cosine": float(offdiag.abs().mean().item()),
        "adapter_basis_cosine": float(boff.abs().mean().item()) if boff.numel() else 0.0,
        "adapter_delta_norm": float((model.up.weight.detach().cpu() - initial_up).norm().item()),
    }


def run_one(args: argparse.Namespace, method: str, seed: int, device: torch.device) -> AdapterRun:
    if method not in METHODS:
        raise ValueError(f"Unknown method={method!r}; choose from {METHODS}")
    set_seed(seed)
    payload = torch.load(args.feature_cache, map_location="cpu")
    x_train = payload["x_train"].float()
    y_train = payload["y_train"].long()
    x_test = payload["x_test"].float()
    y_test = payload["y_test"].long()
    tasks = make_tasks(args.num_classes, args.classes_per_task, seed, args.class_order)
    model = AdapterHead(x_train.shape[1], args.num_classes, args.rank, args.tau, args.adapter_scale).to(device)
    initial_up = model.up.weight.detach().cpu().clone()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    teacher: AdapterHead | None = None
    acc_matrix = []

    for tid, classes in enumerate(tasks):
        loader = subset_loader(x_train, y_train, classes, args.batch_size, True)
        for _ in range(args.epochs):
            model.train()
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = F.cross_entropy(logits, yb)
                if method in {"biocs", "biocs_kd"}:
                    kernel = (args.a_exc, args.a_inh, args.sigma_exc, args.sigma_inh, args.kernel_family)
                    if args.lambda_cls > 0:
                        loss = loss + args.lambda_cls * biocs_loss(
                            model.classifier.weight,
                            *kernel,
                            distance_metric=args.prototype_distance_metric,
                        )
                    if args.lambda_adapter > 0:
                        loss = loss + args.lambda_adapter * biocs_loss(
                            model.adapter_basis(),
                            *kernel,
                            distance_metric=args.adapter_distance_metric,
                        )
                if method in {"kd", "biocs_kd"} and teacher is not None:
                    with torch.no_grad():
                        target = teacher(xb)
                    t = args.kd_temperature
                    loss = loss + args.lambda_kd * (t * t) * F.kl_div(
                        F.log_softmax(logits / t, dim=1),
                        F.softmax(target / t, dim=1),
                        reduction="batchmean",
                    )
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                opt.step()
        acc_matrix.append(evaluate_til(model, x_test, y_test, tasks, tid + 1, args.batch_size, device))
        if method in {"kd", "biocs_kd"}:
            teacher = AdapterHead(x_train.shape[1], args.num_classes, args.rank, args.tau, args.adapter_scale).to(device)
            teacher.load_state_dict(model.state_dict())
            teacher.eval()

    final = acc_matrix[-1]
    forgetting = []
    for task_idx in range(len(tasks) - 1):
        best = max(row[task_idx] for row in acc_matrix[task_idx:])
        forgetting.append(best - final[task_idx])
    geo = geometry(model, initial_up)
    return AdapterRun(
        method=method,
        seed=seed,
        avg_accuracy=float(np.mean(final)),
        avg_forgetting=float(np.mean(forgetting)) if forgetting else 0.0,
        **geo,
    )


def summarize(rows: list[AdapterRun]) -> dict:
    out = {}
    for method in sorted({r.method for r in rows}):
        items = [r for r in rows if r.method == method]
        out[method] = {}
        for key in [
            "avg_accuracy",
            "avg_forgetting",
            "effective_rank",
            "mean_abs_offdiag_cosine",
            "adapter_basis_cosine",
            "adapter_delta_norm",
        ]:
            vals = np.asarray([getattr(r, key) for r in items], dtype=float)
            out[method][key] = {"mean": float(vals.mean()), "std": float(vals.std()), "n": int(vals.size)}
    return out


def write_summary_md(summary: dict, path: Path) -> None:
    lines = [
        "# Low-Rank Adapter SSR Probe",
        "",
        "Frozen CUB200 ResNet-18 features with a trainable bottleneck adapter and classifier head.",
        "",
        "| Method | AA | AF | Eff. rank | |offdiag cos| | Adapter basis cos | Adapter delta norm |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method, stats in summary.items():
        def fmt(key: str) -> str:
            s = stats[key]
            return f"{s['mean']:.3f} +/- {s['std']:.3f}"

        lines.append(
            "| "
            + " | ".join(
                [
                    method,
                    fmt("avg_accuracy"),
                    fmt("avg_forgetting"),
                    fmt("effective_rank"),
                    fmt("mean_abs_offdiag_cosine"),
                    fmt("adapter_basis_cosine"),
                    fmt("adapter_delta_norm"),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Interpretation: this is a lightweight PEFT-style transfer probe, not a LoRA SOTA benchmark.",
            "A useful positive signal would be lower forgetting or better geometry at matched accuracy.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--feature_cache", default=str(FEATURE_CACHE))
    p.add_argument("--output_dir", default="results/lowrank_adapter_probe_20260514")
    p.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--num_classes", type=int, default=200)
    p.add_argument("--classes_per_task", type=int, default=10)
    p.add_argument("--class_order", choices=["semantic", "random"], default="semantic")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--adapter_scale", type=float, default=0.5)
    p.add_argument("--tau", type=float, default=12.0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--lambda_cls", type=float, default=1.0)
    p.add_argument("--lambda_adapter", type=float, default=0.2)
    p.add_argument("--lambda_kd", type=float, default=2.0)
    p.add_argument("--kd_temperature", type=float, default=3.0)
    p.add_argument("--a-exc", type=float, default=1.0)
    p.add_argument("--a-inh", type=float, default=0.8)
    p.add_argument("--sigma-exc", type=float, default=0.2)
    p.add_argument("--sigma-inh", type=float, default=0.5)
    p.add_argument("--kernel-family", choices=KERNEL_FAMILIES, default="gaussian")
    p.add_argument(
        "--prototype-distance-metric",
        choices=["cosine", "projective"],
        default="cosine",
    )
    p.add_argument(
        "--adapter-distance-metric",
        choices=["cosine", "projective"],
        default="projective",
    )
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rows = []
    runs_path = outdir / "runs.jsonl"
    if runs_path.exists():
        runs_path.unlink()
    for seed in args.seeds:
        for method in args.methods:
            print(f"[adapter] method={method} seed={seed} device={device}", flush=True)
            row = run_one(args, method, seed, device)
            rows.append(row)
            with runs_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")
            print(json.dumps(asdict(row), ensure_ascii=False), flush=True)
    summary = summarize(rows)
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_summary_md(summary, outdir / "summary.md")
    print(f"Saved to {outdir}", flush=True)


if __name__ == "__main__":
    main()
