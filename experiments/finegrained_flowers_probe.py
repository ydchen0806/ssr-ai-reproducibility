"""Fine-grained continual-learning probe on Flowers102 frozen features.

Flowers102 is used as a lightweight CUB-style fine-grained classification
benchmark available through torchvision in this environment. The script extracts
ImageNet ResNet-18 features once, then trains a sequential classifier head with
baseline / SSR / SSR+KD variants.
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
from torch.utils.data import ConcatDataset, DataLoader, Subset, TensorDataset
from torchvision import datasets, models, transforms


@dataclass
class FlowerRun:
    method: str
    seed: int
    avg_accuracy: float
    avg_forgetting: float
    effective_rank: float
    mean_abs_offdiag_cosine: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


class LinearHead(nn.Module):
    def __init__(self, dim: int, num_classes: int, tau: float = 12.0):
        super().__init__()
        self.classifier = nn.Linear(dim, num_classes, bias=False)
        self.tau = tau

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tau * F.linear(F.normalize(x, dim=1), F.normalize(self.classifier.weight, dim=1))


def extract_features(args: argparse.Namespace, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    cache = Path(args.cache)
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        payload = torch.load(cache, map_location="cpu")
        return payload["x_train"], payload["y_train"], payload["x_test"], payload["y_test"]

    weights = models.ResNet18_Weights.DEFAULT
    preprocess = weights.transforms()
    train_set = ConcatDataset(
        [
            datasets.Flowers102(args.data_root, split="train", download=True, transform=preprocess),
            datasets.Flowers102(args.data_root, split="val", download=True, transform=preprocess),
        ]
    )
    test_set = datasets.Flowers102(args.data_root, split="test", download=True, transform=preprocess)

    backbone = models.resnet18(weights=weights)
    backbone.fc = nn.Identity()
    backbone = backbone.to(device).eval()

    def collect(dataset) -> tuple[torch.Tensor, torch.Tensor]:
        loader = DataLoader(dataset, batch_size=args.feature_batch_size, shuffle=False, num_workers=args.workers)
        xs, ys = [], []
        with torch.no_grad():
            for xb, yb in loader:
                feats = backbone(xb.to(device)).cpu()
                xs.append(feats)
                ys.append(yb.cpu().long())
        return torch.cat(xs), torch.cat(ys)

    x_train, y_train = collect(train_set)
    x_test, y_test = collect(test_set)
    torch.save({"x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test}, cache)
    return x_train, y_train, x_test, y_test


def make_tasks(y_train: torch.Tensor, num_classes: int, classes_per_task: int, seed: int) -> list[list[int]]:
    order = list(range(num_classes))
    rng = random.Random(seed)
    rng.shuffle(order)
    return [order[i : i + classes_per_task] for i in range(0, num_classes, classes_per_task) if len(order[i : i + classes_per_task]) == classes_per_task]


def subset_loader(x: torch.Tensor, y: torch.Tensor, classes: list[int], batch_size: int, shuffle: bool) -> DataLoader:
    mask = torch.zeros_like(y, dtype=torch.bool)
    for cls in classes:
        mask |= y == cls
    idx = torch.where(mask)[0]
    ds = TensorDataset(x[idx], y[idx])
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def evaluate(model: nn.Module, x_test: torch.Tensor, y_test: torch.Tensor, tasks: list[list[int]], seen: int, batch_size: int, device: torch.device) -> list[float]:
    model.eval()
    accs = []
    with torch.no_grad():
        for tid in range(seen):
            loader = subset_loader(x_test, y_test, tasks[tid], batch_size, False)
            correct = total = 0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb).argmax(dim=1)
                correct += (pred == yb).sum().item()
                total += yb.numel()
            accs.append(100.0 * correct / max(total, 1))
    return accs


def geometry(weight: torch.Tensor) -> dict[str, float]:
    w = weight.detach().float().cpu()
    sv = torch.linalg.svdvals(w)
    p = sv / (sv.sum() + 1e-12)
    eff_rank = float(torch.exp(-(p * torch.log(p + 1e-12)).sum()).item())
    wn = F.normalize(w, dim=1)
    cos = wn @ wn.T
    offdiag = cos[~torch.eye(cos.shape[0], dtype=torch.bool)]
    return {"effective_rank": eff_rank, "mean_abs_offdiag_cosine": float(offdiag.abs().mean().item())}


def run_one(args: argparse.Namespace, method: str, seed: int, device: torch.device) -> FlowerRun:
    set_seed(seed)
    x_train, y_train, x_test, y_test = extract_features(args, device)
    tasks = make_tasks(y_train, args.num_classes, args.classes_per_task, seed)
    model = LinearHead(x_train.shape[1], args.num_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay if method == "l2" else 0.0)
    teacher: LinearHead | None = None
    acc_matrix = []

    for tid, task_classes in enumerate(tasks):
        loader = subset_loader(x_train, y_train, task_classes, args.batch_size, True)
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
        acc_matrix.append(evaluate(model, x_test, y_test, tasks, tid + 1, args.batch_size, device))
        if method == "biocs_kd":
            teacher = LinearHead(x_train.shape[1], args.num_classes).to(device)
            teacher.load_state_dict(model.state_dict())
            teacher.eval()

    final = acc_matrix[-1]
    forgetting = []
    for task_idx in range(len(tasks) - 1):
        best = max(row[task_idx] for row in acc_matrix[task_idx:])
        forgetting.append(best - final[task_idx])
    geo = geometry(model.classifier.weight)
    return FlowerRun(
        method=method,
        seed=seed,
        avg_accuracy=float(np.mean(final)),
        avg_forgetting=float(np.mean(forgetting)) if forgetting else 0.0,
        **geo,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="./data")
    p.add_argument("--cache", default="data/flowers102_resnet18_features.pt")
    p.add_argument("--output_dir", default="results/finegrained_flowers_probe_20260429")
    p.add_argument("--methods", nargs="+", default=["baseline", "l2", "biocs", "biocs_kd"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--num_classes", type=int, default=102)
    p.add_argument("--classes_per_task", type=int, default=6)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--feature_batch_size", type=int, default=128)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--lambda_sp", type=float, default=2.0)
    p.add_argument("--lambda_kd", type=float, default=2.0)
    p.add_argument("--kd_temperature", type=float, default=3.0)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rows = []
    for seed in args.seeds:
        for method in args.methods:
            print(f"[flowers] method={method} seed={seed}", flush=True)
            row = run_one(args, method, seed, device)
            rows.append(row)
            with (outdir / "runs.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")
            print(json.dumps(asdict(row), ensure_ascii=False), flush=True)

    summary = {}
    for method in args.methods:
        items = [r for r in rows if r.method == method]
        summary[method] = {}
        for key in ["avg_accuracy", "avg_forgetting", "effective_rank", "mean_abs_offdiag_cosine"]:
            vals = np.asarray([getattr(r, key) for r in items], dtype=float)
            summary[method][key] = {"mean": float(vals.mean()), "std": float(vals.std())}
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved to {outdir}", flush=True)


if __name__ == "__main__":
    main()
