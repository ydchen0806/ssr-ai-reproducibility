#!/usr/bin/env python3
"""Matched task-loss versus task-loss+SSR low-rank adapter experiment.

The image encoder is frozen and evaluated once per dataset.  Each paired run
then trains the same low-rank residual adapter and classifier from the same
initial state, with the same class stream, minibatch order, optimizer and
number of updates.  The sole treatment difference is the SSR term.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.lowrank_adapter_mapping import AdapterHead, biocs_loss  # noqa: E402


DATASET_SPECS = {
    "cifar100": {"classes": 100, "tasks": 10},
    "flowers102": {"classes": 102, "tasks": 10},
    "food101": {"classes": 101, "tasks": 10},
    "oxfordiiitpet": {"classes": 37, "tasks": 7},
    "dtd": {"classes": 47, "tasks": 8},
}
METHODS = ("task", "task_ssr")


@dataclass(frozen=True)
class RunResult:
    dataset: str
    method: str
    seed: int
    rank: int
    ssr_scale: float
    avg_accuracy: float
    avg_forgetting: float
    cil_last_accuracy: float
    effective_rank: float
    prototype_overlap: float
    adapter_basis_overlap: float
    optimizer_steps: int
    initial_state_sha256: str
    schedule_sha256: str
    feature_cache_sha256: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_state_dict(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _dataset_splits(name: str, data_root: Path, transform, download: bool):
    from torchvision import datasets

    if name == "cifar100":
        train = datasets.CIFAR100(data_root, train=True, transform=transform, download=download)
        test = datasets.CIFAR100(data_root, train=False, transform=transform, download=download)
    elif name == "flowers102":
        train = ConcatDataset(
            [
                datasets.Flowers102(
                    data_root, split="train", transform=transform, download=download
                ),
                datasets.Flowers102(data_root, split="val", transform=transform, download=download),
            ]
        )
        test = datasets.Flowers102(data_root, split="test", transform=transform, download=download)
    elif name == "food101":
        train = datasets.Food101(data_root, split="train", transform=transform, download=download)
        test = datasets.Food101(data_root, split="test", transform=transform, download=download)
    elif name == "oxfordiiitpet":
        train = datasets.OxfordIIITPet(
            data_root,
            split="trainval",
            target_types="category",
            transform=transform,
            download=download,
        )
        test = datasets.OxfordIIITPet(
            data_root,
            split="test",
            target_types="category",
            transform=transform,
            download=download,
        )
    elif name == "dtd":
        train = ConcatDataset(
            [
                datasets.DTD(
                    data_root,
                    split="train",
                    partition=1,
                    transform=transform,
                    download=download,
                ),
                datasets.DTD(
                    data_root,
                    split="val",
                    partition=1,
                    transform=transform,
                    download=download,
                ),
            ]
        )
        test = datasets.DTD(
            data_root,
            split="test",
            partition=1,
            transform=transform,
            download=download,
        )
    else:
        raise ValueError(f"Unknown dataset {name!r}; choose from {sorted(DATASET_SPECS)}")
    return train, test


@torch.inference_mode()
def _extract_features(
    encoder: nn.Module,
    dataset: Dataset,
    batch_size: int,
    workers: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    features: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for images, targets in loader:
        encoded = encoder(images.to(device, non_blocking=True)).float().cpu()
        features.append(encoded)
        labels.append(torch.as_tensor(targets).long().cpu())
    return torch.cat(features), torch.cat(labels)


def prepare_feature_cache(args: argparse.Namespace) -> Path:
    from torchvision.models import ResNet18_Weights, resnet18

    dataset = args.dataset.lower()
    if dataset not in DATASET_SPECS:
        raise ValueError(f"Unknown dataset {dataset!r}")
    cache = Path(args.feature_cache)
    if cache.exists() and not args.force_rebuild:
        return cache

    cache.parent.mkdir(parents=True, exist_ok=True)
    weights = ResNet18_Weights.IMAGENET1K_V1
    transform = weights.transforms()
    train_set, test_set = _dataset_splits(
        dataset,
        Path(args.data_root),
        transform,
        args.download,
    )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    encoder = resnet18(weights=weights)
    encoder.fc = nn.Identity()
    encoder.eval().to(device)
    x_train, y_train = _extract_features(
        encoder, train_set, args.feature_batch_size, args.workers, device
    )
    x_test, y_test = _extract_features(
        encoder, test_set, args.feature_batch_size, args.workers, device
    )

    classes = sorted(set(y_train.tolist()) | set(y_test.tolist()))
    remap = {old: new for new, old in enumerate(classes)}
    y_train = torch.tensor([remap[int(value)] for value in y_train], dtype=torch.long)
    y_test = torch.tensor([remap[int(value)] for value in y_test], dtype=torch.long)
    expected = DATASET_SPECS[dataset]["classes"]
    if len(classes) != expected:
        raise RuntimeError(f"{dataset} exposed {len(classes)} classes; expected {expected}")
    if set(y_train.tolist()) != set(y_test.tolist()):
        raise RuntimeError(f"{dataset} train/test class sets differ")

    payload = {
        "schema_version": 1,
        "dataset": dataset,
        "backbone": "torchvision/resnet18-imagenet1k-v1",
        "x_train": x_train.contiguous(),
        "y_train": y_train.contiguous(),
        "x_test": x_test.contiguous(),
        "y_test": y_test.contiguous(),
        "class_remap": remap,
    }
    temporary = cache.with_name(f".{cache.name}.tmp.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, cache)
    return cache


def load_feature_cache(path: Path, dataset: str) -> dict:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    required = {"dataset", "x_train", "y_train", "x_test", "y_test"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"Feature cache is missing keys: {sorted(missing)}")
    if payload["dataset"] != dataset:
        raise ValueError(
            f"Feature cache dataset={payload['dataset']!r}, requested {dataset!r}"
        )
    return payload


def make_tasks(num_classes: int, num_tasks: int, seed: int) -> list[list[int]]:
    order = list(range(num_classes))
    random.Random(seed).shuffle(order)
    return [list(map(int, part)) for part in np.array_split(order, num_tasks)]


def schedule_sha256(tasks: list[list[int]], seed: int, epochs: int) -> str:
    payload = {
        "tasks": tasks,
        "loader_seeds": [seed + 100_003 * task for task in range(len(tasks))],
        "epochs": epochs,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def subset_indices(labels: torch.Tensor, classes: Iterable[int]) -> torch.Tensor:
    mask = torch.zeros(labels.shape[0], dtype=torch.bool)
    for cls in classes:
        mask |= labels == int(cls)
    return torch.where(mask)[0]


def task_loader(
    x: torch.Tensor,
    y: torch.Tensor,
    classes: list[int],
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    indices = subset_indices(y, classes)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(x[indices], y[indices]),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
    )


def masked_prediction(logits: torch.Tensor, allowed: list[int]) -> torch.Tensor:
    masked = torch.full_like(logits, -torch.inf)
    masked[:, allowed] = logits[:, allowed]
    return masked.argmax(dim=1)


@torch.inference_mode()
def evaluate_accuracy(
    model: AdapterHead,
    x: torch.Tensor,
    y: torch.Tensor,
    classes: list[int],
    batch_size: int,
    device: torch.device,
) -> float:
    model.eval()
    loader = task_loader(x, y, classes, batch_size, False, 0)
    correct = total = 0
    for xb, yb in loader:
        yb = yb.to(device)
        prediction = masked_prediction(model(xb.to(device)), classes)
        correct += int((prediction == yb).sum())
        total += int(yb.numel())
    return 100.0 * correct / max(total, 1)


def geometry(model: AdapterHead) -> tuple[float, float, float]:
    prototypes = F.normalize(model.classifier.weight.detach().float(), dim=1)
    singular = torch.linalg.svdvals(prototypes)
    mass = singular.square()
    mass = mass / mass.sum().clamp_min(1e-12)
    effective_rank = torch.exp(
        -(mass * mass.clamp_min(1e-12).log()).sum()
    ).item()
    cosine = prototypes @ prototypes.T
    offdiag = cosine[~torch.eye(cosine.size(0), dtype=torch.bool, device=cosine.device)]

    basis = F.normalize(model.adapter_basis().detach().float(), dim=1)
    basis_cosine = basis @ basis.T
    basis_offdiag = basis_cosine[
        ~torch.eye(basis_cosine.size(0), dtype=torch.bool, device=basis_cosine.device)
    ]
    return (
        float(effective_rank),
        float(offdiag.abs().mean()),
        float(basis_offdiag.abs().mean()),
    )


def run_one(
    args: argparse.Namespace,
    payload: dict,
    method: str,
    feature_hash: str,
    device: torch.device,
) -> RunResult:
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}")
    set_seed(args.seed)
    x_train = payload["x_train"].float()
    y_train = payload["y_train"].long()
    x_test = payload["x_test"].float()
    y_test = payload["y_test"].long()
    num_classes = len(torch.unique(y_train))
    tasks = make_tasks(num_classes, args.num_tasks, args.seed)
    schedule_hash = schedule_sha256(tasks, args.seed, args.epochs)

    model = AdapterHead(
        x_train.shape[1],
        num_classes,
        args.rank,
        args.tau,
        args.adapter_scale,
    ).to(device)
    initial_hash = sha256_state_dict(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    accuracy_matrix: list[list[float]] = []
    cil_curve: list[float] = []
    optimizer_steps = 0
    seen_classes: list[int] = []

    for task_id, classes in enumerate(tasks):
        seen_classes.extend(classes)
        loader_seed = args.seed + 100_003 * task_id
        loader = task_loader(
            x_train,
            y_train,
            classes,
            args.batch_size,
            True,
            loader_seed,
        )
        for _ in range(args.epochs):
            model.train()
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                loss = F.cross_entropy(model(xb), yb)
                if method == "task_ssr":
                    kernel = (
                        args.a_exc,
                        args.a_inh,
                        args.sigma_exc,
                        args.sigma_inh,
                        args.kernel_family,
                    )
                    if args.lambda_cls > 0 and len(seen_classes) > 1:
                        loss = loss + args.ssr_scale * args.lambda_cls * biocs_loss(
                            model.classifier.weight[seen_classes],
                            *kernel,
                            distance_metric=args.distance_metric,
                        )
                    if args.lambda_adapter > 0 and args.rank > 1:
                        loss = loss + args.ssr_scale * args.lambda_adapter * biocs_loss(
                            model.adapter_basis(),
                            *kernel,
                            distance_metric=args.distance_metric,
                        )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer_steps += 1

        accuracy_matrix.append(
            [
                evaluate_accuracy(
                    model, x_test, y_test, tasks[index], args.batch_size, device
                )
                for index in range(task_id + 1)
            ]
        )
        cil_curve.append(
            evaluate_accuracy(
                model, x_test, y_test, seen_classes, args.batch_size, device
            )
        )

    final = accuracy_matrix[-1]
    forgetting = [
        max(row[task] for row in accuracy_matrix[task:]) - final[task]
        for task in range(len(tasks) - 1)
    ]
    effective_rank, prototype_overlap, adapter_overlap = geometry(model)
    return RunResult(
        dataset=args.dataset,
        method=method,
        seed=args.seed,
        rank=args.rank,
        ssr_scale=args.ssr_scale,
        avg_accuracy=float(np.mean(final)),
        avg_forgetting=float(np.mean(forgetting)) if forgetting else 0.0,
        cil_last_accuracy=float(cil_curve[-1]),
        effective_rank=effective_rank,
        prototype_overlap=prototype_overlap,
        adapter_basis_overlap=adapter_overlap,
        optimizer_steps=optimizer_steps,
        initial_state_sha256=initial_hash,
        schedule_sha256=schedule_hash,
        feature_cache_sha256=feature_hash,
    )


def paired_record(task: RunResult, treated: RunResult) -> dict:
    for field in (
        "dataset",
        "seed",
        "rank",
        "optimizer_steps",
        "initial_state_sha256",
        "schedule_sha256",
        "feature_cache_sha256",
    ):
        if getattr(task, field) != getattr(treated, field):
            raise RuntimeError(f"Unmatched pair: {field} differs")
    return {
        "schema_version": 1,
        "comparison": "task+SSR - task",
        "control": asdict(task),
        "treatment": asdict(treated),
        "delta": {
            "avg_accuracy": treated.avg_accuracy - task.avg_accuracy,
            "avg_forgetting_reduction": task.avg_forgetting - treated.avg_forgetting,
            "cil_last_accuracy": treated.cil_last_accuracy - task.cil_last_accuracy,
            "effective_rank": treated.effective_rank - task.effective_rank,
            "prototype_overlap_reduction": task.prototype_overlap - treated.prototype_overlap,
            "adapter_basis_overlap_reduction": (
                task.adapter_basis_overlap - treated.adapter_basis_overlap
            ),
        },
        "matched_budget": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASET_SPECS), required=True)
    parser.add_argument("--data-root", default=str(ROOT / "data" / "torchvision"))
    parser.add_argument("--feature-cache")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--feature-batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=4101)
    parser.add_argument("--num-tasks", type=int)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--adapter-scale", type=float, default=0.5)
    parser.add_argument("--tau", type=float, default=12.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lambda-cls", type=float, default=1.0)
    parser.add_argument("--lambda-adapter", type=float, default=0.2)
    parser.add_argument("--ssr-scale", type=float, default=1.0)
    parser.add_argument("--a-exc", type=float, default=1.0)
    parser.add_argument("--a-inh", type=float, default=0.8)
    parser.add_argument("--sigma-exc", type=float, default=0.2)
    parser.add_argument("--sigma-inh", type=float, default=0.5)
    parser.add_argument(
        "--kernel-family",
        choices=["gaussian", "laplace", "cauchy", "inverse"],
        default="gaussian",
    )
    parser.add_argument(
        "--distance-metric", choices=["cosine", "projective"], default="cosine"
    )
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.feature_cache is None:
        args.feature_cache = str(
            Path(args.data_root) / "feature_cache" / f"{args.dataset}_resnet18.pt"
        )
    if args.num_tasks is None:
        args.num_tasks = DATASET_SPECS[args.dataset]["tasks"]
    for name in ("epochs", "batch_size", "rank", "num_tasks"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.ssr_scale <= 0:
        parser.error("--ssr-scale must be positive")
    return args


def main() -> None:
    args = parse_args()
    cache = Path(args.feature_cache)
    if args.prepare_only or not cache.exists():
        cache = prepare_feature_cache(args)
    if args.prepare_only:
        print(json.dumps({"feature_cache": str(cache), "sha256": sha256_file(cache)}))
        return

    payload = load_feature_cache(cache, args.dataset)
    feature_hash = sha256_file(cache)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    task = run_one(args, payload, "task", feature_hash, device)
    treated = run_one(args, payload, "task_ssr", feature_hash, device)
    record = paired_record(task, treated)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / f"pair.json.tmp.{os.getpid()}"
    temporary.write_text(json.dumps(record, indent=2), encoding="utf-8")
    os.replace(temporary, output / "pair.json")
    (output / "args.json").write_text(
        json.dumps(vars(args), indent=2), encoding="utf-8"
    )
    print(json.dumps(record["delta"], sort_keys=True))


if __name__ == "__main__":
    main()
