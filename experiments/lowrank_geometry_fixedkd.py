#!/usr/bin/env python3
"""Matched fixed-KD CUB low-rank geometry benchmark.

The executable has three commands:
  run       train one method/seed/configuration and save a structured record;
  select    select one coefficient per family on development seeds only;
  summarize summarize the frozen ten-seed confirmation without filtering.

Every arm uses the same frozen CUB ResNet-18 features, GELU bottleneck adapter,
cosine classifier, semantic class order, optimizer and training budget. KD is
on in every arm and matches the historical Figure 6 scaffold: the teacher is
the preceding task-boundary model, all logits are distilled at T=3 and
lambda_KD=2. Center-surround SSR and three geometry controls receive the same
four-coefficient development budget before a disjoint ten-seed paired
confirmation. All geometry terms act on the same seen classifier prototypes
and low-rank adapter basis.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from torch.utils.data import DataLoader, TensorDataset


PROTOCOL = "cub_lowrank_fixedkd_geometry_v1"
METHODS = ("kd", "kd_ssr", "kd_orthogonal", "kd_protodecor", "kd_spectral")
TUNED_METHODS = ("kd_ssr", "kd_orthogonal", "kd_protodecor", "kd_spectral")
METRICS = ("avg_accuracy", "avg_forgetting")


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def radial_kernel(distance: torch.Tensor, sigma: float) -> torch.Tensor:
    return torch.exp(-(distance.square()) / (2.0 * sigma * sigma))


def ssr_loss(
    weights: torch.Tensor,
    *,
    a_exc: float,
    a_inh: float,
    sigma_exc: float,
    sigma_inh: float,
) -> torch.Tensor:
    if weights.shape[0] < 2:
        return weights.sum() * 0.0
    directions = F.normalize(weights.float(), dim=1)
    cosine = torch.clamp(directions @ directions.T, -1.0, 1.0)
    distance = torch.sqrt(torch.clamp(1.0 - cosine, min=1e-8))
    pair_loss = (
        a_inh * radial_kernel(distance, sigma_inh)
        - a_exc * radial_kernel(distance, sigma_exc)
        + a_exc
        - a_inh
    )
    offdiag = ~torch.eye(weights.shape[0], dtype=torch.bool, device=weights.device)
    return pair_loss[offdiag].mean()


def orthogonal_loss(weights: torch.Tensor) -> torch.Tensor:
    """Penalize squared off-diagonal cosine similarity."""
    if weights.shape[0] < 2:
        return weights.sum() * 0.0
    directions = F.normalize(weights.float(), dim=1)
    cosine = directions @ directions.T
    offdiag = ~torch.eye(weights.shape[0], dtype=torch.bool, device=weights.device)
    return cosine[offdiag].square().mean()


def prototype_decorrelation_loss(weights: torch.Tensor) -> torch.Tensor:
    """Penalize squared correlations after centering every row."""
    if weights.shape[0] < 2 or weights.shape[1] < 2:
        return weights.sum() * 0.0
    centered = weights.float() - weights.float().mean(dim=1, keepdim=True)
    directions = F.normalize(centered, dim=1)
    correlation = directions @ directions.T
    offdiag = ~torch.eye(weights.shape[0], dtype=torch.bool, device=weights.device)
    return correlation[offdiag].square().mean()


def spectral_flatness_loss(weights: torch.Tensor) -> torch.Tensor:
    """KL divergence from a uniform singular-value spectrum."""
    if weights.shape[0] < 2:
        return weights.sum() * 0.0
    singular_values = torch.linalg.svdvals(weights.float())
    if singular_values.numel() < 2:
        return weights.sum() * 0.0
    probabilities = singular_values / (singular_values.sum() + 1e-8)
    uniform = torch.full_like(probabilities, 1.0 / probabilities.numel())
    return F.kl_div((probabilities + 1e-8).log(), uniform, reduction="sum")


def geometry_loss(
    method: str,
    weights: torch.Tensor,
    *,
    a_exc: float,
    a_inh: float,
    sigma_exc: float,
    sigma_inh: float,
) -> torch.Tensor:
    if method == "kd_ssr":
        return ssr_loss(
            weights,
            a_exc=a_exc,
            a_inh=a_inh,
            sigma_exc=sigma_exc,
            sigma_inh=sigma_inh,
        )
    if method == "kd_orthogonal":
        return orthogonal_loss(weights)
    if method == "kd_protodecor":
        return prototype_decorrelation_loss(weights)
    if method == "kd_spectral":
        return spectral_flatness_loss(weights)
    raise ValueError(f"unsupported geometry method: {method}")


class AdapterHead(nn.Module):
    def __init__(self, dim: int, num_classes: int, rank: int, tau: float, scale: float):
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        self.classifier = nn.Linear(dim, num_classes, bias=False)
        self.tau = tau
        self.scale = scale
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        return features + self.scale * self.up(F.gelu(self.down(features)))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        encoded = F.normalize(self.encode(features), dim=1)
        prototypes = F.normalize(self.classifier.weight, dim=1)
        return self.tau * F.linear(encoded, prototypes)

    def adapter_basis(self) -> torch.Tensor:
        return self.up.weight.T


def make_tasks(num_classes: int, classes_per_task: int, seed: int, order: str) -> list[list[int]]:
    classes = list(range(num_classes))
    if order == "random":
        random.Random(seed).shuffle(classes)
    return [classes[i : i + classes_per_task] for i in range(0, num_classes, classes_per_task)]


def subset_loader(
    features: torch.Tensor,
    labels: torch.Tensor,
    classes: list[int],
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    mask = torch.zeros_like(labels, dtype=torch.bool)
    for class_id in classes:
        mask |= labels == class_id
    indices = torch.where(mask)[0]
    return DataLoader(
        TensorDataset(features[indices], labels[indices]),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )


def task_accuracies(
    model: AdapterHead,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    tasks: list[list[int]],
    seen_tasks: int,
    batch_size: int,
    device: torch.device,
) -> list[float]:
    model.eval()
    accuracies: list[float] = []
    with torch.no_grad():
        for task_id in range(seen_tasks):
            allowed = tasks[task_id]
            loader = subset_loader(x_test, y_test, allowed, batch_size, False)
            correct = 0
            total = 0
            for features, labels in loader:
                logits = model(features.to(device))
                masked = torch.full_like(logits, -1e9)
                masked[:, allowed] = logits[:, allowed]
                predictions = masked.argmax(dim=1).cpu()
                correct += int((predictions == labels).sum())
                total += int(labels.numel())
            accuracies.append(100.0 * correct / max(total, 1))
    return accuracies


def trainable_snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().clone() for name, parameter in model.named_parameters() if parameter.requires_grad}


def zeros_like_snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: torch.zeros_like(parameter) for name, parameter in model.named_parameters() if parameter.requires_grad}


def quadratic_penalty(
    model: nn.Module,
    reference: dict[str, torch.Tensor],
    importance: dict[str, torch.Tensor],
) -> torch.Tensor:
    terms = [
        (importance[name] * (parameter - reference[name]).square()).sum()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and name in reference
    ]
    return torch.stack(terms).sum() if terms else next(model.parameters()).sum() * 0.0


def estimate_ewc_importance(
    model: AdapterHead,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    importance = zeros_like_snapshot(model)
    total = 0
    model.eval()
    for features, labels in loader:
        features, labels = features.to(device), labels.to(device)
        model.zero_grad(set_to_none=True)
        F.cross_entropy(model(features), labels).backward()
        batch_size = int(labels.numel())
        total += batch_size
        for name, parameter in model.named_parameters():
            if parameter.grad is not None and name in importance:
                importance[name].add_(parameter.grad.detach().square(), alpha=batch_size)
    for name in importance:
        importance[name].div_(max(total, 1))
    return importance


def estimate_mas_importance(
    model: AdapterHead,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    importance = zeros_like_snapshot(model)
    total = 0
    model.eval()
    for features, _ in loader:
        features = features.to(device)
        model.zero_grad(set_to_none=True)
        model(features).square().mean().backward()
        batch_size = int(features.shape[0])
        total += batch_size
        for name, parameter in model.named_parameters():
            if parameter.grad is not None and name in importance:
                importance[name].add_(parameter.grad.detach().abs(), alpha=batch_size)
    for name in importance:
        importance[name].div_(max(total, 1))
    return importance


def add_importance(
    cumulative: dict[str, torch.Tensor], current: dict[str, torch.Tensor]
) -> None:
    for name in cumulative:
        cumulative[name].add_(current[name])


def model_geometry(model: AdapterHead) -> dict[str, float]:
    classifier = model.classifier.weight.detach().float().cpu()
    singular_values = torch.linalg.svdvals(classifier)
    probabilities = singular_values / (singular_values.sum() + 1e-12)
    normalized = F.normalize(classifier, dim=1)
    cosine = normalized @ normalized.T
    offdiag = cosine[~torch.eye(cosine.shape[0], dtype=torch.bool)]

    basis = model.adapter_basis().detach().float().cpu()
    basis_normalized = F.normalize(basis, dim=1)
    basis_cosine = basis_normalized @ basis_normalized.T
    basis_offdiag = basis_cosine[~torch.eye(basis_cosine.shape[0], dtype=torch.bool)]
    return {
        "effective_rank": float(torch.exp(-(probabilities * torch.log(probabilities + 1e-12)).sum())),
        "prototype_abs_cosine": float(offdiag.abs().mean()),
        "adapter_basis_abs_cosine": float(basis_offdiag.abs().mean()),
    }


def run_one(args: argparse.Namespace) -> dict[str, Any]:
    if args.method not in METHODS:
        raise ValueError(f"unsupported method: {args.method}")
    if args.method == "kd" and args.coefficient != 0:
        raise ValueError(f"{args.method} coefficient must be zero")
    set_seed(args.seed)
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cache = Path(args.feature_cache)
    payload = torch.load(cache, map_location="cpu")
    x_train = payload["x_train"].float()
    y_train = payload["y_train"].long()
    x_test = payload["x_test"].float()
    y_test = payload["y_test"].long()
    tasks = make_tasks(args.num_classes, args.classes_per_task, args.seed, args.class_order)[: args.max_tasks]

    model = AdapterHead(x_train.shape[1], args.num_classes, args.rank, args.tau, args.adapter_scale).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    teacher: AdapterHead | None = None
    accuracy_matrix: list[list[float]] = []

    use_geometry = args.method != "kd"
    classifier_coefficient = args.coefficient if use_geometry else 0.0
    adapter_coefficient = args.geometry_adapter_ratio * args.coefficient if use_geometry else 0.0
    for task_id, classes in enumerate(tasks):
        seen_classes = [class_id for task in tasks[: task_id + 1] for class_id in task]
        loader = subset_loader(x_train, y_train, classes, args.batch_size, True)
        for _ in range(args.epochs):
            model.train()
            for features, labels in loader:
                features, labels = features.to(device), labels.to(device)
                logits = model(features)
                loss = F.cross_entropy(logits, labels)
                if use_geometry:
                    loss = loss + classifier_coefficient * geometry_loss(
                        args.method,
                        model.classifier.weight[seen_classes],
                        a_exc=args.a_exc,
                        a_inh=args.a_inh,
                        sigma_exc=args.sigma_exc,
                        sigma_inh=args.sigma_inh,
                    )
                    loss = loss + adapter_coefficient * geometry_loss(
                        args.method,
                        model.adapter_basis(),
                        a_exc=args.a_exc,
                        a_inh=args.a_inh,
                        sigma_exc=args.sigma_exc,
                        sigma_inh=args.sigma_inh,
                    )
                if teacher is not None:
                    with torch.no_grad():
                        target_logits = teacher(features)
                    current_logits = logits
                    temperature = args.kd_temperature
                    distillation = F.kl_div(
                        F.log_softmax(current_logits / temperature, dim=1),
                        F.softmax(target_logits / temperature, dim=1),
                        reduction="batchmean",
                    )
                    loss = loss + args.kd_coefficient * temperature * temperature * distillation

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

        accuracy_matrix.append(
            task_accuracies(model, x_test, y_test, tasks, task_id + 1, args.batch_size, device)
        )

        teacher = deepcopy(model).eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)

    final = accuracy_matrix[-1]
    forgetting = []
    for task_id in range(len(tasks) - 1):
        best = max(row[task_id] for row in accuracy_matrix[task_id:])
        forgetting.append(best - final[task_id])
    result = {
        "status": "complete",
        "protocol": PROTOCOL,
        "stage": args.stage,
        "method": args.method,
        "seed": args.seed,
        "coefficient": args.coefficient,
        "rank": args.rank,
        "avg_accuracy": float(np.mean(final)),
        "avg_forgetting": float(np.mean(forgetting)) if forgetting else 0.0,
        "accuracy_matrix": accuracy_matrix,
        "geometry": model_geometry(model),
        "model": {
            "feature_backbone": "ImageNet ResNet-18 (frozen cached features)",
            "adapter": "GELU bottleneck residual adapter",
            "rank": args.rank,
            "adapter_scale": args.adapter_scale,
            "classifier": "cosine classifier",
            "tau": args.tau,
        },
        "data": {
            "dataset": "CUB-200-2011",
            "feature_cache": str(cache),
            "feature_cache_sha256": file_sha256(cache),
            "num_classes": args.num_classes,
            "classes_per_task": args.classes_per_task,
            "tasks": tasks,
            "task_order": args.class_order,
        },
        "optimization": {
            "epochs_per_task": args.epochs,
            "batch_size": args.batch_size,
            "optimizer": "AdamW",
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "gradient_clip": args.grad_clip,
        },
        "regularizer": {
            "kd_enabled": True,
            "kd_scope": "all_logits_from_preceding_task_boundary_teacher",
            "kd_coefficient": args.kd_coefficient,
            "kd_temperature": args.kd_temperature,
            "geometry_family": args.method.removeprefix("kd_") if use_geometry else "none",
            "geometry_scope": "seen_classifier_prototypes_and_adapter_basis",
            "classifier_coefficient": classifier_coefficient,
            "adapter_coefficient": adapter_coefficient,
            "ssr_enabled": args.method == "kd_ssr",
            "ssr_mapping": "shared cosine" if args.method == "kd_ssr" else "none",
            "ssr_kernel": "Gaussian center-surround" if args.method == "kd_ssr" else "none",
            "ssr_sigma_exc": args.sigma_exc if args.method == "kd_ssr" else None,
            "ssr_sigma_inh": args.sigma_inh if args.method == "kd_ssr" else None,
        },
        "peak_cuda_memory_mb": (
            float(torch.cuda.max_memory_allocated(device) / (1024 ** 2)) if device.type == "cuda" else 0.0
        ),
    }
    atomic_json(Path(args.output), result)
    return result


def load_records(root: Path, stage: str) -> list[dict[str, Any]]:
    records = []
    for path in sorted((root / stage).glob("**/result.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        record["_path"] = str(path)
        records.append(record)
    return records


def select(args: argparse.Namespace) -> dict[str, Any]:
    records = load_records(Path(args.result_root), "development")
    seeds = set(args.seeds)
    kd_by_seed = {
        int(row["seed"]): row
        for row in records
        if row["method"] == "kd" and int(row["seed"]) in seeds
    }
    if set(kd_by_seed) != seeds:
        raise RuntimeError(f"missing paired KD development records: {sorted(seeds - set(kd_by_seed))}")
    candidates: dict[str, list[dict[str, Any]]] = {}
    for method in TUNED_METHODS:
        rows = [row for row in records if row["method"] == method and row["seed"] in seeds]
        by_coefficient: dict[float, list[dict[str, Any]]] = {}
        for row in rows:
            by_coefficient.setdefault(float(row["coefficient"]), []).append(row)
        method_candidates = []
        for coefficient, items in sorted(by_coefficient.items()):
            if len(items) != len(seeds):
                continue
            items = sorted(items, key=lambda item: int(item["seed"]))
            aa = np.asarray([item["avg_accuracy"] for item in items])
            af = np.asarray([item["avg_forgetting"] for item in items])
            aa_gain = np.asarray([
                item["avg_accuracy"] - kd_by_seed[int(item["seed"])]["avg_accuracy"]
                for item in items
            ])
            af_reduction = np.asarray([
                kd_by_seed[int(item["seed"])]["avg_forgetting"] - item["avg_forgetting"]
                for item in items
            ])
            method_candidates.append(
                {
                    "coefficient": coefficient,
                    "mean_aa": float(aa.mean()),
                    "mean_af": float(af.mean()),
                    "mean_aa_gain_vs_kd": float(aa_gain.mean()),
                    "mean_af_reduction_vs_kd": float(af_reduction.mean()),
                    "paired_nonregression_gate": bool(aa_gain.mean() > 0 and af_reduction.mean() > 0),
                    "selection_utility": float(aa_gain.mean() + af_reduction.mean()),
                    "seeds": sorted(seeds),
                }
            )
        if not method_candidates:
            raise RuntimeError(f"no complete development candidates for {method}")
        candidates[method] = method_candidates
    selected = {
        method: sorted(
            items,
            key=lambda item: (
                not item["paired_nonregression_gate"],
                -item["selection_utility"],
                item["coefficient"],
            ),
        )[0]
        for method, items in candidates.items()
    }
    kd_reference = {
        "mean_aa": float(np.mean([row["avg_accuracy"] for row in kd_by_seed.values()])),
        "mean_af": float(np.mean([row["avg_forgetting"] for row in kd_by_seed.values()])),
    }
    ssr = selected["kd_ssr"]
    comparator_margins = {
        method: {
            "aa_gain_pp": float(ssr["mean_aa"] - row["mean_aa"]),
            "af_reduction_pp": float(row["mean_af"] - ssr["mean_af"]),
        }
        for method, row in selected.items()
        if method != "kd_ssr"
    }
    confirmation_gate = {
        "definition": (
            "Selected KD+SSR must improve mean AA and reduce mean AF versus KD, "
            "and exceed every equally tuned geometry control on both endpoints"
        ),
        "passed": bool(
            ssr["mean_aa_gain_vs_kd"] > 0
            and ssr["mean_af_reduction_vs_kd"] > 0
            and all(
                margin["aa_gain_pp"] > 0 and margin["af_reduction_pp"] > 0
                for margin in comparator_margins.values()
            )
        ),
        "ssr_vs_kd": {
            "aa_gain_pp": ssr["mean_aa_gain_vs_kd"],
            "af_reduction_pp": ssr["mean_af_reduction_vs_kd"],
        },
        "ssr_vs_selected_geometry_controls": comparator_margins,
    }
    payload = {
        "status": "selected",
        "protocol": PROTOCOL,
        "development_seeds": sorted(seeds),
        "selection_rule": "prefer positive paired mean AA gain and AF reduction versus KD, then maximize their sum; ties choose smaller coefficient",
        "candidate_budget_per_tuned_family": len(next(iter(candidates.values()))),
        "kd_reference": kd_reference,
        "candidates": candidates,
        "selected": selected,
        "confirmation_gate": confirmation_gate,
        "confirmation_was_not_read": True,
    }
    atomic_json(Path(args.output), payload)
    return payload


def paired_interval(values: np.ndarray) -> tuple[float, float, float]:
    mean = float(values.mean())
    if len(values) < 2:
        return mean, math.nan, math.nan
    half = float(stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / math.sqrt(len(values)))
    return mean, mean - half, mean + half


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.result_root)
    records = load_records(root, "confirmation")
    expected_seeds = sorted(set(args.seeds))
    expected_methods = list(METHODS)
    indexed = {(row["method"], int(row["seed"])): row for row in records}
    missing = [
        (method, seed)
        for method in expected_methods
        for seed in expected_seeds
        if (method, seed) not in indexed
    ]
    if missing:
        raise RuntimeError(f"missing confirmation records: {missing}")

    methods: dict[str, Any] = {}
    for method in expected_methods:
        items = [indexed[(method, seed)] for seed in expected_seeds]
        methods[method] = {
            metric: {
                "mean": float(np.mean([row[metric] for row in items])),
                "sd": float(np.std([row[metric] for row in items], ddof=1)),
            }
            for metric in METRICS
        }

    contrasts: dict[str, Any] = {}
    for reference, treatments in {
        "kd": ("kd_ssr", "kd_orthogonal", "kd_protodecor", "kd_spectral"),
        "kd_orthogonal": ("kd_ssr",),
        "kd_protodecor": ("kd_ssr",),
        "kd_spectral": ("kd_ssr",),
    }.items():
        for treatment in treatments:
            aa = np.asarray([
                indexed[(treatment, seed)]["avg_accuracy"] - indexed[(reference, seed)]["avg_accuracy"]
                for seed in expected_seeds
            ])
            af_reduction = np.asarray([
                indexed[(reference, seed)]["avg_forgetting"] - indexed[(treatment, seed)]["avg_forgetting"]
                for seed in expected_seeds
            ])
            aa_mean, aa_low, aa_high = paired_interval(aa)
            af_mean, af_low, af_high = paired_interval(af_reduction)
            contrasts[f"{treatment}_vs_{reference}"] = {
                "aa_gain_pp": {"mean": aa_mean, "ci95": [aa_low, aa_high]},
                "af_reduction_pp": {"mean": af_mean, "ci95": [af_low, af_high]},
                "dual_favorable_seeds": int(np.sum((aa > 0) & (af_reduction > 0))),
                "n": len(expected_seeds),
            }

    payload = {
        "status": "complete",
        "protocol": PROTOCOL,
        "rank": int(records[0]["rank"]),
        "confirmation_seeds": expected_seeds,
        "all_confirmation_records_retained": True,
        "methods": methods,
        "contrasts": contrasts,
        "main_text_gate": {
            "definition": "KD+SSR must have positive 95% CIs for AA gain and AF reduction against KD and every matched geometry control",
            "passed": all(
                contrasts[f"kd_ssr_vs_{reference}"]["aa_gain_pp"]["ci95"][0] > 0
                and contrasts[f"kd_ssr_vs_{reference}"]["af_reduction_pp"]["ci95"][0] > 0
                for reference in ("kd", "kd_orthogonal", "kd_protodecor", "kd_spectral")
            ),
        },
    }
    atomic_json(Path(args.output), payload)

    csv_path = Path(args.csv_output)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["rank", "method", "seed", "coefficient", "avg_accuracy", "avg_forgetting", "result_file"],
        )
        writer.writeheader()
        for method in expected_methods:
            for seed in expected_seeds:
                row = indexed[(method, seed)]
                writer.writerow(
                    {
                        "rank": row["rank"],
                        "method": method,
                        "seed": seed,
                        "coefficient": row["coefficient"],
                        "avg_accuracy": row["avg_accuracy"],
                        "avg_forgetting": row["avg_forgetting"],
                        "result_file": row["_path"],
                    }
                )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--stage", choices=("canary", "development", "confirmation"), required=True)
    run_parser.add_argument("--method", choices=METHODS, required=True)
    run_parser.add_argument("--seed", type=int, required=True)
    run_parser.add_argument("--coefficient", type=float, required=True)
    run_parser.add_argument("--feature-cache", required=True)
    run_parser.add_argument("--output", required=True)
    run_parser.add_argument("--device", default="cuda")
    run_parser.add_argument("--cpu-threads", type=int, default=4)
    run_parser.add_argument("--num-classes", type=int, default=200)
    run_parser.add_argument("--classes-per-task", type=int, default=10)
    run_parser.add_argument("--class-order", choices=("semantic", "random"), default="semantic")
    run_parser.add_argument("--max-tasks", type=int, default=20)
    run_parser.add_argument("--epochs", type=int, default=40)
    run_parser.add_argument("--batch-size", type=int, default=128)
    run_parser.add_argument("--rank", type=int, default=32)
    run_parser.add_argument("--adapter-scale", type=float, default=0.5)
    run_parser.add_argument("--tau", type=float, default=12.0)
    run_parser.add_argument("--lr", type=float, default=1e-3)
    run_parser.add_argument("--weight-decay", type=float, default=1e-3)
    run_parser.add_argument("--grad-clip", type=float, default=5.0)
    run_parser.add_argument("--geometry-adapter-ratio", type=float, default=0.2)
    run_parser.add_argument("--a-exc", type=float, default=1.0)
    run_parser.add_argument("--a-inh", type=float, default=0.8)
    run_parser.add_argument("--sigma-exc", type=float, default=0.55)
    run_parser.add_argument("--sigma-inh", type=float, default=1.25)
    run_parser.add_argument("--kd-coefficient", type=float, default=2.0)
    run_parser.add_argument("--kd-temperature", type=float, default=3.0)

    select_parser = subparsers.add_parser("select")
    select_parser.add_argument("--result-root", required=True)
    select_parser.add_argument("--seeds", nargs="+", type=int, required=True)
    select_parser.add_argument("--output", required=True)

    summary_parser = subparsers.add_parser("summarize")
    summary_parser.add_argument("--result-root", required=True)
    summary_parser.add_argument("--seeds", nargs="+", type=int, required=True)
    summary_parser.add_argument("--output", required=True)
    summary_parser.add_argument("--csv-output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "run":
        result = run_one(args)
    elif args.command == "select":
        result = select(args)
    else:
        result = summarize(args)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
