#!/usr/bin/env python3
"""Functional-space SSR benchmark on low-rank continual adapters for Fig. 6.

The executable has four commands:
  run       train one method/seed/configuration and save a structured record;
  select    select one coefficient per family on development seeds only;
  summarize summarize the frozen ten-seed confirmation without filtering.
  robustness summarize a frozen full-grid replication without selecting cells.

Every arm uses the same frozen ResNet-18 features, GELU bottleneck adapter,
cosine classifier, class order, optimizer, centroid replay and training
budget. KD is on in every arm. Functional SSR acts on the adapted responses of
seen-class feature centroids and on classifier prototypes. At task boundaries,
it also preserves the teacher's center--surround-weighted response topology on
old-class centroids. It therefore constrains the function represented by both
low-rank factors rather than one non-identifiable factor. EWC, MAS and SI cover
the same trainable adapter and classifier tensors.
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


PROTOCOL = "fig6_lowrank_fixedkd_functionalspace_v16"
METHODS = ("kd", "kd_ssr", "kd_ewc", "kd_mas", "kd_si")
TUNED_METHODS = ("kd_ssr", "kd_ewc", "kd_mas", "kd_si")
METRICS = ("avg_accuracy", "avg_forgetting", "final_old_accuracy", "final_new_accuracy")
REGULARIZED_PARAMETER_NAMES = frozenset({"down.weight", "up.weight", "classifier.weight"})


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


def functional_topology_anchor(
    current: torch.Tensor,
    reference: torch.Tensor,
    *,
    a_exc: float,
    a_inh: float,
    sigma_exc: float,
    sigma_inh: float,
) -> torch.Tensor:
    """Preserve reference response geometry where the SSR kernel is active."""
    if current.shape[0] < 2:
        return current.sum() * 0.0
    current_directions = F.normalize(current.float(), dim=1)
    reference_directions = F.normalize(reference.detach().float(), dim=1)
    current_cosine = torch.clamp(current_directions @ current_directions.T, -1.0, 1.0)
    reference_cosine = torch.clamp(reference_directions @ reference_directions.T, -1.0, 1.0)
    reference_distance = torch.sqrt(torch.clamp(1.0 - reference_cosine, min=1e-8))
    interaction = torch.abs(
        a_exc * radial_kernel(reference_distance, sigma_exc)
        - a_inh * radial_kernel(reference_distance, sigma_inh)
    )
    offdiag = ~torch.eye(current.shape[0], dtype=torch.bool, device=current.device)
    pair_weights = interaction[offdiag]
    pair_weights = pair_weights / (pair_weights.mean().detach() + 1e-8)
    pair_loss = (
        pair_weights * (current_cosine[offdiag] - reference_cosine[offdiag]).square()
    ).mean()

    # Pairwise geometry is globally rotation-invariant; this weighted node term
    # keeps the response frame aligned with the task-boundary teacher.
    node_weights = interaction.masked_fill(~offdiag, 0.0).mean(dim=1)
    node_weights = node_weights / (node_weights.mean().detach() + 1e-8)
    node_loss = (
        node_weights * (1.0 - (current_directions * reference_directions).sum(dim=1))
    ).mean()
    return pair_loss + node_loss


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
    generator_seed: int,
) -> DataLoader:
    mask = torch.zeros_like(labels, dtype=torch.bool)
    for class_id in classes:
        mask |= labels == class_id
    indices = torch.where(mask)[0]
    generator = torch.Generator().manual_seed(generator_seed)
    return DataLoader(
        TensorDataset(features[indices], labels[indices]),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
    )


def class_feature_centroids(
    features: torch.Tensor, labels: torch.Tensor, num_classes: int
) -> torch.Tensor:
    """Compute one deterministic training-feature centroid per class."""
    centroids = torch.zeros(num_classes, features.shape[1], dtype=features.dtype)
    counts = torch.zeros(num_classes, dtype=features.dtype)
    centroids.index_add_(0, labels, features)
    counts.index_add_(0, labels, torch.ones_like(labels, dtype=features.dtype))
    if bool((counts == 0).any()):
        missing = torch.where(counts == 0)[0].tolist()
        raise ValueError(f"feature cache has classes without training examples: {missing}")
    return centroids / counts[:, None]


def task_accuracies(
    model: AdapterHead,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    tasks: list[list[int]],
    seen_tasks: int,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> list[float]:
    model.eval()
    accuracies: list[float] = []
    with torch.no_grad():
        for task_id in range(seen_tasks):
            allowed = tasks[task_id]
            loader = subset_loader(
                x_test,
                y_test,
                allowed,
                batch_size,
                False,
                seed + 700_000 + 1_000 * seen_tasks + task_id,
            )
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


def regularized_snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and name in REGULARIZED_PARAMETER_NAMES
    }


def zeros_like_snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: torch.zeros_like(parameter) for name, parameter in model.named_parameters() if name in REGULARIZED_PARAMETER_NAMES}


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


def estimate_spatial_importance(
    model: AdapterHead,
    centroids: torch.Tensor,
    seen_classes: list[int],
    *,
    a_exc: float,
    a_inh: float,
    sigma_exc: float,
    sigma_inh: float,
) -> dict[str, torch.Tensor]:
    """Measure parameter sensitivity weighted by SSR interaction mass."""
    importance = zeros_like_snapshot(model)
    model.eval()
    model.zero_grad(set_to_none=True)

    def interaction_weighted_energy(values: torch.Tensor) -> torch.Tensor:
        directions = F.normalize(values.float(), dim=1)
        cosine = torch.clamp(directions @ directions.T, -1.0, 1.0)
        distance = torch.sqrt(torch.clamp(1.0 - cosine, min=1e-8))
        interaction = torch.abs(
            a_exc * radial_kernel(distance, sigma_exc)
            - a_inh * radial_kernel(distance, sigma_inh)
        )
        offdiag = ~torch.eye(values.shape[0], dtype=torch.bool, device=values.device)
        node_weights = interaction.masked_fill(~offdiag, 0.0).mean(dim=1)
        node_weights = node_weights / (node_weights.mean().detach() + 1e-8)
        return (node_weights.detach()[:, None] * values.float().square()).mean()

    responses = model.encode(centroids[seen_classes])
    prototypes = model.classifier.weight[seen_classes]
    objective = interaction_weighted_energy(responses) + interaction_weighted_energy(prototypes)
    objective.backward()
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and name in importance:
            importance[name].copy_(parameter.grad.detach().abs())
    model.zero_grad(set_to_none=True)
    return importance


def add_importance(
    cumulative: dict[str, torch.Tensor], current: dict[str, torch.Tensor]
) -> None:
    for name in cumulative:
        cumulative[name].add_(current[name])


def mask_unseen_classifier_rows(
    importance: dict[str, torch.Tensor], seen_classes: list[int]
) -> None:
    classifier_importance = importance.get("classifier.weight")
    if classifier_importance is None:
        return
    active = torch.zeros(
        classifier_importance.shape[0], dtype=torch.bool, device=classifier_importance.device
    )
    active[torch.as_tensor(seen_classes, device=classifier_importance.device)] = True
    classifier_importance[~active] = 0


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
    centroids = class_feature_centroids(x_train, y_train, args.num_classes).to(device)
    tasks = make_tasks(args.num_classes, args.classes_per_task, args.seed, args.class_order)[: args.max_tasks]

    model = AdapterHead(x_train.shape[1], args.num_classes, args.rank, args.tau, args.adapter_scale).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    reference = regularized_snapshot(model)
    importance = zeros_like_snapshot(model)
    si_path = zeros_like_snapshot(model)
    teacher: AdapterHead | None = None
    accuracy_matrix: list[list[float]] = []

    use_ssr = args.method == "kd_ssr"
    use_ewc = args.method == "kd_ewc"
    use_mas = args.method == "kd_mas"
    use_si = args.method == "kd_si"
    ssr_classifier_coefficient = args.coefficient if use_ssr else 0.0
    ssr_response_coefficient = args.ssr_response_ratio * args.coefficient if use_ssr else 0.0
    ssr_topology_coefficient = args.ssr_topology_ratio * args.coefficient if use_ssr else 0.0
    ssr_importance_coefficient = args.ssr_importance_ratio * args.coefficient if use_ssr else 0.0
    for task_id, classes in enumerate(tasks):
        if use_ssr and args.ssr_consolidation_power > 0:
            consolidation_scale = (
                task_id / max(len(tasks) - 1, 1)
            ) ** args.ssr_consolidation_power
        else:
            consolidation_scale = 1.0
        seen_classes = [class_id for task in tasks[: task_id + 1] for class_id in task]
        loader = subset_loader(
            x_train,
            y_train,
            classes,
            args.batch_size,
            True,
            args.seed + 100_003 * task_id,
        )
        for _ in range(args.epochs):
            model.train()
            for features, labels in loader:
                features, labels = features.to(device), labels.to(device)
                logits = model(features)
                loss = F.cross_entropy(logits, labels)
                if use_ssr:
                    loss = loss + ssr_classifier_coefficient * ssr_loss(
                        model.classifier.weight[seen_classes],
                        a_exc=args.a_exc,
                        a_inh=args.a_inh,
                        sigma_exc=args.sigma_exc,
                        sigma_inh=args.sigma_inh,
                    )
                    loss = loss + ssr_response_coefficient * ssr_loss(
                        model.encode(centroids[seen_classes]),
                        a_exc=args.a_exc,
                        a_inh=args.a_inh,
                        sigma_exc=args.sigma_exc,
                        sigma_inh=args.sigma_inh,
                    )
                if (use_ewc or use_mas or use_si) and task_id > 0:
                    loss = loss + args.coefficient * quadratic_penalty(model, reference, importance)
                if use_ssr and task_id > 0 and ssr_importance_coefficient > 0:
                    loss = loss + consolidation_scale * ssr_importance_coefficient * quadratic_penalty(
                        model, reference, importance
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
                    old_classes = [
                        class_id for previous in tasks[:task_id] for class_id in previous
                    ]
                    if old_classes and args.centroid_kd_coefficient > 0:
                        centroid_features = centroids[old_classes]
                        with torch.no_grad():
                            centroid_targets = teacher(centroid_features)[:, old_classes]
                        centroid_logits = model(centroid_features)[:, old_classes]
                        centroid_distillation = F.kl_div(
                            F.log_softmax(centroid_logits / temperature, dim=1),
                            F.softmax(centroid_targets / temperature, dim=1),
                            reduction="batchmean",
                        )
                        loss = loss + (
                            args.centroid_kd_coefficient
                            * temperature
                            * temperature
                            * centroid_distillation
                        )
                    if old_classes and use_ssr and ssr_topology_coefficient > 0:
                        old_centroid_features = centroids[old_classes]
                        with torch.no_grad():
                            reference_responses = teacher.encode(old_centroid_features)
                        current_responses = model.encode(old_centroid_features)
                        response_anchor = functional_topology_anchor(
                            current_responses, reference_responses,
                            a_exc=args.a_exc, a_inh=args.a_inh,
                            sigma_exc=args.sigma_exc, sigma_inh=args.sigma_inh,
                        )
                        prototype_anchor = functional_topology_anchor(
                            model.classifier.weight[old_classes],
                            teacher.classifier.weight[old_classes],
                            a_exc=args.a_exc, a_inh=args.a_inh,
                            sigma_exc=args.sigma_exc, sigma_inh=args.sigma_inh,
                        )
                        loss = loss + 0.5 * consolidation_scale * ssr_topology_coefficient * (
                            response_anchor + prototype_anchor
                        )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                before = regularized_snapshot(model) if use_si else None
                gradients = {
                    name: parameter.grad.detach().clone()
                    for name, parameter in model.named_parameters()
                    if use_si and parameter.grad is not None and name in before
                }
                optimizer.step()
                if use_si and before is not None:
                    for name, parameter in model.named_parameters():
                        if name in gradients:
                            delta = parameter.detach() - before[name]
                            si_path[name].add_(-gradients[name] * delta)

        accuracy_matrix.append(
            task_accuracies(
                model,
                x_test,
                y_test,
                tasks,
                task_id + 1,
                args.batch_size,
                device,
                args.seed,
            )
        )

        boundary_loader = subset_loader(
            x_train,
            y_train,
            classes,
            args.batch_size,
            False,
            args.seed + 900_003 + task_id,
        )
        if use_ewc:
            current_importance = estimate_ewc_importance(model, boundary_loader, device)
            mask_unseen_classifier_rows(current_importance, seen_classes)
            add_importance(importance, current_importance)
            reference = regularized_snapshot(model)
        elif use_mas:
            current_importance = estimate_mas_importance(model, boundary_loader, device)
            mask_unseen_classifier_rows(current_importance, seen_classes)
            add_importance(importance, current_importance)
            reference = regularized_snapshot(model)
        elif use_si:
            current = regularized_snapshot(model)
            for name in importance:
                displacement = current[name] - reference[name]
                importance[name].add_(torch.clamp(si_path[name] / (displacement.square() + args.si_epsilon), min=0.0))
                si_path[name].zero_()
            mask_unseen_classifier_rows(importance, seen_classes)
            reference = current
        elif use_ssr and args.ssr_importance_ratio > 0:
            current_importance = estimate_spatial_importance(
                model,
                centroids,
                seen_classes,
                a_exc=args.a_exc,
                a_inh=args.a_inh,
                sigma_exc=args.sigma_exc,
                sigma_inh=args.sigma_inh,
            )
            mask_unseen_classifier_rows(current_importance, seen_classes)
            add_importance(importance, current_importance)
            reference = regularized_snapshot(model)
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
        "candidate_id": args.candidate_id,
        "rank": args.rank,
        "avg_accuracy": float(np.mean(final)),
        "avg_forgetting": float(np.mean(forgetting)) if forgetting else 0.0,
        "final_old_accuracy": float(np.mean(final[:-1])) if len(final) > 1 else float(final[0]),
        "final_new_accuracy": float(final[-1]),
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
            "dataset": args.dataset_name,
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
            "kd_scope": "current_examples_plus_one_old_class_centroid_per_seen_class_from_preceding_task_boundary_teacher",
            "kd_coefficient": args.kd_coefficient,
            "kd_temperature": args.kd_temperature,
            "centroid_kd_coefficient": args.centroid_kd_coefficient,
            "ssr_enabled": use_ssr,
            "ssr_classifier_coefficient": ssr_classifier_coefficient if use_ssr else 0.0,
            "ssr_response_coefficient": ssr_response_coefficient if use_ssr else 0.0,
            "ssr_response_ratio": args.ssr_response_ratio if use_ssr else 0.0,
            "ssr_topology_coefficient": ssr_topology_coefficient if use_ssr else 0.0,
            "ssr_topology_ratio": args.ssr_topology_ratio if use_ssr else 0.0,
            "ssr_importance_coefficient": ssr_importance_coefficient if use_ssr else 0.0,
            "ssr_importance_ratio": args.ssr_importance_ratio if use_ssr else 0.0,
            "ssr_consolidation_power": args.ssr_consolidation_power if use_ssr else 0.0,
            "ssr_mapping": "shared cosine",
            "ssr_kernel": "Gaussian center-surround",
            "regularizer_parameter_scope": sorted(REGULARIZED_PARAMETER_NAMES),
            "ssr_active_rows": "seen_classifier_prototypes_only",
            "ssr_functional_target": "adapted_seen_class_training_feature_centroids_and_seen_classifier_prototypes",
            "a_exc": args.a_exc,
            "a_inh": args.a_inh,
            "sigma_exc": args.sigma_exc,
            "sigma_inh": args.sigma_inh,
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
    candidates: dict[str, list[dict[str, Any]]] = {}
    for method in TUNED_METHODS:
        rows = [row for row in records if row["method"] == method and row["seed"] in seeds]
        by_candidate: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            by_candidate.setdefault(str(row["candidate_id"]), []).append(row)
        method_candidates = []
        for candidate_id, items in sorted(by_candidate.items()):
            if len(items) != len(seeds):
                continue
            aa = np.asarray([item["avg_accuracy"] for item in items])
            af = np.asarray([item["avg_forgetting"] for item in items])
            regularizer = items[0]["regularizer"]
            method_candidates.append(
                {
                    "candidate_id": candidate_id,
                    "coefficient": float(items[0]["coefficient"]),
                    "sigma_exc": float(regularizer["sigma_exc"]),
                    "sigma_inh": float(regularizer["sigma_inh"]),
                    "ssr_response_ratio": float(regularizer["ssr_response_ratio"]),
                    "ssr_topology_ratio": float(regularizer["ssr_topology_ratio"]),
                    "ssr_importance_ratio": float(regularizer["ssr_importance_ratio"]),
                    "ssr_consolidation_power": float(regularizer["ssr_consolidation_power"]),
                    "mean_aa": float(aa.mean()),
                    "mean_af": float(af.mean()),
                    "mean_new_accuracy": float(np.mean([item["final_new_accuracy"] for item in items])),
                    "selection_utility": float(aa.mean() - 0.25 * af.mean()),
                    "seeds": sorted(seeds),
                }
            )
        if not method_candidates:
            raise RuntimeError(f"no complete development candidates for {method}")
        candidates[method] = method_candidates
    candidate_counts = {method: len(items) for method, items in candidates.items()}
    if len(set(candidate_counts.values())) != 1:
        raise RuntimeError(f"unequal development budgets: {candidate_counts}")
    kd_rows = [row for row in records if row["method"] == "kd" and row["seed"] in seeds]
    if len(kd_rows) != len(seeds):
        raise RuntimeError("one KD development record is required for every development seed")
    kd_new_accuracy = float(np.mean([row["final_new_accuracy"] for row in kd_rows]))
    acquisition_floor = kd_new_accuracy - 0.5
    for items in candidates.values():
        for item in items:
            item["acquisition_eligible"] = item["mean_new_accuracy"] >= acquisition_floor
    selected = {}
    for method, items in candidates.items():
        eligible = [item for item in items if item["acquisition_eligible"]]
        pool = eligible if eligible else items
        selected[method] = sorted(
            pool,
            key=lambda item: (-item["selection_utility"], -item["mean_aa"], item["mean_af"], item["candidate_id"]),
        )[0]
    ssr_aa = selected["kd_ssr"]["mean_aa"]
    comparator_aa = {method: selected[method]["mean_aa"] for method in ("kd_ewc", "kd_mas", "kd_si")}
    ssr_af = selected["kd_ssr"]["mean_af"]
    comparator_af = {method: selected[method]["mean_af"] for method in ("kd_ewc", "kd_mas", "kd_si")}
    development_gate_passed = (
        all(item["acquisition_eligible"] for item in selected.values())
        and all(ssr_aa > value for value in comparator_aa.values())
        and all(ssr_af < value for value in comparator_af.values())
    )
    payload = {
        "status": "selected",
        "protocol": PROTOCOL,
        "development_seeds": sorted(seeds),
        "selection_rule": "among settings within 0.5 pp of KD final-new accuracy, maximize mean AA - 0.25*AF; fixed tie breaks",
        "kd_mean_final_new_accuracy": kd_new_accuracy,
        "acquisition_floor": acquisition_floor,
        "candidate_budget_per_tuned_family": next(iter(candidate_counts.values())),
        "candidates": candidates,
        "selected": selected,
        "development_gate": {
            "definition": "all selected arms pass the acquisition floor and selected KD+SSR has higher AA and lower AF than KD+EWC, KD+MAS and KD+SI",
            "comparator_mean_aa": comparator_aa,
            "comparator_mean_af": comparator_af,
            "passed": development_gate_passed,
        },
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
    roots = [Path(item) for item in args.result_root]
    records = [row for root in roots for row in load_records(root, "confirmation")]
    expected_seeds = sorted(set(args.seeds))
    expected_methods = list(METHODS)
    indexed = {(row["method"], int(row["seed"])): row for row in records}
    if len(indexed) != len(records):
        raise RuntimeError("duplicate method/seed records across confirmation roots")
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
        "kd": ("kd_ssr", "kd_ewc", "kd_mas", "kd_si"),
        "kd_ewc": ("kd_ssr",),
        "kd_mas": ("kd_ssr",),
        "kd_si": ("kd_ssr",),
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
            new_accuracy = np.asarray([
                indexed[(treatment, seed)]["final_new_accuracy"]
                - indexed[(reference, seed)]["final_new_accuracy"]
                for seed in expected_seeds
            ])
            aa_mean, aa_low, aa_high = paired_interval(aa)
            af_mean, af_low, af_high = paired_interval(af_reduction)
            new_mean, new_low, new_high = paired_interval(new_accuracy)
            contrasts[f"{treatment}_vs_{reference}"] = {
                "aa_gain_pp": {"mean": aa_mean, "ci95": [aa_low, aa_high]},
                "af_reduction_pp": {"mean": af_mean, "ci95": [af_low, af_high]},
                "final_new_accuracy_gain_pp": {
                    "mean": new_mean,
                    "ci95": [new_low, new_high],
                },
                "dual_favorable_seeds": int(np.sum((aa > 0) & (af_reduction > 0))),
                "n": len(expected_seeds),
            }

    comparator_keys = (
        "kd_ssr_vs_kd_ewc",
        "kd_ssr_vs_kd_mas",
        "kd_ssr_vs_kd_si",
    )
    regularizer_comparison_gate_passed = all(
        contrasts[key]["aa_gain_pp"]["ci95"][0] > 0
        and contrasts[key]["af_reduction_pp"]["ci95"][0] > 0
        for key in comparator_keys
    )
    confirmation_gate_passed = (
        regularizer_comparison_gate_passed
        and contrasts["kd_ssr_vs_kd"]["final_new_accuracy_gain_pp"]["ci95"][0] > -0.5
    )

    payload = {
        "status": "complete",
        "protocol": PROTOCOL,
        "rank": int(records[0]["rank"]),
        "result_roots": [str(root) for root in roots],
        "confirmation_seeds": expected_seeds,
        "all_confirmation_records_retained": True,
        "methods": methods,
        "contrasts": contrasts,
        "regularizer_comparison_gate": {
            "definition": "KD+SSR paired AA-gain and AF-reduction 95% CI lower bounds both exceed zero against KD+EWC, KD+MAS and KD+SI",
            "passed": regularizer_comparison_gate_passed,
        },
        "main_text_gate": {
            "definition": "KD+SSR paired AA-gain and AF-reduction 95% CI lower bounds both exceed zero against KD+EWC, KD+MAS and KD+SI, and final-new accuracy is non-inferior to KD within 0.5 pp",
            "passed": confirmation_gate_passed,
        },
    }
    atomic_json(Path(args.output), payload)

    csv_path = Path(args.csv_output)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["rank", "method", "seed", "candidate_id", "coefficient", "avg_accuracy", "avg_forgetting", "final_old_accuracy", "final_new_accuracy", "result_file"],
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
                        "candidate_id": row["candidate_id"],
                        "coefficient": row["coefficient"],
                        "avg_accuracy": row["avg_accuracy"],
                        "avg_forgetting": row["avg_forgetting"],
                        "final_old_accuracy": row["final_old_accuracy"],
                        "final_new_accuracy": row["final_new_accuracy"],
                        "result_file": row["_path"],
                    }
                )
    return payload


def summarize_robustness(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.result_root)
    grid_source = Path(args.grid_source)
    source = json.loads(grid_source.read_text(encoding="utf-8"))
    if source.get("protocol") != PROTOCOL:
        raise RuntimeError("grid source protocol mismatch")
    multipliers = (0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
    source_candidates: dict[str, dict[str, dict[str, Any]]] = {}
    for method in TUNED_METHODS:
        base = source["selected"][method]
        candidates = {}
        for multiplier in multipliers:
            multiplier_label = str(multiplier).replace(".", "p")
            candidate_id = f"scale_m{multiplier_label}"
            candidates[candidate_id] = {
                "base_candidate_id": str(base["candidate_id"]),
                "scale_multiplier": multiplier,
                "coefficient": float(base["coefficient"]) * multiplier,
                "sigma_exc": float(base["sigma_exc"]),
                "sigma_inh": float(base["sigma_inh"]),
                "ssr_response_ratio": float(base["ssr_response_ratio"]),
                "ssr_topology_ratio": float(base["ssr_topology_ratio"]),
                "ssr_importance_ratio": float(base["ssr_importance_ratio"]),
                "ssr_consolidation_power": float(base["ssr_consolidation_power"]),
            }
        source_candidates[method] = candidates

    expected_seeds = sorted(set(args.seeds))
    if len(expected_seeds) != len(args.seeds):
        raise RuntimeError("robustness seeds must be unique")
    if set(expected_seeds) & set(source.get("development_seeds", [])):
        raise RuntimeError("robustness seeds overlap development seeds")

    records = load_records(root, "robustness")
    indexed = {
        (str(row["method"]), str(row["candidate_id"]), int(row["seed"])): row
        for row in records
    }
    if len(indexed) != len(records):
        raise RuntimeError("duplicate method/candidate/seed robustness records")

    expected = {("kd", "baseline", seed) for seed in expected_seeds}
    for method, candidates in source_candidates.items():
        expected.update(
            (method, candidate_id, seed)
            for candidate_id in candidates
            for seed in expected_seeds
        )
    missing = sorted(expected - set(indexed))
    unexpected = sorted(set(indexed) - expected)
    if missing or unexpected:
        raise RuntimeError(
            f"robustness grid mismatch; missing={missing[:10]}, unexpected={unexpected[:10]}"
        )

    kd_rows = {seed: indexed[("kd", "baseline", seed)] for seed in expected_seeds}
    families: dict[str, Any] = {}
    csv_rows: list[dict[str, Any]] = []
    for seed, row in kd_rows.items():
        csv_rows.append({
            "rank": int(row["rank"]),
            "method": "kd",
            "candidate_id": "baseline",
            "seed": seed,
            "avg_accuracy": row["avg_accuracy"],
            "avg_forgetting": row["avg_forgetting"],
            "final_new_accuracy": row["final_new_accuracy"],
            "aa_gain_vs_kd_pp": 0.0,
            "af_reduction_vs_kd_pp": 0.0,
            "final_new_gain_vs_kd_pp": 0.0,
            "result_file": row["_path"],
        })
    for method, candidate_source in source_candidates.items():
        candidate_payload: dict[str, Any] = {}
        for candidate_id, frozen in sorted(candidate_source.items()):
            rows = [indexed[(method, candidate_id, seed)] for seed in expected_seeds]
            first = rows[0]
            regularizer = first["regularizer"]
            frozen_fields = {
                "coefficient": float(frozen["coefficient"]),
                "sigma_exc": float(frozen["sigma_exc"]),
                "sigma_inh": float(frozen["sigma_inh"]),
                "ssr_response_ratio": float(frozen["ssr_response_ratio"]),
                "ssr_topology_ratio": float(frozen["ssr_topology_ratio"]),
                "ssr_importance_ratio": float(frozen["ssr_importance_ratio"]),
                "ssr_consolidation_power": float(frozen["ssr_consolidation_power"]),
            }
            observed_fields = {
                "coefficient": float(first["coefficient"]),
                "sigma_exc": float(regularizer["sigma_exc"]),
                "sigma_inh": float(regularizer["sigma_inh"]),
                "ssr_response_ratio": float(regularizer["ssr_response_ratio"]),
                "ssr_topology_ratio": float(regularizer["ssr_topology_ratio"]),
                "ssr_importance_ratio": float(regularizer["ssr_importance_ratio"]),
                "ssr_consolidation_power": float(regularizer["ssr_consolidation_power"]),
            }
            if observed_fields != frozen_fields:
                raise RuntimeError(
                    f"configuration drift for {method}/{candidate_id}: "
                    f"observed={observed_fields}, frozen={frozen_fields}"
                )

            aa_gain = np.asarray([
                row["avg_accuracy"] - kd_rows[int(row["seed"])]["avg_accuracy"]
                for row in rows
            ])
            af_reduction = np.asarray([
                kd_rows[int(row["seed"])]["avg_forgetting"] - row["avg_forgetting"]
                for row in rows
            ])
            new_gain = np.asarray([
                row["final_new_accuracy"] - kd_rows[int(row["seed"])]["final_new_accuracy"]
                for row in rows
            ])
            aa_mean, aa_low, aa_high = paired_interval(aa_gain)
            af_mean, af_low, af_high = paired_interval(af_reduction)
            new_mean, new_low, new_high = paired_interval(new_gain)
            acquisition_eligible = new_mean >= -0.5
            dual_favorable = aa_mean > 0 and af_mean > 0
            dual_ci_confirmed = aa_low > 0 and af_low > 0
            candidate_payload[candidate_id] = {
                "base_candidate_id": frozen["base_candidate_id"],
                "scale_multiplier": frozen["scale_multiplier"],
                **frozen_fields,
                "mean_avg_accuracy": float(np.mean([row["avg_accuracy"] for row in rows])),
                "mean_avg_forgetting": float(np.mean([row["avg_forgetting"] for row in rows])),
                "mean_final_new_accuracy": float(np.mean([row["final_new_accuracy"] for row in rows])),
                "aa_gain_pp": {"mean": aa_mean, "ci95": [aa_low, aa_high]},
                "af_reduction_pp": {"mean": af_mean, "ci95": [af_low, af_high]},
                "final_new_accuracy_gain_pp": {
                    "mean": new_mean,
                    "ci95": [new_low, new_high],
                },
                "acquisition_eligible": acquisition_eligible,
                "dual_favorable_means": dual_favorable,
                "dual_favorable_cis": dual_ci_confirmed,
                "robust_success": acquisition_eligible and dual_favorable,
                "n": len(expected_seeds),
            }
            for row in rows:
                seed = int(row["seed"])
                csv_rows.append({
                    "rank": int(row["rank"]),
                    "method": method,
                    "candidate_id": candidate_id,
                    "seed": seed,
                    "avg_accuracy": row["avg_accuracy"],
                    "avg_forgetting": row["avg_forgetting"],
                    "final_new_accuracy": row["final_new_accuracy"],
                    "aa_gain_vs_kd_pp": row["avg_accuracy"] - kd_rows[seed]["avg_accuracy"],
                    "af_reduction_vs_kd_pp": kd_rows[seed]["avg_forgetting"] - row["avg_forgetting"],
                    "final_new_gain_vs_kd_pp": row["final_new_accuracy"] - kd_rows[seed]["final_new_accuracy"],
                    "result_file": row["_path"],
                })

        mean_aa = [item["mean_avg_accuracy"] for item in candidate_payload.values()]
        mean_af = [item["mean_avg_forgetting"] for item in candidate_payload.values()]
        mean_new = [item["mean_final_new_accuracy"] for item in candidate_payload.values()]
        eligible_count = sum(item["acquisition_eligible"] for item in candidate_payload.values())
        dual_count = sum(item["dual_favorable_means"] for item in candidate_payload.values())
        confirmed_count = sum(item["dual_favorable_cis"] for item in candidate_payload.values())
        robust_count = sum(item["robust_success"] for item in candidate_payload.values())
        families[method] = {
            "candidate_count": len(candidate_payload),
            "acquisition_eligible_count": eligible_count,
            "dual_favorable_mean_count": dual_count,
            "dual_favorable_ci_count": confirmed_count,
            "robust_success_count": robust_count,
            "robust_success_fraction": robust_count / len(candidate_payload),
            "mean_metric_ranges_pp": {
                "avg_accuracy": max(mean_aa) - min(mean_aa),
                "avg_forgetting": max(mean_af) - min(mean_af),
                "final_new_accuracy": max(mean_new) - min(mean_new),
            },
            "candidates": candidate_payload,
        }

    payload = {
        "status": "complete",
        "protocol": f"{PROTOCOL}_full_grid_robustness_v1",
        "rank": int(records[0]["rank"]),
        "result_root": str(root),
        "grid_source": str(grid_source),
        "grid_source_sha256": file_sha256(grid_source),
        "source_development_seeds": sorted(source["development_seeds"]),
        "robustness_confirmation_seeds": expected_seeds,
        "all_grid_records_retained": True,
        "criterion": {
            "acquisition_eligible": "mean final-new accuracy no more than 0.5 pp below paired KD",
            "dual_favorable_means": "mean AA gain > 0 and mean AF reduction > 0 versus paired KD",
            "dual_favorable_cis": "paired 95% CI lower bounds for AA gain and AF reduction both > 0",
            "robust_success": "acquisition eligible and dual-favorable means",
        },
        "relative_strength_multipliers": list(multipliers),
        "sweep_definition": "each family is scaled over the same 1/16x--16x range around its development-selected center; SSR scales all coefficient-derived terms together while keeping its schedule shape fixed",
        "families": families,
    }
    atomic_json(Path(args.output), payload)

    csv_path = Path(args.csv_output)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument(
        "--stage",
        choices=("canary", "development", "confirmation", "robustness"),
        required=True,
    )
    run_parser.add_argument("--method", choices=METHODS, required=True)
    run_parser.add_argument("--seed", type=int, required=True)
    run_parser.add_argument("--coefficient", type=float, required=True)
    run_parser.add_argument("--candidate-id", required=True)
    run_parser.add_argument("--feature-cache", required=True)
    run_parser.add_argument("--dataset-name", required=True)
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
    run_parser.add_argument("--ssr-classifier-coefficient", type=float, default=1.0)
    run_parser.add_argument("--ssr-response-ratio", type=float, default=0.5)
    run_parser.add_argument("--ssr-topology-ratio", type=float, default=2.0)
    run_parser.add_argument("--ssr-importance-ratio", type=float, default=0.0)
    run_parser.add_argument("--ssr-consolidation-power", type=float, default=0.0)
    run_parser.add_argument("--a-exc", type=float, default=1.0)
    run_parser.add_argument("--a-inh", type=float, default=0.8)
    run_parser.add_argument("--sigma-exc", type=float, default=0.55)
    run_parser.add_argument("--sigma-inh", type=float, default=1.25)
    run_parser.add_argument("--kd-coefficient", type=float, default=2.0)
    run_parser.add_argument("--kd-temperature", type=float, default=3.0)
    run_parser.add_argument("--centroid-kd-coefficient", type=float, default=1.0)
    run_parser.add_argument("--si-epsilon", type=float, default=1e-3)

    select_parser = subparsers.add_parser("select")
    select_parser.add_argument("--result-root", required=True)
    select_parser.add_argument("--seeds", nargs="+", type=int, required=True)
    select_parser.add_argument("--output", required=True)

    summary_parser = subparsers.add_parser("summarize")
    summary_parser.add_argument("--result-root", action="append", required=True)
    summary_parser.add_argument("--seeds", nargs="+", type=int, required=True)
    summary_parser.add_argument("--output", required=True)
    summary_parser.add_argument("--csv-output", required=True)

    robustness_parser = subparsers.add_parser("robustness")
    robustness_parser.add_argument("--result-root", required=True)
    robustness_parser.add_argument("--grid-source", required=True)
    robustness_parser.add_argument("--seeds", nargs="+", type=int, required=True)
    robustness_parser.add_argument("--output", required=True)
    robustness_parser.add_argument("--csv-output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "run":
        result = run_one(args)
    elif args.command == "select":
        result = select(args)
    elif args.command == "summarize":
        result = summarize(args)
    else:
        result = summarize_robustness(args)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
