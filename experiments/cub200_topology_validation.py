#!/usr/bin/env python3
"""Matched CUB200 validation of bounded interaction topology.

The frozen feature cache, task stream, optimizer, KD scaffold, and training
budget are identical across methods. The experiment changes only the geometry
term and evaluates both endpoints and a semantic-neighborhood diagnostic built
from the frozen feature space before continual training.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.cub200_continual_benchmark import (  # noqa: E402
    LinearHead,
    evaluate_classification_cil,
    evaluate_classification_til,
    extract_classification_features,
    geometry,
    make_tasks,
    set_seed,
    subset_loader,
)
from methods.bioreg import compute_spectral_flatness  # noqa: E402
from methods.geometry_controls import (  # noqa: E402
    cosine_orthogonal_loss,
    prototype_decorrelation_loss,
)


METHODS = (
    "kd",
    "ssr_gaussian",
    "ssr_gen_p1",
    "ssr_gen_p4",
    "ssr_gaussian_quantile",
    "ssr_compact_quantile",
    "ssr_soft_annulus_quantile",
    "center_only",
    "surround_only",
    "cosine_orthogonal",
    "prototype_decorrelation",
    "spectral",
)


def class_centroids(
    features: torch.Tensor, labels: torch.Tensor, num_classes: int
) -> torch.Tensor:
    centroids = []
    for class_id in range(num_classes):
        selected = features[labels == class_id]
        if selected.numel() == 0:
            raise ValueError(f"No frozen features for CUB class {class_id}")
        centroids.append(selected.float().mean(dim=0))
    return F.normalize(torch.stack(centroids), dim=1)


def pairwise_cosine_distance(directions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    normalized = F.normalize(directions.float(), dim=1)
    cosine = torch.clamp(normalized @ normalized.T, -1.0, 1.0)
    distance = torch.sqrt(torch.clamp(1.0 - cosine, min=1e-8))
    return cosine, distance


def reference_quantile_anchors(reference: torch.Tensor) -> torch.Tensor:
    _, distance = pairwise_cosine_distance(reference)
    mask = torch.triu(torch.ones_like(distance, dtype=torch.bool), diagonal=1)
    probabilities = torch.linspace(0.05, 0.95, 19, device=distance.device)
    return torch.quantile(distance[mask], probabilities).detach()


def soft_empirical_quantile(
    distance: torch.Tensor, anchors: torch.Tensor, temperature: float
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("quantile temperature must be positive")
    return torch.sigmoid(
        (distance.unsqueeze(-1) - anchors.view(1, 1, -1)) / temperature
    ).mean(dim=-1)


def generalized_response(distance: torch.Tensor, half_width: float, exponent: float) -> torch.Tensor:
    if half_width <= 0 or exponent <= 0:
        raise ValueError("half-width and exponent must be positive")
    return torch.exp(-math.log(2.0) * (distance / half_width).pow(exponent))


def wendland_response(distance: torch.Tensor, support: float) -> torch.Tensor:
    if support <= 0:
        raise ValueError("support must be positive")
    ratio = distance / support
    remaining = torch.clamp(1.0 - ratio, min=0.0)
    return remaining.pow(4) * (4.0 * ratio + 1.0)


def ssr_potential(
    method: str,
    distance: torch.Tensor,
    anchors: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    gaussian_hwhm = math.sqrt(2.0 * math.log(2.0))
    raw_h_exc = args.sigma_exc * gaussian_hwhm
    raw_h_inh = args.sigma_inh * gaussian_hwhm

    if method in {"ssr_gaussian", "ssr_gen_p1", "ssr_gen_p4", "center_only", "surround_only"}:
        exponent = {"ssr_gen_p1": 1.0, "ssr_gen_p4": 4.0}.get(method, 2.0)
        exc = args.a_exc * generalized_response(distance, raw_h_exc, exponent)
        inh = args.a_inh * generalized_response(distance, raw_h_inh, exponent)
        if method == "center_only":
            return -exc
        if method == "surround_only":
            return inh
        return inh - exc

    quantile_distance = soft_empirical_quantile(
        distance, anchors, args.quantile_temperature
    )
    if method == "ssr_gaussian_quantile":
        exc = args.a_exc * generalized_response(quantile_distance, 0.05, 2.0)
        inh = args.a_inh * generalized_response(quantile_distance, 0.25, 2.0)
        return inh - exc
    if method == "ssr_compact_quantile":
        exc = args.a_exc * wendland_response(quantile_distance, 0.10)
        inh = args.a_inh * wendland_response(quantile_distance, 0.30)
        return inh - exc
    if method == "ssr_soft_annulus_quantile":
        edge = args.quantile_edge_temperature
        center = args.a_exc * generalized_response(quantile_distance, 0.05, 2.0)
        annulus = torch.sigmoid((quantile_distance - 0.05) / edge) - torch.sigmoid(
            (quantile_distance - 0.25) / edge
        )
        return args.a_inh * annulus - center
    raise ValueError(f"Unsupported SSR method: {method}")


def regularizer(
    method: str,
    weights: torch.Tensor,
    anchors: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    if method == "kd":
        return weights.sum() * 0.0
    if method == "cosine_orthogonal":
        return cosine_orthogonal_loss(weights)
    if method == "prototype_decorrelation":
        return prototype_decorrelation_loss(weights)
    if method == "spectral":
        return compute_spectral_flatness(weights.float())

    _, distance = pairwise_cosine_distance(weights)
    potential = ssr_potential(method, distance, anchors, args)
    off_diagonal = ~torch.eye(
        potential.shape[0], dtype=torch.bool, device=potential.device
    )
    return potential[off_diagonal].mean()


def semantic_geometry(
    weights: torch.Tensor,
    reference: torch.Tensor,
    tasks: list[list[int]],
    knn_k: int = 5,
) -> dict[str, float | dict[str, float]]:
    learned_cosine, _ = pairwise_cosine_distance(weights)
    reference_cosine, reference_distance = pairwise_cosine_distance(reference)
    pair_mask = torch.triu(
        torch.ones_like(reference_distance, dtype=torch.bool), diagonal=1
    )
    distances = reference_distance[pair_mask]
    q10, q35, q75 = torch.quantile(
        distances, torch.tensor([0.10, 0.35, 0.75], device=distances.device)
    )
    band_masks = {
        "close": pair_mask & (reference_distance <= q10),
        "intermediate": pair_mask
        & (reference_distance > q10)
        & (reference_distance <= q35),
        "distant": pair_mask & (reference_distance >= q75),
    }
    bands: dict[str, dict[str, float]] = {}
    for name, mask in band_masks.items():
        bands[name] = {
            "reference_mean_cosine": float(reference_cosine[mask].mean().item()),
            "learned_mean_cosine": float(learned_cosine[mask].mean().item()),
            "learned_mean_abs_cosine": float(learned_cosine[mask].abs().mean().item()),
            "pairs": int(mask.sum().item()),
        }

    class_count = weights.shape[0]
    diagonal = torch.eye(class_count, dtype=torch.bool, device=weights.device)
    reference_rank = reference_cosine.masked_fill(diagonal, -torch.inf).topk(
        knn_k, dim=1
    ).indices
    learned_rank = learned_cosine.masked_fill(diagonal, -torch.inf).topk(
        knn_k, dim=1
    ).indices
    recalls = []
    for class_id in range(class_count):
        overlap = set(reference_rank[class_id].tolist()) & set(
            learned_rank[class_id].tolist()
        )
        recalls.append(len(overlap) / knn_k)

    task_id = torch.empty(class_count, dtype=torch.long, device=weights.device)
    for index, classes in enumerate(tasks):
        task_id[torch.as_tensor(classes, device=weights.device)] = index
    same_task = pair_mask & (task_id[:, None] == task_id[None, :])
    adjacent_task = pair_mask & ((task_id[:, None] - task_id[None, :]).abs() == 1)
    separated_task = pair_mask & ((task_id[:, None] - task_id[None, :]).abs() > 1)

    return {
        "reference_distance_q10": float(q10.item()),
        "reference_distance_q35": float(q35.item()),
        "reference_distance_q75": float(q75.item()),
        "semantic_knn_recall_at_5": float(np.mean(recalls)),
        "within_task_mean_cosine": float(learned_cosine[same_task].mean().item()),
        "adjacent_task_mean_abs_cosine": float(
            learned_cosine[adjacent_task].abs().mean().item()
        ),
        "separated_task_mean_abs_cosine": float(
            learned_cosine[separated_task].abs().mean().item()
        ),
        "semantic_bands": bands,
    }


def run(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    requested_device = args.device if torch.cuda.is_available() else "cpu"
    device = torch.device(requested_device)
    x_train, y_train, x_test, y_test = extract_classification_features(args, device)
    num_classes = int(args.max_classes or args.num_classes)
    tasks = make_tasks(num_classes, args.classes_per_task, args.seed, args.class_order)
    if not tasks:
        raise ValueError("No complete CUB tasks were created")

    reference = class_centroids(x_train, y_train, num_classes).to(device)
    anchors = reference_quantile_anchors(reference).to(device)
    model = LinearHead(x_train.shape[1], num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    teacher: LinearHead | None = None
    acc_matrix: list[list[float]] = []
    cil_curve: list[float] = []

    for task_id, task_classes in enumerate(tasks):
        regularized_classes = [
            class_id for task in tasks[: task_id + 1] for class_id in task
        ]
        loader = subset_loader(x_train, y_train, task_classes, args.batch_size, True)
        for _ in range(args.epochs):
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = F.cross_entropy(logits, yb)
                if args.strength > 0 and args.method != "kd":
                    with torch.amp.autocast("cuda", enabled=False):
                        loss = loss + args.strength * regularizer(
                            args.method,
                            model.classifier.weight[regularized_classes].float(),
                            anchors,
                            args,
                        )
                if teacher is not None:
                    with torch.no_grad():
                        target = teacher(xb)
                    temperature = args.kd_temperature
                    loss = loss + args.lambda_kd * (temperature**2) * F.kl_div(
                        F.log_softmax(logits / temperature, dim=1),
                        F.softmax(target / temperature, dim=1),
                        reduction="batchmean",
                    )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        acc_matrix.append(
            evaluate_classification_til(
                model, x_test, y_test, tasks, task_id + 1, args.batch_size, device
            )
        )
        seen_classes = [cls for task in tasks[: task_id + 1] for cls in task]
        cil_curve.append(
            evaluate_classification_cil(
                model, x_test, y_test, seen_classes, args.batch_size, device
            )
        )
        teacher = copy.deepcopy(model).eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)

    final = acc_matrix[-1]
    forgetting = []
    for task_id in range(len(tasks) - 1):
        best = max(row[task_id] for row in acc_matrix[task_id:])
        forgetting.append(best - final[task_id])

    weights = model.classifier.weight.detach().float().cpu()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = output_dir / "classifier_weights.pt"
    torch.save(weights, weights_path)
    semantic = semantic_geometry(weights, reference.detach().cpu(), tasks)
    payload = {
        "benchmark": "CUB-200-2011 frozen ResNet-18 classifier head",
        "method": args.method,
        "seed": args.seed,
        "strength": args.strength,
        "num_classes": num_classes,
        "classes_per_task": args.classes_per_task,
        "class_order": args.class_order,
        "epochs": args.epochs,
        "lambda_kd": args.lambda_kd,
        "kd_temperature": args.kd_temperature,
        "a_exc": args.a_exc,
        "a_inh": args.a_inh,
        "sigma_exc": args.sigma_exc,
        "sigma_inh": args.sigma_inh,
        "quantile_temperature": args.quantile_temperature,
        "quantile_edge_temperature": args.quantile_edge_temperature,
        "quantile_anchor_values": [float(v) for v in anchors.detach().cpu()],
        "avg_accuracy": float(np.mean(final)),
        "avg_forgetting": float(np.mean(forgetting)) if forgetting else 0.0,
        "final_cil_accuracy": float(cil_curve[-1]),
        "avg_cil_accuracy": float(np.mean(cil_curve)),
        "til_curve": [float(np.mean(row)) for row in acc_matrix],
        "cil_curve": [float(value) for value in cil_curve],
        "task_order": tasks,
        "classifier_weights": str(weights_path),
        **geometry(weights),
        **semantic,
    }
    (output_dir / "result.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--strength", type=float, default=0.0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-root", default="data/cub200")
    parser.add_argument(
        "--classification-cache", default="data/cub200/cub200_resnet18_features.pt"
    )
    parser.add_argument("--num-classes", type=int, default=200)
    parser.add_argument("--max-classes", type=int, default=None)
    parser.add_argument("--classes-per-task", type=int, default=10)
    parser.add_argument("--class-order", choices=["semantic", "random"], default="semantic")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda-kd", type=float, default=2.0)
    parser.add_argument("--kd-temperature", type=float, default=3.0)
    parser.add_argument("--a-exc", type=float, default=1.2)
    parser.add_argument("--a-inh", type=float, default=0.9)
    parser.add_argument("--sigma-exc", type=float, default=0.16)
    parser.add_argument("--sigma-inh", type=float, default=0.45)
    parser.add_argument("--quantile-temperature", type=float, default=0.025)
    parser.add_argument("--quantile-edge-temperature", type=float, default=0.02)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--feature-batch-size", type=int, default=128)
    parser.add_argument("--no-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
