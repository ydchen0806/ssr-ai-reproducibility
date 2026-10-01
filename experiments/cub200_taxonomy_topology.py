#!/usr/bin/env python3
"""Taxonomy-aware CUB validation with an active density-calibrated SSR map."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments import cub200_topology_validation as base


ORIGINAL_REGULARIZER = base.regularizer
TAXONOMY_ORDERS = ("taxonomy_blocked", "taxonomy_interleaved", "random")
METHODS = (
    "kd",
    "ssr_dynamic",
    "center_dynamic",
    "surround_dynamic",
    "cosine_orthogonal",
    "prototype_decorrelation",
    "spectral",
)


def load_taxonomy(path: Path, num_classes: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("classes", [])
    by_id = {int(row["class_id"]): row for row in rows}
    selected = [by_id[index] for index in range(num_classes)]
    for row in selected:
        if not row.get("order") or not row.get("family") or not row.get("genus"):
            raise ValueError(f"Incomplete taxonomy row: {row}")
    return payload, selected


def taxonomy_tasks(
    rows: list[dict[str, Any]],
    classes_per_task: int,
    seed: int,
    order_mode: str,
) -> list[list[int]]:
    if len(rows) % classes_per_task:
        raise ValueError("Taxonomy experiment requires complete equal-size tasks")
    rng = random.Random(seed)
    if order_mode == "random":
        ordered = [int(row["class_id"]) for row in rows]
        rng.shuffle(ordered)
    elif order_mode == "taxonomy_blocked":
        grouped: dict[tuple[str, str], list[int]] = {}
        for row in rows:
            grouped.setdefault((row["order"], row["family"]), []).append(int(row["class_id"]))
        ordered = []
        for key in sorted(grouped):
            values = grouped[key]
            rng.shuffle(values)
            ordered.extend(values)
    elif order_mode == "taxonomy_interleaved":
        grouped: dict[tuple[str, str], list[int]] = {}
        for row in rows:
            grouped.setdefault((row["order"], row["family"]), []).append(int(row["class_id"]))
        keys = sorted(grouped)
        rng.shuffle(keys)
        for values in grouped.values():
            rng.shuffle(values)
        ordered = []
        while any(grouped[key] for key in keys):
            for key in keys:
                if grouped[key]:
                    ordered.append(grouped[key].pop())
    else:
        raise ValueError(f"Unknown taxonomy order: {order_mode}")
    return [ordered[start : start + classes_per_task] for start in range(0, len(ordered), classes_per_task)]


def dynamic_quantile_coordinate(distance: torch.Tensor, temperature: float) -> torch.Tensor:
    mask = torch.triu(torch.ones_like(distance, dtype=torch.bool), diagonal=1)
    anchors = torch.quantile(
        distance[mask].detach(),
        torch.linspace(0.025, 0.975, 39, device=distance.device),
    )
    return torch.sigmoid(
        (distance.unsqueeze(-1) - anchors.view(1, 1, -1)) / temperature
    ).mean(dim=-1)


def dynamic_regularizer(
    method: str,
    weights: torch.Tensor,
    anchors: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    if method not in {"ssr_dynamic", "center_dynamic", "surround_dynamic"}:
        return ORIGINAL_REGULARIZER(method, weights, anchors, args)
    _, distance = base.pairwise_cosine_distance(weights)
    coordinate = dynamic_quantile_coordinate(distance, args.dynamic_temperature)
    exc = args.a_exc * base.generalized_response(coordinate, args.center_hwhm, 2.0)
    inh = args.a_inh * base.generalized_response(coordinate, args.surround_hwhm, 2.0)
    if method == "center_dynamic":
        potential = -exc
    elif method == "surround_dynamic":
        potential = inh
    else:
        potential = inh - exc
    mask = ~torch.eye(potential.shape[0], dtype=torch.bool, device=potential.device)
    return potential[mask].mean()


def pair_mask(values: list[str], mode: str, device: torch.device) -> torch.Tensor:
    size = len(values)
    equal = torch.zeros((size, size), dtype=torch.bool, device=device)
    for i in range(size):
        for j in range(size):
            equal[i, j] = values[i] == values[j]
    upper = torch.triu(torch.ones_like(equal), diagonal=1)
    if mode == "equal":
        return upper & equal
    return upper & ~equal


def taxonomy_geometry(
    weights: torch.Tensor,
    rows: list[dict[str, Any]],
    tasks: list[list[int]],
) -> dict[str, Any]:
    normalized = F.normalize(weights.float(), dim=1)
    cosine = normalized @ normalized.T
    families = [str(row["family"]) for row in rows]
    orders = [str(row["order"]) for row in rows]
    genera = [str(row["genus"]) for row in rows]
    upper = torch.triu(torch.ones_like(cosine, dtype=torch.bool), diagonal=1)
    same_genus = pair_mask(genera, "equal", cosine.device)
    same_family = pair_mask(families, "equal", cosine.device)
    same_order = pair_mask(orders, "equal", cosine.device)
    same_family_diff_genus = same_family & ~same_genus
    same_order_diff_family = same_order & ~same_family
    cross_order = upper & ~same_order

    def summary(mask: torch.Tensor) -> dict[str, float | int]:
        values = cosine[mask]
        return {
            "pairs": int(values.numel()),
            "mean_cosine": float(values.mean().item()),
            "mean_abs_cosine": float(values.abs().mean().item()),
        }

    task_same_family = []
    task_same_order = []
    for task in tasks:
        pairs = [(i, j) for offset, i in enumerate(task) for j in task[offset + 1 :]]
        task_same_family.extend(families[i] == families[j] for i, j in pairs)
        task_same_order.extend(orders[i] == orders[j] for i, j in pairs)

    diagonal = torch.eye(weights.shape[0], dtype=torch.bool, device=weights.device)
    neighbors = cosine.masked_fill(diagonal, -torch.inf).topk(5, dim=1).indices
    family_purity = []
    order_purity = []
    for class_id in range(weights.shape[0]):
        neighbor_ids = neighbors[class_id].tolist()
        family_purity.append(sum(families[index] == families[class_id] for index in neighbor_ids) / 5)
        order_purity.append(sum(orders[index] == orders[class_id] for index in neighbor_ids) / 5)

    return {
        "same_genus": summary(same_genus),
        "same_family_diff_genus": summary(same_family_diff_genus),
        "same_order_diff_family": summary(same_order_diff_family),
        "cross_order": summary(cross_order),
        "task_pair_same_family_fraction": float(np.mean(task_same_family)),
        "task_pair_same_order_fraction": float(np.mean(task_same_order)),
        "taxonomy_knn5_family_purity": float(np.mean(family_purity)),
        "taxonomy_knn5_order_purity": float(np.mean(order_purity)),
    }


def dynamic_audit(weights: torch.Tensor, args: argparse.Namespace) -> dict[str, float]:
    trainable = weights.detach().float().clone().requires_grad_(True)
    _, distance = base.pairwise_cosine_distance(trainable)
    coordinate = dynamic_quantile_coordinate(distance, args.dynamic_temperature)
    mask = ~torch.eye(coordinate.shape[0], dtype=torch.bool, device=coordinate.device)
    loss = dynamic_regularizer(args.method, trainable, torch.empty(0), args)
    gradient = torch.autograd.grad(loss, trainable)[0]
    values = coordinate[mask]
    return {
        "center_occupancy": float((values <= args.center_hwhm).float().mean().item()),
        "annulus_occupancy": float(
            ((values > args.center_hwhm) & (values <= args.surround_hwhm)).float().mean().item()
        ),
        "release_occupancy": float((values > args.surround_hwhm).float().mean().item()),
        "regularizer_gradient_norm": float(gradient.norm().item()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--strength", type=float, default=0.0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--taxonomy-file", type=Path, required=True)
    parser.add_argument("--taxonomy-order", choices=TAXONOMY_ORDERS, required=True)
    parser.add_argument("--data-root", default="data/cub200")
    parser.add_argument("--classification-cache", default="data/cub200/cub200_resnet18_features.pt")
    parser.add_argument("--num-classes", type=int, default=200)
    parser.add_argument("--max-classes", type=int, default=None)
    parser.add_argument("--classes-per-task", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda-kd", type=float, default=2.0)
    parser.add_argument("--kd-temperature", type=float, default=3.0)
    parser.add_argument("--a-exc", type=float, default=1.2)
    parser.add_argument("--a-inh", type=float, default=0.9)
    parser.add_argument("--center-hwhm", type=float, default=0.08)
    parser.add_argument("--surround-hwhm", type=float, default=0.30)
    parser.add_argument("--dynamic-temperature", type=float, default=0.02)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--feature-batch-size", type=int, default=128)
    parser.add_argument("--no-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    num_classes = int(args.max_classes or args.num_classes)
    taxonomy_payload, rows = load_taxonomy(args.taxonomy_file, num_classes)

    def make_tasks(_: int, classes_per_task: int, seed: int, __: str) -> list[list[int]]:
        return taxonomy_tasks(rows, classes_per_task, seed, args.taxonomy_order)

    base.make_tasks = make_tasks
    base.regularizer = dynamic_regularizer
    args.class_order = args.taxonomy_order
    args.sigma_exc = 0.0
    args.sigma_inh = 0.0
    args.quantile_temperature = args.dynamic_temperature
    args.quantile_edge_temperature = args.dynamic_temperature
    result = base.run(args)

    weights = torch.load(result["classifier_weights"], map_location="cpu", weights_only=True)
    result["taxonomy"] = {
        "schema": taxonomy_payload["schema"],
        "source": taxonomy_payload["source"],
        "sha256": hashlib.sha256(args.taxonomy_file.read_bytes()).hexdigest(),
        "order_mode": args.taxonomy_order,
    }
    result["dynamic_mapping"] = {
        "center_hwhm": args.center_hwhm,
        "surround_hwhm": args.surround_hwhm,
        "temperature": args.dynamic_temperature,
        "anchors_detached_each_step": True,
    }
    result["taxonomy_geometry"] = taxonomy_geometry(weights, rows, result["task_order"])
    if args.method in {"ssr_dynamic", "center_dynamic", "surround_dynamic"}:
        result["dynamic_mapping"].update(dynamic_audit(weights, args))
    output = Path(args.output_dir) / "result.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
