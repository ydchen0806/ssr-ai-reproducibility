#!/usr/bin/env python3
"""Run one matched CUB200 classifier-head mechanism control.

Every method uses the same frozen ResNet-18 feature cache, sequential task
order, optimizer, and KD objective. Only the geometric regularizer changes.
"""

from __future__ import annotations

import argparse
import copy
import json
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
    biocs_loss,
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
    "ssr_kd",
    "center_attraction_kd",
    "surround_repulsion_kd",
    "cosine_orthogonal_kd",
    "prototype_decorrelation_kd",
    "spectral_kd",
)


def regularizer(method: str, weights: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    if method == "kd":
        return weights.sum() * 0.0
    if method == "ssr_kd":
        return biocs_loss(
            weights,
            args.a_exc,
            args.a_inh,
            args.sigma_exc,
            args.sigma_inh,
            args.kernel_family,
        )
    if method == "center_attraction_kd":
        return biocs_loss(
            weights,
            args.a_exc,
            0.0,
            args.sigma_exc,
            args.sigma_inh,
            args.kernel_family,
        )
    if method == "surround_repulsion_kd":
        return biocs_loss(
            weights,
            0.0,
            args.a_inh,
            args.sigma_exc,
            args.sigma_inh,
            args.kernel_family,
        )
    if method == "cosine_orthogonal_kd":
        return cosine_orthogonal_loss(weights)
    if method == "prototype_decorrelation_kd":
        return prototype_decorrelation_loss(weights)
    if method == "spectral_kd":
        return compute_spectral_flatness(weights.float())
    raise ValueError(f"Unsupported method: {method}")


def run(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    requested_device = args.device if torch.cuda.is_available() else "cpu"
    device = torch.device(requested_device)
    x_train, y_train, x_test, y_test = extract_classification_features(args, device)
    num_classes = int(args.max_classes or args.num_classes)
    tasks = make_tasks(num_classes, args.classes_per_task, args.seed, args.class_order)
    if not tasks:
        raise ValueError("No complete CUB tasks were created")

    model = LinearHead(x_train.shape[1], num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    teacher: LinearHead | None = None
    acc_matrix: list[list[float]] = []
    cil_curve: list[float] = []

    for task_id, task_classes in enumerate(tasks):
        loader = subset_loader(x_train, y_train, task_classes, args.batch_size, True)
        for _ in range(args.epochs):
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = F.cross_entropy(logits, yb)
                if args.strength > 0 and args.method != "kd":
                    with torch.amp.autocast("cuda", enabled=False):
                        loss = loss + args.strength * regularizer(
                            args.method, model.classifier.weight.float(), args
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
        "kernel_family": args.kernel_family,
        "a_exc": args.a_exc,
        "a_inh": args.a_inh,
        "sigma_exc": args.sigma_exc,
        "sigma_inh": args.sigma_inh,
        "avg_accuracy": float(np.mean(final)),
        "avg_forgetting": float(np.mean(forgetting)) if forgetting else 0.0,
        "final_cil_accuracy": float(cil_curve[-1]),
        "avg_cil_accuracy": float(np.mean(cil_curve)),
        "til_curve": [float(np.mean(row)) for row in acc_matrix],
        "cil_curve": [float(value) for value in cil_curve],
        "task_order": tasks,
        "classifier_weights": str(weights_path),
        **geometry(weights),
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
    parser.add_argument("--kernel-family", default="gaussian")
    parser.add_argument("--a-exc", type=float, default=1.0)
    parser.add_argument("--a-inh", type=float, default=0.8)
    parser.add_argument("--sigma-exc", type=float, default=0.2)
    parser.add_argument("--sigma-inh", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--feature-batch-size", type=int, default=128)
    parser.add_argument("--no-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(args)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
