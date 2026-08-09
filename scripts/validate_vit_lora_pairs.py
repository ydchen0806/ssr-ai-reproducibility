#!/usr/bin/env python3
"""Validate strict plain/task+SSR pairing for raw-image ViT-LoRA runs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ssr_utils.paired_stats import paired_summary
from ssr_utils.result_schema import OBJECTIVE_COMPONENTS, normalize_objective


METRICS = (
    "avg_accuracy",
    "avg_forgetting",
    "cil_last_accuracy",
    "effective_rank",
    "prototype_overlap",
)
LOWER_IS_BETTER = {"avg_forgetting", "prototype_overlap"}
ALLOWED_METHOD_DIFFERENCES = {
    "recipe",
    "lambda_spatial",
    "lambda_adapter",
    "ssr_start_task",
    "ssr_ramp_tasks",
    "A_exc",
    "A_inh",
    "sigma_exc",
    "sigma_inh",
}
CONTROL_OBJECTIVE = {
    component: component == "task" for component in OBJECTIVE_COMPONENTS
}
TREATMENT_OBJECTIVE = dict(CONTROL_OBJECTIVE, ssr=True)


def load_run(path: Path) -> tuple[dict, dict]:
    record = json.loads((path / "result_record.json").read_text(encoding="utf-8"))
    config = yaml.safe_load((path / "config.yaml").read_text(encoding="utf-8"))
    return record, config


def validate_pair(control_path: Path, treatment_path: Path) -> dict:
    control, control_config = load_run(control_path)
    treatment, treatment_config = load_run(treatment_path)
    for field in (
        "dataset",
        "model",
        "seed",
        "dataset_hash",
        "pairing_hash",
        "initial_model_hash",
        "optimizer_steps",
        "training_batches",
    ):
        if control.get(field) != treatment.get(field):
            raise ValueError(f"Unmatched ViT-LoRA pair: {field} differs")
    if int(control["optimizer_steps"]) <= 0:
        raise ValueError("optimizer_steps must be positive")
    if int(control["training_batches"]) != int(control["optimizer_steps"]):
        raise ValueError("ViT-LoRA confirmation requires one optimizer step per batch")
    for name, record in (("control", control), ("treatment", treatment)):
        final_hash = record.get("final_model_hash")
        if not isinstance(final_hash, str) or len(final_hash) != 64:
            raise ValueError(f"{name} is missing a valid final_model_hash")
        if final_hash == record["initial_model_hash"]:
            raise ValueError(f"{name} model did not change during training")
    if control["final_model_hash"] == treatment["final_model_hash"]:
        raise ValueError("SSR treatment produced the same final model as task-only control")
    if normalize_objective(control["objective"]) != CONTROL_OBJECTIVE:
        raise ValueError("Control is not task-loss only")
    if normalize_objective(treatment["objective"]) != TREATMENT_OBJECTIVE:
        raise ValueError("Treatment is not task-loss+SSR only")
    if control.get("active_regularizer") != "none" or float(
        control.get("active_regularizer_weight", 0.0)
    ) != 0.0:
        raise ValueError("Control unexpectedly activates a regularizer")
    if treatment.get("active_regularizer") != "ssr" or float(
        treatment.get("active_regularizer_weight", 0.0)
    ) <= 0.0:
        raise ValueError("SSR treatment did not activate a positive-weight SSR term")
    if control["recipe"] != "plain" or treatment["recipe"] != "ssr_only":
        raise ValueError("Expected plain and ssr_only recipes")
    if control["distance_mapping"] != "none" or treatment["distance_mapping"] != "cosine":
        raise ValueError("Unexpected distance mapping")

    if control_config["model"] != treatment_config["model"]:
        raise ValueError("Model configs differ")
    if control_config["dataset"] != treatment_config["dataset"]:
        raise ValueError("Dataset configs differ")
    if control_config["training"] != treatment_config["training"]:
        raise ValueError("Training budgets differ")
    control_method = control_config["method"]
    treatment_method = treatment_config["method"]
    if any(
        float(control_method.get(field, 0.0)) != 0.0
        for field in ("lambda_spatial", "lambda_adapter", "lambda_spectral")
    ):
        raise ValueError("Control contains a non-task regularizer")
    if "lambda_spectral" not in treatment_method or float(
        treatment_method["lambda_spectral"]
    ) != 0.0:
        raise ValueError("Treatment must explicitly set lambda_spectral=0")
    if not any(
        float(treatment_method.get(field, 0.0)) > 0.0
        for field in ("lambda_spatial", "lambda_adapter")
    ):
        raise ValueError("Treatment must enable at least one declared SSR penalty")
    extra_active_lambdas = sorted(
        field
        for field, value in treatment_method.items()
        if field.startswith("lambda_")
        and field not in {"lambda_spatial", "lambda_adapter", "lambda_spectral"}
        and float(value) != 0.0
    )
    if extra_active_lambdas:
        raise ValueError(
            "Treatment contains non-SSR regularization coefficients: "
            f"{extra_active_lambdas}"
        )
    method_differences = {
        field
        for field in set(control_method) | set(treatment_method)
        if control_method.get(field) != treatment_method.get(field)
    }
    unexpected_differences = method_differences - ALLOWED_METHOD_DIFFERENCES
    if unexpected_differences:
        raise ValueError(
            "Method configs differ outside the SSR treatment allowlist: "
            f"{sorted(unexpected_differences)}"
        )

    metrics = {}
    for metric in METRICS:
        if metric not in control["metrics"] or metric not in treatment["metrics"]:
            raise ValueError(f"Missing paired metric: {metric}")
        control_value = float(control["metrics"][metric])
        treatment_value = float(treatment["metrics"][metric])
        metrics[f"{metric}_control"] = control_value
        metrics[f"{metric}_treatment"] = treatment_value
        if metric in LOWER_IS_BETTER:
            metrics[f"{metric}_reduction"] = (
                control_value - treatment_value
            )
        else:
            metrics[f"{metric}_delta"] = treatment_value - control_value
    return {
        "dataset": control["dataset"],
        "seed": int(control["seed"]),
        "model": control["model"],
        "lora_rank": int(control_config["method"]["lora_rank"]),
        "lora_alpha": float(control_config["method"]["lora_alpha"]),
        "optimizer_steps": int(control["optimizer_steps"]),
        "training_batches": int(control["training_batches"]),
        "initial_model_hash": control["initial_model_hash"],
        "pairing_hash": control["pairing_hash"],
        **metrics,
    }


def summarize(rows: list[dict]) -> dict:
    output = {}
    for dataset in sorted({row["dataset"] for row in rows}):
        items = [row for row in rows if row["dataset"] == dataset]
        stats = {
            "n": len(items),
            "control_arm": "task",
            "treatment_arm": "task_ssr",
            "metrics": {},
        }
        for metric in METRICS:
            summary = paired_summary(
                [row[f"{metric}_control"] for row in items],
                [row[f"{metric}_treatment"] for row in items],
                higher_is_better=metric not in LOWER_IS_BETTER,
            )
            stats["metrics"][metric] = {
                "higher_is_better": metric not in LOWER_IS_BETTER,
                **summary.as_dict(),
            }
        stats["dual_wins"] = sum(
            row["avg_accuracy_delta"] > 0
            and row["avg_forgetting_reduction"] > 0
            for row in items
        )
        output[dataset] = stats
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    rows = []
    for dataset in args.datasets:
        for seed in args.seeds:
            rows.append(
                validate_pair(
                    args.results_root / dataset / "task" / f"seed_{seed}",
                    args.results_root / dataset / "task_ssr" / f"seed_{seed}",
                )
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    temporary = args.output_dir / f".pairs.csv.tmp.{os.getpid()}"
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, args.output_dir / "pairs.csv")
    (args.output_dir / "summary.json").write_text(
        json.dumps(summarize(rows), indent=2), encoding="utf-8"
    )
    print(json.dumps({"validated_pairs": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
