#!/usr/bin/env python3
"""Aggregate all predeclared v24 PASCAL VOC 10-1 sequential streams."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path


STEPS = tuple(range(1, 11))
ENDPOINTS = ("all_miou", "old_miou", "new_miou", "old_boundary_iou", "old_boundary_margin")
TRAJECTORY_ENDPOINTS = ("all_miou", "old_miou", "old_boundary_iou", "old_boundary_margin")


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_trial(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "plop_ssr_lowrank_v24_sequential_trial_v1" or payload.get("status") != "complete":
        raise ValueError(f"invalid or incomplete trial: {path}")
    return payload


def finite(value, label: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"missing finite {label}")
    return float(value)


def step_metrics(raw: dict, step: int) -> dict[str, float]:
    class_iou = raw.get("Class IoU", {})
    old_count = 9 + step
    old = [finite(class_iou.get(str(index), class_iou.get(index)), f"step {step} class {index} IoU") for index in range(1, old_count + 1)]
    new_index = old_count + 1
    return {
        "all_miou": finite(raw.get("Mean IoU"), f"step {step} Mean IoU"),
        "old_miou": sum(old) / len(old),
        "new_miou": finite(class_iou.get(str(new_index), class_iou.get(new_index)), f"step {step} new class IoU"),
        "old_boundary_iou": finite(raw.get("Old Boundary IoU"), f"step {step} Old Boundary IoU"),
        "old_boundary_margin": finite(raw.get("Old Boundary Logit Margin"), f"step {step} old boundary margin"),
    }


def trial_trajectory(trial: dict) -> dict[int, dict[str, float]]:
    records = trial.get("step_metrics", [])
    if len(records) != len(STEPS):
        raise ValueError(f"expected {len(STEPS)} incremental records, found {len(records)}")
    return {step: step_metrics(records[position], step) for position, step in enumerate(STEPS)}


def bootstrap_ci(values: list[float], draws: int = 20_000) -> list[float]:
    generator = random.Random(20260825)
    means = [sum(generator.choice(values) for _ in values) / len(values) for _ in range(draws)]
    means.sort()
    return [means[int(0.025 * (draws - 1))], means[int(0.975 * (draws - 1))]]


def paired_summary(values: list[float]) -> dict:
    return {
        "paired_delta_mean": sum(values) / len(values),
        "bootstrap_95_ci": bootstrap_ci(values),
        "per_seed_deltas": values,
    }


def peak_memory(trial: dict) -> float:
    values = [finite(value, "peak GPU memory") for value in trial.get("peak_gpu_memory_mib", {}).values()]
    if not values:
        raise ValueError("trial has no GPU-memory samples")
    return max(values)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--frozen-selection", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selection = json.loads(args.frozen_selection.read_text(encoding="utf-8"))
    selected = selection.get("selected")
    if selection.get("status") != "selected" or not isinstance(selected, dict):
        raise ValueError("the v23 frozen development selection is unavailable")
    rank = int(selected["rank"])
    tag = str(selected["tag"])
    arm_dirs = {
        "plop": "plop",
        "lowrank": f"lowrank_r{rank}",
        "lowrank_ssr": f"lowrank_ssr_{tag}",
    }
    root = args.run_root.resolve() / "confirmation" / "10-1"
    trials = {arm: {} for arm in arm_dirs}
    trajectories = {arm: {} for arm in arm_dirs}
    for seed in args.seeds:
        for arm, directory in arm_dirs.items():
            trial = load_trial(root / directory / f"seed_{seed}" / "trial.json")
            if trial.get("seed") != seed or trial.get("task") != "10-1" or trial.get("arm") != arm:
                raise ValueError(f"mismatched {arm} trial for seed {seed}")
            trials[arm][seed] = trial
            trajectories[arm][seed] = trial_trajectory(trial)
        parent_hashes = {trials[arm][seed].get("parent_checkpoint_sha256") for arm in arm_dirs}
        if len(parent_hashes) != 1 or None in parent_hashes:
            raise ValueError(f"arms do not share one exact step-0 checkpoint for seed {seed}")
        if trials["plop"][seed]["low_rank_classifier"]["rank"] != 0:
            raise ValueError(f"PLOP rank is not zero for seed {seed}")
        for arm in ("lowrank", "lowrank_ssr"):
            low_rank = trials[arm][seed]["low_rank_classifier"]
            if int(low_rank["rank"]) != rank or float(low_rank["alpha"]) != float(selected["alpha"]):
                raise ValueError(f"low-rank settings differ from the frozen selection for {arm}, seed {seed}")
        expected_ssr = {
            key: selected[key] for key in (
                "lambda", "normalization", "a_exc", "a_inh", "sigma_exc", "sigma_inh",
                "distance", "start_step", "target_scope", "gradient_gate", "warmup_epochs",
                "repulsion_margin", "target",
            )
        }
        if trials["lowrank_ssr"][seed].get("ssr") != expected_ssr:
            raise ValueError(f"SSR settings differ from the frozen v23 selection for seed {seed}")

    comparisons = {
        "lowrank_ssr_vs_lowrank": ("lowrank", "lowrank_ssr"),
        "lowrank_ssr_vs_plop": ("plop", "lowrank_ssr"),
        "lowrank_vs_plop": ("plop", "lowrank"),
    }
    report = {
        "schema_version": "plop_ssr_lowrank_v24_10_1_summary_v1",
        "status": "complete",
        "task": "10-1",
        "confirmation_seeds": args.seeds,
        "frozen_selection_path": str(args.frozen_selection.resolve()),
        "frozen_selection": selection,
        "per_seed": [],
        "final_comparisons": {},
        "trajectory_auc_comparisons": {},
        "per_step_comparisons": {},
        "efficiency": {},
    }
    for seed in args.seeds:
        report["per_seed"].append({
            "seed": seed,
            "parent_checkpoint_sha256": trials["plop"][seed]["parent_checkpoint_sha256"],
            **{
                arm: {
                    "final": trajectories[arm][seed][STEPS[-1]],
                    "trajectory": {str(step): trajectories[arm][seed][step] for step in STEPS},
                }
                for arm in arm_dirs
            },
        })
    for name, (left, right) in comparisons.items():
        report["final_comparisons"][name] = {
            endpoint: paired_summary([
                trajectories[right][seed][STEPS[-1]][endpoint]
                - trajectories[left][seed][STEPS[-1]][endpoint]
                for seed in args.seeds
            ])
            for endpoint in ENDPOINTS
        }
        report["trajectory_auc_comparisons"][name] = {
            endpoint: paired_summary([
                sum(trajectories[right][seed][step][endpoint] - trajectories[left][seed][step][endpoint] for step in STEPS) / len(STEPS)
                for seed in args.seeds
            ])
            for endpoint in TRAJECTORY_ENDPOINTS
        }
        report["per_step_comparisons"][name] = {
            str(step): {
                endpoint: paired_summary([
                    trajectories[right][seed][step][endpoint] - trajectories[left][seed][step][endpoint]
                    for seed in args.seeds
                ])
                for endpoint in ENDPOINTS
            }
            for step in STEPS
        }
    for arm in arm_dirs:
        report["efficiency"][arm] = {
            "mean_incremental_wall_time_s": sum(finite(trials[arm][seed].get("wall_time_s"), "wall time") for seed in args.seeds) / len(args.seeds),
            "mean_peak_gpu_memory_mib": sum(peak_memory(trials[arm][seed]) for seed in args.seeds) / len(args.seeds),
        }
    atomic_json(args.output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
