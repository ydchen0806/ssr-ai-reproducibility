#!/usr/bin/env python3
"""Paired summary for a frozen low-rank SSR confirmation cohort."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


CHECKPOINTS = (1, 2, 5, 10, 25, 50, 100)
PERFORMANCE_METRICS = (
    "history_efficacy",
    "pre_edit_output_consistency",
    "immediate_efficacy",
    "immediate_rephrase",
    "immediate_portability",
    "immediate_locality_target_consistency",
)
GEOMETRY_METRICS = (
    "effective_rank",
    "effective_rank_fraction",
    "near_row_cosine",
    "surround_row_cosine",
    "center_surround_contrast",
    "kernel_alignment",
)
PROTOCOL = "lowrank_lora_ssr_v1"


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def bootstrap_ci(values: list[float], key: str, draws: int = 10000) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size < 2 or not np.isfinite(array).all():
        raise ValueError(f"Invalid paired bootstrap vector for {key}: {values}")
    seed = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(draws, array.size))
    means = array[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def summarize_values(baseline: list[float], ssr: list[float], key: str) -> dict:
    deltas = [right - left for left, right in zip(baseline, ssr)]
    return {
        "baseline_mean": float(np.mean(baseline)),
        "baseline_ci95": bootstrap_ci(baseline, key + ":baseline"),
        "ssr_mean": float(np.mean(ssr)),
        "ssr_ci95": bootstrap_ci(ssr, key + ":ssr"),
        "delta_mean": float(np.mean(deltas)),
        "delta_ci95": bootstrap_ci(deltas, key + ":delta"),
        "paired_deltas": deltas,
    }


def checkpoint(record: dict, edit_count: int) -> dict:
    rows = [row for row in record["checkpoints"] if int(row["after_edits"]) == edit_count]
    if len(rows) != 1:
        raise ValueError(f"Expected one checkpoint at {edit_count}, found {len(rows)}")
    return rows[0]


def performance_value(row: dict, metric: str) -> float:
    if metric == "history_efficacy":
        return float(row["history"]["efficacy"])
    if metric == "pre_edit_output_consistency":
        return float(row["pre_edit_output_consistency"]["score"])
    immediate_key = metric.removeprefix("immediate_")
    return 100.0 * float(row["immediate"][immediate_key])


def validate_pair(baseline: dict, ssr: dict, seed: int, rank: int,
                  learning_rate: float, lora_steps: int, ssr_lambda: float) -> None:
    for record in (baseline, ssr):
        if record.get("protocol") != PROTOCOL:
            raise ValueError("Unexpected protocol")
        if record.get("status") != "complete" or int(record["stream_seed"]) != seed:
            raise ValueError(f"Incomplete or mismatched seed {seed}")
        if (
            int(record["n_edits"]) != 100
            or int(record["lora"]["rank"]) != rank
            or not np.isclose(float(record["lora"]["learning_rate"]), learning_rate)
            or int(record["lora"]["num_steps"]) != lora_steps
        ):
            raise ValueError(
                f"This confirmation requires rank {rank}, lr {learning_rate}, "
                f"{lora_steps} steps, and 100 edits"
            )
        if [int(row["after_edits"]) for row in record["checkpoints"]] != list(CHECKPOINTS):
            raise ValueError("Unexpected checkpoint schedule")
    if float(baseline["ssr"]["lambda"]) != 0.0 or not np.isclose(
            float(ssr["ssr"]["lambda"]), ssr_lambda):
        raise ValueError(
            f"The independent confirmation must compare lambda 0 with {ssr_lambda}"
        )
    for key in (
        "model_label",
        "model_path",
        "dataset_sha256",
        "stream_manifest_sha256",
        "n_edits",
        "hparams_sha256",
        "lora",
        "history_checkpoints",
        "history_max_samples",
        "pre_edit_control_count",
        "output_preservation_anchor",
    ):
        if baseline[key] != ssr[key]:
            raise ValueError(f"Paired configuration mismatch for seed {seed}: {key}")


def history_auc(record: dict) -> float:
    x = np.asarray(CHECKPOINTS, dtype=np.float64)
    y = np.asarray([checkpoint(record, value)["history"]["efficacy"] for value in CHECKPOINTS])
    integrate = getattr(np, "trapezoid", None) or np.trapz
    return float(integrate(y, x) / (x[-1] - x[0]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--lora-steps", type=int, required=True)
    parser.add_argument("--ssr-lambda", type=float, required=True)
    parser.add_argument("--frozen-from", required=True)
    args = parser.parse_args()
    seeds = sorted(args.seeds)
    if len(seeds) != 10 or len(set(seeds)) != 10:
        raise ValueError("Confirmation requires exactly ten distinct prespecified seeds")

    records: dict[float, dict[int, dict]] = {0.0: {}, args.ssr_lambda: {}}
    for path in sorted(args.result_root.rglob("results.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        coefficient = float(record["ssr"]["lambda"])
        seed = int(record["stream_seed"])
        if coefficient not in records or seed not in seeds:
            raise ValueError(f"Undeclared confirmation arm: {path}")
        if seed in records[coefficient]:
            raise ValueError(f"Duplicate confirmation arm: lambda={coefficient}, seed={seed}")
        records[coefficient][seed] = record
    if sorted(records[0.0]) != seeds or sorted(records[args.ssr_lambda]) != seeds:
        raise ValueError("Incomplete ten-seed paired confirmation")
    for seed in seeds:
        validate_pair(records[0.0][seed], records[args.ssr_lambda][seed], seed,
                      args.rank, args.learning_rate, args.lora_steps, args.ssr_lambda)

    curves = []
    csv_rows = []
    for edit_count in CHECKPOINTS:
        performance = {}
        geometry = {}
        for metric in PERFORMANCE_METRICS:
            baseline = [performance_value(checkpoint(records[0.0][seed], edit_count), metric) for seed in seeds]
            ssr = [performance_value(checkpoint(records[args.ssr_lambda][seed], edit_count), metric)
                   for seed in seeds]
            performance[metric] = summarize_values(baseline, ssr, f"{edit_count}:{metric}")
        for metric in GEOMETRY_METRICS:
            baseline = [float(checkpoint(records[0.0][seed], edit_count)["lora_geometry"][metric]) for seed in seeds]
            ssr = [float(checkpoint(records[args.ssr_lambda][seed], edit_count)
                         ["lora_geometry"][metric]) for seed in seeds]
            geometry[metric] = summarize_values(baseline, ssr, f"{edit_count}:geometry:{metric}")
        curves.append({"after_edits": edit_count, "performance": performance, "geometry": geometry})
        for family, values in (("performance", performance), ("geometry", geometry)):
            for metric, summary in values.items():
                csv_rows.append(
                    {
                        "edit_count": edit_count,
                        "family": family,
                        "metric": metric,
                        "baseline_mean": summary["baseline_mean"],
                        "ssr_mean": summary["ssr_mean"],
                        "delta_mean": summary["delta_mean"],
                        "delta_ci95_low": summary["delta_ci95"][0],
                        "delta_ci95_high": summary["delta_ci95"][1],
                    }
                )

    final = curves[-1]["performance"]
    baseline_auc = [history_auc(records[0.0][seed]) for seed in seeds]
    ssr_auc = [history_auc(records[args.ssr_lambda][seed]) for seed in seeds]
    history_auc_summary = summarize_values(baseline_auc, ssr_auc, "history_auc")
    runtime = summarize_values(
        [float(records[0.0][seed]["elapsed_s"]) for seed in seeds],
        [float(records[args.ssr_lambda][seed]["elapsed_s"]) for seed in seeds],
        "runtime",
    )
    memory = summarize_values(
        [float(records[0.0][seed]["peak_cuda_memory_mb"]) for seed in seeds],
        [float(records[args.ssr_lambda][seed]["peak_cuda_memory_mb"]) for seed in seeds],
        "memory",
    )
    payload = {
        "schema_version": "lowrank_lora_ssr_confirmation_v1",
        "protocol": "independent_frozen_effloc_lr_100_edit_confirmation",
        "model": "Qwen2.5-7B-Instruct",
        "dataset": "ZsRE",
        "method": f"EasyEdit LoRA rank {args.rank}, lr {args.learning_rate:g}",
        "rank": args.rank,
        "learning_rate": args.learning_rate,
        "lora_steps": args.lora_steps,
        "frozen_ssr_lambda": args.ssr_lambda,
        "frozen_from": args.frozen_from,
        "confirmation_seeds": seeds,
        "n_pairs": len(seeds),
        "n_edits": 100,
        "checkpoints": list(CHECKPOINTS),
        "curves": curves,
        "normalized_history_auc": history_auc_summary,
        "runtime_seconds": runtime,
        "peak_cuda_memory_mb": memory,
        "primary_claim_gate": {
            "immediate_efficacy_delta_ci95_above_zero": final["immediate_efficacy"]["delta_ci95"][0] > 0,
            "output_consistency_locality_delta_ci95_above_zero": (
                final["pre_edit_output_consistency"]["delta_ci95"][0] > 0
            ),
            "dual_positive_primary_gate": (
                final["immediate_efficacy"]["delta_ci95"][0] > 0
                and final["pre_edit_output_consistency"]["delta_ci95"][0] > 0
            ),
        },
        "reporting": (
            "Rank, learning rate, step count, coefficient, edit horizon, checkpoints, and the two primary "
            "endpoints (immediate efficacy and pre-edit output-distribution consistency locality) were frozen "
            "before these ten new streams. All paired outcomes are retained. Target-consistency locality, "
            "rephrase, portability, retention, effective rank, and kernel alignment are secondary endpoints."
        ),
    }
    atomic_json(args.output, payload)
    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.csv_output.with_suffix(args.csv_output.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    temporary.replace(args.csv_output)
    print(json.dumps(payload["primary_claim_gate"], sort_keys=True))


if __name__ == "__main__":
    main()
