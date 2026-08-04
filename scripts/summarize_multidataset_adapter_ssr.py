#!/usr/bin/env python3
"""Validate and summarize matched multi-dataset adapter SSR pairs."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


METRICS = (
    "avg_accuracy",
    "avg_forgetting_reduction",
    "cil_last_accuracy",
    "effective_rank",
    "prototype_overlap_reduction",
    "adapter_basis_overlap_reduction",
)


def interval(values: list[float]) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    mean = float(array.mean())
    if array.size < 2:
        return mean, mean, mean
    half = float(stats.t.ppf(0.975, array.size - 1) * stats.sem(array))
    return mean, mean - half, mean + half


def load_pairs(root: Path) -> list[dict]:
    rows: list[dict] = []
    seen: set[tuple] = set()
    for path in sorted(root.rglob("pair.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if not record.get("matched_budget"):
            raise ValueError(f"Unmatched record: {path}")
        control = record["control"]
        treatment = record["treatment"]
        key = (
            control["dataset"],
            int(control["seed"]),
            int(control["rank"]),
            float(treatment["ssr_scale"]),
        )
        if key in seen:
            raise ValueError(f"Duplicate pair {key}: {path}")
        seen.add(key)
        rows.append(
            {
                "dataset": key[0],
                "seed": key[1],
                "rank": key[2],
                "ssr_scale": key[3],
                "path": str(path),
                **{metric: float(record["delta"][metric]) for metric in METRICS},
            }
        )
    if not rows:
        raise ValueError(f"No pair.json records found under {root}")
    return rows


def validate_grid(
    rows: list[dict], expected_datasets: list[str], expected_seeds: list[int]
) -> None:
    if not expected_datasets and not expected_seeds:
        return
    by_condition: dict[tuple, set[int]] = defaultdict(set)
    for row in rows:
        by_condition[(row["dataset"], row["rank"], row["ssr_scale"])].add(row["seed"])
    datasets = expected_datasets or sorted({row["dataset"] for row in rows})
    ranks = sorted({row["rank"] for row in rows})
    scales = sorted({row["ssr_scale"] for row in rows})
    expected_seed_set = set(expected_seeds)
    for dataset in datasets:
        for rank in ranks:
            for scale in scales:
                actual = by_condition.get((dataset, rank, scale), set())
                if expected_seeds and actual != expected_seed_set:
                    raise ValueError(
                        f"Incomplete grid dataset={dataset} rank={rank} scale={scale}: "
                        f"seeds={sorted(actual)}, expected={sorted(expected_seed_set)}"
                    )


def summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["rank"], row["ssr_scale"])].append(row)
    output: list[dict] = []
    for (dataset, rank, scale), items in sorted(grouped.items()):
        summary = {
            "dataset": dataset,
            "rank": rank,
            "ssr_scale": scale,
            "n": len(items),
            "dual_wins": sum(
                row["avg_accuracy"] > 0 and row["avg_forgetting_reduction"] > 0
                for row in items
            ),
        }
        for metric in METRICS:
            mean, low, high = interval([row[metric] for row in items])
            summary[f"{metric}_mean"] = mean
            summary[f"{metric}_ci_low"] = low
            summary[f"{metric}_ci_high"] = high
        output.append(summary)
    return output


def select_global_scale(summary: list[dict]) -> dict:
    by_scale: dict[tuple[int, float], list[dict]] = defaultdict(list)
    for row in summary:
        by_scale[(int(row["rank"]), float(row["ssr_scale"]))].append(row)
    candidates = []
    for (rank, scale), rows in by_scale.items():
        dual_datasets = sum(
            row["avg_accuracy_mean"] > 0
            and row["avg_forgetting_reduction_mean"] > 0
            for row in rows
        )
        dual_wins = sum(int(row["dual_wins"]) for row in rows)
        total = sum(int(row["n"]) for row in rows)
        combined = float(
            np.mean(
                [
                    row["avg_accuracy_mean"]
                    + row["avg_forgetting_reduction_mean"]
                    for row in rows
                ]
            )
        )
        candidates.append(
            {
                "rank": rank,
                "ssr_scale": scale,
                "dataset_dual_positive": dual_datasets,
                "datasets": len(rows),
                "seed_dual_wins": dual_wins,
                "seed_pairs": total,
                "combined_endpoint_score": combined,
            }
        )
    selected = max(
        candidates,
        key=lambda row: (
            row["dataset_dual_positive"],
            row["seed_dual_wins"],
            row["combined_endpoint_score"],
            -row["ssr_scale"],
        ),
    )
    return {
        "schema_version": 1,
        "selection_cohort": "development only",
        "selection_rule": (
            "maximize datasets with dual-positive paired means; then paired-seed "
            "dual wins; then mean delta-AA plus forgetting reduction; then smaller scale"
        ),
        "selected": selected,
        "candidates": sorted(candidates, key=lambda row: (row["rank"], row["ssr_scale"])),
    }


def write_outputs(rows: list[dict], summary: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, records in (("pairs.csv", rows), ("summary.csv", summary)):
        temporary = output_dir / f".{name}.tmp.{os.getpid()}"
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        os.replace(temporary, output_dir / name)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-datasets", nargs="*", default=[])
    parser.add_argument("--expected-seeds", nargs="*", type=int, default=[])
    parser.add_argument("--select-lock", action="store_true")
    args = parser.parse_args()

    rows = load_pairs(args.input_root)
    validate_grid(rows, args.expected_datasets, args.expected_seeds)
    summary = summarize(rows)
    write_outputs(rows, summary, args.output_dir)
    if args.select_lock:
        lock = select_global_scale(summary)
        temporary = args.output_dir / f".selection_lock.json.tmp.{os.getpid()}"
        temporary.write_text(json.dumps(lock, indent=2), encoding="utf-8")
        os.replace(temporary, args.output_dir / "selection_lock.json")
        print(json.dumps(lock["selected"], sort_keys=True))
    else:
        print(json.dumps({"pairs": len(rows), "conditions": len(summary)}))


if __name__ == "__main__":
    main()
