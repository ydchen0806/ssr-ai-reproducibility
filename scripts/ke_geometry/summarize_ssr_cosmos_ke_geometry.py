#!/usr/bin/env python3
"""Aggregate paired FT/SSR KE performance and cumulative-update geometry."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path


METRICS = {
    "immediate_efficacy": ("higher", lambda row: float(row["efficacy"])),
    "rephrase": ("higher", lambda row: float(row["rephrase"])),
    "portability": ("higher", lambda row: float(row["portability"])),
    "auxiliary_relation_consistency": ("higher", lambda row: float(row["locality"])),
    "final_history_efficacy": (
        "higher",
        lambda row: float(row["historical_retention"]["final"]["efficacy"]),
    ),
    "pre_edit_output_consistency": (
        "higher",
        lambda row: 100.0
        * float(row["pre_edit_output_consistency"]["final"]["score"]),
    ),
    "update_effective_rank": (
        "higher",
        lambda row: float(row["update_geometry"]["update_effective_rank"]),
    ),
    "update_offdiag_cosine_reduction": (
        "lower",
        lambda row: float(row["update_geometry"]["update_mean_abs_offdiag_cosine"]),
    ),
    "update_rms": ("descriptive", lambda row: float(row["update_geometry"]["update_rms"])),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bootstrap_ci(values: list[float], seed: int, draws: int = 20000) -> list[float]:
    if not values:
        raise ValueError("cannot bootstrap an empty vector")
    rng = random.Random(seed)
    means = sorted(
        sum(values[rng.randrange(len(values))] for _ in values) / len(values)
        for _ in range(draws)
    )
    return [means[int(0.025 * draws)], means[min(draws - 1, int(0.975 * draws))]]


def load_complete(path: Path, expected_edits: int) -> dict:
    row = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "status": "complete",
        "requested_n_edits": expected_edits,
        "attempted": expected_edits,
        "succeeded": expected_edits,
        "failed": 0,
    }
    for key, expected in required.items():
        if row.get(key) != expected:
            raise ValueError(f"{path}: expected {key}={expected!r}, got {row.get(key)!r}")
    if row.get("update_geometry", {}).get("protocol") != "cumulative_edit_update_rows_v1":
        raise ValueError(f"{path}: missing locked update geometry")
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--n-edits", type=int, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, required=True)
    args = parser.parse_args()

    summary = {
        "protocol": "ssr_cosmos_ke_geometry_confirmation_v1",
        "source_commit": args.source_commit,
        "n_edits": args.n_edits,
        "n_prespecified_seeds": len(args.seeds),
        "claim_scope": (
            "paired mechanism replication; performance endpoints are reported without "
            "selection and require a larger confirmation cohort for a standalone main claim"
        ),
        "datasets": {},
    }
    csv_rows = []
    for dataset_index, dataset in enumerate(args.datasets):
        pairs = []
        for seed in args.seeds:
            ft_path = args.result_root / dataset / "ft" / f"seed_{seed}" / "results.json"
            ssr_path = args.result_root / dataset / "ssr" / f"seed_{seed}" / "results.json"
            ft = load_complete(ft_path, args.n_edits)
            ssr = load_complete(ssr_path, args.n_edits)
            if ft.get("data_indices_sha256") != ssr.get("data_indices_sha256"):
                raise ValueError(f"{dataset} seed {seed}: FT/SSR stream mismatch")
            pair = {"seed": seed, "stream_sha256": ft["data_indices_sha256"], "metrics": {}}
            for metric, (direction, getter) in METRICS.items():
                ft_value = getter(ft)
                ssr_value = getter(ssr)
                if direction == "lower":
                    delta = ft_value - ssr_value
                else:
                    delta = ssr_value - ft_value
                pair["metrics"][metric] = {
                    "ft": ft_value,
                    "ssr": ssr_value,
                    "favorable_delta": delta,
                    "direction": direction,
                }
                csv_rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "metric": metric,
                        "direction": direction,
                        "ft": ft_value,
                        "ssr": ssr_value,
                        "favorable_delta": delta,
                    }
                )
            pairs.append(pair)

        aggregates = {}
        for metric in METRICS:
            deltas = [pair["metrics"][metric]["favorable_delta"] for pair in pairs]
            aggregates[metric] = {
                "mean_favorable_delta": sum(deltas) / len(deltas),
                "bootstrap_95_ci": bootstrap_ci(
                    deltas, seed=93000 + dataset_index * 100 + list(METRICS).index(metric)
                ),
                "all_seed_favorable_deltas": deltas,
            }
        summary["datasets"][dataset] = {"paired": pairs, "paired_metrics": aggregates}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"wrote {args.output} ({sha256(args.output)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
