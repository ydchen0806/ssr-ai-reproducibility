#!/usr/bin/env python3
"""Select one matched LoRA rank, learning rate, and SSR coefficient."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


PROTOCOL = "lowrank_lora_ssr_v1"


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def vector(record: dict) -> dict[str, float]:
    immediate = record["immediate"]
    return {
        "efficacy": 100.0 * float(immediate["efficacy"]),
        "locality": float(record["final_pre_edit_output_consistency"]),
        "locality_target_consistency": 100.0 * float(immediate["locality_target_consistency"]),
        "rephrase": 100.0 * float(immediate["rephrase"]),
        "portability": 100.0 * float(immediate["portability"]),
        "final_history": float(record["final_history_efficacy"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--ranks", type=int, nargs="+", required=True)
    parser.add_argument("--learning-rates", type=float, nargs="+", required=True)
    parser.add_argument("--lora-steps", type=int, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--lambdas", type=float, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ranks = sorted(set(args.ranks))
    learning_rates = sorted(set(args.learning_rates))
    seeds = sorted(set(args.seeds))
    coefficients = sorted(set(args.lambdas))
    if (
        len(seeds) != 4 or len(ranks) < 2 or len(learning_rates) < 2
        or learning_rates[0] <= 0 or args.lora_steps < 1
        or not coefficients or coefficients[0] <= 0
    ):
        raise ValueError(
            "Require four seeds, at least two ranks/rates, positive steps, and positive coefficients"
        )

    records: dict[tuple[int, float, float], dict[int, dict]] = defaultdict(dict)
    for path in sorted(args.result_root.rglob("results.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("protocol") != PROTOCOL or record.get("status") != "complete":
            raise ValueError(f"Invalid development record: {path}")
        if int(record["n_edits"]) != 100:
            raise ValueError(f"Development record is not 100 edits: {path}")
        rank = int(record["lora"]["rank"])
        learning_rate = float(record["lora"]["learning_rate"])
        if int(record["lora"]["num_steps"]) != args.lora_steps:
            raise ValueError(f"Unexpected LoRA step count in {path}")
        coefficient = float(record["ssr"]["lambda"])
        seed = int(record["stream_seed"])
        key = (rank, learning_rate, coefficient)
        if seed in records[key]:
            raise ValueError(
                f"Duplicate development arm: rank={rank}, lr={learning_rate}, "
                f"lambda={coefficient}, seed={seed}"
            )
        records[key][seed] = record

    for rank in ranks:
        for learning_rate in learning_rates:
            for coefficient in [0.0, *coefficients]:
                if sorted(records.get((rank, learning_rate, coefficient), {})) != seeds:
                    raise ValueError(
                        f"Incomplete development cell: rank={rank}, lr={learning_rate}, "
                        f"lambda={coefficient}"
                    )

    scaffold = None
    for cell in records.values():
        for record in cell.values():
            observed = record.get("output_preservation_anchor", {"enabled": False})
            if scaffold is None:
                scaffold = observed
            elif observed != scaffold:
                raise ValueError("Output-preservation scaffold changed inside the matched matrix")

    candidates = []
    for rank in ranks:
        for learning_rate in learning_rates:
            for coefficient in coefficients:
                per_seed = {}
                for seed in seeds:
                    baseline = vector(records[(rank, learning_rate, 0.0)][seed])
                    candidate = vector(records[(rank, learning_rate, coefficient)][seed])
                    per_seed[str(seed)] = {
                        metric: candidate[metric] - baseline[metric] for metric in baseline
                    }
                means = {
                    metric: float(np.mean([per_seed[str(seed)][metric] for seed in seeds]))
                    for metric in (
                        "efficacy", "locality", "locality_target_consistency",
                        "rephrase", "portability", "final_history",
                    )
                }
                efficacy_nonnegative = sum(per_seed[str(seed)]["efficacy"] >= 0 for seed in seeds)
                locality_nonnegative = sum(per_seed[str(seed)]["locality"] >= 0 for seed in seeds)
                eligible = (
                    means["efficacy"] >= 5.0
                    and means["locality"] >= 0.5
                    and efficacy_nonnegative >= 3
                    and locality_nonnegative >= 3
                    and means["locality_target_consistency"] >= -0.25
                    and means["rephrase"] >= -1.0
                    and means["portability"] >= -1.0
                    and means["final_history"] >= -0.5
                )
                balance_score = min(means["efficacy"] / 5.0, means["locality"] / 0.5)
                candidates.append({
                    "rank": rank,
                    "learning_rate": learning_rate,
                    "lora_steps": args.lora_steps,
                    "lambda": coefficient,
                    "mean_delta_pp": means,
                    "efficacy_nonnegative_seed_count": efficacy_nonnegative,
                    "locality_nonnegative_seed_count": locality_nonnegative,
                    "eligible": eligible,
                    "balance_score": balance_score,
                    "per_seed_delta_pp": per_seed,
                })

    eligible = [row for row in candidates if row["eligible"]]
    selected = None
    if eligible:
        selected = max(
            eligible,
            key=lambda row: (
                row["balance_score"],
                row["mean_delta_pp"]["efficacy"] + row["mean_delta_pp"]["locality"],
                -row["rank"],
                -row["learning_rate"],
                -row["lambda"],
            ),
        )
    atomic_json(args.output, {
        "schema_version": "lowrank_lora_ssr_development_selection_v1",
        "status": "selected" if selected else "no_eligible_configuration",
        "development_seeds": seeds,
        "development_edits": 100,
        "candidate_ranks": ranks,
        "candidate_learning_rates": learning_rates,
        "lora_steps": args.lora_steps,
        "candidate_lambdas": coefficients,
        "output_preservation_anchor": scaffold,
        "primary_endpoints": ["immediate_efficacy", "pre_edit_output_consistency_locality"],
        "selection_rule": (
            "Require mean efficacy gain >=5 pp and pre-edit output-consistency locality gain >=0.5 pp; "
            "require at least three of four nonnegative paired deltas for both primary endpoints; "
            "require target-consistency locality >=-0.25 pp, rephrase and portability >=-1 pp, and "
            "final-history efficacy >=-0.5 pp; maximize the smaller normalized primary gain."
        ),
        "selected": selected,
        "candidates": candidates,
        "confirmation_was_not_read": True,
    })


if __name__ == "__main__":
    main()
