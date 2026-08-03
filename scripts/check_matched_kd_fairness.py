#!/usr/bin/env python3
"""Verify that fixed-KD comparisons change only the selected regularizer."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


EXPECTED = {"kd", "kd_ewc", "kd_mas", "kd_si", "kd_ssr"}


def check(
    root: Path,
    *,
    expected_datasets: set[str] | None = None,
    expected_seeds: set[int] | None = None,
) -> dict:
    groups: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    invalid = []
    for path in sorted(root.rglob("result_record.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("task_family") != "classification":
            continue
        recipe = record.get("recipe")
        if recipe not in EXPECTED:
            continue
        key = (record["dataset"], record["model"], int(record["seed"]))
        groups[key].append({"path": str(path), **record})

    for key, rows in sorted(groups.items()):
        recipes = {row["recipe"] for row in rows}
        if recipes != EXPECTED:
            invalid.append(
                {
                    "identity": key,
                    "error": "incomplete_recipe_set",
                    "present": sorted(recipes),
                    "missing": sorted(EXPECTED - recipes),
                }
            )
        hashes = {row.get("pairing_hash") for row in rows}
        if None in hashes or len(hashes) != 1:
            invalid.append(
                {
                    "identity": key,
                    "error": "scaffold_hash_mismatch",
                    "pairing_hashes": sorted(str(value) for value in hashes),
                }
            )
        teacher_identities = {
            (row.get("teacher_protocol"), row.get("teacher_trajectory_hash"))
            for row in rows
        }
        valid_teacher_identity = (
            len(teacher_identities) == 1
            and next(iter(teacher_identities))[0] == "locked_kd_only_trajectory_v1"
            and isinstance(next(iter(teacher_identities))[1], str)
            and len(next(iter(teacher_identities))[1]) == 64
        )
        if not valid_teacher_identity:
            invalid.append(
                {
                    "identity": key,
                    "error": "teacher_trajectory_mismatch",
                    "teacher_identities": sorted(
                        [str(value) for value in teacher_identities]
                    ),
                }
            )
        kd_settings = {
            (
                row["objective"].get("kd", False),
                row["dataset_hash"],
            )
            for row in rows
        }
        if kd_settings != {(True, rows[0]["dataset_hash"])}:
            invalid.append(
                {"identity": key, "error": "kd_or_dataset_identity_mismatch"}
            )

    if expected_datasets is not None and expected_seeds is not None:
        observed = {(dataset, seed) for dataset, _, seed in groups}
        expected = {
            (dataset, seed)
            for dataset in expected_datasets
            for seed in expected_seeds
        }
        for dataset, seed in sorted(expected - observed):
            invalid.append(
                {
                    "identity": [dataset, "resnet18", seed],
                    "error": "missing_expected_group",
                }
            )

    return {
        "root": str(root),
        "group_count": len(groups),
        "expected_recipes": sorted(EXPECTED),
        "status": "PASS" if groups and not invalid else "FAIL",
        "errors": invalid or ([{"error": "no_matched_kd_records"}] if not groups else []),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-datasets", nargs="+")
    parser.add_argument("--expected-seeds", nargs="+", type=int)
    args = parser.parse_args()
    if (args.expected_datasets is None) != (args.expected_seeds is None):
        parser.error("--expected-datasets and --expected-seeds must be supplied together")
    report = check(
        args.results_root.resolve(),
        expected_datasets=set(args.expected_datasets) if args.expected_datasets else None,
        expected_seeds=set(args.expected_seeds) if args.expected_seeds else None,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Matched-KD fairness: {report['status']} ({report['group_count']} groups)")
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
