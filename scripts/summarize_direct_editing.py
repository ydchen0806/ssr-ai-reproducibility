#!/usr/bin/env python3
"""Create strict paired summaries for plain versus SSR-only editing runs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ssr_utils.paired_stats import paired_summary
from ssr_utils.result_schema import validate_result_record


CONTROL = "plain"
TREATMENT = "ssr_only"
PAIR_FIELDS = (
    "dataset",
    "model",
    "seed",
    "data_offset",
    "n_edits",
    "dataset_hash",
    "evaluator",
    "evaluator_version",
    "evaluation_protocol_hash",
    "model_hash",
    "pairing_protocol_hash",
)


def read_plan(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise ValueError(f"Empty editing plan: {path}")
    return rows


def load_pairs(rows: list[dict[str, str]]) -> dict[str, list[tuple[dict, dict]]]:
    records = {}
    for row in rows:
        if row["recipe"] not in {CONTROL, TREATMENT}:
            raise ValueError(f"Unexpected direct-editing recipe: {row['recipe']}")
        path = Path(row["output"]) / "result_record.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        record = validate_result_record(json.loads(path.read_text(encoding="utf-8")))
        key = (row["dataset"], int(row["seed"]), row["recipe"])
        if key in records:
            raise ValueError(f"Duplicate direct-editing cell: {key}")
        if (
            record["dataset"] != row["dataset"]
            or int(record["seed"]) != int(row["seed"])
            or record["recipe"] != row["recipe"]
        ):
            raise ValueError(f"Result identity mismatch: {path}")
        records[key] = record

    pairs: dict[str, list[tuple[dict, dict]]] = {}
    for dataset in sorted({dataset for dataset, _, _ in records}):
        seeds = sorted({seed for row_dataset, seed, _ in records if row_dataset == dataset})
        dataset_pairs = []
        for seed in seeds:
            control = records.get((dataset, seed, CONTROL))
            treatment = records.get((dataset, seed, TREATMENT))
            if control is None or treatment is None:
                raise ValueError(f"Missing direct-editing pair: {dataset}/seed_{seed}")
            mismatches = {
                field: {"control": control.get(field), "treatment": treatment.get(field)}
                for field in PAIR_FIELDS
                if control.get(field) != treatment.get(field)
            }
            if mismatches:
                raise ValueError(
                    f"Unmatched direct-editing pair {dataset}/seed_{seed}: {mismatches}"
                )
            if control["distance_mapping"] != "none":
                raise ValueError("Plain editing control must use distance_mapping=none")
            if treatment["distance_mapping"] not in {"cosine", "projective"}:
                raise ValueError("SSR editing treatment must use a geometric mapping")
            dataset_pairs.append((control, treatment))
        pairs[dataset] = dataset_pairs
    return pairs


def summarize(pairs: dict[str, list[tuple[dict, dict]]]) -> tuple[dict, list[dict]]:
    payload = {"control": CONTROL, "treatment": TREATMENT, "datasets": {}}
    rows = []
    for dataset, dataset_pairs in pairs.items():
        if len(dataset_pairs) < 2:
            raise ValueError(f"{dataset}: at least two paired seeds are required")
        common_metrics = set.intersection(
            *(
                set(control["metrics"]) & set(treatment["metrics"])
                for control, treatment in dataset_pairs
            )
        )
        dataset_payload = {
            "seeds": [int(control["seed"]) for control, _ in dataset_pairs],
            "metrics": {},
        }
        for metric in sorted(common_metrics):
            try:
                control_values = [float(control["metrics"][metric]) for control, _ in dataset_pairs]
                treatment_values = [float(treatment["metrics"][metric]) for _, treatment in dataset_pairs]
            except (TypeError, ValueError):
                continue
            directions = {
                bool(record.get("metric_directions", {}).get(metric, True))
                for pair in dataset_pairs
                for record in pair
            }
            if len(directions) != 1:
                raise ValueError(f"{dataset}/{metric}: inconsistent metric direction")
            higher_is_better = directions.pop()
            summary = paired_summary(
                control_values,
                treatment_values,
                higher_is_better=higher_is_better,
            ).as_dict()
            dataset_payload["metrics"][metric] = {
                "higher_is_better": higher_is_better,
                **summary,
            }
            rows.append(
                {
                    "dataset": dataset,
                    "control": CONTROL,
                    "treatment": TREATMENT,
                    "metric": metric,
                    "higher_is_better": higher_is_better,
                    **summary,
                }
            )
        payload["datasets"][dataset] = dataset_payload
    return payload, rows


def write_outputs(output_dir: Path, payload: dict, rows: list[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "summary.json"
    json_tmp = json_path.with_name(f".{json_path.name}.tmp.{os.getpid()}")
    json_tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(json_tmp, json_path)
    csv_path = output_dir / "summary.csv"
    csv_tmp = csv_path.with_name(f".{csv_path.name}.tmp.{os.getpid()}")
    with csv_tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(csv_tmp, csv_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    payload, rows = summarize(load_pairs(read_plan(args.plan.resolve())))
    write_outputs(args.output_dir.resolve(), payload, rows)
    print(f"Direct-editing paired summary: {len(rows)} metric contrasts")


if __name__ == "__main__":
    main()
