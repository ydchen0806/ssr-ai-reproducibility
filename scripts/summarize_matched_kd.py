#!/usr/bin/env python3
"""Summarize matched-KD regularizers with strict same-seed pairing."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ssr_utils.paired_stats import paired_summary
from ssr_utils.result_schema import validate_result_record


CONTROL = "kd"
TREATMENTS = (
    "kd_ewc",
    "kd_mas",
    "kd_si",
    "kd_center",
    "kd_protodecor",
    "kd_spectral",
    "kd_ssr",
)
REQUIRED_METRICS = (
    "avg_accuracy",
    "last_accuracy",
    "avg_forgetting",
    "backward_transfer",
    "cil_avg_accuracy",
    "cil_last_accuracy",
    "effective_rank",
    "prototype_overlap",
)
DEFAULT_DIRECTIONS = {
    "avg_accuracy": True,
    "average_accuracy": True,
    "last_accuracy": True,
    "avg_forgetting": False,
    "average_forgetting": False,
    "backward_transfer": True,
    "forward_transfer": True,
    "cil_avg_accuracy": True,
    "cil_last_accuracy": True,
    "effective_rank": True,
    "prototype_overlap": False,
    "mean_abs_offdiag_cosine": False,
}


def read_plan(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise ValueError(f"Empty matched-KD plan: {path}")
    return rows


def load_records(rows: list[dict[str, str]]) -> dict[tuple[str, str, int], dict]:
    records = {}
    for row in rows:
        path = Path(row["output"]) / "result_record.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        record = validate_result_record(json.loads(path.read_text(encoding="utf-8")))
        key = (row["dataset"], row["method"], int(row["seed"]))
        if key in records:
            raise ValueError(f"Duplicate matched-KD cell: {key}")
        expected = (record["dataset"], record["recipe"], int(record["seed"]))
        if expected != key:
            raise ValueError(f"Result identity mismatch for {path}: {expected} != {key}")
        records[key] = record
    return records


def summarize(records: dict[tuple[str, str, int], dict]) -> tuple[dict, list[dict]]:
    datasets = sorted({dataset for dataset, _, _ in records})
    output = {"control": CONTROL, "datasets": {}, "ssr_vs_controls": {}}
    csv_rows = []
    for dataset in datasets:
        by_method: dict[str, dict[int, dict]] = defaultdict(dict)
        for (row_dataset, method, seed), record in records.items():
            if row_dataset == dataset:
                by_method[method][seed] = record
        expected_methods = {CONTROL, *TREATMENTS}
        if set(by_method) != expected_methods:
            raise ValueError(
                f"{dataset}: expected methods {sorted(expected_methods)}, "
                f"found {sorted(by_method)}"
            )
        control_seeds = set(by_method[CONTROL])
        if len(control_seeds) < 2:
            raise ValueError(f"{dataset}: at least two paired seeds are required")
        for method in expected_methods:
            method_seeds = set(by_method[method])
            if method_seeds != control_seeds:
                raise ValueError(
                    f"{dataset}/{method}: non-identical paired seeds "
                    f"control={sorted(control_seeds)}, method={sorted(method_seeds)}"
                )
            for seed, record in by_method[method].items():
                missing = set(REQUIRED_METRICS) - set(record.get("metrics", {}))
                if missing:
                    raise ValueError(
                        f"{dataset}/{method}/seed_{seed}: missing metrics {sorted(missing)}"
                    )

        def contrast_payload(
            control: str,
            treatment: str,
            comparison: str,
        ) -> dict:
            seeds = sorted(control_seeds)
            payload = {
                "control": control,
                "treatment": treatment,
                "seeds": seeds,
                "metrics": {},
            }
            for metric in REQUIRED_METRICS:
                control_values = [
                    float(by_method[control][seed]["metrics"][metric])
                    for seed in seeds
                ]
                treatment_values = [
                    float(by_method[treatment][seed]["metrics"][metric])
                    for seed in seeds
                ]
                direction = DEFAULT_DIRECTIONS[metric]
                summary = paired_summary(
                    control_values,
                    treatment_values,
                    higher_is_better=direction,
                ).as_dict()
                payload["metrics"][metric] = {
                    "higher_is_better": direction,
                    **summary,
                }
                csv_rows.append(
                    {
                        "comparison": comparison,
                        "dataset": dataset,
                        "control": control,
                        "treatment": treatment,
                        "metric": metric,
                        "higher_is_better": direction,
                        **summary,
                    }
                )
            return payload

        dataset_payload = {}
        for treatment in TREATMENTS:
            dataset_payload[treatment] = contrast_payload(
                CONTROL,
                treatment,
                "regularizer_vs_kd",
            )
        output["datasets"][dataset] = dataset_payload
        output["ssr_vs_controls"][dataset] = {
            control: contrast_payload(control, "kd_ssr", "ssr_vs_regularizer")
            for control in TREATMENTS
            if control != "kd_ssr"
        }
    return output, csv_rows


def write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    payload, rows = summarize(load_records(read_plan(args.plan.resolve())))
    output_dir = args.output_dir.resolve()
    write_atomic(output_dir / "summary.json", json.dumps(payload, indent=2) + "\n")
    csv_path = output_dir / "summary.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = csv_path.with_name(f".{csv_path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, csv_path)
    print(f"Matched-KD paired summary: {len(rows)} metric contrasts")


if __name__ == "__main__":
    main()
