#!/usr/bin/env python3
"""Aggregate meeting-revision records with strict paired joins."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ssr_utils.paired_stats import complete_pairs, paired_summary
from ssr_utils.result_schema import canonical_json, validate_result_record


DEFAULT_DIRECTIONS = {
    "aa": True,
    "average_accuracy": True,
    "accuracy": True,
    "effective_rank": True,
    "efficacy": True,
    "locality": True,
    "mean_iou": True,
    "miou": True,
    "mean_dice": True,
    "dice": True,
    "af": False,
    "average_forgetting": False,
    "iou_forgetting": False,
    "avg_forgetting_iou": False,
    "prototype_overlap": False,
    "mean_abs_offdiag_cosine": False,
}


def _record_identity(plan: dict, record: dict[str, Any]) -> tuple[Any, ...]:
    """Return the experiment cell identity used by the planned aggregation."""
    matching_cohorts = [
        (name, cohort)
        for name, cohort in plan["cohorts"].items()
        if record["task_family"] == cohort["task_family"]
        and record["dataset"] in cohort["datasets"]
        and record["model"] == cohort["model"]
    ]
    if len(matching_cohorts) > 1:
        names = [name for name, _ in matching_cohorts]
        raise ValueError(
            f"result matches multiple aggregation cohorts {names}: "
            f"{record.get('_path', '<unknown>')}"
        )
    if matching_cohorts:
        cohort_name, cohort = matching_cohorts[0]
        identity_fields = tuple(
            cohort.get("paired_identity", ("dataset", "model", "seed"))
        )
        missing = [field for field in identity_fields if field not in record]
        if missing:
            raise ValueError(
                f"result is missing aggregation identity fields {missing}: "
                f"{record.get('_path', '<unknown>')}"
            )
        return (
            "cohort",
            cohort_name,
            record["recipe"],
            record["distance_mapping"],
            *(record[field] for field in identity_fields),
        )

    # Unplanned records still enter the long-form audit. Include execution
    # identifiers so unrelated exploratory runs are not collapsed together.
    return (
        "unplanned",
        record["task_family"],
        record["dataset"],
        record["model"],
        record["seed"],
        record["recipe"],
        record["distance_mapping"],
        record.get("data_offset"),
        record.get("n_edits"),
        record["dataset_hash"],
        record["config_hash"],
        record["run_id"],
    )


def read_records(root: Path, plan: dict) -> list[dict[str, Any]]:
    """Read records once, tolerating only exact copies of one experiment cell."""
    records_by_identity: dict[tuple[Any, ...], dict[str, Any]] = {}
    payloads_by_identity: dict[tuple[Any, ...], str] = {}
    for path in sorted(root.rglob("result_record.json")):
        record = validate_result_record(json.loads(path.read_text(encoding="utf-8")))
        record["_path"] = str(path)
        identity = _record_identity(plan, record)
        payload = {key: value for key, value in record.items() if key != "_path"}
        canonical_payload = canonical_json(payload)
        if identity not in records_by_identity:
            records_by_identity[identity] = record
            payloads_by_identity[identity] = canonical_payload
            continue
        if payloads_by_identity[identity] == canonical_payload:
            continue

        original = records_by_identity[identity]
        original_payload = {
            key: value for key, value in original.items() if key != "_path"
        }
        differing_fields = sorted(
            key
            for key in set(original_payload) | set(payload)
            if original_payload.get(key) != payload.get(key)
        )
        raise ValueError(
            "conflicting duplicate aggregation identity "
            f"{identity!r}: first={original['_path']!r}, duplicate={str(path)!r}, "
            f"differing_fields={differing_fields}"
        )
    return list(records_by_identity.values())


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def long_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        for metric, value in sorted(record["metrics"].items()):
            if not isinstance(value, (int, float)):
                continue
            rows.append(
                {
                    "task_family": record["task_family"],
                    "dataset": record["dataset"],
                    "model": record["model"],
                    "run_id": record["run_id"],
                    "recipe": record.get("recipe", ""),
                    "distance_mapping": record["distance_mapping"],
                    "seed": record["seed"],
                    "data_offset": record.get("data_offset", ""),
                    "n_edits": record.get("n_edits", ""),
                    "metric": metric,
                    "value": value,
                    "result_path": record["_path"],
                }
            )
    return rows


def paired_contrasts(plan: dict, records: list[dict[str, Any]]) -> tuple[list[dict], list[dict]]:
    summaries = []
    exclusions = []
    for cohort_name, cohort in plan["cohorts"].items():
        confirmation_seeds = set(cohort.get("confirmation_seeds", []))
        cohort_records = [
            row
            for row in records
            if row["task_family"] == cohort["task_family"]
            and row["dataset"] in cohort["datasets"]
            and row["model"] == cohort["model"]
            and (not confirmation_seeds or row["seed"] in confirmation_seeds)
        ]
        identity_fields = tuple(
            cohort.get("paired_identity", ("dataset", "model", "seed"))
        )
        mapping_dependent = set(cohort.get("mapping_dependent_recipes", []))
        contrasts = list(cohort.get("primary_contrasts", []))
        if cohort.get("transfer_contrast"):
            contrasts.append(cohort["transfer_contrast"])
        for treatment_name, control_name in contrasts:
            for dataset in cohort["datasets"]:
                dataset_rows = [row for row in cohort_records if row["dataset"] == dataset]
                treatments = [row for row in dataset_rows if row.get("recipe") == treatment_name]
                controls = [row for row in dataset_rows if row.get("recipe") == control_name]
                if treatment_name in mapping_dependent:
                    mappings = list(cohort.get("mappings", []))
                    if not mappings:
                        mappings = sorted(
                            {row["distance_mapping"] for row in treatments}
                        ) or ["none"]
                else:
                    mappings = ["none"]
                for mapping in mappings:
                    mapped_treatments = [
                        row for row in treatments if row["distance_mapping"] == mapping
                    ]
                    mapped_controls = [
                        row
                        for row in controls
                        if row["distance_mapping"] in {"none", mapping}
                    ]
                    pairs, missing = complete_pairs(
                        mapped_controls,
                        mapped_treatments,
                        identity_fields=identity_fields,
                    )
                    paired_seeds = {control["seed"] for control, _ in pairs}
                    expected_seeds = confirmation_seeds or {
                        row["seed"] for row in mapped_controls + mapped_treatments
                    }
                    missing_expected = sorted(expected_seeds - paired_seeds)
                    if missing or missing_expected:
                        exclusions.append(
                            {
                                "cohort": cohort_name,
                                "dataset": dataset,
                                "contrast": f"{treatment_name}_minus_{control_name}",
                                "mapping": mapping,
                                "missing_identity": json.dumps(
                                    {
                                        "unmatched_observed_identities": missing,
                                        "missing_expected_seeds": missing_expected,
                                        "identity_fields": identity_fields,
                                    }
                                ),
                            }
                        )
                    if len(pairs) < 2:
                        continue
                    common_metrics = set.intersection(
                        *(set(control["metrics"]) & set(treatment["metrics"]) for control, treatment in pairs)
                    )
                    for metric in sorted(common_metrics):
                        try:
                            control_values = [float(control["metrics"][metric]) for control, _ in pairs]
                            treatment_values = [float(treatment["metrics"][metric]) for _, treatment in pairs]
                        except (TypeError, ValueError):
                            continue
                        direction = DEFAULT_DIRECTIONS.get(metric)
                        if direction is None:
                            direction = bool(
                                mapped_treatments[0].get("metric_directions", {}).get(metric, True)
                            )
                        summary = paired_summary(
                            control_values,
                            treatment_values,
                            higher_is_better=direction,
                        ).as_dict()
                        summaries.append(
                            {
                                "cohort": cohort_name,
                                "task_family": cohort["task_family"],
                                "dataset": dataset,
                                "model": cohort["model"],
                                "contrast": f"{treatment_name}_minus_{control_name}",
                                "mapping": mapping,
                                "metric": metric,
                                "higher_is_better": direction,
                                **summary,
                            }
                        )
    return summaries, exclusions


def editing_mapping_contrasts(
    plan: dict,
    records: list[dict[str, Any]],
) -> tuple[list[dict], list[dict]]:
    """Compare projective against cosine for the same SSR recipe and seed."""
    summaries = []
    exclusions = []
    for cohort_name, cohort in plan["cohorts"].items():
        if cohort["task_family"] != "editing":
            continue
        if not {"cosine", "projective"}.issubset(set(cohort.get("mappings", []))):
            continue
        confirmation_seeds = set(cohort.get("confirmation_seeds", []))
        identity_fields = tuple(
            cohort.get("paired_identity", ("dataset", "model", "seed"))
        )
        mapping_recipes = set(cohort.get("mapping_dependent_recipes", []))
        cohort_records = [
            row
            for row in records
            if row["task_family"] == "editing"
            and row["dataset"] in cohort["datasets"]
            and row["model"] == cohort["model"]
            and (not confirmation_seeds or row["seed"] in confirmation_seeds)
        ]
        for dataset in cohort["datasets"]:
            for recipe in sorted(mapping_recipes):
                selected = [
                    row
                    for row in cohort_records
                    if row["dataset"] == dataset and row.get("recipe") == recipe
                ]
                cosine = [row for row in selected if row["distance_mapping"] == "cosine"]
                projective = [
                    row for row in selected if row["distance_mapping"] == "projective"
                ]
                pairs, missing = complete_pairs(
                    cosine,
                    projective,
                    identity_fields=identity_fields,
                )
                paired_seeds = {control["seed"] for control, _ in pairs}
                expected_seeds = confirmation_seeds or {
                    row["seed"] for row in cosine + projective
                }
                missing_expected = sorted(expected_seeds - paired_seeds)
                if missing or missing_expected:
                    exclusions.append(
                        {
                            "cohort": cohort_name,
                            "dataset": dataset,
                            "contrast": f"{recipe}:projective_minus_cosine",
                            "mapping": "projective_vs_cosine",
                            "missing_identity": json.dumps(
                                {
                                    "unmatched_observed_identities": missing,
                                    "missing_expected_seeds": missing_expected,
                                    "identity_fields": identity_fields,
                                }
                            ),
                        }
                    )
                if len(pairs) < 2:
                    continue
                common_metrics = set.intersection(
                    *(
                        set(control["metrics"]) & set(treatment["metrics"])
                        for control, treatment in pairs
                    )
                )
                for metric in sorted(common_metrics):
                    try:
                        control_values = [
                            float(control["metrics"][metric]) for control, _ in pairs
                        ]
                        treatment_values = [
                            float(treatment["metrics"][metric]) for _, treatment in pairs
                        ]
                    except (TypeError, ValueError):
                        continue
                    direction = DEFAULT_DIRECTIONS.get(metric)
                    if direction is None:
                        direction = bool(
                            projective[0].get("metric_directions", {}).get(metric, True)
                        )
                    summaries.append(
                        {
                            "cohort": cohort_name,
                            "task_family": "editing",
                            "dataset": dataset,
                            "model": cohort["model"],
                            "recipe": recipe,
                            "contrast": "projective_minus_cosine",
                            "mapping": "projective_vs_cosine",
                            "metric": metric,
                            "higher_is_better": direction,
                            **paired_summary(
                                control_values,
                                treatment_values,
                                higher_is_better=direction,
                            ).as_dict(),
                        }
                    )
    return summaries, exclusions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Write a partial audit without failing on absent confirmatory pairs.",
    )
    args = parser.parse_args()
    plan = yaml.safe_load(args.manifest.read_text(encoding="utf-8"))
    records = read_records(args.results_root, plan)
    output = args.output
    table_root = output / "tables"
    long = long_rows(records)
    summaries, exclusions = paired_contrasts(plan, records)
    mapping_summaries, mapping_exclusions = editing_mapping_contrasts(plan, records)
    exclusions.extend(mapping_exclusions)
    long_fields = [
        "task_family", "dataset", "model", "run_id", "recipe", "distance_mapping",
        "seed", "data_offset", "n_edits", "metric", "value", "result_path",
    ]
    summary_fields = [
        "cohort", "task_family", "dataset", "model", "contrast", "mapping", "metric",
        "recipe", "higher_is_better", "n", "control_mean", "control_sd",
        "treatment_mean", "treatment_sd", "raw_difference_mean", "raw_difference_sd",
        "difference_mean", "difference_sd", "ci95_low", "ci95_high", "favorable_pairs",
    ]
    write_csv(table_root / "all_results_long.csv", long, long_fields)
    write_csv(table_root / "paired_contrasts.csv", summaries, summary_fields)
    write_csv(table_root / "editing_mapping.csv", mapping_summaries, summary_fields)
    write_csv(
        output / "missing_pairs.csv",
        exclusions,
        ["cohort", "dataset", "contrast", "mapping", "missing_identity"],
    )
    for task_family, filename in {
        "editing": "editing_ablation.csv",
        "classification": "kd_matched_cl.csv",
        "segmentation": "cub_segmentation.csv",
        "adapter": "adapter_ablation.csv",
    }.items():
        write_csv(
            table_root / filename,
            [row for row in summaries if row["task_family"] == task_family],
            summary_fields,
        )
    print(
        f"Aggregated {len(records)} records, {len(summaries)} paired summaries, "
        f"{len(exclusions)} incomplete pair sets"
    )
    if not args.allow_incomplete and (not records or exclusions):
        raise SystemExit(
            "Aggregation is incomplete: use --allow-incomplete only for a staged audit"
        )


if __name__ == "__main__":
    main()
