#!/usr/bin/env python3
"""Index validated per-run JSON records without rewriting their contents."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ssr_utils.result_schema import sha256_file, validate_result_record


DEFAULT_PLAN = PROJECT_ROOT / "configs" / "meeting_20260803" / "manifest.yaml"
OPTIONAL_INDEX_FIELDS = (
    "data_offset",
    "n_edits",
    "status",
    "evaluator",
    "evaluator_version",
    "pairing_hash",
    "model_hash",
    "evaluation_protocol_hash",
    "pairing_protocol_hash",
    "teacher_protocol",
    "teacher_trajectory_hash",
    "selection_lock_hash",
)
CELL_FIELDS = (
    "task_family",
    "dataset",
    "model",
    "recipe",
    "distance_mapping",
    "seed",
)


def _index_record(path: Path, record: dict) -> dict:
    item = {
        "path": str(path),
        "sha256": sha256_file(path),
        "git_commit": record["git_commit"],
        "run_id": record["run_id"],
        "task_family": record["task_family"],
        "dataset": record["dataset"],
        "model": record["model"],
        "seed": record["seed"],
        "recipe": record["recipe"],
        "distance_mapping": record["distance_mapping"],
        "config_hash": record["config_hash"],
        "dataset_hash": record["dataset_hash"],
        "objective": record["objective"],
        "kernel": record["kernel"],
        "metric_names": sorted(record["metrics"]),
        "runtime_names": sorted(record["runtime"]),
    }
    for field in OPTIONAL_INDEX_FIELDS:
        if field in record:
            item[field] = record[field]
    return item


def _failure_markers(root: Path) -> list[dict]:
    markers = []
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        name = path.name
        has_result = (path.parent / "result_record.json").is_file()
        if not has_result and (name.startswith("FAILED") or name == "status.tsv"):
            has_result = next(path.parent.rglob("result_record.json"), None) is not None
        if name == "FAILED" or name == "FAILED.json" or name.startswith("FAILED."):
            archived = name.startswith("FAILED.previous.")
            markers.append(
                {
                    "path": str(path),
                    "kind": "archived_failure" if archived else "failure_file",
                    "active": not archived and not has_result,
                    "sha256": sha256_file(path),
                }
            )
            continue
        if name != "status.tsv":
            continue
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(),
            start=1,
        ):
            columns = line.split("\t")
            if not any(column.strip().upper() == "FAILED" for column in columns):
                continue
            markers.append(
                {
                    "path": str(path),
                    "kind": "failed_status_entry",
                    "active": not has_result,
                    "line_number": line_number,
                    "columns": columns,
                }
            )
    return sorted(
        markers,
        key=lambda item: (
            item["path"],
            int(item.get("line_number", 0)),
            item["kind"],
        ),
    )


def _cell(record: dict) -> dict:
    return {field: record[field] for field in CELL_FIELDS}


def _cell_key(cell: dict) -> tuple:
    return tuple(cell[field] for field in CELL_FIELDS)


def _expected_cells(cohort: dict, seeds: list[int]) -> list[dict]:
    mappings = sorted(set(cohort.get("mappings", [])))
    mapping_dependent = set(cohort.get("mapping_dependent_recipes", []))
    cells = []
    for dataset in sorted(set(cohort["datasets"])):
        for recipe in sorted(set(cohort["recipes"])):
            recipe_mappings = mappings if recipe in mapping_dependent else ["none"]
            if not recipe_mappings:
                raise ValueError(
                    f"mapping-dependent recipe {recipe!r} has no mappings in cohort"
                )
            for mapping in recipe_mappings:
                for seed in sorted(set(seeds)):
                    cells.append(
                        {
                            "task_family": cohort["task_family"],
                            "dataset": dataset,
                            "model": cohort["model"],
                            "recipe": recipe,
                            "distance_mapping": mapping,
                            "seed": seed,
                        }
                    )
    return sorted(cells, key=_cell_key)


def _observed_cells(items: list[dict]) -> list[dict]:
    paths_by_cell: dict[tuple, list[str]] = {}
    cells_by_key: dict[tuple, dict] = {}
    for item in items:
        cell = _cell(item)
        key = _cell_key(cell)
        cells_by_key[key] = cell
        paths_by_cell.setdefault(key, []).append(item["path"])
    return [
        {
            **cells_by_key[key],
            "record_count": len(paths_by_cell[key]),
            "paths": sorted(paths_by_cell[key]),
        }
        for key in sorted(cells_by_key)
    ]


def _cohort_contract_errors(item: dict, cohort: dict) -> list[str]:
    errors = []
    for field in ("n_edits", "data_offset"):
        if field in cohort and item.get(field) != cohort[field]:
            errors.append(
                f"{field}: expected {cohort[field]!r}, observed {item.get(field)!r}"
            )
    missing_metrics = sorted(
        set(cohort.get("required_metrics", [])) - set(item.get("metric_names", []))
    )
    missing_runtime = sorted(
        set(cohort.get("required_runtime", [])) - set(item.get("runtime_names", []))
    )
    if missing_metrics:
        errors.append(f"missing metrics: {missing_metrics}")
    if missing_runtime:
        errors.append(f"missing runtime: {missing_runtime}")
    missing_identity = sorted(
        field for field in cohort.get("paired_identity", []) if field not in item
    )
    if missing_identity:
        errors.append(f"missing paired identity: {missing_identity}")
    return errors


def _cohort_matrices(plan: dict, items: list[dict]) -> dict:
    matrices = {}
    for name in sorted(plan.get("cohorts", {})):
        cohort = plan["cohorts"][name]
        confirmation_seeds = list(cohort.get("confirmation_seeds", []))
        seed_field = "confirmation_seeds"
        if not confirmation_seeds:
            confirmation_seeds = list(cohort.get("development_seeds", []))
            seed_field = "development_seeds"
        expected = _expected_cells(cohort, confirmation_seeds)
        expected_keys = {_cell_key(cell) for cell in expected}
        development_seeds = set(cohort.get("development_seeds", []))
        cohort_items = [
            item
            for item in items
            if item["task_family"] == cohort["task_family"]
            and item["dataset"] in cohort["datasets"]
            and item["model"] == cohort["model"]
            and item["recipe"] in cohort["recipes"]
        ]
        expected_items = [item for item in cohort_items if _cell_key(_cell(item)) in expected_keys]
        protocol_mismatches = []
        eligible_items = []
        for item in expected_items:
            errors = _cohort_contract_errors(item, cohort)
            if errors:
                protocol_mismatches.append(
                    {"cell": _cell(item), "path": item["path"], "errors": errors}
                )
            else:
                eligible_items.append(item)

        items_by_cell: dict[tuple, list[dict]] = {}
        for item in eligible_items:
            items_by_cell.setdefault(_cell_key(_cell(item)), []).append(item)
        nonconflicting_items = []
        conflicting = []
        for key in sorted(items_by_cell):
            cell_items = items_by_cell[key]
            unique_hashes = {item["sha256"] for item in cell_items}
            if len(unique_hashes) > 1:
                conflicting.append(
                    {
                        "cell": _cell(cell_items[0]),
                        "records": [
                            {"path": item["path"], "sha256": item["sha256"]}
                            for item in sorted(cell_items, key=lambda value: value["path"])
                        ],
                    }
                )
            else:
                nonconflicting_items.extend(cell_items)
        observed = _observed_cells(nonconflicting_items)
        observed_keys = {_cell_key(cell) for cell in observed}
        missing = [cell for cell in expected if _cell_key(cell) not in observed_keys]
        development = _observed_cells(
            [
                item
                for item in cohort_items
                if item["seed"] in development_seeds
                and _cell_key(_cell(item)) not in expected_keys
            ]
        )
        unexpected = _observed_cells(
            [
                item
                for item in cohort_items
                if item["seed"] not in development_seeds
                and _cell_key(_cell(item)) not in expected_keys
            ]
        )
        matrices[name] = {
            "seed_source": seed_field,
            "expected_count": len(expected),
            "observed_count": len(observed),
            "missing_count": len(missing),
            "development_observed_count": len(development),
            "unexpected_count": len(unexpected),
            "protocol_mismatch_count": len(protocol_mismatches),
            "conflicting_cell_count": len(conflicting),
            "expected_matrix": expected,
            "observed_matrix": observed,
            "missing_cells": missing,
            "development_observed_cells": development,
            "unexpected_cells": unexpected,
            "protocol_mismatches": sorted(
                protocol_mismatches,
                key=lambda item: (_cell_key(item["cell"]), item["path"]),
            ),
            "conflicting_cells": conflicting,
        }
    return matrices


def build_manifest(root: Path, plan: dict | None = None) -> dict:
    if plan is None:
        plan = yaml.safe_load(DEFAULT_PLAN.read_text(encoding="utf-8"))
    records = []
    failures = []
    for path in sorted(root.rglob("result_record.json")):
        try:
            record = validate_result_record(json.loads(path.read_text(encoding="utf-8")))
        except Exception as error:
            failures.append({"path": str(path), "error": str(error)})
            continue
        records.append(_index_record(path, record))
    return {
        "schema_version": "1.0",
        "result_root": str(root),
        "complete_records": records,
        "invalid_records": failures,
        "failure_markers": _failure_markers(root),
        "cohort_matrices": _cohort_matrices(plan, records),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    args = parser.parse_args()
    plan_path = args.plan.resolve()
    plan = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    manifest = build_manifest(args.root.resolve(), plan)
    manifest["plan"] = {
        "path": str(plan_path),
        "sha256": sha256_file(plan_path),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    if manifest["invalid_records"]:
        raise SystemExit(f"Found {len(manifest['invalid_records'])} invalid result records")
    print(f"Indexed {len(manifest['complete_records'])} validated result records")


if __name__ == "__main__":
    main()
