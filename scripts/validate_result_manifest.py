#!/usr/bin/env python3
"""Validate a planned experiment manifest or an indexed result manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ssr_utils.result_schema import SCHEMA_VERSION, sha256_file, validate_result_record


RECIPE_KEYS = {
    "plain",
    "anchor",
    "spectral",
    "stabilized",
    "ssr_only",
    "full",
    "kd",
    "kd_ewc",
    "kd_mas",
    "kd_si",
    "kd_ssr",
    "ssr_kd",
}


def _ensure_unique(values: list, label: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{label} contains duplicate entries")


def validate_plan(payload: dict) -> None:
    cohorts = payload.get("cohorts")
    if not isinstance(cohorts, dict) or not cohorts:
        raise ValueError("planned manifest requires a non-empty cohorts object")
    for name, cohort in cohorts.items():
        for field in ("task_family", "model", "datasets", "recipes"):
            if field not in cohort:
                raise ValueError(f"cohort {name!r} is missing {field!r}")
        _ensure_unique(cohort["datasets"], f"{name}.datasets")
        _ensure_unique(cohort["recipes"], f"{name}.recipes")
        unknown_recipes = set(cohort["recipes"]) - RECIPE_KEYS
        if unknown_recipes:
            raise ValueError(f"cohort {name!r} has unknown recipes {sorted(unknown_recipes)}")
        development = cohort.get("development_seeds", [])
        confirmation = cohort.get("confirmation_seeds", [])
        _ensure_unique(development, f"{name}.development_seeds")
        _ensure_unique(confirmation, f"{name}.confirmation_seeds")
        overlap = set(development) & set(confirmation)
        if overlap:
            raise ValueError(f"cohort {name!r} reuses development seeds in confirmation: {sorted(overlap)}")
        recipes = set(cohort["recipes"])
        for treatment, control in cohort.get("primary_contrasts", []):
            if treatment not in recipes or control not in recipes:
                raise ValueError(
                    f"cohort {name!r} contrast {(treatment, control)} is outside its recipe matrix"
                )


def validate_index(payload: dict, manifest_path: Path) -> None:
    invalid = payload.get("invalid_records", [])
    if invalid:
        raise ValueError(f"indexed manifest contains {len(invalid)} invalid records")
    for item in payload.get("complete_records", []):
        path = Path(item["path"])
        if not path.is_absolute():
            path = manifest_path.parent / path
        if not path.is_file():
            raise ValueError(f"indexed result is missing: {path}")
        if sha256_file(path) != item["sha256"]:
            raise ValueError(f"checksum mismatch: {path}")
        validate_result_record(json.loads(path.read_text(encoding="utf-8")))


def validate_manifest(path: Path) -> None:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("manifest root must be an object")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"manifest schema_version must be {SCHEMA_VERSION!r}")
    if "cohorts" in payload:
        validate_plan(payload)
    elif "complete_records" in payload:
        validate_index(payload, path)
    else:
        raise ValueError("manifest is neither a plan nor a result index")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    validate_manifest(args.manifest.resolve())
    print(f"Manifest validated: {args.manifest}")


if __name__ == "__main__":
    main()
