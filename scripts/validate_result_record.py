#!/usr/bin/env python3
"""Validate one result sidecar and its expected experiment identity."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ssr_utils.result_schema import ResultSchemaError, validate_result_record


def validate_identity(
    path: Path,
    expected: dict[str, Any],
    *,
    expected_git_commit: str | None = None,
    allow_legacy_git: bool = False,
) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ResultSchemaError(f"Cannot read result record {path}: {error}") from error
    record = validate_result_record(raw)
    mismatches = {
        key: {"expected": value, "observed": record.get(key)}
        for key, value in expected.items()
        if value is not None and record.get(key) != value
    }
    if mismatches:
        raise ResultSchemaError(
            f"Result identity mismatch for {path}: {json.dumps(mismatches, sort_keys=True)}"
        )
    if expected_git_commit is not None and record["git_commit"] != expected_git_commit:
        is_legacy = str(record["git_commit"]).startswith("legacy-source-fingerprint:")
        if not (allow_legacy_git and is_legacy):
            raise ResultSchemaError(
                f"Result git_commit mismatch for {path}: expected {expected_git_commit}, "
                f"observed {record['git_commit']}"
            )
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--task-family", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--mapping", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data-offset", type=int)
    parser.add_argument("--n-edits", type=int)
    parser.add_argument("--expected-git-commit")
    parser.add_argument("--allow-legacy-git", action="store_true")
    args = parser.parse_args()
    expected = {
        "task_family": args.task_family,
        "dataset": args.dataset,
        "model": args.model,
        "recipe": args.recipe,
        "distance_mapping": args.mapping,
        "seed": args.seed,
        "data_offset": args.data_offset,
        "n_edits": args.n_edits,
    }
    try:
        validate_identity(
            args.record.resolve(),
            expected,
            expected_git_commit=args.expected_git_commit,
            allow_legacy_git=args.allow_legacy_git,
        )
    except ResultSchemaError as error:
        parser.exit(1, f"invalid result record: {error}\n")
    print(f"Validated result record: {args.record}")


if __name__ == "__main__":
    main()
