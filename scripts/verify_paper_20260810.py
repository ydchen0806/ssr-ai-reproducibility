#!/usr/bin/env python3
"""Verify manuscript-facing CounterFact values against the full factorial."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "artifacts" / "paper_20260810"


def _load(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _resolve(record: Any, path: list[str]) -> Any:
    for key in path:
        record = record[key]
    return record


def _assert_equal(actual: Any, expected: Any, label: str) -> None:
    if isinstance(expected, float):
        if not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-10):
            raise AssertionError(f"{label}: expected {expected}, found {actual}")
    elif actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, found {actual!r}")


def verify() -> list[str]:
    index = _load(BUNDLE / "reported_values.json")
    source = _load(BUNDLE / index["source"])
    verified: list[str] = []

    for claim in index["claims"]:
        record = _resolve(source, claim["path"])
        for field in (
            "n",
            "control_mean",
            "treatment_mean",
            "difference_mean",
            "ci95_low",
            "ci95_high",
            "favorable_pairs",
        ):
            _assert_equal(record[field], claim[field], f"{claim['id']}.{field}")
        if record["ci95_low"] <= 0:
            raise AssertionError(f"{claim['id']}: favorable interval is not above zero")
        verified.append(claim["id"])

    endpoint = index["matched_task_endpoint"]
    record = _resolve(source, endpoint["path"])
    for field in ("n", "control_mean", "treatment_mean", "difference_mean"):
        _assert_equal(record[field], endpoint[field], f"matched_task_endpoint.{field}")

    return verified


if __name__ == "__main__":
    claims = verify()
    print(f"Verified {len(claims)} manuscript-facing CounterFact contrasts.")
    for claim_id in claims:
        print(f"  - {claim_id}")
