#!/usr/bin/env python3
"""Check registered manuscript claims against generated CSV rows."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_registered_path(path_text: str, registry_path: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    local_candidate = registry_path.parent / path
    if local_candidate.exists():
        return local_candidate
    return PROJECT_ROOT / path


def select_row(path: Path, filters: dict[str, str]) -> dict[str, str]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if all(row.get(key) == str(value) for key, value in filters.items())
        ]
    if len(rows) != 1:
        raise ValueError(f"Expected one row in {path} for {filters}, found {len(rows)}")
    return rows[0]


def check_claim(claim: dict, registry_path: Path) -> list[str]:
    errors = []
    source = resolve_registered_path(claim["source_csv"], registry_path)
    row = select_row(source.resolve(), claim.get("filters", {}))
    value = float(row[claim["value_column"]])
    tolerance = float(claim.get("tolerance", 1e-9))
    if "expected" in claim and abs(value - float(claim["expected"])) > tolerance:
        errors.append(
            f"{claim['id']}: source value {value} differs from expected {claim['expected']}"
        )
    for target in claim.get("targets", []):
        path = resolve_registered_path(target["path"], registry_path)
        text = path.read_text(encoding="utf-8")
        match = re.search(target["pattern"], text)
        if not match:
            errors.append(f"{claim['id']}: pattern not found in {path}")
            continue
        target_value = float(match.group(target.get("group", 1)))
        if abs(target_value - value) > float(target.get("tolerance", tolerance)):
            errors.append(
                f"{claim['id']}: {path} reports {target_value}, source row reports {value}"
            )
    if "relative_from" in claim:
        numerator = float(row[claim["relative_from"]["numerator"]])
        denominator = float(row[claim["relative_from"]["denominator"]])
        calculated = 100.0 * numerator / denominator
        if abs(value - calculated) > tolerance:
            errors.append(
                f"{claim['id']}: relative percentage {value} != {calculated} from source columns"
            )
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Mark an experiment-stage empty registry as STAGED instead of failing.",
    )
    args = parser.parse_args()
    payload = yaml.safe_load(args.registry.read_text(encoding="utf-8"))
    claims = payload.get("claims", [])
    errors = []
    if not claims and not args.allow_empty:
        errors.append("claim registry is empty in submission-validation mode")
    for claim in claims:
        if claim.get("enabled", True):
            errors.extend(check_claim(claim, args.registry.resolve()))
    report = args.registry.parent / "consistency_report.md"
    lines = ["# Result consistency report", ""]
    if not claims and args.allow_empty:
        lines.extend(
            [
                "Status: **STAGED**",
                "",
                "No numerical claims are enabled while confirmatory experiments are incomplete.",
            ]
        )
    elif errors:
        lines.extend(["Status: **FAIL**", "", *[f"- {error}" for error in errors]])
    else:
        lines.extend(["Status: **PASS**", "", "All enabled claims match their source rows."])
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Consistency check complete: {report}")


if __name__ == "__main__":
    main()
