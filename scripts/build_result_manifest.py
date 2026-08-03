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


def build_manifest(root: Path) -> dict:
    records = []
    failures = []
    for path in sorted(root.rglob("result_record.json")):
        try:
            record = validate_result_record(json.loads(path.read_text(encoding="utf-8")))
        except Exception as error:
            failures.append({"path": str(path), "error": str(error)})
            continue
        records.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "run_id": record["run_id"],
                "task_family": record["task_family"],
                "dataset": record["dataset"],
                "model": record["model"],
                "seed": record["seed"],
            }
        )
    return {
        "schema_version": "1.0",
        "result_root": str(root),
        "complete_records": records,
        "invalid_records": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_manifest(args.root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    if manifest["invalid_records"]:
        raise SystemExit(f"Found {len(manifest['invalid_records'])} invalid result records")
    print(f"Indexed {len(manifest['complete_records'])} validated result records")


if __name__ == "__main__":
    main()
