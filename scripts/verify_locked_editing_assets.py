#!/usr/bin/env python3
"""Verify the locked GPT-2 XL and KnowEdit assets before formal runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ssr_utils.result_schema import sha256_file, sha256_value


def verify_assets(config_path: Path, model_path: Path) -> dict:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    errors = []
    observed_model_files = {}
    for relative, expected in config["model"]["fingerprint_files"].items():
        path = model_path / relative
        if not path.is_file():
            errors.append(f"missing model file: {path}")
            continue
        observed = sha256_file(path)
        observed_model_files[relative] = observed
        if observed != expected:
            errors.append(
                f"model hash mismatch for {path}: expected {expected}, observed {observed}"
            )
    observed_model_hash = sha256_value(observed_model_files)
    expected_model_hash = config["model"]["fingerprint_sha256"]
    if observed_model_hash != expected_model_hash:
        errors.append(
            "combined model fingerprint mismatch: "
            f"expected {expected_model_hash}, observed {observed_model_hash}"
        )

    datasets = {}
    for name, spec in config["datasets"].items():
        path = PROJECT_ROOT / spec["path"]
        if not path.is_file():
            errors.append(f"missing dataset file: {path}")
            continue
        observed = sha256_file(path)
        datasets[name] = {
            "path": str(path.resolve()),
            "expected_sha256": spec["sha256"],
            "observed_sha256": observed,
        }
        if observed != spec["sha256"]:
            errors.append(
                f"dataset hash mismatch for {path}: expected {spec['sha256']}, "
                f"observed {observed}"
            )

    return {
        "status": "PASS" if not errors else "FAIL",
        "config": str(config_path.resolve()),
        "model_path": str(model_path.resolve()),
        "model_fingerprint_sha256": observed_model_hash,
        "model_files": observed_model_files,
        "datasets": datasets,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = verify_assets(args.config.resolve(), args.model_path.resolve())
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        temporary.replace(args.output)
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
