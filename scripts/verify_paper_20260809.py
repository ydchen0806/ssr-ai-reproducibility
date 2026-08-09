#!/usr/bin/env python3
"""Verify the locked 2026-08-09 manuscript evidence from paired records."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ssr_utils.paired_stats import paired_summary  # noqa: E402


DEFAULT_ARTIFACT_DIR = ROOT / "artifacts" / "paper_20260809"
CHECKSUM_FILE = "checksums.sha256"

PET_CONFIG = {
    "candidate_id": "r16_joint_c0.006_a0.0012",
    "rank": 16,
    "lambda_classifier": 0.006,
    "lambda_adapter": 0.0012,
    "sigma_exc": 0.2,
    "sigma_inh": 0.5,
    "start_task": 0,
    "ramp_tasks": 0,
}

# These values lock the manuscript-facing analysis to the completed 30-seed run.
PET_EXPECTED = {
    "avg_accuracy": {
        "difference_mean": 0.07228490317506366,
        "ci95_low": 0.018341727797605574,
        "ci95_high": 0.12622807855252174,
        "favorable_pairs": 22,
    },
    "avg_forgetting": {
        "difference_mean": 0.25331350989668344,
        "ci95_low": 0.007919958129636945,
        "ci95_high": 0.4987070616637299,
        "favorable_pairs": 21,
    },
    "cil_last_accuracy": {
        "difference_mean": 0.01907876805669133,
        "ci95_low": -0.012786910451483006,
        "ci95_high": 0.05094444656486567,
        "favorable_pairs": 15,
    },
    "effective_rank": {
        "difference_mean": 0.605174446105957,
        "ci95_low": 0.5319496766334371,
        "ci95_high": 0.678399215578477,
        "favorable_pairs": 30,
    },
    "prototype_overlap": {
        "difference_mean": 0.010490044951438904,
        "ci95_low": 0.009221775432941181,
        "ci95_high": 0.011758314469936627,
        "favorable_pairs": 30,
    },
}

PET_METRICS = {
    "avg_accuracy": ("avg_accuracy_control", "avg_accuracy_treatment", True),
    "avg_forgetting": ("avg_forgetting_control", "avg_forgetting_treatment", False),
    "cil_last_accuracy": (
        "cil_last_accuracy_control",
        "cil_last_accuracy_treatment",
        True,
    ),
    "effective_rank": ("effective_rank_control", "effective_rank_treatment", True),
    "prototype_overlap": (
        "prototype_overlap_control",
        "prototype_overlap_treatment",
        False,
    ),
}

MANUSCRIPT_ENDPOINTS = (
    "avg_accuracy",
    "avg_forgetting",
    "effective_rank",
    "prototype_overlap",
)


class VerificationError(AssertionError):
    """Raised when an artifact no longer matches its locked evidence."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def close(actual: float, expected: float, *, tolerance: float = 1e-12) -> bool:
    return math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=tolerance)


def require_close(actual: float, expected: float, label: str) -> None:
    require(close(actual, expected), f"{label}: expected {expected!r}, found {actual!r}")


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing artifact: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), f"artifact must contain a JSON object: {path}")
    return payload


def verify_checksums(artifact_dir: Path) -> None:
    checksum_path = artifact_dir / CHECKSUM_FILE
    require(checksum_path.is_file(), f"missing artifact: {checksum_path}")
    expected_files = {
        "direct_attribution_summary.json",
        "pet_vit_lora_confirmation.json",
        "segmentation_direct_confirmation.json",
    }
    recorded: dict[str, str] = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        recorded[relative] = digest
    require(set(recorded) == expected_files, "checksum inventory changed")
    for relative, expected in recorded.items():
        path = artifact_dir / relative
        require(path.is_file(), f"missing checksummed artifact: {path}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        require(actual == expected, f"checksum mismatch: {relative}")


def verify_pet(payload: dict[str, Any]) -> dict[str, Any]:
    require(payload.get("status") == "PASS_DUAL", "Pet confirmation status is not PASS_DUAL")
    require(payload.get("selected") == PET_CONFIG, "Pet selected configuration changed")
    for gate in (
        "selection_gate_passed",
        "primary_accuracy_gate_passed",
        "key_secondary_forgetting_gate_passed",
        "mechanism_gate_passed",
    ):
        require(payload.get(gate) is True, f"Pet locked gate failed: {gate}")

    rows = payload.get("rows")
    require(isinstance(rows, list) and len(rows) == 30, "Pet artifact must have 30 paired rows")
    seeds = [int(row["seed"]) for row in rows]
    require(len(set(seeds)) == 30, "Pet paired seeds must be unique")
    require(all(row.get("dataset") == "oxfordiiitpet_cl" for row in rows), "unexpected Pet dataset")
    require(
        all(row.get("initial_model_hash") and row.get("pairing_hash") for row in rows),
        "each Pet row must retain model and pairing hashes",
    )

    summary = payload.get("summary", {})
    require(summary.get("n") == 30, "Pet summary n must be 30")
    require(summary.get("control_arm") == "task", "Pet control must be Task-only")
    require(summary.get("treatment_arm") == "task_ssr", "Pet treatment must be Task+SSR")
    require(summary.get("dual_wins") == 19, "Pet locked dual-win count changed")
    stored_metrics = summary.get("metrics", {})

    for metric, (control_key, treatment_key, higher_is_better) in PET_METRICS.items():
        controls = [float(row[control_key]) for row in rows]
        treatments = [float(row[treatment_key]) for row in rows]
        recomputed = paired_summary(
            controls,
            treatments,
            higher_is_better=higher_is_better,
        ).as_dict()
        stored = stored_metrics.get(metric, {})
        require(stored.get("higher_is_better") is higher_is_better, f"direction changed: {metric}")
        for field, actual in recomputed.items():
            if isinstance(actual, int):
                require(stored.get(field) == actual, f"Pet summary mismatch: {metric}/{field}")
            else:
                require_close(stored[field], actual, f"Pet summary mismatch: {metric}/{field}")
        for field, expected in PET_EXPECTED[metric].items():
            if isinstance(expected, int):
                require(stored.get(field) == expected, f"locked Pet value changed: {metric}/{field}")
            else:
                require_close(stored[field], expected, f"locked Pet value changed: {metric}/{field}")

    delta_checks = {
        "avg_accuracy_delta": lambda row: row["avg_accuracy_treatment"] - row["avg_accuracy_control"],
        "avg_forgetting_reduction": lambda row: row["avg_forgetting_control"]
        - row["avg_forgetting_treatment"],
        "cil_last_accuracy_delta": lambda row: row["cil_last_accuracy_treatment"]
        - row["cil_last_accuracy_control"],
        "effective_rank_delta": lambda row: row["effective_rank_treatment"]
        - row["effective_rank_control"],
        "prototype_overlap_reduction": lambda row: row["prototype_overlap_control"]
        - row["prototype_overlap_treatment"],
    }
    for row in rows:
        for field, calculate in delta_checks.items():
            require_close(row[field], calculate(row), f"paired-row arithmetic: seed {row['seed']}/{field}")

    for metric in MANUSCRIPT_ENDPOINTS:
        require(stored_metrics[metric]["ci95_low"] > 0.0, f"manuscript CI is not positive: {metric}")
    cil = stored_metrics["cil_last_accuracy"]
    require(cil["ci95_low"] < 0.0 < cil["ci95_high"], "CIL-last must remain a cross-zero boundary")

    return {
        "n": 30,
        "selected": payload["selected"],
        "manuscript_endpoints": {
            metric: {
                "difference_mean": stored_metrics[metric]["difference_mean"],
                "ci95": [stored_metrics[metric]["ci95_low"], stored_metrics[metric]["ci95_high"]],
                "favorable_pairs": stored_metrics[metric]["favorable_pairs"],
            }
            for metric in MANUSCRIPT_ENDPOINTS
        },
        "cil_last_boundary": {
            "difference_mean": cil["difference_mean"],
            "ci95": [cil["ci95_low"], cil["ci95_high"]],
            "crosses_zero": True,
            "manuscript_endpoint": False,
        },
    }


def verify_segmentation(payload: dict[str, Any]) -> dict[str, Any]:
    require(payload.get("protocol") == "direct_ssr_prototype_segmentation_v1", "unexpected protocol")
    require(payload.get("comparison") == "Task-only versus Task+SSR", "unexpected comparison")
    require(payload.get("primary_metric") == "mean_iou", "unexpected segmentation primary metric")
    require(payload.get("all_primary_gates") is False, "segmentation must remain a boundary audit")

    datasets = payload.get("datasets", {})
    require(
        set(datasets) == {"cub200", "oxford_iiit_pet", "oxford_flowers102"},
        "segmentation dataset inventory changed",
    )
    report: dict[str, Any] = {}
    for dataset, record in sorted(datasets.items()):
        require(record.get("primary_gate") is False, f"unexpected primary pass: {dataset}")
        rows = record.get("per_seed")
        require(isinstance(rows, list) and len(rows) == record.get("n"), f"row count mismatch: {dataset}")
        require(len({int(row["seed"]) for row in rows}) == len(rows), f"duplicate seeds: {dataset}")
        recomputed = paired_summary(
            [float(row["control_mean_iou"]) for row in rows],
            [float(row["treatment_mean_iou"]) for row in rows],
            higher_is_better=True,
        ).as_dict()
        stored = record["metrics"]["mean_iou"]
        for field, actual in recomputed.items():
            if isinstance(actual, int):
                require(stored.get(field) == actual, f"segmentation summary mismatch: {dataset}/{field}")
            else:
                require_close(stored[field], actual, f"segmentation summary mismatch: {dataset}/{field}")
        require(
            stored["ci95_low"] < 0.0 < stored["ci95_high"],
            f"segmentation primary CI must remain cross-zero: {dataset}",
        )
        report[dataset] = {
            "n": record["n"],
            "primary_gate": False,
            "difference_mean": stored["difference_mean"],
            "ci95": [stored["ci95_low"], stored["ci95_high"]],
            "crosses_zero": True,
        }
    return {"role": "boundary_audit", "all_primary_gates": False, "datasets": report}


def verify_artifacts(artifact_dir: Path = DEFAULT_ARTIFACT_DIR) -> dict[str, Any]:
    verify_checksums(artifact_dir)
    return {
        "pet_vit_lora": verify_pet(load_json(artifact_dir / "pet_vit_lora_confirmation.json")),
        "segmentation_direct": verify_segmentation(
            load_json(artifact_dir / "segmentation_direct_confirmation.json")
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = verify_artifacts(args.artifact_dir)
    except (KeyError, TypeError, ValueError, VerificationError) as exc:
        print(f"verification failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
