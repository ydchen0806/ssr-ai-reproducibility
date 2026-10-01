"""Validation helpers for the locked KD-only teacher trajectory."""

from __future__ import annotations

import json
from pathlib import Path

from ssr_utils.result_schema import sha256_file, sha256_value


TEACHER_PROTOCOL = "locked_kd_only_trajectory_v1"


def validate_teacher_trajectory(
    root: Path,
    *,
    dataset: str,
    model: str,
    seed: int,
    expected_checkpoints: int,
    expected_trajectory_hash: str | None = None,
) -> dict:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"teacher manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_identity = {
        "protocol": TEACHER_PROTOCOL,
        "dataset": dataset,
        "model": model,
        "seed": seed,
    }
    mismatches = {
        key: {"expected": value, "observed": manifest.get(key)}
        for key, value in expected_identity.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(f"teacher manifest identity mismatch: {mismatches}")

    checkpoints = manifest.get("checkpoints")
    if not isinstance(checkpoints, list):
        raise ValueError("teacher manifest checkpoints must be a list")
    task_ids = [entry.get("task_id") for entry in checkpoints]
    if task_ids != list(range(expected_checkpoints)):
        raise ValueError(
            "teacher checkpoint sequence mismatch: "
            f"expected {list(range(expected_checkpoints))}, observed {task_ids}"
        )
    for entry in checkpoints:
        checkpoint = root / str(entry.get("file", ""))
        if not checkpoint.is_file():
            raise ValueError(f"teacher checkpoint is missing: {checkpoint}")
        observed = sha256_file(checkpoint)
        if observed != entry.get("sha256"):
            raise ValueError(
                f"teacher checkpoint hash mismatch for {checkpoint}: "
                f"expected {entry.get('sha256')}, observed {observed}"
            )

    trajectory_hash = sha256_value(manifest)
    if expected_trajectory_hash and trajectory_hash != expected_trajectory_hash:
        raise ValueError(
            "teacher trajectory hash mismatch: "
            f"expected {expected_trajectory_hash}, observed {trajectory_hash}"
        )
    return {"manifest": str(manifest_path), "sha256": trajectory_hash, **manifest}
