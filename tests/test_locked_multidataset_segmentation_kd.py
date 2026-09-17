from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.run_locked_multidataset_segmentation_kd import (
    CONFIRMATION_SEEDS,
    METHODS,
    confirmation_jobs,
    load_selection_lock,
    validate_paired_records,
)
from ssr_utils.result_schema import sha256_value


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def write_lock(path: Path) -> dict:
    payload = {
        "protocol": "meeting_20260803_cub_segmentation_2x2_old_class_kd_v1",
        "selected": {
            "spec_id": "locked",
            "lambda_sp": 0.075,
            "sigma_exc": 0.16,
            "sigma_inh": 0.45,
        },
        "development_seeds": [9101, 9103, 9105],
    }
    payload["lock_sha256"] = sha256_value(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def test_cub_lock_is_verified_and_parameters_are_read(tmp_path):
    path = tmp_path / "SELECTION_LOCK.json"
    payload = write_lock(path)
    spec = load_selection_lock(path)
    assert spec.lambda_ssr == 0.075
    assert spec.sigma_exc == 0.16
    assert spec.sigma_inh == 0.45
    assert spec.lock_sha256 == payload["lock_sha256"]

    payload["selected"]["lambda_sp"] = 9.0
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        load_selection_lock(path)


def test_locked_confirmation_matrix_has_twenty_independent_pairs(tmp_path):
    jobs = confirmation_jobs(tmp_path)
    assert len(CONFIRMATION_SEEDS) == 20
    assert CONFIRMATION_SEEDS == tuple(range(9501, 9540, 2))
    assert len(jobs) == 40
    for seed in CONFIRMATION_SEEDS:
        subset = [job for job in jobs if job.seed == seed]
        assert {job.method for job in subset} == set(METHODS)
        assert len(subset) == 2


def test_pair_audit_rejects_training_budget_mismatch():
    shared = {
        "git_commit": "a" * 40,
        "dataset": "oxford_iiit_pet_masks",
        "seed": 9501,
        "dataset_hash": "b" * 64,
        "source_dataset_fingerprint": "c" * 64,
        "feature_cache_sha256": "d" * 64,
        "encoder_state_sha256": "e" * 64,
        "torchvision_version": "0.0",
        "initial_model_hash": "f" * 64,
        "task_schedule_hash": "0" * 64,
        "optimizer_steps": 100,
        "lambda_kd": 2.0,
        "selection_lock_hash": "1" * 64,
        "objective": {
            "task": True,
            "kd": True,
            "ssr": False,
            "anchor": False,
            "spectral": False,
            "ewc": False,
            "mas": False,
            "si": False,
            "center": False,
            "protodecor": False,
        },
    }
    treatment = dict(shared)
    treatment["objective"] = dict(shared["objective"], ssr=True)
    validate_paired_records(shared, treatment, identity="pet/seed_9501")

    treatment["optimizer_steps"] = 101
    with pytest.raises(RuntimeError, match="optimizer_steps"):
        validate_paired_records(shared, treatment, identity="pet/seed_9501")


def test_manifest_only_needs_lock_but_not_dataset_payload(tmp_path):
    lock_path = tmp_path / "SELECTION_LOCK.json"
    payload = write_lock(lock_path)
    result_root = tmp_path / "results"
    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_locked_multidataset_segmentation_kd.py"),
            "--result-root",
            str(result_root),
            "--dataset",
            "oxford_iiit_pet",
            "--dataset-root",
            str(tmp_path / "absent_dataset"),
            "--feature-cache",
            str(tmp_path / "absent_cache.pt"),
            "--selection-lock",
            str(lock_path),
            "--manifest-only",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    manifest = json.loads((result_root / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["primary_contrast"] == "kd_ssr_minus_kd"
    assert manifest["selection_lock_sha256"] == payload["lock_sha256"]
    assert manifest["jobs"] == {"kd": 20, "kd_ssr": 20, "total": 40}
    assert manifest["training"]["lambda_kd"] == 2.0
    assert manifest["training"]["epochs_per_task"] == 20
    assert '"total": 40' in completed.stdout
