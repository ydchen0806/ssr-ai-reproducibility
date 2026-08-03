from __future__ import annotations

import json

import pytest

from scripts.validate_result_record import validate_identity
from ssr_utils.result_schema import ResultSchemaError, build_result_record


def _record(tmp_path, *, seed=7, git_commit="abc123"):
    record = build_result_record(
        git_commit=git_commit,
        run_id="test-run",
        task_family="classification",
        dataset="split_cifar100",
        model="resnet18",
        seed=seed,
        objective={"task": True, "kd": True},
        distance_mapping="none",
        kernel={},
        metrics={"avg_accuracy": 70.0},
        runtime={},
        config={"method": {"name": "kd"}},
        dataset_hash="dataset-v1",
        recipe="kd",
    )
    path = tmp_path / "result_record.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def test_identity_validator_accepts_exact_record(tmp_path):
    path = _record(tmp_path)

    record = validate_identity(
        path,
        {
            "task_family": "classification",
            "dataset": "split_cifar100",
            "model": "resnet18",
            "recipe": "kd",
            "distance_mapping": "none",
            "seed": 7,
        },
        expected_git_commit="abc123",
    )

    assert record["seed"] == 7


def test_identity_validator_rejects_stale_seed_or_commit(tmp_path):
    path = _record(tmp_path)

    with pytest.raises(ResultSchemaError, match="identity mismatch"):
        validate_identity(path, {"seed": 9})
    with pytest.raises(ResultSchemaError, match="git_commit mismatch"):
        validate_identity(path, {"seed": 7}, expected_git_commit="different")


def test_identity_validator_allows_explicit_legacy_provenance(tmp_path):
    path = _record(tmp_path, git_commit="legacy-source-fingerprint:locked")

    validate_identity(
        path,
        {"seed": 7},
        expected_git_commit="current",
        allow_legacy_git=True,
    )
