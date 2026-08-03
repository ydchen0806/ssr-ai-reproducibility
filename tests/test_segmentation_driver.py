from __future__ import annotations

import json
import hashlib
from argparse import Namespace

import pytest

from scripts.run_meeting_segmentation_ablation import (
    CONFIRMATION_SEEDS,
    DEVELOPMENT_SEEDS,
    confirmation_jobs,
    result_record,
    screen_jobs,
    select_spec,
    validate_selection_lock,
)
from ssr_utils.dataset_fingerprint import cub_dataset_fingerprint
from ssr_utils.result_schema import build_result_record


def test_segmentation_matrix_sizes(tmp_path):
    screen = screen_jobs(tmp_path)
    assert len(screen) == 21
    assert {job.seed for job in screen} == set(DEVELOPMENT_SEEDS)
    confirm = confirmation_jobs(tmp_path, screen[1].spec)
    assert len(confirm) == 40
    assert {job.seed for job in confirm} == set(CONFIRMATION_SEEDS)
    assert {job.method for job in confirm} == {"baseline", "kd", "biocs", "biocs_kd"}


def test_segmentation_fingerprint_includes_source_and_cache_hashes(tmp_path):
    metadata = tmp_path / "CUB_200_2011"
    metadata.mkdir()
    (metadata / "images.txt").write_text("1 image.jpg\n", encoding="utf-8")

    first = cub_dataset_fingerprint(
        tmp_path,
        segmentation_source_sha256="mask-v1",
        segmentation_cache_sha256="cache-v1",
    )
    second = cub_dataset_fingerprint(
        tmp_path,
        segmentation_source_sha256="mask-v1",
        segmentation_cache_sha256="cache-v2",
    )

    assert first != second


def test_selection_lock_binds_commit_data_and_screen_records(tmp_path):
    jobs = screen_jobs(tmp_path)
    for job in jobs:
        job.output.mkdir(parents=True, exist_ok=True)
        values = {
            "mean_iou": 70.0 + (1.0 if job.method == "biocs_kd" else 0.0),
            "mean_dice": 80.0 + (1.0 if job.method == "biocs_kd" else 0.0),
            "avg_forgetting_iou": 4.0 - (1.0 if job.method == "biocs_kd" else 0.0),
        }
        record = result_record(job)
        record.parent.mkdir(parents=True, exist_ok=True)
        uses_ssr = "biocs" in job.method
        payload = build_result_record(
            git_commit="commit-a",
            run_id=job.output.name,
            task_family="segmentation",
            dataset="cub200_masks",
            model="resnet18_dense_decoder",
            seed=job.seed,
            objective={"task": True, "kd": True, "ssr": uses_ssr},
            distance_mapping="cosine" if uses_ssr else "none",
            kernel=(
                {
                    "family": "gaussian",
                    "A_exc": 1.0,
                    "A_inh": 0.8,
                    "sigma_exc": job.spec.sigma_exc,
                    "sigma_inh": job.spec.sigma_inh,
                }
                if uses_ssr
                else {}
            ),
            metrics=values,
            runtime={"elapsed_s": 1.0},
            config={"job": job.output.name},
            dataset_hash="dataset-a",
            recipe="ssr_kd" if uses_ssr else "kd",
            selection_lock_hash="development_screen",
        )
        record.write_text(json.dumps(payload), encoding="utf-8")

    args = Namespace(
        segmentation_source_sha256="a" * 64,
        segmentation_cache_sha256="b" * 64,
    )
    selected = select_spec(jobs, tmp_path, args, "commit-a")
    lock_path = tmp_path / "SELECTION_LOCK.json"

    assert validate_selection_lock(lock_path, args, "commit-a") == selected

    forged = json.loads(lock_path.read_text(encoding="utf-8"))
    forged["screen_result_records"].pop()
    unhashed = {key: value for key, value in forged.items() if key != "lock_sha256"}
    forged["lock_sha256"] = hashlib.sha256(
        json.dumps(unhashed, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    lock_path.write_text(json.dumps(forged), encoding="utf-8")
    with pytest.raises(RuntimeError, match="screen inventory mismatch"):
        validate_selection_lock(lock_path, args, "commit-a")

    select_spec(jobs, tmp_path, args, "commit-a")

    first_record = result_record(jobs[0])
    first_record.write_text("tampered", encoding="utf-8")
    with pytest.raises(RuntimeError, match="screen artifact mismatch"):
        validate_selection_lock(lock_path, args, "commit-a")
