from __future__ import annotations

import json
import hashlib
from argparse import Namespace

import pytest

from scripts.run_meeting_segmentation_ablation import (
    CONFIRMATION_SEEDS,
    DEVELOPMENT_SEEDS,
    FACTORIAL_CONFIRMATION_SEEDS,
    METHODS,
    PROTOCOL_FACTORIAL,
    SPECS,
    confirmation_jobs,
    manifest,
    result_record,
    screen_jobs,
    select_spec,
    validate_job_record,
    validate_paired_records,
    validate_selection_lock,
    write_confirmation_summary,
)
from ssr_utils.dataset_fingerprint import cub_dataset_fingerprint
from ssr_utils.result_schema import build_result_record


def _objective(method: str) -> dict[str, bool]:
    return {
        "task": True,
        "kd": method in {"kd", "biocs_kd"},
        "ssr": method in {"biocs", "biocs_kd"},
    }


def _write_factorial_record(
    job,
    *,
    selected_spec,
    selection_lock_hash: str,
) -> None:
    uses_ssr = job.method in {"biocs", "biocs_kd"}
    base_iou = 70.0 + (job.seed % 7) / 100.0
    base_dice = 80.0 + (job.seed % 7) / 100.0
    base_forgetting = 4.0 + (job.seed % 3) / 100.0
    if job.method in {"kd", "biocs_kd"}:
        base_iou += 0.25
        base_dice += 0.15
        base_forgetting -= 0.10
    if job.method == "biocs":
        effect = (3.0, 2.0, 1.0) if job.spec == selected_spec else (0.4, 0.3, 0.2)
        base_iou += effect[0]
        base_dice += effect[1]
        base_forgetting -= effect[2]
    if job.method == "biocs_kd":
        effect = (4.0, 3.0, 1.5) if job.spec == selected_spec else (0.5, 0.4, 0.25)
        base_iou += effect[0]
        base_dice += effect[1]
        base_forgetting -= effect[2]
    payload = build_result_record(
        git_commit="commit-a",
        run_id=job.output.name,
        task_family="segmentation",
        dataset="cub200_masks",
        model="resnet18_dense_decoder",
        seed=job.seed,
        objective=_objective(job.method),
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
        metrics={
            "mean_iou": base_iou,
            "mean_dice": base_dice,
            "avg_forgetting_iou": base_forgetting,
        },
        runtime={"elapsed_s": 1.0},
        config={"job": job.output.name},
        dataset_hash=f"dataset-{job.seed}",
        recipe={
            "baseline": "plain",
            "kd": "kd",
            "biocs": "ssr_only",
            "biocs_kd": "ssr_kd",
        }[job.method],
        selection_lock_hash=selection_lock_hash,
        lambda_ssr=job.spec.lambda_sp if uses_ssr else 0.0,
        initial_model_hash=f"{job.seed:064x}",
        task_schedule_hash=f"{job.seed + 1:064x}",
        optimizer_steps=480,
        lambda_kd=2.0,
        lambda_kd_seg=2.0,
        segmentation_source_sha256="a" * 64,
        segmentation_cache_sha256="b" * 64,
    )
    path = result_record(job)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_segmentation_matrix_sizes(tmp_path):
    screen = screen_jobs(tmp_path)
    assert len(screen) == 21
    assert {job.seed for job in screen} == set(DEVELOPMENT_SEEDS)
    confirm = confirmation_jobs(tmp_path, screen[1].spec)
    assert len(confirm) == 40
    assert {job.seed for job in confirm} == set(CONFIRMATION_SEEDS)
    assert {job.method for job in confirm} == {"baseline", "kd", "biocs", "biocs_kd"}


def test_factorial_segmentation_manifest_has_42_screen_and_80_confirmation_jobs(tmp_path):
    args = Namespace(
        result_root=tmp_path,
        protocol=PROTOCOL_FACTORIAL,
        confirmation_seeds=FACTORIAL_CONFIRMATION_SEEDS,
        confirmation_methods=METHODS,
    )
    payload = manifest(args)
    assert payload["jobs"] == {"development": 42, "confirmation": 80, "total": 122}
    assert payload["selection_contexts"] == ["task_to_task_ssr", "kd_to_kd_ssr"]

    screen = screen_jobs(tmp_path, PROTOCOL_FACTORIAL)
    assert len(screen) == 42
    for seed in DEVELOPMENT_SEEDS:
        jobs = [job for job in screen if job.seed == seed]
        assert {job.method for job in jobs if job.spec.spec_id == "control"} == {
            "baseline",
            "kd",
        }
        assert len([job for job in jobs if job.method == "biocs"]) == len(SPECS)
        assert len([job for job in jobs if job.method == "biocs_kd"]) == len(SPECS)


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


def test_factorial_lock_summary_and_pair_audit_cover_all_four_arms(tmp_path):
    selected_spec = SPECS[1]
    screen = screen_jobs(tmp_path, PROTOCOL_FACTORIAL)
    for job in screen:
        _write_factorial_record(
            job,
            selected_spec=selected_spec,
            selection_lock_hash="development_screen",
        )

    args = Namespace(
        protocol=PROTOCOL_FACTORIAL,
        confirmation_seeds=FACTORIAL_CONFIRMATION_SEEDS,
        confirmation_methods=METHODS,
        segmentation_source_sha256="a" * 64,
        segmentation_cache_sha256="b" * 64,
    )
    selected = select_spec(screen, tmp_path, args, "commit-a")
    assert selected == selected_spec
    lock_path = tmp_path / "SELECTION_LOCK.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    assert lock["selection_contexts"] == ["task_to_task_ssr", "kd_to_kd_ssr"]
    assert len(lock["screen_result_records"]) == 42
    assert validate_selection_lock(lock_path, args, "commit-a") == selected_spec

    args.selection_lock_sha256 = lock["lock_sha256"]
    confirmation = confirmation_jobs(
        tmp_path,
        selected_spec,
        FACTORIAL_CONFIRMATION_SEEDS,
        METHODS,
    )
    for job in confirmation:
        _write_factorial_record(
            job,
            selected_spec=selected_spec,
            selection_lock_hash=args.selection_lock_sha256,
        )
    write_confirmation_summary(confirmation, tmp_path, args, "commit-a")

    summary = json.loads((tmp_path / "CONFIRMATION_SUMMARY.json").read_text())
    assert set(summary["contrasts"]) == {
        "task_to_task_ssr",
        "kd_to_kd_ssr",
        "task_to_kd",
        "task_ssr_to_kd_ssr",
        "task_to_full_recipe",
    }
    assert summary["contrasts"]["task_to_task_ssr"]["metrics"]["mean_iou"]["n"] == 20
    assert summary["factorial_interaction"]["metrics"]["mean_iou"][
        "difference_mean"
    ] == pytest.approx(1.0)

    incomplete = [
        job
        for job in confirmation
        if not (job.seed == FACTORIAL_CONFIRMATION_SEEDS[0] and job.method == "baseline")
    ]
    with pytest.raises(RuntimeError, match="Factorial confirmation inventory mismatch"):
        write_confirmation_summary(incomplete, tmp_path, args, "commit-a")


def test_optional_record_data_hashes_must_match_the_launcher_assets(tmp_path):
    job = screen_jobs(tmp_path, PROTOCOL_FACTORIAL)[0]
    _write_factorial_record(
        job,
        selected_spec=SPECS[0],
        selection_lock_hash="development_screen",
    )
    validate_job_record(
        job,
        "commit-a",
        "development_screen",
        expected_segmentation_source_sha256="a" * 64,
        expected_segmentation_cache_sha256="b" * 64,
    )

    path = result_record(job)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["segmentation_cache_sha256"] = "c" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="segmentation_cache_sha256"):
        validate_job_record(
            job,
            "commit-a",
            "development_screen",
            expected_segmentation_source_sha256="a" * 64,
            expected_segmentation_cache_sha256="b" * 64,
        )


def test_pair_audit_rejects_new_trainer_field_mismatch():
    control = {
        "git_commit": "commit-a",
        "task_family": "segmentation",
        "dataset": "cub200_masks",
        "model": "resnet18_dense_decoder",
        "seed": 9201,
        "dataset_hash": "dataset-a",
        "selection_lock_hash": "lock-a",
        "initial_model_hash": "a" * 64,
        "task_schedule_hash": "b" * 64,
        "optimizer_steps": 480,
        "lambda_kd": 2.0,
        "lambda_kd_seg": 2.0,
        "objective": _objective("baseline"),
    }
    treatment = dict(control)
    treatment["objective"] = _objective("biocs")
    validate_paired_records(
        control,
        treatment,
        identity="factorial/seed_9201",
        allowed_objective_changes=("ssr",),
    )

    treatment["optimizer_steps"] = 481
    with pytest.raises(RuntimeError, match="optimizer_steps"):
        validate_paired_records(
            control,
            treatment,
            identity="factorial/seed_9201",
            allowed_objective_changes=("ssr",),
        )
