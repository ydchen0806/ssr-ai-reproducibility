from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from scripts.run_segmentation_nested_dataset import (
    CANDIDATES,
    CONFIRMATION_SEEDS,
    CONTEXTS,
    REFINE_SEEDS,
    SCREEN_SEEDS,
    Job,
    command,
    confirmation_jobs,
    control_jobs,
    manifest,
    method_name,
    treatment_jobs,
)


def _args(tmp_path: Path, dataset: str) -> Namespace:
    return Namespace(
        dataset=dataset,
        dataset_root=tmp_path / dataset,
        feature_cache=tmp_path / "cache.pt",
        feature_cache_sha256="a" * 64,
        segmentation_source_sha256="b" * 64 if dataset == "cub200" else None,
        result_root=tmp_path / "results",
        gpus=[str(index) for index in range(8)],
        jobs_per_gpu=2,
        workers=0,
        batch_size=24 if dataset == "cub200" else 32,
        python="python3",
    )


def test_manifest_prespecifies_bounded_three_stage_budget(tmp_path):
    args = _args(tmp_path, "oxford_flowers102")
    payload = manifest(args, "commit-a")

    assert len(payload["candidate_matrix"]) == 54
    assert payload["jobs"] == {
        "screen": 336,
        "refine_maximum": 60,
        "confirmation": 40,
        "total_maximum": 436,
    }
    assert payload["screen_seeds"] == list(SCREEN_SEEDS)
    assert payload["refine_seeds"] == list(REFINE_SEEDS)
    assert payload["confirmation_seeds"] == list(CONFIRMATION_SEEDS)
    assert payload["selection_contexts"] == [
        "task_to_task_ssr",
        "kd_to_kd_ssr",
    ]
    assert payload["training_epochs_by_phase"] == {
        "screen": 8,
        "refine": 15,
        "confirmation": 20,
    }
    assert payload["evaluation_policy"]["confirmation"] == "official_test_after_lock"


def test_screen_has_target_matched_controls_for_both_host_objectives(tmp_path):
    root = tmp_path / "results"
    contexts = {context: CANDIDATES for context in CONTEXTS}
    jobs = control_jobs(root, "screen", SCREEN_SEEDS) + treatment_jobs(
        root, "screen", SCREEN_SEEDS, contexts
    )

    assert len(jobs) == 336
    for context in CONTEXTS:
        for seed in SCREEN_SEEDS:
            controls = [
                job
                for job in jobs
                if job.context == context and job.seed == seed and not job.treatment
            ]
            assert {(job.target, job.scope) for job in controls} == {
                ("class", "new_old"),
                ("channels", "all"),
            }
            treatments = [
                job
                for job in jobs
                if job.context == context and job.seed == seed and job.treatment
            ]
            assert len(treatments) == 54


def test_commands_keep_base_budget_and_change_only_ssr_arm(tmp_path):
    spec = next(candidate for candidate in CANDIDATES if candidate.scope == "new_old")
    for dataset in ("cub200", "oxford_iiit_pet", "oxford_flowers102"):
        args = _args(tmp_path, dataset)
        for context in CONTEXTS:
            job = Job(
                phase="screen",
                context=context,
                seed=SCREEN_SEEDS[0],
                target=spec.target,
                scope=spec.scope,
                treatment=True,
                spec=spec,
                output=tmp_path / dataset / context,
            )
            argv = command(job, args, None)
            expected_method = method_name(dataset, context, True)
            method_flag = "--methods" if dataset == "cub200" else "--method"
            assert argv[argv.index(method_flag) + 1] == expected_method
            scope_flag = "--seg_biocs_scope" if dataset == "cub200" else "--ssr-scope"
            assert argv[argv.index(scope_flag) + 1] == "new_old"
            start_flag = (
                "--seg_biocs_start_task"
                if dataset == "cub200"
                else "--ssr-start-task"
            )
            assert argv[argv.index(start_flag) + 1] == "1"
            kd_flag = "--lambda_kd_seg" if dataset == "cub200" else "--lambda-kd"
            assert argv[argv.index(kd_flag) + 1] == "2.0"
            epoch_flag = "--seg_epochs" if dataset == "cub200" else "--epochs"
            assert argv[argv.index(epoch_flag) + 1] == "8"
            evaluation_flag = (
                "--seg_evaluation_split"
                if dataset == "cub200"
                else "--evaluation-split"
            )
            assert argv[argv.index(evaluation_flag) + 1] == "validation"


def test_confirmation_inventory_is_exactly_ten_pairs_per_context(tmp_path):
    selected = {
        "task": CANDIDATES[0],
        "kd": CANDIDATES[-1],
    }
    jobs = confirmation_jobs(tmp_path, selected)

    assert len(jobs) == 40
    for context in CONTEXTS:
        context_jobs = [job for job in jobs if job.context == context]
        assert len(context_jobs) == 20
        assert {job.seed for job in context_jobs} == set(CONFIRMATION_SEEDS)
        assert sum(job.treatment for job in context_jobs) == 10

    confirmation_treatment = next(job for job in jobs if job.treatment)
    args = _args(tmp_path, "oxford_flowers102")
    argv = command(confirmation_treatment, args, "f" * 64)
    assert argv[argv.index("--evaluation-split") + 1] == "test"
