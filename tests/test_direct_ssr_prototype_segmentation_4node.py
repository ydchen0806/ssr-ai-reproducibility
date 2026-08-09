from __future__ import annotations

import json
import os
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from scripts.run_direct_ssr_prototype_segmentation_4node import (
    CANDIDATES,
    CONFIRMATION_SEEDS,
    DEVELOPMENT_DATASETS,
    REFINE_SEEDS,
    SCREEN_SEEDS,
    DatasetSpec,
    PairJob,
    command,
    manifest,
    node_groups,
    pair_jobs,
    sign_flip_test,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WRAPPER = (
    PROJECT_ROOT
    / "scripts"
    / "submit_direct_ssr_prototype_segmentation_4node_20260809.sh"
)
RANK_VARIABLES = (
    "GLOBAL_NODE_RANK",
    "NODE_RANK",
    "PADDLE_TRAINER_ID",
    "SLURM_NODEID",
    "GROUP_RANK",
)


def _args(tmp_path: Path, *, rank: int = 0) -> Namespace:
    return Namespace(
        result_root=tmp_path / "results",
        node_rank=rank,
        nodes=4,
        gpus=[str(index) for index in range(8)],
        jobs_per_gpu=2,
        workers=0,
        cub_batch_size=24,
        multidataset_batch_size=32,
        python=sys.executable,
        barrier_timeout=30,
        cub_root=tmp_path / "cub200",
        cub_cache=tmp_path / "cub200" / "cache.pt",
        pet_root=tmp_path / "oxford_iiit_pet",
        pet_cache=tmp_path / "oxford_iiit_pet" / "cache.pt",
        flowers_root=tmp_path / "flowers-102",
        flowers_cache=tmp_path / "flowers-102" / "cache.pt",
    )


def test_manifest_supports_three_node_sharding_for_combined_attribution_run(tmp_path):
    args = _args(tmp_path)
    args.nodes = 3

    payload = manifest(args, "a" * 40, None)

    assert payload["compute"]["nodes"] == 3
    assert payload["jobs"]["total_runs"] == 332


def _datasets(tmp_path: Path) -> dict[str, DatasetSpec]:
    return {
        "cub200": DatasetSpec(
            "cub200", tmp_path / "cub200", tmp_path / "cub.pt", "a" * 64, "b" * 64
        ),
        "oxford_iiit_pet": DatasetSpec(
            "oxford_iiit_pet", tmp_path / "pet", tmp_path / "pet.pt", "c" * 64
        ),
        "oxford_flowers102": DatasetSpec(
            "oxford_flowers102",
            tmp_path / "flowers",
            tmp_path / "flowers.pt",
            "d" * 64,
        ),
    }


def test_manifest_locks_direct_ssr_matrix_and_hidden_confirmation_budget(tmp_path):
    payload = manifest(_args(tmp_path), "f" * 40, _datasets(tmp_path))

    assert len(CANDIDATES) == 16
    assert payload["comparison"] == (
        "Task-only versus Task+SSR; no KD, anchor or spectral term"
    )
    assert payload["jobs"] == {
        "screen_pairs": 96,
        "screen_controls": 6,
        "screen_treatments": 96,
        "screen_runs": 102,
        "refine_pairs": 56,
        "refine_controls": 14,
        "refine_treatments": 56,
        "refine_runs": 70,
        "confirmation_pairs": 80,
        "confirmation_controls": 80,
        "confirmation_treatments": 80,
        "confirmation_runs": 160,
        "total_pairs": 232,
        "total_runs": 332,
    }
    assert payload["development"]["screen_seeds"] == list(SCREEN_SEEDS)
    assert payload["development"]["refine_seeds"] == list(REFINE_SEEDS)
    assert payload["development"]["selection_uses_geometry"] is False
    assert {
        name: len(seeds) for name, seeds in payload["confirmation"]["datasets_and_seeds"].items()
    } == {"cub200": 30, "oxford_iiit_pet": 30, "oxford_flowers102": 20}


@pytest.mark.parametrize("dataset", ("cub200", "oxford_iiit_pet"))
def test_commands_change_only_task_ssr_arm_and_opt_into_prototype_head(tmp_path, dataset):
    args = _args(tmp_path)
    datasets = _datasets(tmp_path)
    spec = CANDIDATES[0]
    job = PairJob(
        "screen",
        dataset,
        SCREEN_SEEDS[0],
        spec,
        tmp_path / "pair",
        tmp_path / "control",
    )
    control = command(job, False, args, datasets, None)
    treatment = command(job, True, args, datasets, None)

    assert control[control.index("--segmentation-head") + 1] == "prototype_cosine"
    assert treatment[treatment.index("--segmentation-head") + 1] == "prototype_cosine"
    assert control[control.index("--ssr-warmup-fraction") + 1] == "0.2"
    assert control[control.index("--ssr-ramp-fraction") + 1] == "0.2"
    method_flag = "--methods" if dataset == "cub200" else "--method"
    assert control[control.index(method_flag) + 1] in {"baseline", "task_only"}
    assert treatment[treatment.index(method_flag) + 1] in {"biocs", "task_ssr"}
    kd_flag = "--lambda_kd_seg" if dataset == "cub200" else "--lambda-kd"
    assert control[control.index(kd_flag) + 1] == "0.0"
    assert treatment[treatment.index(kd_flag) + 1] == "0.0"
    assert treatment[treatment.index("--kernel-family") + 1] == spec.kernel


def test_seed_group_sharding_reuses_one_control_without_cross_node_races(tmp_path):
    jobs = pair_jobs(
        tmp_path,
        "screen",
        DEVELOPMENT_DATASETS,
        {dataset: SCREEN_SEEDS for dataset in DEVELOPMENT_DATASETS},
        CANDIDATES,
    )
    shards = [node_groups(jobs, rank, 4) for rank in range(4)]
    identities = [
        {job.pair_id for group in shard for job in group} for shard in shards
    ]

    assert len(jobs) == 96
    assert [len(shard) for shard in shards] == [2, 2, 1, 1]
    assert sorted(len(group) for shard in shards for group in shard) == [16] * 6
    assert len(set().union(*identities)) == len(jobs)
    for left_index, left in enumerate(identities):
        for right in identities[left_index + 1 :]:
            assert left.isdisjoint(right)
    for shard in shards:
        for group in shard:
            assert len({job.control_output for job in group}) == 1
            assert len({(job.phase, job.dataset, job.seed) for job in group}) == 1
    assert all("task_only" not in job.pair_id and "task_ssr" not in job.pair_id for job in jobs)


def test_sign_flip_test_is_reproducible_and_directional():
    first = sign_flip_test([0.5] * 12, label="fixed")
    second = sign_flip_test([0.5] * 12, label="fixed")

    assert first == second
    assert first["p_value"] < 0.01
    assert first["alternative"] == "favorable_difference_greater_than_zero"


def _environment(result_root: Path, rank: int) -> dict[str, str]:
    environment = dict(os.environ)
    for key in RANK_VARIABLES:
        environment.pop(key, None)
    environment.update(
        {
            "PROJECT_ROOT": str(PROJECT_ROOT),
            "PYTHON": sys.executable,
            "SSR_DIRECT_SEG_RUN_ID": "pytest_direct_ssr_prototype",
            "SSR_DIRECT_SEG_RESULT_ROOT": str(result_root),
            "SSR_DIRECT_SEG_DRY_RUN": "1",
            "EXPECTED_NNODES": "4",
            "GPU_COUNT": "8",
            "JOBS_PER_GPU": "2",
            "INIT_WAIT_SEC": "30",
            "FINAL_WAIT_SEC": "60",
            "HOSTNAME": "pytest-host",
            "GLOBAL_NODE_RANK": str(rank),
        }
    )
    return environment


def test_submit_wrapper_has_valid_bash_syntax():
    completed = subprocess.run(
        ["bash", "-n", str(WRAPPER)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_four_node_dry_run_validates_the_locked_budget(tmp_path):
    result_root = tmp_path / "four-node-plan"
    processes = [
        subprocess.Popen(
            ["bash", str(WRAPPER)],
            cwd=PROJECT_ROOT,
            env=_environment(result_root, rank),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for rank in range(4)
    ]
    outputs = [process.communicate(timeout=90)[0] for process in processes]

    assert [process.returncode for process in processes] == [0, 0, 0, 0], outputs
    validation = json.loads(
        (result_root / "MATRIX_PLAN_VALIDATION.json").read_text(encoding="utf-8")
    )
    assert validation["status"] == "PASS"
    assert validation["screen_pairs"] == 96
    assert validation["refine_pairs"] == 56
    assert validation["confirmation_pairs"] == 80
    assert validation["total_runs"] == 332
    assert validation["nodes"] == 4
