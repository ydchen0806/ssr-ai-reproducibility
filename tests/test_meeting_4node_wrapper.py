from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WRAPPER = PROJECT_ROOT / "scripts" / "submit_meeting_revision_4node.sh"
RANK_VARIABLES = (
    "GLOBAL_NODE_RANK",
    "NODE_RANK",
    "PADDLE_TRAINER_ID",
    "SLURM_NODEID",
    "GROUP_RANK",
)


def base_environment(result_root: Path) -> dict[str, str]:
    environment = dict(os.environ)
    for key in RANK_VARIABLES:
        environment.pop(key, None)
    for key in (
        "PADDLE_TRAINERS_NUM",
        "PADDLE_TRAINER_ENDPOINTS",
        "NNODES",
        "SLURM_NNODES",
        "PET_NNODES",
    ):
        environment.pop(key, None)
    environment.update(
        {
            "PROJECT_ROOT": str(PROJECT_ROOT),
            "PYTHON": sys.executable,
            "SSR_MEETING_RUN_ID": "pytest_meeting_four_node",
            "SSR_MEETING_LAUNCH_TOKEN": "pytest_meeting_four_node_token",
            "SSR_MEETING_DRY_RUN": "1",
            "EXPECTED_NNODES": "4",
            "GPU_COUNT": "8",
            "MEETING_RESULT_ROOT": str(result_root),
            "REUSE_RESULTS_ROOTS": str(result_root.parent / "no_reuse"),
            "CLAIM_WAIT_SEC": "30",
            "INIT_WAIT_SEC": "30",
            "FINAL_WAIT_SEC": "60",
            "HOSTNAME": "pytest-host",
        }
    )
    return environment


def test_four_node_dry_run_builds_all_planned_matrices(tmp_path):
    result_root = tmp_path / "meeting"
    processes = []
    for rank in range(4):
        environment = base_environment(result_root)
        environment["GLOBAL_NODE_RANK"] = str(rank)
        processes.append(
            subprocess.Popen(
                ["bash", str(WRAPPER)],
                cwd=PROJECT_ROOT,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        )

    outputs = []
    for process in processes:
        output, _ = process.communicate(timeout=90)
        outputs.append(output)

    assert [process.returncode for process in processes] == [0, 0, 0, 0], outputs
    report = json.loads((result_root / "DRY_RUN_VALIDATION.json").read_text())
    assert report == {
        "status": "PASS",
        "editing_cells": 240,
        "editing_unique_cells": 240,
        "matched_kd_jobs": 50,
        "matched_kd_unique_jobs": 50,
        "segmentation_development_jobs": 21,
        "segmentation_confirmation_jobs": 40,
    }
    assert (result_root / "launcher_status" / "final.done").is_file()
    assert (result_root / "editing" / "planned_runs.tsv").is_file()
    assert (result_root / "matched_kd_cl" / "planned_runs.tsv").is_file()
    assert (result_root / "cub_segmentation" / "MANIFEST.json").is_file()


def test_wrapper_rejects_missing_rank_instead_of_defaulting_to_zero(tmp_path):
    result = subprocess.run(
        ["bash", str(WRAPPER)],
        cwd=PROJECT_ROOT,
        env=base_environment(tmp_path / "missing-rank"),
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "Cannot resolve GLOBAL_NODE_RANK" in result.stderr


def test_wrapper_rejects_conflicting_rank_variables(tmp_path):
    environment = base_environment(tmp_path / "conflicting-rank")
    environment["GLOBAL_NODE_RANK"] = "1"
    environment["PADDLE_TRAINER_ID"] = "2"

    result = subprocess.run(
        ["bash", str(WRAPPER)],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "Conflicting node-rank variables" in result.stderr
