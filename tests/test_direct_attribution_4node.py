from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WRAPPER = PROJECT_ROOT / "scripts/submit_direct_attribution_multidata_4node_20260804.sh"
RANK_VARIABLES = (
    "GLOBAL_NODE_RANK",
    "NODE_RANK",
    "PADDLE_TRAINER_ID",
    "SLURM_NODEID",
    "GROUP_RANK",
)


def environment(result_root: Path, rank: int | None) -> dict[str, str]:
    output = dict(os.environ)
    for key in RANK_VARIABLES:
        output.pop(key, None)
    for key in ("PADDLE_TRAINERS_NUM", "NNODES", "SLURM_NNODES", "PET_NNODES"):
        output.pop(key, None)
    output.update(
        {
            "PROJECT_ROOT": str(PROJECT_ROOT),
            "PYTHON": sys.executable,
            "SSR_DIRECT_RUN_ID": "pytest_direct_four_node",
            "SSR_DIRECT_LAUNCH_TOKEN": "pytest_direct_four_node_token",
            "SSR_DIRECT_DRY_RUN": "1",
            "EXPECTED_NNODES": "4",
            "GPU_COUNT": "8",
            "DIRECT_RESULT_ROOT": str(result_root),
            "INIT_WAIT_SEC": "30",
            "FINAL_WAIT_SEC": "90",
            "HOSTNAME": "pytest-host",
        }
    )
    if rank is not None:
        output["GLOBAL_NODE_RANK"] = str(rank)
    return output


def test_expanded_four_node_dry_run_has_all_317_cells(tmp_path):
    result_root = tmp_path / "direct-four-node"
    processes = [
        subprocess.Popen(
            ["bash", str(WRAPPER)],
            cwd=PROJECT_ROOT,
            env=environment(result_root, rank),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for rank in range(4)
    ]
    outputs = [process.communicate(timeout=100)[0] for process in processes]

    assert [process.returncode for process in processes] == [0, 0, 0, 0], outputs
    report = json.loads(
        (result_root / "MATRIX_PLAN_VALIDATION.json").read_text(encoding="utf-8")
    )
    assert report["status"] == "PASS"
    assert report["total_jobs"] == 317
    assert report["jobs"] == {
        "ke_existing": 60,
        "ke_wikibio": 20,
        "vit_lora": 20,
        "matched_kd_cl": 80,
        "cub_segmentation": 61,
        "multidataset_segmentation": 76,
    }
    assert report["matched_kd_contract"]["planned_cells"] == 80
    assert (result_root / "launcher_status" / "final.done").is_file()


def test_expanded_wrapper_rejects_implicit_rank(tmp_path):
    completed = subprocess.run(
        ["bash", str(WRAPPER)],
        cwd=PROJECT_ROOT,
        env=environment(tmp_path / "missing-rank", None),
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "Set GLOBAL_NODE_RANK explicitly" in completed.stderr
