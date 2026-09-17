from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WRAPPER = PROJECT_ROOT / "scripts" / "submit_ssr_segmentation_factorial_4node_20260808.sh"
RANK_VARIABLES = (
    "GLOBAL_NODE_RANK",
    "NODE_RANK",
    "PADDLE_TRAINER_ID",
    "SLURM_NODEID",
    "GROUP_RANK",
)


def _environment(result_root: Path, rank: int) -> dict[str, str]:
    environment = dict(os.environ)
    for key in RANK_VARIABLES:
        environment.pop(key, None)
    environment.update(
        {
            "PROJECT_ROOT": str(PROJECT_ROOT),
            "PYTHON": sys.executable,
            "SSR_SEG_RUN_ID": "pytest_factorial_four_node",
            "SSR_SEG_RESULT_ROOT": str(result_root),
            "SSR_SEG_DRY_RUN": "1",
            "EXPECTED_NNODES": "4",
            "GPU_COUNT": "8",
            "JOBS_PER_GPU": "2",
            "SEG_BATCH_SIZE": "24",
            "INIT_WAIT_SEC": "30",
            "LOCK_WAIT_SEC": "30",
            "FINISH_WAIT_SEC": "60",
            "HOSTNAME": "pytest-host",
            "GLOBAL_NODE_RANK": str(rank),
        }
    )
    return environment


def test_factorial_submit_wrapper_has_valid_bash_syntax():
    completed = subprocess.run(
        ["bash", "-n", str(WRAPPER)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_factorial_four_node_dry_run_emits_the_42_by_80_manifest(tmp_path):
    result_root = tmp_path / "factorial-four-node"
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
    assert validation == {
        "status": "PASS",
        "protocol": "factorial_2x2",
        "development": 42,
        "confirmation": 80,
        "total": 122,
        "nodes": 4,
    }
    manifest = json.loads(
        (result_root / "cub200_masks" / "MANIFEST.json").read_text(encoding="utf-8")
    )
    assert manifest["jobs"] == {"development": 42, "confirmation": 80, "total": 122}
    identity = (result_root / ".segmentation_factorial_4node_identity").read_text(
        encoding="utf-8"
    )
    assert "jobs_per_gpu=2\n" in identity
    assert "seg_batch_size=24\n" in identity
