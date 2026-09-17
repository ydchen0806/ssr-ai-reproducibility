from __future__ import annotations

import json

import pytest

from ssr_utils.result_schema import sha256_file, sha256_value
from ssr_utils.teacher_trajectory import TEACHER_PROTOCOL, validate_teacher_trajectory


def _trajectory(tmp_path):
    root = tmp_path / "teacher"
    root.mkdir()
    checkpoints = []
    for task_id in range(2):
        path = root / f"task_{task_id:02d}.pt"
        path.write_bytes(f"checkpoint-{task_id}".encode())
        checkpoints.append(
            {"task_id": task_id, "file": path.name, "sha256": sha256_file(path)}
        )
    manifest = {
        "protocol": TEACHER_PROTOCOL,
        "seed": 7,
        "dataset": "toy",
        "model": "resnet18",
        "checkpoints": checkpoints,
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root, manifest


def test_teacher_trajectory_validates_files_and_combined_hash(tmp_path):
    root, manifest = _trajectory(tmp_path)
    report = validate_teacher_trajectory(
        root,
        dataset="toy",
        model="resnet18",
        seed=7,
        expected_checkpoints=2,
        expected_trajectory_hash=sha256_value(manifest),
    )
    assert report["sha256"] == sha256_value(manifest)


def test_teacher_trajectory_rejects_checkpoint_tampering(tmp_path):
    root, _ = _trajectory(tmp_path)
    (root / "task_01.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checkpoint hash mismatch"):
        validate_teacher_trajectory(
            root,
            dataset="toy",
            model="resnet18",
            seed=7,
            expected_checkpoints=2,
        )
