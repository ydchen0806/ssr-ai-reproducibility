from __future__ import annotations

import json

from scripts.check_matched_kd_fairness import EXPECTED, REGULARIZERS, check


def test_fairness_checker_accepts_complete_same_scaffold(tmp_path):
    for recipe in EXPECTED:
        path = tmp_path / recipe / "result_record.json"
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(
                {
                    "task_family": "classification",
                    "dataset": "split_cifar100",
                    "model": "resnet18",
                    "seed": 1,
                    "recipe": recipe,
                    "objective": {"kd": True},
                    "dataset_hash": "data",
                    "pairing_hash": "same",
                    "teacher_protocol": "locked_kd_only_trajectory_v1",
                    "teacher_trajectory_hash": "a" * 64,
                    "initial_model_hash": "b" * 64,
                    "optimizer_steps": 100,
                    "training_batches": 100,
                    "kd_weight": 5.0,
                    "kd_temperature": 2.0,
                    "active_regularizer": REGULARIZERS[recipe],
                    "active_regularizer_weight": 0.0 if recipe == "kd" else 1.0,
                    "auxiliary_trainable_parameter_count": (
                        32 if recipe == "kd_center" else 0
                    ),
                }
            ),
            encoding="utf-8",
        )
    assert check(tmp_path)["status"] == "PASS"


def test_fairness_checker_rejects_missing_method(tmp_path):
    path = tmp_path / "kd" / "result_record.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "task_family": "classification",
                "dataset": "split_cifar100",
                "model": "resnet18",
                "seed": 1,
                "recipe": "kd",
                "objective": {"kd": True},
                "dataset_hash": "data",
                "pairing_hash": "same",
                "teacher_protocol": "locked_kd_only_trajectory_v1",
                "teacher_trajectory_hash": "a" * 64,
                "initial_model_hash": "b" * 64,
                "optimizer_steps": 100,
                "training_batches": 100,
                "kd_weight": 5.0,
                "kd_temperature": 2.0,
                "active_regularizer": "none",
                "active_regularizer_weight": 0.0,
                "auxiliary_trainable_parameter_count": 0,
            }
        ),
        encoding="utf-8",
    )
    assert check(tmp_path)["status"] == "FAIL"


def test_fairness_checker_rejects_an_entirely_missing_expected_seed(tmp_path):
    report = check(
        tmp_path,
        expected_datasets={"split_cifar100"},
        expected_seeds={3101},
    )

    assert report["status"] == "FAIL"
    assert report["errors"] == [
        {
            "identity": ["split_cifar100", "resnet18", 3101],
            "error": "missing_expected_group",
        }
    ]


def test_fairness_checker_rejects_shared_amp_skips(tmp_path):
    for recipe in EXPECTED:
        path = tmp_path / recipe / "result_record.json"
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(
                {
                    "task_family": "classification",
                    "dataset": "split_cifar100",
                    "model": "resnet18",
                    "seed": 1,
                    "recipe": recipe,
                    "objective": {"kd": True},
                    "dataset_hash": "data",
                    "pairing_hash": "same",
                    "teacher_protocol": "locked_kd_only_trajectory_v1",
                    "teacher_trajectory_hash": "a" * 64,
                    "initial_model_hash": "b" * 64,
                    "optimizer_steps": 99,
                    "training_batches": 100,
                    "kd_weight": 5.0,
                    "kd_temperature": 2.0,
                    "active_regularizer": REGULARIZERS[recipe],
                    "active_regularizer_weight": 0.0 if recipe == "kd" else 1.0,
                    "auxiliary_trainable_parameter_count": (
                        32 if recipe == "kd_center" else 0
                    ),
                }
            ),
            encoding="utf-8",
        )

    report = check(tmp_path)

    assert report["status"] == "FAIL"
    assert any(
        error["error"] == "skipped_optimizer_updates"
        for error in report["errors"]
    )
