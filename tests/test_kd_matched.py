from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from methods.builder import build_method
from methods.kd_matched import KDMatched


def tiny_model():
    return nn.Sequential(nn.Flatten(), nn.Linear(4, 3))


def test_registry_selects_one_common_kd_implementation():
    for method_name, regularizer in {
        "kd": "none",
        "kd_ewc": "ewc",
        "kd_mas": "mas",
        "kd_si": "si",
        "kd_ssr": "ssr",
    }.items():
        method = build_method(
            {"name": method_name, "lambda_kd": 3.0},
            model=tiny_model(),
            device=torch.device("cpu"),
        )
        assert isinstance(method, KDMatched)
        assert method.regularizer == regularizer
        assert method.lambda_kd == 3.0


def test_matched_kd_rejects_method_regularizer_mismatch():
    with pytest.raises(ValueError, match="Contradictory matched-KD config"):
        KDMatched(
            tiny_model(),
            torch.device("cpu"),
            {"name": "kd_ssr", "regularizer": "none"},
        )


def test_kd_scaffold_changes_only_after_teacher_exists():
    model = tiny_model()
    method = KDMatched(model, torch.device("cpu"), {"name": "kd", "lambda_kd": 2.0})
    x = torch.randn(2, 1, 2, 2)
    y = torch.tensor([0, 1])
    initial, _ = method.training_step(x, y, 0)
    method.teacher = copy_model = copy_module(model)
    with torch.no_grad():
        copy_model[-1].weight[0].add_(0.5)
    with_teacher, _ = method.training_step(x, y, 1)
    assert with_teacher > initial


def test_method_name_rejects_a_contradictory_explicit_regularizer():
    with pytest.raises(ValueError, match="Contradictory matched-KD config"):
        KDMatched(
            tiny_model(),
            torch.device("cpu"),
            {"name": "kd_ssr", "regularizer": "mas"},
        )


def test_locked_teacher_is_shared_from_kd_only_trajectory(tmp_path):
    teacher_root = tmp_path / "teachers"
    common = {
        "teacher_mode": "locked_kd_trajectory",
        "teacher_checkpoint_root": str(teacher_root),
        "teacher_dataset": "split_cifar100",
        "teacher_model": "resnet18",
    }
    kd = KDMatched(tiny_model(), torch.device("cpu"), {"name": "kd", **common})
    kd._run_seed = 7
    with torch.no_grad():
        kd.model[-1].weight.fill_(0.25)
    kd._save_locked_teacher(0)

    treatment = KDMatched(
        tiny_model(), torch.device("cpu"), {"name": "kd_ssr", **common}
    )
    treatment._run_seed = 7
    with torch.no_grad():
        treatment.model[-1].weight.fill_(9.0)
    treatment._load_locked_teacher(0)

    assert treatment.teacher is not None
    assert torch.allclose(treatment.teacher[-1].weight, kd.model[-1].weight)
    assert treatment.teacher_trajectory_identity() == kd.teacher_trajectory_identity()


def test_locked_teacher_rejects_checkpoint_tampering(tmp_path):
    config = {
        "name": "kd",
        "teacher_mode": "locked_kd_trajectory",
        "teacher_checkpoint_root": str(tmp_path),
        "teacher_dataset": "split_cifar100",
        "teacher_model": "resnet18",
    }
    kd = KDMatched(tiny_model(), torch.device("cpu"), config)
    kd._run_seed = 3
    kd._save_locked_teacher(0)
    with (tmp_path / "task_00.pt").open("ab") as handle:
        handle.write(b"tampered")

    with pytest.raises(RuntimeError, match="checkpoint hash mismatch"):
        kd._load_locked_teacher(0)


def copy_module(module: nn.Module) -> nn.Module:
    import copy

    cloned = copy.deepcopy(module)
    cloned.eval()
    for parameter in cloned.parameters():
        parameter.requires_grad_(False)
    return cloned
