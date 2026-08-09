from __future__ import annotations

import copy
import json
from argparse import Namespace
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import yaml

from experiments.multidataset_adapter_ssr import (
    load_feature_cache,
    make_tasks,
    paired_record,
    run_one,
    schedule_sha256,
    sha256_file,
)
from methods.biocs_lora import BioCsLoRA, LoRAInjectedLinear
from main import objective_for_method
from scripts.summarize_multidataset_adapter_ssr import select_global_scale, summarize
from scripts.validate_vit_lora_pairs import CONTROL_OBJECTIVE, TREATMENT_OBJECTIVE
from scripts.validate_vit_lora_pairs import summarize as summarize_vit_lora
from scripts.validate_vit_lora_pairs import validate_pair
from ssr_utils.result_schema import OBJECTIVE_COMPONENTS


class TinyAttentionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(4, 4)
        self.proj = nn.Linear(4, 4)
        self.head = nn.Linear(4, 3)

    def forward(self, x):
        return self.head(torch.tanh(self.proj(torch.tanh(self.qkv(x)))))


class TinyViTLikeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(4, 4)
        self.mlp = nn.Module()
        self.mlp.fc1 = nn.Linear(4, 8)
        self.mlp.fc2 = nn.Linear(8, 4)
        self.head = nn.Linear(4, 3)

    def get_classifier(self):
        return self.head

    def forward(self, x):
        return self.head(self.mlp.fc2(torch.tanh(self.mlp.fc1(self.qkv(x)))))


def test_biocs_lora_is_registered_and_optimized():
    learner = BioCsLoRA(
        TinyAttentionModel(),
        torch.device("cpu"),
        {
            "lora_targets": ["qkv", "proj"],
            "lora_rank": 2,
            "lambda_spatial": 0.01,
            "lambda_adapter": 0.01,
        },
    )
    wrappers = [
        module for module in learner.model.modules() if isinstance(module, LoRAInjectedLinear)
    ]
    assert len(wrappers) == 2
    named = dict(learner.model.named_parameters())
    lora = {name: parameter for name, parameter in named.items() if ".lora." in name}
    assert lora
    assert all(parameter.requires_grad for parameter in lora.values())
    assert all(not wrapper.base.weight.requires_grad for wrapper in wrappers)

    optimizer = learner._build_optimizer({"optimizer": "adamw", "lr": 0.05})
    optimized = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    assert all(id(parameter) in optimized for parameter in lora.values())

    before = wrappers[0].lora.lora_B.detach().clone()
    loss, _ = learner.training_step(torch.randn(8, 4), torch.arange(8) % 3, 0)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    assert not torch.equal(before, wrappers[0].lora.lora_B)


def test_biocs_lora_rejects_empty_target_set():
    with pytest.raises(ValueError, match="No LoRA target modules matched"):
        BioCsLoRA(
            TinyAttentionModel(),
            torch.device("cpu"),
            {"lora_targets": ["does_not_exist"]},
        )


def test_biocs_lora_only_unfreezes_exact_classifier_and_lora_parameters():
    learner = BioCsLoRA(
        TinyViTLikeModel(),
        torch.device("cpu"),
        {"lora_targets": ["qkv"], "lora_rank": 2},
    )
    trainable = {
        name for name, parameter in learner.model.named_parameters() if parameter.requires_grad
    }
    assert "head.weight" in trainable
    assert "head.bias" in trainable
    assert not any(name.startswith("mlp.fc") for name in trainable)
    assert any(".lora." in name for name in trainable)
    assert learner._get_classifier() is learner.model.head


def synthetic_cache(path: Path) -> dict:
    generator = torch.Generator().manual_seed(19)
    centers = torch.randn(6, 8, generator=generator)

    def split(samples_per_class: int):
        labels = torch.arange(6).repeat_interleave(samples_per_class)
        noise = 0.2 * torch.randn(labels.numel(), 8, generator=generator)
        return centers[labels] + noise, labels

    x_train, y_train = split(10)
    x_test, y_test = split(4)
    payload = {
        "schema_version": 1,
        "dataset": "cifar100",
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,
    }
    torch.save(payload, path)
    return payload


def adapter_args(tmp_path: Path) -> Namespace:
    return Namespace(
        dataset="cifar100",
        seed=23,
        num_tasks=3,
        epochs=2,
        batch_size=12,
        rank=3,
        tau=8.0,
        adapter_scale=0.5,
        lr=0.01,
        weight_decay=0.0,
        ssr_scale=0.1,
        lambda_cls=0.2,
        lambda_adapter=0.1,
        a_exc=1.0,
        a_inh=0.8,
        sigma_exc=0.2,
        sigma_inh=0.5,
        kernel_family="gaussian",
        distance_metric="cosine",
        grad_clip=5.0,
    )


def test_matched_adapter_pair_has_identical_budget_and_initialization(tmp_path):
    cache = tmp_path / "features.pt"
    synthetic_cache(cache)
    payload = load_feature_cache(cache, "cifar100")
    args = adapter_args(tmp_path)
    digest = sha256_file(cache)
    control = run_one(args, payload, "task", digest, torch.device("cpu"))
    treated = run_one(args, payload, "task_ssr", digest, torch.device("cpu"))
    record = paired_record(control, treated)

    assert record["matched_budget"] is True
    assert control.initial_state_sha256 == treated.initial_state_sha256
    assert control.schedule_sha256 == treated.schedule_sha256
    assert control.optimizer_steps == treated.optimizer_steps
    assert set(record["delta"]) == {
        "avg_accuracy",
        "avg_forgetting_reduction",
        "cil_last_accuracy",
        "effective_rank",
        "prototype_overlap_reduction",
        "adapter_basis_overlap_reduction",
    }


def test_task_schedule_is_deterministic_and_seed_specific():
    tasks = make_tasks(23, 5, 7)
    assert tasks == make_tasks(23, 5, 7)
    assert sorted(item for task in tasks for item in task) == list(range(23))
    assert max(map(len, tasks)) - min(map(len, tasks)) <= 1
    assert schedule_sha256(tasks, 7, 2) != schedule_sha256(tasks, 8, 2)


def test_global_lock_prefers_cross_dataset_dual_positive_scale():
    rows = []
    for dataset in ("flowers102", "dtd"):
        for seed in (1, 2, 3):
            for scale, aa, af in ((0.1, 0.3, 0.2), (1.0, 0.8, -0.1)):
                rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "rank": 16,
                        "ssr_scale": scale,
                        "avg_accuracy": aa,
                        "avg_forgetting_reduction": af,
                        "cil_last_accuracy": 0.0,
                        "effective_rank": 0.0,
                        "prototype_overlap_reduction": 0.0,
                        "adapter_basis_overlap_reduction": 0.0,
                    }
                )
    lock = select_global_scale(summarize(rows))
    assert lock["selected"]["ssr_scale"] == pytest.approx(0.1)
    assert lock["selected"]["dataset_dual_positive"] == 2


def test_biocs_lora_objective_is_task_only_exactly_when_ssr_weights_are_zero():
    assert objective_for_method(
        "biocs_lora", {"lambda_spatial": 0.0, "lambda_adapter": 0.0}
    ) == {"task": True}
    assert objective_for_method(
        "biocs_lora", {"lambda_spatial": 0.01, "lambda_adapter": 0.002}
    ) == {"task": True, "ssr": True}
    assert objective_for_method(
        "biocs_lora",
        {
            "lambda_spatial": 0.01,
            "lambda_adapter": 0.0,
            "lambda_spectral": 0.1,
        },
    ) == {"task": True, "ssr": True, "spectral": True}


def test_biocs_lora_task_only_skips_inactive_spatial_penalty(monkeypatch):
    learner = BioCsLoRA(
        TinyAttentionModel(),
        torch.device("cpu"),
        {
            "lora_targets": ["qkv", "proj"],
            "lora_rank": 2,
            "lambda_spatial": 0.0,
            "lambda_adapter": 0.0,
        },
    )

    def unexpected_penalty(*_args, **_kwargs):
        raise AssertionError("inactive SSR penalty was evaluated")

    monkeypatch.setattr("methods.biocs_lora.compute_spatial_biocs", unexpected_penalty)
    loss, _ = learner.training_step(torch.randn(4, 4), torch.arange(4) % 3, 0)
    assert torch.isfinite(loss)
    assert learner.regularizer == "none"
    assert learner.active_regularizer_weight() == 0.0


def test_biocs_lora_delays_and_ramps_ssr_by_task():
    learner = BioCsLoRA(
        TinyAttentionModel(),
        torch.device("cpu"),
        {
            "lora_targets": ["qkv", "proj"],
            "lora_rank": 2,
            "lambda_spatial": 0.0,
            "lambda_adapter": 0.01,
            "ssr_start_task": 1,
            "ssr_ramp_tasks": 3,
        },
    )

    assert learner._ssr_scale(0) == 0.0
    assert learner._ssr_scale(1) == pytest.approx(1 / 3)
    assert learner._ssr_scale(2) == pytest.approx(2 / 3)
    assert learner._ssr_scale(3) == 1.0
    assert learner._ssr_scale(8) == 1.0


def test_biocs_lora_rejects_negative_ssr_schedule():
    with pytest.raises(ValueError, match="must be non-negative"):
        BioCsLoRA(
            TinyAttentionModel(),
            torch.device("cpu"),
            {
                "lora_targets": ["qkv"],
                "ssr_start_task": -1,
            },
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA autocast regression")
def test_spatial_biocs_has_finite_gradients_inside_cuda_autocast():
    from methods.bioreg import compute_spatial_biocs

    weights = torch.randn(11, 32, device="cuda", requires_grad=True)
    with torch.amp.autocast("cuda"):
        penalty = compute_spatial_biocs(weights)
    penalty.backward()
    assert torch.isfinite(penalty)
    assert torch.isfinite(weights.grad).all()


def test_vit_lora_pair_validator_enforces_initialization_and_budget(tmp_path):
    normalized_task = {
        "task": True,
        "kd": False,
        "ssr": False,
        "anchor": False,
        "spectral": False,
        "ewc": False,
        "mas": False,
        "si": False,
    }
    normalized_ssr = dict(normalized_task, ssr=True)
    common_record = {
        "dataset": "flowers102_cl",
        "model": "vit_tiny_patch16_224",
        "seed": 5201,
        "dataset_hash": "a" * 64,
        "pairing_hash": "b" * 64,
        "initial_model_hash": "c" * 64,
        "final_model_hash": "d" * 64,
        "optimizer_steps": 123,
        "training_batches": 123,
        "active_regularizer": "none",
        "active_regularizer_weight": 0.0,
        "metrics": {
            "avg_accuracy": 40.0,
            "avg_forgetting": 12.0,
            "cil_last_accuracy": 20.0,
            "effective_rank": 7.0,
            "prototype_overlap": 0.2,
        },
    }
    common_config = {
        "model": {"name": "vit_tiny_patch16_224", "pretrained": True},
        "dataset": {"name": "flowers102_cl"},
        "training": {"epochs": 2, "batch_size": 4},
        "method": {
            "name": "biocs_lora",
            "lora_targets": ["qkv", "proj"],
            "lora_rank": 8,
            "lora_alpha": 16.0,
            "lambda_spatial": 0.0,
            "lambda_adapter": 0.0,
            "lambda_spectral": 0.0,
            "recipe": "plain",
        },
    }
    control = tmp_path / "task"
    treatment = tmp_path / "task_ssr"
    control.mkdir()
    treatment.mkdir()
    (control / "result_record.json").write_text(
        json.dumps(
            dict(common_record, objective=normalized_task, recipe="plain", distance_mapping="none")
        ),
        encoding="utf-8",
    )
    (control / "config.yaml").write_text(
        yaml.safe_dump(common_config), encoding="utf-8"
    )
    treated_record = dict(common_record)
    treated_record["final_model_hash"] = "e" * 64
    treated_record["metrics"] = dict(common_record["metrics"], avg_accuracy=41.0)
    treated_record.update(
        objective=normalized_ssr,
        recipe="ssr_only",
        distance_mapping="cosine",
        active_regularizer="ssr",
        active_regularizer_weight=0.002,
    )
    treated_config = copy.deepcopy(common_config)
    treated_config["method"].update(
        recipe="ssr_only", lambda_spatial=0.0, lambda_adapter=0.002
    )
    (treatment / "result_record.json").write_text(
        json.dumps(treated_record), encoding="utf-8"
    )
    (treatment / "config.yaml").write_text(
        yaml.safe_dump(treated_config), encoding="utf-8"
    )

    row = validate_pair(control, treatment)
    assert row["avg_accuracy_delta"] == pytest.approx(1.0)
    treated_record["final_model_hash"] = common_record["final_model_hash"]
    (treatment / "result_record.json").write_text(
        json.dumps(treated_record), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="same final model"):
        validate_pair(control, treatment)

    treated_record["final_model_hash"] = "e" * 64
    treated_record["optimizer_steps"] = 124
    (treatment / "result_record.json").write_text(
        json.dumps(treated_record), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="optimizer_steps differs"):
        validate_pair(control, treatment)

    treated_record["optimizer_steps"] = 123
    (treatment / "result_record.json").write_text(
        json.dumps(treated_record), encoding="utf-8"
    )
    treated_config["method"]["lambda_spectral"] = 0.01
    (treatment / "config.yaml").write_text(
        yaml.safe_dump(treated_config), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="lambda_spectral=0"):
        validate_pair(control, treatment)

    treated_config["method"]["lambda_spectral"] = 0.0
    treated_config["method"]["hidden_regularizer"] = 1.0
    (treatment / "config.yaml").write_text(
        yaml.safe_dump(treated_config), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="outside the SSR treatment allowlist"):
        validate_pair(control, treatment)

    treated_config["method"].pop("hidden_regularizer")
    (treatment / "config.yaml").write_text(
        yaml.safe_dump(treated_config), encoding="utf-8"
    )
    for forbidden_component in ("kd", "anchor", "spectral", "center", "protodecor"):
        treated_record["objective"] = dict(
            normalized_ssr, **{forbidden_component: True}
        )
        (treatment / "result_record.json").write_text(
            json.dumps(treated_record), encoding="utf-8"
        )
        with pytest.raises(ValueError, match=r"task-loss\+SSR only"):
            validate_pair(control, treatment)

    treated_record["objective"] = normalized_ssr
    (treatment / "result_record.json").write_text(
        json.dumps(treated_record), encoding="utf-8"
    )
    treated_config["method"]["lambda_kd"] = 0.5
    (treatment / "config.yaml").write_text(
        yaml.safe_dump(treated_config), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="non-SSR regularization coefficients"):
        validate_pair(control, treatment)


def test_vit_lora_objectives_track_the_canonical_component_inventory():
    assert set(CONTROL_OBJECTIVE) == set(OBJECTIVE_COMPONENTS)
    assert set(TREATMENT_OBJECTIVE) == set(OBJECTIVE_COMPONENTS)
    assert {name for name, enabled in CONTROL_OBJECTIVE.items() if enabled} == {"task"}
    assert {name for name, enabled in TREATMENT_OBJECTIVE.items() if enabled} == {
        "task",
        "ssr",
    }


def test_vit_lora_summary_reports_raw_arms_ci_and_favorable_pairs():
    rows = []
    for seed, control_accuracy, treatment_accuracy in (
        (1, 40.0, 41.0),
        (2, 42.0, 43.5),
        (3, 44.0, 43.5),
    ):
        row = {"dataset": "flowers102_cl", "seed": seed}
        for metric in (
            "avg_accuracy",
            "cil_last_accuracy",
            "effective_rank",
        ):
            row[f"{metric}_control"] = control_accuracy
            row[f"{metric}_treatment"] = treatment_accuracy
            row[f"{metric}_delta"] = treatment_accuracy - control_accuracy
        for metric in ("avg_forgetting", "prototype_overlap"):
            row[f"{metric}_control"] = 2.0
            row[f"{metric}_treatment"] = 1.0
            row[f"{metric}_reduction"] = 1.0
        rows.append(row)

    summary = summarize_vit_lora(rows)["flowers102_cl"]
    accuracy = summary["metrics"]["avg_accuracy"]
    assert accuracy["n"] == 3
    assert accuracy["control_mean"] == pytest.approx(42.0)
    assert accuracy["treatment_mean"] == pytest.approx(128.0 / 3.0)
    assert accuracy["difference_mean"] == pytest.approx(2.0 / 3.0)
    assert accuracy["ci95_low"] < accuracy["difference_mean"] < accuracy["ci95_high"]
    assert accuracy["favorable_pairs"] == 2
    forgetting = summary["metrics"]["avg_forgetting"]
    assert forgetting["higher_is_better"] is False
    assert forgetting["difference_mean"] == pytest.approx(1.0)
    assert forgetting["favorable_pairs"] == 3


def test_vit_lora_asset_fetch_uses_live_oxford_archive_urls():
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "fetch_vit_lora_assets.sh"
    ).read_text(encoding="utf-8")
    assert 'PETS_ARCHIVE_URL="https://thor.robots.ox.ac.uk/pets"' in script
    assert "/pets/data/images.tar.gz" not in script
    assert "/pets/data/annotations.tar.gz" not in script
