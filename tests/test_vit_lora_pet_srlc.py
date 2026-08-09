from argparse import Namespace

from scripts import run_vit_lora_pet_srlc as pet


def test_pet_candidate_matrix_covers_targets_ranks_and_task_ramps():
    assert len(pet.CANDIDATES) == 36
    assert {candidate.rank for candidate in pet.CANDIDATES} == {8, 16, 32}
    assert {
        rank: sum(candidate.rank == rank for candidate in pet.CANDIDATES)
        for rank in (8, 16, 32)
    } == {8: 12, 16: 12, 32: 12}
    assert {
        target: sum(pet.candidate_target(candidate) == target for candidate in pet.CANDIDATES)
        for target in ("classifier", "adapter", "joint")
    } == {"classifier": 9, "adapter": 9, "joint": 18}
    assert sum(candidate.ramp_tasks > 0 for candidate in pet.CANDIDATES) == 6
    assert any(
        candidate.rank == 8
        and candidate.lambda_classifier == 0.01
        and candidate.lambda_adapter == 0.002
        and candidate.sigma_exc == 0.20
        and candidate.sigma_inh == 0.50
        and candidate.ramp_tasks == 0
        for candidate in pet.CANDIDATES
    )


def test_pet_seed_partitions_and_job_counts_are_locked():
    assert len(pet.SCREEN_SEEDS) == 3
    assert len(pet.REFINE_SEEDS) == 7
    assert len(pet.CONFIRMATION_SEEDS) == 30
    assert not set(pet.SCREEN_SEEDS) & set(pet.REFINE_SEEDS)
    assert not set(pet.SCREEN_SEEDS) & set(pet.CONFIRMATION_SEEDS)
    assert not set(pet.REFINE_SEEDS) & set(pet.CONFIRMATION_SEEDS)
    assert pet.job_counts(6) == {
        "candidate_count": 36,
        "screen_controls": 9,
        "screen_treatments": 108,
        "screen_jobs": 117,
        "maximum_refine_controls": 21,
        "refine_treatments": 42,
        "maximum_refine_jobs": 63,
        "confirmation_controls": 30,
        "confirmation_treatments": 30,
        "confirmation_jobs": 60,
        "maximum_total_jobs": 240,
    }


def test_pet_pair_commands_share_rank_and_training_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(pet.driver, "CONFIG", pet.CONFIG)
    candidate = next(
        candidate
        for candidate in pet.CANDIDATES
        if candidate.candidate_id == "r16_joint_c0.01_a0.002_start1_ramp2"
    )
    control = pet.Job("screen", 8101, 8, candidate.rank, None)
    treatment = pet.Job("screen", 8101, 8, candidate.rank, candidate)
    args = Namespace(
        data_root=tmp_path / "data",
        pretrained_checkpoint=tmp_path / "model.safetensors",
        workers=2,
        python="python",
        result_root=tmp_path / "results",
    )

    def overrides(job):
        command = pet.driver.command(job, args)
        start = command.index("--overrides") + 1
        return dict(item.split("=", 1) for item in command[start:])

    control_overrides = overrides(control)
    treatment_overrides = overrides(treatment)
    assert control_overrides["training.epochs"] == treatment_overrides["training.epochs"] == "8"
    assert control_overrides["method.lora_rank"] == treatment_overrides["method.lora_rank"] == "16"
    assert control_overrides["method.lora_alpha"] == treatment_overrides["method.lora_alpha"] == "32.0"
    assert control_overrides["dataset.data_root"] == treatment_overrides["dataset.data_root"]
    assert control_overrides["model.pretrained_checkpoint"] == treatment_overrides["model.pretrained_checkpoint"]
    assert control_overrides["method.recipe"] == "plain"
    assert treatment_overrides["method.recipe"] == "ssr_only"
    assert control_overrides["method.lambda_spatial"] == "0.0"
    assert control_overrides["method.lambda_adapter"] == "0.0"
    assert treatment_overrides["method.lambda_spatial"] == "0.01"
    assert treatment_overrides["method.lambda_adapter"] == "0.002"
    assert treatment_overrides["method.ssr_start_task"] == "1"
    assert treatment_overrides["method.ssr_ramp_tasks"] == "2"


def test_pet_selection_uses_endpoints_not_geometry():
    strong_endpoint = {
        "candidate": {"candidate_id": "a"},
        "means": {
            "avg_accuracy_delta": 0.3,
            "avg_forgetting_reduction": 0.2,
            "effective_rank_delta": -10.0,
            "prototype_overlap_reduction": -10.0,
        },
        "endpoint_gate": True,
        "geometry_gate": False,
        "dual_wins": 6,
        "geometry_wins": 0,
        "balanced_endpoint_score": 0.2,
    }
    weak_endpoint = {
        "candidate": {"candidate_id": "b"},
        "means": {
            "avg_accuracy_delta": 0.1,
            "avg_forgetting_reduction": 0.1,
            "effective_rank_delta": 10.0,
            "prototype_overlap_reduction": 10.0,
        },
        "endpoint_gate": True,
        "geometry_gate": True,
        "dual_wins": 5,
        "geometry_wins": 7,
        "balanced_endpoint_score": 0.1,
    }
    assert pet.rank_for_selection([weak_endpoint, strong_endpoint])[0] is strong_endpoint


def test_pet_selection_rejects_gain_from_a_weakened_task_only_rank():
    weak_baseline_large_gain = {
        "candidate": {"candidate_id": "weak"},
        "means": {
            "avg_accuracy_delta": 2.0,
            "avg_forgetting_reduction": 2.0,
        },
        "endpoint_gate": True,
        "geometry_gate": True,
        "dual_wins": 7,
        "geometry_wins": 7,
        "balanced_endpoint_score": 2.0,
        "rows": [{"avg_accuracy_control": 70.0}],
    }
    adequate_baseline = {
        "candidate": {"candidate_id": "adequate"},
        "means": {
            "avg_accuracy_delta": 0.2,
            "avg_forgetting_reduction": 0.2,
        },
        "endpoint_gate": True,
        "geometry_gate": True,
        "dual_wins": 4,
        "geometry_wins": 4,
        "balanced_endpoint_score": 0.2,
        "rows": [{"avg_accuracy_control": 75.0}],
    }

    ranked = pet.rank_for_selection([weak_baseline_large_gain, adequate_baseline])

    assert ranked[0] is adequate_baseline
    assert adequate_baseline["baseline_adequacy_gate"] is True
    assert weak_baseline_large_gain["baseline_adequacy_gate"] is False


def test_pet_protocol_manifest_records_auditable_boundaries():
    manifest = pet.protocol_manifest("a" * 40, 6)
    assert manifest["protocol"] == pet.PROTOCOL_NAME
    assert manifest["seed_partitions_are_disjoint"] is True
    assert manifest["confirmation_primary_endpoint"] == "avg_accuracy"
    assert manifest["confirmation_key_secondary"] == "avg_forgetting"
    assert "Geometry is an audit outcome" in manifest["selection_rule"]
    assert manifest["config_sha256"]
    assert manifest["runner_sha256"]
    assert manifest["execution_driver_sha256"]
    assert len(manifest["candidate_inventory"]) == 36
    assert all("target" in candidate for candidate in manifest["candidate_inventory"])
