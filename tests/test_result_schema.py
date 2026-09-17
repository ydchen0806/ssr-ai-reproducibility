from __future__ import annotations

import pytest

from ssr_utils.result_schema import ResultSchemaError, build_result_record, validate_result_record


def example_record(**overrides):
    values = {
        "git_commit": "abc123",
        "run_id": "smoke",
        "task_family": "editing",
        "dataset": "zsre",
        "model": "gpt2-xl",
        "seed": 1,
        "objective": {"task": True, "ssr": True},
        "distance_mapping": "projective",
        "kernel": {
            "family": "gaussian_dog",
            "A_exc": 1.0,
            "A_inh": 0.8,
            "sigma_exc": 0.16,
            "sigma_inh": 0.45,
        },
        "metrics": {"efficacy": 100.0, "locality": 42.0},
        "runtime": {"elapsed_s": 1.0},
        "config": {"recipe": "ssr_only"},
        "dataset_hash": "def456",
        "recipe": "ssr_only",
        "requested": 10,
        "attempted": 10,
        "succeeded": 10,
        "failed": 0,
        "status": "complete",
        "evaluator": "custom_substring_any_ground_truth_all_items",
        "evaluator_version": "1.0",
        "evaluation_protocol_hash": "a" * 64,
        "model_hash": "b" * 64,
        "pairing_protocol_hash": "c" * 64,
    }
    values.update(overrides)
    return build_result_record(**values)


def test_build_result_record_normalizes_objective_and_hashes_config():
    record = example_record()
    assert record["objective"]["task"] is True
    assert record["objective"]["anchor"] is False
    assert len(record["config_hash"]) == 64


def test_non_ssr_result_rejects_spatial_mapping():
    with pytest.raises(ResultSchemaError, match="Non-SSR"):
        example_record(
            objective={"task": True, "anchor": True},
            distance_mapping="cosine",
            kernel={},
            recipe="anchor",
        )


def test_ssr_result_requires_kernel_parameters():
    record = example_record()
    record["kernel"] = {"family": "gaussian_dog"}
    with pytest.raises(ResultSchemaError, match="SSR kernel is missing"):
        validate_result_record(record)


def test_known_recipe_must_match_recorded_objective():
    with pytest.raises(ResultSchemaError, match="Recipe 'ssr_only' requires"):
        example_record(objective={"task": True, "ssr": True, "anchor": True})


def test_known_kd_recipe_rejects_contradictory_method_config():
    with pytest.raises(ResultSchemaError, match="requires regularizer='ewc'"):
        build_result_record(
            git_commit="abc123",
            run_id="kd-smoke",
            task_family="classification",
            dataset="split_cifar100",
            model="resnet18",
            seed=1,
            objective={"task": True, "kd": True, "ewc": True},
            distance_mapping="none",
            kernel={},
            metrics={"avg_accuracy": 0.5},
            runtime={"elapsed_s": 1.0},
            config={"method": {"name": "kd_ewc", "regularizer": "mas"}},
            dataset_hash="def456",
            recipe="kd_ewc",
        )


def test_incomplete_editing_record_is_not_official():
    with pytest.raises(ResultSchemaError, match="Official editing results"):
        example_record(
            attempted=10,
            succeeded=9,
            failed=1,
            status="incomplete",
        )

    diagnostic = example_record(
        attempted=10,
        succeeded=9,
        failed=1,
        status="incomplete",
        allow_incomplete=True,
    )
    assert diagnostic["status"] == "incomplete"


def test_editing_completion_counts_must_balance():
    with pytest.raises(ResultSchemaError, match=r"succeeded \+ failed"):
        example_record(
            attempted=10,
            succeeded=8,
            failed=1,
            status="incomplete",
            allow_incomplete=True,
        )
