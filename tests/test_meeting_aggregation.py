from __future__ import annotations

import json

import pytest

from scripts.aggregate_meeting_revision import paired_contrasts, read_records
from ssr_utils.result_schema import build_result_record


def _record(recipe: str, seed: int, value: float, mapping: str) -> dict:
    return {
        "task_family": "editing",
        "dataset": "zsre",
        "model": "gpt2-xl",
        "seed": seed,
        "recipe": recipe,
        "distance_mapping": mapping,
        "data_offset": 0,
        "n_edits": 100,
        "dataset_hash": "dataset-v1",
        "metrics": {"locality": value},
        "metric_directions": {"locality": True},
    }


def _valid_record(*, value: float = 31.0) -> dict:
    return build_result_record(
        git_commit="test-commit",
        run_id="editing-zsre-11",
        task_family="editing",
        dataset="zsre",
        model="gpt2-xl",
        seed=11,
        objective={"task": True, "ssr": True},
        distance_mapping="cosine",
        kernel={
            "family": "gaussian",
            "A_exc": 1.2,
            "A_inh": 0.9,
            "sigma_exc": 0.22,
            "sigma_inh": 0.6,
        },
        metrics={"locality": value},
        runtime={"elapsed_s": 1.0},
        config={"recipe": "ssr_only", "lambda_ssr": 0.003},
        dataset_hash="dataset-v1",
        recipe="ssr_only",
        data_offset=500,
        n_edits=100,
        requested=100,
        attempted=100,
        succeeded=100,
        failed=0,
        status="complete",
        evaluator="test-evaluator",
        evaluator_version="1",
        evaluation_protocol_hash="a" * 64,
        model_hash="b" * 64,
        pairing_protocol_hash="c" * 64,
    )


def _editing_plan() -> dict:
    return {
        "cohorts": {
            "editing": {
                "task_family": "editing",
                "datasets": ["zsre"],
                "model": "gpt2-xl",
                "paired_identity": [
                    "dataset",
                    "model",
                    "seed",
                    "data_offset",
                    "n_edits",
                    "dataset_hash",
                ],
            }
        }
    }


def test_read_records_deduplicates_exact_copies_deterministically(tmp_path):
    record = _valid_record()
    first = tmp_path / "a_imported" / "result_record.json"
    second = tmp_path / "z_reused" / "result_record.json"
    for path in (second, first):
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(record), encoding="utf-8")

    records = read_records(tmp_path, _editing_plan())

    assert len(records) == 1
    assert records[0]["_path"] == str(first)


def test_read_records_rejects_conflicting_copies_of_same_cell(tmp_path):
    paths = [
        tmp_path / "imported" / "result_record.json",
        tmp_path / "reused" / "result_record.json",
    ]
    for path, value in zip(paths, (31.0, 32.0)):
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(_valid_record(value=value)), encoding="utf-8")

    with pytest.raises(ValueError, match="conflicting duplicate aggregation identity") as error:
        read_records(tmp_path, _editing_plan())

    message = str(error.value)
    assert str(paths[0]) in message
    assert str(paths[1]) in message
    assert "metrics" in message


def test_read_records_enforces_cohort_metric_contract(tmp_path):
    path = tmp_path / "run" / "result_record.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(_valid_record()), encoding="utf-8")
    plan = _editing_plan()
    plan["cohorts"]["editing"]["required_metrics"] = ["efficacy", "locality"]
    plan["cohorts"]["editing"]["required_runtime"] = ["elapsed_s"]

    with pytest.raises(ValueError, match="cohort metric contract") as error:
        read_records(tmp_path, plan)

    assert "efficacy" in str(error.value)


def test_aggregation_excludes_development_and_reports_missing_confirm_seed():
    plan = {
        "cohorts": {
            "editing": {
                "task_family": "editing",
                "datasets": ["zsre"],
                "model": "gpt2-xl",
                "recipes": ["plain", "ssr_only"],
                "mappings": ["projective"],
                "mapping_dependent_recipes": ["ssr_only"],
                "paired_identity": [
                    "dataset",
                    "model",
                    "seed",
                    "data_offset",
                    "n_edits",
                    "dataset_hash",
                ],
                "development_seeds": [1],
                "confirmation_seeds": [11, 13, 15],
                "primary_contrasts": [["ssr_only", "plain"]],
            }
        }
    }
    records = [
        _record("plain", 1, 20.0, "none"),
        _record("ssr_only", 1, 99.0, "projective"),
        _record("plain", 11, 30.0, "none"),
        _record("ssr_only", 11, 31.0, "projective"),
        _record("plain", 13, 40.0, "none"),
        _record("ssr_only", 13, 42.0, "projective"),
    ]

    summaries, exclusions = paired_contrasts(plan, records)

    assert len(summaries) == 1
    assert summaries[0]["n"] == 2
    assert summaries[0]["difference_mean"] == 1.5
    missing = json.loads(exclusions[0]["missing_identity"])
    assert missing["missing_expected_seeds"] == [15]


def test_pairing_does_not_join_different_edit_prefixes():
    plan = {
        "cohorts": {
            "editing": {
                "task_family": "editing",
                "datasets": ["zsre"],
                "model": "gpt2-xl",
                "recipes": ["plain", "ssr_only"],
                "mappings": ["cosine"],
                "mapping_dependent_recipes": ["ssr_only"],
                "paired_identity": ["dataset", "model", "seed", "data_offset", "n_edits"],
                "confirmation_seeds": [11, 13],
                "primary_contrasts": [["ssr_only", "plain"]],
            }
        }
    }
    controls = [_record("plain", seed, 30.0, "none") for seed in (11, 13)]
    treatments = [_record("ssr_only", seed, 35.0, "cosine") for seed in (11, 13)]
    for row in treatments:
        row["data_offset"] = 100

    summaries, exclusions = paired_contrasts(plan, controls + treatments)

    assert summaries == []
    assert exclusions


def test_aggregation_rejects_fixed_edit_count_drift():
    plan = {
        "cohorts": {
            "editing": {
                "task_family": "editing",
                "datasets": ["zsre"],
                "model": "gpt2-xl",
                "recipes": ["plain", "ssr_only"],
                "mappings": ["cosine"],
                "mapping_dependent_recipes": ["ssr_only"],
                "paired_identity": ["dataset", "model", "seed", "n_edits"],
                "confirmation_seeds": [11, 13],
                "n_edits": 100,
                "primary_contrasts": [["ssr_only", "plain"]],
            }
        }
    }
    records = [
        _record(recipe, seed, 30.0, "cosine" if recipe == "ssr_only" else "none")
        for seed in (11, 13)
        for recipe in ("plain", "ssr_only")
    ]
    records[-1]["n_edits"] = 5

    with pytest.raises(ValueError, match="requires n_edits=100"):
        paired_contrasts(plan, records)


def test_aggregation_rejects_mixed_cohort_provenance():
    plan = {
        "cohorts": {
            "editing": {
                "task_family": "editing",
                "datasets": ["zsre"],
                "model": "gpt2-xl",
                "recipes": ["plain", "ssr_only"],
                "mappings": ["cosine"],
                "mapping_dependent_recipes": ["ssr_only"],
                "paired_identity": [
                    "dataset", "model", "seed", "pairing_protocol_hash"
                ],
                "cohort_invariants": ["pairing_protocol_hash"],
                "confirmation_seeds": [11, 13],
                "primary_contrasts": [["ssr_only", "plain"]],
            }
        }
    }
    records = []
    for seed, protocol in ((11, "a" * 64), (13, "b" * 64)):
        for recipe in ("plain", "ssr_only"):
            row = _record(
                recipe,
                seed,
                30.0,
                "cosine" if recipe == "ssr_only" else "none",
            )
            row["pairing_protocol_hash"] = protocol
            records.append(row)

    with pytest.raises(ValueError, match="mixes provenance"):
        paired_contrasts(plan, records)


def test_aggregation_rejects_multiple_complete_protocols_per_seed():
    plan = {
        "cohorts": {
            "editing": {
                "task_family": "editing",
                "datasets": ["zsre"],
                "model": "gpt2-xl",
                "recipes": ["plain", "ssr_only"],
                "mappings": ["cosine"],
                "mapping_dependent_recipes": ["ssr_only"],
                "paired_identity": ["dataset", "model", "seed", "pairing_hash"],
                "confirmation_seeds": [11, 13],
                "primary_contrasts": [["ssr_only", "plain"]],
            }
        }
    }
    records = []
    for seed in (11, 13):
        for pairing_hash in ("protocol-a", "protocol-b"):
            for recipe in ("plain", "ssr_only"):
                row = _record(
                    recipe,
                    seed,
                    30.0,
                    "cosine" if recipe == "ssr_only" else "none",
                )
                row["pairing_hash"] = pairing_hash
                records.append(row)

    with pytest.raises(ValueError, match="ambiguous paired protocols"):
        paired_contrasts(plan, records)


def test_kd_ssr_cosine_treatment_pairs_with_mapping_free_kd_control():
    plan = {
        "cohorts": {
            "matched_kd": {
                "task_family": "classification",
                "datasets": ["split_cifar100"],
                "model": "resnet18",
                "recipes": ["kd", "kd_ssr"],
                "mappings": ["cosine"],
                "mapping_dependent_recipes": ["kd_ssr"],
                "paired_identity": ["dataset", "model", "seed", "dataset_hash", "pairing_hash"],
                "confirmation_seeds": [3101, 3103],
                "primary_contrasts": [["kd_ssr", "kd"]],
            }
        }
    }
    common = {
        "task_family": "classification",
        "dataset": "split_cifar100",
        "model": "resnet18",
        "seed": 3101,
        "dataset_hash": "dataset-v1",
        "pairing_hash": "same-scaffold",
        "metrics": {"avg_accuracy": 70.0},
        "metric_directions": {"avg_accuracy": True},
    }
    records = []
    for seed in (3101, 3103):
        records.extend(
            [
                {**common, "seed": seed, "recipe": "kd", "distance_mapping": "none"},
                {
                    **common,
                    "seed": seed,
                    "recipe": "kd_ssr",
                    "distance_mapping": "cosine",
                    "metrics": {"avg_accuracy": 71.0},
                },
            ]
        )

    summaries, exclusions = paired_contrasts(plan, records)

    assert exclusions == []
    assert len(summaries) == 1
    assert summaries[0]["mapping"] == "cosine"
    assert summaries[0]["difference_mean"] == 1.0
