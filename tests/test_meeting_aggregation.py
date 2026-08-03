from __future__ import annotations

import json

from scripts.aggregate_meeting_revision import paired_contrasts


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
