from __future__ import annotations

import pytest

from scripts.summarize_matched_kd import summarize


def record(dataset: str, recipe: str, seed: int, accuracy: float, forgetting: float):
    return {
        "dataset": dataset,
        "recipe": recipe,
        "seed": seed,
        "metrics": {
            "avg_accuracy": accuracy,
            "last_accuracy": accuracy - 1.0,
            "avg_forgetting": forgetting,
            "backward_transfer": -forgetting,
            "cil_avg_accuracy": accuracy - 2.0,
            "cil_last_accuracy": accuracy - 3.0,
            "effective_rank": accuracy / 10.0,
            "prototype_overlap": forgetting / 10.0,
        },
    }


def test_summary_pairs_every_regularizer_against_same_kd_seed():
    records = {}
    shifts = {
        "kd_ewc": 0.1,
        "kd_mas": 0.2,
        "kd_si": 0.3,
        "kd_center": 0.4,
        "kd_protodecor": 0.5,
        "kd_spectral": 0.6,
        "kd_ssr": 0.7,
    }
    for seed in (1, 3, 5):
        records[("split_cifar100", "kd", seed)] = record(
            "split_cifar100", "kd", seed, 70.0 + seed, 5.0
        )
        for treatment, shift in shifts.items():
            records[("split_cifar100", treatment, seed)] = record(
                "split_cifar100", treatment, seed, 70.0 + seed + shift, 5.0 - shift
            )

    payload, rows = summarize(records)

    assert set(payload["datasets"]["split_cifar100"]) == set(shifts)
    assert len(rows) == 104
    ssr = payload["datasets"]["split_cifar100"]["kd_ssr"]["metrics"]
    assert ssr["avg_accuracy"]["difference_mean"] == pytest.approx(0.7)
    assert ssr["avg_forgetting"]["difference_mean"] == pytest.approx(0.7)
    assert ssr["avg_accuracy"]["favorable_pairs"] == 3
    assert ssr["avg_forgetting"]["favorable_pairs"] == 3
    ssr_vs_center = payload["ssr_vs_controls"]["split_cifar100"]["kd_center"]
    assert ssr_vs_center["control"] == "kd_center"
    assert ssr_vs_center["treatment"] == "kd_ssr"
    assert ssr_vs_center["metrics"]["avg_accuracy"][
        "difference_mean"
    ] == pytest.approx(0.3)


def test_summary_rejects_a_silently_missing_required_metric():
    records = {}
    for seed in (1, 3):
        for recipe in (
            "kd",
            "kd_ewc",
            "kd_mas",
            "kd_si",
            "kd_center",
            "kd_protodecor",
            "kd_spectral",
            "kd_ssr",
        ):
            records[("split_cifar100", recipe, seed)] = record(
                "split_cifar100", recipe, seed, 70.0, 5.0
            )
    del records[("split_cifar100", "kd_ssr", 3)]["metrics"]["effective_rank"]

    with pytest.raises(ValueError, match="missing metrics"):
        summarize(records)
