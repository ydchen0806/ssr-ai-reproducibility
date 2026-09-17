from __future__ import annotations

import csv
import json

import pytest

from scripts.summarize_direct_editing import (
    load_pairs,
    read_plan,
    summarize,
    write_outputs,
)
from scripts.validate_direct_attribution_matrix import (
    validate_factorial_editing_summary,
)
from ssr_utils.result_schema import build_result_record


OBJECTIVES = {
    "plain": {"task": True},
    "anchor": {"task": True, "anchor": True},
    "spectral": {"task": True, "spectral": True},
    "stabilized": {"task": True, "anchor": True, "spectral": True},
    "ssr_only": {"task": True, "ssr": True},
    "full": {"task": True, "ssr": True, "anchor": True, "spectral": True},
}
FACTORIAL_ARMS = (
    ("plain", "none"),
    ("anchor", "none"),
    ("spectral", "none"),
    ("stabilized", "none"),
    ("ssr_only", "projective"),
    ("ssr_only", "cosine"),
    ("full", "projective"),
    ("full", "cosine"),
)


def editing_record(
    dataset: str,
    recipe: str,
    seed: int,
    efficacy: float,
    *,
    mapping: str | None = None,
    pairing_protocol_hash: str = "b" * 64,
) -> dict:
    uses_ssr = recipe in {"ssr_only", "full"}
    mapping = mapping or ("projective" if uses_ssr else "none")
    return build_result_record(
        git_commit="abc123",
        run_id=f"{dataset}-{recipe}-{seed}",
        task_family="editing",
        dataset=dataset,
        model="gpt2-xl",
        seed=seed,
        objective=OBJECTIVES[recipe],
        distance_mapping=mapping,
        kernel=(
            {
                "family": "gaussian",
                "A_exc": 1.2,
                "A_inh": 0.9,
                "sigma_exc": 0.22,
                "sigma_inh": 0.60,
            }
            if uses_ssr
            else {}
        ),
        metrics={
            "efficacy": efficacy,
            "locality": efficacy / 2.0,
            "final_history_efficacy": efficacy / 3.0,
            "final_history_locality": efficacy / 4.0,
        },
        metric_directions={
            "efficacy": True,
            "locality": True,
            "final_history_efficacy": True,
            "final_history_locality": True,
        },
        runtime={"elapsed_s": 1.0},
        config={"recipe": recipe},
        dataset_hash="d" * 64,
        recipe=recipe,
        requested=100,
        attempted=100,
        succeeded=100,
        failed=0,
        status="complete",
        evaluator="locked-evaluator",
        evaluator_version="1.0",
        evaluation_protocol_hash="e" * 64,
        model_hash="a" * 64,
        pairing_protocol_hash=pairing_protocol_hash,
        data_offset=500,
        n_edits=100,
    )


def write_condition(output, record: dict) -> None:
    (output / "result_record.json").write_text(
        json.dumps(record),
        encoding="utf-8",
    )
    (output / "results.json").write_text(
        json.dumps(
            {
                "results": [
                    {
                        "idx": index,
                        "prompt": f"prompt-{index}",
                        "target_new": f"target-{index}",
                    }
                    for index in range(500, 600)
                ]
            }
        ),
        encoding="utf-8",
    )


def test_direct_editing_summary_has_absolute_means_delta_and_ci(tmp_path):
    plan = tmp_path / "planned_runs.tsv"
    rows = []
    for seed in (1, 3, 5):
        for recipe, efficacy in (("plain", 70.0 + seed), ("ssr_only", 71.0 + seed)):
            output = tmp_path / recipe / f"seed_{seed}"
            output.mkdir(parents=True)
            write_condition(
                output,
                editing_record("zsre", recipe, seed, efficacy),
            )
            rows.append(
                {
                    "phase": "confirm",
                    "dataset": "zsre",
                    "recipe": recipe,
                    "mapping": "none" if recipe == "plain" else "projective",
                    "seed": seed,
                    "data_offset": 500,
                    "n_edits": 100,
                    "action": "run",
                    "source": "",
                    "output": output,
                }
            )
    with plan.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    payload, flat = summarize(load_pairs(read_plan(plan)))

    efficacy = payload["datasets"]["zsre"]["metrics"]["efficacy"]
    assert efficacy["n"] == 3
    assert efficacy["control_mean"] == pytest.approx(73.0)
    assert efficacy["treatment_mean"] == pytest.approx(74.0)
    assert efficacy["difference_mean"] == pytest.approx(1.0)
    assert efficacy["favorable_pairs"] == 3
    assert efficacy["favorable_pair_fraction"] == pytest.approx(1.0)
    assert len(flat) == 4


def factorial_plan(
    tmp_path,
    *,
    datasets=("zsre",),
    seeds=(1, 3, 5),
):
    plan = tmp_path / "factorial_runs.tsv"
    rows = []
    offsets = {
        ("plain", "none"): 0.0,
        ("anchor", "none"): 0.4,
        ("spectral", "none"): 0.5,
        ("stabilized", "none"): 0.8,
        ("ssr_only", "projective"): 1.5,
        ("ssr_only", "cosine"): 1.0,
        ("full", "projective"): 2.0,
        ("full", "cosine"): 1.4,
    }
    for dataset_index, dataset in enumerate(datasets):
        for seed in seeds:
            for recipe, mapping in FACTORIAL_ARMS:
                output = tmp_path / dataset / recipe / mapping / f"seed_{seed}"
                output.mkdir(parents=True)
                efficacy = 70.0 + dataset_index + seed + offsets[(recipe, mapping)]
                write_condition(
                    output,
                    editing_record(
                        dataset, recipe, seed, efficacy, mapping=mapping
                    ),
                )
                rows.append(
                    {
                        "phase": "confirm",
                        "dataset": dataset,
                        "recipe": recipe,
                        "mapping": mapping,
                        "seed": seed,
                        "data_offset": 500,
                        "n_edits": 100,
                        "action": "run",
                        "source": "",
                        "output": output,
                    }
                )
    with plan.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return plan, rows


def test_factorial_summary_covers_recipes_mappings_and_key_contrasts(tmp_path):
    plan, _ = factorial_plan(tmp_path)
    payload, flat = summarize(load_pairs(read_plan(plan)))

    assert payload["mode"] == "factorial"
    assert payload["recipes"] == [
        "plain",
        "anchor",
        "spectral",
        "stabilized",
        "ssr_only",
        "full",
    ]
    assert set(payload["arms"]["zsre"]) == {
        f"{recipe}:{mapping}" for recipe, mapping in FACTORIAL_ARMS
    }
    expected_contrasts = {
        "ssr_only_minus_plain",
        "full_minus_stabilized",
        "ssr_only_minus_anchor",
        "ssr_only_minus_spectral",
        "full_minus_plain",
    }
    assert set(payload["contrasts"]["zsre"]) == expected_contrasts
    assert all(
        set(payload["contrasts"]["zsre"][contrast]) == {"projective", "cosine"}
        for contrast in expected_contrasts
    )

    efficacy = payload["contrasts"]["zsre"]["ssr_only_minus_plain"][
        "projective"
    ]["metrics"]["efficacy"]
    assert efficacy["control_mean"] == pytest.approx(73.0)
    assert efficacy["treatment_mean"] == pytest.approx(74.5)
    assert efficacy["difference_mean"] == pytest.approx(1.5)
    assert efficacy["ci95_low"] == pytest.approx(1.5)
    assert efficacy["ci95_high"] == pytest.approx(1.5)
    assert efficacy["favorable_pairs"] == 3
    assert efficacy["favorable_pair_fraction"] == pytest.approx(1.0)
    assert [pair["seed"] for pair in efficacy["pairs"]] == [1, 3, 5]
    assert all(pair["data_offset"] == 500 for pair in efficacy["pairs"])

    mapping = payload["mapping_contrasts"]["zsre"]["ssr_only"]["metrics"][
        "efficacy"
    ]
    assert mapping["control_mean"] == pytest.approx(74.0)
    assert mapping["treatment_mean"] == pytest.approx(74.5)
    assert mapping["difference_mean"] == pytest.approx(0.5)
    assert mapping["favorable_pair_fraction"] == pytest.approx(1.0)

    assert len(flat) == 48
    output_dir = tmp_path / "summary"
    write_outputs(output_dir, payload, flat)
    assert {
        path.name for path in output_dir.iterdir()
    } == {"summary.json", "summary.csv", "arms.csv", "paired_deltas.csv"}
    with (output_dir / "arms.csv").open(encoding="utf-8", newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 32
    with (output_dir / "paired_deltas.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        pair_rows = list(csv.DictReader(handle))
    assert len(pair_rows) == 144
    assert {row["seed"] for row in pair_rows} == {"1", "3", "5"}


def test_factorial_summary_satisfies_final_four_node_contract(tmp_path):
    seeds = tuple(range(9311, 9330, 2))
    plan, _ = factorial_plan(
        tmp_path,
        datasets=("zsre", "cf", "recent"),
        seeds=seeds,
    )
    payload, flat = summarize(load_pairs(read_plan(plan)))
    output_dir = tmp_path / "paired_summary"
    write_outputs(output_dir, payload, flat)

    validate_factorial_editing_summary(output_dir / "summary.json")


def test_factorial_summary_rejects_nonidentical_record_cohort(tmp_path):
    plan, rows = factorial_plan(tmp_path)
    treatment_path = next(
        row["output"] / "results.json"
        for row in rows
        if row["recipe"] == "ssr_only"
        and row["mapping"] == "projective"
        and row["seed"] == 3
    )
    payload = json.loads(treatment_path.read_text(encoding="utf-8"))
    payload["results"][7]["target_new"] = "different-target"
    treatment_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Unmatched same-record conditions"):
        load_pairs(read_plan(plan))


def test_factorial_summary_requires_per_edit_results(tmp_path):
    plan, rows = factorial_plan(tmp_path)
    results_path = rows[0]["output"] / "results.json"
    results_path.unlink()

    with pytest.raises(FileNotFoundError, match="results.json"):
        load_pairs(read_plan(plan))


def test_factorial_summary_rejects_noncontiguous_record_indices(tmp_path):
    plan, rows = factorial_plan(tmp_path)
    results_path = rows[0]["output"] / "results.json"
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    payload["results"][4]["idx"] = 999
    results_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="item 4 has idx=999, expected 504"):
        load_pairs(read_plan(plan))


def test_factorial_summary_rejects_missing_or_duplicate_cells(tmp_path):
    plan, rows = factorial_plan(tmp_path)
    missing = rows[:-1]
    with pytest.raises(ValueError, match="Incomplete arm"):
        load_pairs(missing)
    with pytest.raises(ValueError, match="Duplicate editing cell"):
        load_pairs(rows + [rows[0]])
