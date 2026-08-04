from __future__ import annotations

import csv
import json

import pytest

from scripts.summarize_direct_editing import load_pairs, read_plan, summarize
from ssr_utils.result_schema import build_result_record


def editing_record(dataset: str, recipe: str, seed: int, efficacy: float) -> dict:
    uses_ssr = recipe == "ssr_only"
    return build_result_record(
        git_commit="abc123",
        run_id=f"{dataset}-{recipe}-{seed}",
        task_family="editing",
        dataset=dataset,
        model="gpt2-xl",
        seed=seed,
        objective={"task": True, "ssr": uses_ssr},
        distance_mapping="projective" if uses_ssr else "none",
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
        metrics={"efficacy": efficacy, "locality": efficacy / 2.0},
        metric_directions={"efficacy": True, "locality": True},
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
        pairing_protocol_hash="b" * 64,
        data_offset=500,
        n_edits=100,
    )


def test_direct_editing_summary_has_absolute_means_delta_and_ci(tmp_path):
    plan = tmp_path / "planned_runs.tsv"
    rows = []
    for seed in (1, 3, 5):
        for recipe, efficacy in (("plain", 70.0 + seed), ("ssr_only", 71.0 + seed)):
            output = tmp_path / recipe / f"seed_{seed}"
            output.mkdir(parents=True)
            (output / "result_record.json").write_text(
                json.dumps(editing_record("zsre", recipe, seed, efficacy)),
                encoding="utf-8",
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
    assert len(flat) == 2
