from __future__ import annotations

import json
from argparse import Namespace

import pytest
import torch

from llm_ke.biocs_editor import run_sequential_editing
from llm_ke.locality import (
    LOCALITY_EVALUATOR,
    LOCALITY_EVALUATOR_VERSION,
    evaluate_locality,
)
from ssr_utils.result_schema import ResultSchemaError
from scripts import run_llm_ke_easyedit


def test_empty_locality_preserves_historical_zero_and_records_no_items():
    result = evaluate_locality({}, lambda prompt: "unused")

    assert result["score"] == 0.0
    assert result["n_items"] == 0
    assert result["n_matched"] == 0
    assert result["evaluator"] == LOCALITY_EVALUATOR
    assert result["evaluator_version"] == LOCALITY_EVALUATOR_VERSION


def test_multiple_locality_items_are_all_evaluated():
    locality = {
        "neighborhood": [
            {"prompt": "capital", "ground_truth": "Paris"},
            {"prompt": "planet", "ground_truth": "Mars"},
        ],
        "other": [{"prompt": "author", "ground_truth": "Austen"}],
    }
    predictions = {
        "capital": "The answer is Paris.",
        "planet": "The answer is Venus.",
        "author": "Jane Austen wrote it.",
    }

    result = evaluate_locality(locality, predictions.__getitem__)

    assert result["score"] == pytest.approx(2 / 3)
    assert result["n_items"] == 3
    assert result["n_matched"] == 2


def test_nested_ground_truth_matches_any_flattened_alias():
    locality = {
        "neighborhood": [
            {
                "prompt": "city",
                "ground_truth": [
                    {"str": "New York City"},
                    [{"text": "NYC"}],
                ],
            }
        ]
    }

    result = evaluate_locality(locality, lambda prompt: "NYC is in the United States")

    assert result["score"] == 1.0
    assert result["evaluations"][0]["ground_truths"] == ["New York City", "NYC"]


def test_easyedit_locality_arrays_remain_aligned_with_edit_prompts(tmp_path, monkeypatch):
    data_path = tmp_path / "data.json"
    data_path.write_text(
        json.dumps(
            [
                {
                    "prompt": "p1",
                    "target_new": {"str": "t1"},
                    "ground_truth": {"str": "g1"},
                    "locality": {
                        "n": [
                            {
                                "prompt": "l1",
                                "ground_truth": [[{"text": "nested label"}]],
                            }
                        ]
                    },
                },
                {"prompt": "p2", "target_new": "t2", "ground_truth": "g2"},
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setitem(run_llm_ke_easyedit.DATASET_MAP, "zsre", str(data_path))

    loaded = run_llm_ke_easyedit.load_dataset("zsre", 2)

    assert loaded["locality_inputs"] == ["l1", ""]
    assert loaded["locality_labels"] == ["nested label", ""]
    assert len(loaded["prompts"]) == len(loaded["locality_inputs"])


class _FakeDataset:
    offset = 0

    def __len__(self):
        return 2

    def __getitem__(self, index):
        return {
            "prompt": f"p{index}",
            "target_new": f"t{index}",
            "ground_truth": "old",
            "locality": {},
        }


class _PartiallyFailingEditor:
    input_device = torch.device("cpu")

    def metadata(self):
        return {
            "locality_evaluator": LOCALITY_EVALUATOR,
            "locality_evaluator_version": LOCALITY_EVALUATOR_VERSION,
        }

    def edit(self, prompt, target):
        return {"success": prompt == "p0", "reason": "intentional test failure"}

    def evaluate_edit(self, *args):
        return {"efficacy": 1.0, "locality": 1.0}


def test_partial_custom_run_is_explicitly_incomplete(tmp_path):
    summary = run_sequential_editing(
        _PartiallyFailingEditor(),
        _FakeDataset(),
        n_edits=2,
        output_path=str(tmp_path / "results.json"),
    )

    assert summary["attempted"] == 2
    assert summary["succeeded"] == 1
    assert summary["failed"] == 1
    assert summary["status"] == "incomplete"
    assert summary["n_edits"] == 1
    assert summary["failures"][0]["idx"] == 1


def test_incomplete_writer_uses_a_nonofficial_diagnostic_sidecar(tmp_path, monkeypatch):
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text("[]", encoding="utf-8")
    monkeypatch.setitem(run_llm_ke_easyedit.DATASET_MAP, "zsre", str(dataset_path))
    output = tmp_path / "run"
    run_manifest = output / "run_manifest.json"
    run_manifest.parent.mkdir(parents=True)
    run_manifest.write_text("{}", encoding="utf-8")
    args = Namespace(
        method="ft",
        dataset="zsre",
        output=str(output),
        model_name="gpt2-xl",
        n_edits=2,
        seed=7,
        result_manifest=None,
    )
    summary = {
        "efficacy": 100.0,
        "locality": 100.0,
        "n_edits": 1,
        "requested_n_edits": 2,
        "attempted": 2,
        "succeeded": 1,
        "failed": 1,
        "status": "incomplete",
        "locality_evaluator": LOCALITY_EVALUATOR,
        "locality_evaluator_version": LOCALITY_EVALUATOR_VERSION,
    }

    with pytest.raises(ResultSchemaError, match="not official"):
        run_llm_ke_easyedit.write_result_record(args, summary, None, run_manifest)

    assert (output / "incomplete_result_record.json").is_file()
    assert not (output / "result_record.json").exists()
