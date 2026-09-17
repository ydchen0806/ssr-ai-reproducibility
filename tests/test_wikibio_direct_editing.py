from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

import llm_ke.biocs_editor as editor_module
from llm_ke.biocs_editor import KnowEditDataset, normalize_knowedit_record
from scripts import run_llm_ke_easyedit
from ssr_utils.editing_protocol import (
    GPT2_XL_FINGERPRINT_FILES,
    GPT2_XL_MODEL_HASH,
    WIKIBIO_DATASET_FILES,
    WIKIBIO_EVALUATION_PROTOCOL_HASH,
    WIKIBIO_PAIRING_PROTOCOL_HASH,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PROJECT_ROOT / "configs/meeting_20260804/direct_editing_wikibio_gpt2xl.yaml"
)


def _wikibio_row() -> dict:
    return {
        "text": "This is a Wikipedia passage about Ada Lovelace.",
        "labels": "She wrote notes on the Analytical Engine.",
        "concept": "Ada Lovelace",
        "locality": {
            "Relation_Specificity": [
                {"prompt": "Ada Lovelace was born in", "ground_truth": ["London"]}
            ]
        },
    }


def test_wikibio_fields_normalize_to_the_same_editor_schema(tmp_path):
    normalized = normalize_knowedit_record(_wikibio_row())
    assert normalized["prompt"].startswith("This is a Wikipedia passage")
    assert normalized["target_new"] == "She wrote notes on the Analytical Engine."
    assert normalized["subject"] == "Ada Lovelace"
    assert normalized["portability"] == {}

    path = tmp_path / "wikibio.json"
    path.write_text(json.dumps([_wikibio_row()]), encoding="utf-8")
    assert KnowEditDataset(str(path))[0] == normalized


def test_shared_loader_accepts_wikibio_without_a_second_conversion(tmp_path, monkeypatch):
    path = tmp_path / "wikibio.json"
    path.write_text(json.dumps([_wikibio_row()]), encoding="utf-8")
    monkeypatch.setitem(run_llm_ke_easyedit.DATASET_MAP, "wikibio", str(path))

    loaded = run_llm_ke_easyedit.load_dataset("wikibio", 1)
    assert loaded["prompts"] == [_wikibio_row()["text"]]
    assert loaded["target_new"] == [_wikibio_row()["labels"]]
    assert loaded["subject"] == [_wikibio_row()["concept"]]


def test_wikibio_lock_matches_code_derived_hashes_and_direct_objective():
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["model"]["fingerprint_files"] == GPT2_XL_FINGERPRINT_FILES
    assert config["model"]["fingerprint_sha256"] == GPT2_XL_MODEL_HASH
    assert config["datasets"] == WIKIBIO_DATASET_FILES
    assert config["evaluation"]["protocol_sha256"] == WIKIBIO_EVALUATION_PROTOCOL_HASH
    assert config["pairing"]["protocol_sha256"] == WIKIBIO_PAIRING_PROTOCOL_HASH
    assert config["objective"]["recipes"] == ["plain", "ssr_only"]
    assert config["objective"]["lambda_anchor"] == 0.0
    assert config["objective"]["lambda_spectral"] == 0.0


def test_custom_editors_share_wikibio_context_and_decoding_limits(
    tmp_path, monkeypatch
):
    captured = []

    class FakeEditor:
        def __init__(self, *args, **kwargs):
            captured.append(kwargs)

    class FakeDataset:
        def __init__(self, *args, **kwargs):
            pass

    def fake_run(editor, dataset, n_edits, output_path, **kwargs):
        return {"n_edits": n_edits, "efficacy": 1.0, "locality": 1.0}

    monkeypatch.setattr(editor_module, "BioCsLLMEditor", FakeEditor)
    monkeypatch.setattr(editor_module, "FTBaselineEditor", FakeEditor)
    monkeypatch.setattr(editor_module, "KnowEditDataset", FakeDataset)
    monkeypatch.setattr(editor_module, "run_sequential_editing", fake_run)
    monkeypatch.setenv("KE_MAX_LENGTH", "256")
    monkeypatch.setenv("KE_MAX_NEW_TOKENS", "96")

    config = {
        "recipe": "ssr_only",
        "lambda_ssr": 0.003,
        "lambda_anchor": 0.0,
        "lambda_spectral": 0.0,
        "distance_mapping": "projective",
        "sampler_seed": 9401,
    }
    run_llm_ke_easyedit.run_biocs_method(
        "wikibio", 1, str(tmp_path / "ssr"), "gpt2-xl", config
    )
    run_llm_ke_easyedit.run_ft_method(
        "wikibio", 1, str(tmp_path / "plain"), "gpt2-xl"
    )

    assert len(captured) == 2
    assert {kwargs["max_length"] for kwargs in captured} == {256}
    assert {kwargs["max_new_tokens"] for kwargs in captured} == {96}
    assert {kwargs["lr"] for kwargs in captured} == {0.0001}
    assert {kwargs["num_steps"] for kwargs in captured} == {25}


def test_direct_launcher_plans_only_matched_plain_and_ssr(tmp_path):
    result_root = tmp_path / "results"
    env = {
        **os.environ,
        "PROJECT_ROOT": str(PROJECT_ROOT),
        "PYTHON": sys.executable,
        "RESULT_ROOT": str(result_root),
        "RUN_ID": "wikibio_test",
        "MODEL_NAME": "gpt2-xl",
    }
    subprocess.run(
        [
            "bash",
            str(PROJECT_ROOT / "scripts/run_wikibio_direct_editing_8gpu.sh"),
            "--dry-run",
        ],
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    rows = [
        line.split("\t")
        for line in (result_root / "planned_runs.tsv")
        .read_text(encoding="utf-8")
        .splitlines()[1:]
    ]
    assert len(rows) == 20
    assert {row[1] for row in rows} == {"wikibio"}
    assert {row[2] for row in rows} == {"plain", "ssr_only"}
    assert {row[3] for row in rows if row[2] == "plain"} == {"none"}
    assert {row[3] for row in rows if row[2] == "ssr_only"} == {"projective"}
    assert {row[5] for row in rows} == {"0"}
    assert {row[6] for row in rows} == {"200"}
