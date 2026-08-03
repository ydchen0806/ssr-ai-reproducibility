from __future__ import annotations

import json
import os
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest

import llm_ke.biocs_editor as editor_module
from scripts.run_llm_ke_easyedit import (
    resolve_history_run_config,
    run_biocs_method,
    run_ft_method,
    write_run_manifest,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _history_args(**overrides) -> Namespace:
    values = {
        "method": "biocs",
        "n_edits": 100,
        "evaluate_history": False,
        "history_checkpoints": None,
        "history_max_samples": 0,
    }
    values.update(overrides)
    return Namespace(**values)


def test_history_cli_is_explicit_and_validated():
    resolved = resolve_history_run_config(
        _history_args(
            evaluate_history=True,
            history_checkpoints=[100, 50, 50],
            history_max_samples=20,
        )
    )
    assert resolved == {
        "evaluate_history": True,
        "history_checkpoints": [50, 100],
        "history_max_samples": 20,
    }

    with pytest.raises(ValueError, match="requires --evaluate_history"):
        resolve_history_run_config(
            _history_args(history_checkpoints=[50])
        )
    with pytest.raises(ValueError, match="cannot exceed"):
        resolve_history_run_config(
            _history_args(evaluate_history=True, history_checkpoints=[101])
        )


def test_external_easyedit_methods_do_not_accept_custom_history_evaluation():
    assert resolve_history_run_config(_history_args(method="ROME")) is None
    with pytest.raises(ValueError, match="only supported"):
        resolve_history_run_config(
            _history_args(method="MEMIT", evaluate_history=True)
        )


@pytest.mark.parametrize("runner_name", ["biocs", "ft"])
def test_custom_runner_forwards_history_arguments(monkeypatch, tmp_path, runner_name):
    captured = {}

    class FakeEditor:
        def __init__(self, *args, **kwargs):
            captured["editor_kwargs"] = kwargs

    class FakeDataset:
        def __init__(self, *args, **kwargs):
            captured["dataset_kwargs"] = kwargs

    def fake_run(editor, dataset, n_edits, output_path, **kwargs):
        captured["history_kwargs"] = kwargs
        return {"n_edits": n_edits, "efficacy": 1.0, "locality": 1.0}

    monkeypatch.setattr(editor_module, "BioCsLLMEditor", FakeEditor)
    monkeypatch.setattr(editor_module, "FTBaselineEditor", FakeEditor)
    monkeypatch.setattr(editor_module, "KnowEditDataset", FakeDataset)
    monkeypatch.setattr(editor_module, "run_sequential_editing", fake_run)
    history = {
        "evaluate_history": True,
        "history_checkpoints": [50],
        "history_max_samples": 0,
    }

    if runner_name == "biocs":
        run_biocs_method(
            "zsre",
            100,
            str(tmp_path),
            "gpt2-xl",
            {
                "recipe": "full",
                "lambda_ssr": 0.003,
                "lambda_anchor": 0.001,
                "lambda_spectral": 0.01,
                "distance_mapping": "projective",
                "sampler_seed": 7,
            },
            history,
        )
    else:
        run_ft_method("zsre", 100, str(tmp_path), "gpt2-xl", history)

    assert captured["history_kwargs"] == history


def test_history_protocol_is_written_to_run_manifest(tmp_path):
    args = Namespace(
        output=str(tmp_path),
        method="biocs",
        hparams_dir="unused",
        hparams_model="unused",
        evaluate_history=True,
        history_checkpoints=[50],
        history_max_samples=0,
    )
    history = resolve_history_run_config(
        _history_args(
            evaluate_history=True,
            history_checkpoints=[50],
        )
    )
    path = write_run_manifest(args, {"recipe": "full"}, history)
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["historical_retention_evaluation"] == history


def _dry_run_launcher(tmp_path: Path, *, full_matrix: bool) -> list[list[str]]:
    result_root = tmp_path / ("full" if full_matrix else "deduplicated")
    command = [
        "bash",
        str(PROJECT_ROOT / "scripts/run_meeting_editing_ablation.sh"),
        "--phase",
        "confirm",
        "--dataset",
        "zsre",
        "--recipe",
        "plain,anchor,spectral,stabilized,ssr_only,full",
        "--mapping",
        "projective,cosine",
        "--seed",
        "7",
        "--dry-run",
    ]
    if full_matrix:
        command.append("--full-matrix")
    env = {
        **os.environ,
        "PROJECT_ROOT": str(PROJECT_ROOT),
        "PYTHON": sys.executable,
        "RESULT_ROOT": str(result_root),
        "MODEL_NAME": "gpt2-xl",
    }
    subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True, capture_output=True, text=True)
    lines = (result_root / "planned_runs.tsv").read_text(encoding="utf-8").splitlines()
    return [line.split("\t") for line in lines[1:]]


def test_launcher_deduplicates_mapping_independent_controls(tmp_path):
    rows = _dry_run_launcher(tmp_path, full_matrix=False)
    assert len(rows) == 8
    controls = [row for row in rows if row[2] in {"plain", "anchor", "spectral", "stabilized"}]
    ssr = [row for row in rows if row[2] in {"ssr_only", "full"}]
    assert len(controls) == 4
    assert {row[3] for row in controls} == {"none"}
    assert len(ssr) == 4
    assert {row[3] for row in ssr} == {"projective", "cosine"}
    assert {row[4] for row in rows} == {"7"}
    assert {row[5] for row in rows} == {"500"}


def test_launcher_can_request_the_redundant_full_matrix(tmp_path):
    rows = _dry_run_launcher(tmp_path, full_matrix=True)
    assert len(rows) == 12
    assert {row[3] for row in rows} == {"projective", "cosine"}
