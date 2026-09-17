from __future__ import annotations

from pathlib import Path

import pytest

from scripts.check_result_consistency import check_claim


def test_claim_matches_source_and_target(tmp_path):
    source = tmp_path / "values.csv"
    source.write_text("dataset,value\na,1.25\n", encoding="utf-8")
    target = tmp_path / "main.tex"
    target.write_text("gain=1.25", encoding="utf-8")
    errors = check_claim(
        {
            "id": "gain",
            "source_csv": "values.csv",
            "filters": {"dataset": "a"},
            "value_column": "value",
            "targets": [{"path": "main.tex", "pattern": r"gain=([0-9.]+)"}],
        },
        tmp_path / "registry.yaml",
    )
    assert errors == []


def test_claim_detects_mismatch(tmp_path):
    source = tmp_path / "values.csv"
    source.write_text("dataset,value\na,1.25\n", encoding="utf-8")
    target = tmp_path / "main.tex"
    target.write_text("gain=1.30", encoding="utf-8")
    errors = check_claim(
        {
            "id": "gain",
            "source_csv": "values.csv",
            "filters": {"dataset": "a"},
            "value_column": "value",
            "targets": [{"path": "main.tex", "pattern": r"gain=([0-9.]+)"}],
        },
        tmp_path / "registry.yaml",
    )
    assert errors
