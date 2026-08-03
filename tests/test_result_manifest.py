from __future__ import annotations

from pathlib import Path

import pytest

from scripts.validate_result_manifest import validate_manifest


def test_meeting_plan_manifest_is_valid():
    validate_manifest(Path("configs/meeting_20260803/manifest.yaml").resolve())


def test_matched_kd_contrasts_are_registered():
    import yaml

    payload = yaml.safe_load(
        Path("configs/meeting_20260803/manifest.yaml").read_text(encoding="utf-8")
    )
    cohort = payload["cohorts"]["continual_learning_matched_kd"]
    assert cohort["primary_contrasts"] == [
        ["kd_ewc", "kd"],
        ["kd_mas", "kd"],
        ["kd_si", "kd"],
        ["kd_ssr", "kd"],
    ]


def test_development_confirmation_overlap_is_rejected(tmp_path):
    manifest = tmp_path / "bad.yaml"
    manifest.write_text(
        """schema_version: '1.0'
cohorts:
  x:
    task_family: editing
    model: m
    datasets: [d]
    recipes: [plain]
    development_seeds: [1]
    confirmation_seeds: [1]
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="reuses development seeds"):
        validate_manifest(manifest)
