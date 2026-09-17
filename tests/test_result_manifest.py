from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.build_result_manifest import build_manifest
from scripts.validate_result_manifest import validate_manifest
from ssr_utils.result_schema import build_result_record


def _editing_record(
    *,
    recipe: str,
    mapping: str,
    seed: int,
    run_id: str,
) -> dict:
    ssr = recipe == "ssr_only"
    return build_result_record(
        git_commit="test-commit",
        run_id=run_id,
        task_family="editing",
        dataset="zsre",
        model="gpt2-xl",
        seed=seed,
        objective={"task": True, "ssr": ssr},
        distance_mapping=mapping,
        kernel=(
            {
                "family": "gaussian",
                "A_exc": 1.2,
                "A_inh": 0.9,
                "sigma_exc": 0.22,
                "sigma_inh": 0.6,
            }
            if ssr
            else {}
        ),
        metrics={"locality": 31.0},
        runtime={"elapsed_s": 1.0},
        config={"recipe": recipe},
        dataset_hash="dataset-v1",
        recipe=recipe,
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
        pairing_hash="paired-v1",
        teacher_protocol="locked_kd_only_trajectory_v1",
        teacher_trajectory_hash="d" * 64,
        selection_lock_hash="e" * 64,
    )


def _small_plan() -> dict:
    return {
        "cohorts": {
            "editing": {
                "task_family": "editing",
                "model": "gpt2-xl",
                "datasets": ["zsre"],
                "recipes": ["ssr_only", "plain"],
                "mappings": ["projective", "cosine"],
                "mapping_dependent_recipes": ["ssr_only"],
                "n_edits": 100,
                "required_metrics": ["locality"],
                "required_runtime": ["elapsed_s"],
                "development_seeds": [3],
                "confirmation_seeds": [13, 11],
            }
        }
    }


def _write_record(root: Path, name: str, record: dict) -> Path:
    path = root / name / "result_record.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


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
        ["kd_center", "kd"],
        ["kd_protodecor", "kd"],
        ["kd_spectral", "kd"],
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


def test_result_index_includes_reproducibility_identity_fields(tmp_path):
    record = _editing_record(
        recipe="ssr_only", mapping="cosine", seed=11, run_id="run-11"
    )
    path = _write_record(tmp_path, "run", record)

    manifest = build_manifest(tmp_path, _small_plan())

    item = manifest["complete_records"][0]
    assert item["path"] == str(path)
    assert item["git_commit"] == record["git_commit"]
    assert item["objective"] == record["objective"]
    assert item["kernel"] == record["kernel"]
    for field in (
        "data_offset",
        "n_edits",
        "status",
        "evaluator",
        "evaluator_version",
        "recipe",
        "distance_mapping",
        "config_hash",
        "dataset_hash",
        "pairing_hash",
        "model_hash",
        "evaluation_protocol_hash",
        "pairing_protocol_hash",
        "teacher_protocol",
        "teacher_trajectory_hash",
        "selection_lock_hash",
    ):
        assert item[field] == record[field]


def test_result_index_scans_persistent_failure_markers(tmp_path):
    failed = tmp_path / "failed_job"
    failed.mkdir()
    (failed / "FAILED").write_text("log=run.log\n", encoding="utf-8")
    (failed / "FAILED.json").write_text('{"returncode": 1}\n', encoding="utf-8")
    (failed / "FAILED.20260804_010203").write_text("1\n", encoding="utf-8")
    (failed / "FAILED.previous.20260804.json").write_text(
        '{"returncode": 2}\n', encoding="utf-8"
    )
    (failed / "status.tsv").write_text(
        "attempt-1\tRUNNING\n"
        "attempt-1\tFAILED\t70\trun.log\n"
        "attempt-2\tSUCCESS\t0\trun-2.log\n",
        encoding="utf-8",
    )

    manifest = build_manifest(tmp_path, _small_plan())

    markers = manifest["failure_markers"]
    assert len(markers) == 5
    assert [item["kind"] for item in markers].count("failure_file") == 3
    assert [item["kind"] for item in markers].count("archived_failure") == 1
    status = next(item for item in markers if item["kind"] == "failed_status_entry")
    assert status["line_number"] == 2
    assert status["columns"][1] == "FAILED"


def test_cohort_matrix_reports_expected_observed_and_missing_cells(tmp_path):
    _write_record(
        tmp_path,
        "plain-11",
        _editing_record(recipe="plain", mapping="none", seed=11, run_id="plain-11"),
    )
    _write_record(
        tmp_path,
        "ssr-11",
        _editing_record(
            recipe="ssr_only", mapping="cosine", seed=11, run_id="ssr-11"
        ),
    )
    _write_record(
        tmp_path,
        "development",
        _editing_record(
            recipe="ssr_only", mapping="projective", seed=3, run_id="screen-3"
        ),
    )

    manifest = build_manifest(tmp_path, _small_plan())

    matrix = manifest["cohort_matrices"]["editing"]
    assert matrix["seed_source"] == "confirmation_seeds"
    assert matrix["expected_count"] == 6
    assert matrix["observed_count"] == 2
    assert matrix["missing_count"] == 4
    assert matrix["development_observed_count"] == 1
    assert matrix["unexpected_count"] == 0
    assert [cell["seed"] for cell in matrix["expected_matrix"]] == [11, 13] * 3
    assert all("record_count" in cell for cell in matrix["observed_matrix"])
    missing_keys = {
        (cell["recipe"], cell["distance_mapping"], cell["seed"])
        for cell in matrix["missing_cells"]
    }
    assert missing_keys == {
        ("plain", "none", 13),
        ("ssr_only", "cosine", 13),
        ("ssr_only", "projective", 11),
        ("ssr_only", "projective", 13),
    }


def test_cohort_matrix_rejects_wrong_fixed_protocol(tmp_path):
    record = _editing_record(
        recipe="plain", mapping="none", seed=11, run_id="wrong-n-edits"
    )
    record["n_edits"] = 5
    _write_record(tmp_path, "wrong", record)

    matrix = build_manifest(tmp_path, _small_plan())["cohort_matrices"]["editing"]

    assert matrix["observed_count"] == 0
    assert matrix["protocol_mismatch_count"] == 1
    assert matrix["missing_count"] == 6
    assert "n_edits" in matrix["protocol_mismatches"][0]["errors"][0]


def test_cohort_matrix_rejects_conflicting_records_for_one_cell(tmp_path):
    first = _editing_record(
        recipe="plain", mapping="none", seed=11, run_id="first"
    )
    second = _editing_record(
        recipe="plain", mapping="none", seed=11, run_id="second"
    )
    _write_record(tmp_path, "first", first)
    _write_record(tmp_path, "second", second)

    matrix = build_manifest(tmp_path, _small_plan())["cohort_matrices"]["editing"]

    assert matrix["observed_count"] == 0
    assert matrix["conflicting_cell_count"] == 1
    assert matrix["missing_count"] == 6
