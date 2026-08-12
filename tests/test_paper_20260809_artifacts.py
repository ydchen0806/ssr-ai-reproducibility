from __future__ import annotations

import copy
import json

import pytest

from scripts.verify_paper_20260809 import (
    DEFAULT_ARTIFACT_DIR,
    PET_METRICS,
    VerificationError,
    verify_artifacts,
    verify_pet,
)


def test_pet_archive_is_exact_but_not_manuscript_evidence():
    report = verify_artifacts()
    pet = report["pet_vit_lora"]

    assert pet["n"] == 30
    assert pet["role"] == "invalidated_archive_not_manuscript_evidence"
    assert pet["selected"]["candidate_id"] == "r16_joint_c0.006_a0.0012"
    assert set(pet["archived_summary"]) == set(PET_METRICS)
    assert pet["archived_summary"]["avg_accuracy"]["ci95"][0] > 0.0


def test_nonconfirmatory_endpoints_remain_explicit_boundary_evidence():
    report = verify_artifacts()

    cil = report["pet_vit_lora"]["cil_last_boundary"]
    assert cil["manuscript_endpoint"] is False
    assert cil["crosses_zero"] is True
    assert cil["ci95"][0] < 0.0 < cil["ci95"][1]

    segmentation = report["segmentation_direct"]
    assert segmentation["role"] == "boundary_audit"
    assert segmentation["all_primary_gates"] is False
    assert all(record["primary_gate"] is False for record in segmentation["datasets"].values())
    assert all(record["ci95"][0] < 0.0 < record["ci95"][1] for record in segmentation["datasets"].values())
    cub_geometry = segmentation["datasets"]["cub200"]["geometry"]
    assert cub_geometry["effective_rank"]["ci95"][0] > 0.0
    assert cub_geometry["mean_abs_offdiag_cosine"]["ci95"][0] > 0.0


def test_verifier_rejects_a_tampered_paired_row():
    payload = json.loads(
        (DEFAULT_ARTIFACT_DIR / "pet_vit_lora_confirmation.json").read_text(encoding="utf-8")
    )
    tampered = copy.deepcopy(payload)
    tampered["rows"][0]["avg_accuracy_treatment"] += 1.0

    with pytest.raises(VerificationError, match="Pet summary mismatch"):
        verify_pet(tampered)
