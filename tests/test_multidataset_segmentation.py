from __future__ import annotations

import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from ssr_utils.multidataset_segmentation import (
    flower_foreground,
    make_tasks,
    pet_foreground_and_valid,
)
from scripts.prepare_multidataset_segmentation import (
    DATASETS,
    HF_PET_REVISION,
    HF_PET_SHARDS,
    resolve_pet_source,
    safe_extract,
    validate_pet_trimap_payload,
)
from scripts.run_multidataset_segmentation import (
    CONFIRMATION_SEEDS,
    DATASETS as DRIVER_DATASETS,
    DEVELOPMENT_SEEDS,
    SPECS,
    confirmation_jobs,
    cache_path,
    screen_jobs,
    validate_paired_audit_fields,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_direct_segmentation_matrix_is_paired_and_complete(tmp_path):
    screen = screen_jobs(tmp_path)
    assert len(screen) == len(DRIVER_DATASETS) * len(DEVELOPMENT_SEEDS) * (
        1 + len(SPECS)
    )
    for dataset in DRIVER_DATASETS:
        for seed in DEVELOPMENT_SEEDS:
            subset = [job for job in screen if job.dataset == dataset and job.seed == seed]
            assert sum(job.method == "task_only" for job in subset) == 1
            assert sum(job.method == "task_ssr" for job in subset) == len(SPECS)

    confirm = confirmation_jobs(tmp_path, SPECS[0])
    assert len(confirm) == len(DRIVER_DATASETS) * len(CONFIRMATION_SEEDS) * 2
    assert {job.method for job in confirm} == {"task_only", "task_ssr"}


def test_partial_final_task_is_retained():
    pet_tasks = make_tasks(37, 5, 11)
    flower_tasks = make_tasks(102, 10, 13)
    assert [len(task) for task in pet_tasks] == [5] * 7 + [2]
    assert [len(task) for task in flower_tasks] == [10] * 10 + [2]
    assert sorted(class_id for task in pet_tasks for class_id in task) == list(range(37))
    assert sorted(class_id for task in flower_tasks for class_id in task) == list(range(102))


def test_flowers_blue_background_policy():
    pixels = np.asarray(
        [
            [[0, 0, 255], [3, 2, 249]],
            [[80, 20, 210], [220, 177, 36]],
        ],
        dtype=np.uint8,
    )
    mask = flower_foreground(Image.fromarray(pixels, mode="RGB"))
    assert mask.tolist() == [[False, False], [True, True]]


def test_pet_trimap_foreground_and_ignore_policy():
    trimap = Image.fromarray(np.asarray([[1, 2, 3]], dtype=np.uint8), mode="L")
    foreground, valid = pet_foreground_and_valid(trimap)
    assert foreground.tolist() == [[True, False, False]]
    assert valid.tolist() == [[True, True, False]]


def test_hf_pet_trimap_payload_enforces_three_value_semantics():
    trimap = Image.fromarray(np.asarray([[1, 2, 3]], dtype=np.uint8), mode="L")
    payload = io.BytesIO()
    trimap.save(payload, format="PNG")
    assert validate_pet_trimap_payload(payload.getvalue()) == (1, 2, 3)

    background_only = Image.fromarray(np.asarray([[2, 2]], dtype=np.uint8), mode="L")
    payload = io.BytesIO()
    background_only.save(payload, format="PNG")
    assert validate_pet_trimap_payload(payload.getvalue()) == (2,)

    invalid = Image.fromarray(np.asarray([[0, 1, 2]], dtype=np.uint8), mode="L")
    payload = io.BytesIO()
    invalid.save(payload, format="PNG")
    with pytest.raises(RuntimeError, match="trimap values"):
        validate_pet_trimap_payload(payload.getvalue())


def test_pet_source_auto_prefers_complete_local_mirror(tmp_path):
    assert resolve_pet_source("auto", tmp_path) == "official"
    for name in HF_PET_SHARDS:
        (tmp_path / name).touch()
    assert resolve_pet_source("auto", tmp_path) == "hf_parquet"
    assert resolve_pet_source("hf_parquet", tmp_path) == "hf_parquet"
    assert len(HF_PET_SHARDS) == 6
    assert len(HF_PET_REVISION) == 40


def test_pet_source_auto_rejects_partial_local_mirror(tmp_path):
    first = next(iter(HF_PET_SHARDS))
    (tmp_path / first).touch()
    with pytest.raises(RuntimeError, match="partial HF Pet mirror"):
        resolve_pet_source("auto", tmp_path)
    with pytest.raises(RuntimeError, match="shard inventory mismatch"):
        resolve_pet_source("hf_parquet", tmp_path)


def test_safe_extract_rejects_archive_traversal(tmp_path):
    archive = tmp_path / "unsafe.tgz"
    payload = b"outside"
    with tarfile.open(archive, "w:gz") as handle:
        member = tarfile.TarInfo("../outside.txt")
        member.size = len(payload)
        handle.addfile(member, io.BytesIO(payload))
    with pytest.raises(RuntimeError, match="escapes extraction root"):
        safe_extract(archive, tmp_path / "extract")


def test_manifest_only_requires_no_dataset_download(tmp_path):
    result_root = tmp_path / "manifest"
    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/run_multidataset_segmentation.py"),
            "--result-root",
            str(result_root),
            "--manifest-only",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    manifest = json.loads((result_root / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["datasets"] == list(DATASETS)
    assert manifest["primary_contrast"] == "task_ssr_minus_task_only"
    assert manifest["jobs"] == {"development": 36, "confirmation": 40, "total": 76}
    assert '"total": 76' in completed.stdout


def test_segmentation_pair_audit_requires_same_initialization_schedule_and_budget():
    shared = {
        "initial_model_hash": "a" * 64,
        "task_schedule_hash": "b" * 64,
        "optimizer_steps": 42,
        "encoder_state_sha256": "c" * 64,
        "feature_cache_sha256": "d" * 64,
    }
    validate_paired_audit_fields(shared, dict(shared), identity="pet/seed_1")
    treatment = dict(shared, optimizer_steps=43)
    with pytest.raises(RuntimeError, match="optimizer_steps"):
        validate_paired_audit_fields(shared, treatment, identity="pet/seed_1")


def test_segmentation_cache_path_uses_provenance_schema_version(tmp_path):
    assert cache_path(tmp_path, "oxford_iiit_pet", 160).name.endswith(
        "_provenance_v2.pt"
    )
