from __future__ import annotations

from scripts.run_meeting_segmentation_ablation import (
    CONFIRMATION_SEEDS,
    DEVELOPMENT_SEEDS,
    confirmation_jobs,
    screen_jobs,
)
from ssr_utils.dataset_fingerprint import cub_dataset_fingerprint


def test_segmentation_matrix_sizes(tmp_path):
    screen = screen_jobs(tmp_path)
    assert len(screen) == 21
    assert {job.seed for job in screen} == set(DEVELOPMENT_SEEDS)
    confirm = confirmation_jobs(tmp_path, screen[1].spec)
    assert len(confirm) == 40
    assert {job.seed for job in confirm} == set(CONFIRMATION_SEEDS)
    assert {job.method for job in confirm} == {"baseline", "kd", "biocs", "biocs_kd"}


def test_segmentation_fingerprint_includes_source_and_cache_hashes(tmp_path):
    metadata = tmp_path / "CUB_200_2011"
    metadata.mkdir()
    (metadata / "images.txt").write_text("1 image.jpg\n", encoding="utf-8")

    first = cub_dataset_fingerprint(
        tmp_path,
        segmentation_source_sha256="mask-v1",
        segmentation_cache_sha256="cache-v1",
    )
    second = cub_dataset_fingerprint(
        tmp_path,
        segmentation_source_sha256="mask-v1",
        segmentation_cache_sha256="cache-v2",
    )

    assert first != second
