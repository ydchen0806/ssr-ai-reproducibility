"""Dataset fingerprints that do not import model-training dependencies."""

from __future__ import annotations

from pathlib import Path

from ssr_utils.result_schema import sha256_file, sha256_value


def cub_dataset_fingerprint(
    data_root: Path,
    *,
    segmentation_source_sha256: str | None = None,
    segmentation_cache_sha256: str | None = None,
) -> str:
    cub_dir = data_root / "CUB_200_2011"
    files = [
        cub_dir / "images.txt",
        cub_dir / "image_class_labels.txt",
        cub_dir / "train_test_split.txt",
        cub_dir / "classes.txt",
    ]
    present = {
        str(path.relative_to(data_root)): sha256_file(path)
        for path in files
        if path.is_file()
    }
    if not present:
        raise FileNotFoundError(f"CUB metadata files are missing under {cub_dir}")
    if segmentation_source_sha256 is not None:
        present["segmentations_source_sha256"] = segmentation_source_sha256
    if segmentation_cache_sha256 is not None:
        present["segmentation_cache_sha256"] = segmentation_cache_sha256
    return sha256_value(present)
