from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from datasets.materialized_oxford_pet import (
    MaterializedOxfordIIITPet,
    find_materialized_root,
)


def build_materialized_root(tmp_path: Path) -> Path:
    root = tmp_path / "oxford_iiit_pet"
    (root / "images").mkdir(parents=True)
    (root / "annotations").mkdir()
    rows = []
    for index, split in enumerate(("train", "test")):
        stem = f"hf_{split}_00_{index:06d}"
        Image.new("RGB", (5, 4), color=(index * 20, 10, 30)).save(
            root / "images" / f"{stem}.png"
        )
        rows.append(
            {
                "image": f"images/{stem}.jpg",
                "label": index,
                "split": split,
                "stem": stem,
            }
        )
    with (root / "annotations/hf_parquet_rows.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    (root / "annotations/hf_label_names.json").write_text(
        json.dumps({"label": [f"class_{index}" for index in range(37)]}),
        encoding="utf-8",
    )
    return root


def test_materialized_pet_reuses_underscore_root_and_png_alternative(tmp_path):
    root = build_materialized_root(tmp_path)
    assert find_materialized_root(tmp_path) == root
    train = MaterializedOxfordIIITPet(tmp_path, split="trainval")
    test = MaterializedOxfordIIITPet(tmp_path, split="test")

    assert len(train) == len(test) == 1
    assert train.samples[0][0].endswith(".png")
    assert train.samples[0][1] == 0
    image, label = test[0]
    assert image.mode == "RGB"
    assert image.size == (5, 4)
    assert label == 1
    assert len(test.classes) == 37


def test_materialized_pet_rejects_unknown_split(tmp_path):
    build_materialized_root(tmp_path)
    with pytest.raises(ValueError, match="split must be"):
        MaterializedOxfordIIITPet(tmp_path, split="validation")
