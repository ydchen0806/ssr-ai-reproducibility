"""Read the locally materialized Oxford-IIIT Pet HF-Parquet payload."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


MANIFEST_NAME = "annotations/hf_parquet_rows.jsonl"


def find_materialized_root(data_root: str | Path) -> Path | None:
    root = Path(data_root)
    candidates = [root, root / "oxford_iiit_pet"]
    for candidate in candidates:
        if (candidate / MANIFEST_NAME).is_file():
            return candidate
    return None


def resolve_payload_path(root: Path, relative: str, stem: str | None = None) -> Path:
    declared = root / relative
    if declared.is_file():
        return declared
    base = declared.with_suffix("")
    candidates = [base.with_suffix(extension) for extension in (".png", ".jpg", ".jpeg")]
    if stem:
        candidates.extend(
            root / "images" / f"{stem}{extension}"
            for extension in (".png", ".jpg", ".jpeg")
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"No image payload for declared path {relative!r}; tried "
        f"{[str(path) for path in candidates]}"
    )


class MaterializedOxfordIIITPet(Dataset):
    """Classification view over materialized PNG/JPEG payloads without copying."""

    def __init__(self, root: str | Path, split: str, transform=None):
        if split not in {"train", "trainval", "test"}:
            raise ValueError("split must be train, trainval, or test")
        materialized = find_materialized_root(root)
        if materialized is None:
            raise FileNotFoundError(
                f"No materialized Oxford Pet manifest below {Path(root)}"
            )
        requested = "train" if split == "trainval" else split
        self.root = materialized
        self.transform = transform
        self.samples: list[tuple[str, int]] = []
        self._labels: list[int] = []
        manifest = materialized / MANIFEST_NAME
        with manifest.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("split") != requested:
                    continue
                label = int(row["label"])
                if not 0 <= label < 37:
                    raise ValueError(f"Invalid label {label} at {manifest}:{line_number}")
                image_path = resolve_payload_path(
                    materialized,
                    str(row["image"]),
                    str(row.get("stem", "")) or None,
                )
                self.samples.append((str(image_path), label))
                self._labels.append(label)
        if not self.samples:
            raise RuntimeError(f"No {requested} rows in {manifest}")

        label_path = materialized / "annotations/hf_label_names.json"
        if label_path.is_file():
            labels = json.loads(label_path.read_text(encoding="utf-8"))["label"]
            if len(labels) != 37:
                raise ValueError(f"Expected 37 label names in {label_path}")
            self.classes = [str(label) for label in labels]
        else:
            self.classes = [str(index) for index in range(37)]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label
