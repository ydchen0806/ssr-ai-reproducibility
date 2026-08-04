#!/usr/bin/env python3
"""Prepare and fingerprint the Oxford Pet and Flowers segmentation datasets.

Flowers uses official Oxford VGG sources. Pet can use either the official
archives or a hash-locked local Parquet mirror of the Hugging Face conversion.
Source artifacts are retained outside version control so every training result
can be bound to their SHA256 digests.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tarfile
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("oxford_iiit_pet", "oxford_flowers102")
PET_SOURCES = ("auto", "official", "hf_parquet")
HF_PET_REPO_URL = "https://huggingface.co/datasets/dpdl-benchmark/oxford_iiit_pet"
HF_PET_REVISION = "18515194612b584bf54692372ac035ef773a5b41"
HF_PET_SHARDS = {
    "test-00000-of-00003.parquet": {
        "size": 385_333_214,
        "sha256": "cc4a6ca162c95b297116a4462ff7accf8a7f8ae3ce123d9be56d4c91c00380fa",
    },
    "test-00001-of-00003.parquet": {
        "size": 386_004_143,
        "sha256": "c5fb1c1cff8c27809d11f827819088464311a17bf325703a39334031b28af280",
    },
    "test-00002-of-00003.parquet": {
        "size": 392_854_014,
        "sha256": "65d370e76ca49a131ed874053ac372828303ef771ba4c961878faf4870217fda",
    },
    "train-00000-of-00003.parquet": {
        "size": 381_518_702,
        "sha256": "473606455eb66b1ee9b2a8675e41bf8c7494370fc27eccc5eae5890f882b7442",
    },
    "train-00001-of-00003.parquet": {
        "size": 369_025_027,
        "sha256": "3bc7e8a62a1756044d1ecc4750a246afda63d0e8ac8042f43789e661c87036a8",
    },
    "train-00002-of-00003.parquet": {
        "size": 362_885_962,
        "sha256": "c75713efefddd8e7d692e666eb76072e58db9649c7c2fdccfbae05fba2acd7a3",
    },
}


@dataclass(frozen=True)
class Source:
    name: str
    url: str
    size: int
    archive: bool = False


SOURCES = {
    "oxford_iiit_pet": (
        Source(
            "images.tar.gz",
            "https://thor.robots.ox.ac.uk/~vgg/data/pets/images.tar.gz",
            791_918_971,
            True,
        ),
        Source(
            "annotations.tar.gz",
            "https://thor.robots.ox.ac.uk/~vgg/data/pets/annotations.tar.gz",
            19_173_078,
            True,
        ),
    ),
    "oxford_flowers102": (
        Source(
            "102flowers.tgz",
            "https://thor.robots.ox.ac.uk/flowers/102/102flowers.tgz",
            344_862_509,
            True,
        ),
        Source(
            "102segmentations.tgz",
            "https://thor.robots.ox.ac.uk/flowers/102/102segmentations.tgz",
            203_577_493,
            True,
        ),
        Source(
            "imagelabels.mat",
            "https://thor.robots.ox.ac.uk/flowers/102/imagelabels.mat",
            502,
        ),
        Source(
            "setid.mat",
            "https://thor.robots.ox.ac.uk/flowers/102/setid.mat",
            14_989,
        ),
    ),
}


DATASET_NOTES = {
    "oxford_iiit_pet": {
        "official_page": "https://www.robots.ox.ac.uk/~vgg/data/pets/",
        "license": "CC BY-SA 4.0; image copyright remains with original owners",
        "citation": "Parkhi et al., Cats and Dogs, CVPR 2012",
        "mask_policy": (
            "trimap value 1 is foreground, value 2 is background, and value 3 "
            "is an ignored border/void region"
        ),
    },
    "oxford_flowers102": {
        "official_page": "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/",
        "license": (
            "The official dataset page provides research downloads but does not "
            "state a redistribution licence. Review Oxford VGG terms before use "
            "and do not redistribute the source archives."
        ),
        "citation": "Nilsback and Zisserman, ICVGIP 2008; BMVC 2007",
        "mask_policy": (
            "official segmim JPEG; foreground is separated from the nominal "
            "RGB(0,0,255) background with Euclidean threshold 40"
        ),
    },
}

EXTRACTION_SENTINELS = {
    "oxford_iiit_pet": {
        "images.tar.gz": ("images",),
        "annotations.tar.gz": ("annotations/trainval.txt", "annotations/trimaps"),
    },
    "oxford_flowers102": {
        "102flowers.tgz": ("jpg",),
        "102segmentations.tgz": ("segmim",),
    },
}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def download_resumable(source: Source, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size == source.size:
        return
    if destination.exists() and destination.stat().st_size != source.size:
        raise RuntimeError(
            f"Existing source has unexpected size: {destination} "
            f"({destination.stat().st_size} != {source.size})"
        )

    partial = destination.with_name(destination.name + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > source.size:
        raise RuntimeError(f"Partial download is larger than its source: {partial}")
    headers = {"User-Agent": "SSR-reproducibility/1.0"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(source.url, headers=headers)
    print(f"[download] {source.url} -> {destination} (resume={offset})", flush=True)
    with urllib.request.urlopen(request, timeout=120) as response:
        status = getattr(response, "status", response.getcode())
        if offset and status != 206:
            raise RuntimeError(
                f"Server did not honor resume request for {source.url}; "
                f"move {partial} aside and retry"
            )
        mode = "ab" if offset else "wb"
        with partial.open(mode) as handle:
            while True:
                chunk = response.read(4 * 1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
    observed = partial.stat().st_size
    if observed != source.size:
        raise RuntimeError(
            f"Incomplete source download: {partial} ({observed} != {source.size})"
        )
    partial.replace(destination)


def safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(archive, "r:gz") as handle:
        members = handle.getmembers()
        for member in members:
            target = (destination / member.name).resolve()
            if target != root and root not in target.parents:
                raise RuntimeError(f"Archive path escapes extraction root: {member.name}")
            if member.issym() or member.islnk():
                raise RuntimeError(f"Archive links are not accepted: {member.name}")
        handle.extractall(destination, members=members)


def resolve_pet_source(requested: str, parquet_root: Path) -> str:
    if requested not in PET_SOURCES:
        raise ValueError(f"Unknown Pet source: {requested}")
    present = {path.name for path in parquet_root.glob("*.parquet")} if parquet_root.is_dir() else set()
    expected = set(HF_PET_SHARDS)
    if requested == "official":
        return requested
    if requested == "hf_parquet":
        missing = sorted(expected - present)
        unexpected = sorted(present - expected)
        if missing or unexpected:
            raise RuntimeError(
                "HF Pet shard inventory mismatch: "
                f"missing={missing}, unexpected={unexpected}, root={parquet_root}"
            )
        return requested
    if present:
        missing = sorted(expected - present)
        unexpected = sorted(present - expected)
        if missing or unexpected:
            raise RuntimeError(
                "Auto-detected a partial HF Pet mirror; refusing silent fallback: "
                f"missing={missing}, unexpected={unexpected}, root={parquet_root}"
            )
        return "hf_parquet"
    return "official"


def verify_hf_pet_shards(parquet_root: Path) -> list[dict]:
    rows = []
    for name, expected in sorted(HF_PET_SHARDS.items()):
        path = parquet_root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        observed_size = path.stat().st_size
        if observed_size != expected["size"]:
            raise RuntimeError(
                f"HF Pet shard size mismatch: {path} "
                f"({observed_size} != {expected['size']})"
            )
        observed_sha256 = sha256_file(path)
        if observed_sha256 != expected["sha256"]:
            raise RuntimeError(
                f"HF Pet shard SHA256 mismatch: {path} "
                f"({observed_sha256} != {expected['sha256']})"
            )
        rows.append(
            {
                "name": name,
                "url": f"{HF_PET_REPO_URL}/resolve/{HF_PET_REVISION}/data/{name}",
                "size": observed_size,
                "archive": False,
                "path": str(path.resolve()),
                "sha256": observed_sha256,
                "lfs_oid": expected["sha256"],
            }
        )
    return rows


def _feature_names(parquet_file: object) -> tuple[list[str], list[str]]:
    metadata = parquet_file.schema_arrow.metadata or {}
    try:
        payload = json.loads(metadata[b"huggingface"])
        features = payload["info"]["features"]
        labels = list(features["label"]["names"])
        species = list(features["species"]["names"])
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise RuntimeError("HF Pet Parquet lacks expected Hugging Face feature metadata") from error
    if len(labels) != 37 or species != ["Cat", "Dog"]:
        raise RuntimeError(
            f"Unexpected HF Pet labels/species: labels={len(labels)}, species={species}"
        )
    return labels, species


def _payload_bytes(cell: object, field: str) -> tuple[bytes, str | None]:
    if not isinstance(cell, dict):
        raise RuntimeError(f"HF Pet {field} cell is not a struct")
    payload = cell.get("bytes")
    source_path = cell.get("path")
    if isinstance(payload, memoryview):
        payload = payload.tobytes()
    if not isinstance(payload, bytes) or not payload:
        raise RuntimeError(f"HF Pet {field} cell does not contain embedded bytes")
    if source_path is not None and not isinstance(source_path, str):
        raise RuntimeError(f"HF Pet {field}.path is not a string or null")
    return payload, source_path


def _image_extension(payload: bytes) -> str:
    if payload.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if payload.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    raise RuntimeError("HF Pet image payload is neither PNG nor JPEG")


def validate_pet_trimap_payload(payload: bytes) -> tuple[int, ...]:
    from PIL import Image

    with Image.open(io.BytesIO(payload)) as image:
        histogram = image.convert("L").histogram()
        values = tuple(value for value, count in enumerate(histogram) if count)
    if not values or not set(values) <= {1, 2, 3}:
        raise RuntimeError(f"Unexpected HF Pet trimap values: {values}")
    return values


def _write_payload(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        if path.stat().st_size != len(payload) or sha256_file(path) != hashlib.sha256(payload).hexdigest():
            raise RuntimeError(f"Existing materialized Pet payload differs: {path}")
        return
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_bytes(payload)
    temporary.replace(path)


def materialize_hf_pet(
    parquet_root: Path,
    destination: Path,
    source_rows: list[dict],
) -> dict:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError(
            "HF Parquet Pet preparation requires pyarrow at execution time; "
            "install pyarrow or use --pet-source official"
        ) from error

    images_root = destination / "images"
    annotations_root = destination / "annotations"
    trimaps_root = annotations_root / "trimaps"
    images_root.mkdir(parents=True, exist_ok=True)
    trimaps_root.mkdir(parents=True, exist_ok=True)

    split_lines = {"train": [], "test": []}
    list_lines = []
    provenance_rows = []
    reference_labels = reference_species = None
    split_counts = {"train": 0, "test": 0}
    trimap_values = set()

    for source in source_rows:
        name = source["name"]
        split = "train" if name.startswith("train-") else "test"
        shard_index = int(name.split("-")[1])
        parquet_file = pq.ParquetFile(parquet_root / name)
        labels, species_names = _feature_names(parquet_file)
        if reference_labels is None:
            reference_labels, reference_species = labels, species_names
        elif labels != reference_labels or species_names != reference_species:
            raise RuntimeError(f"HF Pet feature metadata differs across shards: {name}")

        row_index = 0
        for batch in parquet_file.iter_batches(
            batch_size=32,
            columns=("image", "label", "species", "segmentation_mask"),
        ):
            for row in batch.to_pylist():
                label = int(row["label"])
                species = int(row["species"])
                if label not in range(37) or species not in (0, 1):
                    raise RuntimeError(
                        f"HF Pet label/species out of range in {name}:{row_index}: "
                        f"label={label}, species={species}"
                    )
                image_bytes, image_source_path = _payload_bytes(row["image"], "image")
                mask_bytes, mask_source_path = _payload_bytes(
                    row["segmentation_mask"], "segmentation_mask"
                )
                observed_values = validate_pet_trimap_payload(mask_bytes)
                trimap_values.update(observed_values)
                stem = f"hf_{split}_{shard_index:02d}_{row_index:06d}"
                image_relative = Path("images") / f"{stem}{_image_extension(image_bytes)}"
                mask_relative = Path("annotations") / "trimaps" / f"{stem}.png"
                _write_payload(destination / image_relative, image_bytes)
                _write_payload(destination / mask_relative, mask_bytes)
                fields = f"{stem} {label + 1} {species + 1} {label + 1}"
                split_lines[split].append(fields)
                list_lines.append(fields)
                provenance_rows.append(
                    {
                        "stem": stem,
                        "split": split,
                        "shard": name,
                        "row_index": row_index,
                        "label": label,
                        "label_name": labels[label],
                        "species": species,
                        "species_name": species_names[species],
                        "image": str(image_relative),
                        "segmentation_mask": str(mask_relative),
                        "source_image_path": image_source_path,
                        "source_mask_path": mask_source_path,
                    }
                )
                row_index += 1
                split_counts[split] += 1
        if row_index != parquet_file.metadata.num_rows:
            raise RuntimeError(
                f"HF Pet shard row count mismatch: {name} ({row_index} != {parquet_file.metadata.num_rows})"
            )

    if split_counts != {"train": 3680, "test": 3669}:
        raise RuntimeError(f"HF Pet split counts are incorrect: {split_counts}")
    if trimap_values != {1, 2, 3}:
        raise RuntimeError(f"HF Pet aggregate trimap values are incorrect: {sorted(trimap_values)}")

    (annotations_root / "trainval.txt").write_text(
        "\n".join(split_lines["train"]) + "\n", encoding="utf-8"
    )
    (annotations_root / "test.txt").write_text(
        "\n".join(split_lines["test"]) + "\n", encoding="utf-8"
    )
    (annotations_root / "list.txt").write_text(
        "\n".join(list_lines) + "\n", encoding="utf-8"
    )
    write_json_atomic(
        annotations_root / "hf_label_names.json",
        {"label": reference_labels, "species": reference_species},
    )
    provenance_path = annotations_root / "hf_parquet_rows.jsonl"
    temporary = provenance_path.with_name(f".{provenance_path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in provenance_rows),
        encoding="utf-8",
    )
    temporary.replace(provenance_path)
    return {
        "repo_url": HF_PET_REPO_URL,
        "revision": HF_PET_REVISION,
        "split_counts": split_counts,
        "trimap_values": sorted(trimap_values),
        "label_count": len(reference_labels or []),
        "species_names": reference_species,
        "payload_policy": "original Parquet image and segmentation_mask bytes",
    }


def inventory(root: Path, directories: Iterable[str], suffixes: set[str]) -> list[dict]:
    rows = []
    for directory in directories:
        base = root / directory
        if not base.is_dir():
            raise FileNotFoundError(base)
        for path in sorted(base.rglob("*")):
            if path.is_file() and path.suffix.lower() in suffixes:
                rows.append(
                    {
                        "path": str(path.relative_to(root)),
                        "size": path.stat().st_size,
                    }
                )
    return rows


def validate_layout(dataset: str, root: Path) -> list[dict]:
    if dataset == "oxford_iiit_pet":
        required = (
            root / "annotations/trainval.txt",
            root / "annotations/test.txt",
            root / "annotations/list.txt",
        )
        for path in required:
            if not path.is_file():
                raise FileNotFoundError(path)
        rows = inventory(root, ("images",), {".jpg", ".jpeg", ".png"})
        masks = inventory(root, ("annotations/trimaps",), {".png"})
        if len(rows) != 7_349 or len(masks) != 7_349:
            raise RuntimeError(
                f"Oxford Pet extraction is incomplete: images={len(rows)}, masks={len(masks)}"
            )
        return rows + masks

    for path in (root / "imagelabels.mat", root / "setid.mat"):
        if not path.is_file():
            raise FileNotFoundError(path)
    rows = inventory(root, ("jpg",), {".jpg"})
    masks = inventory(root, ("segmim",), {".jpg"})
    if len(rows) != 8_189 or len(masks) != 8_189:
        raise RuntimeError(
            f"Oxford Flowers extraction is incomplete: images={len(rows)}, masks={len(masks)}"
        )
    return rows + masks


def metadata_hashes(dataset: str, root: Path) -> dict[str, str]:
    if dataset == "oxford_iiit_pet":
        names = [
            "annotations/trainval.txt",
            "annotations/test.txt",
            "annotations/list.txt",
        ]
        for optional in (
            "annotations/hf_label_names.json",
            "annotations/hf_parquet_rows.jsonl",
        ):
            if (root / optional).is_file():
                names.append(optional)
    else:
        names = ("imagelabels.mat", "setid.mat")
    return {name: sha256_file(root / name) for name in names}


def prepare_dataset(
    dataset: str,
    data_root: Path,
    *,
    pet_source: str = "auto",
    pet_parquet_root: Path | None = None,
) -> dict:
    root = data_root / dataset
    if dataset == "oxford_flowers102":
        legacy_root = data_root / "flowers-102"
        if not root.exists() and legacy_root.is_dir():
            root = legacy_root
    source_root = root / "sources"
    source_root.mkdir(parents=True, exist_ok=True)
    source_rows = []
    source_kind = "official"
    source_provenance = None
    if dataset == "oxford_iiit_pet":
        parquet_root = (
            pet_parquet_root.resolve()
            if pet_parquet_root is not None
            else (data_root / "oxford_hf_parquet" / "data").resolve()
        )
        source_kind = resolve_pet_source(pet_source, parquet_root)
        if source_kind == "hf_parquet":
            source_rows = verify_hf_pet_shards(parquet_root)
            source_provenance = materialize_hf_pet(parquet_root, root, source_rows)

    if source_kind == "official":
        for source in SOURCES[dataset]:
            root_source = root / source.name
            destination = (
                root_source
                if root_source.is_file() and root_source.stat().st_size == source.size
                else source_root / source.name
                if source.archive
                else root_source
            )
            download_resumable(source, destination)
            source_hash = sha256_file(destination)
            source_rows.append(
                {
                    **asdict(source),
                    "path": str(destination.resolve()),
                    "sha256": source_hash,
                }
            )
            if source.archive:
                marker = source_root / f".{source.name}.extracted.sha256"
                sentinels = [
                    root / item for item in EXTRACTION_SENTINELS[dataset][source.name]
                ]
                marker_matches = (
                    marker.is_file()
                    and marker.read_text(encoding="utf-8").strip() == source_hash
                    and all(path.exists() for path in sentinels)
                )
                if not marker_matches:
                    print(f"[extract] {destination} -> {root}", flush=True)
                    safe_extract(destination, root)
                    marker.write_text(source_hash + "\n", encoding="utf-8")

    files = validate_layout(dataset, root)
    fingerprint_payload = {
        "dataset": dataset,
        "sources": [{"name": row["name"], "sha256": row["sha256"]} for row in source_rows],
        "source_kind": source_kind,
        "source_provenance": source_provenance,
        "metadata": metadata_hashes(dataset, root),
        "inventory": files,
        "mask_policy": DATASET_NOTES[dataset]["mask_policy"],
    }
    manifest = {
        "schema_version": 1,
        "dataset": dataset,
        "root": str(root.resolve()),
        **DATASET_NOTES[dataset],
        "source_kind": source_kind,
        "source_provenance": source_provenance,
        "sources": source_rows,
        "inventory_count": len(files),
        "inventory_hash": canonical_sha256(files),
        "metadata": fingerprint_payload["metadata"],
        "dataset_fingerprint": canonical_sha256(fingerprint_payload),
    }
    write_json_atomic(root / "DATASET_MANIFEST.json", manifest)
    if dataset == "oxford_flowers102" and root.name == "flowers-102":
        alias = data_root / dataset
        if alias.is_symlink() and alias.resolve() != root.resolve():
            raise RuntimeError(f"Conflicting Oxford Flowers alias: {alias}")
        if not alias.exists():
            alias.symlink_to(root.name, target_is_directory=True)
    print(
        f"[ready] {dataset}: {manifest['dataset_fingerprint']} "
        f"({manifest['inventory_count']} files)",
        flush=True,
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--pet-source", choices=PET_SOURCES, default="auto")
    parser.add_argument("--pet-parquet-root", type=Path, default=None)
    args = parser.parse_args()
    manifests = [
        prepare_dataset(
            dataset,
            args.data_root.resolve(),
            pet_source=args.pet_source,
            pet_parquet_root=args.pet_parquet_root,
        )
        for dataset in args.datasets
    ]
    print(json.dumps(manifests, indent=2))


if __name__ == "__main__":
    main()
