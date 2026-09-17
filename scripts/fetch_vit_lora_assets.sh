#!/usr/bin/env bash
# Run on a machine with Internet access, then stage DEST_ROOT on the cluster.
set -Eeuo pipefail

DEST_ROOT="${DEST_ROOT:?Set DEST_ROOT to the torchvision data directory}"
PRETRAINED_ROOT="${PRETRAINED_ROOT:-$(dirname "$DEST_ROOT")/pretrained}"
mkdir -p "$DEST_ROOT/flowers-102" "$DEST_ROOT/oxford-iiit-pet" "$PRETRAINED_ROOT"

download() {
  local url="$1" output="$2"
  curl -L --fail --retry 3 --continue-at - -o "$output" "$url"
}

FLOWERS_ROOT="$DEST_ROOT/flowers-102"
download \
  https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz \
  "$FLOWERS_ROOT/102flowers.tgz"
download \
  https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat \
  "$FLOWERS_ROOT/imagelabels.mat"
download \
  https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat \
  "$FLOWERS_ROOT/setid.mat"
[[ -d "$FLOWERS_ROOT/jpg" ]] || tar -xzf "$FLOWERS_ROOT/102flowers.tgz" -C "$FLOWERS_ROOT"

MATERIALIZED_PETS="$DEST_ROOT/oxford_iiit_pet/annotations/hf_parquet_rows.jsonl"
if [[ -s "$MATERIALIZED_PETS" ]]; then
  printf 'Reusing materialized Oxford Pet payload: %s\n' "$MATERIALIZED_PETS"
else
  PETS_ROOT="$DEST_ROOT/oxford-iiit-pet"
  PETS_ARCHIVE_URL="https://thor.robots.ox.ac.uk/pets"
  download \
    "$PETS_ARCHIVE_URL/images.tar.gz" \
    "$PETS_ROOT/images.tar.gz"
  download \
    "$PETS_ARCHIVE_URL/annotations.tar.gz" \
    "$PETS_ROOT/annotations.tar.gz"
  [[ -d "$PETS_ROOT/images" ]] || tar -xzf "$PETS_ROOT/images.tar.gz" -C "$PETS_ROOT"
  [[ -d "$PETS_ROOT/annotations" ]] || tar -xzf "$PETS_ROOT/annotations.tar.gz" -C "$PETS_ROOT"
fi

WEIGHT="$PRETRAINED_ROOT/vit_tiny_patch16_224_augreg_in21k_ft_in1k.safetensors"
download \
  https://huggingface.co/timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k/resolve/main/model.safetensors \
  "$WEIGHT"

"${PYTHON:-python3}" - "$DEST_ROOT" "$WEIGHT" <<'PY'
import hashlib
import sys
from pathlib import Path

root, weight = map(Path, sys.argv[1:])
expected = "fecf81b492bd13ee7a5297cb74d1d417aac8bf7e1b7d96aed89c4691984587ed"
actual = hashlib.sha256(weight.read_bytes()).hexdigest()
if actual != expected:
    raise SystemExit(f"ViT checkpoint checksum mismatch: {actual}")
required = [
    root / "flowers-102/jpg",
    root / "flowers-102/imagelabels.mat",
    root / "flowers-102/setid.mat",
]
materialized = root / "oxford_iiit_pet/annotations/hf_parquet_rows.jsonl"
if materialized.is_file():
    required.extend(
        [
            root / "oxford_iiit_pet/images",
            root / "oxford_iiit_pet/annotations/hf_label_names.json",
        ]
    )
else:
    required.extend(
        [
            root / "oxford-iiit-pet/images",
            root / "oxford-iiit-pet/annotations/trainval.txt",
            root / "oxford-iiit-pet/annotations/test.txt",
        ]
    )
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit(f"Missing staged assets: {missing}")
md5s = {
    root / "flowers-102/102flowers.tgz": "52808999861908f626f3c1f4e79d11fa",
    root / "flowers-102/imagelabels.mat": "e0620be6f572b9609742df49c70aed4d",
    root / "flowers-102/setid.mat": "a5357ecc9cb78c4bef273ce3793fc85c",
}
if not materialized.is_file():
    md5s.update(
        {
            root / "oxford-iiit-pet/images.tar.gz": "5c4f3ee8e5d25df40f4fd59a7f44e54c",
            root / "oxford-iiit-pet/annotations.tar.gz": "95a8c909bbe2e81eed6a22bccdf3f68f",
        }
    )
for path, checksum in md5s.items():
    hasher = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    digest = hasher.hexdigest()
    if digest != checksum:
        raise SystemExit(f"Dataset archive checksum mismatch for {path}: {digest}")
print(f"Assets ready under {root}; checkpoint_sha256={actual}")
PY
