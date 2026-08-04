#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
TARGET_DIR="${TARGET_DIR:-$PROJECT_ROOT/dataset/knowedit/benchmark/WikiBio}"
BASE_URL="https://huggingface.co/datasets/zjunlp/KnowEdit/resolve/main/benchmark/WikiBio"

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

fetch_one() {
  local name="$1"
  local expected="$2"
  local destination="$TARGET_DIR/$name"
  if [[ -f "$destination" ]]; then
    local observed
    observed="$(sha256_file "$destination")"
    [[ "$observed" == "$expected" ]] || {
      printf 'Existing dataset hash mismatch: %s\nexpected=%s\nobserved=%s\n' \
        "$destination" "$expected" "$observed" >&2
      return 1
    }
    printf 'Verified existing %s\n' "$destination"
    return
  fi

  mkdir -p "$TARGET_DIR"
  local temporary
  temporary="$(mktemp "$TARGET_DIR/.${name}.tmp.XXXXXX")"
  trap 'rm -f "$temporary"' RETURN
  curl --fail --location --retry 4 --retry-delay 2 \
    "$BASE_URL/$name?download=true" --output "$temporary"
  local observed
  observed="$(sha256_file "$temporary")"
  [[ "$observed" == "$expected" ]] || {
    printf 'Downloaded dataset hash mismatch: %s\nexpected=%s\nobserved=%s\n' \
      "$name" "$expected" "$observed" >&2
    return 1
  }
  mv "$temporary" "$destination"
  trap - RETURN
  printf 'Downloaded and verified %s\n' "$destination"
}

fetch_one wikibio-test-all.json \
  00933f69b47d8281d01f1f4b84f3f266b7482a65a7d74022e660a8a9ca2264a4
fetch_one wikibio-train-all.json \
  6ee88049c1c6c00d6f81d2d942350b6ec78b45917d2ec8a3eec3eafa89c2e209
