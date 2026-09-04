#!/usr/bin/env python3
"""Create fixed, non-overlapping KnowEdit streams for the KE history study."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


DEVELOPMENT_SEEDS = (3718, 4141)
CONFIRMATION_SEEDS = tuple(range(201, 211))
CONFIRMATION_EDITS = 100
CONFIRMATION_CONTROLS = 128
PROTOCOL = "knowedit_semantic_history_stream_v2"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-label")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--development-edits",
        type=int,
        default=250,
        help="Number of edits in each fixed development stream (default: 250).",
    )
    parser.add_argument(
        "--development-seeds",
        type=int,
        nargs="+",
        default=list(DEVELOPMENT_SEEDS),
        help="Fixed seeds for development streams (default: 3718 4141).",
    )
    parser.add_argument(
        "--confirmation-seeds",
        type=int,
        nargs="+",
        default=list(CONFIRMATION_SEEDS),
        help="Fixed seeds for confirmation streams (default: 201 through 210).",
    )
    parser.add_argument(
        "--confirmation-edits",
        type=int,
        default=CONFIRMATION_EDITS,
        help="Number of edits in each untouched confirmation stream (default: 100).",
    )
    args = parser.parse_args()
    dataset_label = args.dataset_label or args.dataset.stem
    if not dataset_label.strip():
        raise SystemExit("--dataset-label must not be empty")

    development_seeds = tuple(args.development_seeds)
    confirmation_seeds = tuple(args.confirmation_seeds)
    if not development_seeds or not confirmation_seeds:
        raise SystemExit("Both development and confirmation seed lists must be non-empty")
    if len(development_seeds) != len(set(development_seeds)):
        raise SystemExit("Development stream seeds must be unique")
    if len(confirmation_seeds) != len(set(confirmation_seeds)):
        raise SystemExit("Confirmation stream seeds must be unique")
    if args.confirmation_edits < 1:
        raise SystemExit("--confirmation-edits must be positive")

    rows = json.loads(args.dataset.read_text(encoding="utf-8"))
    required_rows = max(args.development_edits + 64, args.confirmation_edits + CONFIRMATION_CONTROLS)
    if not isinstance(rows, list) or len(rows) < required_rows:
        raise SystemExit(f"KnowEdit stream generation requires at least {required_rows} dataset rows")
    if not 100 <= args.development_edits <= len(rows) - 64:
        raise SystemExit(
            "--development-edits must be between 100 and the dataset size minus 64 controls"
        )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    dataset_hash = sha256(args.dataset)
    index = {
        "protocol": PROTOCOL,
        "dataset_label": dataset_label,
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": dataset_hash,
        "development_seeds": list(development_seeds),
        "confirmation_seeds": list(confirmation_seeds),
        "confirmation_edits": args.confirmation_edits,
        "streams": [],
    }
    splits = {
        "development": (development_seeds, args.development_edits, 64),
        "confirmation": (confirmation_seeds, args.confirmation_edits, CONFIRMATION_CONTROLS),
    }
    for split, (seeds, edit_count, probe_count) in splits.items():
        for seed in seeds:
            order = list(range(len(rows)))
            random.Random(seed).shuffle(order)
            edits = order[:edit_count]
            probes = order[edit_count:edit_count + probe_count]
            payload = {
                "protocol": PROTOCOL,
                "dataset_label": dataset_label,
                "split": split,
                "random_order_seed": seed,
                "dataset_sha256": dataset_hash,
                "dataset_rows": len(rows),
                "n_edits": edit_count,
                "n_pre_edit_controls": probe_count,
                "edit_indices": edits,
                "pre_edit_indices": probes,
            }
            path = output / f"{split}_seed_{seed}.json"
            write_json(path, payload)
            index["streams"].append({
                "split": split,
                "seed": seed,
                "path": path.name,
                "sha256": sha256(path),
            })
    write_json(output / "INDEX.json", index)
    print(f"wrote {len(index['streams'])} semantic-history streams to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
