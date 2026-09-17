#!/usr/bin/env python3
"""Create fixed edit/control streams for the SSR-COSMOS geometry study."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


PROTOCOL = "ssr_cosmos_ke_geometry_stream_v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-label", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-edits", type=int, default=100)
    parser.add_argument("--n-controls", type=int, default=64)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    args = parser.parse_args()

    if args.n_edits <= 0 or args.n_controls <= 0:
        raise SystemExit("n-edits and n-controls must be positive")
    if not args.seeds or len(args.seeds) != len(set(args.seeds)):
        raise SystemExit("seeds must be a non-empty unique list")
    rows = json.loads(args.dataset.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or len(rows) < args.n_edits + args.n_controls:
        raise SystemExit("dataset is too small for disjoint edit and control sets")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    dataset_hash = sha256(args.dataset)
    streams = []
    for seed in args.seeds:
        order = list(range(len(rows)))
        random.Random(seed).shuffle(order)
        edits = order[: args.n_edits]
        controls = order[args.n_edits : args.n_edits + args.n_controls]
        payload = {
            "protocol": PROTOCOL,
            "split": "prespecified_confirmation",
            "dataset_label": args.dataset_label,
            "dataset": str(args.dataset.resolve()),
            "dataset_sha256": dataset_hash,
            "dataset_rows": len(rows),
            "random_order_seed": seed,
            "n_edits": args.n_edits,
            "n_pre_edit_controls": args.n_controls,
            "edit_indices": edits,
            "pre_edit_indices": controls,
        }
        path = output / f"seed_{seed}.json"
        write_json(path, payload)
        streams.append({"seed": seed, "path": path.name, "sha256": sha256(path)})

    write_json(
        output / "INDEX.json",
        {
            "protocol": PROTOCOL,
            "dataset_label": args.dataset_label,
            "dataset": str(args.dataset.resolve()),
            "dataset_sha256": dataset_hash,
            "n_edits": args.n_edits,
            "n_pre_edit_controls": args.n_controls,
            "seeds": args.seeds,
            "streams": streams,
        },
    )
    print(f"wrote {len(streams)} {args.dataset_label} streams to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
