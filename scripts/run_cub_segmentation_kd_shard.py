#!/usr/bin/env python3
"""Run a sharded locked CUB-mask KD versus KD+SSR confirmation."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_meeting_segmentation_ablation import (
    confirmation_jobs,
    current_git_commit,
    result_record,
    run_jobs,
    validate_job_record,
    validate_selection_lock,
)
from ssr_utils.paired_stats import paired_summary
from ssr_utils.result_schema import sha256_file


METHODS = ("kd", "biocs_kd")


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def selected_jobs(args: argparse.Namespace, spec, seeds: tuple[int, ...]) -> tuple[list, list]:
    complete = [
        job
        for job in confirmation_jobs(args.result_root, spec, seeds)
        if job.method in METHODS
    ]
    assigned_seeds = set(
        seeds[args.shard_index :: args.shard_count]
    )
    return complete, [job for job in complete if job.seed in assigned_seeds]


def write_summary(
    jobs: list,
    args: argparse.Namespace,
    git_commit: str,
    seeds: tuple[int, ...],
) -> None:
    records = {
        (job.seed, job.method): validate_job_record(
            job,
            git_commit,
            args.selection_lock_sha256,
        )
        for job in jobs
    }
    rows = []
    payload = {
        "protocol": "cub_mask_locked_kd_vs_kd_ssr_v2",
        "selection_lock_sha256": args.selection_lock_sha256,
        "seeds": list(seeds),
        "metrics": {},
    }
    for metric, higher_is_better in (
        ("mean_iou", True),
        ("mean_dice", True),
        ("avg_forgetting_iou", False),
        ("effective_rank", True),
        ("mean_abs_offdiag_cosine", False),
    ):
        control = [records[(seed, "kd")]["metrics"][metric] for seed in seeds]
        treatment = [
            records[(seed, "biocs_kd")]["metrics"][metric]
            for seed in seeds
        ]
        summary = paired_summary(
            control,
            treatment,
            higher_is_better=higher_is_better,
        ).as_dict()
        payload["metrics"][metric] = summary
        rows.append({"metric": metric, **summary})

    payload["primary_gate"] = {
        "miou_ci_positive": payload["metrics"]["mean_iou"]["ci95_low"] > 0,
        "dice_ci_positive": payload["metrics"]["mean_dice"]["ci95_low"] > 0,
        "forgetting_ci_positive": payload["metrics"]["avg_forgetting_iou"]["ci95_low"] > 0,
    }
    payload["primary_gate"]["passes"] = all(payload["primary_gate"].values())
    write_json_atomic(args.result_root / "KD_SSR_CONFIRMATION_SUMMARY.json", payload)
    csv_path = args.result_root / "KD_SSR_CONFIRMATION_SUMMARY.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--segmentation-cache", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--gpus", nargs="+", default=[str(index) for index in range(8)])
    parser.add_argument("--jobs-per-gpu", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seg-batch-size", type=int, default=24)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("shard-index must be in [0, shard-count)")
    for field in ("result_root", "selection_lock", "data_root", "segmentation_cache"):
        setattr(args, field, getattr(args, field).resolve())
    return args


def main() -> None:
    args = parse_args()
    git_commit = current_git_commit()
    source = args.data_root / "segmentations.tgz"
    if not source.is_file() or not args.segmentation_cache.is_file():
        raise FileNotFoundError(f"Missing CUB segmentation source/cache: {source}, {args.segmentation_cache}")
    args.segmentation_source_sha256 = sha256_file(source)
    args.segmentation_cache_sha256 = sha256_file(args.segmentation_cache)
    lock_payload = json.loads(args.selection_lock.read_text(encoding="utf-8"))
    raw_seeds = lock_payload.get("confirmation_seeds_hidden_during_selection")
    if (
        not isinstance(raw_seeds, list)
        or len(raw_seeds) < 2
        or any(not isinstance(seed, int) for seed in raw_seeds)
        or len(set(raw_seeds)) != len(raw_seeds)
    ):
        raise RuntimeError("Selection lock has an invalid confirmation-seed inventory")
    seeds = tuple(raw_seeds)
    args.confirmation_seeds = seeds
    args.confirmation_methods = METHODS
    spec = validate_selection_lock(args.selection_lock, args, git_commit)
    args.selection_lock_sha256 = lock_payload["lock_sha256"]
    complete, assigned = selected_jobs(args, spec, seeds)
    if not args.summary_only:
        run_jobs(assigned, args, git_commit)
    if args.summary_only:
        write_summary(complete, args, git_commit, seeds)
    else:
        write_json_atomic(
            args.result_root / f"SHARD_{args.shard_index}.json",
            {
                "shard_index": args.shard_index,
                "shard_count": args.shard_count,
                "seeds": sorted({job.seed for job in assigned}),
                "records": [str(result_record(job)) for job in assigned],
            },
        )


if __name__ == "__main__":
    main()
