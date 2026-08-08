#!/usr/bin/env python3
"""Run one four-way shard of the locked CUB mask factorial confirmation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_meeting_segmentation_ablation import (
    FACTORIAL_CONFIRMATION_SEEDS,
    METHODS,
    PROTOCOL_FACTORIAL,
    confirmation_jobs,
    current_git_commit,
    result_record,
    run_jobs,
    validate_selection_lock,
    write_confirmation_summary,
    write_json_atomic,
)
from ssr_utils.result_schema import sha256_file


def selected_jobs(args: argparse.Namespace, spec, seeds: tuple[int, ...]) -> tuple[list, list]:
    complete = confirmation_jobs(args.result_root, spec, seeds, METHODS)
    assigned_seeds = set(seeds[args.shard_index :: args.shard_count])
    assigned = [job for job in complete if job.seed in assigned_seeds]
    return complete, assigned


def manifest(args: argparse.Namespace, spec, seeds: tuple[int, ...]) -> dict:
    complete, assigned = selected_jobs(args, spec, seeds)
    return {
        "protocol": PROTOCOL_FACTORIAL,
        "selection_lock": str(args.selection_lock),
        "selection_lock_sha256": args.selection_lock_sha256,
        "selected": {
            "spec_id": spec.spec_id,
            "lambda_sp": spec.lambda_sp,
            "sigma_exc": spec.sigma_exc,
            "sigma_inh": spec.sigma_inh,
        },
        "confirmation_seeds": list(seeds),
        "confirmation_methods": list(METHODS),
        "shard": {
            "index": args.shard_index,
            "count": args.shard_count,
            "seeds": sorted({job.seed for job in assigned}),
        },
        "jobs": {
            "complete_confirmation": len(complete),
            "assigned": len(assigned),
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--segmentation-cache", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument(
        "--gpus", nargs="+", default=[str(index) for index in range(8)]
    )
    parser.add_argument("--jobs-per-gpu", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seg-batch-size", type=int, default=24)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args(argv)
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("shard-index must be in [0, shard-count)")
    if args.jobs_per_gpu < 1 or args.workers < 0 or args.seg_batch_size < 1:
        parser.error(
            "jobs-per-gpu and seg-batch-size must be positive; workers cannot be negative"
        )
    for field in ("result_root", "selection_lock", "data_root", "segmentation_cache"):
        setattr(args, field, getattr(args, field).resolve())
    return args


def main() -> None:
    args = parse_args()
    source = args.data_root / "segmentations.tgz"
    if not source.is_file() or not args.segmentation_cache.is_file():
        raise FileNotFoundError(
            f"Missing CUB segmentation source/cache: {source}, {args.segmentation_cache}"
        )
    args.protocol = PROTOCOL_FACTORIAL
    args.confirmation_seeds = FACTORIAL_CONFIRMATION_SEEDS
    args.confirmation_methods = METHODS
    args.segmentation_source_sha256 = sha256_file(source)
    args.segmentation_cache_sha256 = sha256_file(args.segmentation_cache)
    git_commit = current_git_commit()
    spec = validate_selection_lock(args.selection_lock, args, git_commit)
    lock_payload = json.loads(args.selection_lock.read_text(encoding="utf-8"))
    args.selection_lock_sha256 = lock_payload["lock_sha256"]
    args.result_root.mkdir(parents=True, exist_ok=True)
    complete, assigned = selected_jobs(args, spec, FACTORIAL_CONFIRMATION_SEEDS)
    if args.summary_only:
        # The aggregate call must not overwrite rank 0's 20-job shard manifest.
        write_confirmation_summary(complete, args.result_root, args, git_commit)
        return
    write_json_atomic(
        args.result_root / f"FACTORIAL_SHARD_{args.shard_index}.json",
        manifest(args, spec, FACTORIAL_CONFIRMATION_SEEDS),
    )

    subprocess.run(
        [
            args.python,
            str(PROJECT_ROOT / "scripts" / "check_experiment_worktree.py"),
            "--project-root",
            str(PROJECT_ROOT),
        ],
        check=True,
    )
    run_jobs(assigned, args, git_commit)
    write_json_atomic(
        args.result_root / f"SHARD_{args.shard_index}.json",
        {
            "protocol": PROTOCOL_FACTORIAL,
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
            "seeds": sorted({job.seed for job in assigned}),
            "methods": list(METHODS),
            "records": [
                str(result_record(job).relative_to(args.result_root)) for job in assigned
            ],
        },
    )


if __name__ == "__main__":
    main()
