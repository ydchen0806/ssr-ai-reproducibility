#!/usr/bin/env python3
"""Resumable screen-lock-confirm driver for the CUB mask 2x2 ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ssr_utils.result_schema import ResultSchemaError, sha256_file, validate_result_record

RUNNER = PROJECT_ROOT / "experiments" / "cub200_continual_benchmark.py"
DEVELOPMENT_SEEDS = (9101, 9103, 9105)
CONFIRMATION_SEEDS = tuple(range(9201, 9221, 2))
METHODS = ("baseline", "kd", "biocs", "biocs_kd")
EXPECTED_RECIPES = {
    "baseline": "plain",
    "kd": "kd",
    "biocs": "ssr_only",
    "biocs_kd": "ssr_kd",
}


@dataclass(frozen=True)
class Spec:
    spec_id: str
    lambda_sp: float
    sigma_exc: float
    sigma_inh: float


SPECS = (
    Spec("l0p025_e0p20_i0p50", 0.025, 0.20, 0.50),
    Spec("l0p050_e0p20_i0p50", 0.050, 0.20, 0.50),
    Spec("l0p075_e0p20_i0p50", 0.075, 0.20, 0.50),
    Spec("l0p100_e0p20_i0p50", 0.100, 0.20, 0.50),
    Spec("l0p050_e0p16_i0p45", 0.050, 0.16, 0.45),
    Spec("l0p075_e0p16_i0p45", 0.075, 0.16, 0.45),
)
CONTROL_SPEC = Spec("control", 0.05, 0.20, 0.50)
PRINT_LOCK = Lock()


@dataclass(frozen=True)
class Job:
    phase: str
    method: str
    seed: int
    spec: Spec
    output: Path


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def result_record(job: Job) -> Path:
    return (
        job.output
        / "result_records"
        / "segmentation"
        / job.method
        / f"seed_{job.seed}"
        / "result_record.json"
    )


def current_git_commit() -> str:
    return subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def validate_job_record(
    job: Job,
    expected_git_commit: str,
    expected_selection_lock_hash: str,
) -> dict:
    path = result_record(job)
    try:
        record = validate_result_record(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, ResultSchemaError) as error:
        raise RuntimeError(f"Invalid result record {path}: {error}") from error
    expected = {
        "git_commit": expected_git_commit,
        "task_family": "segmentation",
        "dataset": "cub200_masks",
        "model": "resnet18_dense_decoder",
        "seed": job.seed,
        "recipe": EXPECTED_RECIPES[job.method],
        "distance_mapping": "cosine" if "biocs" in job.method else "none",
        "selection_lock_hash": (
            expected_selection_lock_hash
            if job.phase == "confirmation"
            else "development_screen"
        ),
    }
    mismatches = {
        key: {"expected": value, "observed": record.get(key)}
        for key, value in expected.items()
        if record.get(key) != value
    }
    if "biocs" in job.method:
        expected_kernel = {
            "sigma_exc": job.spec.sigma_exc,
            "sigma_inh": job.spec.sigma_inh,
        }
        for key, value in expected_kernel.items():
            if float(record["kernel"].get(key, float("nan"))) != value:
                mismatches[f"kernel.{key}"] = {
                    "expected": value,
                    "observed": record["kernel"].get(key),
                }
        if float(record.get("lambda_ssr", float("nan"))) != job.spec.lambda_sp:
            mismatches["lambda_ssr"] = {
                "expected": job.spec.lambda_sp,
                "observed": record.get("lambda_ssr"),
            }
    if mismatches:
        raise RuntimeError(
            f"Result identity mismatch for {path}: {json.dumps(mismatches, sort_keys=True)}"
        )
    return record


def command(job: Job, args: argparse.Namespace) -> list[str]:
    output = [
        args.python,
        str(RUNNER),
        "--data_root", str(args.data_root),
        "--segmentation_cache", str(args.segmentation_cache),
        "--segmentation_source_sha256", args.segmentation_source_sha256,
        "--segmentation_cache_sha256", args.segmentation_cache_sha256,
        "--output_dir", str(job.output),
        "--tasks", "segmentation",
        "--methods", job.method,
        "--seeds", str(job.seed),
        "--num_classes", "200",
        "--classes_per_task", "10",
        "--class_order", "semantic",
        "--seg_epochs", "24",
        "--seg_batch_size", "24",
        "--seg_image_size", "192",
        "--seg_hidden_dim", "256",
        "--seg_biocs_target", "class",
        "--seg_biocs_scope", "seen",
        "--lambda_sp", str(job.spec.lambda_sp),
        "--lambda_kd_seg", "2.0",
        "--a-exc", "1.0",
        "--a-inh", "0.8",
        "--sigma-exc", str(job.spec.sigma_exc),
        "--sigma-inh", str(job.spec.sigma_inh),
        "--kernel-family", "gaussian",
        "--workers", "0",
        "--device", "cuda",
        "--no_download",
    ]
    if job.phase == "confirmation":
        output.extend(["--selection_lock_sha256", args.selection_lock_sha256])
    return output


def run_worker(
    gpu: str,
    jobs: list[Job],
    args: argparse.Namespace,
    expected_git_commit: str,
) -> None:
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = gpu
    for job in jobs:
        record = result_record(job)
        if record.is_file():
            validate_job_record(
                job,
                expected_git_commit,
                getattr(args, "selection_lock_sha256", "development_screen"),
            )
            with PRINT_LOCK:
                print(f"SKIP gpu={gpu} {job.output}", flush=True)
            continue
        job.output.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_path = job.output / f"launcher_{timestamp}.log"
        failure_path = job.output / "FAILED.json"
        if failure_path.is_file():
            failure_path.rename(job.output / f"FAILED.previous.{timestamp}.json")
        with PRINT_LOCK:
            print(
                f"RUN gpu={gpu} phase={job.phase} method={job.method} "
                f"spec={job.spec.spec_id} seed={job.seed}",
                flush=True,
            )
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(
                command(job, args),
                cwd=PROJECT_ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        if result.returncode != 0 or not record.is_file():
            write_json(
                failure_path,
                {
                    "failed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "returncode": result.returncode,
                    "job": {
                        "phase": job.phase,
                        "method": job.method,
                        "seed": job.seed,
                        "spec": asdict(job.spec),
                    },
                    "log": str(log_path),
                },
            )
            raise RuntimeError(
                f"Segmentation job failed ({result.returncode}): {job.output}; see {log_path}"
            )
        validate_job_record(
            job,
            expected_git_commit,
            getattr(args, "selection_lock_sha256", "development_screen"),
        )


def run_jobs(jobs: list[Job], args: argparse.Namespace, expected_git_commit: str) -> None:
    from concurrent.futures import ThreadPoolExecutor

    assignments = [jobs[index :: len(args.gpus)] for index in range(len(args.gpus))]
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        futures = [
            pool.submit(run_worker, gpu, assigned, args, expected_git_commit)
            for gpu, assigned in zip(args.gpus, assignments)
            if assigned
        ]
        for future in futures:
            future.result()


def read_row(job: Job) -> dict:
    path = result_record(job)
    try:
        record = validate_result_record(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, ResultSchemaError) as error:
        raise RuntimeError(f"Invalid screen result record {path}: {error}") from error
    required = {"mean_iou", "mean_dice", "avg_forgetting_iou"}
    missing = sorted(required - set(record["metrics"]))
    if missing:
        raise RuntimeError(f"Screen result record lacks selection metrics {missing}: {path}")
    return record["metrics"]


def select_spec(
    jobs: list[Job],
    result_root: Path,
    args: argparse.Namespace,
    git_commit: str,
) -> Spec:
    rows = {(job.seed, job.method, job.spec.spec_id): read_row(job) for job in jobs}
    summaries = []
    for spec in SPECS:
        changes = {"miou": [], "dice": [], "forgetting": []}
        per_seed = []
        for seed in DEVELOPMENT_SEEDS:
            control = rows[(seed, "kd", "control")]
            treatment = rows[(seed, "biocs_kd", spec.spec_id)]
            values = {
                "miou": float(treatment["mean_iou"]) - float(control["mean_iou"]),
                "dice": float(treatment["mean_dice"]) - float(control["mean_dice"]),
                "forgetting": float(control["avg_forgetting_iou"])
                - float(treatment["avg_forgetting_iou"]),
            }
            for metric, value in values.items():
                changes[metric].append(value)
            per_seed.append({"seed": seed, **values})
        means = {metric: statistics.mean(values) for metric, values in changes.items()}
        summaries.append(
            {
                "spec": asdict(spec),
                "mean_changes": means,
                "all_means_positive": all(value > 0 for value in means.values()),
                "balanced_score": min(means.values()),
                "triple_wins": sum(
                    row["miou"] > 0 and row["dice"] > 0 and row["forgetting"] > 0
                    for row in per_seed
                ),
                "sum_score": sum(means.values()),
                "per_seed": per_seed,
            }
        )
    summaries.sort(
        key=lambda row: (
            row["all_means_positive"],
            row["balanced_score"],
            row["triple_wins"],
            row["sum_score"],
        ),
        reverse=True,
    )
    selected = Spec(**summaries[0]["spec"])
    payload = {
        "selected": asdict(selected),
        "selection_rule": (
            "Prefer positive mean mIoU, Dice and IoU-forgetting changes versus KD; "
            "then maximize the weakest change, triple wins and their sum."
        ),
        "development_seeds": list(DEVELOPMENT_SEEDS),
        "confirmation_seeds_hidden_during_selection": list(CONFIRMATION_SEEDS),
        "screen_summary": summaries,
        "git_commit": git_commit,
        "segmentation_source_sha256": args.segmentation_source_sha256,
        "segmentation_cache_sha256": args.segmentation_cache_sha256,
        "screen_result_records": [
            {
                "path": str(result_record(job).relative_to(result_root)),
                "sha256": sha256_file(result_record(job)),
            }
            for job in sorted(jobs, key=lambda item: str(result_record(item)))
        ],
    }
    payload["lock_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    write_json(result_root / "SELECTION_LOCK.json", payload)
    return selected


def validate_selection_lock(
    lock_path: Path,
    args: argparse.Namespace,
    git_commit: str,
) -> Spec:
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    observed_hash = payload.get("lock_sha256")
    unhashed = {key: value for key, value in payload.items() if key != "lock_sha256"}
    expected_hash = hashlib.sha256(
        json.dumps(unhashed, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if observed_hash != expected_hash:
        raise RuntimeError(
            f"Selection lock hash mismatch: expected {expected_hash}, observed {observed_hash}"
        )
    expected_identity = {
        "git_commit": git_commit,
        "segmentation_source_sha256": args.segmentation_source_sha256,
        "segmentation_cache_sha256": args.segmentation_cache_sha256,
    }
    mismatches = {
        key: {"expected": value, "observed": payload.get(key)}
        for key, value in expected_identity.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Selection lock identity mismatch: {mismatches}")
    selected = Spec(**payload["selected"])
    if selected not in SPECS:
        raise RuntimeError(f"Selection lock contains an unplanned spec: {selected}")
    screen_records = payload.get("screen_result_records")
    expected_paths = {
        str(result_record(job).relative_to(lock_path.parent))
        for job in screen_jobs(lock_path.parent)
    }
    if not isinstance(screen_records, list):
        raise RuntimeError("Selection lock is missing the screen result inventory")
    if any(
        not isinstance(item, dict)
        or not isinstance(item.get("path"), str)
        or not isinstance(item.get("sha256"), str)
        for item in screen_records
    ):
        raise RuntimeError("Selection lock screen inventory contains an invalid entry")
    observed_paths = [item["path"] for item in screen_records]
    if len(screen_records) != len(expected_paths) or set(observed_paths) != expected_paths:
        raise RuntimeError(
            "Selection lock screen inventory mismatch: "
            f"expected={len(expected_paths)}, observed={len(screen_records)}, "
            f"missing={sorted(expected_paths - set(observed_paths))}, "
            f"unexpected={sorted(set(observed_paths) - expected_paths)}"
        )
    for item in screen_records:
        path = lock_path.parent / item["path"]
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"Selection lock screen artifact mismatch: {path}")
    return selected


def screen_jobs(result_root: Path) -> list[Job]:
    jobs = []
    for seed in DEVELOPMENT_SEEDS:
        jobs.append(
            Job("development", "kd", seed, CONTROL_SPEC, result_root / "development" / f"kd_control_seed{seed}")
        )
        for spec in SPECS:
            jobs.append(
                Job(
                    "development",
                    "biocs_kd",
                    seed,
                    spec,
                    result_root / "development" / f"biocs_kd_{spec.spec_id}_seed{seed}",
                )
            )
    return jobs


def confirmation_jobs(result_root: Path, spec: Spec) -> list[Job]:
    return [
        Job(
            "confirmation",
            method,
            seed,
            spec,
            result_root / "confirmation" / f"{method}_{spec.spec_id}_seed{seed}",
        )
        for seed in CONFIRMATION_SEEDS
        for method in METHODS
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--phase", choices=["all", "screen", "confirm"], default="all")
    parser.add_argument("--gpus", nargs="+", default=["0"])
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data/cub200")
    parser.add_argument(
        "--segmentation-cache",
        type=Path,
        default=PROJECT_ROOT / "data/cub200/cub200_seg_resnet18_dense_192.pt",
    )
    parser.add_argument("--manifest-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_root = args.result_root.resolve()
    result_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "protocol": "meeting_20260803_cub_segmentation_2x2_old_class_kd_v1",
        "development_seeds": list(DEVELOPMENT_SEEDS),
        "confirmation_seeds": list(CONFIRMATION_SEEDS),
        "development_specs": [asdict(spec) for spec in SPECS],
        "methods": list(METHODS),
        "primary_contrast": "biocs_kd_minus_kd",
        "kd_protocol": "old_class_conditions_on_current_features_v1",
        "jobs": {"development": 21, "confirmation": 40},
    }
    write_json(result_root / "MANIFEST.json", manifest)
    if args.manifest_only:
        print(json.dumps(manifest, indent=2))
        return
    subprocess.run(
        [
            args.python,
            str(PROJECT_ROOT / "scripts/check_experiment_worktree.py"),
            "--project-root",
            str(PROJECT_ROOT),
        ],
        check=True,
    )
    git_commit = current_git_commit()
    for required in (RUNNER, args.data_root, args.segmentation_cache):
        if not required.exists():
            raise FileNotFoundError(required)
    segmentation_source = args.data_root / "segmentations.tgz"
    if not segmentation_source.is_file():
        raise FileNotFoundError(
            f"The locked segmentation source archive is missing: {segmentation_source}"
        )
    args.segmentation_source_sha256 = sha256_file(segmentation_source)
    args.segmentation_cache_sha256 = sha256_file(args.segmentation_cache)
    manifest["segmentation_source"] = {
        "path": str(segmentation_source.resolve()),
        "sha256": args.segmentation_source_sha256,
    }
    manifest["segmentation_cache"] = {
        "path": str(args.segmentation_cache.resolve()),
        "sha256": args.segmentation_cache_sha256,
    }
    write_json(result_root / "MANIFEST.json", manifest)

    selected = None
    if args.phase in {"all", "screen"}:
        jobs = screen_jobs(result_root)
        run_jobs(jobs, args, git_commit)
        select_spec(jobs, result_root, args, git_commit)
        selected = validate_selection_lock(
            result_root / "SELECTION_LOCK.json", args, git_commit
        )
    if args.phase in {"all", "confirm"}:
        lock_path = result_root / "SELECTION_LOCK.json"
        if selected is None:
            if not lock_path.is_file():
                raise FileNotFoundError(
                    f"Confirmation requires a completed development lock: {lock_path}"
                )
            selected = validate_selection_lock(lock_path, args, git_commit)
        args.selection_lock_sha256 = json.loads(
            lock_path.read_text(encoding="utf-8")
        )["lock_sha256"]
        run_jobs(confirmation_jobs(result_root, selected), args, git_commit)
    print(f"Segmentation phase {args.phase} complete: {result_root}")


if __name__ == "__main__":
    main()
