#!/usr/bin/env python3
"""Locked transfer confirmation of KD versus KD+SSR for one mask dataset.

The SSR hyperparameters are read from, and cryptographically tied to, the CUB
segmentation selection lock.  No target-dataset result participates in model or
hyperparameter selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ssr_utils.multidataset_segmentation import make_tasks
from ssr_utils.paired_stats import paired_summary
from ssr_utils.result_schema import (
    ResultSchemaError,
    sha256_file,
    sha256_value,
    validate_result_record,
)


RUNNER = PROJECT_ROOT / "experiments" / "multidataset_segmentation.py"
DATASET_CONFIGS = {
    "oxford_iiit_pet": (37, 5),
    "oxford_flowers102": (102, 10),
}
METHODS = ("kd", "kd_ssr")
CONFIRMATION_SEEDS = tuple(range(9501, 9540, 2))
PRIMARY_METRICS = (
    ("mean_iou", True),
    ("mean_dice", True),
    ("avg_forgetting_iou", False),
)
PRINT_LOCK = Lock()


@dataclass(frozen=True)
class LockedSpec:
    lambda_ssr: float
    sigma_exc: float
    sigma_inh: float
    lock_sha256: str


@dataclass(frozen=True)
class Job:
    method: str
    seed: int
    output: Path


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def current_git_commit() -> str:
    return subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def load_selection_lock(path: Path) -> LockedSpec:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Cannot read CUB selection lock {path}: {error}") from error
    observed_hash = payload.get("lock_sha256")
    unhashed = {key: value for key, value in payload.items() if key != "lock_sha256"}
    expected_hash = sha256_value(unhashed)
    if observed_hash != expected_hash:
        raise RuntimeError(
            f"CUB selection-lock hash mismatch: expected {expected_hash}, "
            f"observed {observed_hash}"
        )
    selected = payload.get("selected")
    if not isinstance(selected, dict):
        raise RuntimeError("CUB selection lock is missing selected hyperparameters")
    required = ("lambda_sp", "sigma_exc", "sigma_inh")
    missing = [name for name in required if name not in selected]
    if missing:
        raise RuntimeError(f"CUB selection lock selected block lacks {missing}")
    values = {name: float(selected[name]) for name in required}
    if values["lambda_sp"] <= 0 or values["sigma_exc"] <= 0 or values["sigma_inh"] <= 0:
        raise RuntimeError("Locked SSR strength and scales must be positive")
    return LockedSpec(
        lambda_ssr=values["lambda_sp"],
        sigma_exc=values["sigma_exc"],
        sigma_inh=values["sigma_inh"],
        lock_sha256=observed_hash,
    )


def confirmation_jobs(
    result_root: Path,
    seeds: tuple[int, ...] = CONFIRMATION_SEEDS,
) -> list[Job]:
    return [
        Job(
            method=method,
            seed=seed,
            output=result_root / "per_seed" / f"seed_{seed}" / method,
        )
        for seed in seeds
        for method in METHODS
    ]


def result_record(job: Job) -> Path:
    return job.output / "result_record.json"


def command(
    job: Job,
    args: argparse.Namespace,
    spec: LockedSpec,
    feature_cache_sha256: str,
) -> list[str]:
    return [
        args.python,
        str(RUNNER),
        "--dataset",
        args.dataset,
        "--data-root",
        str(args.dataset_root.parent),
        "--dataset-root",
        str(args.dataset_root),
        "--feature-cache",
        str(args.feature_cache),
        "--feature-cache-sha256",
        feature_cache_sha256,
        "--output-dir",
        str(job.output),
        "--method",
        job.method,
        "--seed",
        str(job.seed),
        "--device",
        "cuda",
        "--workers",
        str(args.workers),
        "--image-size",
        str(args.image_size),
        "--feature-batch-size",
        str(args.feature_batch_size),
        "--batch-size",
        str(args.batch_size),
        "--hidden-dim",
        str(args.hidden_dim),
        "--epochs",
        str(args.epochs),
        "--learning-rate",
        str(args.learning_rate),
        "--weight-decay",
        str(args.weight_decay),
        "--lambda-kd",
        str(args.lambda_kd),
        "--lambda-ssr",
        str(spec.lambda_ssr),
        "--a-exc",
        "1.0",
        "--a-inh",
        "0.8",
        "--sigma-exc",
        str(spec.sigma_exc),
        "--sigma-inh",
        str(spec.sigma_inh),
        "--kernel-family",
        "gaussian",
        "--selection-lock-sha256",
        spec.lock_sha256,
    ]


def load_input_manifests(args: argparse.Namespace) -> tuple[dict, dict, str]:
    dataset_manifest_path = args.dataset_root / "DATASET_MANIFEST.json"
    cache_manifest_path = args.feature_cache.with_name(
        args.feature_cache.name + ".manifest.json"
    )
    try:
        dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
        cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Invalid prepared segmentation input: {error}") from error
    if dataset_manifest.get("dataset") != args.dataset:
        raise RuntimeError("Dataset manifest identity does not match --dataset")
    if cache_manifest.get("dataset") != args.dataset:
        raise RuntimeError("Feature-cache manifest identity does not match --dataset")
    if not args.feature_cache.is_file():
        raise FileNotFoundError(args.feature_cache)
    actual_cache_sha256 = sha256_file(args.feature_cache)
    if cache_manifest.get("cache_sha256") != actual_cache_sha256:
        raise RuntimeError("Feature-cache bytes do not match the cache manifest")
    for label, value in (
        ("dataset fingerprint", dataset_manifest.get("dataset_fingerprint")),
        ("encoder state", cache_manifest.get("encoder_state_sha256")),
    ):
        if not _is_sha256(value):
            raise RuntimeError(f"Prepared input has invalid {label} SHA256")
    if cache_manifest.get("dataset_fingerprint") != dataset_manifest.get(
        "dataset_fingerprint"
    ):
        raise RuntimeError("Dataset and feature-cache fingerprints disagree")
    return dataset_manifest, cache_manifest, actual_cache_sha256


def _expected_dataset_hash(
    args: argparse.Namespace,
    dataset_manifest: dict,
    cache_sha256: str,
) -> str:
    return sha256_value(
        {
            "dataset_fingerprint": dataset_manifest["dataset_fingerprint"],
            "feature_cache_sha256": cache_sha256,
            "mask_policy": dataset_manifest["mask_policy"],
            "image_size": args.image_size,
        }
    )


def validate_job_record(
    job: Job,
    args: argparse.Namespace,
    spec: LockedSpec,
    expected_git_commit: str,
    dataset_manifest: dict,
    cache_manifest: dict,
    cache_sha256: str,
) -> dict:
    path = result_record(job)
    try:
        record = validate_result_record(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, ResultSchemaError) as error:
        raise RuntimeError(f"Invalid result record {path}: {error}") from error
    uses_ssr = job.method == "kd_ssr"
    expected_objective = {
        "task": True,
        "kd": True,
        "ssr": uses_ssr,
        "anchor": False,
        "spectral": False,
        "ewc": False,
        "mas": False,
        "si": False,
        "center": False,
        "protodecor": False,
    }
    expected = {
        "git_commit": expected_git_commit,
        "task_family": "segmentation",
        "dataset": f"{args.dataset}_masks",
        "model": "resnet18_dense_class_conditioned_decoder",
        "seed": job.seed,
        "recipe": "kd_ssr" if uses_ssr else "kd",
        "objective": expected_objective,
        "distance_mapping": "cosine" if uses_ssr else "none",
        "selection_lock_hash": spec.lock_sha256,
        "dataset_hash": _expected_dataset_hash(args, dataset_manifest, cache_sha256),
        "source_dataset_fingerprint": dataset_manifest["dataset_fingerprint"],
        "feature_cache_sha256": cache_sha256,
        "encoder_state_sha256": cache_manifest["encoder_state_sha256"],
        "torchvision_version": cache_manifest["torchvision_version"],
        "task_schedule_hash": sha256_value(
            make_tasks(*DATASET_CONFIGS[args.dataset], job.seed)
        ),
    }
    mismatches = {
        key: {"expected": value, "observed": record.get(key)}
        for key, value in expected.items()
        if record.get(key) != value
    }
    expected_scalars = {
        "lambda_kd": args.lambda_kd,
        "lambda_ssr": spec.lambda_ssr if uses_ssr else 0.0,
    }
    for key, value in expected_scalars.items():
        try:
            matches = float(record.get(key)) == value
        except (TypeError, ValueError):
            matches = False
        if not matches:
            mismatches[key] = {"expected": value, "observed": record.get(key)}
    expected_kernel = (
        {
            "family": "gaussian",
            "A_exc": 1.0,
            "A_inh": 0.8,
            "sigma_exc": spec.sigma_exc,
            "sigma_inh": spec.sigma_inh,
        }
        if uses_ssr
        else {}
    )
    if record.get("kernel") != expected_kernel:
        mismatches["kernel"] = {
            "expected": expected_kernel,
            "observed": record.get("kernel"),
        }
    for field in ("initial_model_hash", "task_schedule_hash"):
        if not _is_sha256(record.get(field)):
            mismatches[field] = {
                "expected": "lowercase SHA256",
                "observed": record.get(field),
            }
    optimizer_steps = record.get("optimizer_steps")
    if not isinstance(optimizer_steps, int) or isinstance(optimizer_steps, bool) or optimizer_steps <= 0:
        mismatches["optimizer_steps"] = {
            "expected": "positive integer",
            "observed": optimizer_steps,
        }
    args_path = job.output / "args.json"
    try:
        run_args = json.loads(args_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Invalid run arguments {args_path}: {error}") from error
    if record.get("config_hash") != sha256_value(run_args):
        mismatches["config_hash"] = {
            "expected": sha256_value(run_args),
            "observed": record.get("config_hash"),
        }
    expected_run_args = {
        "dataset": args.dataset,
        "dataset_root": str(args.dataset_root),
        "feature_cache": str(args.feature_cache),
        "feature_cache_sha256": cache_sha256,
        "method": job.method,
        "seed": job.seed,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "hidden_dim": args.hidden_dim,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "lambda_kd": args.lambda_kd,
        "lambda_ssr": spec.lambda_ssr,
        "sigma_exc": spec.sigma_exc,
        "sigma_inh": spec.sigma_inh,
        "selection_lock_sha256": spec.lock_sha256,
    }
    for key, value in expected_run_args.items():
        if run_args.get(key) != value:
            mismatches[f"config.{key}"] = {
                "expected": value,
                "observed": run_args.get(key),
            }
    required_metrics = {name for name, _ in PRIMARY_METRICS}
    if not required_metrics <= set(record.get("metrics", {})):
        mismatches["metrics"] = {
            "expected": sorted(required_metrics),
            "observed": sorted(record.get("metrics", {})),
        }
    if mismatches:
        raise RuntimeError(
            f"Result identity mismatch for {path}: {json.dumps(mismatches, sort_keys=True)}"
        )
    return record


def validate_paired_records(control: dict, treatment: dict, *, identity: str) -> None:
    fields = (
        "git_commit",
        "dataset",
        "seed",
        "dataset_hash",
        "source_dataset_fingerprint",
        "feature_cache_sha256",
        "encoder_state_sha256",
        "torchvision_version",
        "initial_model_hash",
        "task_schedule_hash",
        "optimizer_steps",
        "lambda_kd",
        "selection_lock_hash",
    )
    mismatches = {
        field: {"kd": control.get(field), "kd_ssr": treatment.get(field)}
        for field in fields
        if control.get(field) != treatment.get(field)
    }
    control_objective = dict(control.get("objective", {}))
    treatment_objective = dict(treatment.get("objective", {}))
    treatment_objective["ssr"] = False
    if control_objective != treatment_objective:
        mismatches["objective"] = {
            "kd": control.get("objective"),
            "kd_ssr_without_ssr": treatment_objective,
        }
    if mismatches:
        raise RuntimeError(f"Unmatched KD pair {identity}: {mismatches}")


def run_worker(
    gpu: str,
    jobs: list[Job],
    args: argparse.Namespace,
    spec: LockedSpec,
    git_commit: str,
    dataset_manifest: dict,
    cache_manifest: dict,
    cache_sha256: str,
) -> None:
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = gpu
    for job in jobs:
        record_path = result_record(job)
        if record_path.is_file():
            validate_job_record(
                job,
                args,
                spec,
                git_commit,
                dataset_manifest,
                cache_manifest,
                cache_sha256,
            )
            with PRINT_LOCK:
                print(f"SKIP gpu={gpu} {job.output}", flush=True)
            continue
        job.output.mkdir(parents=True, exist_ok=True)
        log_path = job.output / "launcher.log"
        failure_path = job.output / "FAILED.json"
        if failure_path.is_file():
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            failure_path.rename(job.output / f"FAILED.previous.{timestamp}.json")
        with PRINT_LOCK:
            print(f"RUN gpu={gpu} method={job.method} seed={job.seed}", flush=True)
        with log_path.open("a", encoding="utf-8") as log:
            completed = subprocess.run(
                command(job, args, spec, cache_sha256),
                cwd=PROJECT_ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        if completed.returncode != 0 or not record_path.is_file():
            write_json_atomic(
                failure_path,
                {
                    "failed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "returncode": completed.returncode,
                    "job": {**asdict(job), "output": str(job.output)},
                    "log": str(log_path),
                },
            )
            raise RuntimeError(f"Job failed: {job.output}; see {log_path}")
        validate_job_record(
            job,
            args,
            spec,
            git_commit,
            dataset_manifest,
            cache_manifest,
            cache_sha256,
        )


def run_jobs(
    jobs: list[Job],
    args: argparse.Namespace,
    spec: LockedSpec,
    git_commit: str,
    dataset_manifest: dict,
    cache_manifest: dict,
    cache_sha256: str,
) -> None:
    slots = [gpu for gpu in args.gpus for _ in range(args.jobs_per_gpu)]
    assignments = [jobs[index :: len(slots)] for index in range(len(slots))]
    with ThreadPoolExecutor(max_workers=len(slots)) as pool:
        futures = [
            pool.submit(
                run_worker,
                gpu,
                assigned,
                args,
                spec,
                git_commit,
                dataset_manifest,
                cache_manifest,
                cache_sha256,
            )
            for gpu, assigned in zip(slots, assignments)
            if assigned
        ]
        for future in futures:
            future.result()


def write_summary(
    jobs: list[Job],
    args: argparse.Namespace,
    spec: LockedSpec,
    git_commit: str,
    dataset_manifest: dict,
    cache_manifest: dict,
    cache_sha256: str,
) -> None:
    records = {
        (job.seed, job.method): validate_job_record(
            job,
            args,
            spec,
            git_commit,
            dataset_manifest,
            cache_manifest,
            cache_sha256,
        )
        for job in jobs
    }
    paired_rows = []
    for seed in args.seeds:
        control = records[(seed, "kd")]
        treatment = records[(seed, "kd_ssr")]
        validate_paired_records(control, treatment, identity=f"{args.dataset}/seed_{seed}")
        row = {"dataset": args.dataset, "seed": seed}
        for metric, higher_is_better in PRIMARY_METRICS:
            control_value = float(control["metrics"][metric])
            treatment_value = float(treatment["metrics"][metric])
            row[f"kd_{metric}"] = control_value
            row[f"kd_ssr_{metric}"] = treatment_value
            raw_difference = treatment_value - control_value
            row[f"delta_{metric}"] = (
                raw_difference if higher_is_better else -raw_difference
            )
        paired_rows.append(row)
    summaries = {}
    summary_rows = []
    for metric, higher_is_better in PRIMARY_METRICS:
        summary = paired_summary(
            [records[(seed, "kd")]["metrics"][metric] for seed in args.seeds],
            [
                records[(seed, "kd_ssr")]["metrics"][metric]
                for seed in args.seeds
            ],
            higher_is_better=higher_is_better,
        ).as_dict()
        summaries[metric] = summary
        summary_rows.append({"dataset": args.dataset, "metric": metric, **summary})
    payload = {
        "protocol": "locked_cub_to_multidataset_segmentation_kd_transfer_v1",
        "dataset": args.dataset,
        "contrast": "kd_ssr_minus_kd",
        "locked_spec": asdict(spec),
        "confirmation_seeds": list(args.seeds),
        "metric_directions": {
            metric: "higher" if higher else "lower"
            for metric, higher in PRIMARY_METRICS
        },
        "metrics": summaries,
        "per_seed": paired_rows,
    }
    write_json_atomic(args.result_root / "CONFIRMATION_SUMMARY.json", payload)
    with (args.result_root / "CONFIRMATION_SUMMARY.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    write_json_atomic(args.result_root / "PER_SEED_RESULTS.json", paired_rows)
    with (args.result_root / "PER_SEED_RESULTS.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired_rows[0]))
        writer.writeheader()
        writer.writerows(paired_rows)


def manifest(args: argparse.Namespace, spec: LockedSpec) -> dict:
    jobs = confirmation_jobs(args.result_root, args.seeds)
    seed_count = len(args.seeds)
    return {
        "protocol": "locked_cub_to_multidataset_segmentation_kd_transfer_v1",
        "dataset": args.dataset,
        "primary_contrast": "kd_ssr_minus_kd",
        "selection_lock": str(args.selection_lock),
        "selection_lock_sha256": spec.lock_sha256,
        "locked_spec": asdict(spec),
        "confirmation_seeds": list(args.seeds),
        "training": {
            "epochs_per_task": args.epochs,
            "lambda_kd": args.lambda_kd,
            "image_size": args.image_size,
            "feature_cache": str(args.feature_cache),
            "dataset_root": str(args.dataset_root),
            "batch_size": args.batch_size,
            "hidden_dim": args.hidden_dim,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "encoder": "frozen torchvision ResNet-18 through layer3",
            "decoder": "matched class-conditioned binary mask decoder",
            "kd_protocol": "old class conditions on current features",
        },
        "jobs": {"kd": seed_count, "kd_ssr": seed_count, "total": 2 * seed_count},
        "job_ids": [str(job.output.relative_to(args.result_root)) for job in jobs],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=tuple(DATASET_CONFIGS), required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(CONFIRMATION_SEEDS),
    )
    parser.add_argument("--gpus", nargs="+", default=[str(index) for index in range(8)])
    parser.add_argument("--jobs-per-gpu", type=int, default=1)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--feature-batch-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lambda-kd", type=float, default=2.0)
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args(argv)
    args.result_root = args.result_root.resolve()
    args.dataset_root = args.dataset_root.resolve()
    args.feature_cache = args.feature_cache.resolve()
    args.selection_lock = args.selection_lock.resolve()
    if args.jobs_per_gpu < 1 or args.workers < 0 or args.epochs < 1:
        parser.error("jobs-per-gpu and epochs must be positive; workers cannot be negative")
    if args.lambda_kd <= 0:
        parser.error("lambda-kd must be positive")
    if len(args.seeds) < 2 or len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must contain at least two unique values")
    args.seeds = tuple(args.seeds)
    return args


def main() -> None:
    args = parse_args()
    spec = load_selection_lock(args.selection_lock)
    args.result_root.mkdir(parents=True, exist_ok=True)
    payload = manifest(args, spec)
    write_json_atomic(args.result_root / "MANIFEST.json", payload)
    if args.manifest_only:
        print(json.dumps(payload, indent=2))
        return
    subprocess.run(
        [
            args.python,
            str(PROJECT_ROOT / "scripts" / "check_experiment_worktree.py"),
            "--project-root",
            str(PROJECT_ROOT),
        ],
        check=True,
    )
    git_commit = current_git_commit()
    dataset_manifest, cache_manifest, cache_sha256 = load_input_manifests(args)
    payload.update(
        {
            "git_commit": git_commit,
            "dataset_fingerprint": dataset_manifest["dataset_fingerprint"],
            "feature_cache_sha256": cache_sha256,
            "encoder_state_sha256": cache_manifest["encoder_state_sha256"],
        }
    )
    write_json_atomic(args.result_root / "MANIFEST.json", payload)
    jobs = confirmation_jobs(args.result_root, args.seeds)
    run_jobs(
        jobs,
        args,
        spec,
        git_commit,
        dataset_manifest,
        cache_manifest,
        cache_sha256,
    )
    write_summary(
        jobs,
        args,
        spec,
        git_commit,
        dataset_manifest,
        cache_manifest,
        cache_sha256,
    )
    print(f"Locked {args.dataset} KD confirmation complete: {args.result_root}")


if __name__ == "__main__":
    main()
