#!/usr/bin/env python3
"""Resumable pooled screen-lock-confirm protocol for two segmentation datasets."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
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
PREPARER = PROJECT_ROOT / "scripts" / "prepare_multidataset_segmentation.py"
DATASETS = ("oxford_iiit_pet", "oxford_flowers102")
TASK_CONFIGS = {
    "oxford_iiit_pet": (37, 5),
    "oxford_flowers102": (102, 10),
}
DEVELOPMENT_SEEDS = (9301, 9303, 9305)
CONFIRMATION_SEEDS = tuple(range(9401, 9421, 2))
PRINT_LOCK = Lock()


@dataclass(frozen=True)
class Spec:
    spec_id: str
    lambda_ssr: float
    sigma_exc: float
    sigma_inh: float


SPECS = (
    Spec("l0p025_e0p16_i0p45", 0.025, 0.16, 0.45),
    Spec("l0p050_e0p16_i0p45", 0.050, 0.16, 0.45),
    Spec("l0p075_e0p16_i0p45", 0.075, 0.16, 0.45),
    Spec("l0p050_e0p20_i0p50", 0.050, 0.20, 0.50),
    Spec("l0p075_e0p20_i0p50", 0.075, 0.20, 0.50),
)
CONTROL = Spec("control", 0.0, 0.16, 0.45)


@dataclass(frozen=True)
class Job:
    phase: str
    dataset: str
    method: str
    seed: int
    spec: Spec
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


def cache_path(data_root: Path, dataset: str, image_size: int) -> Path:
    return data_root / dataset / f"resnet18_layer3_masks_{image_size}_provenance_v2.pt"


def result_record(job: Job) -> Path:
    return job.output / "result_record.json"


def screen_jobs(result_root: Path) -> list[Job]:
    jobs = []
    for dataset in DATASETS:
        for seed in DEVELOPMENT_SEEDS:
            jobs.append(
                Job(
                    "development",
                    dataset,
                    "task_only",
                    seed,
                    CONTROL,
                    result_root / "development" / dataset / f"task_only_seed{seed}",
                )
            )
            for spec in SPECS:
                jobs.append(
                    Job(
                        "development",
                        dataset,
                        "task_ssr",
                        seed,
                        spec,
                        result_root
                        / "development"
                        / dataset
                        / f"task_ssr_{spec.spec_id}_seed{seed}",
                    )
                )
    return jobs


def confirmation_jobs(result_root: Path, selected: Spec) -> list[Job]:
    return [
        Job(
            "confirmation",
            dataset,
            method,
            seed,
            selected if method == "task_ssr" else CONTROL,
            result_root
            / "confirmation"
            / dataset
            / f"{method}_{selected.spec_id if method == 'task_ssr' else 'control'}_seed{seed}",
        )
        for dataset in DATASETS
        for seed in CONFIRMATION_SEEDS
        for method in ("task_only", "task_ssr")
    ]


def command(job: Job, args: argparse.Namespace) -> list[str]:
    output = [
        args.python,
        str(RUNNER),
        "--dataset",
        job.dataset,
        "--data-root",
        str(args.data_root),
        "--feature-cache",
        str(cache_path(args.data_root, job.dataset, args.image_size)),
        "--feature-cache-sha256",
        args.cache_sha256[job.dataset],
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
        "--batch-size",
        str(args.batch_size),
        "--feature-batch-size",
        str(args.feature_batch_size),
        "--hidden-dim",
        str(args.hidden_dim),
        "--epochs",
        str(args.epochs),
        "--learning-rate",
        str(args.learning_rate),
        "--lambda-ssr",
        str(job.spec.lambda_ssr),
        "--a-exc",
        "1.0",
        "--a-inh",
        "0.8",
        "--sigma-exc",
        str(job.spec.sigma_exc),
        "--sigma-inh",
        str(job.spec.sigma_inh),
        "--kernel-family",
        "gaussian",
        "--selection-lock-sha256",
        args.selection_lock_sha256
        if job.phase == "confirmation"
        else "development_screen",
    ]
    return output


def validate_job_record(
    job: Job,
    args: argparse.Namespace,
    expected_git_commit: str,
    dataset_manifests: dict[str, dict],
    cache_manifests: dict[str, dict],
) -> dict:
    path = result_record(job)
    try:
        record = validate_result_record(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, ResultSchemaError) as error:
        raise RuntimeError(f"Invalid result record {path}: {error}") from error
    uses_ssr = job.method == "task_ssr"
    expected = {
        "git_commit": expected_git_commit,
        "task_family": "segmentation",
        "dataset": f"{job.dataset}_masks",
        "model": "resnet18_dense_class_conditioned_decoder",
        "seed": job.seed,
        "recipe": "ssr_only" if uses_ssr else "plain",
        "distance_mapping": "cosine" if uses_ssr else "none",
        "selection_lock_hash": (
            args.selection_lock_sha256
            if job.phase == "confirmation"
            else "development_screen"
        ),
        "source_dataset_fingerprint": dataset_manifests[job.dataset][
            "dataset_fingerprint"
        ],
        "feature_cache_sha256": cache_manifests[job.dataset]["cache_sha256"],
        "encoder_state_sha256": cache_manifests[job.dataset]["encoder_state_sha256"],
        "torchvision_version": cache_manifests[job.dataset]["torchvision_version"],
        "task_schedule_hash": sha256_value(
            make_tasks(*TASK_CONFIGS[job.dataset], job.seed)
        ),
    }
    mismatches = {
        key: {"expected": value, "observed": record.get(key)}
        for key, value in expected.items()
        if record.get(key) != value
    }
    if uses_ssr:
        for key, value in {
            "sigma_exc": job.spec.sigma_exc,
            "sigma_inh": job.spec.sigma_inh,
        }.items():
            if float(record["kernel"].get(key, float("nan"))) != value:
                mismatches[f"kernel.{key}"] = {
                    "expected": value,
                    "observed": record["kernel"].get(key),
                }
        if float(record.get("lambda_ssr", float("nan"))) != job.spec.lambda_ssr:
            mismatches["lambda_ssr"] = {
                "expected": job.spec.lambda_ssr,
                "observed": record.get("lambda_ssr"),
            }
    for field in ("initial_model_hash", "task_schedule_hash", "encoder_state_sha256"):
        value = record.get(field)
        if not isinstance(value, str) or len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            mismatches[field] = {"expected": "lowercase SHA256", "observed": value}
    optimizer_steps = record.get("optimizer_steps")
    if not isinstance(optimizer_steps, int) or optimizer_steps <= 0:
        mismatches["optimizer_steps"] = {
            "expected": "positive integer",
            "observed": optimizer_steps,
        }
    if mismatches:
        raise RuntimeError(
            f"Result identity mismatch for {path}: {json.dumps(mismatches, sort_keys=True)}"
        )
    required_metrics = {"mean_iou", "mean_dice", "avg_forgetting_iou"}
    if not required_metrics <= set(record["metrics"]):
        raise RuntimeError(f"Result lacks primary metrics: {path}")
    return record


def validate_paired_audit_fields(
    control: dict,
    treatment: dict,
    *,
    identity: str,
) -> None:
    fields = (
        "initial_model_hash",
        "task_schedule_hash",
        "optimizer_steps",
        "encoder_state_sha256",
        "feature_cache_sha256",
    )
    mismatches = {
        field: {"control": control.get(field), "treatment": treatment.get(field)}
        for field in fields
        if control.get(field) != treatment.get(field)
    }
    if mismatches:
        raise RuntimeError(f"Unmatched segmentation pair {identity}: {mismatches}")


def run_worker(
    gpu: str,
    jobs: list[Job],
    args: argparse.Namespace,
    expected_git_commit: str,
    dataset_manifests: dict[str, dict],
    cache_manifests: dict[str, dict],
) -> None:
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = gpu
    for job in jobs:
        record_path = result_record(job)
        if record_path.is_file():
            validate_job_record(
                job,
                args,
                expected_git_commit,
                dataset_manifests,
                cache_manifests,
            )
            with PRINT_LOCK:
                print(f"SKIP gpu={gpu} {job.output}", flush=True)
            continue
        job.output.mkdir(parents=True, exist_ok=True)
        log_path = job.output / "launcher.log"
        failure_path = job.output / "FAILED.json"
        with PRINT_LOCK:
            print(
                f"RUN gpu={gpu} phase={job.phase} dataset={job.dataset} "
                f"method={job.method} spec={job.spec.spec_id} seed={job.seed}",
                flush=True,
            )
        with log_path.open("a", encoding="utf-8") as log:
            completed = subprocess.run(
                command(job, args),
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
                    "job": {
                        **asdict(job),
                        "output": str(job.output),
                    },
                    "log": str(log_path),
                },
            )
            raise RuntimeError(f"Job failed: {job.output}; see {log_path}")
        validate_job_record(
            job,
            args,
            expected_git_commit,
            dataset_manifests,
            cache_manifests,
        )


def run_jobs(
    jobs: list[Job],
    args: argparse.Namespace,
    expected_git_commit: str,
    dataset_manifests: dict[str, dict],
    cache_manifests: dict[str, dict],
) -> None:
    groups = {}
    for job in jobs:
        groups.setdefault((job.dataset, job.seed), []).append(job)
    ordered_groups = [groups[key] for key in sorted(groups)]
    assignments = [
        [job for group in ordered_groups[index :: len(args.gpus)] for job in group]
        for index in range(len(args.gpus))
    ]
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        futures = [
            pool.submit(
                run_worker,
                gpu,
                assigned,
                args,
                expected_git_commit,
                dataset_manifests,
                cache_manifests,
            )
            for gpu, assigned in zip(args.gpus, assignments)
            if assigned
        ]
        for future in futures:
            future.result()


def select_spec(
    jobs: list[Job],
    result_root: Path,
    args: argparse.Namespace,
    git_commit: str,
    dataset_manifests: dict[str, dict],
    cache_manifests: dict[str, dict],
) -> Spec:
    records = {
        (job.dataset, job.seed, job.method, job.spec.spec_id): validate_job_record(
            job, args, git_commit, dataset_manifests, cache_manifests
        )
        for job in jobs
    }
    summaries = []
    for spec in SPECS:
        dataset_summaries = {}
        all_changes = []
        for dataset in DATASETS:
            changes = {"miou": [], "dice": [], "forgetting": []}
            triple_wins = 0
            for seed in DEVELOPMENT_SEEDS:
                control_record = records[(dataset, seed, "task_only", CONTROL.spec_id)]
                treatment_record = records[(dataset, seed, "task_ssr", spec.spec_id)]
                validate_paired_audit_fields(
                    control_record,
                    treatment_record,
                    identity=f"development/{dataset}/{spec.spec_id}/seed_{seed}",
                )
                control = control_record["metrics"]
                treatment = treatment_record["metrics"]
                row = {
                    "miou": float(treatment["mean_iou"]) - float(control["mean_iou"]),
                    "dice": float(treatment["mean_dice"]) - float(control["mean_dice"]),
                    "forgetting": float(control["avg_forgetting_iou"])
                    - float(treatment["avg_forgetting_iou"]),
                }
                triple_wins += int(all(value > 0 for value in row.values()))
                for metric, value in row.items():
                    changes[metric].append(value)
                    all_changes.append(value)
            means = {metric: statistics.mean(values) for metric, values in changes.items()}
            dataset_summaries[dataset] = {
                "mean_changes": means,
                "all_means_positive": all(value > 0 for value in means.values()),
                "triple_wins": triple_wins,
            }
        dataset_positive = sum(
            row["all_means_positive"] for row in dataset_summaries.values()
        )
        summaries.append(
            {
                "spec": asdict(spec),
                "datasets": dataset_summaries,
                "dataset_positive_count": dataset_positive,
                "minimum_change": min(all_changes),
                "mean_change": statistics.mean(all_changes),
                "triple_wins": sum(
                    row["triple_wins"] for row in dataset_summaries.values()
                ),
            }
        )
    summaries.sort(
        key=lambda row: (
            row["dataset_positive_count"],
            row["minimum_change"],
            row["triple_wins"],
            row["mean_change"],
        ),
        reverse=True,
    )
    selected = Spec(**summaries[0]["spec"])
    inventory = [
        {
            "path": str(result_record(job).relative_to(result_root)),
            "sha256": sha256_file(result_record(job)),
        }
        for job in jobs
    ]
    payload = {
        "protocol": "multidataset_segmentation_pooled_lock_v1",
        "git_commit": git_commit,
        "development_seeds": list(DEVELOPMENT_SEEDS),
        "datasets": list(DATASETS),
        "selection_rule": (
            "lexicographic: dataset-positive count, worst cell change, triple wins, "
            "pooled mean change; metrics are delta mIoU, delta Dice, and forgetting reduction"
        ),
        "selected": asdict(selected),
        "ranked_candidates": summaries,
        "dataset_fingerprints": {
            dataset: dataset_manifests[dataset]["dataset_fingerprint"]
            for dataset in DATASETS
        },
        "feature_cache_sha256": {
            dataset: cache_manifests[dataset]["cache_sha256"] for dataset in DATASETS
        },
        "screen_result_records": inventory,
    }
    payload["lock_sha256"] = sha256_value(payload)
    write_json_atomic(result_root / "SELECTION_LOCK.json", payload)
    return selected


def validate_selection_lock(
    result_root: Path,
    args: argparse.Namespace,
    git_commit: str,
    dataset_manifests: dict[str, dict],
    cache_manifests: dict[str, dict],
) -> Spec:
    path = result_root / "SELECTION_LOCK.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Invalid selection lock: {path}: {error}") from error
    unhashed = {key: value for key, value in payload.items() if key != "lock_sha256"}
    if payload.get("lock_sha256") != sha256_value(unhashed):
        raise RuntimeError(f"Selection lock hash mismatch: {path}")
    if payload.get("git_commit") != git_commit:
        raise RuntimeError("Selection lock git commit mismatch")
    expected_data = {
        dataset: dataset_manifests[dataset]["dataset_fingerprint"]
        for dataset in DATASETS
    }
    expected_cache = {
        dataset: cache_manifests[dataset]["cache_sha256"] for dataset in DATASETS
    }
    if payload.get("dataset_fingerprints") != expected_data:
        raise RuntimeError("Selection lock dataset fingerprint mismatch")
    if payload.get("feature_cache_sha256") != expected_cache:
        raise RuntimeError("Selection lock feature-cache fingerprint mismatch")
    expected_records = {
        str(result_record(job).relative_to(result_root)): sha256_file(result_record(job))
        for job in screen_jobs(result_root)
    }
    observed_records = {
        row.get("path"): row.get("sha256")
        for row in payload.get("screen_result_records", [])
        if isinstance(row, dict)
    }
    if observed_records != expected_records:
        raise RuntimeError("Selection lock screen inventory mismatch")
    return Spec(**payload["selected"])


def write_confirmation_summary(
    jobs: list[Job],
    result_root: Path,
    args: argparse.Namespace,
    git_commit: str,
    dataset_manifests: dict[str, dict],
    cache_manifests: dict[str, dict],
) -> None:
    rows = []
    json_summary = {}
    for dataset in DATASETS:
        records = {
            (job.seed, job.method): validate_job_record(
                job, args, git_commit, dataset_manifests, cache_manifests
            )
            for job in jobs
            if job.dataset == dataset
        }
        json_summary[dataset] = {}
        for seed in CONFIRMATION_SEEDS:
            validate_paired_audit_fields(
                records[(seed, "task_only")],
                records[(seed, "task_ssr")],
                identity=f"confirmation/{dataset}/seed_{seed}",
            )
        for metric, higher_is_better in (
            ("mean_iou", True),
            ("mean_dice", True),
            ("avg_forgetting_iou", False),
        ):
            controls = [records[(seed, "task_only")]["metrics"][metric] for seed in CONFIRMATION_SEEDS]
            treatments = [records[(seed, "task_ssr")]["metrics"][metric] for seed in CONFIRMATION_SEEDS]
            summary = paired_summary(
                controls, treatments, higher_is_better=higher_is_better
            ).as_dict()
            json_summary[dataset][metric] = summary
            rows.append({"dataset": dataset, "metric": metric, **summary})
    write_json_atomic(result_root / "CONFIRMATION_SUMMARY.json", json_summary)
    with (result_root / "CONFIRMATION_SUMMARY.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare_inputs(args: argparse.Namespace) -> tuple[dict[str, dict], dict[str, dict]]:
    if args.prepare_data:
        subprocess.run(
            [
                args.python,
                str(PREPARER),
                "--data-root",
                str(args.data_root),
                "--datasets",
                *DATASETS,
            ],
            cwd=PROJECT_ROOT,
            check=True,
        )
    dataset_manifests = {}
    cache_manifests = {}
    for dataset in DATASETS:
        manifest_path = args.data_root / dataset / "DATASET_MANIFEST.json"
        dataset_manifests[dataset] = json.loads(manifest_path.read_text(encoding="utf-8"))
        cache = cache_path(args.data_root, dataset, args.image_size)
        output = args.result_root / "cache_preparation" / dataset
        subprocess.run(
            [
                args.python,
                str(RUNNER),
                "--dataset",
                dataset,
                "--data-root",
                str(args.data_root),
                "--feature-cache",
                str(cache),
                "--output-dir",
                str(output),
                "--device",
                "cuda",
                "--workers",
                str(args.workers),
                "--image-size",
                str(args.image_size),
                "--feature-batch-size",
                str(args.feature_batch_size),
                "--prepare-cache-only",
            ],
            cwd=PROJECT_ROOT,
            check=True,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": args.gpus[0]},
        )
        cache_manifests[dataset] = json.loads(
            cache.with_name(cache.name + ".manifest.json").read_text(encoding="utf-8")
        )
    args.cache_sha256 = {
        dataset: cache_manifests[dataset]["cache_sha256"] for dataset in DATASETS
    }
    return dataset_manifests, cache_manifests


def manifest(args: argparse.Namespace) -> dict:
    screen = screen_jobs(args.result_root)
    confirmation = confirmation_jobs(args.result_root, SPECS[0])
    return {
        "protocol": "multidataset_segmentation_direct_ssr_v1",
        "primary_contrast": "task_ssr_minus_task_only",
        "datasets": list(DATASETS),
        "development_seeds": list(DEVELOPMENT_SEEDS),
        "confirmation_seeds": list(CONFIRMATION_SEEDS),
        "development_specs": [asdict(spec) for spec in SPECS],
        "training": {
            "image_size": args.image_size,
            "epochs_per_task": args.epochs,
            "batch_size": args.batch_size,
            "hidden_dim": args.hidden_dim,
            "learning_rate": args.learning_rate,
            "encoder": "frozen torchvision ResNet-18 through layer3",
            "decoder": "matched class-conditioned binary mask decoder",
        },
        "jobs": {
            "development": len(screen),
            "confirmation": len(confirmation),
            "total": len(screen) + len(confirmation),
        },
        "development_job_ids": [str(job.output.relative_to(args.result_root)) for job in screen],
        "confirmation_job_templates": [
            str(job.output.relative_to(args.result_root)) for job in confirmation
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--phase", choices=("all", "screen", "confirm"), default="all")
    parser.add_argument("--gpus", nargs="+", default=[str(index) for index in range(8)])
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--prepare-data", action="store_true")
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--feature-batch-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()
    args.result_root = args.result_root.resolve()
    args.data_root = args.data_root.resolve()
    args.selection_lock_sha256 = "development_screen"
    args.cache_sha256 = {}
    return args


def main() -> None:
    args = parse_args()
    args.result_root.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.result_root / "MANIFEST.json", manifest(args))
    if args.manifest_only:
        print(json.dumps(manifest(args), indent=2))
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
    dataset_manifests, cache_manifests = prepare_inputs(args)
    selected = None
    if args.phase in {"all", "screen"}:
        jobs = screen_jobs(args.result_root)
        run_jobs(jobs, args, git_commit, dataset_manifests, cache_manifests)
        selected = select_spec(
            jobs,
            args.result_root,
            args,
            git_commit,
            dataset_manifests,
            cache_manifests,
        )
    if args.phase in {"all", "confirm"}:
        if selected is None:
            selected = validate_selection_lock(
                args.result_root,
                args,
                git_commit,
                dataset_manifests,
                cache_manifests,
            )
        lock = json.loads(
            (args.result_root / "SELECTION_LOCK.json").read_text(encoding="utf-8")
        )
        args.selection_lock_sha256 = lock["lock_sha256"]
        jobs = confirmation_jobs(args.result_root, selected)
        run_jobs(jobs, args, git_commit, dataset_manifests, cache_manifests)
        write_confirmation_summary(
            jobs,
            args.result_root,
            args,
            git_commit,
            dataset_manifests,
            cache_manifests,
        )
    print(f"Multidataset segmentation phase {args.phase} complete: {args.result_root}")


if __name__ == "__main__":
    main()
