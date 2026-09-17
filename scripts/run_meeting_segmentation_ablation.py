#!/usr/bin/env python3
"""Resumable CUB mask screen-lock-confirm segmentation protocols.

``legacy_kd_screen`` preserves the original KD-only screen used by the
meeting-revision wrapper.  ``factorial_2x2`` is the auditable primary
protocol: one SSR setting is selected jointly on task-only and KD contexts,
then all four factorial arms are confirmed on held-out paired seeds.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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

from ssr_utils.paired_stats import paired_summary
from ssr_utils.result_schema import ResultSchemaError, sha256_file, validate_result_record


RUNNER = PROJECT_ROOT / "experiments" / "cub200_continual_benchmark.py"
DEVELOPMENT_SEEDS = (9101, 9103, 9105)
# Retained for the legacy meeting wrapper. The factorial protocol requires the
# distinct 20-seed cohort below.
CONFIRMATION_SEEDS = tuple(range(9201, 9221, 2))
FACTORIAL_CONFIRMATION_SEEDS = tuple(range(9201, 9240, 2))
METHODS = ("baseline", "kd", "biocs", "biocs_kd")
PROTOCOL_LEGACY = "legacy_kd_screen"
PROTOCOL_FACTORIAL = "factorial_2x2"
EXPECTED_RECIPES = {
    "baseline": "plain",
    "kd": "kd",
    "biocs": "ssr_only",
    "biocs_kd": "ssr_kd",
}
PRIMARY_METRICS = (
    ("mean_iou", True),
    ("mean_dice", True),
    ("avg_forgetting_iou", False),
)
OPTIONAL_AUDIT_HASH_FIELDS = (
    "initial_model_hash",
    "task_schedule_hash",
    "segmentation_source_sha256",
    "segmentation_cache_sha256",
)
OPTIONAL_AUDIT_EQUALITY_FIELDS = (
    "optimizer_steps",
    "lambda_kd",
    "lambda_kd_seg",
)


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


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_csv_atomic(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write an empty summary: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


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


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _objective_for_method(method: str) -> dict[str, bool]:
    return {
        "task": True,
        "kd": method in {"kd", "biocs_kd"},
        "ssr": method in {"biocs", "biocs_kd"},
        "anchor": False,
        "spectral": False,
        "ewc": False,
        "mas": False,
        "si": False,
        "center": False,
        "protodecor": False,
    }


def _protocol_from_args(args: argparse.Namespace) -> str:
    return getattr(args, "protocol", PROTOCOL_LEGACY)


def _confirmation_seeds_from_args(args: argparse.Namespace) -> tuple[int, ...]:
    """Keep programmatic legacy callers compatible with the old defaults."""

    return tuple(getattr(args, "confirmation_seeds", CONFIRMATION_SEEDS))


def _confirmation_methods_from_args(args: argparse.Namespace) -> tuple[str, ...]:
    """Keep programmatic legacy callers compatible with the old defaults."""

    return tuple(getattr(args, "confirmation_methods", METHODS))


def _selection_contexts(protocol: str) -> tuple[tuple[str, str, str], ...]:
    if protocol == PROTOCOL_LEGACY:
        return (("kd_to_kd_ssr", "kd", "biocs_kd"),)
    if protocol == PROTOCOL_FACTORIAL:
        return (
            ("task_to_task_ssr", "baseline", "biocs"),
            ("kd_to_kd_ssr", "kd", "biocs_kd"),
        )
    raise ValueError(f"Unknown segmentation protocol: {protocol}")


def validate_job_record(
    job: Job,
    expected_git_commit: str,
    expected_selection_lock_hash: str,
    *,
    expected_segmentation_source_sha256: str | None = None,
    expected_segmentation_cache_sha256: str | None = None,
) -> dict:
    path = result_record(job)
    try:
        record = validate_result_record(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, ResultSchemaError) as error:
        raise RuntimeError(f"Invalid result record {path}: {error}") from error

    uses_ssr = "biocs" in job.method
    expected = {
        "git_commit": expected_git_commit,
        "task_family": "segmentation",
        "dataset": "cub200_masks",
        "model": "resnet18_dense_decoder",
        "seed": job.seed,
        "recipe": EXPECTED_RECIPES[job.method],
        "objective": _objective_for_method(job.method),
        "distance_mapping": "cosine" if uses_ssr else "none",
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

    expected_kernel = (
        {
            "family": "gaussian",
            "A_exc": 1.0,
            "A_inh": 0.8,
            "sigma_exc": job.spec.sigma_exc,
            "sigma_inh": job.spec.sigma_inh,
        }
        if uses_ssr
        else {}
    )
    if record.get("kernel") != expected_kernel:
        mismatches["kernel"] = {
            "expected": expected_kernel,
            "observed": record.get("kernel"),
        }

    expected_lambda_ssr = job.spec.lambda_sp if uses_ssr else 0.0
    if "lambda_ssr" in record:
        try:
            lambda_matches = float(record["lambda_ssr"]) == expected_lambda_ssr
        except (TypeError, ValueError):
            lambda_matches = False
        if not lambda_matches:
            mismatches["lambda_ssr"] = {
                "expected": expected_lambda_ssr,
                "observed": record.get("lambda_ssr"),
            }

    # New trainer records carry these fields. Older records remain readable so
    # the legacy protocol and its artifacts are not invalidated retroactively.
    expected_audit_hashes = {
        "segmentation_source_sha256": expected_segmentation_source_sha256,
        "segmentation_cache_sha256": expected_segmentation_cache_sha256,
    }
    for field in OPTIONAL_AUDIT_HASH_FIELDS:
        if field not in record:
            continue
        if not _is_sha256(record[field]):
            mismatches[field] = {
                "expected": "lowercase SHA256 when present",
                "observed": record.get(field),
            }
        elif (
            expected_audit_hashes.get(field) is not None
            and record[field] != expected_audit_hashes[field]
        ):
            mismatches[field] = {
                "expected": expected_audit_hashes[field],
                "observed": record.get(field),
            }
    if "optimizer_steps" in record and (
        not isinstance(record["optimizer_steps"], int)
        or isinstance(record["optimizer_steps"], bool)
        or record["optimizer_steps"] <= 0
    ):
        mismatches["optimizer_steps"] = {
            "expected": "positive integer when present",
            "observed": record.get("optimizer_steps"),
        }

    required_metrics = {metric for metric, _ in PRIMARY_METRICS}
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


def validate_paired_records(
    control: dict,
    treatment: dict,
    *,
    identity: str,
    allowed_objective_changes: tuple[str, ...],
) -> None:
    """Reject mismatched paired runs while allowing the named factorial delta."""

    fields = (
        "git_commit",
        "task_family",
        "dataset",
        "model",
        "seed",
        "dataset_hash",
        "selection_lock_hash",
    )
    mismatches = {
        field: {"control": control.get(field), "treatment": treatment.get(field)}
        for field in fields
        if control.get(field) != treatment.get(field)
    }
    for field in OPTIONAL_AUDIT_HASH_FIELDS + OPTIONAL_AUDIT_EQUALITY_FIELDS:
        control_value = control.get(field)
        treatment_value = treatment.get(field)
        if control_value is None and treatment_value is None:
            continue
        if control_value != treatment_value:
            mismatches[field] = {
                "control": control_value,
                "treatment": treatment_value,
            }

    control_objective = dict(control.get("objective", {}))
    treatment_objective = dict(treatment.get("objective", {}))
    for component in allowed_objective_changes:
        control_objective.pop(component, None)
        treatment_objective.pop(component, None)
    if control_objective != treatment_objective:
        mismatches["objective"] = {
            "control": control.get("objective"),
            "treatment": treatment.get("objective"),
            "allowed_changes": list(allowed_objective_changes),
        }
    if mismatches:
        raise RuntimeError(f"Unmatched segmentation pair {identity}: {mismatches}")


def command(job: Job, args: argparse.Namespace) -> list[str]:
    output = [
        args.python,
        str(RUNNER),
        "--data_root",
        str(args.data_root),
        "--segmentation_cache",
        str(args.segmentation_cache),
        "--segmentation_source_sha256",
        args.segmentation_source_sha256,
        "--segmentation_cache_sha256",
        args.segmentation_cache_sha256,
        "--output_dir",
        str(job.output),
        "--tasks",
        "segmentation",
        "--methods",
        job.method,
        "--seeds",
        str(job.seed),
        "--num_classes",
        "200",
        "--classes_per_task",
        "10",
        "--class_order",
        "semantic",
        "--seg_epochs",
        "24",
        "--seg_batch_size",
        str(args.seg_batch_size),
        "--seg_image_size",
        "192",
        "--seg_hidden_dim",
        "256",
        "--seg_biocs_target",
        "class",
        "--seg_biocs_scope",
        "seen",
        "--lambda_sp",
        str(job.spec.lambda_sp),
        "--lambda_kd_seg",
        "2.0",
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
        "--workers",
        str(args.workers),
        "--device",
        "cuda",
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
    expected_source_hash = getattr(args, "segmentation_source_sha256", None)
    expected_cache_hash = getattr(args, "segmentation_cache_sha256", None)
    for job in jobs:
        record = result_record(job)
        selection_hash = getattr(args, "selection_lock_sha256", "development_screen")
        if record.is_file():
            validate_job_record(
                job,
                expected_git_commit,
                selection_hash,
                expected_segmentation_source_sha256=expected_source_hash,
                expected_segmentation_cache_sha256=expected_cache_hash,
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
            write_json_atomic(
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
            selection_hash,
            expected_segmentation_source_sha256=expected_source_hash,
            expected_segmentation_cache_sha256=expected_cache_hash,
        )


def run_jobs(jobs: list[Job], args: argparse.Namespace, expected_git_commit: str) -> None:
    slots = [gpu for gpu in args.gpus for _ in range(args.jobs_per_gpu)]
    if not slots:
        raise ValueError("at least one GPU slot is required")
    assignments = [jobs[index :: len(slots)] for index in range(len(slots))]
    with ThreadPoolExecutor(max_workers=len(slots)) as pool:
        futures = [
            pool.submit(run_worker, gpu, assigned, args, expected_git_commit)
            for gpu, assigned in zip(slots, assignments)
            if assigned
        ]
        for future in futures:
            future.result()


def _favorable_delta(
    control: dict,
    treatment: dict,
    metric: str,
    higher_is_better: bool,
) -> float:
    delta = float(treatment["metrics"][metric]) - float(control["metrics"][metric])
    return delta if higher_is_better else -delta


def screen_jobs(
    result_root: Path,
    protocol: str = PROTOCOL_LEGACY,
) -> list[Job]:
    jobs: list[Job] = []
    for seed in DEVELOPMENT_SEEDS:
        if protocol == PROTOCOL_LEGACY:
            jobs.append(
                Job(
                    "development",
                    "kd",
                    seed,
                    CONTROL_SPEC,
                    result_root / "development" / f"kd_control_seed{seed}",
                )
            )
            for spec in SPECS:
                jobs.append(
                    Job(
                        "development",
                        "biocs_kd",
                        seed,
                        spec,
                        result_root
                        / "development"
                        / f"biocs_kd_{spec.spec_id}_seed{seed}",
                    )
                )
        elif protocol == PROTOCOL_FACTORIAL:
            for method in ("baseline", "kd"):
                jobs.append(
                    Job(
                        "development",
                        method,
                        seed,
                        CONTROL_SPEC,
                        result_root / "development" / f"{method}_control_seed{seed}",
                    )
                )
            for spec in SPECS:
                for method in ("biocs", "biocs_kd"):
                    jobs.append(
                        Job(
                            "development",
                            method,
                            seed,
                            spec,
                            result_root
                            / "development"
                            / f"{method}_{spec.spec_id}_seed{seed}",
                        )
                    )
        else:
            raise ValueError(f"Unknown segmentation protocol: {protocol}")
    return jobs


def confirmation_jobs(
    result_root: Path,
    spec: Spec,
    seeds: tuple[int, ...] = CONFIRMATION_SEEDS,
    methods: tuple[str, ...] = METHODS,
) -> list[Job]:
    return [
        Job(
            "confirmation",
            method,
            seed,
            spec,
            result_root / "confirmation" / f"{method}_{spec.spec_id}_seed{seed}",
        )
        for seed in seeds
        for method in methods
    ]


def select_spec(
    jobs: list[Job],
    result_root: Path,
    args: argparse.Namespace,
    git_commit: str,
) -> Spec:
    protocol = _protocol_from_args(args)
    expected_jobs = screen_jobs(result_root, protocol)
    if {job.output for job in jobs} != {job.output for job in expected_jobs}:
        raise RuntimeError("Selection jobs do not match the prespecified screen inventory")
    records = {
        (job.seed, job.method, job.spec.spec_id): validate_job_record(
            job,
            git_commit,
            "development_screen",
            expected_segmentation_source_sha256=args.segmentation_source_sha256,
            expected_segmentation_cache_sha256=args.segmentation_cache_sha256,
        )
        for job in jobs
    }
    contexts = _selection_contexts(protocol)
    summaries = []
    for spec in SPECS:
        context_summaries = {}
        all_means: list[float] = []
        all_triple_wins = 0
        for context_name, control_method, treatment_method in contexts:
            changes = {metric: [] for metric, _ in PRIMARY_METRICS}
            per_seed = []
            for seed in DEVELOPMENT_SEEDS:
                control = records[(seed, control_method, CONTROL_SPEC.spec_id)]
                treatment = records[(seed, treatment_method, spec.spec_id)]
                validate_paired_records(
                    control,
                    treatment,
                    identity=f"development/{context_name}/seed_{seed}/{spec.spec_id}",
                    allowed_objective_changes=("ssr",),
                )
                row = {"seed": seed}
                for metric, higher_is_better in PRIMARY_METRICS:
                    value = _favorable_delta(control, treatment, metric, higher_is_better)
                    changes[metric].append(value)
                    row[metric] = value
                per_seed.append(row)
            means = {metric: statistics.mean(values) for metric, values in changes.items()}
            all_means.extend(means.values())
            all_triple_wins += sum(
                all(row[metric] > 0 for metric, _ in PRIMARY_METRICS)
                for row in per_seed
            )
            context_summaries[context_name] = {
                "control": control_method,
                "treatment": treatment_method,
                "mean_changes": means,
                "all_means_positive": all(value > 0 for value in means.values()),
                "triple_wins": sum(
                    all(row[metric] > 0 for metric, _ in PRIMARY_METRICS)
                    for row in per_seed
                ),
                "per_seed": per_seed,
            }
        positive_context_count = sum(
            row["all_means_positive"] for row in context_summaries.values()
        )
        summaries.append(
            {
                "spec": asdict(spec),
                "contexts": context_summaries,
                "all_contexts_positive": positive_context_count == len(contexts),
                "positive_context_count": positive_context_count,
                "positive_endpoint_count": sum(value > 0 for value in all_means),
                "balanced_score": min(all_means),
                "triple_wins": all_triple_wins,
                "sum_score": sum(all_means),
            }
        )
    summaries.sort(
        key=lambda row: (
            row["all_contexts_positive"],
            row["positive_context_count"],
            row["positive_endpoint_count"],
            row["balanced_score"],
            row["triple_wins"],
            row["sum_score"],
        ),
        reverse=True,
    )
    selected = Spec(**summaries[0]["spec"])
    if protocol == PROTOCOL_FACTORIAL:
        selection_rule = (
            "Joint factorial lexicographic rule across task-only and KD contexts: "
            "positive-context count, positive endpoint count, worst favorable "
            "mean change, triple wins, then summed favorable changes. Endpoints "
            "are delta mIoU, delta Dice, and IoU-forgetting reduction."
        )
    else:
        selection_rule = (
            "KD-context lexicographic rule: positive endpoint count, worst favorable "
            "mean change, triple wins, then summed favorable changes."
        )
    payload = {
        "protocol": protocol,
        "selected": asdict(selected),
        "selection_rule": selection_rule,
        "selection_contexts": [context[0] for context in contexts],
        "development_seeds": list(DEVELOPMENT_SEEDS),
        "confirmation_seeds_hidden_during_selection": list(
            _confirmation_seeds_from_args(args)
        ),
        "confirmation_methods": list(_confirmation_methods_from_args(args)),
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
    payload["lock_sha256"] = _canonical_sha256(payload)
    write_json_atomic(result_root / "SELECTION_LOCK.json", payload)
    return selected


def validate_selection_lock(
    lock_path: Path,
    args: argparse.Namespace,
    git_commit: str,
) -> Spec:
    try:
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Invalid selection lock: {lock_path}: {error}") from error
    observed_hash = payload.get("lock_sha256")
    unhashed = {key: value for key, value in payload.items() if key != "lock_sha256"}
    expected_hash = _canonical_sha256(unhashed)
    if observed_hash != expected_hash:
        raise RuntimeError(
            f"Selection lock hash mismatch: expected {expected_hash}, observed {observed_hash}"
        )
    protocol = _protocol_from_args(args)
    expected_identity = {
        "protocol": protocol,
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
    if payload.get("confirmation_seeds_hidden_during_selection") != list(
        _confirmation_seeds_from_args(args)
    ):
        raise RuntimeError("Selection lock confirmation-seed inventory mismatch")
    if payload.get("confirmation_methods") != list(
        _confirmation_methods_from_args(args)
    ):
        raise RuntimeError("Selection lock confirmation-method inventory mismatch")
    if payload.get("selection_contexts") != [
        context[0] for context in _selection_contexts(protocol)
    ]:
        raise RuntimeError("Selection lock context inventory mismatch")
    selected_payload = payload.get("selected")
    try:
        selected = Spec(**selected_payload)
    except (TypeError, ValueError) as error:
        raise RuntimeError("Selection lock has an invalid selected specification") from error
    if selected not in SPECS:
        raise RuntimeError(f"Selection lock contains an unplanned spec: {selected}")
    screen_records = payload.get("screen_result_records")
    expected_paths = {
        str(result_record(job).relative_to(lock_path.parent))
        for job in screen_jobs(lock_path.parent, protocol)
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


def _confirmation_contrasts(
    methods: set[str],
    *,
    require_factorial: bool,
) -> tuple[tuple[str, str, str, tuple[str, ...]], ...]:
    contrasts = (
        ("task_to_task_ssr", "baseline", "biocs", ("ssr",)),
        ("kd_to_kd_ssr", "kd", "biocs_kd", ("ssr",)),
        ("task_to_kd", "baseline", "kd", ("kd",)),
        ("task_ssr_to_kd_ssr", "biocs", "biocs_kd", ("kd",)),
        ("task_to_full_recipe", "baseline", "biocs_kd", ("kd", "ssr")),
    )
    if require_factorial:
        missing = sorted(set(METHODS) - methods)
        unexpected = sorted(methods - set(METHODS))
        if missing or unexpected:
            raise RuntimeError(
                "Factorial confirmation requires exactly all four arms: "
                f"missing={missing}, unexpected={unexpected}"
            )
        return contrasts
    return tuple(
        contrast
        for contrast in contrasts
        if {contrast[1], contrast[2]} <= methods
    )


def write_confirmation_summary(
    jobs: list[Job],
    result_root: Path,
    args: argparse.Namespace,
    git_commit: str,
) -> None:
    confirmation_seeds = _confirmation_seeds_from_args(args)
    confirmation_methods = _confirmation_methods_from_args(args)
    protocol = _protocol_from_args(args)
    observed_keys = [(job.seed, job.method) for job in jobs]
    if protocol == PROTOCOL_FACTORIAL:
        expected_keys = {
            (seed, method)
            for seed in confirmation_seeds
            for method in confirmation_methods
        }
        if (
            len(observed_keys) != len(expected_keys)
            or len(set(observed_keys)) != len(observed_keys)
            or set(observed_keys) != expected_keys
        ):
            raise RuntimeError(
                "Factorial confirmation inventory mismatch: "
                f"expected={len(expected_keys)}, observed={len(observed_keys)}, "
                f"missing={sorted(expected_keys - set(observed_keys))}, "
                f"unexpected={sorted(set(observed_keys) - expected_keys)}"
            )
    records = {
        (job.seed, job.method): validate_job_record(
            job,
            git_commit,
            args.selection_lock_sha256,
            expected_segmentation_source_sha256=getattr(
                args, "segmentation_source_sha256", None
            ),
            expected_segmentation_cache_sha256=getattr(
                args, "segmentation_cache_sha256", None
            ),
        )
        for job in jobs
    }
    methods = {job.method for job in jobs}
    contrasts = _confirmation_contrasts(
        methods,
        require_factorial=protocol == PROTOCOL_FACTORIAL,
    )
    if not contrasts:
        raise RuntimeError("Confirmation has no complete prespecified contrasts")
    for seed in confirmation_seeds:
        available = {method for candidate_seed, method in records if candidate_seed == seed}
        if protocol == PROTOCOL_FACTORIAL and available != set(METHODS):
            raise RuntimeError(
                f"Factorial confirmation seed {seed} is incomplete: {sorted(available)}"
            )

    payload = {
        "protocol": protocol,
        "selection_lock_sha256": args.selection_lock_sha256,
        "confirmation_seeds": list(confirmation_seeds),
        "confirmation_methods": list(confirmation_methods),
        "contrasts": {},
    }
    rows: list[dict] = []
    for contrast_name, control_method, treatment_method, allowed_changes in contrasts:
        for seed in confirmation_seeds:
            validate_paired_records(
                records[(seed, control_method)],
                records[(seed, treatment_method)],
                identity=f"confirmation/{contrast_name}/seed_{seed}",
                allowed_objective_changes=allowed_changes,
            )
        summary_payload = {
            "control": control_method,
            "treatment": treatment_method,
            "metrics": {},
        }
        for metric, higher_is_better in PRIMARY_METRICS:
            summary = paired_summary(
                [
                    records[(seed, control_method)]["metrics"][metric]
                    for seed in confirmation_seeds
                ],
                [
                    records[(seed, treatment_method)]["metrics"][metric]
                    for seed in confirmation_seeds
                ],
                higher_is_better=higher_is_better,
            ).as_dict()
            summary_payload["metrics"][metric] = summary
            rows.append(
                {
                    "summary_type": "contrast",
                    "contrast": contrast_name,
                    "control": control_method,
                    "treatment": treatment_method,
                    "metric": metric,
                    **summary,
                }
            )
        payload["contrasts"][contrast_name] = summary_payload

    if protocol == PROTOCOL_FACTORIAL:
        interaction = {
            "definition": "(KD+SSR - KD) - (task+SSR - task)",
            "control": "task_to_task_ssr",
            "treatment": "kd_to_kd_ssr",
            "metrics": {},
        }
        for metric, higher_is_better in PRIMARY_METRICS:
            task_effect = [
                _favorable_delta(
                    records[(seed, "baseline")],
                    records[(seed, "biocs")],
                    metric,
                    higher_is_better,
                )
                for seed in confirmation_seeds
            ]
            kd_effect = [
                _favorable_delta(
                    records[(seed, "kd")],
                    records[(seed, "biocs_kd")],
                    metric,
                    higher_is_better,
                )
                for seed in confirmation_seeds
            ]
            summary = paired_summary(task_effect, kd_effect).as_dict()
            interaction["metrics"][metric] = summary
            rows.append(
                {
                    "summary_type": "factorial_interaction",
                    "contrast": "ssr_by_kd_interaction",
                    "control": "task_to_task_ssr",
                    "treatment": "kd_to_kd_ssr",
                    "metric": metric,
                    **summary,
                }
            )
        payload["factorial_interaction"] = interaction

    write_json_atomic(result_root / "CONFIRMATION_SUMMARY.json", payload)
    write_csv_atomic(result_root / "CONFIRMATION_SUMMARY.csv", rows)


def manifest(args: argparse.Namespace) -> dict:
    protocol = _protocol_from_args(args)
    confirmation_seeds = _confirmation_seeds_from_args(args)
    confirmation_methods = _confirmation_methods_from_args(args)
    screen = screen_jobs(args.result_root, protocol)
    confirmation = confirmation_jobs(
        args.result_root,
        SPECS[0],
        confirmation_seeds,
        confirmation_methods,
    )
    return {
        "protocol": protocol,
        "development_seeds": list(DEVELOPMENT_SEEDS),
        "confirmation_seeds": list(confirmation_seeds),
        "development_specs": [asdict(spec) for spec in SPECS],
        "methods": list(METHODS),
        "confirmation_methods": list(confirmation_methods),
        "selection_contexts": [
            context[0] for context in _selection_contexts(protocol)
        ],
        "primary_contrasts": [
            "task_to_task_ssr",
            "kd_to_kd_ssr",
        ],
        "kd_protocol": "old_class_conditions_on_current_features_v1",
        "jobs": {
            "development": len(screen),
            "confirmation": len(confirmation),
            "total": len(screen) + len(confirmation),
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--phase", choices=["all", "screen", "confirm"], default="all")
    parser.add_argument(
        "--protocol",
        choices=(PROTOCOL_LEGACY, PROTOCOL_FACTORIAL),
        default=PROTOCOL_LEGACY,
    )
    parser.add_argument("--gpus", nargs="+", default=["0"])
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--jobs-per-gpu", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seg-batch-size", type=int, default=48)
    parser.add_argument(
        "--confirmation-seeds",
        nargs="+",
        type=int,
        default=list(CONFIRMATION_SEEDS),
    )
    parser.add_argument(
        "--confirmation-methods",
        nargs="+",
        choices=METHODS,
        default=list(METHODS),
    )
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data/cub200")
    parser.add_argument(
        "--segmentation-cache",
        type=Path,
        default=PROJECT_ROOT / "data/cub200/cub200_seg_resnet18_dense_192.pt",
    )
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args(argv)
    if args.jobs_per_gpu < 1 or args.workers < 0 or args.seg_batch_size < 1:
        parser.error(
            "jobs-per-gpu and seg-batch-size must be positive; workers cannot be negative"
        )
    if len(args.confirmation_seeds) < 2 or len(set(args.confirmation_seeds)) != len(
        args.confirmation_seeds
    ):
        parser.error("confirmation-seeds must contain at least two unique values")
    if len(set(args.confirmation_methods)) != len(args.confirmation_methods):
        parser.error("confirmation-methods must be unique")
    args.confirmation_seeds = tuple(args.confirmation_seeds)
    args.confirmation_methods = tuple(args.confirmation_methods)
    if args.protocol == PROTOCOL_FACTORIAL:
        if args.confirmation_seeds != FACTORIAL_CONFIRMATION_SEEDS:
            parser.error(
                "factorial_2x2 requires the prespecified 20-seed confirmation cohort"
            )
        if args.confirmation_methods != METHODS:
            parser.error("factorial_2x2 requires all four confirmation arms")
    return args


def main() -> None:
    args = parse_args()
    args.result_root = args.result_root.resolve()
    args.data_root = args.data_root.resolve()
    args.segmentation_cache = args.segmentation_cache.resolve()
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
    launch_manifest = manifest(args)
    launch_manifest["segmentation_source"] = {
        "path": str(segmentation_source.resolve()),
        "sha256": args.segmentation_source_sha256,
    }
    launch_manifest["segmentation_cache"] = {
        "path": str(args.segmentation_cache.resolve()),
        "sha256": args.segmentation_cache_sha256,
    }
    write_json_atomic(args.result_root / "MANIFEST.json", launch_manifest)

    selected = None
    if args.phase in {"all", "screen"}:
        jobs = screen_jobs(args.result_root, args.protocol)
        run_jobs(jobs, args, git_commit)
        selected = select_spec(jobs, args.result_root, args, git_commit)
        selected = validate_selection_lock(
            args.result_root / "SELECTION_LOCK.json", args, git_commit
        )
    if args.phase in {"all", "confirm"}:
        lock_path = args.result_root / "SELECTION_LOCK.json"
        if selected is None:
            if not lock_path.is_file():
                raise FileNotFoundError(
                    f"Confirmation requires a completed development lock: {lock_path}"
                )
            selected = validate_selection_lock(lock_path, args, git_commit)
        args.selection_lock_sha256 = json.loads(
            lock_path.read_text(encoding="utf-8")
        )["lock_sha256"]
        jobs = confirmation_jobs(
            args.result_root,
            selected,
            args.confirmation_seeds,
            args.confirmation_methods,
        )
        run_jobs(jobs, args, git_commit)
        write_confirmation_summary(jobs, args.result_root, args, git_commit)
    print(f"Segmentation phase {args.phase} complete: {args.result_root}")


if __name__ == "__main__":
    main()
