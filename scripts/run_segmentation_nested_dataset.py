#!/usr/bin/env python3
"""One-node locked segmentation screen--refine--confirm experiment.

Each invocation owns one dataset and uses all eight local GPUs.  A higher-level
four-node launcher can run CUB, Pet and Flowers invocations independently while
using the fourth node for the separate LoRA matrix.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ssr_utils.paired_stats import paired_summary
from ssr_utils.result_schema import (
    ResultSchemaError,
    sha256_file,
    sha256_value,
    validate_result_record,
)
from ssr_utils.segmentation_nested_search import (
    Candidate,
    candidate_matrix,
    lock_selection,
    select_top,
    validate_paired_identities,
)


CUB_RUNNER = PROJECT_ROOT / "experiments" / "cub200_continual_benchmark.py"
MULTIDATASET_RUNNER = PROJECT_ROOT / "experiments" / "multidataset_segmentation.py"
CONTEXTS = ("task", "kd")
SCREEN_SEEDS = (10101, 10103, 10105)
REFINE_SEEDS = (10201, 10203, 10205, 10207, 10209)
CONFIRMATION_SEEDS = tuple(range(10301, 10320, 2))
WIDTHS = ((0.12, 0.36), (0.188, 0.530), (0.26, 0.72))
LAMBDAS = (0.01, 0.05, 0.20)
CANDIDATES = candidate_matrix(
    ("gaussian", "laplace", "cauchy"),
    WIDTHS,
    LAMBDAS,
    (("class", "new_old"), ("channels", "all")),
)
if len(CANDIDATES) != 54:
    raise RuntimeError(f"Expected 54 prespecified candidates, got {len(CANDIDATES)}")

PRIMARY_METRICS = (
    ("mean_iou", True),
    ("mean_dice", True),
    ("avg_forgetting_iou", False),
)
GEOMETRY_METRICS = (
    ("effective_rank", True),
    ("mean_abs_offdiag_cosine", False),
)
SUMMARY_METRICS = PRIMARY_METRICS + GEOMETRY_METRICS
PRINT_LOCK = Lock()


@dataclass(frozen=True)
class Job:
    phase: str
    context: str
    seed: int
    target: str
    scope: str
    treatment: bool
    output: Path
    spec: Candidate | None = None

    @property
    def spec_id(self) -> str:
        return self.spec.spec_id if self.spec is not None else f"control_{self.target}"


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_csv_atomic(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise ValueError("CSV rows do not share one ordered schema")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def current_git_commit() -> str:
    return subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def method_name(dataset: str, context: str, treatment: bool) -> str:
    if dataset == "cub200":
        return {
            ("task", False): "baseline",
            ("task", True): "biocs",
            ("kd", False): "kd",
            ("kd", True): "biocs_kd",
        }[(context, treatment)]
    return {
        ("task", False): "task_only",
        ("task", True): "task_ssr",
        ("kd", False): "kd",
        ("kd", True): "kd_ssr",
    }[(context, treatment)]


def representative_for_target(target: str) -> Candidate:
    return next(candidate for candidate in CANDIDATES if candidate.target == target)


def control_jobs(result_root: Path, phase: str, seeds: Iterable[int]) -> list[Job]:
    jobs = []
    for context in CONTEXTS:
        for target, scope in (("class", "new_old"), ("channels", "all")):
            for seed in seeds:
                jobs.append(
                    Job(
                        phase=phase,
                        context=context,
                        seed=seed,
                        target=target,
                        scope=scope,
                        treatment=False,
                        spec=None,
                        output=result_root
                        / phase
                        / context
                        / f"control_{target}"
                        / f"seed_{seed}",
                    )
                )
    return jobs


def treatment_jobs(
    result_root: Path,
    phase: str,
    seeds: Iterable[int],
    candidates_by_context: dict[str, Iterable[Candidate]],
) -> list[Job]:
    jobs = []
    for context in CONTEXTS:
        for spec in candidates_by_context[context]:
            for seed in seeds:
                jobs.append(
                    Job(
                        phase=phase,
                        context=context,
                        seed=seed,
                        target=spec.target,
                        scope=spec.scope,
                        treatment=True,
                        spec=spec,
                        output=result_root
                        / phase
                        / context
                        / spec.spec_id
                        / f"seed_{seed}",
                    )
                )
    return jobs


def result_record(job: Job, dataset: str) -> Path:
    if dataset == "cub200":
        return (
            job.output
            / "result_records"
            / "segmentation"
            / method_name(dataset, job.context, job.treatment)
            / f"seed_{job.seed}"
            / "result_record.json"
        )
    return job.output / "result_record.json"


def stage_selection_hash(phase: str, lock_hash: str | None) -> str:
    if phase == "confirmation":
        if not lock_hash:
            raise ValueError("confirmation jobs require a selection-lock hash")
        return lock_hash
    return f"segmentation_nested_{phase}_v1"


def command(job: Job, args: argparse.Namespace, lock_hash: str | None) -> list[str]:
    spec = job.spec or representative_for_target(job.target)
    method = method_name(args.dataset, job.context, job.treatment)
    selection_hash = stage_selection_hash(job.phase, lock_hash)
    evaluation_split = "test" if job.phase == "confirmation" else "validation"
    if args.dataset == "cub200":
        epochs = {"screen": 8, "refine": 16, "confirmation": 24}[job.phase]
        return [
            args.python,
            str(CUB_RUNNER),
            "--data_root",
            str(args.dataset_root),
            "--segmentation_cache",
            str(args.feature_cache),
            "--segmentation_source_sha256",
            args.segmentation_source_sha256,
            "--segmentation_cache_sha256",
            args.feature_cache_sha256,
            "--selection_lock_sha256",
            selection_hash,
            "--output_dir",
            str(job.output),
            "--tasks",
            "segmentation",
            "--methods",
            method,
            "--seeds",
            str(job.seed),
            "--num_classes",
            "200",
            "--classes_per_task",
            "10",
            "--class_order",
            "semantic",
            "--seg_epochs",
            str(epochs),
            "--seg_batch_size",
            str(args.batch_size),
            "--seg_image_size",
            "192",
            "--seg_hidden_dim",
            "256",
            "--seg_biocs_target",
            spec.target,
            "--seg_biocs_scope",
            spec.scope,
            "--seg_biocs_start_task",
            "1",
            "--seg_evaluation_split",
            evaluation_split,
            "--seg_validation_fraction",
            "0.2",
            "--seg_validation_seed",
            "20260809",
            "--lambda_sp",
            str(spec.lambda_ssr),
            "--lambda_kd_seg",
            "2.0",
            "--a-exc",
            str(spec.a_exc),
            "--a-inh",
            str(spec.a_inh),
            "--sigma-exc",
            str(spec.sigma_exc),
            "--sigma-inh",
            str(spec.sigma_inh),
            "--kernel-family",
            spec.kernel,
            "--workers",
            str(args.workers),
            "--device",
            "cuda",
            "--no_download",
        ]
    epochs = {"screen": 8, "refine": 15, "confirmation": 20}[job.phase]
    return [
        args.python,
        str(MULTIDATASET_RUNNER),
        "--dataset",
        args.dataset,
        "--data-root",
        str(args.dataset_root.parent),
        "--dataset-root",
        str(args.dataset_root),
        "--feature-cache",
        str(args.feature_cache),
        "--feature-cache-sha256",
        args.feature_cache_sha256,
        "--output-dir",
        str(job.output),
        "--method",
        method,
        "--seed",
        str(job.seed),
        "--device",
        "cuda",
        "--workers",
        str(args.workers),
        "--image-size",
        "160",
        "--feature-batch-size",
        "64",
        "--batch-size",
        str(args.batch_size),
        "--hidden-dim",
        "192",
        "--epochs",
        str(epochs),
        "--learning-rate",
        "0.001",
        "--weight-decay",
        "0.0",
        "--lambda-kd",
        "2.0",
        "--lambda-ssr",
        str(spec.lambda_ssr),
        "--ssr-target",
        spec.target,
        "--ssr-scope",
        spec.scope,
        "--ssr-start-task",
        "1",
        "--evaluation-split",
        evaluation_split,
        "--validation-fraction",
        "0.2",
        "--validation-seed",
        "20260809",
        "--a-exc",
        str(spec.a_exc),
        "--a-inh",
        str(spec.a_inh),
        "--sigma-exc",
        str(spec.sigma_exc),
        "--sigma-inh",
        str(spec.sigma_inh),
        "--kernel-family",
        spec.kernel,
        "--selection-lock-sha256",
        selection_hash,
    ]


def expected_identity(args: argparse.Namespace) -> tuple[str, str]:
    if args.dataset == "cub200":
        return "cub200_masks", "resnet18_dense_decoder"
    return f"{args.dataset}_masks", "resnet18_dense_class_conditioned_decoder"


def load_args(job: Job) -> dict:
    try:
        return json.loads((job.output / "args.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Invalid job arguments for {job.output}: {error}") from error


def validate_job_record(
    job: Job,
    args: argparse.Namespace,
    git_commit: str,
    lock_hash: str | None,
) -> dict:
    path = result_record(job, args.dataset)
    try:
        record = validate_result_record(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, ResultSchemaError) as error:
        raise RuntimeError(f"Invalid segmentation record {path}: {error}") from error
    dataset_name, model_name = expected_identity(args)
    method = method_name(args.dataset, job.context, job.treatment)
    expected_recipe = {
        ("task", False): "plain",
        ("task", True): "ssr_only",
        ("kd", False): "kd",
        ("kd", True): "ssr_kd" if args.dataset == "cub200" else "kd_ssr",
    }[(job.context, job.treatment)]
    expected = {
        "git_commit": git_commit,
        "task_family": "segmentation",
        "dataset": dataset_name,
        "model": model_name,
        "seed": job.seed,
        "recipe": expected_recipe,
        "distance_mapping": "cosine" if job.treatment else "none",
        "selection_lock_hash": stage_selection_hash(job.phase, lock_hash),
    }
    mismatches = {
        key: {"expected": value, "observed": record.get(key)}
        for key, value in expected.items()
        if record.get(key) != value
    }
    objective = record["objective"]
    if objective["task"] is not True or objective["kd"] != (job.context == "kd"):
        mismatches["objective.context"] = {
            "expected": job.context,
            "observed": objective,
        }
    if objective["ssr"] != job.treatment:
        mismatches["objective.ssr"] = {
            "expected": job.treatment,
            "observed": objective["ssr"],
        }
    run_args = load_args(job)
    if args.dataset == "cub200":
        if run_args.get("methods") != [method]:
            mismatches["config.methods"] = {
                "expected": [method],
                "observed": run_args.get("methods"),
            }
    elif run_args.get("method") != method:
        mismatches["config.method"] = {
            "expected": method,
            "observed": run_args.get("method"),
        }
    if job.treatment:
        assert job.spec is not None
        kernel = record.get("kernel", {})
        expected_kernel = {
            "family": job.spec.kernel,
            "A_exc": job.spec.a_exc,
            "A_inh": job.spec.a_inh,
            "sigma_exc": job.spec.sigma_exc,
            "sigma_inh": job.spec.sigma_inh,
        }
        for key, value in expected_kernel.items():
            observed = kernel.get(key)
            matches = observed == value if isinstance(value, str) else math.isclose(
                float(observed), float(value), rel_tol=1e-12, abs_tol=1e-12
            )
            if not matches:
                mismatches[f"kernel.{key}"] = {"expected": value, "observed": observed}
        if not math.isclose(
            float(record.get("lambda_ssr", float("nan"))),
            job.spec.lambda_ssr,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            mismatches["lambda_ssr"] = {
                "expected": job.spec.lambda_ssr,
                "observed": record.get("lambda_ssr"),
            }
    required_metrics = {
        "mean_iou",
        "mean_dice",
        "avg_forgetting_iou",
        "effective_rank",
        "mean_abs_offdiag_cosine",
    }
    if not required_metrics <= set(record["metrics"]):
        mismatches["metrics"] = {
            "expected": sorted(required_metrics),
            "observed": sorted(record["metrics"]),
        }
    if mismatches:
        raise RuntimeError(f"Result identity mismatch for {path}: {mismatches}")
    return record


def validate_pair(
    control_job: Job,
    treatment_job: Job,
    args: argparse.Namespace,
    git_commit: str,
    lock_hash: str | None,
) -> tuple[dict, dict]:
    control = validate_job_record(control_job, args, git_commit, lock_hash)
    treatment = validate_job_record(treatment_job, args, git_commit, lock_hash)
    validate_paired_identities([control], [treatment])
    control_objective = dict(control["objective"])
    treatment_objective = dict(treatment["objective"])
    treatment_objective["ssr"] = False
    if treatment_objective != control_objective:
        raise RuntimeError(
            f"Objective mismatch for {treatment_job.output}: "
            f"{control_objective} vs {treatment_objective}"
        )
    control_args = load_args(control_job)
    treatment_args = load_args(treatment_job)
    if args.dataset == "cub200":
        fields = (
            "num_classes",
            "classes_per_task",
            "class_order",
            "seg_epochs",
            "seg_batch_size",
            "seg_image_size",
            "seg_hidden_dim",
            "seg_lr",
            "seg_weight_decay",
            "lambda_kd_seg",
            "seg_biocs_target",
            "seg_biocs_scope",
            "seg_biocs_start_task",
            "seg_evaluation_split",
            "seg_validation_fraction",
            "seg_validation_seed",
        )
    else:
        fields = (
            "dataset",
            "image_size",
            "batch_size",
            "hidden_dim",
            "epochs",
            "learning_rate",
            "weight_decay",
            "lambda_kd",
            "ssr_target",
            "ssr_scope",
            "ssr_start_task",
            "evaluation_split",
            "validation_fraction",
            "validation_seed",
        )
    mismatches = {
        field: (control_args.get(field), treatment_args.get(field))
        for field in fields
        if control_args.get(field) != treatment_args.get(field)
    }
    if mismatches:
        raise RuntimeError(f"Unmatched base training budget: {mismatches}")
    return control, treatment


def run_worker(
    gpu: str,
    jobs: list[Job],
    args: argparse.Namespace,
    git_commit: str,
    lock_hash: str | None,
) -> None:
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = gpu
    for job in jobs:
        record_path = result_record(job, args.dataset)
        if record_path.is_file():
            validate_job_record(job, args, git_commit, lock_hash)
            with PRINT_LOCK:
                print(f"SKIP gpu={gpu} {job.output}", flush=True)
            continue
        job.output.mkdir(parents=True, exist_ok=True)
        log_path = job.output / "launcher.log"
        with PRINT_LOCK:
            print(
                f"RUN gpu={gpu} phase={job.phase} context={job.context} "
                f"spec={job.spec_id} seed={job.seed} treatment={job.treatment}",
                flush=True,
            )
        with log_path.open("a", encoding="utf-8") as log:
            completed = subprocess.run(
                command(job, args, lock_hash),
                cwd=PROJECT_ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        if completed.returncode != 0 or not record_path.is_file():
            write_json_atomic(
                job.output / "FAILED.json",
                {
                    "returncode": completed.returncode,
                    "job": {**asdict(job), "output": str(job.output)},
                    "log": str(log_path),
                },
            )
            raise RuntimeError(f"Job failed: {job.output}; see {log_path}")
        validate_job_record(job, args, git_commit, lock_hash)


def run_jobs(
    jobs: list[Job],
    args: argparse.Namespace,
    git_commit: str,
    lock_hash: str | None,
) -> None:
    slots = [gpu for gpu in args.gpus for _ in range(args.jobs_per_gpu)]
    assignments = [jobs[index::len(slots)] for index in range(len(slots))]
    with ThreadPoolExecutor(max_workers=len(slots)) as pool:
        futures = [
            pool.submit(run_worker, gpu, assigned, args, git_commit, lock_hash)
            for gpu, assigned in zip(slots, assignments)
            if assigned
        ]
        for future in futures:
            future.result()


def matching_control(
    result_root: Path, phase: str, context: str, target: str, seed: int
) -> Job:
    scope = "new_old" if target == "class" else "all"
    return Job(
        phase=phase,
        context=context,
        seed=seed,
        target=target,
        scope=scope,
        treatment=False,
        spec=None,
        output=result_root / phase / context / f"control_{target}" / f"seed_{seed}",
    )


def score_candidates(
    result_root: Path,
    phase: str,
    contexts: dict[str, Iterable[Candidate]],
    seeds: Iterable[int],
    args: argparse.Namespace,
    git_commit: str,
    lock_hash: str | None = None,
) -> list[dict]:
    rows = []
    for context in CONTEXTS:
        for spec in contexts[context]:
            pairs = []
            triple_wins = 0
            for seed in seeds:
                control_job = matching_control(result_root, phase, context, spec.target, seed)
                treatment_job = Job(
                    phase=phase,
                    context=context,
                    seed=seed,
                    target=spec.target,
                    scope=spec.scope,
                    treatment=True,
                    spec=spec,
                    output=result_root / phase / context / spec.spec_id / f"seed_{seed}",
                )
                control, treatment = validate_pair(
                    control_job, treatment_job, args, git_commit, lock_hash
                )
                deltas = {}
                for metric, higher_is_better in PRIMARY_METRICS:
                    raw = float(treatment["metrics"][metric]) - float(
                        control["metrics"][metric]
                    )
                    deltas[metric] = raw if higher_is_better else -raw
                triple_wins += int(all(value > 0 for value in deltas.values()))
                pairs.append((control, treatment, deltas))
            means = {
                metric: sum(pair[2][metric] for pair in pairs) / len(pairs)
                for metric, _ in PRIMARY_METRICS
            }
            rank_gain = sum(
                float(treatment["metrics"]["effective_rank"])
                - float(control["metrics"]["effective_rank"])
                for control, treatment, _ in pairs
            ) / len(pairs)
            overlap_reduction = sum(
                float(control["metrics"]["mean_abs_offdiag_cosine"])
                - float(treatment["metrics"]["mean_abs_offdiag_cosine"])
                for control, treatment, _ in pairs
            ) / len(pairs)
            endpoint_values = list(means.values())
            rows.append(
                {
                    "stage": phase,
                    "context": context,
                    "spec_id": spec.spec_id,
                    "kernel": spec.kernel,
                    "hwhm_exc": spec.hwhm_exc,
                    "hwhm_inh": spec.hwhm_inh,
                    "lambda_ssr": spec.lambda_ssr,
                    "target": spec.target,
                    "scope": spec.scope,
                    "delta_mean_iou": means["mean_iou"],
                    "delta_mean_dice": means["mean_dice"],
                    "forgetting_reduction": means["avg_forgetting_iou"],
                    "rank_gain": rank_gain,
                    "overlap_reduction": overlap_reduction,
                    "triple_wins": triple_wins,
                    "endpoint_gate": all(value > 0 for value in endpoint_values),
                    "geometry_gate": rank_gain > 0 and overlap_reduction > 0,
                    "selection_score": min(endpoint_values),
                    "sum_score": sum(endpoint_values),
                }
            )
    return rows


def find_candidate(spec_id: str) -> Candidate:
    try:
        return next(candidate for candidate in CANDIDATES if candidate.spec_id == spec_id)
    except StopIteration as error:
        raise RuntimeError(f"Unknown selected candidate {spec_id!r}") from error


def top_candidates(rows: list[dict], stage: str, k: int) -> dict[str, tuple[Candidate, ...]]:
    selected = {}
    for context in CONTEXTS:
        context_rows = [row for row in rows if row["context"] == context]
        selected[context] = tuple(
            find_candidate(row["spec_id"])
            for row in select_top(context_rows, k=k, stage=stage)
        )
    return selected


def write_selection_lock(
    args: argparse.Namespace,
    git_commit: str,
    screen_rows: list[dict],
    refine_rows: list[dict],
) -> tuple[dict, dict[str, Candidate]]:
    confirmation_root = args.result_root / "confirmation"
    if confirmation_root.exists() and any(confirmation_root.rglob("result_record.json")):
        raise RuntimeError("Refusing to select after confirmation results exist")
    selected = {}
    selected_specs = {}
    for context in CONTEXTS:
        context_rows = [row for row in refine_rows if row["context"] == context]
        lock_row = lock_selection(context_rows)
        selected[context] = lock_row
        selected_specs[context] = find_candidate(lock_row["selected_spec_id"])
    record_inventory = []
    for phase in ("screen", "refine"):
        for record in sorted((args.result_root / phase).rglob("result_record.json")):
            record_inventory.append(
                {
                    "path": str(record.relative_to(args.result_root)),
                    "sha256": sha256_file(record),
                }
            )
    payload = {
        "protocol": "segmentation_nested_screen_refine_confirm_v1",
        "dataset": args.dataset,
        "git_commit": git_commit,
        "feature_cache_sha256": args.feature_cache_sha256,
        "segmentation_source_sha256": args.segmentation_source_sha256,
        "candidate_count": len(CANDIDATES),
        "screen_seeds": list(SCREEN_SEEDS),
        "refine_seeds": list(REFINE_SEEDS),
        "confirmation_seeds_hidden_during_selection": list(CONFIRMATION_SEEDS),
        "selection_rule": (
            "Within each host objective, prioritize all-positive endpoint means, "
            "then intended geometry, then maximize the weakest favorable endpoint."
        ),
        "selected": {
            context: {
                **selected[context],
                "spec": selected_specs[context].as_dict(),
            }
            for context in CONTEXTS
        },
        "screen_summary_sha256": sha256_value(screen_rows),
        "refine_summary_sha256": sha256_value(refine_rows),
        "development_record_inventory": record_inventory,
    }
    payload["lock_sha256"] = sha256_value(payload)
    write_json_atomic(args.result_root / "SELECTION_LOCK.json", payload)
    return payload, selected_specs


def load_selection_lock(
    args: argparse.Namespace,
    git_commit: str,
    screen_rows: list[dict],
    refine_rows: list[dict],
) -> tuple[dict, dict[str, Candidate]]:
    path = args.result_root / "SELECTION_LOCK.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Invalid selection lock {path}: {error}") from error
    observed_hash = payload.get("lock_sha256")
    unhashed = {key: value for key, value in payload.items() if key != "lock_sha256"}
    if observed_hash != sha256_value(unhashed):
        raise RuntimeError("Selection-lock hash mismatch")
    expected = {
        "protocol": "segmentation_nested_screen_refine_confirm_v1",
        "dataset": args.dataset,
        "git_commit": git_commit,
        "candidate_count": len(CANDIDATES),
        "feature_cache_sha256": args.feature_cache_sha256,
        "segmentation_source_sha256": args.segmentation_source_sha256,
        "screen_summary_sha256": sha256_value(screen_rows),
        "refine_summary_sha256": sha256_value(refine_rows),
    }
    mismatches = {
        key: {"expected": value, "observed": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Selection lock does not match this run: {mismatches}")
    selected_specs = {}
    for context in CONTEXTS:
        try:
            selected_payload = payload["selected"][context]["spec"]
            spec = find_candidate(selected_payload["spec_id"])
        except (KeyError, TypeError) as error:
            raise RuntimeError(f"Selection lock lacks {context!r} spec") from error
        if spec.as_dict() != selected_payload:
            raise RuntimeError(f"Selection lock contains altered {context!r} spec")
        selected_specs[context] = spec
    return payload, selected_specs


def confirmation_jobs(
    result_root: Path, selected: dict[str, Candidate]
) -> list[Job]:
    jobs = []
    for context in CONTEXTS:
        spec = selected[context]
        for seed in CONFIRMATION_SEEDS:
            jobs.append(matching_control(result_root, "confirmation", context, spec.target, seed))
            jobs.append(
                Job(
                    phase="confirmation",
                    context=context,
                    seed=seed,
                    target=spec.target,
                    scope=spec.scope,
                    treatment=True,
                    spec=spec,
                    output=result_root
                    / "confirmation"
                    / context
                    / spec.spec_id
                    / f"seed_{seed}",
                )
            )
    return jobs


def write_confirmation_summary(
    args: argparse.Namespace,
    selected: dict[str, Candidate],
    git_commit: str,
    lock_hash: str,
) -> dict:
    payload = {
        "protocol": "segmentation_nested_screen_refine_confirm_v1",
        "dataset": args.dataset,
        "selection_lock_sha256": lock_hash,
        "confirmation_seeds": list(CONFIRMATION_SEEDS),
        "contexts": {},
    }
    csv_rows = []
    for context in CONTEXTS:
        spec = selected[context]
        controls = []
        treatments = []
        per_seed = []
        for seed in CONFIRMATION_SEEDS:
            control_job = matching_control(
                args.result_root, "confirmation", context, spec.target, seed
            )
            treatment_job = Job(
                phase="confirmation",
                context=context,
                seed=seed,
                target=spec.target,
                scope=spec.scope,
                treatment=True,
                spec=spec,
                output=args.result_root
                / "confirmation"
                / context
                / spec.spec_id
                / f"seed_{seed}",
            )
            control, treatment = validate_pair(
                control_job, treatment_job, args, git_commit, lock_hash
            )
            controls.append(control)
            treatments.append(treatment)
            row = {"seed": seed}
            for metric, higher_is_better in SUMMARY_METRICS:
                control_value = float(control["metrics"][metric])
                treatment_value = float(treatment["metrics"][metric])
                delta = treatment_value - control_value
                row[f"control_{metric}"] = control_value
                row[f"treatment_{metric}"] = treatment_value
                row[f"favorable_delta_{metric}"] = (
                    delta if higher_is_better else -delta
                )
            per_seed.append(row)
        summaries = {}
        for metric, higher_is_better in SUMMARY_METRICS:
            summary = paired_summary(
                [record["metrics"][metric] for record in controls],
                [record["metrics"][metric] for record in treatments],
                higher_is_better=higher_is_better,
            ).as_dict()
            summaries[metric] = summary
            csv_rows.append({"context": context, "metric": metric, **summary})
        payload["contexts"][context] = {
            "control": method_name(args.dataset, context, False),
            "treatment": method_name(args.dataset, context, True),
            "selected_spec": spec.as_dict(),
            "metrics": summaries,
            "per_seed": per_seed,
            "confirmation_gate": all(
                summaries[metric]["ci95_low"] > 0 for metric, _ in PRIMARY_METRICS
            ),
            "geometry_gate": all(
                summaries[metric]["ci95_low"] > 0 for metric, _ in GEOMETRY_METRICS
            ),
        }
    write_json_atomic(args.result_root / "CONFIRMATION_SUMMARY.json", payload)
    write_csv_atomic(args.result_root / "CONFIRMATION_SUMMARY.csv", csv_rows)
    return payload


def manifest(args: argparse.Namespace, git_commit: str) -> dict:
    screen_jobs_count = len(control_jobs(args.result_root, "screen", SCREEN_SEEDS)) + len(
        treatment_jobs(
            args.result_root,
            "screen",
            SCREEN_SEEDS,
            {context: CANDIDATES for context in CONTEXTS},
        )
    )
    refine_count = len(REFINE_SEEDS) * len(CONTEXTS) * (2 + 4)
    confirmation_count = len(CONFIRMATION_SEEDS) * len(CONTEXTS) * 2
    return {
        "protocol": "segmentation_nested_screen_refine_confirm_v1",
        "dataset": args.dataset,
        "git_commit": git_commit,
        "candidate_matrix": [candidate.as_dict() for candidate in CANDIDATES],
        "screen_seeds": list(SCREEN_SEEDS),
        "refine_seeds": list(REFINE_SEEDS),
        "confirmation_seeds": list(CONFIRMATION_SEEDS),
        "selection_contexts": ["task_to_task_ssr", "kd_to_kd_ssr"],
        "evaluation_policy": {
            "screen": "fixed_stratified_train_validation",
            "refine": "fixed_stratified_train_validation",
            "confirmation": "official_test_after_lock",
            "validation_fraction": 0.2,
            "validation_seed": 20260809,
        },
        "training_epochs_by_phase": (
            {"screen": 8, "refine": 16, "confirmation": 24}
            if args.dataset == "cub200"
            else {"screen": 8, "refine": 15, "confirmation": 20}
        ),
        "jobs": {
            "screen": screen_jobs_count,
            "refine_maximum": refine_count,
            "confirmation": confirmation_count,
            "total_maximum": screen_jobs_count + refine_count + confirmation_count,
        },
        "compute": {
            "gpus": list(args.gpus),
            "jobs_per_gpu": args.jobs_per_gpu,
            "workers": args.workers,
            "batch_size": args.batch_size,
        },
        "data": {
            "dataset_root": str(args.dataset_root),
            "feature_cache": str(args.feature_cache),
            "feature_cache_sha256": args.feature_cache_sha256,
            "segmentation_source_sha256": args.segmentation_source_sha256,
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=("cub200", "oxford_iiit_pet", "oxford_flowers102"),
        required=True,
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--gpus", nargs="+", default=[str(index) for index in range(8)])
    parser.add_argument("--jobs-per-gpu", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args(argv)
    args.dataset_root = args.dataset_root.resolve()
    args.feature_cache = args.feature_cache.resolve()
    args.result_root = args.result_root.resolve()
    if args.batch_size is None:
        args.batch_size = 24 if args.dataset == "cub200" else 32
    if args.jobs_per_gpu < 1 or args.workers < 0 or args.batch_size < 1:
        parser.error("jobs-per-gpu and batch-size must be positive; workers cannot be negative")
    if len(args.gpus) != 8 or len(set(args.gpus)) != 8:
        parser.error("this runner requires eight distinct local GPU identifiers")
    if not args.feature_cache.is_file():
        parser.error(f"feature cache does not exist: {args.feature_cache}")
    args.feature_cache_sha256 = sha256_file(args.feature_cache)
    args.segmentation_source_sha256 = None
    if args.dataset == "cub200":
        source = args.dataset_root / "segmentations.tgz"
        if not source.is_file():
            parser.error(f"CUB segmentation source does not exist: {source}")
        args.segmentation_source_sha256 = sha256_file(source)
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    git_commit = current_git_commit()
    if args.result_root.exists() and not args.resume:
        raise SystemExit(f"Fresh --result-root required: {args.result_root}")
    args.result_root.mkdir(parents=True, exist_ok=True)
    manifest_payload = manifest(args, git_commit)
    write_json_atomic(args.result_root / "MANIFEST.json", manifest_payload)
    if args.manifest_only:
        print(json.dumps(manifest_payload, indent=2))
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

    screen_contexts = {context: CANDIDATES for context in CONTEXTS}
    screen = control_jobs(args.result_root, "screen", SCREEN_SEEDS) + treatment_jobs(
        args.result_root, "screen", SCREEN_SEEDS, screen_contexts
    )
    run_jobs(screen, args, git_commit, None)
    screen_rows = score_candidates(
        args.result_root,
        "screen",
        screen_contexts,
        SCREEN_SEEDS,
        args,
        git_commit,
    )
    write_json_atomic(args.result_root / "SCREEN_SUMMARY.json", screen_rows)
    write_csv_atomic(args.result_root / "SCREEN_SUMMARY.csv", screen_rows)

    refine_contexts = top_candidates(screen_rows, "screen", 4)
    refine = control_jobs(args.result_root, "refine", REFINE_SEEDS) + treatment_jobs(
        args.result_root, "refine", REFINE_SEEDS, refine_contexts
    )
    run_jobs(refine, args, git_commit, None)
    refine_rows = score_candidates(
        args.result_root,
        "refine",
        refine_contexts,
        REFINE_SEEDS,
        args,
        git_commit,
    )
    write_json_atomic(args.result_root / "REFINE_SUMMARY.json", refine_rows)
    write_csv_atomic(args.result_root / "REFINE_SUMMARY.csv", refine_rows)

    if (args.result_root / "SELECTION_LOCK.json").is_file():
        lock, selected = load_selection_lock(
            args, git_commit, screen_rows, refine_rows
        )
    else:
        lock, selected = write_selection_lock(
            args, git_commit, screen_rows, refine_rows
        )
    lock_hash = lock["lock_sha256"]
    confirmation = confirmation_jobs(args.result_root, selected)
    run_jobs(confirmation, args, git_commit, lock_hash)
    summary = write_confirmation_summary(args, selected, git_commit, lock_hash)
    write_json_atomic(
        args.result_root / "COMPLETED.json",
        {
            "status": "complete",
            "dataset": args.dataset,
            "git_commit": git_commit,
            "selection_lock_sha256": lock_hash,
            "confirmation_gates": {
                context: summary["contexts"][context]["confirmation_gate"]
                for context in CONTEXTS
            },
            "geometry_gates": {
                context: summary["contexts"][context]["geometry_gate"]
                for context in CONTEXTS
            },
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
    )
    print(f"Completed locked segmentation study: {args.result_root}")


if __name__ == "__main__":
    main()
