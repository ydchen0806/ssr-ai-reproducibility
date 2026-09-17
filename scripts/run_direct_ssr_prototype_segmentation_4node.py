#!/usr/bin/env python3
"""Locked multi-node Task versus Task+SSR segmentation study.

The study is intentionally narrow: every pair has the same task loss, data,
initialization and optimization budget; the treatment arm adds only SSR.  CUB
and Oxford-IIIT Pet are used for development, one universal SSR setting is
locked, and untouched test splits from CUB, Pet and Flowers102 are evaluated
only after the lock has been written.

Every node runs this same driver against a shared result directory.  Each
phase/dataset/seed group is deterministically assigned to one node and GPU: its
task-only control is run once, then reused by all SSR candidates in that group.
File-system barriers separate screen, refine and confirmation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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
from typing import Iterable, Sequence

import numpy as np


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
    validate_paired_identities,
)


PROTOCOL = "direct_ssr_prototype_segmentation_v1"
CUB_RUNNER = PROJECT_ROOT / "experiments" / "cub200_continual_benchmark.py"
MULTIDATASET_RUNNER = PROJECT_ROOT / "experiments" / "multidataset_segmentation.py"
DEVELOPMENT_DATASETS = ("cub200", "oxford_iiit_pet")
CONFIRMATION_DATASETS = (*DEVELOPMENT_DATASETS, "oxford_flowers102")
SCREEN_SEEDS = (21001, 21003, 21005)
REFINE_SEEDS = (22001, 22003, 22005, 22007, 22009, 22011, 22013)
CONFIRMATION_SEEDS = {
    "cub200": tuple(range(23001, 23060, 2)),
    "oxford_iiit_pet": tuple(range(24001, 24060, 2)),
    "oxford_flowers102": tuple(range(25001, 25040, 2)),
}
WIDTHS = ((0.188, 0.530), (0.260, 0.720))
LAMBDAS = (0.005, 0.015, 0.05, 0.15)
CANDIDATES = candidate_matrix(
    ("gaussian", "cauchy"),
    WIDTHS,
    LAMBDAS,
    (("class", "new_old"),),
)
if len(CANDIDATES) != 16:
    raise RuntimeError(f"Expected 16 prespecified candidates, got {len(CANDIDATES)}")

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
PRIMARY_EFFECT_FLOOR = 0.20
SIGN_FLIP_DRAWS = 100_000
PRINT_LOCK = Lock()


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    root: Path
    feature_cache: Path
    feature_cache_sha256: str
    source_sha256: str | None = None


@dataclass(frozen=True)
class PairJob:
    phase: str
    dataset: str
    seed: int
    spec: Candidate
    output: Path
    control_output: Path

    @property
    def pair_id(self) -> str:
        return f"{self.phase}__{self.dataset}__{self.spec.spec_id}__seed_{self.seed}"


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_csv_atomic(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty table: {path}")
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise ValueError("CSV rows do not share one ordered schema")
    path.parent.mkdir(parents=True, exist_ok=True)
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


def find_candidate(spec_id: str) -> Candidate:
    try:
        return next(candidate for candidate in CANDIDATES if candidate.spec_id == spec_id)
    except StopIteration as error:
        raise RuntimeError(f"Unknown candidate {spec_id!r}") from error


def stage_selection_hash(phase: str, lock_hash: str | None) -> str:
    if phase == "confirmation":
        if not lock_hash:
            raise ValueError("confirmation requires a selection-lock hash")
        return lock_hash
    return f"{PROTOCOL}_{phase}"


def method_name(dataset: str, treatment: bool) -> str:
    if dataset == "cub200":
        return "biocs" if treatment else "baseline"
    return "task_ssr" if treatment else "task_only"


def expected_identity(dataset: str) -> tuple[str, str]:
    if dataset == "cub200":
        return "cub200_masks", "resnet18_dense_prototype_cosine_decoder"
    return f"{dataset}_masks", "resnet18_dense_prototype_cosine_decoder"


def phase_epochs(dataset: str, phase: str) -> int:
    if dataset == "cub200":
        return {"screen": 8, "refine": 16, "confirmation": 24}[phase]
    return {"screen": 8, "refine": 15, "confirmation": 20}[phase]


def arm_output(job: PairJob, treatment: bool) -> Path:
    return job.output / "task_ssr" if treatment else job.control_output


def result_record_path(job: PairJob, treatment: bool) -> Path:
    output = arm_output(job, treatment)
    if job.dataset == "cub200":
        return (
            output
            / "result_records"
            / "segmentation"
            / method_name(job.dataset, treatment)
            / f"seed_{job.seed}"
            / "result_record.json"
        )
    return output / "result_record.json"


def command(
    job: PairJob,
    treatment: bool,
    args: argparse.Namespace,
    datasets: dict[str, DatasetSpec],
    lock_hash: str | None,
) -> list[str]:
    dataset = datasets[job.dataset]
    method = method_name(job.dataset, treatment)
    evaluation_split = "test" if job.phase == "confirmation" else "validation"
    selection_hash = stage_selection_hash(job.phase, lock_hash)
    spec = job.spec
    if job.dataset == "cub200":
        if not dataset.source_sha256:
            raise ValueError("CUB jobs require the segmentation-source SHA256")
        return [
            args.python,
            str(CUB_RUNNER),
            "--data_root",
            str(dataset.root),
            "--segmentation_cache",
            str(dataset.feature_cache),
            "--segmentation_source_sha256",
            dataset.source_sha256,
            "--segmentation_cache_sha256",
            dataset.feature_cache_sha256,
            "--selection_lock_sha256",
            selection_hash,
            "--output_dir",
            str(arm_output(job, treatment)),
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
            str(phase_epochs(job.dataset, job.phase)),
            "--seg_batch_size",
            str(args.cub_batch_size),
            "--seg_image_size",
            "192",
            "--seg_hidden_dim",
            "256",
            "--segmentation-head",
            "prototype_cosine",
            "--prototype-logit-scale",
            "10.0",
            "--seg_biocs_target",
            "class",
            "--seg_biocs_scope",
            "new_old",
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
            "0.0",
            "--ssr-warmup-fraction",
            "0.2",
            "--ssr-ramp-fraction",
            "0.2",
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
    return [
        args.python,
        str(MULTIDATASET_RUNNER),
        "--dataset",
        job.dataset,
        "--data-root",
        str(dataset.root.parent),
        "--dataset-root",
        str(dataset.root),
        "--feature-cache",
        str(dataset.feature_cache),
        "--feature-cache-sha256",
        dataset.feature_cache_sha256,
        "--output-dir",
        str(arm_output(job, treatment)),
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
        str(args.multidataset_batch_size),
        "--hidden-dim",
        "192",
        "--segmentation-head",
        "prototype_cosine",
        "--prototype-logit-scale",
        "10.0",
        "--epochs",
        str(phase_epochs(job.dataset, job.phase)),
        "--learning-rate",
        "0.001",
        "--weight-decay",
        "0.0",
        "--lambda-kd",
        "0.0",
        "--lambda-ssr",
        str(spec.lambda_ssr),
        "--ssr-warmup-fraction",
        "0.2",
        "--ssr-ramp-fraction",
        "0.2",
        "--ssr-target",
        "class",
        "--ssr-scope",
        "new_old",
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


def load_args(path: Path) -> dict:
    try:
        return json.loads((path / "args.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Invalid arguments in {path}: {error}") from error


def validate_record(
    job: PairJob,
    treatment: bool,
    args: argparse.Namespace,
    git_commit: str,
    lock_hash: str | None,
) -> dict:
    path = result_record_path(job, treatment)
    try:
        record = validate_result_record(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, ResultSchemaError) as error:
        raise RuntimeError(f"Invalid result record {path}: {error}") from error

    dataset_name, model_name = expected_identity(job.dataset)
    expected = {
        "git_commit": git_commit,
        "task_family": "segmentation",
        "dataset": dataset_name,
        "model": model_name,
        "seed": job.seed,
        "recipe": "ssr_only" if treatment else "plain",
        "distance_mapping": "cosine" if treatment else "none",
        "selection_lock_hash": stage_selection_hash(job.phase, lock_hash),
    }
    mismatches = {
        key: {"expected": value, "observed": record.get(key)}
        for key, value in expected.items()
        if record.get(key) != value
    }
    objective = record.get("objective", {})
    expected_objective = {
        "task": True,
        "kd": False,
        "ssr": treatment,
        "anchor": False,
        "spectral": False,
        "ewc": False,
        "mas": False,
        "si": False,
        "center": False,
        "protodecor": False,
    }
    for name, expected_value in expected_objective.items():
        if objective.get(name, False) != expected_value:
            mismatches[f"objective.{name}"] = {
                "expected": expected_value,
                "observed": objective.get(name),
            }
    model_config = record.get("model_config", {})
    if model_config.get("head") != "prototype_cosine":
        mismatches["model_config.head"] = {
            "expected": "prototype_cosine",
            "observed": model_config.get("head"),
        }
    run_args = load_args(arm_output(job, treatment))
    method_field = "methods" if job.dataset == "cub200" else "method"
    expected_method: object = (
        [method_name(job.dataset, treatment)]
        if job.dataset == "cub200"
        else method_name(job.dataset, treatment)
    )
    if run_args.get(method_field) != expected_method:
        mismatches[f"config.{method_field}"] = {
            "expected": expected_method,
            "observed": run_args.get(method_field),
        }
    head_field = "segmentation_head"
    if run_args.get(head_field) != "prototype_cosine":
        mismatches[f"config.{head_field}"] = {
            "expected": "prototype_cosine",
            "observed": run_args.get(head_field),
        }
    kd_field = "lambda_kd_seg" if job.dataset == "cub200" else "lambda_kd"
    if not math.isclose(float(run_args.get(kd_field, math.nan)), 0.0, abs_tol=1e-12):
        mismatches[f"config.{kd_field}"] = {
            "expected": 0.0,
            "observed": run_args.get(kd_field),
        }
    for field, expected_value in (
        ("ssr_warmup_fraction", 0.2),
        ("ssr_ramp_fraction", 0.2),
    ):
        if not math.isclose(
            float(run_args.get(field, math.nan)), expected_value, abs_tol=1e-12
        ):
            mismatches[f"config.{field}"] = {
                "expected": expected_value,
                "observed": run_args.get(field),
            }
    if treatment:
        expected_kernel = {
            "family": job.spec.kernel,
            "A_exc": job.spec.a_exc,
            "A_inh": job.spec.a_inh,
            "sigma_exc": job.spec.sigma_exc,
            "sigma_inh": job.spec.sigma_inh,
        }
        kernel = record.get("kernel", {})
        for key, expected_value in expected_kernel.items():
            observed = kernel.get(key)
            matches = observed == expected_value if isinstance(expected_value, str) else (
                isinstance(observed, (int, float))
                and math.isclose(
                    float(observed), float(expected_value), rel_tol=1e-12, abs_tol=1e-12
                )
            )
            if not matches:
                mismatches[f"kernel.{key}"] = {
                    "expected": expected_value,
                    "observed": observed,
                }
        if not math.isclose(
            float(record.get("lambda_ssr", math.nan)),
            job.spec.lambda_ssr,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            mismatches["lambda_ssr"] = {
                "expected": job.spec.lambda_ssr,
                "observed": record.get("lambda_ssr"),
            }
    required_metrics = {name for name, _ in SUMMARY_METRICS}
    if not required_metrics <= set(record.get("metrics", {})):
        mismatches["metrics"] = {
            "expected": sorted(required_metrics),
            "observed": sorted(record.get("metrics", {})),
        }
    if mismatches:
        raise RuntimeError(f"Result identity mismatch for {path}: {mismatches}")
    return record


def paired_records(
    job: PairJob,
    args: argparse.Namespace,
    git_commit: str,
    lock_hash: str | None,
) -> tuple[dict, dict]:
    control = validate_record(job, False, args, git_commit, lock_hash)
    treatment = validate_record(job, True, args, git_commit, lock_hash)
    validate_paired_identities([control], [treatment])
    control_objective = dict(control["objective"])
    treatment_objective = dict(treatment["objective"])
    treatment_objective["ssr"] = False
    if treatment_objective != control_objective:
        raise RuntimeError(
            f"The pair differs by more than SSR: {job.pair_id}: "
            f"{control_objective} vs {treatment_objective}"
        )

    control_args = load_args(arm_output(job, False))
    treatment_args = load_args(arm_output(job, True))
    if job.dataset == "cub200":
        base_fields = (
            "data_root",
            "segmentation_cache",
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
            "segmentation_head",
            "prototype_logit_scale",
            "ssr_warmup_fraction",
            "ssr_ramp_fraction",
        )
    else:
        base_fields = (
            "dataset",
            "dataset_root",
            "feature_cache",
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
            "segmentation_head",
            "prototype_logit_scale",
            "ssr_warmup_fraction",
            "ssr_ramp_fraction",
        )
    differing = {
        key: (control_args.get(key), treatment_args.get(key))
        for key in base_fields
        if control_args.get(key) != treatment_args.get(key)
    }
    if differing:
        raise RuntimeError(f"Unmatched pair configuration {job.pair_id}: {differing}")
    return control, treatment


def run_arm(
    gpu: str,
    job: PairJob,
    treatment: bool,
    args: argparse.Namespace,
    datasets: dict[str, DatasetSpec],
    git_commit: str,
    lock_hash: str | None,
) -> None:
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = gpu
    job.output.mkdir(parents=True, exist_ok=True)
    record_path = result_record_path(job, treatment)
    if record_path.is_file():
        validate_record(job, treatment, args, git_commit, lock_hash)
        with PRINT_LOCK:
            print(
                f"SKIP gpu={gpu} {job.pair_id} treatment={treatment}", flush=True
            )
        return
    output = arm_output(job, treatment)
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "launcher.log"
    with PRINT_LOCK:
        print(
            f"RUN gpu={gpu} pair={job.pair_id} treatment={treatment}",
            flush=True,
        )
    with log_path.open("a", encoding="utf-8") as log:
        completed = subprocess.run(
            command(job, treatment, args, datasets, lock_hash),
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    if completed.returncode != 0 or not record_path.is_file():
        write_json_atomic(
            output / "FAILED.json",
            {
                "returncode": completed.returncode,
                "pair": {**asdict(job), "output": str(job.output)},
                "treatment": treatment,
                "log": str(log_path),
            },
        )
        raise RuntimeError(f"Arm failed: {job.pair_id}; see {log_path}")
    validate_record(job, treatment, args, git_commit, lock_hash)


def grouped_jobs(jobs: Sequence[PairJob]) -> list[list[PairJob]]:
    groups: dict[tuple[str, str, int], list[PairJob]] = {}
    for job in jobs:
        key = (job.phase, job.dataset, job.seed)
        groups.setdefault(key, []).append(job)
    return [
        sorted(groups[key], key=lambda job: job.spec.spec_id)
        for key in sorted(groups)
    ]


def node_groups(jobs: Sequence[PairJob], node_rank: int, nodes: int) -> list[list[PairJob]]:
    return grouped_jobs(jobs)[node_rank::nodes]


def run_group(
    gpu: str,
    jobs: Sequence[PairJob],
    args: argparse.Namespace,
    datasets: dict[str, DatasetSpec],
    git_commit: str,
    lock_hash: str | None,
) -> None:
    if not jobs:
        return
    representative = jobs[0]
    run_arm(gpu, representative, False, args, datasets, git_commit, lock_hash)

    def treatment(job: PairJob) -> None:
        run_arm(gpu, job, True, args, datasets, git_commit, lock_hash)
        paired_records(job, args, git_commit, lock_hash)

    with ThreadPoolExecutor(max_workers=args.jobs_per_gpu) as pool:
        futures = [pool.submit(treatment, job) for job in jobs]
        for future in futures:
            future.result()


def run_shard(
    jobs: Sequence[PairJob],
    args: argparse.Namespace,
    datasets: dict[str, DatasetSpec],
    git_commit: str,
    lock_hash: str | None,
) -> None:
    shard = node_groups(jobs, args.node_rank, args.nodes)
    assignments = [shard[index::len(args.gpus)] for index in range(len(args.gpus))]

    def worker(gpu: str, assigned: Sequence[Sequence[PairJob]]) -> None:
        for group in assigned:
            run_group(gpu, group, args, datasets, git_commit, lock_hash)

    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        futures = [
            pool.submit(worker, gpu, assigned)
            for gpu, assigned in zip(args.gpus, assignments)
            if assigned
        ]
        for future in futures:
            future.result()


def pair_jobs(
    result_root: Path,
    phase: str,
    datasets: Iterable[str],
    seeds_by_dataset: dict[str, Sequence[int]],
    candidates: Sequence[Candidate],
) -> list[PairJob]:
    jobs = [
        PairJob(
            phase=phase,
            dataset=dataset,
            seed=seed,
            spec=spec,
            output=(
                result_root
                / phase
                / dataset
                / spec.spec_id
                / f"seed_{seed}"
            ),
            control_output=(
                result_root
                / phase
                / dataset
                / "task_only_control"
                / f"seed_{seed}"
            ),
        )
        for dataset in datasets
        for spec in candidates
        for seed in seeds_by_dataset[dataset]
    ]
    return sorted(jobs, key=lambda job: job.pair_id)


def favorable_delta(control: dict, treatment: dict, metric: str, higher: bool) -> float:
    raw = float(treatment["metrics"][metric]) - float(control["metrics"][metric])
    return raw if higher else -raw


def score_candidates(
    jobs: Sequence[PairJob],
    candidates: Sequence[Candidate],
    args: argparse.Namespace,
    git_commit: str,
) -> list[dict]:
    rows: list[dict] = []
    jobs_by_spec = {
        spec.spec_id: [job for job in jobs if job.spec.spec_id == spec.spec_id]
        for spec in candidates
    }
    for spec in candidates:
        dataset_metrics: dict[str, dict[str, float | int | bool]] = {}
        all_endpoint_means: list[float] = []
        favorable_pair_count = 0
        pair_count = 0
        rank_deltas: list[float] = []
        overlap_deltas: list[float] = []
        for dataset in DEVELOPMENT_DATASETS:
            dataset_jobs = [job for job in jobs_by_spec[spec.spec_id] if job.dataset == dataset]
            endpoint_samples = {metric: [] for metric, _ in PRIMARY_METRICS}
            dataset_triple_wins = 0
            for job in dataset_jobs:
                control, treatment = paired_records(job, args, git_commit, None)
                deltas = {
                    metric: favorable_delta(control, treatment, metric, higher)
                    for metric, higher in PRIMARY_METRICS
                }
                dataset_triple_wins += int(all(value > 0.0 for value in deltas.values()))
                favorable_pair_count += int(all(value > 0.0 for value in deltas.values()))
                pair_count += 1
                for metric, value in deltas.items():
                    endpoint_samples[metric].append(value)
                rank_deltas.append(
                    favorable_delta(control, treatment, "effective_rank", True)
                )
                overlap_deltas.append(
                    favorable_delta(
                        control,
                        treatment,
                        "mean_abs_offdiag_cosine",
                        False,
                    )
                )
            endpoint_means = {
                metric: float(np.mean(values))
                for metric, values in endpoint_samples.items()
            }
            all_endpoint_means.extend(endpoint_means.values())
            dataset_metrics[dataset] = {
                **{f"delta_{metric}": value for metric, value in endpoint_means.items()},
                "triple_wins": dataset_triple_wins,
                "endpoint_gate": all(value > 0.0 for value in endpoint_means.values()),
            }
        rows.append(
            {
                "stage": jobs[0].phase,
                "spec_id": spec.spec_id,
                "kernel": spec.kernel,
                "hwhm_exc": spec.hwhm_exc,
                "hwhm_inh": spec.hwhm_inh,
                "lambda_ssr": spec.lambda_ssr,
                "target": spec.target,
                "scope": spec.scope,
                "dataset_endpoint_gate_count": sum(
                    bool(dataset_metrics[dataset]["endpoint_gate"])
                    for dataset in DEVELOPMENT_DATASETS
                ),
                "weakest_dataset_endpoint": min(all_endpoint_means),
                "mean_endpoint_gain": float(np.mean(all_endpoint_means)),
                "favorable_pair_rate": favorable_pair_count / pair_count,
                "rank_gain": float(np.mean(rank_deltas)),
                "overlap_reduction": float(np.mean(overlap_deltas)),
                "geometry_gate": (
                    float(np.mean(rank_deltas)) > 0.0
                    and float(np.mean(overlap_deltas)) > 0.0
                ),
                "per_dataset": dataset_metrics,
            }
        )
    return sorted(rows, key=selection_sort_key)


def selection_sort_key(row: dict) -> tuple:
    """Sort by task endpoints only; geometry is deliberately not a selector."""
    return (
        -int(row["dataset_endpoint_gate_count"]),
        -float(row["weakest_dataset_endpoint"]),
        -float(row["mean_endpoint_gain"]),
        -float(row["favorable_pair_rate"]),
        str(row["spec_id"]),
    )


def flat_development_rows(rows: Sequence[dict]) -> list[dict]:
    flattened = []
    for row in rows:
        flat = {key: value for key, value in row.items() if key != "per_dataset"}
        for dataset in DEVELOPMENT_DATASETS:
            for key, value in row["per_dataset"][dataset].items():
                flat[f"{dataset}_{key}"] = value
        flattened.append(flat)
    return flattened


def sign_flip_test(values: Sequence[float], *, label: str) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 2 or not np.isfinite(array).all():
        raise ValueError("sign-flip input must contain at least two finite values")
    observed = float(array.mean())
    seed = int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    extreme = 0
    completed = 0
    chunk_size = 10_000
    while completed < SIGN_FLIP_DRAWS:
        count = min(chunk_size, SIGN_FLIP_DRAWS - completed)
        signs = rng.choice(np.asarray((-1.0, 1.0)), size=(count, array.size))
        randomized = (signs * array).mean(axis=1)
        extreme += int(np.count_nonzero(randomized >= observed - 1e-15))
        completed += count
    return {
        "alternative": "favorable_difference_greater_than_zero",
        "method": "deterministic_monte_carlo_sign_flip",
        "draws": SIGN_FLIP_DRAWS,
        "seed": seed,
        "p_value": (extreme + 1.0) / (SIGN_FLIP_DRAWS + 1.0),
        "observed_mean": observed,
    }


def development_inventory(result_root: Path) -> list[dict]:
    inventory = []
    for phase in ("screen", "refine"):
        for path in sorted((result_root / phase).rglob("result_record.json")):
            inventory.append(
                {
                    "path": str(path.relative_to(result_root)),
                    "sha256": sha256_file(path),
                }
            )
    return inventory


def write_selection_lock(
    args: argparse.Namespace,
    git_commit: str,
    screen_rows: Sequence[dict],
    refine_rows: Sequence[dict],
) -> tuple[dict, Candidate]:
    confirmation_root = args.result_root / "confirmation"
    if confirmation_root.exists() and any(confirmation_root.rglob("result_record.json")):
        raise RuntimeError("Refusing to select after confirmation records exist")
    selected_row = sorted(refine_rows, key=selection_sort_key)[0]
    selected = find_candidate(selected_row["spec_id"])
    payload = {
        "protocol": PROTOCOL,
        "git_commit": git_commit,
        "development_datasets": list(DEVELOPMENT_DATASETS),
        "candidate_count": len(CANDIDATES),
        "screen_seeds": list(SCREEN_SEEDS),
        "refine_seeds": list(REFINE_SEEDS),
        "confirmation_seeds_hidden_during_selection": {
            dataset: list(seeds) for dataset, seeds in CONFIRMATION_SEEDS.items()
        },
        "selection_rule": (
            "Task endpoints only: maximize the number of development datasets with "
            "positive mean mIoU, Dice and forgetting reduction; then maximize the "
            "weakest dataset-metric mean, aggregate endpoint mean and paired win rate. "
            "Geometry is reported but is not used for selection."
        ),
        "selected_spec": selected.as_dict(),
        "selected_refine_row": selected_row,
        "screen_summary_sha256": sha256_value(list(screen_rows)),
        "refine_summary_sha256": sha256_value(list(refine_rows)),
        "development_record_inventory": development_inventory(args.result_root),
    }
    payload["lock_sha256"] = sha256_value(payload)
    write_json_atomic(args.result_root / "SELECTION_LOCK.json", payload)
    return payload, selected


def load_selection_lock(
    args: argparse.Namespace,
    git_commit: str,
    screen_rows: Sequence[dict],
    refine_rows: Sequence[dict],
) -> tuple[dict, Candidate]:
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
        "protocol": PROTOCOL,
        "git_commit": git_commit,
        "candidate_count": len(CANDIDATES),
        "screen_summary_sha256": sha256_value(list(screen_rows)),
        "refine_summary_sha256": sha256_value(list(refine_rows)),
    }
    mismatches = {
        key: {"expected": value, "observed": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Selection lock does not match the run: {mismatches}")
    selected = find_candidate(payload["selected_spec"]["spec_id"])
    if payload["selected_spec"] != selected.as_dict():
        raise RuntimeError("Selection lock contains an altered candidate")
    return payload, selected


def write_confirmation_summary(
    jobs: Sequence[PairJob],
    selected: Candidate,
    args: argparse.Namespace,
    git_commit: str,
    lock_hash: str,
) -> dict:
    payload: dict = {
        "protocol": PROTOCOL,
        "git_commit": git_commit,
        "selection_lock_sha256": lock_hash,
        "comparison": "Task-only versus Task+SSR",
        "selected_spec": selected.as_dict(),
        "primary_metric": "mean_iou",
        "primary_effect_floor_percentage_points": PRIMARY_EFFECT_FLOOR,
        "datasets": {},
    }
    csv_rows = []
    for dataset in CONFIRMATION_DATASETS:
        dataset_jobs = [job for job in jobs if job.dataset == dataset]
        controls = []
        treatments = []
        per_seed = []
        for job in dataset_jobs:
            control, treatment = paired_records(job, args, git_commit, lock_hash)
            controls.append(control)
            treatments.append(treatment)
            row = {"seed": job.seed}
            for metric, higher in SUMMARY_METRICS:
                control_value = float(control["metrics"][metric])
                treatment_value = float(treatment["metrics"][metric])
                row[f"control_{metric}"] = control_value
                row[f"treatment_{metric}"] = treatment_value
                row[f"favorable_delta_{metric}"] = (
                    treatment_value - control_value
                    if higher
                    else control_value - treatment_value
                )
            per_seed.append(row)
        summaries = {}
        for metric, higher in SUMMARY_METRICS:
            summary = paired_summary(
                [record["metrics"][metric] for record in controls],
                [record["metrics"][metric] for record in treatments],
                higher_is_better=higher,
            ).as_dict()
            favorable_values = [row[f"favorable_delta_{metric}"] for row in per_seed]
            permutation = sign_flip_test(
                favorable_values,
                label=f"{PROTOCOL}:{lock_hash}:{dataset}:{metric}",
            )
            summaries[metric] = {**summary, "sign_flip": permutation}
            csv_rows.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    **summary,
                    "sign_flip_p": permutation["p_value"],
                    "sign_flip_draws": permutation["draws"],
                }
            )
        primary = summaries["mean_iou"]
        primary_gate = (
            primary["difference_mean"] >= PRIMARY_EFFECT_FLOOR
            and primary["ci95_low"] > 0.0
            and primary["sign_flip"]["p_value"] < 0.05
        )
        triple_gate = all(
            summaries[metric]["ci95_low"] > 0.0
            and summaries[metric]["sign_flip"]["p_value"] < 0.05
            for metric, _ in PRIMARY_METRICS
        )
        geometry_gate = all(
            summaries[metric]["ci95_low"] > 0.0
            for metric, _ in GEOMETRY_METRICS
        )
        payload["datasets"][dataset] = {
            "n": len(dataset_jobs),
            "metrics": summaries,
            "per_seed": per_seed,
            "primary_gate": primary_gate,
            "triple_endpoint_gate": triple_gate,
            "geometry_gate": geometry_gate,
        }
    payload["all_primary_gates"] = all(
        payload["datasets"][dataset]["primary_gate"]
        for dataset in CONFIRMATION_DATASETS
    )
    write_json_atomic(args.result_root / "CONFIRMATION_SUMMARY.json", payload)
    write_csv_atomic(args.result_root / "CONFIRMATION_SUMMARY.csv", csv_rows)
    return payload


def dataset_paths(args: argparse.Namespace) -> dict[str, tuple[Path, Path]]:
    return {
        "cub200": (args.cub_root, args.cub_cache),
        "oxford_iiit_pet": (args.pet_root, args.pet_cache),
        "oxford_flowers102": (args.flowers_root, args.flowers_cache),
    }


def prepare_datasets(args: argparse.Namespace) -> dict[str, DatasetSpec]:
    datasets = {}
    for name, (root, cache) in dataset_paths(args).items():
        if not root.is_dir():
            raise FileNotFoundError(f"Dataset root does not exist: {root}")
        if not cache.is_file():
            raise FileNotFoundError(f"Feature cache does not exist: {cache}")
        source_sha256 = None
        if name == "cub200":
            source = root / "segmentations.tgz"
            if not source.is_file():
                raise FileNotFoundError(f"CUB segmentation source does not exist: {source}")
            source_sha256 = sha256_file(source)
        datasets[name] = DatasetSpec(
            name=name,
            root=root,
            feature_cache=cache,
            feature_cache_sha256=sha256_file(cache),
            source_sha256=source_sha256,
        )
    return datasets


def manifest(
    args: argparse.Namespace,
    git_commit: str,
    datasets: dict[str, DatasetSpec] | None,
) -> dict:
    screen_pairs = len(DEVELOPMENT_DATASETS) * len(CANDIDATES) * len(SCREEN_SEEDS)
    screen_controls = len(DEVELOPMENT_DATASETS) * len(SCREEN_SEEDS)
    refine_pairs = len(DEVELOPMENT_DATASETS) * 4 * len(REFINE_SEEDS)
    refine_controls = len(DEVELOPMENT_DATASETS) * len(REFINE_SEEDS)
    confirmation_pairs = sum(len(CONFIRMATION_SEEDS[name]) for name in CONFIRMATION_DATASETS)
    confirmation_controls = confirmation_pairs
    data_payload = {}
    for name, (root, cache) in dataset_paths(args).items():
        spec = datasets.get(name) if datasets else None
        data_payload[name] = {
            "dataset_root": str(root),
            "feature_cache": str(cache),
            "feature_cache_sha256": spec.feature_cache_sha256 if spec else None,
            "segmentation_source_sha256": spec.source_sha256 if spec else None,
        }
    return {
        "protocol": PROTOCOL,
        "git_commit": git_commit,
        "comparison": "Task-only versus Task+SSR; no KD, anchor or spectral term",
        "model": "ResNet18 dense prototype-cosine mask decoder",
        "model_constraints": {
            "foreground_readout": "normalized class prototype cosine",
            "background_readout": "normalized shared background prototype cosine",
            "logit": "scale * (foreground cosine - background cosine) + class bias",
            "prototype_logit_scale_initial": 10.0,
            "prototype_logit_scale_clamp": [1.0, 30.0],
            "ssr_target": "class",
            "ssr_scope": "new_old",
            "ssr_start_task": 1,
            "ssr_warmup_fraction": 0.2,
            "ssr_ramp_fraction": 0.2,
        },
        "candidate_matrix": [candidate.as_dict() for candidate in CANDIDATES],
        "development": {
            "datasets": list(DEVELOPMENT_DATASETS),
            "screen_seeds": list(SCREEN_SEEDS),
            "screen_top_k": 4,
            "refine_seeds": list(REFINE_SEEDS),
            "evaluation_split": "fixed stratified validation",
            "validation_fraction": 0.2,
            "validation_seed": 20260809,
            "selection_uses_geometry": False,
        },
        "confirmation": {
            "datasets_and_seeds": {
                name: list(seeds) for name, seeds in CONFIRMATION_SEEDS.items()
            },
            "evaluation_split": "official test after selection lock",
            "primary_metric": "mean_iou",
            "primary_effect_floor_percentage_points": PRIMARY_EFFECT_FLOOR,
            "sign_flip_draws": SIGN_FLIP_DRAWS,
        },
        "jobs": {
            "screen_pairs": screen_pairs,
            "screen_controls": screen_controls,
            "screen_treatments": screen_pairs,
            "screen_runs": screen_controls + screen_pairs,
            "refine_pairs": refine_pairs,
            "refine_controls": refine_controls,
            "refine_treatments": refine_pairs,
            "refine_runs": refine_controls + refine_pairs,
            "confirmation_pairs": confirmation_pairs,
            "confirmation_controls": confirmation_controls,
            "confirmation_treatments": confirmation_pairs,
            "confirmation_runs": confirmation_controls + confirmation_pairs,
            "total_pairs": screen_pairs + refine_pairs + confirmation_pairs,
            "total_runs": (
                screen_controls
                + screen_pairs
                + refine_controls
                + refine_pairs
                + confirmation_controls
                + confirmation_pairs
            ),
        },
        "compute": {
            "nodes": args.nodes,
            "gpus_per_node": len(args.gpus),
            "jobs_per_gpu": args.jobs_per_gpu,
            "workers": args.workers,
            "same_pair_same_gpu": True,
            "control_reuse": "one task-only run per phase, dataset and seed",
            "sharding": (
                "sorted phase/dataset/seed groups, round-robin by node rank; each "
                "group remains on one local GPU"
            ),
        },
        "data": data_payload,
    }


def barrier_path(args: argparse.Namespace, phase: str, suffix: str) -> Path:
    return args.result_root / ".barriers" / phase / suffix


def mark_node(args: argparse.Namespace, phase: str, status: str, extra: dict | None = None) -> None:
    payload = {
        "protocol": PROTOCOL,
        "node_rank": args.node_rank,
        "phase": phase,
        "status": status,
        "host": os.environ.get("HOSTNAME", "unknown"),
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        **(extra or {}),
    }
    write_json_atomic(
        barrier_path(args, phase, f"node_{args.node_rank}.{status}.json"),
        payload,
    )


def wait_for_file(args: argparse.Namespace, path: Path, description: str) -> None:
    deadline = time.monotonic() + args.barrier_timeout
    while not path.is_file():
        failures = sorted((args.result_root / ".barriers").rglob("node_*.failed.json"))
        if failures:
            details = failures[0].read_text(encoding="utf-8")
            raise RuntimeError(f"A peer failed while waiting for {description}: {details}")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for {description}: {path}")
        time.sleep(2)


def wait_for_nodes(args: argparse.Namespace, phase: str) -> None:
    for rank in range(args.nodes):
        wait_for_file(
            args,
            barrier_path(args, phase, f"node_{rank}.done.json"),
            f"{phase} node {rank}",
        )


def clear_own_barrier(args: argparse.Namespace, phase: str) -> None:
    for status in ("done", "failed"):
        barrier_path(args, phase, f"node_{args.node_rank}.{status}.json").unlink(
            missing_ok=True
        )


def publish_ready(args: argparse.Namespace, phase: str, payload: dict) -> None:
    write_json_atomic(barrier_path(args, phase, "READY.json"), payload)


def stage_ready(args: argparse.Namespace, phase: str) -> dict:
    path = barrier_path(args, phase, "READY.json")
    wait_for_file(args, path, f"{phase} readiness")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Invalid phase readiness file {path}: {error}") from error


def initialize_manifest(
    args: argparse.Namespace,
    git_commit: str,
    datasets: dict[str, DatasetSpec],
) -> dict:
    payload = manifest(args, git_commit, datasets)
    path = args.result_root / "MANIFEST.json"
    if args.node_rank == 0:
        if path.is_file():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing != payload:
                raise RuntimeError("Existing MANIFEST.json does not match this launch")
        else:
            write_json_atomic(path, payload)
        publish_ready(
            args,
            "screen",
            {"protocol": PROTOCOL, "manifest_sha256": sha256_value(payload)},
        )
    ready = stage_ready(args, "screen")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if ready.get("manifest_sha256") != sha256_value(loaded) or loaded != payload:
        raise RuntimeError("Nodes disagree on the experiment manifest")
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--node-rank", type=int, required=True)
    parser.add_argument("--nodes", type=int, default=4)
    parser.add_argument("--gpus", nargs="+", default=[str(index) for index in range(8)])
    parser.add_argument("--jobs-per-gpu", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--cub-batch-size", type=int, default=24)
    parser.add_argument("--multidataset-batch-size", type=int, default=32)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--barrier-timeout", type=int, default=50_400)
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--cub-root", type=Path, default=PROJECT_ROOT / "data" / "cub200")
    parser.add_argument(
        "--cub-cache",
        type=Path,
        default=PROJECT_ROOT / "data" / "cub200" / "cub200_seg_resnet18_dense_192.pt",
    )
    parser.add_argument(
        "--pet-root", type=Path, default=PROJECT_ROOT / "data" / "oxford_iiit_pet"
    )
    parser.add_argument(
        "--pet-cache",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "oxford_iiit_pet"
            / "resnet18_layer3_masks_160_provenance_v2.pt"
        ),
    )
    parser.add_argument(
        "--flowers-root", type=Path, default=PROJECT_ROOT / "data" / "flowers-102"
    )
    parser.add_argument(
        "--flowers-cache",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "flowers-102"
            / "resnet18_layer3_masks_160_provenance_v2.pt"
        ),
    )
    args = parser.parse_args(argv)
    for name in (
        "result_root",
        "cub_root",
        "cub_cache",
        "pet_root",
        "pet_cache",
        "flowers_root",
        "flowers_cache",
    ):
        setattr(args, name, getattr(args, name).resolve())
    if args.nodes < 1 or not 0 <= args.node_rank < args.nodes:
        parser.error("require --nodes >= 1 and 0 <= --node-rank < --nodes")
    if len(args.gpus) != 8 or len(set(args.gpus)) != 8:
        parser.error("this protocol requires eight distinct local GPU identifiers")
    if (
        args.jobs_per_gpu < 1
        or args.workers < 0
        or args.cub_batch_size < 1
        or args.multidataset_batch_size < 1
        or args.barrier_timeout < 1
    ):
        parser.error("job, batch and timeout values are invalid")
    return args


def write_result_inventory(args: argparse.Namespace) -> dict:
    records = [
        {
            "path": str(path.relative_to(args.result_root)),
            "sha256": sha256_file(path),
        }
        for path in sorted(args.result_root.rglob("result_record.json"))
    ]
    payload = {
        "protocol": PROTOCOL,
        "record_count": len(records),
        "records": records,
        "inventory_sha256": sha256_value(records),
    }
    write_json_atomic(args.result_root / "RESULT_INVENTORY.json", payload)
    return payload


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    git_commit = current_git_commit()
    args.result_root.mkdir(parents=True, exist_ok=True)
    if args.manifest_only:
        payload = manifest(args, git_commit, None)
        write_json_atomic(args.result_root / "MANIFEST.json", payload)
        print(json.dumps(payload, indent=2))
        return

    datasets = prepare_datasets(args)
    phase = "initialization"
    try:
        initialize_manifest(args, git_commit, datasets)

        phase = "screen"
        clear_own_barrier(args, phase)
        screen_jobs = pair_jobs(
            args.result_root,
            phase,
            DEVELOPMENT_DATASETS,
            {dataset: SCREEN_SEEDS for dataset in DEVELOPMENT_DATASETS},
            CANDIDATES,
        )
        run_shard(screen_jobs, args, datasets, git_commit, None)
        local_screen_groups = node_groups(screen_jobs, args.node_rank, args.nodes)
        mark_node(
            args,
            phase,
            "done",
            {
                "local_seed_groups": len(local_screen_groups),
                "local_treatments": sum(len(group) for group in local_screen_groups),
            },
        )
        if args.node_rank == 0:
            wait_for_nodes(args, phase)
            screen_rows = score_candidates(
                screen_jobs, CANDIDATES, args, git_commit
            )
            write_json_atomic(args.result_root / "SCREEN_SUMMARY.json", screen_rows)
            write_csv_atomic(
                args.result_root / "SCREEN_SUMMARY.csv",
                flat_development_rows(screen_rows),
            )
            top_four = [find_candidate(row["spec_id"]) for row in screen_rows[:4]]
            publish_ready(
                args,
                "refine",
                {
                    "protocol": PROTOCOL,
                    "screen_summary_sha256": sha256_value(screen_rows),
                    "candidate_spec_ids": [spec.spec_id for spec in top_four],
                },
            )
        refine_ready = stage_ready(args, "refine")
        top_four = tuple(
            find_candidate(spec_id) for spec_id in refine_ready["candidate_spec_ids"]
        )
        if len(top_four) != 4 or len({spec.spec_id for spec in top_four}) != 4:
            raise RuntimeError("Refinement readiness does not contain four candidates")

        phase = "refine"
        clear_own_barrier(args, phase)
        refine_jobs = pair_jobs(
            args.result_root,
            phase,
            DEVELOPMENT_DATASETS,
            {dataset: REFINE_SEEDS for dataset in DEVELOPMENT_DATASETS},
            top_four,
        )
        run_shard(refine_jobs, args, datasets, git_commit, None)
        local_refine_groups = node_groups(refine_jobs, args.node_rank, args.nodes)
        mark_node(
            args,
            phase,
            "done",
            {
                "local_seed_groups": len(local_refine_groups),
                "local_treatments": sum(len(group) for group in local_refine_groups),
            },
        )
        if args.node_rank == 0:
            wait_for_nodes(args, phase)
            screen_rows = json.loads(
                (args.result_root / "SCREEN_SUMMARY.json").read_text(encoding="utf-8")
            )
            refine_rows = score_candidates(
                refine_jobs, top_four, args, git_commit
            )
            write_json_atomic(args.result_root / "REFINE_SUMMARY.json", refine_rows)
            write_csv_atomic(
                args.result_root / "REFINE_SUMMARY.csv",
                flat_development_rows(refine_rows),
            )
            if (args.result_root / "SELECTION_LOCK.json").is_file():
                lock, selected = load_selection_lock(
                    args, git_commit, screen_rows, refine_rows
                )
            else:
                lock, selected = write_selection_lock(
                    args, git_commit, screen_rows, refine_rows
                )
            publish_ready(
                args,
                "confirmation",
                {
                    "protocol": PROTOCOL,
                    "selection_lock_sha256": lock["lock_sha256"],
                    "selected_spec_id": selected.spec_id,
                },
            )
        confirmation_ready = stage_ready(args, "confirmation")
        selected = find_candidate(confirmation_ready["selected_spec_id"])
        lock_hash = str(confirmation_ready["selection_lock_sha256"])

        phase = "confirmation"
        clear_own_barrier(args, phase)
        confirmation_jobs = pair_jobs(
            args.result_root,
            phase,
            CONFIRMATION_DATASETS,
            CONFIRMATION_SEEDS,
            (selected,),
        )
        run_shard(confirmation_jobs, args, datasets, git_commit, lock_hash)
        mark_node(
            args,
            phase,
            "done",
            {
                "local_seed_groups": len(
                    node_groups(confirmation_jobs, args.node_rank, args.nodes)
                ),
                "local_treatments": sum(
                    len(group)
                    for group in node_groups(
                        confirmation_jobs, args.node_rank, args.nodes
                    )
                ),
            },
        )
        if args.node_rank == 0:
            wait_for_nodes(args, phase)
            summary = write_confirmation_summary(
                confirmation_jobs,
                selected,
                args,
                git_commit,
                lock_hash,
            )
            inventory = write_result_inventory(args)
            completed = {
                "status": "complete",
                "protocol": PROTOCOL,
                "git_commit": git_commit,
                "selection_lock_sha256": lock_hash,
                "selected_spec_id": selected.spec_id,
                "primary_gates": {
                    dataset: summary["datasets"][dataset]["primary_gate"]
                    for dataset in CONFIRMATION_DATASETS
                },
                "all_primary_gates": summary["all_primary_gates"],
                "record_count": inventory["record_count"],
                "result_inventory_sha256": inventory["inventory_sha256"],
                "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
            write_json_atomic(args.result_root / "COMPLETED.json", completed)
            publish_ready(args, "complete", completed)
        stage_ready(args, "complete")
        print(
            f"Completed node {args.node_rank} of direct SSR segmentation study: "
            f"{args.result_root}",
            flush=True,
        )
    except BaseException as error:
        mark_node(
            args,
            phase,
            "failed",
            {"error_type": type(error).__name__, "error": str(error)},
        )
        raise


if __name__ == "__main__":
    main()
