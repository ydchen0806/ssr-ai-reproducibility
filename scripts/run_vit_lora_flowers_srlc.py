#!/usr/bin/env python3
"""Screen, refine, lock, and confirm Flowers102 task-loss + SSR ViT-LoRA."""

from __future__ import annotations

import argparse
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

from scripts.validate_vit_lora_pairs import summarize, validate_pair
from ssr_utils.result_schema import sha256_file, validate_result_record


CONFIG = PROJECT_ROOT / "configs" / "meeting_20260804" / "vit_lora_flowers102.yaml"
SCREEN_SEEDS = (6101, 6103, 6105)
REFINE_SEEDS = (6201, 6203, 6205, 6207, 6209)
CONFIRMATION_SEEDS = tuple(range(7201, 7220, 2))
PRINT_LOCK = Lock()


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    rank: int
    lambda_classifier: float
    lambda_adapter: float
    sigma_exc: float
    sigma_inh: float
    start_task: int = 0
    ramp_tasks: int = 0

    @property
    def alpha(self) -> float:
        return float(2 * self.rank)


def candidates() -> tuple[Candidate, ...]:
    rows: list[Candidate] = []
    for classifier, adapter in (
        (0.001, 0.0002),
        (0.003, 0.0006),
        (0.006, 0.0012),
        (0.010, 0.0020),
    ):
        rows.append(
            Candidate(
                f"r8_joint_c{classifier:g}_a{adapter:g}",
                8,
                classifier,
                adapter,
                0.20,
                0.50,
            )
        )
    for classifier in (0.001, 0.003):
        rows.append(
            Candidate(
                f"r8_classifier_c{classifier:g}",
                8,
                classifier,
                0.0,
                0.20,
                0.50,
            )
        )
    for adapter in (0.0005, 0.001, 0.002, 0.004):
        rows.append(
            Candidate(
                f"r8_adapter_a{adapter:g}",
                8,
                0.0,
                adapter,
                0.20,
                0.50,
            )
        )
    for rank in (16, 32):
        for adapter in (0.001, 0.002, 0.004):
            rows.append(
                Candidate(
                    f"r{rank}_adapter_broad_a{adapter:g}",
                    rank,
                    0.0,
                    adapter,
                    0.30,
                    0.90,
                )
            )
        rows.append(
            Candidate(
                f"r{rank}_adapter_broad_a0.002_ramp3",
                rank,
                0.0,
                0.002,
                0.30,
                0.90,
                start_task=1,
                ramp_tasks=3,
            )
        )
        for width_id, sigma_exc, sigma_inh in (
            ("narrow", 0.16, 0.45),
            ("reference", 0.20, 0.50),
        ):
            rows.append(
                Candidate(
                    f"r{rank}_adapter_{width_id}_a0.002",
                    rank,
                    0.0,
                    0.002,
                    sigma_exc,
                    sigma_inh,
                )
            )
    identifiers = [row.candidate_id for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("Duplicate Flowers102 candidate identifiers")
    return tuple(rows)


CANDIDATES = candidates()


@dataclass(frozen=True)
class Job:
    phase: str
    seed: int
    epochs: int
    rank: int
    candidate: Candidate | None

    @property
    def experiment_name(self) -> str:
        if self.candidate is None:
            return f"task_rank{self.rank}"
        return self.candidate.candidate_id


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def output_path(result_root: Path, job: Job) -> Path:
    return result_root / job.phase / job.experiment_name / f"seed_{job.seed}"


def control_path(result_root: Path, phase: str, rank: int, seed: int) -> Path:
    return result_root / phase / f"task_rank{rank}" / f"seed_{seed}"


def command(job: Job, args: argparse.Namespace) -> list[str]:
    candidate = job.candidate
    rank = candidate.rank if candidate is not None else job.rank
    alpha = candidate.alpha if candidate is not None else float(2 * rank)
    method_values = {
        "lambda_spatial": candidate.lambda_classifier if candidate else 0.0,
        "lambda_adapter": candidate.lambda_adapter if candidate else 0.0,
        "sigma_exc": candidate.sigma_exc if candidate else 0.20,
        "sigma_inh": candidate.sigma_inh if candidate else 0.50,
        "ssr_start_task": candidate.start_task if candidate else 0,
        "ssr_ramp_tasks": candidate.ramp_tasks if candidate else 0,
    }
    overrides = [
        f"experiment_name={job.experiment_name}",
        f"dataset.data_root={args.data_root}",
        f"model.pretrained_checkpoint={args.pretrained_checkpoint}",
        f"training.epochs={job.epochs}",
        f"training.num_workers={args.workers}",
        f"method.lora_rank={rank}",
        f"method.lora_alpha={alpha}",
        f"method.recipe={'ssr_only' if candidate else 'plain'}",
        "method.lambda_spectral=0.0",
    ]
    overrides.extend(f"method.{key}={value}" for key, value in method_values.items())
    return [
        args.python,
        str(PROJECT_ROOT / "main.py"),
        "--config",
        str(CONFIG),
        "--seed",
        str(job.seed),
        "--device",
        "cuda",
        "--output_dir",
        str(args.result_root / job.phase),
        "--overrides",
        *overrides,
    ]


def validate_job(job: Job, args: argparse.Namespace, git_commit: str) -> None:
    path = output_path(args.result_root, job) / "result_record.json"
    record = validate_result_record(json.loads(path.read_text(encoding="utf-8")))
    expected = {
        "git_commit": git_commit,
        "dataset": "flowers102_cl",
        "model": "vit_tiny_patch16_224",
        "seed": job.seed,
        "recipe": "ssr_only" if job.candidate else "plain",
    }
    mismatches = {
        key: {"expected": value, "observed": record.get(key)}
        for key, value in expected.items()
        if record.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Flowers102 result identity mismatch: {mismatches}")


def run_worker(
    gpu: str,
    jobs: list[Job],
    args: argparse.Namespace,
    git_commit: str,
) -> None:
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = gpu
    for job in jobs:
        output = output_path(args.result_root, job)
        record = output / "result_record.json"
        if record.is_file():
            validate_job(job, args, git_commit)
            continue
        output.mkdir(parents=True, exist_ok=True)
        log = args.result_root / "logs" / f"{job.phase}_{job.experiment_name}_{job.seed}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with PRINT_LOCK:
            print(
                f"RUN gpu={gpu} phase={job.phase} arm={job.experiment_name} "
                f"seed={job.seed}",
                flush=True,
            )
        with log.open("a", encoding="utf-8") as handle:
            completed = subprocess.run(
                command(job, args),
                cwd=PROJECT_ROOT,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
        if completed.returncode != 0 or not record.is_file():
            raise RuntimeError(f"Flowers102 job failed; see {log}")
        validate_job(job, args, git_commit)


def run_jobs(jobs: list[Job], args: argparse.Namespace, git_commit: str) -> None:
    slots = [gpu for gpu in args.gpus for _ in range(args.jobs_per_gpu)]
    assignments = [jobs[index :: len(slots)] for index in range(len(slots))]
    with ThreadPoolExecutor(max_workers=len(slots)) as pool:
        futures = [
            pool.submit(run_worker, gpu, assigned, args, git_commit)
            for gpu, assigned in zip(slots, assignments)
            if assigned
        ]
        for future in futures:
            future.result()


def phase_jobs(
    phase: str,
    seeds: tuple[int, ...],
    epochs: int,
    selected_candidates: tuple[Candidate, ...],
) -> tuple[list[Job], list[Job]]:
    ranks = sorted({candidate.rank for candidate in selected_candidates})
    controls = [Job(phase, seed, epochs, rank, None) for rank in ranks for seed in seeds]
    treatments = [
        Job(phase, seed, epochs, candidate.rank, candidate)
        for candidate in selected_candidates
        for seed in seeds
    ]
    return controls, treatments


def summarize_candidates(
    result_root: Path,
    phase: str,
    seeds: tuple[int, ...],
    selected_candidates: tuple[Candidate, ...],
) -> list[dict]:
    summaries = []
    for candidate in selected_candidates:
        rows = [
            validate_pair(
                control_path(result_root, phase, candidate.rank, seed),
                result_root / phase / candidate.candidate_id / f"seed_{seed}",
            )
            for seed in seeds
        ]
        means = {
            key: statistics.mean(float(row[key]) for row in rows)
            for key in (
                "avg_accuracy_delta",
                "avg_forgetting_reduction",
                "effective_rank_delta",
                "prototype_overlap_reduction",
            )
        }
        endpoint_gate = (
            means["avg_accuracy_delta"] > 0
            and means["avg_forgetting_reduction"] > 0
        )
        geometry_gate = (
            means["effective_rank_delta"] > 0
            and means["prototype_overlap_reduction"] > 0
        )
        summaries.append(
            {
                "candidate": asdict(candidate),
                "means": means,
                "endpoint_gate": endpoint_gate,
                "geometry_gate": geometry_gate,
                "dual_wins": sum(
                    row["avg_accuracy_delta"] > 0
                    and row["avg_forgetting_reduction"] > 0
                    for row in rows
                ),
                "geometry_wins": sum(
                    row["effective_rank_delta"] > 0
                    and row["prototype_overlap_reduction"] > 0
                    for row in rows
                ),
                "balanced_endpoint_score": min(
                    means["avg_accuracy_delta"],
                    means["avg_forgetting_reduction"],
                ),
                "rows": rows,
            }
        )
    summaries.sort(
        key=lambda row: (
            row["endpoint_gate"] and row["geometry_gate"],
            row["endpoint_gate"],
            row["dual_wins"],
            row["balanced_endpoint_score"],
            row["geometry_gate"],
            row["geometry_wins"],
        ),
        reverse=True,
    )
    return summaries


def prepare_assets(args: argparse.Namespace) -> None:
    command_line = [
        args.python,
        str(PROJECT_ROOT / "scripts" / "prepare_vit_lora_assets.py"),
        "--configs",
        str(CONFIG),
        "--data-root",
        str(args.data_root),
        "--pretrained-checkpoint",
        str(args.pretrained_checkpoint),
    ]
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = args.gpus[0]
    subprocess.run(command_line, cwd=PROJECT_ROOT, env=environment, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument(
        "--pretrained-checkpoint",
        type=Path,
        default=PROJECT_ROOT
        / "data/pretrained/vit_tiny_patch16_224_augreg_in21k_ft_in1k.safetensors",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpus", nargs="+", default=[str(index) for index in range(8)])
    parser.add_argument("--jobs-per-gpu", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--screen-epochs", type=int, default=8)
    parser.add_argument("--full-epochs", type=int, default=15)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.result_root = args.result_root.resolve()
    args.data_root = args.data_root.resolve()
    args.pretrained_checkpoint = args.pretrained_checkpoint.resolve()
    if args.jobs_per_gpu < 1 or args.top_k < 1 or args.top_k > len(CANDIDATES):
        raise ValueError("jobs-per-gpu and top-k are outside the supported range")

    screen_controls, screen_treatments = phase_jobs(
        "screen", SCREEN_SEEDS, args.screen_epochs, CANDIDATES
    )
    if args.dry_run:
        payload = {
            "candidates": len(CANDIDATES),
            "screen_jobs": len(screen_controls) + len(screen_treatments),
            "maximum_refine_jobs": args.top_k * len(REFINE_SEEDS)
            + 3 * len(REFINE_SEEDS),
            "confirmation_jobs": 2 * len(CONFIRMATION_SEEDS),
            "gpus": args.gpus,
            "jobs_per_gpu": args.jobs_per_gpu,
        }
        print(json.dumps(payload, indent=2))
        return

    git_commit = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    args.result_root.mkdir(parents=True, exist_ok=True)
    prepare_assets(args)

    run_jobs(screen_controls, args, git_commit)
    run_jobs(screen_treatments, args, git_commit)
    screen_summary = summarize_candidates(
        args.result_root, "screen", SCREEN_SEEDS, CANDIDATES
    )
    write_json_atomic(args.result_root / "SCREEN_SUMMARY.json", screen_summary)
    finalist_ids = {
        row["candidate"]["candidate_id"] for row in screen_summary[: args.top_k]
    }
    finalists = tuple(
        candidate for candidate in CANDIDATES if candidate.candidate_id in finalist_ids
    )

    refine_controls, refine_treatments = phase_jobs(
        "refine", REFINE_SEEDS, args.full_epochs, finalists
    )
    run_jobs(refine_controls, args, git_commit)
    run_jobs(refine_treatments, args, git_commit)
    refine_summary = summarize_candidates(
        args.result_root, "refine", REFINE_SEEDS, finalists
    )
    write_json_atomic(args.result_root / "REFINE_SUMMARY.json", refine_summary)
    selected = Candidate(**refine_summary[0]["candidate"])
    selection_gate = bool(
        refine_summary[0]["endpoint_gate"]
        and refine_summary[0]["geometry_gate"]
        and refine_summary[0]["dual_wins"] >= 3
    )
    evidence_paths = sorted(
        list((args.result_root / "screen").glob("**/result_record.json"))
        + list((args.result_root / "refine").glob("**/result_record.json"))
    )
    lock = {
        "protocol": "flowers102_vit_lora_screen_refine_lock_confirm_v1",
        "git_commit": git_commit,
        "screen_seeds": list(SCREEN_SEEDS),
        "refine_seeds": list(REFINE_SEEDS),
        "confirmation_seeds_hidden_during_selection": list(CONFIRMATION_SEEDS),
        "candidate_inventory": [asdict(candidate) for candidate in CANDIDATES],
        "selected": asdict(selected),
        "selection_gate_passed": selection_gate,
        "selection_rule": (
            "Lexicographic: joint endpoint-and-geometry gate, endpoint gate, "
            "dual wins, worst mean endpoint gain, geometry gate, geometry wins."
        ),
        "evidence": [
            {
                "path": str(path.relative_to(args.result_root)),
                "sha256": sha256_file(path),
            }
            for path in evidence_paths
        ],
    }
    lock["lock_sha256"] = canonical_sha256(lock)
    write_json_atomic(args.result_root / "SELECTION_LOCK.json", lock)

    confirm_controls, confirm_treatments = phase_jobs(
        "confirmation", CONFIRMATION_SEEDS, args.full_epochs, (selected,)
    )
    run_jobs(confirm_controls, args, git_commit)
    run_jobs(confirm_treatments, args, git_commit)
    rows = [
        validate_pair(
            control_path(args.result_root, "confirmation", selected.rank, seed),
            args.result_root
            / "confirmation"
            / selected.candidate_id
            / f"seed_{seed}",
        )
        for seed in CONFIRMATION_SEEDS
    ]
    summary = summarize(rows)["flowers102_cl"]
    metrics = summary["metrics"]
    primary_gate = (
        metrics["avg_accuracy"]["ci95_low"] > 0
        and metrics["avg_forgetting"]["ci95_low"] > 0
    )
    mechanism_gate = (
        metrics["effective_rank"]["ci95_low"] > 0
        and metrics["prototype_overlap"]["ci95_low"] > 0
    )
    payload = {
        "status": "PASS" if primary_gate else "BOUNDARY",
        "selected": asdict(selected),
        "selection_lock_sha256": lock["lock_sha256"],
        "selection_gate_passed": selection_gate,
        "primary_endpoint_gate_passed": primary_gate,
        "mechanism_gate_passed": mechanism_gate,
        "summary": summary,
        "rows": rows,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    write_json_atomic(args.result_root / "CONFIRMATION_SUMMARY.json", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
