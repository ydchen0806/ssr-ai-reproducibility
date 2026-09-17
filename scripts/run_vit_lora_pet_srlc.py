#!/usr/bin/env python3
"""Screen, refine, lock, and confirm Task+SSR on Oxford-IIIT Pet ViT-LoRA.

The development phases may select the SSR target, strength, LoRA rank, and
task-wise warm-in.  Confirmation seeds are disjoint and are not read until a
selection lock has been written.  Each treatment is paired with a task-only
control of the same rank, seed, model, data order, and optimizer budget.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Reuse the execution and strict paired-validation plumbing without changing
# the established Flowers102 protocol.
from scripts import run_vit_lora_flowers_srlc as driver
from scripts.validate_vit_lora_pairs import summarize, validate_pair
from ssr_utils.result_schema import sha256_file, validate_result_record


CONFIG = PROJECT_ROOT / "configs" / "meeting_20260804" / "vit_lora_oxfordiiitpet.yaml"
SCREEN_SEEDS = (8101, 8103, 8105)
REFINE_SEEDS = (8201, 8203, 8205, 8207, 8209, 8211, 8213)
CONFIRMATION_SEEDS = tuple(range(83001, 83060, 2))
PROTOCOL_NAME = "oxford_pet_vit_lora_task_ssr_srlc_v1"
BASELINE_ADEQUACY_TOLERANCE = 0.5

Candidate = driver.Candidate
Job = driver.Job


def candidates() -> tuple[Candidate, ...]:
    """Return the prespecified target/strength/rank/warm-in search matrix."""
    rows: list[Candidate] = []
    for rank in (8, 16, 32):
        for value in (0.001, 0.003, 0.010):
            rows.append(
                Candidate(
                    f"r{rank}_classifier_c{value:g}",
                    rank,
                    value,
                    0.0,
                    0.20,
                    0.50,
                )
            )
        for value in (0.0005, 0.002, 0.006):
            rows.append(
                Candidate(
                    f"r{rank}_adapter_a{value:g}",
                    rank,
                    0.0,
                    value,
                    0.20,
                    0.50,
                )
            )
        for classifier, adapter in (
            (0.003, 0.0006),
            (0.006, 0.0012),
            (0.010, 0.0020),
        ):
            rows.append(
                Candidate(
                    f"r{rank}_joint_c{classifier:g}_a{adapter:g}",
                    rank,
                    classifier,
                    adapter,
                    0.20,
                    0.50,
                )
            )
        for ramp_tasks in (2, 4):
            rows.append(
                Candidate(
                    f"r{rank}_joint_c0.01_a0.002_start1_ramp{ramp_tasks}",
                    rank,
                    0.010,
                    0.002,
                    0.20,
                    0.50,
                    start_task=1,
                    ramp_tasks=ramp_tasks,
                )
            )
        rows.append(
            Candidate(
                f"r{rank}_joint_narrow_c0.01_a0.002",
                rank,
                0.010,
                0.002,
                0.16,
                0.45,
            )
        )
    identifiers = [row.candidate_id for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("Duplicate Oxford-IIIT Pet candidate identifiers")
    return tuple(rows)


CANDIDATES = candidates()


def configure_driver() -> None:
    """Point shared execution plumbing at this protocol before launching jobs."""
    driver.CONFIG = CONFIG
    driver.validate_job = validate_job


def validate_job(job: Job, args: argparse.Namespace, git_commit: str) -> None:
    path = driver.output_path(args.result_root, job) / "result_record.json"
    record = validate_result_record(json.loads(path.read_text(encoding="utf-8")))
    expected = {
        "git_commit": git_commit,
        "dataset": "oxfordiiitpet_cl",
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
        raise RuntimeError(f"Oxford-IIIT Pet result identity mismatch: {mismatches}")


def candidate_target(candidate: Candidate) -> str:
    if candidate.lambda_classifier > 0 and candidate.lambda_adapter > 0:
        return "joint"
    if candidate.lambda_classifier > 0:
        return "classifier"
    if candidate.lambda_adapter > 0:
        return "adapter"
    raise ValueError("SSR candidate has no active target")


def selection_key(row: dict) -> tuple:
    """Endpoint-only ranking; geometry is reported but never selects a model."""
    means = row["means"]
    return (
        row.get("baseline_adequacy_gate", True),
        row["endpoint_gate"],
        row["dual_wins"],
        row["balanced_endpoint_score"],
        means["avg_accuracy_delta"],
        means["avg_forgetting_reduction"],
        row["candidate"]["candidate_id"],
    )


def rank_for_selection(rows: list[dict]) -> list[dict]:
    control_means = []
    for row in rows:
        paired_rows = row.get("rows", [])
        if paired_rows:
            row["control_avg_accuracy"] = statistics.mean(
                float(item["avg_accuracy_control"]) for item in paired_rows
            )
            control_means.append(row["control_avg_accuracy"])
    if control_means:
        best_control = max(control_means)
        for row in rows:
            row["best_control_avg_accuracy"] = best_control
            row["baseline_adequacy_gate"] = (
                row["control_avg_accuracy"]
                >= best_control - BASELINE_ADEQUACY_TOLERANCE
            )
    return sorted(rows, key=selection_key, reverse=True)


def job_counts(top_k: int) -> dict[str, int]:
    ranks = len({candidate.rank for candidate in CANDIDATES})
    screen_controls = ranks * len(SCREEN_SEEDS)
    screen_treatments = len(CANDIDATES) * len(SCREEN_SEEDS)
    maximum_refine_controls = ranks * len(REFINE_SEEDS)
    refine_treatments = top_k * len(REFINE_SEEDS)
    confirmation_per_arm = len(CONFIRMATION_SEEDS)
    return {
        "candidate_count": len(CANDIDATES),
        "screen_controls": screen_controls,
        "screen_treatments": screen_treatments,
        "screen_jobs": screen_controls + screen_treatments,
        "maximum_refine_controls": maximum_refine_controls,
        "refine_treatments": refine_treatments,
        "maximum_refine_jobs": maximum_refine_controls + refine_treatments,
        "confirmation_controls": confirmation_per_arm,
        "confirmation_treatments": confirmation_per_arm,
        "confirmation_jobs": 2 * confirmation_per_arm,
        "maximum_total_jobs": (
            screen_controls
            + screen_treatments
            + maximum_refine_controls
            + refine_treatments
            + 2 * confirmation_per_arm
        ),
    }


def protocol_manifest(git_commit: str, top_k: int) -> dict:
    return {
        "protocol": PROTOCOL_NAME,
        "git_commit": git_commit,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "execution_driver_sha256": sha256_file(Path(driver.__file__).resolve()),
        "config": str(CONFIG.relative_to(PROJECT_ROOT)),
        "config_sha256": sha256_file(CONFIG),
        "screen_seeds": list(SCREEN_SEEDS),
        "refine_seeds": list(REFINE_SEEDS),
        "confirmation_seeds": list(CONFIRMATION_SEEDS),
        "seed_partitions_are_disjoint": not (
            set(SCREEN_SEEDS) & set(REFINE_SEEDS)
            or set(SCREEN_SEEDS) & set(CONFIRMATION_SEEDS)
            or set(REFINE_SEEDS) & set(CONFIRMATION_SEEDS)
        ),
        "candidate_inventory": [
            {**asdict(candidate), "target": candidate_target(candidate)}
            for candidate in CANDIDATES
        ],
        "top_k": top_k,
        "selection_metrics": ["avg_accuracy", "avg_forgetting"],
        "selection_rule": (
            "Development data only; require the same-rank task-only control to be "
            f"within {BASELINE_ADEQUACY_TOLERANCE:.1f} accuracy points of the best "
            "task-only rank, then rank by endpoint gate, paired dual wins, "
            "worst mean endpoint gain, AA gain, AF reduction, candidate identifier. "
            "Geometry is an audit outcome and is not used for selection."
        ),
        "confirmation_primary_endpoint": "avg_accuracy",
        "confirmation_key_secondary": "avg_forgetting",
        "job_counts": job_counts(top_k),
    }


def write_or_validate(path: Path, payload: dict) -> None:
    if path.is_file():
        observed = json.loads(path.read_text(encoding="utf-8"))
        if observed != payload:
            raise RuntimeError(f"Existing immutable protocol file differs: {path}")
        return
    driver.write_json_atomic(path, payload)


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
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.result_root = args.result_root.resolve()
    args.data_root = args.data_root.resolve()
    args.pretrained_checkpoint = args.pretrained_checkpoint.resolve()
    if args.jobs_per_gpu < 1 or args.top_k < 1 or args.top_k > len(CANDIDATES):
        raise ValueError("jobs-per-gpu and top-k are outside the supported range")
    if not protocol_manifest("dry-run", args.top_k)["seed_partitions_are_disjoint"]:
        raise RuntimeError("Development and confirmation seeds must be disjoint")
    if args.dry_run:
        print(
            json.dumps(
                {
                    **job_counts(args.top_k),
                    "screen_seeds": list(SCREEN_SEEDS),
                    "refine_seeds": list(REFINE_SEEDS),
                    "confirmation_seed_count": len(CONFIRMATION_SEEDS),
                    "gpus": args.gpus,
                    "jobs_per_gpu": args.jobs_per_gpu,
                },
                indent=2,
            )
        )
        return

    configure_driver()
    git_commit = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    args.result_root.mkdir(parents=True, exist_ok=True)
    manifest = protocol_manifest(git_commit, args.top_k)
    manifest_path = args.result_root / "PROTOCOL_MANIFEST.json"
    write_or_validate(manifest_path, manifest)
    driver.prepare_assets(args)

    screen_controls, screen_treatments = driver.phase_jobs(
        "screen", SCREEN_SEEDS, args.screen_epochs, CANDIDATES
    )
    driver.run_jobs(screen_controls, args, git_commit)
    driver.run_jobs(screen_treatments, args, git_commit)
    screen_summary = rank_for_selection(
        driver.summarize_candidates(
            args.result_root, "screen", SCREEN_SEEDS, CANDIDATES
        )
    )
    driver.write_json_atomic(args.result_root / "SCREEN_SUMMARY.json", screen_summary)
    finalist_ids = {
        row["candidate"]["candidate_id"] for row in screen_summary[: args.top_k]
    }
    finalists = tuple(
        candidate for candidate in CANDIDATES if candidate.candidate_id in finalist_ids
    )

    refine_controls, refine_treatments = driver.phase_jobs(
        "refine", REFINE_SEEDS, args.full_epochs, finalists
    )
    driver.run_jobs(refine_controls, args, git_commit)
    driver.run_jobs(refine_treatments, args, git_commit)
    refine_summary = rank_for_selection(
        driver.summarize_candidates(
            args.result_root, "refine", REFINE_SEEDS, finalists
        )
    )
    driver.write_json_atomic(args.result_root / "REFINE_SUMMARY.json", refine_summary)
    selected = Candidate(**refine_summary[0]["candidate"])

    confirmation_root = args.result_root / "confirmation"
    lock_path = args.result_root / "SELECTION_LOCK.json"
    if not lock_path.is_file() and any(confirmation_root.glob("**/result_record.json")):
        raise RuntimeError("Confirmation records exist before a selection lock")
    evidence_paths = sorted(
        list((args.result_root / "screen").glob("**/result_record.json"))
        + list((args.result_root / "refine").glob("**/result_record.json"))
    )
    lock = {
        "protocol": PROTOCOL_NAME,
        "git_commit": git_commit,
        "protocol_manifest_sha256": sha256_file(manifest_path),
        "selected": asdict(selected),
        "selected_target": candidate_target(selected),
        "selection_gate_passed": bool(
            refine_summary[0]["endpoint_gate"]
            and refine_summary[0]["dual_wins"] >= 4
        ),
        "actual_refine_jobs": len(refine_controls) + len(refine_treatments),
        "development_evidence": [
            {
                "path": str(path.relative_to(args.result_root)),
                "sha256": sha256_file(path),
            }
            for path in evidence_paths
        ],
    }
    lock["lock_sha256"] = driver.canonical_sha256(lock)
    write_or_validate(lock_path, lock)

    confirm_controls, confirm_treatments = driver.phase_jobs(
        "confirmation", CONFIRMATION_SEEDS, args.full_epochs, (selected,)
    )
    driver.run_jobs(confirm_controls, args, git_commit)
    driver.run_jobs(confirm_treatments, args, git_commit)
    rows = [
        validate_pair(
            driver.control_path(args.result_root, "confirmation", selected.rank, seed),
            args.result_root
            / "confirmation"
            / selected.candidate_id
            / f"seed_{seed}",
        )
        for seed in CONFIRMATION_SEEDS
    ]
    summary = summarize(rows)["oxfordiiitpet_cl"]
    metrics = summary["metrics"]
    accuracy_gate = metrics["avg_accuracy"]["ci95_low"] > 0
    forgetting_gate = metrics["avg_forgetting"]["ci95_low"] > 0
    geometry_gate = (
        metrics["effective_rank"]["ci95_low"] > 0
        and metrics["prototype_overlap"]["ci95_low"] > 0
    )
    status = (
        "PASS_DUAL"
        if accuracy_gate and forgetting_gate
        else "PASS_PRIMARY"
        if accuracy_gate
        else "BOUNDARY"
    )
    payload = {
        "status": status,
        "selected": asdict(selected),
        "selection_lock_sha256": lock["lock_sha256"],
        "selection_gate_passed": lock["selection_gate_passed"],
        "primary_accuracy_gate_passed": accuracy_gate,
        "key_secondary_forgetting_gate_passed": forgetting_gate,
        "mechanism_gate_passed": geometry_gate,
        "summary": summary,
        "rows": rows,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    driver.write_json_atomic(args.result_root / "CONFIRMATION_SUMMARY.json", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
