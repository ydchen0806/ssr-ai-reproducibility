#!/usr/bin/env python3
"""Run one auditable PLOP/LowRank/LowRank+SSR 10-1 sequential stage.

This wrapper intentionally keeps the official PLOP step-by-step protocol. It
adds only SSR command-line fields and writes a per-step machine-readable
record, so a later aggregator can require every declared seed instead of
selecting favorable confirmation outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path


TASK_STEPS = {
    "10-1": tuple(range(11)),
}


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def json_load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_dry_command(args: argparse.Namespace, command: list[str]) -> None:
    """Reject a dry command that the selected PLOP parser cannot accept."""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(args.overlay) + os.pathsep + environment.get("PYTHONPATH", "")
    result = subprocess.run(
        [args.python_exe, "run.py", "--help"],
        cwd=args.plop_root,
        capture_output=True,
        text=True,
        env=environment,
    )
    if result.returncode != 0:
        raise RuntimeError("PLOP --help preflight failed: %s" % result.stderr.strip())
    run_index = command.index("run.py")
    supplied = {token for token in command[run_index + 1:] if token.startswith("--")}
    registered = set(re.findall(r"--[a-z0-9_-]+", result.stdout + result.stderr))
    unsupported = sorted(supplied - registered)
    if unsupported:
        raise RuntimeError("PLOP parser rejects dry-run options: %s" % ", ".join(unsupported))


class GpuMemorySampler:
    def __init__(self, gpu_ids: list[str]) -> None:
        self.gpu_ids = gpu_ids
        self.peak_mib = {gpu: 0 for gpu in gpu_ids}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def close(self) -> dict[str, int]:
        self._stop.set()
        self._thread.join(timeout=8)
        return self.peak_mib

    def _run(self) -> None:
        while not self._stop.is_set():
            for gpu in self.gpu_ids:
                try:
                    output = subprocess.check_output(
                        [
                            "nvidia-smi", "--id=" + gpu,
                            "--query-gpu=memory.used", "--format=csv,noheader,nounits",
                        ],
                        text=True,
                        stderr=subprocess.DEVNULL,
                    ).strip()
                    self.peak_mib[gpu] = max(self.peak_mib[gpu], int(output.splitlines()[0]))
                except (OSError, ValueError, subprocess.CalledProcessError):
                    pass
            self._stop.wait(5)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--plop-root", required=True, type=Path)
    result.add_argument("--python", required=True, dest="python_exe")
    result.add_argument("--overlay", required=True, type=Path)
    result.add_argument("--data-root", required=True, type=Path)
    result.add_argument("--trial-root", required=True, type=Path)
    result.add_argument("--task", required=True, choices=["10-1"])
    result.add_argument("--arm", required=True, choices=["plop", "lowrank", "lowrank_ssr"])
    result.add_argument("--seed", required=True, type=int)
    result.add_argument("--gpus", required=True, help="Comma-separated physical local GPU IDs.")
    result.add_argument("--ssr-lambda", type=float, default=0.0)
    result.add_argument(
        "--ssr-normalization", choices=["none", "relative_primary_v1"], default="none",
    )
    result.add_argument("--ssr-a-exc", type=float, default=1.0)
    result.add_argument("--ssr-a-inh", type=float, default=0.8)
    result.add_argument("--ssr-sigma-exc", type=float, default=0.2)
    result.add_argument("--ssr-sigma-inh", type=float, default=0.5)
    result.add_argument("--ssr-distance", choices=["projective", "cosine"], default="projective")
    result.add_argument("--ssr-start-step", type=int, default=1)
    result.add_argument(
        "--ssr-target-scope",
        choices=["current_to_old", "historical_only"],
        default="current_to_old",
    )
    result.add_argument(
        "--ssr-gradient-gate",
        choices=["none", "nonconflicting_v1"],
        default="none",
    )
    result.add_argument("--ssr-warmup-epochs", type=int, default=0)
    result.add_argument("--ssr-repulsion-margin", type=float, default=0.15)
    result.add_argument("--low-rank-classifier-rank", type=int, default=0)
    result.add_argument("--low-rank-classifier-alpha", type=float, default=1.0)
    result.add_argument("--parent-checkpoint", type=Path)
    result.add_argument("--stage0-only", action="store_true")
    result.add_argument("--batch-size", type=int, default=12)
    result.add_argument("--initial-epochs", type=int, default=30)
    result.add_argument("--incremental-epochs", type=int, default=30)
    result.add_argument("--dry-run", action="store_true")
    return result


def main() -> int:
    args = parser().parse_args()
    if args.arm in {"plop", "lowrank"} and args.ssr_lambda != 0.0:
        raise SystemExit("Reference arms must use --ssr-lambda 0.")
    if args.arm == "lowrank_ssr" and args.ssr_lambda <= 0.0:
        raise SystemExit("The LowRank+SSR arm requires a positive --ssr-lambda.")
    if args.arm == "plop" and args.low_rank_classifier_rank != 0:
        raise SystemExit("The PLOP arm must use rank zero.")
    if args.arm in {"lowrank", "lowrank_ssr"} and args.low_rank_classifier_rank <= 0:
        raise SystemExit("Low-rank arms require a positive rank.")
    if args.low_rank_classifier_rank > 0 and args.low_rank_classifier_alpha <= 0:
        raise SystemExit("Low-rank alpha must be positive.")
    if args.stage0_only:
        if args.arm != "plop" or args.parent_checkpoint is not None:
            raise SystemExit("Stage 0 must be a fresh PLOP arm without a parent checkpoint.")
    elif args.parent_checkpoint is None or not args.parent_checkpoint.is_file():
        raise SystemExit("Incremental trials require an existing --parent-checkpoint.")
    if args.ssr_start_step < 0:
        raise SystemExit("--ssr-start-step must be non-negative.")
    if args.ssr_warmup_epochs < 0:
        raise SystemExit("--ssr-warmup-epochs must be non-negative.")
    if not -1.0 < args.ssr_repulsion_margin < 1.0:
        raise SystemExit("--ssr-repulsion-margin must lie in (-1, 1).")
    gpu_ids = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if len(gpu_ids) != 2 or len(set(gpu_ids)) != len(gpu_ids):
        raise SystemExit("Each PLOP stream requires two distinct local GPUs.")
    if not args.data_root.joinpath("VOCdevkit", "VOC2012").is_dir():
        raise SystemExit("PASCAL VOC root is missing VOCdevkit/VOC2012.")
    if not args.plop_root.joinpath("run.py").is_file():
        raise SystemExit("PLOP source is incomplete.")

    trial_root = args.trial_root.resolve()
    if trial_root.exists():
        raise SystemExit("Trial root must be fresh: %s" % trial_root)
    trial_root.mkdir(parents=True)
    steps_root = trial_root / "steps"
    checkpoint_root = trial_root / "checkpoints" / "step"
    log_root = trial_root / "logs"
    log_root.mkdir(parents=True)

    source_commit = subprocess.check_output(
        ["git", "-C", str(args.plop_root), "rev-parse", "HEAD"], text=True
    ).strip()
    runtime_sources = (
        "segmentation_module.py",
        "teacher_ssr.py",
        "train.py",
        "run.py",
        "argparser.py",
    )
    runtime_source_hashes = {
        name: hashlib.sha256(args.plop_root.joinpath(name).read_bytes()).hexdigest()
        for name in runtime_sources
    }
    manifest = {
        "schema_version": "plop_ssr_lowrank_v24_sequential_trial_v1",
        "status": "running",
        "task": args.task,
        "arm": args.arm,
        "seed": args.seed,
        "gpus": gpu_ids,
        "plop_source_commit": source_commit,
        "plop_runtime_source_sha256": runtime_source_hashes,
        "trial_runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "ssr": {
            "lambda": args.ssr_lambda,
            "normalization": args.ssr_normalization,
            "a_exc": args.ssr_a_exc,
            "a_inh": args.ssr_a_inh,
            "sigma_exc": args.ssr_sigma_exc,
            "sigma_inh": args.ssr_sigma_inh,
            "distance": args.ssr_distance,
            "start_step": args.ssr_start_step,
            "target_scope": args.ssr_target_scope,
            "gradient_gate": args.ssr_gradient_gate,
            "warmup_epochs": args.ssr_warmup_epochs,
            "repulsion_margin": args.ssr_repulsion_margin,
            "target": (
                "effective_historical_low_rank_decoder_geometry"
                if args.low_rank_classifier_rank > 0 else
                "incremental_decoder_classifier_channels"
            ),
        },
        "low_rank_classifier": {
            "rank": args.low_rank_classifier_rank,
            "alpha": args.low_rank_classifier_alpha,
            "scope": "historical_decoder_classifier_residual",
        },
        "parent_checkpoint": str(args.parent_checkpoint.resolve()) if args.parent_checkpoint else None,
        "parent_checkpoint_sha256": (
            sha256_file(args.parent_checkpoint)
            if args.parent_checkpoint else None
        ),
        "started_at_unix": time.time(),
        "commands": [],
    }
    atomic_json(trial_root / "trial.json", manifest)

    trial_tag = "%s_%s_seed_%s" % (args.task, args.arm, args.seed)
    base_environment = os.environ.copy()
    base_environment["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
    base_environment["PYTHONPATH"] = str(args.overlay) + os.pathsep + base_environment.get("PYTHONPATH", "")
    sampler = GpuMemorySampler(gpu_ids)
    sampler.start()
    started = time.monotonic()
    try:
        selected_steps = (0,) if args.stage0_only else TASK_STEPS[args.task][1:]
        for step in selected_steps:
            metric_path = steps_root / ("step_%d.json" % step)
            step_epochs = args.initial_epochs if step == 0 else args.incremental_epochs
            step_lr = "0.01" if step == 0 else "0.001"
            # Distinct lambda trials run concurrently during development; include
            # the complete trial identity so their local rendezvous ports differ.
            port_key = "%s:%s:%s:%s:%s:%s" % (
                args.task,
                args.arm,
                args.seed,
                (
                    args.ssr_lambda, args.ssr_normalization,
                    args.ssr_distance, args.ssr_sigma_exc,
                    args.ssr_sigma_inh, args.ssr_start_step,
                    args.ssr_target_scope, args.ssr_gradient_gate,
                    args.ssr_warmup_epochs, args.ssr_repulsion_margin,
                    args.low_rank_classifier_rank, args.low_rank_classifier_alpha,
                ),
                trial_root,
                step,
            )
            port = 20000 + int(hashlib.sha256(port_key.encode()).hexdigest()[:6], 16) % 20000
            command = [
                args.python_exe, "-m", "torch.distributed.launch",
                "--nproc_per_node=2", "--master_port=%d" % port, "run.py",
                "--data_root", str(args.data_root), "--overlap",
                "--batch_size", str(args.batch_size), "--dataset", "voc",
                "--name", trial_tag, "--task", args.task, "--step", str(step),
                "--lr", step_lr, "--epochs", str(step_epochs), "--method", "PLOP",
                "--opt_level", "O0", "--checkpoint", str(checkpoint_root),
                "--logdir", str(log_root), "--date", "teacher_revision",
                "--random_seed", str(args.seed), "--ssr-lambda", str(args.ssr_lambda),
                "--ssr-a-exc", str(args.ssr_a_exc), "--ssr-a-inh", str(args.ssr_a_inh),
                "--ssr-sigma-exc", str(args.ssr_sigma_exc), "--ssr-sigma-inh", str(args.ssr_sigma_inh),
                "--ssr-distance", args.ssr_distance, "--ssr-start-step", str(args.ssr_start_step),
                "--ssr-target-scope", args.ssr_target_scope,
                "--ssr-gradient-gate", args.ssr_gradient_gate,
                "--ssr-warmup-epochs", str(args.ssr_warmup_epochs),
                "--ssr-repulsion-margin", str(args.ssr_repulsion_margin),
                "--ssr-normalization", args.ssr_normalization,
                "--low-rank-classifier-rank", str(args.low_rank_classifier_rank),
                "--low-rank-classifier-alpha", str(args.low_rank_classifier_alpha),
                "--teacher-metrics-json", str(metric_path),
            ]
            if step == 1:
                command.extend(["--step_ckpt", str(args.parent_checkpoint.resolve())])
            manifest["commands"].append(command)
            atomic_json(trial_root / "trial.json", manifest)
            if args.dry_run:
                validate_dry_command(args, command)
                print("DRY RUN:", " ".join(command))
                continue
            with (log_root / ("step_%d.log" % step)).open("w", encoding="utf-8") as log:
                process = subprocess.run(
                    command, cwd=args.plop_root, env=base_environment, stdout=log, stderr=subprocess.STDOUT
                )
            if process.returncode != 0:
                raise RuntimeError("PLOP step %d exited %d" % (step, process.returncode))
            payload = json_load(metric_path)
            if payload.get("schema_version") != "plop_ssr_lowrank_teacher_metrics_v2":
                raise RuntimeError("Unexpected metric schema for step %d" % step)
            if len(payload.get("steps", [])) != 1:
                raise RuntimeError("PLOP did not export one score for step %d" % step)

        if args.dry_run:
            manifest.update({
                "status": "dry_run",
                "completed_at_unix": time.time(),
                "wall_time_s": round(time.monotonic() - started, 3),
                "peak_gpu_memory_mib": sampler.close(),
            })
            atomic_json(trial_root / "trial.json", manifest)
            return 0
        step_metrics = [json_load(steps_root / ("step_%d.json" % step))["steps"][0] for step in selected_steps]
        output_checkpoint = checkpoint_root / (
            "%s-voc_%s_%d.pth" % (args.task, trial_tag, selected_steps[-1])
        )
        if not output_checkpoint.is_file():
            raise RuntimeError("Expected checkpoint is missing: %s" % output_checkpoint)
        manifest.update({
            "status": "complete",
            "completed_at_unix": time.time(),
            "wall_time_s": round(time.monotonic() - started, 3),
            "peak_gpu_memory_mib": sampler.close(),
            "step_metrics": step_metrics,
            "final_metrics": step_metrics[-1],
            "output_checkpoint": str(output_checkpoint),
            "output_checkpoint_sha256": sha256_file(output_checkpoint),
        })
        atomic_json(trial_root / "trial.json", manifest)
        return 0
    except Exception as error:
        manifest.update({
            "status": "failed",
            "failed_at_unix": time.time(),
            "wall_time_s": round(time.monotonic() - started, 3),
            "peak_gpu_memory_mib": sampler.close(),
            "error": str(error),
        })
        atomic_json(trial_root / "trial.json", manifest)
        raise
    finally:
        if sampler._thread.is_alive():
            sampler.close()


if __name__ == "__main__":
    raise SystemExit(main())
