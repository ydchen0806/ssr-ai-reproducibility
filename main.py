"""BioReg-CL: Biologically-inspired Regularization for Continual Learning.

Evaluation protocol:
  After each task, we evaluate on ALL tasks seen so far (per-task accuracy)
  to build the full T×T accuracy matrix. From this matrix we derive:
    - CIL Average Accuracy (AA_cil): mean of cumulative-test accuracy at each step
    - TIL Average Accuracy (AA_til): mean of per-task accuracies (task ID given)
    - Average Forgetting (AF): how much old-task accuracy drops
    - Backward Transfer (BWT): negative = forgetting
  This dual-protocol is essential: CIL conflates "task confusion" with
  "forgetting," while TIL isolates true knowledge retention.
"""

import argparse
import copy
import json
import logging
import subprocess
import time
import yaml
import torch
import random
import numpy as np
from pathlib import Path

from ssr_utils.result_schema import build_result_record, sha256_value


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description="BioReg-CL")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--overrides", nargs="*", default=[],
                        help="Config overrides as key=value pairs, e.g. method.lambda_reg=2.0")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def apply_overrides(config: dict, overrides: list[str]):
    for override in overrides:
        key, value = override.split("=", 1)
        parts = key.split(".")
        d = config
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        try:
            d[parts[-1]] = yaml.safe_load(value)
        except yaml.YAMLError:
            d[parts[-1]] = value


def setup_logging(output_dir: Path):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "train.log"),
        ],
    )


def dataset_fingerprint(benchmark, dataset_config: dict) -> str:
    """Fingerprint the evaluated cohort without hashing multi-gigabyte image payloads."""
    payload = {
        "config": dataset_config,
        "class_order": list(getattr(benchmark, "class_order", [])),
        "n_classes": int(getattr(benchmark, "n_classes", 0)),
        "n_tasks": int(getattr(benchmark, "n_tasks", 0)),
        "train_targets": getattr(benchmark, "train_targets", []).tolist()
        if hasattr(getattr(benchmark, "train_targets", None), "tolist")
        else list(getattr(benchmark, "train_targets", [])),
        "test_targets": getattr(benchmark, "test_targets", []).tolist()
        if hasattr(getattr(benchmark, "test_targets", None), "tolist")
        else list(getattr(benchmark, "test_targets", [])),
    }
    for split in ("train_data", "test_data"):
        dataset = getattr(benchmark, split, None)
        samples = getattr(dataset, "samples", None)
        if samples is not None:
            payload[f"{split}_samples"] = [
                [str(Path(path).name), int(label)] for path, label in samples
            ]
    return sha256_value(payload)


def current_git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown-uncommitted-environment"


def objective_for_method(method_name: str) -> dict[str, bool]:
    method_name = method_name.lower()
    objective = {"task": True}
    if method_name.startswith("kd"):
        objective["kd"] = True
    suffix_map = {
        "kd_ewc": "ewc",
        "kd_mas": "mas",
        "kd_si": "si",
        "kd_ssr": "ssr",
    }
    if method_name in suffix_map:
        objective[suffix_map[method_name]] = True
    return objective


def main():
    args = parse_args()
    config = load_config(args.config)
    apply_overrides(config, args.overrides)
    set_seed(args.seed)

    output_dir = Path(args.output_dir) / config.get("experiment_name", "default") / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir)
    logger = logging.getLogger("main")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device} | Seed: {args.seed}")

    with open(output_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    if args.wandb:
        import wandb
        wandb.init(project="bioreg-cl", config=config, name=output_dir.name)

    from datasets.builder import build_benchmark
    benchmark = build_benchmark(config["dataset"])
    data_hash = dataset_fingerprint(benchmark, config["dataset"])

    from models.builder import build_model
    model = build_model(config["model"], num_classes=benchmark.n_classes)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {config['model']['name']} | Trainable params: {n_params:,}")

    from methods.builder import build_method
    config.setdefault("training", {})["run_seed"] = args.seed
    method = build_method(config["method"], model=model, device=device)
    logger.info(f"Method: {config['method']['name']}")

    from utils.metrics import MetricTracker
    tracker = MetricTracker(n_tasks=benchmark.n_tasks)
    total_start = time.time()

    cpt = getattr(benchmark, 'classes_per_task', benchmark.n_classes // benchmark.n_tasks)

    for task_id, (train_set, test_set) in enumerate(benchmark):
        logger.info(f"{'='*60}")
        logger.info(f"Task {task_id + 1} / {benchmark.n_tasks}")
        logger.info(f"{'='*60}")

        task_start = time.time()
        method.train_task(task_id, train_set, config.get("training", {}))
        task_time = time.time() - task_start

        # --- Full evaluation: per-task TIL (masked) + cumulative CIL ---
        til_accs = []
        for eval_tid in range(task_id + 1):
            task_test = benchmark.get_test_set(eval_tid)
            task_cls = benchmark._get_task_classes(eval_tid) if hasattr(benchmark, '_get_task_classes') else None
            acc = method.evaluate(eval_tid, task_test, task_classes=task_cls)
            tracker.update(task_id, eval_tid, acc)
            til_accs.append(acc)
            logger.info(f"  TIL Task {eval_tid+1} acc: {acc:.2f}%")

        til_mean = np.mean(til_accs)
        logger.info(f"  TIL mean (tasks 1-{task_id+1}): {til_mean:.2f}%")

        cumulative_test = benchmark.get_cumulative_test_set(task_id)
        cil_acc = method.evaluate(task_id, cumulative_test)
        logger.info(f"  CIL accuracy (all {(task_id+1)*cpt} classes): {cil_acc:.2f}%")

        method.after_task(task_id)
        logger.info(f"  Task {task_id+1} wall-clock: {task_time:.1f}s")

        if args.wandb:
            import wandb
            wandb.log({
                "task": task_id + 1,
                "task_time_s": task_time,
                "cil_acc": cil_acc,
                "til_mean": til_mean,
            })

    total_time = time.time() - total_start

    results = tracker.compute_metrics()
    results["total_time_s"] = total_time
    results["seed"] = args.seed

    logger.info(f"{'='*60}")
    logger.info("Final Results:")
    logger.info(f"  TIL Avg Accuracy:   {results['avg_accuracy']:.2f}%")
    logger.info(f"  TIL Last Accuracy:  {results['last_accuracy']:.2f}%")
    logger.info(f"  Avg Forgetting:     {results['avg_forgetting']:.2f}%")
    logger.info(f"  Backward Transfer:  {results['backward_transfer']:+.2f}%")
    logger.info(f"  Total Time:         {total_time:.1f}s")
    logger.info(f"{'='*60}")
    logger.info(str(tracker))

    tracker.save(output_dir / "metrics.json")

    import datetime
    summary = {k: v for k, v in results.items() if k != "accuracy_matrix"}
    summary["experiment_name"] = config.get("experiment_name", "unknown")
    summary["method"] = config["method"]["name"]
    summary["dataset"] = config["dataset"]["name"]
    summary["model"] = config["model"]["name"]
    summary["n_tasks"] = benchmark.n_tasks
    summary["n_classes"] = benchmark.n_classes
    summary["epochs"] = config.get("training", {}).get("epochs", "?")
    summary["batch_size"] = config.get("training", {}).get("batch_size", "?")
    summary["lr"] = config.get("training", {}).get("lr", "?")
    summary["optimizer"] = config.get("training", {}).get("optimizer", "sgd")
    summary["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if "regularization_mode" in config.get("method", {}):
        summary["regularization_mode"] = config["method"]["regularization_mode"]
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    method_name = config["method"]["name"].lower()
    objective = objective_for_method(method_name)
    kernel = {}
    if objective.get("ssr"):
        kernel = {
            "family": "gaussian_dog",
            "A_exc": float(config["method"].get("A_exc", 1.0)),
            "A_inh": float(config["method"].get("A_inh", 0.8)),
            "sigma_exc": float(config["method"].get("sigma_exc", 0.16)),
            "sigma_inh": float(config["method"].get("sigma_inh", 0.45)),
        }
    pairing_config = copy.deepcopy(config)
    pairing_config.pop("experiment_name", None)
    pairing_method = pairing_config.get("method", {})
    pairing_method.pop("name", None)
    pairing_method.pop("regularizer", None)
    result_record = build_result_record(
        git_commit=current_git_commit(),
        run_id=str(output_dir.resolve()),
        task_family="classification",
        dataset=config["dataset"]["name"],
        model=config["model"]["name"],
        seed=args.seed,
        objective=objective,
        distance_mapping="cosine" if objective.get("ssr") else "none",
        kernel=kernel,
        metrics={
            key: float(results[key])
            for key in ("avg_accuracy", "last_accuracy", "avg_forgetting", "backward_transfer")
            if key in results
        },
        runtime={"elapsed_s": float(total_time)},
        config=config,
        dataset_hash=data_hash,
        recipe=method_name,
        pairing_hash=sha256_value(pairing_config),
        metric_directions={
            "avg_accuracy": True,
            "last_accuracy": True,
            "avg_forgetting": False,
            "backward_transfer": True,
        },
    )
    with (output_dir / "result_record.json").open("w", encoding="utf-8") as handle:
        json.dump(result_record, handle, indent=2, ensure_ascii=False)

    checkpoint_path = None
    if config.get("training", {}).get("save_final_model", False):
        checkpoint_dir = output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / "final_model.pt"
        torch.save(
            {
                "model_state_dict": method.model.state_dict(),
                "config": config,
                "summary": summary,
                "metrics": results,
                "seed": args.seed,
            },
            checkpoint_path,
        )
        summary["checkpoint_path"] = str(checkpoint_path)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved to {output_dir}")
    if checkpoint_path is not None:
        logger.info(f"Final model checkpoint saved to {checkpoint_path}")

    if args.wandb:
        import wandb
        wandb.log(results)
        wandb.finish()


if __name__ == "__main__":
    main()
