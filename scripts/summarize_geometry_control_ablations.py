#!/usr/bin/env python3
"""Summarize matched geometry-control ablations.

This script intentionally reports only completed seed folders. It is safe to
run before all jobs finish; incomplete methods are marked with their observed
seed count.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import yaml


DEFAULT_METHODS = [
    "lwf_split_cifar100",
    "biocs_plus_s05_c100",
    "biocs_plus_kd_only_s05_c100",
    "geomctrl_cosorth_c100",
    "geomctrl_proto_decor_c100",
    "geomctrl_spectral_c100",
    "geomctrl_center_loss_c100",
    "geomctrl_supcon_c100",
    "geomctrl_kd_weight_decay_c100",
]

REQUIRED_CONTROLS = [
    "biocs_plus_kd_only_s05_c100",
    "geomctrl_cosorth_c100",
    "geomctrl_proto_decor_c100",
    "geomctrl_spectral_c100",
    "geomctrl_center_loss_c100",
    "geomctrl_supcon_c100",
    "geomctrl_kd_weight_decay_c100",
]

METHOD_QUESTIONS = {
    "lwf_split_cifar100": "KD anchor baseline",
    "biocs_plus_s05_c100": "SSR+KD selected frontier",
    "biocs_plus_kd_only_s05_c100": "same-code SSR+ path with spatial loss disabled",
    "geomctrl_cosorth_c100": "generic pairwise cosine orthogonality",
    "geomctrl_proto_decor_c100": "prototype decorrelation / covariance reduction",
    "geomctrl_spectral_c100": "spectral regularization",
    "geomctrl_center_loss_c100": "class-center compactness",
    "geomctrl_supcon_c100": "standard supervised contrastive geometry",
    "geomctrl_kd_weight_decay_c100": "KD plus optimizer weight decay",
}

EXPECTED_PROTOCOL = {
    "dataset.name": "split_cifar100",
    "dataset.n_tasks": 10,
    "model.name": "resnet18",
    "training.epochs": 50,
    "training.batch_size": 512,
    "training.optimizer": "sgd",
    "training.lr": 0.1,
    "training.weight_decay": 0.0,
    "method.lambda_distill": 5.0,
    "method.temperature": 3.0,
}

CONFIG_PATHS = {
    "biocs_plus_kd_only_s05_c100": "configs/biocs_plus_kd_only_s05_c100.yaml",
    "geomctrl_cosorth_c100": "configs/geomctrl_cosorth_c100.yaml",
    "geomctrl_proto_decor_c100": "configs/geomctrl_proto_decor_c100.yaml",
    "geomctrl_spectral_c100": "configs/geomctrl_spectral_c100.yaml",
    "geomctrl_center_loss_c100": "configs/geomctrl_center_loss_c100.yaml",
    "geomctrl_supcon_c100": "configs/geomctrl_supcon_c100.yaml",
    "geomctrl_kd_weight_decay_c100": "configs/geomctrl_kd_weight_decay_c100.yaml",
}

EXPECTED_CONTROLS = {
    "biocs_plus_kd_only_s05_c100": None,
    "geomctrl_cosorth_c100": "cosine_orthogonal",
    "geomctrl_proto_decor_c100": "prototype_decorrelation",
    "geomctrl_spectral_c100": "spectral",
    "geomctrl_center_loss_c100": "center_loss",
    "geomctrl_supcon_c100": "supervised_contrastive",
    "geomctrl_kd_weight_decay_c100": "none",
}


def mean_sd(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def get_nested(data: dict, dotted: str):
    cur = data
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def isclose_or_equal(observed, expected) -> bool:
    if isinstance(expected, float):
        try:
            return math.isclose(float(observed), expected, rel_tol=1e-9, abs_tol=1e-9)
        except (TypeError, ValueError):
            return False
    return observed == expected


def audit_config(method: str, repo_root: Path) -> dict:
    cfg_rel = CONFIG_PATHS.get(method)
    if cfg_rel is None:
        return {"status": "not_applicable"}
    cfg_path = repo_root / cfg_rel
    if not cfg_path.exists():
        return {"status": "missing_config", "path": cfg_rel}
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    mismatches = []
    expected_control = EXPECTED_CONTROLS.get(method)
    observed_control = get_nested(cfg, "method.control_type")
    if expected_control is not None and observed_control != expected_control:
        mismatches.append(
            {
                "field": "method.control_type",
                "expected": expected_control,
                "observed": observed_control,
            }
        )
    if method == "biocs_plus_kd_only_s05_c100":
        special_expected = {
            "method.name": "biocs_plus",
            "method.lambda_spatial": 0.0,
            "method.lambda_replay": 0.0,
            "method.lambda_spectral": 0.0,
        }
        for field, expected in special_expected.items():
            observed = get_nested(cfg, field)
            if not isclose_or_equal(observed, expected):
                mismatches.append({"field": field, "expected": expected, "observed": observed})
    for field, expected in EXPECTED_PROTOCOL.items():
        if method == "geomctrl_kd_weight_decay_c100" and field == "training.weight_decay":
            expected = 0.05
        observed = get_nested(cfg, field)
        if not isclose_or_equal(observed, expected):
            mismatches.append({"field": field, "expected": expected, "observed": observed})
    return {
        "status": "matched" if not mismatches else "mismatch",
        "path": cfg_rel,
        "mismatches": mismatches,
    }


def sanitize_for_json(value):
    if isinstance(value, dict):
        return {k: sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(v) for v in value]
    if isinstance(value, tuple):
        return [sanitize_for_json(v) for v in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def read_method(result_root: Path, name: str, repo_root: Path) -> dict:
    aas: list[float] = []
    afs: list[float] = []
    seeds: list[str] = []
    paths: list[str] = []
    for summary_path in sorted((result_root / name).glob("seed_*/summary.json")):
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        if "avg_accuracy" not in summary or "avg_forgetting" not in summary:
            continue
        aas.append(float(summary["avg_accuracy"]))
        afs.append(float(summary["avg_forgetting"]))
        seeds.append(summary_path.parent.name.replace("seed_", ""))
        paths.append(str(summary_path))
    aa_mean, aa_sd = mean_sd(aas)
    af_mean, af_sd = mean_sd(afs)
    return {
        "method": name,
        "n": len(aas),
        "seeds": seeds,
        "paths": paths,
        "aa_mean": aa_mean,
        "aa_sd": aa_sd,
        "af_mean": af_mean,
        "af_sd": af_sd,
        "config_audit": audit_config(name, repo_root),
    }


def fmt(mean: float, sd: float) -> str:
    if math.isnan(mean):
        return "NA"
    return f"{mean:.2f}+/-{sd:.2f}"


def missing_seeds(row: dict, required_seeds: list[str]) -> list[str]:
    present = {str(seed) for seed in row["seeds"]}
    return [seed for seed in required_seeds if seed not in present]


def completion_status(row: dict, required_seeds: list[str]) -> str:
    missing = missing_seeds(row, required_seeds)
    if not missing:
        return "completed"
    if row["n"] == 0:
        return "missing"
    return "partial"


def gate_status(rows: list[dict], required_controls: list[str], required_seeds: list[str]) -> tuple[str, list[str]]:
    row_by_method = {row["method"]: row for row in rows}
    missing = []
    for method in required_controls:
        row = row_by_method.get(method)
        if row is None:
            missing.append(f"{method}: method not summarized")
            continue
        seed_gap = missing_seeds(row, required_seeds)
        if seed_gap:
            missing.append(f"{method}: missing seeds {','.join(seed_gap)}")
        cfg = row.get("config_audit", {})
        if cfg.get("status") != "matched":
            missing.append(f"{method}: config audit {cfg.get('status', 'missing')}")
    return ("complete" if not missing else "not_ready", missing)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", default="results")
    parser.add_argument("--methods", nargs="*", default=DEFAULT_METHODS)
    parser.add_argument("--required-seeds", nargs="*", default=["42", "123", "456"])
    parser.add_argument("--required-controls", nargs="*", default=REQUIRED_CONTROLS)
    parser.add_argument("--out", default="docs/geometry_control_ablations_20260523.md")
    parser.add_argument("--json-out", default="docs/geometry_control_ablations_20260523.json")
    args = parser.parse_args()

    root = Path(args.result_root)
    repo_root = Path(".").resolve()
    rows = [read_method(root, name, repo_root) for name in args.methods]
    gate, missing = gate_status(rows, args.required_controls, args.required_seeds)

    out_lines = [
        "# Matched geometry-control ablations",
        "",
        "All rows use the Split-CIFAR-100 continual-learning protocol. Values are mean +/- SD over completed seeds.",
        "",
        f"**Broad AI-method claim gate:** `{gate}`.",
        "",
        "SSR+ should not be claimed to beat generic geometric regularization unless every required control below is completed under the same seed set, training budget, optimizer family, KD strength, and task stream.",
        "",
        f"Required seed set: `{','.join(args.required_seeds)}`.",
        "",
        "| Method | Question tested | Status | Completed seeds | Missing seeds | AA | AF |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        missing_for_row = missing_seeds(row, args.required_seeds)
        status = completion_status(row, args.required_seeds)
        out_lines.append(
            f"| `{row['method']}` | {METHOD_QUESTIONS.get(row['method'], 'additional method')} | "
            f"{status} | {row['n']} ({','.join(row['seeds'])}) | "
            f"{','.join(missing_for_row) if missing_for_row else '-'} | "
            f"{fmt(row['aa_mean'], row['aa_sd'])} | {fmt(row['af_mean'], row['af_sd'])} |"
        )
    out_lines.append("")
    out_lines.append("## Missing controls")
    out_lines.append("")
    if missing:
        for item in missing:
            out_lines.append(f"- {item}")
    else:
        out_lines.append("- none")
    out_lines.append("")
    out_lines.append("## Config audit")
    out_lines.append("")
    out_lines.append("| Method | Config | Protocol status |")
    out_lines.append("|---|---|---:|")
    for row in rows:
        cfg = row["config_audit"]
        if cfg.get("status") == "not_applicable":
            continue
        out_lines.append(f"| `{row['method']}` | `{cfg.get('path', '-')}` | {cfg.get('status')} |")
    out_lines.append("")
    out_lines.append("## Run command")
    out_lines.append("")
    out_lines.append("```bash")
    out_lines.append("SEEDS=\"42 123 456\" GPUS=\"0 1 2 3\" bash scripts/run_geometry_control_gate_4gpu.sh")
    out_lines.append(f"python scripts/summarize_geometry_control_ablations.py --out {args.out}")
    out_lines.append("```")
    out_lines.append("")
    out_lines.append("## Interpretation rule")
    out_lines.append("")
    if gate == "complete":
        out_lines.append(
            "Because the gate is `complete`, manuscript claims about SSR+ relative to "
            "generic geometry controls should use the values above and remain limited to "
            "this matched Split-CIFAR-100 protocol."
        )
    else:
        out_lines.append(
            "Before the gate is `complete`, the manuscript may say only that matched controls "
            "have been implemented and queued. It should not say that SSR+ is distinct from "
            "ordinary cosine orthogonality, prototype decorrelation, center loss, supervised "
            "contrastive learning, spectral regularization, KD itself, or weight decay."
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    json_path = Path(args.json_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(
            sanitize_for_json(
            {
                "result_root": str(root),
                "required_seeds": args.required_seeds,
                "required_controls": args.required_controls,
                "broad_ai_method_claim_gate": gate,
                "missing": missing,
                "rows": rows,
            },
            ),
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(out_path)


if __name__ == "__main__":
    main()
