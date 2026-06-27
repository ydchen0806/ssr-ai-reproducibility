#!/usr/bin/env python3
"""Summarize SSR+ SOTA/frontier sweeps for manuscript rows 2-4."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CONFIG_DIR = ROOT / "configs" / "sota_sweeps"
OUT_JSON = ROOT / "docs" / "sota_sweeps_20260525.json"
OUT_MD = ROOT / "docs" / "sota_sweeps_20260525.md"

COMPARATORS = {
    "split_cifar10": "lwf_split_cifar10",
    "split_tiny_imagenet": "lwf_split_tinyimagenet",
    "five_datasets": "er_ace_5datasets",
}


def mean_sem(values: list[float]) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(statistics.mean(values)), float(statistics.stdev(values) / math.sqrt(len(values)))


def load_experiment(exp_name: str) -> dict:
    rows = []
    for path in sorted((RESULTS / exp_name).glob("seed_*/summary.json")):
        data = json.loads(path.read_text())
        rows.append(data)
    aas = [float(r["avg_accuracy"]) for r in rows]
    afs = [float(r["avg_forgetting"]) for r in rows]
    aa_mean, aa_sem = mean_sem(aas)
    af_mean, af_sem = mean_sem(afs)
    return {
        "experiment_name": exp_name,
        "n": len(rows),
        "dataset": rows[0].get("dataset") if rows else None,
        "method": rows[0].get("method") if rows else None,
        "model": rows[0].get("model") if rows else None,
        "aa_values": aas,
        "af_values": afs,
        "aa_mean": aa_mean,
        "aa_sem": aa_sem,
        "af_mean": af_mean,
        "af_sem": af_sem,
        "seeds": [r.get("seed") for r in rows],
        "checkpoint_paths": [r.get("checkpoint_path") for r in rows if r.get("checkpoint_path")],
    }


def load_config_index() -> dict[str, dict]:
    index = {}
    for path in sorted(CONFIG_DIR.glob("*.yaml")):
        cfg = yaml.safe_load(path.read_text())
        index[cfg["experiment_name"]] = {
            "config_path": str(path.relative_to(ROOT)),
            "dataset": cfg["dataset"]["name"],
            "method": cfg["method"],
            "training": cfg["training"],
            "model": cfg["model"],
        }
    return index


def main() -> None:
    config_index = load_config_index()
    experiments = []
    for exp_name, meta in config_index.items():
        result = load_experiment(exp_name)
        result.update(meta)
        if result["n"] > 0:
            comp = load_experiment(COMPARATORS[result["dataset"]])
            result["comparator"] = {
                "experiment_name": comp["experiment_name"],
                "aa_mean": comp["aa_mean"],
                "aa_sem": comp["aa_sem"],
                "af_mean": comp["af_mean"],
                "af_sem": comp["af_sem"],
                "n": comp["n"],
            }
            result["delta_aa_vs_comparator"] = result["aa_mean"] - comp["aa_mean"]
            result["delta_af_vs_comparator"] = result["af_mean"] - comp["af_mean"]
        experiments.append(result)

    by_dataset: dict[str, list[dict]] = {}
    for row in experiments:
        if row["n"] > 0:
            by_dataset.setdefault(row["dataset"], []).append(row)

    selected = {}
    for dataset, rows in by_dataset.items():
        selected[dataset] = {
            "best_aa": max(rows, key=lambda r: (r["aa_mean"], -r["af_mean"]))["experiment_name"],
            "best_af": min(rows, key=lambda r: (r["af_mean"], -r["aa_mean"]))["experiment_name"],
            "best_frontier": max(
                rows,
                key=lambda r: (
                    r["aa_mean"] - r["comparator"]["aa_mean"]
                    + 0.25 * (r["comparator"]["af_mean"] - r["af_mean"]),
                    r["aa_mean"],
                ),
            )["experiment_name"],
        }

    payload = {
        "comparators": COMPARATORS,
        "selected": selected,
        "experiments": experiments,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# SSR+ SOTA/Frontier Sweeps (2026-05-25)",
        "",
        "Rows are grouped by the manuscript Table 1 rows that did not have the highest average accuracy.",
        "AA/AF are mean ± SEM over completed seeds. Positive ΔAA is better than the matched comparator; negative ΔAF is lower forgetting.",
        "",
    ]
    labels = {
        "split_cifar10": "Split-CIFAR-10",
        "split_tiny_imagenet": "Split-TinyImageNet",
        "five_datasets": "5-dataset stream",
    }
    for dataset in ["split_cifar10", "split_tiny_imagenet", "five_datasets"]:
        rows = sorted(by_dataset.get(dataset, []), key=lambda r: (-r["aa_mean"], r["af_mean"]))
        lines += [f"## {labels[dataset]}", ""]
        comp = load_experiment(COMPARATORS[dataset])
        lines.append(
            f"Comparator `{comp['experiment_name']}`: AA {comp['aa_mean']:.2f}±{comp['aa_sem']:.2f}, "
            f"AF {comp['af_mean']:.2f}±{comp['af_sem']:.2f} (n={comp['n']})."
        )
        lines.append("")
        lines.append("| Experiment | n | AA | AF | ΔAA | ΔAF | Config | Checkpoints |")
        lines.append("|---|---:|---:|---:|---:|---:|---|---:|")
        for r in rows:
            lines.append(
                f"| `{r['experiment_name']}` | {r['n']} | "
                f"{r['aa_mean']:.2f}±{r['aa_sem']:.2f} | {r['af_mean']:.2f}±{r['af_sem']:.2f} | "
                f"{r['delta_aa_vs_comparator']:+.2f} | {r['delta_af_vs_comparator']:+.2f} | "
                f"`{r['config_path']}` | {len(r['checkpoint_paths'])} |"
            )
        if dataset in selected:
            sel = selected[dataset]
            lines.append("")
            lines.append(
                f"Selected: best AA `{sel['best_aa']}`, best AF `{sel['best_af']}`, "
                f"frontier `{sel['best_frontier']}`."
            )
        lines.append("")

    lines += [
        "## Selected Frontier Manifest",
        "",
        "All selected rows use `python main.py` with `methods/biocs_plus.py` and `torchvision` ResNet-18 from `models/builder.py`.",
        "Per-seed hyperparameters are saved in each `results/<experiment>/seed_<seed>/config.yaml`; final model weights are saved in each `checkpoints/final_model.pt`.",
        "",
        "| Dataset | Selected experiment | Model | Config | Saved checkpoints |",
        "|---|---|---|---|---|",
    ]
    for dataset in ["split_cifar10", "split_tiny_imagenet", "five_datasets"]:
        if dataset not in selected:
            continue
        exp_name = selected[dataset]["best_frontier"]
        row = next(r for r in experiments if r["experiment_name"] == exp_name)
        ckpts = "<br>".join(f"`{p}`" for p in row["checkpoint_paths"])
        lines.append(
            f"| {labels[dataset]} | `{exp_name}` | `{row['model']['name']}` | "
            f"`{row['config_path']}` | {ckpts} |"
        )
    lines.append("")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT_JSON.relative_to(ROOT)}")
    print(f"Wrote {OUT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
