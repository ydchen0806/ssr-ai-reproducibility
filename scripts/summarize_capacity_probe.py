#!/usr/bin/env python3
"""Summarize capacity_mechanism_probe.py outputs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


METRICS = [
    "avg_accuracy",
    "capacity_at_50",
    "capacity_at_40",
    "avg_forgetting",
    "effective_rank",
    "spectral_entropy",
    "mean_abs_offdiag_cosine",
    "total_classifier_movement",
    "fourier_high_freq_ratio",
]


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summarize(rows: list[dict]) -> dict[str, dict[str, dict[str, float]]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)

    summary = {}
    for method, items in sorted(grouped.items()):
        summary[method] = {}
        for metric in METRICS:
            vals = np.asarray([float(item[metric]) for item in items], dtype=float)
            summary[method][metric] = {
                "mean": float(vals.mean()),
                "std": float(vals.std()),
                "n": int(vals.size),
            }
    return summary


def write_csv(summary: dict, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "metric", "mean", "std", "n"])
        for method, metrics in summary.items():
            for metric, stat in metrics.items():
                writer.writerow([method, metric, stat["mean"], stat["std"], stat["n"]])


def fmt(stat: dict[str, float], digits: int = 3) -> str:
    return f"{stat['mean']:.{digits}f} ± {stat['std']:.{digits}f}"


def write_md(summary: dict, path: Path) -> None:
    lines = [
        "# Capacity / Mechanism Probe Summary",
        "",
        "This table summarizes completed rows from `runs.jsonl`.",
        "",
        "| Method | AA | Cap@50 | Cap@40 | AF | Eff. rank | |offdiag cos| | Movement | HF ratio |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, metrics in summary.items():
        lines.append(
            "| "
            + " | ".join(
                [
                    method,
                    fmt(metrics["avg_accuracy"]),
                    fmt(metrics["capacity_at_50"], 1),
                    fmt(metrics["capacity_at_40"], 1),
                    fmt(metrics["avg_forgetting"]),
                    fmt(metrics["effective_rank"]),
                    fmt(metrics["mean_abs_offdiag_cosine"]),
                    fmt(metrics["total_classifier_movement"]),
                    fmt(metrics["fourier_high_freq_ratio"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Interpretation notes:",
            "",
            "- `Cap@50/40` is an operational capacity proxy: the largest number of sequentially introduced classes before running mean accuracy drops below the threshold.",
            "- Higher effective rank and lower off-diagonal cosine indicate a less collapsed classifier geometry.",
            "- Lower movement supports lower classifier-update cost; high-frequency ratio should be interpreted separately and is not assumed to improve in every setting.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/capacity_mechanism_probe_20260429/runs.jsonl")
    parser.add_argument("--output_dir", default="results/capacity_mechanism_probe_20260429")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(input_path)
    if not rows:
        raise SystemExit(f"No rows found in {input_path}")
    summary = summarize(rows)
    (output_dir / "summary_aggregated.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_csv(summary, output_dir / "summary_aggregated.csv")
    write_md(summary, output_dir / "summary_aggregated.md")
    print(f"Wrote summary for {len(rows)} rows to {output_dir}")


if __name__ == "__main__":
    main()
