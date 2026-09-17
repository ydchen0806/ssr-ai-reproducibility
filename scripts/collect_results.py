#!/usr/bin/env python3
"""Collect SSR continual-learning summary.json files into a compact table."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


FIELDS = [
    "avg_accuracy",
    "last_accuracy",
    "avg_forgetting",
    "backward_transfer",
    "total_time_s",
]


def sem(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return stdev(values) / math.sqrt(len(values))


def load_summaries(root: Path) -> list[dict]:
    rows = []
    for path in sorted(root.rglob("summary.json")):
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        payload["_path"] = str(path)
        rows.append(payload)
    return rows


def group_key(row: dict) -> tuple[str, str, str, str]:
    return (
        str(row.get("dataset", "unknown")),
        str(row.get("method", "unknown")),
        str(row.get("experiment_name", "unknown")),
        str(row.get("model", "unknown")),
    )


def summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[group_key(row)].append(row)

    out = []
    for (dataset, method, experiment_name, model), items in sorted(grouped.items()):
        summary = {
            "dataset": dataset,
            "method": method,
            "experiment_name": experiment_name,
            "model": model,
            "n": len(items),
            "seeds": ",".join(str(item.get("seed", "?")) for item in items),
        }
        for field in FIELDS:
            values = [float(item[field]) for item in items if field in item and item[field] is not None]
            if values:
                summary[f"{field}_mean"] = mean(values)
                summary[f"{field}_sem"] = sem(values)
        out.append(summary)
    return out


def print_table(rows: list[dict]) -> None:
    if not rows:
        print("No summary.json files found.")
        return
    print(
        "dataset\tmethod\texperiment\tn\tavg_acc_mean\tavg_acc_sem\t"
        "avg_forgetting_mean\tavg_forgetting_sem\ttime_s_mean"
    )
    for row in rows:
        print(
            f"{row['dataset']}\t{row['method']}\t{row['experiment_name']}\t{row['n']}\t"
            f"{row.get('avg_accuracy_mean', float('nan')):.3f}\t"
            f"{row.get('avg_accuracy_sem', float('nan')):.3f}\t"
            f"{row.get('avg_forgetting_mean', float('nan')):.3f}\t"
            f"{row.get('avg_forgetting_sem', float('nan')):.3f}\t"
            f"{row.get('total_time_s_mean', float('nan')):.1f}"
        )


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results", help="Root directory containing summary.json files.")
    parser.add_argument("--csv", default=None, help="Optional CSV output path.")
    args = parser.parse_args()

    rows = summarize(load_summaries(Path(args.results)))
    print_table(rows)
    if args.csv:
        write_csv(rows, Path(args.csv))


if __name__ == "__main__":
    main()

