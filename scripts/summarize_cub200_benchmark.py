#!/usr/bin/env python3
"""Summarize CUB200 continual classification/segmentation benchmark outputs."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_rows(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def summarize(rows: list[dict]) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["task"], row["method"])].append(row)
    out = {}
    for (task, method), items in sorted(grouped.items()):
        out.setdefault(task, {})[method] = {}
        for key in items[0]:
            if key in {"task", "method", "seed"}:
                continue
            vals = np.asarray([float(item[key]) for item in items], dtype=float)
            out[task][method][key] = {
                "mean": float(vals.mean()),
                "std": float(vals.std()),
                "n": int(vals.size),
            }
    return out


def fmt(stat: dict, digits: int = 2) -> str:
    return f"{stat['mean']:.{digits}f} ± {stat['std']:.{digits}f}"


def write_md(summary: dict, output: Path) -> None:
    lines = ["# CUB200 Continual Benchmark Summary", ""]
    if "classification" in summary:
        lines.extend(
            [
                "## Classification",
                "",
                "| Method | TIL AA | CIL AA curve | Final CIL | AF | Eff. rank | |offdiag cos| | n |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for method, m in summary["classification"].items():
            extra = ""
            if "til_capacity_at_80" in m:
                extra = (
                    f"; Cap@80/75/70="
                    f"{m['til_capacity_at_80']['mean']:.0f}/"
                    f"{m['til_capacity_at_75']['mean']:.0f}/"
                    f"{m['til_capacity_at_70']['mean']:.0f}"
                )
            lines.append(
                "| "
                + " | ".join(
                    [
                        method,
                        fmt(m["avg_accuracy"]) + extra,
                        fmt(m["avg_cil_accuracy"]),
                        fmt(m["final_cil_accuracy_seen"]),
                        fmt(m["avg_forgetting"]),
                        fmt(m["effective_rank"]),
                        fmt(m["mean_abs_offdiag_cosine"], 3),
                        str(m["avg_accuracy"]["n"]),
                    ]
                )
                + " |"
            )
        lines.append("")
    if "segmentation" in summary:
        lines.extend(
            [
                "## Segmentation",
                "",
                "| Method | mIoU | Dice | IoU forgetting | Eff. rank | |offdiag cos| | n |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for method, m in summary["segmentation"].items():
            lines.append(
                "| "
                + " | ".join(
                    [
                        method,
                        fmt(m["mean_iou"]),
                        fmt(m["mean_dice"]),
                        fmt(m["avg_forgetting_iou"]),
                        fmt(m["effective_rank"]),
                        fmt(m["mean_abs_offdiag_cosine"], 3),
                        str(m["mean_iou"]["n"]),
                    ]
                )
                + " |"
            )
        lines.append("")
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--classification", default="results/cub200_classification_til_20260429/runs.jsonl")
    p.add_argument("--segmentation", default="results/cub200_segmentation_20260429/runs.jsonl")
    p.add_argument("--output_dir", default="results/cub200_summary_20260429")
    args = p.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = load_rows([Path(args.classification), Path(args.segmentation)])
    if not rows:
        raise SystemExit("No CUB200 rows found.")
    summary = summarize(rows)
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_md(summary, outdir / "summary.md")
    print(f"Wrote {len(rows)} rows to {outdir}")


if __name__ == "__main__":
    main()
