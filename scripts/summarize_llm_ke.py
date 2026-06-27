#!/usr/bin/env python3
"""Summarize LLM KnowEdit runs into CSV and Markdown.

The runner writes one result directory per task:
  results/llm_ke/<RUN_ID>/<method>_<model>_<dataset>_<n_edits>/results.json

This script is intentionally tolerant of failed/missing tasks so that a partial
cluster run still produces a useful report for the next sweep.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def as_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def read_manifest(path: Path) -> dict[str, dict]:
    tasks: dict[str, dict] = {}
    if not path.exists():
        return tasks
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            tasks[row["tag"]] = row
    return tasks


def score_row(row: dict) -> float:
    efficacy = as_float(row.get("efficacy"))
    locality = as_float(row.get("locality"))
    if efficacy + locality <= 0:
        return 0.0
    return 2 * efficacy * locality / (efficacy + locality)


def collect(run_dir: Path, manifest_path: Path) -> list[dict]:
    manifest = read_manifest(manifest_path)
    status_dir = run_dir / "status"
    rows: list[dict] = []
    seen_tags: set[str] = set()

    for result_path in sorted(run_dir.glob("*/results.json")):
        tag = result_path.parent.name
        seen_tags.add(tag)
        with result_path.open(encoding="utf-8") as f:
            data = json.load(f)
        meta = manifest.get(tag, {})
        row = {
            "tag": tag,
            "status": "ok",
            "method": data.get("method", meta.get("method", tag.split("_", 1)[0])),
            "dataset": data.get("dataset", meta.get("dataset", "")),
            "model": data.get("model", meta.get("model", "")),
            "n_edits": data.get("n_edits", meta.get("n_edits", "")),
            "efficacy": round(as_float(data.get("efficacy")), 4),
            "locality": round(as_float(data.get("locality")), 4),
            "elapsed_s": data.get("elapsed_s", ""),
            "output": str(result_path.parent),
            "log": meta.get("log", ""),
        }
        row["score_hmean"] = round(score_row(row), 4)
        rows.append(row)

    for tag, meta in sorted(manifest.items()):
        if tag in seen_tags:
            continue
        status_file = status_dir / f"{tag}.status"
        status = status_file.read_text(encoding="utf-8").strip() if status_file.exists() else meta.get("status", "missing")
        rows.append({
            "tag": tag,
            "status": status,
            "method": meta.get("method", ""),
            "dataset": meta.get("dataset", ""),
            "model": meta.get("model", ""),
            "n_edits": meta.get("n_edits", ""),
            "efficacy": "",
            "locality": "",
            "elapsed_s": "",
            "score_hmean": "",
            "output": meta.get("output", ""),
            "log": meta.get("log", ""),
        })

    rows.sort(key=lambda r: (r["dataset"], str(r["n_edits"]), -as_float(r.get("score_hmean"))))
    return rows


def write_csv(rows: list[dict], path: Path):
    fields = ["tag", "status", "method", "dataset", "model", "n_edits", "efficacy", "locality", "score_hmean", "elapsed_s", "output", "log"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict], path: Path):
    lines = [
        "# LLM KnowEdit Summary",
        "",
        "| Dataset | Edits | Method | Status | Efficacy | Locality | H-mean | Output |",
        "|---|---:|---|---|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['n_edits']} | {row['method']} | {row['status']} | "
            f"{row['efficacy']} | {row['locality']} | {row['score_hmean']} | `{row['output']}` |"
        )

    ok_rows = [r for r in rows if r["status"] == "ok"]
    if ok_rows:
        lines.extend(["", "## Best by dataset/edit count", ""])
        groups: dict[tuple[str, str], list[dict]] = {}
        for row in ok_rows:
            groups.setdefault((str(row["dataset"]), str(row["n_edits"])), []).append(row)
        for key in sorted(groups):
            best = max(groups[key], key=score_row)
            lines.append(
                f"- `{key[0]}` edits `{key[1]}`: `{best['method']}` "
                f"eff={best['efficacy']}, loc={best['locality']}, hmean={best['score_hmean']}"
            )

    failed = [r for r in rows if r["status"] != "ok"]
    if failed:
        lines.extend(["", "## Missing or failed", ""])
        for row in failed:
            lines.append(f"- `{row['tag']}` status `{row['status']}` log `{row['log']}`")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()

    manifest = args.manifest or args.run_dir / "manifest.tsv"
    rows = collect(args.run_dir, manifest)
    write_csv(rows, args.run_dir / "summary.csv")
    write_markdown(rows, args.run_dir / "SUMMARY.md")
    print(f"Wrote {args.run_dir / 'summary.csv'}")
    print(f"Wrote {args.run_dir / 'SUMMARY.md'}")


if __name__ == "__main__":
    main()
