#!/usr/bin/env python3
"""Import the locked 2026-08-03 editing validation runs without rewriting them."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ssr_utils.result_schema import build_result_record


DATASET_HASHES = {
    "zsre": "0e0214dda853a906ef02bef58c7ce2978af5a962e91fc4a10827e0125459905c",
    "cf": "efc01ba398dfe04975939f729aab45523addb3c6b3cf41d7429ba5de1d45225b",
    "recent": "e5c4ce6de1e8e2d10d892164ad6f4cbd28c976ed509e29e78ae853b01de1797b",
}
CONDITIONS = {
    "true_ft_control": ("plain", {"task": True}),
    "anchor_spectral_control": (
        "stabilized",
        {"task": True, "anchor": True, "spectral": True},
    ),
    "ssr_only_": ("ssr_only", {"task": True, "ssr": True}),
    "full_": (
        "full",
        {"task": True, "ssr": True, "anchor": True, "spectral": True},
    ),
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def identify_condition(directory_name: str) -> tuple[str, dict[str, bool]]:
    for marker, condition in CONDITIONS.items():
        if marker in directory_name:
            return condition
    raise ValueError(f"Unrecognized decisive editing condition: {directory_name}")


def import_one(
    raw_path: Path,
    *,
    raw_root: Path,
    output_root: Path,
    source_fingerprint: str,
) -> Path:
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    dataset = raw_path.relative_to(raw_root).parts[0]
    if dataset not in DATASET_HASHES:
        raise ValueError(f"Unknown editing dataset directory: {dataset}")
    recipe, objective = identify_condition(raw_path.parent.name)
    seed_match = re.search(r"seed(\d+)", raw_path.parent.name)
    seed = int(raw.get("seed") or (seed_match.group(1) if seed_match else -1))
    if seed < 0:
        raise ValueError(f"Cannot recover seed from {raw_path}")

    requested = int(raw.get("requested_n_edits", raw.get("n_edits", 0)))
    attempted = int(raw.get("n_edits", len(raw.get("results", []))))
    succeeded = len(raw.get("results", []))
    failed = max(attempted - succeeded, 0)
    status = "complete" if requested == attempted == succeeded and failed == 0 else (
        "failed" if attempted > 0 and succeeded == 0 else "incomplete"
    )
    if status != "complete":
        raise ValueError(
            f"Legacy validation run is not complete: {raw_path} "
            f"(requested={requested}, attempted={attempted}, succeeded={succeeded})"
        )

    uses_ssr = bool(objective.get("ssr"))
    mapping = str(raw.get("distance_metric", "none")) if uses_ssr else "none"
    kernel = {}
    if uses_ssr:
        kernel = {
            "family": str(raw["kernel_family"]),
            "A_exc": float(raw["a_exc"]),
            "A_inh": float(raw["a_inh"]),
            "sigma_exc": float(raw["sigma_exc"]),
            "sigma_inh": float(raw["sigma_inh"]),
        }

    metrics = {
        "efficacy": float(raw["efficacy"]),
        "locality": float(raw["locality"]),
    }
    final = raw.get("historical_retention", {}).get("final")
    if isinstance(final, dict):
        metrics["final_history_efficacy"] = float(final["efficacy"])
        metrics["final_history_locality"] = float(final["locality"])

    raw_config = {
        key: raw.get(key)
        for key in (
            "run_label",
            "objective_name",
            "geometry_regularizer",
            "lambda_geometry",
            "lambda_anchor",
            "lambda_spectral",
            "kernel_family",
            "a_exc",
            "a_inh",
            "sigma_exc",
            "sigma_inh",
            "biocs_target",
            "distance_metric",
            "row_selection",
            "max_spatial_rows",
            "target_layers",
            "lr",
            "num_steps",
            "data_offset",
        )
    }
    raw_config["legacy_source_fingerprint"] = source_fingerprint
    raw_config["legacy_results_sha256"] = file_sha256(raw_path)

    record = build_result_record(
        git_commit=f"legacy-source-fingerprint:{source_fingerprint}",
        run_id=f"legacy:{raw_path.relative_to(raw_root)}",
        task_family="editing",
        dataset=dataset,
        model="gpt2-xl",
        seed=seed,
        objective=objective,
        distance_mapping=mapping,
        kernel=kernel,
        metrics=metrics,
        runtime={
            "elapsed_s": float(raw.get("elapsed_s", 0.0)),
            "peak_cuda_allocated_mb": float(raw.get("peak_cuda_allocated_mb", 0.0)),
            "peak_cuda_reserved_mb": float(raw.get("peak_cuda_reserved_mb", 0.0)),
        },
        config=raw_config,
        dataset_hash=DATASET_HASHES[dataset],
        recipe=recipe,
        n_edits=attempted,
        data_offset=int(raw.get("data_offset", 0)),
        requested=requested,
        attempted=attempted,
        succeeded=succeeded,
        failed=failed,
        status=status,
        evaluator="legacy_custom_substring_any_ground_truth_all_items",
        evaluator_version="2026-08-03-locked",
        metric_directions={metric: True for metric in metrics},
        imported_from=str(raw_path),
        imported_results_sha256=raw_config["legacy_results_sha256"],
    )
    output = output_root / dataset / recipe / mapping / f"seed_{seed}" / "result_record.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    shutil.copy2(raw_path, output.parent / "results.json")
    condition_path = raw_path.with_name("condition.json")
    if condition_path.is_file():
        shutil.copy2(condition_path, output.parent / "condition.json")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-fingerprint-file", type=Path, required=True)
    args = parser.parse_args()

    raw_root = args.raw_root.resolve()
    fingerprint = file_sha256(args.source_fingerprint_file.resolve())
    paths = sorted(raw_root.rglob("results.json"))
    if not paths:
        raise SystemExit(f"No legacy results.json files found under {raw_root}")
    imported = [
        import_one(
            path,
            raw_root=raw_root,
            output_root=args.output_root.resolve(),
            source_fingerprint=fingerprint,
        )
        for path in paths
    ]
    print(f"Imported {len(imported)} complete locked editing records")


if __name__ == "__main__":
    main()
