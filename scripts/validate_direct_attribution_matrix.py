#!/usr/bin/env python3
"""Validate the complete four-node meeting-extension experiment inventory."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from itertools import product
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ssr_utils.result_schema import validate_result_record


EXPECTED = {
    "ke_factorial": 240,
    "ke_wikibio": 20,
    "vit_lora": 20,
    "matched_kd_cl": 80,
    "cub_segmentation": 61,
    "multidataset_segmentation": 76,
}

MATCHED_KD_DATASETS = {"split_cifar100", "split_tiny_imagenet"}
MATCHED_KD_METHODS = {
    "kd",
    "kd_ewc",
    "kd_mas",
    "kd_si",
    "kd_center",
    "kd_protodecor",
    "kd_spectral",
    "kd_ssr",
}
MATCHED_KD_SEEDS = {3101, 3103, 3105, 3107, 3109}
MATCHED_KD_TEACHER_PROTOCOL = "locked_kd_only_trajectory_v1"
MATCHED_KD_REGULARIZERS = {
    "kd": "none",
    "kd_ewc": "ewc",
    "kd_mas": "mas",
    "kd_si": "si",
    "kd_center": "center",
    "kd_protodecor": "protodecor",
    "kd_spectral": "spectral",
    "kd_ssr": "ssr",
}
MATCHED_KD_REQUIRED_METRICS = {
    "avg_accuracy",
    "last_accuracy",
    "avg_forgetting",
    "backward_transfer",
    "cil_avg_accuracy",
    "cil_last_accuracy",
    "effective_rank",
    "prototype_overlap",
}
MATCHED_KD_REQUIRED_RUNTIME = {
    "elapsed_s",
    "peak_cuda_allocated_mb",
    "peak_cuda_reserved_mb",
}
EDITING_SEEDS = {
    "zsre": set(range(9311, 9330, 2)),
    "cf": set(range(9311, 9330, 2)),
    "recent": set(range(9311, 9330, 2)),
    "wikibio": set(range(9401, 9420, 2)),
}

FACTORIAL_EDITING_DATASETS = {"zsre", "cf", "recent"}
FACTORIAL_EDITING_RECIPE_MAPPINGS = {
    "plain": {"none"},
    "anchor": {"none"},
    "spectral": {"none"},
    "stabilized": {"none"},
    "ssr_only": {"cosine", "projective"},
    "full": {"cosine", "projective"},
}
FACTORIAL_EDITING_CONTRASTS = {
    "ssr_only_minus_plain",
    "full_minus_stabilized",
    "ssr_only_minus_anchor",
    "ssr_only_minus_spectral",
    "full_minus_plain",
}
FACTORIAL_EDITING_MAPPING_CONTRASTS = {
    "ssr_only:projective_minus_cosine",
    "full:projective_minus_cosine",
}


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def expected_editing_cells(
    datasets: set[str],
    recipe_mappings: dict[str, set[str]],
) -> set[tuple[str, str, str, int]]:
    return {
        (dataset, recipe, mapping, seed)
        for dataset in datasets
        for recipe, mappings in recipe_mappings.items()
        for mapping in mappings
        for seed in EDITING_SEEDS[dataset]
    }


def validate_editing_plan(
    path: Path,
    datasets: set[str],
    recipe_mappings: dict[str, set[str]],
    *,
    expected_offset: int,
    expected_n_edits: int,
) -> list[Path]:
    rows = read_tsv(path)
    required_columns = {
        "phase",
        "dataset",
        "recipe",
        "mapping",
        "seed",
        "data_offset",
        "n_edits",
        "action",
        "source",
        "output",
    }
    if not rows or not required_columns.issubset(rows[0]):
        raise ValueError(f"{path}: missing editing plan columns")
    expected_cells = expected_editing_cells(datasets, recipe_mappings)
    observed_cells: set[tuple[str, str, str, int]] = set()
    plan_root = path.parent.resolve()
    for row in rows:
        try:
            seed = int(row["seed"])
            data_offset = int(row["data_offset"])
            n_edits = int(row["n_edits"])
        except ValueError as error:
            raise ValueError(f"{path}: invalid numeric editing field in {row}") from error
        cell = (row["dataset"], row["recipe"], row["mapping"], seed)
        if cell in observed_cells:
            raise ValueError(f"{path}: duplicate editing cell {cell}")
        observed_cells.add(cell)
        if row["phase"] != "confirm":
            raise ValueError(f"{path}: editing phase must be confirm")
        if data_offset != expected_offset or n_edits != expected_n_edits:
            raise ValueError(
                f"{path}: {cell} must use offset={expected_offset}, "
                f"n_edits={expected_n_edits}"
            )
        expected_output = (
            plan_root
            / "confirm"
            / row["dataset"]
            / row["recipe"]
            / row["mapping"]
            / f"seed_{seed}"
        ).resolve()
        if Path(row["output"]).resolve() != expected_output:
            raise ValueError(
                f"{path}: output for {cell} must be {expected_output}, "
                f"found {row['output']}"
            )
        if row["action"] not in {"run", "reuse", "skip"}:
            raise ValueError(f"{path}: invalid action {row['action']!r} for {cell}")
    if observed_cells != expected_cells:
        missing = sorted(expected_cells - observed_cells)
        unexpected = sorted(observed_cells - expected_cells)
        raise ValueError(
            f"{path}: incomplete editing matrix; "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    return [Path(row["output"]) / "result_record.json" for row in rows]


def validate_vit_plan(path: Path) -> list[Path]:
    rows = read_tsv(path)
    if len(rows) != EXPECTED["vit_lora"]:
        raise ValueError(f"{path}: unexpected ViT-LoRA job count")
    keys = {(row["dataset"], row["arm"], row["seed"]) for row in rows}
    if len(keys) != EXPECTED["vit_lora"]:
        raise ValueError(f"{path}: duplicate ViT-LoRA cells")
    if {row["dataset"] for row in rows} != {"flowers102", "oxfordiiitpet"}:
        raise ValueError(f"{path}: unexpected ViT-LoRA datasets")
    if {row["arm"] for row in rows} != {"task", "task_ssr"}:
        raise ValueError(f"{path}: expected task and task_ssr arms")
    return [Path(row["output"]) / "result_record.json" for row in rows]


def validate_vit_summary(path: Path) -> None:
    if not path.is_file():
        raise ValueError(f"Missing ViT-LoRA paired summary: {path}")
    summary = load_json(path)
    if set(summary) != {"flowers102_cl", "oxfordiiitpet_cl"}:
        raise ValueError(f"{path}: unexpected ViT-LoRA summary datasets")
    for dataset, dataset_summary in summary.items():
        if dataset_summary.get("n") != 5:
            raise ValueError(f"{path}: {dataset} must contain five paired seeds")
        if dataset_summary.get("control_arm") != "task" or dataset_summary.get(
            "treatment_arm"
        ) != "task_ssr":
            raise ValueError(f"{path}: {dataset} has unexpected arms")
        metrics = dataset_summary.get("metrics", {})
        for metric in ("avg_accuracy", "avg_forgetting"):
            if metrics.get(metric, {}).get("n") != 5:
                raise ValueError(f"{path}: {dataset}/{metric} is incomplete")


def validate_editing_summary(path: Path, expected_datasets: set[str]) -> None:
    if not path.is_file():
        raise ValueError(f"Missing direct-editing paired summary: {path}")
    summary = load_json(path)
    if summary.get("control") != "plain" or summary.get("treatment") != "ssr_only":
        raise ValueError(f"{path}: expected plain versus ssr_only")
    datasets = summary.get("datasets", {})
    if set(datasets) != expected_datasets:
        raise ValueError(f"{path}: unexpected direct-editing summary datasets")
    for dataset, dataset_summary in datasets.items():
        seeds = set(dataset_summary.get("seeds", []))
        if seeds != EDITING_SEEDS[dataset]:
            raise ValueError(f"{path}: {dataset} has incomplete paired seeds")
        metrics = dataset_summary.get("metrics", {})
        for metric in (
            "efficacy",
            "locality",
            "final_history_efficacy",
            "final_history_locality",
        ):
            metric_summary = metrics.get(metric, {})
            if metric_summary.get("n") != len(EDITING_SEEDS[dataset]):
                raise ValueError(f"{path}: {dataset}/{metric} is incomplete")


def validate_factorial_editing_summary(path: Path) -> None:
    if not path.is_file():
        raise ValueError(f"Missing factorial-editing paired summary: {path}")
    summary = load_json(path)
    if summary.get("schema_version") != "ke_factorial_paired_summary_v1":
        raise ValueError(f"{path}: unexpected factorial summary schema")
    if summary.get("mode") != "factorial" or summary.get("phase") != "confirm":
        raise ValueError(f"{path}: expected confirmatory factorial summary")
    if set(summary.get("recipes", [])) != set(FACTORIAL_EDITING_RECIPE_MAPPINGS):
        raise ValueError(f"{path}: incomplete factorial recipe inventory")

    arms = summary.get("arms", {})
    contrasts = summary.get("contrasts", {})
    mapping_contrasts = summary.get("mapping_contrasts", {})
    for container_name, container in (
        ("arms", arms),
        ("contrasts", contrasts),
        ("mapping_contrasts", mapping_contrasts),
    ):
        if set(container) != FACTORIAL_EDITING_DATASETS:
            raise ValueError(f"{path}: {container_name} has unexpected datasets")

    required_metrics = {
        "efficacy",
        "locality",
        "final_history_efficacy",
        "final_history_locality",
    }
    expected_arm_keys = {
        f"{recipe}:{mapping}"
        for recipe, mappings in FACTORIAL_EDITING_RECIPE_MAPPINGS.items()
        for mapping in mappings
    }
    for dataset in sorted(FACTORIAL_EDITING_DATASETS):
        expected_seeds = EDITING_SEEDS[dataset]
        dataset_arms = arms[dataset]
        if set(dataset_arms) != expected_arm_keys:
            raise ValueError(f"{path}: {dataset} has an incomplete factorial arm inventory")
        for arm, arm_summary in dataset_arms.items():
            if set(arm_summary.get("seeds", [])) != expected_seeds:
                raise ValueError(f"{path}: {dataset}/{arm} has incomplete seeds")
            metrics = arm_summary.get("metrics", {})
            if not required_metrics.issubset(metrics):
                raise ValueError(f"{path}: {dataset}/{arm} lacks required metrics")
            for metric in required_metrics:
                if metrics[metric].get("n") != len(expected_seeds):
                    raise ValueError(f"{path}: {dataset}/{arm}/{metric} is incomplete")

        dataset_contrasts = contrasts[dataset]
        if set(dataset_contrasts) != FACTORIAL_EDITING_CONTRASTS:
            raise ValueError(f"{path}: {dataset} lacks preregistered KE contrasts")
        for contrast, mapping_summaries in dataset_contrasts.items():
            if set(mapping_summaries) != {"cosine", "projective"}:
                raise ValueError(f"{path}: {dataset}/{contrast} lacks a mapping arm")
            for mapping, contrast_summary in mapping_summaries.items():
                if set(contrast_summary.get("seeds", [])) != expected_seeds:
                    raise ValueError(
                        f"{path}: {dataset}/{contrast}/{mapping} has incomplete seeds"
                    )
                for metric in required_metrics:
                    metric_summary = contrast_summary.get("metrics", {}).get(metric, {})
                    if metric_summary.get("n") != len(expected_seeds):
                        raise ValueError(
                            f"{path}: {dataset}/{contrast}/{mapping}/{metric} is incomplete"
                        )
                    if len(metric_summary.get("pairs", [])) != len(expected_seeds):
                        raise ValueError(
                            f"{path}: {dataset}/{contrast}/{mapping}/{metric} "
                            "lacks per-seed deltas"
                        )

        dataset_mapping = mapping_contrasts[dataset]
        expected_mapping_recipes = {
            item.split(":", 1)[0] for item in FACTORIAL_EDITING_MAPPING_CONTRASTS
        }
        if set(dataset_mapping) != expected_mapping_recipes:
            raise ValueError(f"{path}: {dataset} lacks mapping contrasts")
        for recipe, mapping_summary in dataset_mapping.items():
            if set(mapping_summary.get("seeds", [])) != expected_seeds:
                raise ValueError(f"{path}: {dataset}/{recipe} mapping seeds are incomplete")
            for metric in required_metrics:
                metric_summary = mapping_summary.get("metrics", {}).get(metric, {})
                if metric_summary.get("n") != len(expected_seeds):
                    raise ValueError(
                        f"{path}: {dataset}/{recipe}/mapping/{metric} is incomplete"
                    )


def validate_segmentation_summaries(cub_path: Path, multidataset_path: Path) -> None:
    cub = load_json(cub_path)
    if set(cub) != {"direct_ssr", "matched_kd_ssr"}:
        raise ValueError(f"{cub_path}: missing direct or matched-KD contrast")
    for contrast, contrast_summary in cub.items():
        for metric in ("mean_iou", "mean_dice", "avg_forgetting_iou"):
            if contrast_summary.get("metrics", {}).get(metric, {}).get("n") != 10:
                raise ValueError(f"{cub_path}: {contrast}/{metric} is incomplete")

    multidataset = load_json(multidataset_path)
    expected_datasets = {"oxford_iiit_pet", "oxford_flowers102"}
    if set(multidataset) != expected_datasets:
        raise ValueError(f"{multidataset_path}: unexpected segmentation datasets")
    for dataset, dataset_summary in multidataset.items():
        for metric in ("mean_iou", "mean_dice", "avg_forgetting_iou"):
            if dataset_summary.get(metric, {}).get("n") != 10:
                raise ValueError(f"{multidataset_path}: {dataset}/{metric} is incomplete")


def validate_matched_kd_plan(
    path: Path,
) -> list[tuple[Path, str, str, int, Path]]:
    rows = read_tsv(path)
    required_columns = {
        "dataset",
        "method",
        "seed",
        "config",
        "teacher_root",
        "output",
    }
    if not rows or not required_columns.issubset(rows[0]):
        raise ValueError(f"{path}: missing matched-KD plan columns")

    observed: set[tuple[str, str, int]] = set()
    records: list[tuple[Path, str, str, int, Path]] = []
    output_root = path.parent.resolve()
    for row in rows:
        try:
            seed = int(row["seed"])
        except ValueError as error:
            raise ValueError(f"{path}: invalid seed {row['seed']!r}") from error
        cell = (row["dataset"], row["method"], seed)
        if cell in observed:
            raise ValueError(f"{path}: duplicate matched-KD cell {cell}")
        observed.add(cell)

        expected_output = (
            output_root / row["dataset"] / row["method"] / f"seed_{seed}"
        ).resolve()
        output = Path(row["output"]).resolve()
        if output != expected_output:
            raise ValueError(
                f"{path}: output for {cell} must be {expected_output}, found {output}"
            )
        expected_teacher = (
            output_root
            / row["dataset"]
            / "kd_teacher_trajectories"
            / f"seed_{seed}"
        ).resolve()
        teacher_root = Path(row["teacher_root"]).resolve()
        if teacher_root != expected_teacher:
            raise ValueError(
                f"{path}: teacher root for {cell} must be {expected_teacher}, "
                f"found {teacher_root}"
            )
        records.append(
            (
                output / "result_record.json",
                row["dataset"],
                row["method"],
                seed,
                teacher_root,
            )
        )

    expected_cells = set(
        product(MATCHED_KD_DATASETS, MATCHED_KD_METHODS, MATCHED_KD_SEEDS)
    )
    if observed != expected_cells:
        missing = sorted(expected_cells - observed)
        unexpected = sorted(observed - expected_cells)
        raise ValueError(
            f"{path}: matched-KD matrix must contain all {len(expected_cells)} cells; "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    return records


def is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def is_git_commit(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_matched_kd_results(
    records: list[tuple[Path, str, str, int, Path]],
    fairness_path: Path,
    summary_path: Path,
) -> None:
    pairing_groups: dict[tuple[str, int], set[str]] = {}
    group_identities: dict[tuple[str, int], set[tuple[str, ...]]] = {}
    for path, dataset, method, seed, _teacher_root in records:
        if not path.is_file():
            raise ValueError(f"Missing matched-KD result record: {path}")
        record = validate_result_record(load_json(path))
        expected_identity = {
            "task_family": "classification",
            "dataset": dataset,
            "model": "resnet18",
            "recipe": method,
            "seed": seed,
            "distance_mapping": "cosine" if method == "kd_ssr" else "none",
            "teacher_protocol": MATCHED_KD_TEACHER_PROTOCOL,
        }
        mismatches = {
            key: {"expected": value, "observed": record.get(key)}
            for key, value in expected_identity.items()
            if record.get(key) != value
        }
        if mismatches:
            raise ValueError(f"{path}: matched-KD identity mismatch {mismatches}")
        if record.get("objective", {}).get("kd") is not True:
            raise ValueError(f"{path}: matched-KD record must enable the KD objective")
        missing_metrics = MATCHED_KD_REQUIRED_METRICS - set(record.get("metrics", {}))
        if missing_metrics:
            raise ValueError(f"{path}: missing matched-KD metrics {sorted(missing_metrics)}")
        missing_runtime = MATCHED_KD_REQUIRED_RUNTIME - set(record.get("runtime", {}))
        if missing_runtime:
            raise ValueError(f"{path}: missing runtime fields {sorted(missing_runtime)}")
        for field in ("teacher_trajectory_hash", "pairing_hash", "dataset_hash"):
            if not is_sha256(record.get(field)):
                raise ValueError(f"{path}: {field} must be a lowercase SHA256")
        if not is_sha256(record.get("initial_model_hash")):
            raise ValueError(f"{path}: initial_model_hash must be a lowercase SHA256")
        if not isinstance(record.get("optimizer_steps"), int) or record["optimizer_steps"] <= 0:
            raise ValueError(f"{path}: optimizer_steps must be a positive integer")
        if not isinstance(record.get("training_batches"), int) or record["training_batches"] <= 0:
            raise ValueError(f"{path}: training_batches must be a positive integer")
        if record["optimizer_steps"] > record["training_batches"]:
            raise ValueError(f"{path}: optimizer_steps cannot exceed training_batches")
        expected_regularizer = MATCHED_KD_REGULARIZERS[method]
        if record.get("active_regularizer") != expected_regularizer:
            raise ValueError(f"{path}: active_regularizer must be {expected_regularizer}")
        regularizer_weight = record.get("active_regularizer_weight")
        if not isinstance(regularizer_weight, (int, float)):
            raise ValueError(f"{path}: active_regularizer_weight must be numeric")
        if (method == "kd" and float(regularizer_weight) != 0.0) or (
            method != "kd" and float(regularizer_weight) <= 0.0
        ):
            raise ValueError(f"{path}: active regularizer weight is inconsistent")
        for field in ("kd_weight", "kd_temperature"):
            if not isinstance(record.get(field), (int, float)) or float(record[field]) <= 0:
                raise ValueError(f"{path}: {field} must be positive")
        if not is_sha256(record.get("auxiliary_state_hash")):
            raise ValueError(f"{path}: auxiliary_state_hash must be a lowercase SHA256")
        auxiliary_count = record.get("auxiliary_trainable_parameter_count")
        trainable_count = record.get("trainable_parameter_count")
        if not isinstance(auxiliary_count, int) or auxiliary_count < 0:
            raise ValueError(f"{path}: invalid auxiliary parameter count")
        if not isinstance(trainable_count, int) or trainable_count <= auxiliary_count:
            raise ValueError(f"{path}: invalid total trainable parameter count")
        if (method == "kd_center") != (auxiliary_count > 0):
            raise ValueError(f"{path}: auxiliary parameters are inconsistent with recipe")
        group = (dataset, seed)
        pairing_groups.setdefault(group, set()).add(method)
        group_identities.setdefault(group, set()).add(
            (
                record["pairing_hash"],
                record["teacher_trajectory_hash"],
                record["dataset_hash"],
                record["initial_model_hash"],
                str(record["optimizer_steps"]),
                str(record["training_batches"]),
                str(record["kd_weight"]),
                str(record["kd_temperature"]),
            )
        )

    if len(pairing_groups) != len(MATCHED_KD_DATASETS) * len(MATCHED_KD_SEEDS):
        raise ValueError("Matched-KD results do not contain exactly ten paired groups")
    incomplete = {
        key: sorted(MATCHED_KD_METHODS - methods)
        for key, methods in pairing_groups.items()
        if methods != MATCHED_KD_METHODS
    }
    if incomplete:
        raise ValueError(f"Incomplete matched-KD result groups: {incomplete}")
    mismatched_groups = {
        group: sorted(identities)
        for group, identities in group_identities.items()
        if len(identities) != 1
    }
    if mismatched_groups:
        raise ValueError(
            "Matched-KD scaffold, teacher, or dataset identity differs within groups: "
            f"{mismatched_groups}"
        )

    if not fairness_path.is_file():
        raise ValueError(f"Missing matched-KD fairness report: {fairness_path}")
    fairness = load_json(fairness_path)
    expected_fairness = {
        "status": "PASS",
        "group_count": 10,
        "expected_recipes": sorted(MATCHED_KD_METHODS),
        "errors": [],
    }
    mismatches = {
        key: {"expected": value, "observed": fairness.get(key)}
        for key, value in expected_fairness.items()
        if fairness.get(key) != value
    }
    if Path(str(fairness.get("root", ""))).resolve() != fairness_path.parent.resolve():
        mismatches["root"] = {
            "expected": str(fairness_path.parent.resolve()),
            "observed": fairness.get("root"),
        }
    if mismatches:
        raise ValueError(f"{fairness_path}: invalid fairness report {mismatches}")

    if not summary_path.is_file():
        raise ValueError(f"Missing matched-KD paired summary: {summary_path}")
    summary = load_json(summary_path)
    if summary.get("control") != "kd":
        raise ValueError(f"{summary_path}: paired-summary control must be kd")
    datasets = summary.get("datasets", {})
    if set(datasets) != MATCHED_KD_DATASETS:
        raise ValueError(f"{summary_path}: unexpected paired-summary datasets")
    ssr_vs_controls = summary.get("ssr_vs_controls", {})
    if set(ssr_vs_controls) != MATCHED_KD_DATASETS:
        raise ValueError(f"{summary_path}: missing SSR-versus-control summaries")
    treatments = MATCHED_KD_METHODS - {"kd"}
    ssr_controls = MATCHED_KD_METHODS - {"kd", "kd_ssr"}
    for dataset, dataset_summary in datasets.items():
        if set(dataset_summary) != treatments:
            raise ValueError(
                f"{summary_path}: {dataset} must summarize {sorted(treatments)}"
            )
        for treatment, treatment_summary in dataset_summary.items():
            if set(treatment_summary.get("seeds", [])) != MATCHED_KD_SEEDS:
                raise ValueError(
                    f"{summary_path}: {dataset}/{treatment} has incomplete paired seeds"
                )
            metrics = treatment_summary.get("metrics", {})
            for metric in MATCHED_KD_REQUIRED_METRICS:
                metric_summary = metrics.get(metric, {})
                if metric_summary.get("n") != len(MATCHED_KD_SEEDS):
                    raise ValueError(
                        f"{summary_path}: {dataset}/{treatment}/{metric} is incomplete"
                    )
                favorable_pairs = metric_summary.get("favorable_pairs")
                if not isinstance(favorable_pairs, int) or not 0 <= favorable_pairs <= 5:
                    raise ValueError(
                        f"{summary_path}: invalid favorable-pair count for "
                        f"{dataset}/{treatment}/{metric}"
                    )
        direct_summary = ssr_vs_controls[dataset]
        if set(direct_summary) != ssr_controls:
            raise ValueError(
                f"{summary_path}: {dataset} must directly compare SSR with "
                f"{sorted(ssr_controls)}"
            )
        for control, comparison in direct_summary.items():
            if comparison.get("control") != control or comparison.get(
                "treatment"
            ) != "kd_ssr":
                raise ValueError(
                    f"{summary_path}: invalid direct SSR comparison for {dataset}/{control}"
                )
            if set(comparison.get("seeds", [])) != MATCHED_KD_SEEDS:
                raise ValueError(
                    f"{summary_path}: {dataset}/{control} has incomplete direct pairs"
                )
            for metric in MATCHED_KD_REQUIRED_METRICS:
                metric_summary = comparison.get("metrics", {}).get(metric, {})
                if metric_summary.get("n") != len(MATCHED_KD_SEEDS):
                    raise ValueError(
                        f"{summary_path}: {dataset}/SSR-vs-{control}/{metric} is incomplete"
                    )


def count_manifest_jobs(path: Path) -> int:
    jobs = load_json(path).get("jobs", {})
    return int(jobs.get("development", 0)) + int(jobs.get("confirmation", 0))


def count_records(root: Path) -> int:
    return sum(1 for _ in root.rglob("result_record.json"))


def read_launcher_identity(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise ValueError(f"Missing launcher identity: {path}")
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            raise ValueError(f"Malformed launcher identity line: {line!r}")
        key, value = line.split("=", 1)
        if not key or key in values:
            raise ValueError(f"Malformed launcher identity key: {key!r}")
        values[key] = value
    return values


def validate_component_git_commits(
    paths: list[Path],
    *,
    component: str,
    allowed_commits: set[str],
) -> dict[str, int]:
    if not allowed_commits or any(not is_git_commit(commit) for commit in allowed_commits):
        raise ValueError(f"{component}: invalid allowed git commit set")
    counts: dict[str, int] = {}
    for path in paths:
        record = validate_result_record(load_json(path))
        commit = record["git_commit"]
        if commit not in allowed_commits:
            raise ValueError(
                f"{component}: disallowed git commit {commit} in {path}; "
                f"allowed={sorted(allowed_commits)}"
            )
        counts[commit] = counts.get(commit, 0) + 1
    return dict(sorted(counts.items()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--mode", choices=("plan", "results"), default="plan")
    args = parser.parse_args()
    root = args.root.resolve()

    factorial_records = validate_editing_plan(
        root / "ke_factorial" / "planned_runs.tsv",
        FACTORIAL_EDITING_DATASETS,
        FACTORIAL_EDITING_RECIPE_MAPPINGS,
        expected_offset=500,
        expected_n_edits=100,
    )
    wikibio_records = validate_editing_plan(
        root / "ke_wikibio" / "planned_runs.tsv",
        {"wikibio"},
        {"plain": {"none"}, "ssr_only": {"projective"}},
        expected_offset=0,
        expected_n_edits=200,
    )
    vit_records = validate_vit_plan(root / "vit_lora" / "planned_runs.tsv")
    expected_records = factorial_records + wikibio_records + vit_records
    matched_kd_records = validate_matched_kd_plan(
        root / "matched_kd_cl" / "planned_runs.tsv"
    )

    observed = {
        "ke_factorial": EXPECTED["ke_factorial"],
        "ke_wikibio": EXPECTED["ke_wikibio"],
        "vit_lora": EXPECTED["vit_lora"],
        "matched_kd_cl": len(matched_kd_records),
        "cub_segmentation": count_manifest_jobs(
            root / "cub_segmentation" / "MANIFEST.json"
        ),
        "multidataset_segmentation": count_manifest_jobs(
            root / "multidataset_segmentation" / "MANIFEST.json"
        ),
    }
    if observed != EXPECTED:
        raise ValueError(f"Matrix mismatch: expected={EXPECTED}, observed={observed}")
    total_jobs = sum(observed.values())
    if total_jobs != 497:
        raise ValueError(f"Expected 497 total jobs, found {total_jobs}")

    provenance = None
    if args.mode == "results":
        missing = [str(path) for path in expected_records if not path.is_file()]
        if missing:
            raise ValueError(f"Missing {len(missing)} editing/LoRA result records: {missing[:5]}")
        validate_matched_kd_results(
            matched_kd_records,
            root / "matched_kd_cl" / "fairness_report.json",
            root / "matched_kd_cl" / "paired_summary" / "summary.json",
        )
        validate_factorial_editing_summary(
            root / "ke_factorial" / "paired_summary" / "summary.json"
        )
        validate_editing_summary(
            root / "ke_wikibio" / "paired_summary" / "summary.json",
            {"wikibio"},
        )
        validate_vit_summary(root / "vit_lora" / "paired_summary" / "summary.json")
        for name in ("cub_segmentation", "multidataset_segmentation"):
            count = count_records(root / name)
            if count != EXPECTED[name]:
                raise ValueError(
                    f"{name}: expected {EXPECTED[name]} result records, found {count}"
                )
        identity = read_launcher_identity(root / ".direct_attribution_4node_identity")
        current_commit = identity.get("git_commit", "")
        recovery_commit = identity.get("recovery_source_commit", "")
        reusable_commits = {current_commit}
        if recovery_commit:
            reusable_commits.add(recovery_commit)
        strict_current = {current_commit}
        cub_records = sorted((root / "cub_segmentation").rglob("result_record.json"))
        multidataset_records = sorted(
            (root / "multidataset_segmentation").rglob("result_record.json")
        )
        provenance = {
            "ke_factorial": validate_component_git_commits(
                factorial_records,
                component="ke_factorial",
                allowed_commits=reusable_commits,
            ),
            "ke_wikibio": validate_component_git_commits(
                wikibio_records,
                component="ke_wikibio",
                allowed_commits=reusable_commits,
            ),
            "vit_lora": validate_component_git_commits(
                vit_records,
                component="vit_lora",
                allowed_commits=reusable_commits,
            ),
            "matched_kd_cl": validate_component_git_commits(
                [record[0] for record in matched_kd_records],
                component="matched_kd_cl",
                allowed_commits=strict_current,
            ),
            "cub_segmentation": validate_component_git_commits(
                cub_records,
                component="cub_segmentation",
                allowed_commits=strict_current,
            ),
            "multidataset_segmentation": validate_component_git_commits(
                multidataset_records,
                component="multidataset_segmentation",
                allowed_commits=strict_current,
            ),
        }
        required = (
            root / "cub_segmentation" / "CONFIRMATION_SUMMARY.json",
            root / "multidataset_segmentation" / "CONFIRMATION_SUMMARY.json",
        )
        missing_summaries = [str(path) for path in required if not path.is_file()]
        if missing_summaries:
            raise ValueError(f"Missing confirmation summaries: {missing_summaries}")
        validate_segmentation_summaries(*required)

    report = {
        "status": "PASS",
        "mode": args.mode,
        "jobs": observed,
        "total_jobs": total_jobs,
        "matched_kd_contract": {
            "planned_cells": len(matched_kd_records),
            "result_records_required": EXPECTED["matched_kd_cl"],
            "fairness_report_required": True,
            "paired_summary_required": True,
        },
        "direct_editing_contract": {
            "result_records_required": EXPECTED["ke_factorial"] + EXPECTED["ke_wikibio"],
            "paired_summaries_required": 2,
            "factorial_cells": EXPECTED["ke_factorial"],
            "external_transfer_cells": EXPECTED["ke_wikibio"],
        },
        "primary_comparisons": [
            "task loss plus SSR versus task loss only",
            "full SSR recipe versus the anchor-plus-spectral stabilized baseline",
            "SSR-only versus anchor-only and spectral-only",
            "projective versus cosine distance mapping within the same SSR recipe",
            "KD plus EWC, MAS, SI, center, prototype decorrelation, spectral, or SSR versus the same KD scaffold",
        ],
    }
    if provenance is not None:
        report["git_provenance"] = provenance
    output = root / f"MATRIX_{args.mode.upper()}_VALIDATION.json"
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
