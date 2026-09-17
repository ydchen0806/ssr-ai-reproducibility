#!/usr/bin/env python3
"""Build strict paired summaries for the knowledge-editing factorial."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ssr_utils.paired_stats import paired_summary
from ssr_utils.result_schema import sha256_value, validate_result_record


RECIPES = ("plain", "anchor", "spectral", "stabilized", "ssr_only", "full")
DIRECT_RECIPES = {"plain", "ssr_only"}
MAPPING_DEPENDENT_RECIPES = {"ssr_only", "full"}
MAPPINGS = ("projective", "cosine")
RECIPE_CONTRASTS = (
    ("ssr_only_minus_plain", "plain", "ssr_only"),
    ("full_minus_stabilized", "stabilized", "full"),
    ("ssr_only_minus_anchor", "anchor", "ssr_only"),
    ("ssr_only_minus_spectral", "spectral", "ssr_only"),
    ("full_minus_plain", "plain", "full"),
)
PAIR_FIELDS = (
    "dataset",
    "model",
    "seed",
    "data_offset",
    "n_edits",
    "requested",
    "attempted",
    "succeeded",
    "failed",
    "status",
    "dataset_hash",
    "record_cohort_hash",
    "evaluator",
    "evaluator_version",
    "evaluation_protocol_hash",
    "model_hash",
    "pairing_protocol_hash",
)
PLAN_FIELDS = {
    "phase",
    "dataset",
    "recipe",
    "mapping",
    "seed",
    "data_offset",
    "n_edits",
    "output",
}


def read_plan(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise ValueError(f"Empty editing plan: {path}")
    missing = PLAN_FIELDS - set(rows[0])
    if missing:
        raise ValueError(f"Editing plan is missing columns: {sorted(missing)}")
    return rows


def _int_field(row: dict[str, str], field: str) -> int:
    try:
        return int(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid plan field {field}={row.get(field)!r}") from error


def _arm_mapping(recipe: str, mapping: str) -> str:
    if recipe in MAPPING_DEPENDENT_RECIPES:
        if mapping not in MAPPINGS:
            raise ValueError(f"{recipe} requires projective or cosine mapping, got {mapping}")
        return mapping
    if mapping != "none":
        raise ValueError(f"Mapping-independent recipe {recipe} must use mapping=none")
    return "none"


def _arm_key(recipe: str, mapping: str) -> str:
    return f"{recipe}:{mapping}"


def _record_key(dataset: str, recipe: str, mapping: str, seed: int) -> tuple:
    return dataset, recipe, mapping, seed


def _identity_mismatches(reference: dict, candidate: dict) -> dict[str, dict]:
    return {
        field: {"reference": reference.get(field), "candidate": candidate.get(field)}
        for field in PAIR_FIELDS
        if reference.get(field) != candidate.get(field)
    }


def _record_cohort_hash(path: Path, *, data_offset: int, n_edits: int) -> str:
    results_path = path / "results.json"
    if not results_path.is_file():
        raise FileNotFoundError(results_path)
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    items = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(items, list) or len(items) != n_edits:
        observed = len(items) if isinstance(items, list) else None
        raise ValueError(
            f"{results_path}: expected {n_edits} per-edit records, found {observed}"
        )
    identities = []
    expected_indices = list(range(data_offset, data_offset + n_edits))
    for position, (item, expected_index) in enumerate(zip(items, expected_indices)):
        if not isinstance(item, dict):
            raise ValueError(f"{results_path}: item {position} is not an object")
        if item.get("idx") != expected_index:
            raise ValueError(
                f"{results_path}: item {position} has idx={item.get('idx')!r}, "
                f"expected {expected_index}"
            )
        prompt = item.get("prompt")
        target = item.get("target_new")
        if not isinstance(prompt, str) or not isinstance(target, str):
            raise ValueError(
                f"{results_path}: item {position} lacks string prompt/target_new"
            )
        identities.append(
            {"idx": expected_index, "prompt": prompt, "target_new": target}
        )
    return sha256_value(identities)


def load_pairs(rows: list[dict[str, str]]) -> dict[str, Any]:
    """Load and audit a direct or full-factorial editing result matrix."""
    phases = {row["phase"] for row in rows}
    if len(phases) != 1:
        raise ValueError(f"A summary must contain exactly one phase, found {sorted(phases)}")

    records: dict[tuple, dict] = {}
    for row in rows:
        recipe = row["recipe"]
        if recipe not in RECIPES:
            raise ValueError(f"Unexpected editing recipe: {recipe}")
        mapping = _arm_mapping(recipe, row["mapping"])
        seed = _int_field(row, "seed")
        data_offset = _int_field(row, "data_offset")
        n_edits = _int_field(row, "n_edits")
        path = Path(row["output"]) / "result_record.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        record = validate_result_record(json.loads(path.read_text(encoding="utf-8")))
        expected_identity = {
            "task_family": "editing",
            "dataset": row["dataset"],
            "recipe": recipe,
            "seed": seed,
            "distance_mapping": mapping,
            "data_offset": data_offset,
            "n_edits": n_edits,
        }
        mismatches = {
            field: {"plan": expected, "record": record.get(field)}
            for field, expected in expected_identity.items()
            if record.get(field) != expected
        }
        if mismatches:
            raise ValueError(f"Result identity mismatch for {path}: {mismatches}")
        if int(record["requested"]) != n_edits:
            raise ValueError(
                f"{path}: requested={record['requested']} does not match n_edits={n_edits}"
            )
        record["record_cohort_hash"] = _record_cohort_hash(
            path.parent,
            data_offset=data_offset,
            n_edits=n_edits,
        )
        key = _record_key(row["dataset"], recipe, mapping, seed)
        if key in records:
            raise ValueError(f"Duplicate editing cell: {key}")
        records[key] = record

    observed_recipes = {key[1] for key in records}
    if observed_recipes == DIRECT_RECIPES:
        mode = "direct"
    elif observed_recipes == set(RECIPES):
        mode = "factorial"
    else:
        raise ValueError(
            "Editing recipe inventory must be either plain+ssr_only or the complete "
            f"six-recipe factorial; found {sorted(observed_recipes)}"
        )

    recipe_mappings = {
        recipe: {key[2] for key in records if key[1] == recipe}
        for recipe in observed_recipes
    }
    for recipe, mappings in recipe_mappings.items():
        expected = {"none"} if recipe not in MAPPING_DEPENDENT_RECIPES else set(mappings)
        if mappings != expected or not mappings:
            raise ValueError(f"Invalid mapping inventory for {recipe}: {sorted(mappings)}")
    if mode == "factorial":
        for recipe in MAPPING_DEPENDENT_RECIPES:
            if recipe_mappings[recipe] != set(MAPPINGS):
                raise ValueError(
                    f"Full factorial requires projective and cosine for {recipe}; "
                    f"found {sorted(recipe_mappings[recipe])}"
                )

    datasets = sorted({key[0] for key in records})
    seeds_by_dataset: dict[str, list[int]] = {}
    for dataset in datasets:
        seeds = sorted(
            key[3]
            for key in records
            if key[0] == dataset and key[1] == "plain" and key[2] == "none"
        )
        if len(seeds) < 2:
            raise ValueError(f"{dataset}: at least two plain-control seeds are required")
        expected_seeds = set(seeds)
        for recipe in sorted(observed_recipes):
            for mapping in sorted(recipe_mappings[recipe]):
                arm_seeds = {
                    key[3]
                    for key in records
                    if key[0] == dataset and key[1:3] == (recipe, mapping)
                }
                if arm_seeds != expected_seeds:
                    raise ValueError(
                        f"Incomplete arm {dataset}/{recipe}/{mapping}: "
                        f"expected={sorted(expected_seeds)}, observed={sorted(arm_seeds)}"
                    )
        for seed in seeds:
            reference = records[_record_key(dataset, "plain", "none", seed)]
            for recipe in sorted(observed_recipes):
                for mapping in sorted(recipe_mappings[recipe]):
                    candidate = records[_record_key(dataset, recipe, mapping, seed)]
                    mismatches = _identity_mismatches(reference, candidate)
                    if mismatches:
                        raise ValueError(
                            f"Unmatched same-record conditions for {dataset}/seed_{seed}/"
                            f"{recipe}/{mapping}: {mismatches}"
                        )
        seeds_by_dataset[dataset] = seeds

    return {
        "mode": mode,
        "phase": next(iter(phases)),
        "records": records,
        "recipes": tuple(recipe for recipe in RECIPES if recipe in observed_recipes),
        "recipe_mappings": recipe_mappings,
        "datasets": datasets,
        "seeds": seeds_by_dataset,
    }


def _records_for_arm(
    matrix: dict[str, Any], dataset: str, recipe: str, mapping: str
) -> list[dict]:
    return [
        matrix["records"][_record_key(dataset, recipe, mapping, seed)]
        for seed in matrix["seeds"][dataset]
    ]


def _metric_inventory(records: Iterable[dict], label: str) -> dict[str, bool]:
    records = list(records)
    metric_sets = [set(record["metrics"]) for record in records]
    if any(metrics != metric_sets[0] for metrics in metric_sets[1:]):
        raise ValueError(f"{label}: metric inventory differs across paired records")
    directions: dict[str, bool] = {}
    for metric in sorted(metric_sets[0]):
        values = []
        for record in records:
            declared = record.get("metric_directions", {})
            if metric not in declared:
                raise ValueError(f"{label}/{metric}: missing metric direction")
            values.append(bool(declared[metric]))
            value = float(record["metrics"][metric])
            if not math.isfinite(value):
                raise ValueError(f"{label}/{metric}: metric value is not finite")
        if len(set(values)) != 1:
            raise ValueError(f"{label}/{metric}: inconsistent metric direction")
        directions[metric] = values[0]
    return directions


def _arm_summary(records: list[dict], label: str) -> dict:
    directions = _metric_inventory(records, label)
    metrics = {}
    for metric, higher_is_better in directions.items():
        values = [float(record["metrics"][metric]) for record in records]
        metrics[metric] = {
            "higher_is_better": higher_is_better,
            "n": len(values),
            "mean": statistics.fmean(values),
            "sd": statistics.stdev(values),
        }
    return {
        "seeds": [int(record["seed"]) for record in records],
        "metrics": metrics,
    }


def _contrast_summary(
    controls: list[dict],
    treatments: list[dict],
    *,
    label: str,
) -> dict:
    if [record["seed"] for record in controls] != [
        record["seed"] for record in treatments
    ]:
        raise ValueError(f"{label}: paired seeds differ")
    control_directions = _metric_inventory(controls, f"{label}/control")
    treatment_directions = _metric_inventory(treatments, f"{label}/treatment")
    if control_directions != treatment_directions:
        raise ValueError(f"{label}: paired metric inventory or directions differ")

    metrics = {}
    for metric, higher_is_better in control_directions.items():
        control_values = [float(record["metrics"][metric]) for record in controls]
        treatment_values = [float(record["metrics"][metric]) for record in treatments]
        summary = paired_summary(
            control_values,
            treatment_values,
            higher_is_better=higher_is_better,
        ).as_dict()
        pair_rows = []
        for control, treatment, control_value, treatment_value in zip(
            controls, treatments, control_values, treatment_values
        ):
            raw_difference = treatment_value - control_value
            pair_rows.append(
                {
                    "seed": int(control["seed"]),
                    "data_offset": int(control["data_offset"]),
                    "n_edits": int(control["n_edits"]),
                    "dataset_hash": control["dataset_hash"],
                    "record_cohort_hash": control["record_cohort_hash"],
                    "pairing_protocol_hash": control["pairing_protocol_hash"],
                    "control_value": control_value,
                    "treatment_value": treatment_value,
                    "raw_difference": raw_difference,
                    "favorable_difference": (
                        raw_difference if higher_is_better else -raw_difference
                    ),
                }
            )
        metrics[metric] = {
            "higher_is_better": higher_is_better,
            **summary,
            "favorable_pair_fraction": summary["favorable_pairs"] / summary["n"],
            "pairs": pair_rows,
        }
    return {
        "seeds": [int(record["seed"]) for record in controls],
        "metrics": metrics,
    }


def _flat_summary_row(
    *,
    dataset: str,
    contrast_type: str,
    contrast: str,
    mapping: str,
    recipe: str,
    control_recipe: str,
    control_mapping: str,
    treatment_recipe: str,
    treatment_mapping: str,
    metric: str,
    metric_summary: dict,
) -> dict:
    return {
        "dataset": dataset,
        "contrast_type": contrast_type,
        "contrast": contrast,
        "mapping": mapping,
        "recipe": recipe,
        "control_recipe": control_recipe,
        "control_mapping": control_mapping,
        "treatment_recipe": treatment_recipe,
        "treatment_mapping": treatment_mapping,
        "metric": metric,
        **{key: value for key, value in metric_summary.items() if key != "pairs"},
    }


def summarize(matrix: dict[str, Any]) -> tuple[dict, list[dict]]:
    payload: dict[str, Any] = {
        "schema_version": "ke_factorial_paired_summary_v1",
        "mode": matrix["mode"],
        "phase": matrix["phase"],
        "recipes": list(matrix["recipes"]),
        "mappings": list(MAPPINGS),
        "pair_identity_fields": list(PAIR_FIELDS),
        "control": "plain",
        "treatment": "ssr_only",
        "datasets": {},
        "arms": {},
        "contrasts": {},
        "mapping_contrasts": {},
    }
    flat_rows: list[dict] = []

    for dataset in matrix["datasets"]:
        payload["arms"][dataset] = {}
        for recipe in matrix["recipes"]:
            for mapping in sorted(matrix["recipe_mappings"][recipe]):
                arm = _records_for_arm(matrix, dataset, recipe, mapping)
                arm_payload = _arm_summary(arm, f"{dataset}/{recipe}/{mapping}")
                payload["arms"][dataset][_arm_key(recipe, mapping)] = {
                    "recipe": recipe,
                    "mapping": mapping,
                    **arm_payload,
                }

        payload["contrasts"][dataset] = {}
        for contrast, control_recipe, treatment_recipe in RECIPE_CONTRASTS:
            if (
                control_recipe not in matrix["recipes"]
                or treatment_recipe not in matrix["recipes"]
            ):
                continue
            payload["contrasts"][dataset][contrast] = {}
            for mapping in sorted(matrix["recipe_mappings"][treatment_recipe]):
                controls = _records_for_arm(matrix, dataset, control_recipe, "none")
                treatments = _records_for_arm(matrix, dataset, treatment_recipe, mapping)
                contrast_payload = _contrast_summary(
                    controls,
                    treatments,
                    label=f"{dataset}/{contrast}/{mapping}",
                )
                contrast_payload.update(
                    {
                        "control_recipe": control_recipe,
                        "control_mapping": "none",
                        "treatment_recipe": treatment_recipe,
                        "treatment_mapping": mapping,
                    }
                )
                payload["contrasts"][dataset][contrast][mapping] = contrast_payload
                for metric, metric_summary in contrast_payload["metrics"].items():
                    flat_rows.append(
                        _flat_summary_row(
                            dataset=dataset,
                            contrast_type="recipe",
                            contrast=contrast,
                            mapping=mapping,
                            recipe="",
                            control_recipe=control_recipe,
                            control_mapping="none",
                            treatment_recipe=treatment_recipe,
                            treatment_mapping=mapping,
                            metric=metric,
                            metric_summary=metric_summary,
                        )
                    )

        payload["mapping_contrasts"][dataset] = {}
        for recipe in sorted(MAPPING_DEPENDENT_RECIPES & set(matrix["recipes"])):
            available = matrix["recipe_mappings"][recipe]
            if not set(MAPPINGS).issubset(available):
                continue
            cosine = _records_for_arm(matrix, dataset, recipe, "cosine")
            projective = _records_for_arm(matrix, dataset, recipe, "projective")
            mapping_payload = _contrast_summary(
                cosine,
                projective,
                label=f"{dataset}/{recipe}/projective_minus_cosine",
            )
            mapping_payload.update(
                {
                    "recipe": recipe,
                    "contrast": "projective_minus_cosine",
                    "control_mapping": "cosine",
                    "treatment_mapping": "projective",
                }
            )
            payload["mapping_contrasts"][dataset][recipe] = mapping_payload
            for metric, metric_summary in mapping_payload["metrics"].items():
                flat_rows.append(
                    _flat_summary_row(
                        dataset=dataset,
                        contrast_type="mapping",
                        contrast="projective_minus_cosine",
                        mapping="projective_vs_cosine",
                        recipe=recipe,
                        control_recipe=recipe,
                        control_mapping="cosine",
                        treatment_recipe=recipe,
                        treatment_mapping="projective",
                        metric=metric,
                        metric_summary=metric_summary,
                    )
                )

        primary_mapping = (
            "projective"
            if "projective" in matrix["recipe_mappings"]["ssr_only"]
            else sorted(matrix["recipe_mappings"]["ssr_only"])[0]
        )
        primary = payload["contrasts"][dataset]["ssr_only_minus_plain"][primary_mapping]
        payload["datasets"][dataset] = {
            "seeds": primary["seeds"],
            "mapping": primary_mapping,
            "metrics": primary["metrics"],
        }

    return payload, flat_rows


def _write_csv_atomic(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty summary table: {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def write_outputs(output_dir: Path, payload: dict, rows: list[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "summary.json"
    json_tmp = json_path.with_name(f".{json_path.name}.tmp.{os.getpid()}")
    json_tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(json_tmp, json_path)
    _write_csv_atomic(output_dir / "summary.csv", rows)

    arm_rows = []
    for dataset, arms in payload["arms"].items():
        for arm in arms.values():
            for metric, summary in arm["metrics"].items():
                arm_rows.append(
                    {
                        "dataset": dataset,
                        "recipe": arm["recipe"],
                        "mapping": arm["mapping"],
                        "metric": metric,
                        "higher_is_better": summary["higher_is_better"],
                        "n": summary["n"],
                        "mean": summary["mean"],
                        "sd": summary["sd"],
                        "seeds": ",".join(str(seed) for seed in arm["seeds"]),
                    }
                )
    _write_csv_atomic(output_dir / "arms.csv", arm_rows)

    pair_rows = []
    for row in rows:
        if row["contrast_type"] == "recipe":
            contrast_payload = payload["contrasts"][row["dataset"]][row["contrast"]][
                row["mapping"]
            ]
        else:
            contrast_payload = payload["mapping_contrasts"][row["dataset"]][row["recipe"]]
        for pair in contrast_payload["metrics"][row["metric"]]["pairs"]:
            pair_rows.append(
                {
                    "dataset": row["dataset"],
                    "contrast_type": row["contrast_type"],
                    "contrast": row["contrast"],
                    "mapping": row["mapping"],
                    "recipe": row["recipe"],
                    "control_recipe": row["control_recipe"],
                    "control_mapping": row["control_mapping"],
                    "treatment_recipe": row["treatment_recipe"],
                    "treatment_mapping": row["treatment_mapping"],
                    "metric": row["metric"],
                    "higher_is_better": row["higher_is_better"],
                    **pair,
                }
            )
    _write_csv_atomic(output_dir / "paired_deltas.csv", pair_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    payload, rows = summarize(load_pairs(read_plan(args.plan.resolve())))
    write_outputs(args.output_dir.resolve(), payload, rows)
    print(
        f"Editing {payload['mode']} paired summary: "
        f"{len(rows)} dataset/contrast/metric rows"
    )


if __name__ == "__main__":
    main()
