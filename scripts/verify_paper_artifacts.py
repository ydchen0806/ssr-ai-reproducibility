#!/usr/bin/env python3
"""Verify the compact 2026-08-03 paper artifact against reported values."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts" / "paper_20260803"
MANIFEST = json.loads((ARTIFACT / "manifest.json").read_text(encoding="utf-8"))
REPORTED = json.loads((ARTIFACT / "reported_values.json").read_text(encoding="utf-8"))
ERRORS: list[str] = []


def require(condition: bool, message: str) -> None:
    if not condition:
        ERRORS.append(message)


def close(actual: float, expected: float, tolerance: float = 1e-10) -> bool:
    return math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=tolerance)


def matches_rounding(actual: float, expected: float, decimals: int) -> bool:
    return abs(float(actual) - float(expected)) <= 0.5 * 10 ** (-decimals) + 1e-12


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def canonical_hash(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def summarize(rows: list[dict], metric: str) -> tuple[float, float]:
    values = [float(row[metric]) for row in rows]
    return statistics.mean(values), statistics.stdev(values)


def paired_interval(values: list[float], critical: float) -> tuple[float, float, float, float]:
    mean = statistics.mean(values)
    spread = statistics.stdev(values)
    half_width = critical * spread / math.sqrt(len(values))
    return mean, spread, mean - half_width, mean + half_width


def verify_checksums() -> None:
    checksum_path = ARTIFACT / MANIFEST["integrity_file"]
    require(checksum_path.is_file(), f"missing checksum file: {checksum_path}")
    if not checksum_path.is_file():
        return
    recorded: dict[str, str] = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        recorded[relative] = digest
    raw_files = {
        str(path.relative_to(ARTIFACT))
        for path in (ARTIFACT / "raw").rglob("*")
        if path.is_file()
    }
    require(set(recorded) == raw_files, "checksum inventory does not match raw artifact files")
    for relative, expected in sorted(recorded.items()):
        path = ARTIFACT / relative
        if not path.is_file():
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        require(actual == expected, f"checksum mismatch: {relative}")


def load_group(spec: dict) -> list[dict]:
    paths = sorted(ARTIFACT.glob(spec["glob"]))
    rows: list[dict] = []
    for path in paths:
        payload = load_jsonl(path) if spec["format"] == "jsonl" else [json.loads(path.read_text(encoding="utf-8"))]
        rows.extend(payload)
    seeds = sorted(int(row["seed"]) for row in rows)
    require(seeds == spec["seeds"], f"seed mismatch for {spec['id']}: {seeds}")
    return rows


def verify_classification() -> None:
    configs = json.loads((ARTIFACT / "raw/classification/configs.json").read_text(encoding="utf-8"))
    actual: dict[str, list[dict]] = {}
    for spec in MANIFEST["classification_groups"]:
        rows = load_group(spec)
        actual[spec["id"]] = rows
        config_record = configs[spec["id"]]
        require(
            canonical_hash(config_record["config"]) == config_record["normalized_sha256"],
            f"classification config digest mismatch: {spec['id']}",
        )
        require(
            all(row.get("_config_sha256") == config_record["normalized_sha256"] for row in rows),
            f"classification config reference mismatch: {spec['id']}",
        )
        summary = []
        for metric in ("avg_accuracy", "avg_forgetting"):
            mean, spread = summarize(rows, metric)
            summary.append(f"{metric}={mean:.12f}+/-{spread:.12f}")
        print(f"classification {spec['id']}: n={len(rows)}; " + "; ".join(summary))

    for group_id, metrics in REPORTED["classification"].items():
        rows = actual[group_id]
        for metric, (expected_mean, expected_sd, decimals) in metrics.items():
            mean, spread = summarize(rows, metric)
            require(matches_rounding(mean, expected_mean, decimals), f"reported mean mismatch: {group_id}/{metric}")
            require(matches_rounding(spread, expected_sd, decimals), f"reported SD mismatch: {group_id}/{metric}")

    for control, treatment in (
        ("cub200_kd", "cub200_ssr_kd"),
        ("cifar100_kd", "cifar100_ssr_kd"),
        ("tinyimagenet_kd", "tinyimagenet_ssr_kd"),
        ("five_datasets_kd", "five_datasets_ssr_kd"),
        ("cifar10_kd", "cifar10_ssr_kd"),
    ):
        require(
            sorted(int(row["seed"]) for row in actual[control])
            == sorted(int(row["seed"]) for row in actual[treatment]),
            f"unmatched paired seeds: {control} vs {treatment}",
        )


def verify_context_baselines() -> None:
    spec = MANIFEST["context_baselines"]
    all_rows = load_jsonl(ARTIFACT / spec["path"])
    configs = json.loads((ARTIFACT / spec["configs"]).read_text(encoding="utf-8"))
    for method in spec["methods"]:
        for dataset in spec["datasets"]:
            group_id = f"{method}_{dataset}"
            rows = [row for row in all_rows if row["group_id"] == group_id]
            require(sorted(int(row["seed"]) for row in rows) == spec["seeds"], f"seed mismatch: {group_id}")
            config_record = configs[group_id]
            require(
                canonical_hash(config_record["config"]) == config_record["normalized_sha256"],
                f"context config digest mismatch: {group_id}",
            )
            require(
                all(row.get("_config_sha256") == config_record["normalized_sha256"] for row in rows),
                f"context config reference mismatch: {group_id}",
            )
            aa, aa_sd = summarize(rows, "avg_accuracy")
            af, af_sd = summarize(rows, "avg_forgetting")
            expected = REPORTED["context_baselines"][group_id]
            for actual, target, label in zip((aa, aa_sd, af, af_sd), expected, ("AA", "AA SD", "AF", "AF SD")):
                require(matches_rounding(actual, target, 2), f"context mismatch: {group_id}/{label}")


def verify_segmentation() -> None:
    protocol = json.loads(
        (ARTIFACT / "raw/segmentation/protocol.json").read_text(encoding="utf-8")
    )
    require(protocol["cohort_seeds"] == [0, 1, 2], "segmentation protocol cohort mismatch")
    require(
        "not recoverable" in protocol["configuration_evidence"]["seed_0"],
        "segmentation seed-0 provenance limitation is not explicit",
    )
    later_configs = protocol["configuration_evidence"]["seeds_1_2"]
    require(
        all(config.get("seeds") == [1, 2] for config in later_configs.values()),
        "segmentation recovered launch manifests must identify seeds 1 and 2",
    )
    for spec in MANIFEST["segmentation_groups"]:
        rows = [row for row in load_jsonl(ARTIFACT / spec["path"]) if row["method"] == spec["method"]]
        require(sorted(int(row["seed"]) for row in rows) == spec["seeds"], f"segmentation seed mismatch: {spec['id']}")
        for metric, expected in REPORTED["segmentation"][spec["id"]].items():
            mean, spread = summarize(rows, metric)
            require(matches_rounding(mean, expected[0], 2), f"segmentation mean mismatch: {spec['id']}/{metric}")
            require(matches_rounding(spread, expected[1], 2), f"segmentation SD mismatch: {spec['id']}/{metric}")
        print(f"segmentation {spec['id']}: n={len(rows)}")


def verify_initial_adapter() -> None:
    rows = load_jsonl(ARTIFACT / "raw/adapter/initial_three_seed/runs.jsonl")
    for method, metrics in REPORTED["initial_adapter_sem"].items():
        selected = [row for row in rows if row["method"] == method]
        require(sorted(int(row["seed"]) for row in selected) == [0, 1, 2], f"initial adapter seeds: {method}")
        for metric, expected in metrics.items():
            values = [float(row[metric]) for row in selected]
            mean = statistics.mean(values)
            sem = statistics.stdev(values) / math.sqrt(len(values))
            decimals = 3 if metric == "mean_abs_offdiag_cosine" else 2
            require(matches_rounding(mean, expected[0], decimals), f"initial adapter mean mismatch: {method}/{metric}")
            require(matches_rounding(sem, expected[1], decimals), f"initial adapter SEM mismatch: {method}/{metric}")


def verify_adapter_validation() -> None:
    spec = MANIFEST["adapter_validation"]
    all_rows = load_jsonl(ARTIFACT / spec["path"])
    configs = json.loads((ARTIFACT / spec["configs"]).read_text(encoding="utf-8"))
    for rank, condition in spec["conditions"]:
        control_rows = [row for row in all_rows if int(row["_rank"]) == rank and row["_condition"] == "kd"]
        treated_rows = [row for row in all_rows if int(row["_rank"]) == rank and row["_condition"] == condition]
        control = {int(row["seed"]): row for row in control_rows}
        treated = {int(row["seed"]): row for row in treated_rows}
        require(sorted(control) == spec["seeds"], f"adapter control seeds: rank {rank}")
        require(sorted(treated) == spec["seeds"], f"adapter treatment seeds: rank {rank}/{condition}")

        for config_key, rows in ((f"rank{rank}/kd", control_rows), (f"rank{rank}/{condition}", treated_rows)):
            config_record = configs[config_key]
            require(int(config_record["config"]["rank"]) == rank, f"adapter rank drift: {config_key}")
            require(
                canonical_hash(config_record["config"]) == config_record["normalized_sha256"],
                f"adapter config digest mismatch: {config_key}",
            )
            require(
                all(row.get("_config_sha256") == config_record["normalized_sha256"] for row in rows),
                f"adapter config reference mismatch: {config_key}",
            )

        aa = [treated[seed]["avg_accuracy"] - control[seed]["avg_accuracy"] for seed in spec["seeds"]]
        af = [control[seed]["avg_forgetting"] - treated[seed]["avg_forgetting"] for seed in spec["seeds"]]
        aa_stats = paired_interval(aa, 2.2621571627409915)
        af_stats = paired_interval(af, 2.2621571627409915)
        wins = sum(a > 0 and f > 0 for a, f in zip(aa, af))
        reported = REPORTED["adapter_validation"][f"rank{rank}/{condition}"]
        for actual, expected, label in zip((aa_stats[0], aa_stats[2], aa_stats[3]), reported[0], ("AA", "AA low", "AA high")):
            require(matches_rounding(actual, expected, 3), f"adapter mismatch: rank {rank}/{condition}/{label}")
        for actual, expected, label in zip((af_stats[0], af_stats[2], af_stats[3]), reported[1], ("AF", "AF low", "AF high")):
            require(matches_rounding(actual, expected, 3), f"adapter mismatch: rank {rank}/{condition}/{label}")
        require(wins == reported[2], f"adapter dual-win mismatch: rank {rank}/{condition}")
        print(f"adapter rank{rank}/{condition}: AA={aa_stats[0]:+.6f}, AFred={af_stats[0]:+.6f}, wins={wins}/10")


def llm_rows(rows: list[dict], dataset: str, variant: str) -> dict[int, dict]:
    return {
        int(row["seed"]): row
        for row in rows
        if row["_dataset"] == dataset and row["_variant"] == variant
    }


def verify_llm() -> None:
    spec = MANIFEST["llm_nested"]
    validation_rows = load_jsonl(ARTIFACT / spec["path"])
    expected_variants = {
        variant
        for contrast in spec["contrasts"]
        for variant in (contrast["treatment"], contrast["control"])
    }
    require(
        len(validation_rows) == len(spec["datasets"]) * len(expected_variants) * len(spec["seeds"]),
        "LLM validation row count does not match the declared contrasts",
    )
    require(
        {row["_variant"] for row in validation_rows} == expected_variants,
        "LLM validation variants do not match the declared contrasts",
    )
    protocol = json.loads((ARTIFACT / "raw/llm/protocol.json").read_text(encoding="utf-8"))
    audit = protocol["implementation_audit"]
    gaussian_hwhm = [
        sigma * math.sqrt(2.0 * math.log(2.0))
        for sigma in audit["gaussian_reference_sigma"]
    ]
    inverse_hwhm = [
        sigma * math.sqrt(3.0)
        for sigma in audit["recorded_inverse_sigma"]
    ]
    matched_inverse_sigma = [width / math.sqrt(3.0) for width in gaussian_hwhm]
    for actual, expected in zip(gaussian_hwhm, audit["gaussian_reference_hwhm"]):
        require(close(actual, expected, 1e-14), "LLM Gaussian HWHM audit mismatch")
    for actual, expected in zip(inverse_hwhm, audit["recorded_inverse_hwhm"]):
        require(close(actual, expected, 1e-14), "LLM inverse HWHM audit mismatch")
    for actual, expected in zip(matched_inverse_sigma, audit["hwhm_matched_inverse_sigma"]):
        require(close(actual, expected, 1e-14), "LLM matched inverse scale audit mismatch")
    require(
        all(actual > target for actual, target in zip(inverse_hwhm, gaussian_hwhm)),
        "recorded ZsRE inverse condition is no longer broader than the HWHM target",
    )
    require(
        protocol["selection_lock"]["selection"]["zsre"]["kernel_family"] == "inverse",
        "LLM implementation audit is no longer attached to the selected ZsRE inverse condition",
    )
    zsre_inverse = llm_rows(validation_rows, "zsre", "full")
    for row in zsre_inverse.values():
        require(
            close(row["sigma_exc"], audit["recorded_inverse_sigma"][0], 1e-14)
            and close(row["sigma_inh"], audit["recorded_inverse_sigma"][1], 1e-14),
            f"ZsRE inverse validation scale mismatch: seed {row['seed']}",
        )
    matched_fields = (
        "model",
        "target_layers",
        "target_module_regex",
        "lambda_spectral",
        "lambda_anchor",
        "lr",
        "num_steps",
        "seed",
    )
    for contrast in spec["contrasts"]:
        contrast_id = contrast["id"]
        for dataset in spec["datasets"]:
            treatment_rows = [
                row
                for row in validation_rows
                if row["_dataset"] == dataset and row["_variant"] == contrast["treatment"]
            ]
            control_rows = [
                row
                for row in validation_rows
                if row["_dataset"] == dataset and row["_variant"] == contrast["control"]
            ]
            treatment = llm_rows(validation_rows, dataset, contrast["treatment"])
            control = llm_rows(validation_rows, dataset, contrast["control"])
            require(
                len(treatment_rows) == len(spec["seeds"])
                and len(control_rows) == len(spec["seeds"]),
                f"LLM duplicate or missing rows: {contrast_id}/{dataset}",
            )
            require(
                sorted(treatment) == spec["seeds"] and sorted(control) == spec["seeds"],
                f"LLM seed mismatch: {contrast_id}/{dataset}",
            )
            for seed in spec["seeds"]:
                for key in matched_fields:
                    require(
                        treatment[seed].get(key) == control[seed].get(key),
                        f"LLM nested config drift: {contrast_id}/{dataset}/{seed}/{key}",
                    )
                require(
                    float(treatment[seed]["lambda_biocs"]) > 0
                    and float(control[seed]["lambda_biocs"]) == 0,
                    f"LLM SSR term mismatch: {contrast_id}/{dataset}/{seed}",
                )
            efficacy = [
                treatment[seed]["efficacy"] - control[seed]["efficacy"]
                for seed in spec["seeds"]
            ]
            locality = [
                treatment[seed]["locality"] - control[seed]["locality"]
                for seed in spec["seeds"]
            ]
            eff_stats = paired_interval(efficacy, 1.96)
            loc_stats = paired_interval(locality, 1.96)
            wins = sum(e >= 0 and loc > 0 for e, loc in zip(efficacy, locality))
            reported = REPORTED["llm_nested"][contrast_id][dataset]
            for actual, expected in zip((eff_stats[0], eff_stats[2], eff_stats[3]), reported[0]):
                require(matches_rounding(actual, expected, 2), f"LLM efficacy mismatch: {contrast_id}/{dataset}")
            for actual, expected in zip((loc_stats[0], loc_stats[2], loc_stats[3]), reported[1]):
                require(matches_rounding(actual, expected, 3), f"LLM locality mismatch: {contrast_id}/{dataset}")
            require(wins == reported[2], f"LLM favorable-pair mismatch: {contrast_id}/{dataset}")
            require(
                loc_stats[2] <= 0 <= loc_stats[3],
                f"LLM null interval no longer crosses zero: {contrast_id}/{dataset}",
            )
            print(
                f"llm {contrast_id}/{dataset}: locality={loc_stats[0]:+.6f} "
                f"[{loc_stats[2]:+.6f}, {loc_stats[3]:+.6f}]"
            )

    print(
        "llm kernel audit: selected ZsRE inverse response is broader than the "
        "intended HWHM-matched control"
    )

    descriptive_rows = load_jsonl(ARTIFACT / spec["descriptive_path"])
    for relative, efficacy, locality in REPORTED["descriptive_llm_pairs"]:
        matches = [row for row in descriptive_rows if row["_source_path"].endswith(relative)]
        require(len(matches) == 1, f"descriptive LLM source mismatch: {relative}")
        if not matches:
            continue
        payload = matches[0]
        require(close(payload["efficacy"], efficacy, 1e-9), f"descriptive LLM efficacy mismatch: {relative}")
        require(matches_rounding(payload["locality"], locality, 2), f"descriptive LLM locality mismatch: {relative}")


def verify_mechanism() -> None:
    spec = MANIFEST["mechanism"]
    taxonomy_path = ARTIFACT / spec["taxonomy"]
    require(
        hashlib.sha256(taxonomy_path.read_bytes()).hexdigest() == spec["taxonomy_sha256"],
        "CUB taxonomy source digest mismatch",
    )
    taxonomy = json.loads(taxonomy_path.read_text(encoding="utf-8"))
    taxonomy_rows = taxonomy.get("classes", [])
    require(
        sorted(int(row["class_id"]) for row in taxonomy_rows) == list(range(200)),
        "CUB taxonomy must contain each class id from 0 through 199 exactly once",
    )
    require(
        all(row.get("order") and row.get("family") and row.get("genus") for row in taxonomy_rows),
        "CUB taxonomy contains incomplete order/family/genus assignments",
    )
    with (ARTIFACT / spec["csv"]).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    report = json.loads((ARTIFACT / spec["report"]).read_text(encoding="utf-8"))
    metrics = (
        "delta_aa", "forgetting_reduction", "top1_confuser_margin_gain",
        "top3_confuser_margin_gain", "semantic_knn_recall_delta",
        "close_displacement", "distant_displacement", "distance_selectivity",
        "locality_ratio", "avg_accuracy", "avg_forgetting",
    )
    for family in spec["families"]:
        selected = [row for row in rows if row["family"] == family]
        require(sorted(int(row["seed"]) for row in selected) == spec["validation_seeds"], f"mechanism seeds: {family}")
        expected = report["validation_summary"][family]
        for metric in metrics:
            values = [float(row[metric]) for row in selected]
            mean, spread, low, high = paired_interval(values, 2.093)
            recorded = expected[metric]
            for actual, target, label in ((mean, recorded["mean"], "mean"), (spread, recorded["std"], "SD"), (low, recorded["ci95_low"], "low"), (high, recorded["ci95_high"], "high")):
                require(close(actual, target, 2e-12), f"mechanism {label} mismatch: {family}/{metric}")
        dual = sum(float(row["delta_aa"]) > 0 and float(row["forgetting_reduction"]) > 0 for row in selected)
        margin = sum(float(row["top1_confuser_margin_gain"]) > 0 for row in selected)
        knn = sum(float(row["semantic_knn_recall_delta"]) >= -0.01 for row in selected)
        require(dual == expected["dual_wins"], f"mechanism dual wins: {family}")
        require(margin == expected["margin_wins"], f"mechanism margin wins: {family}")
        require(knn == expected["knn_noninferior_seeds"], f"mechanism kNN count: {family}")
    require(all(report["primary_gate"].values()), "held-out CUB primary gate is not fully satisfied")
    print("mechanism: 20 held-out seeds x 6 equal-budget families verified")


def main() -> int:
    verify_checksums()
    verify_classification()
    verify_context_baselines()
    verify_segmentation()
    verify_initial_adapter()
    verify_adapter_validation()
    verify_llm()
    verify_mechanism()
    if ERRORS:
        print("\nVERIFICATION FAILED", file=sys.stderr)
        for error in ERRORS:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("\nVERIFICATION PASSED: checksums, cohorts, configurations, and reported values agree.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
