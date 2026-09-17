#!/usr/bin/env python3
"""Verify the compact Figure 5 low-rank KE source tables."""

from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts" / "figure5_lowrank_ke"


def rows(name: str) -> list[dict[str, str]]:
    with (SOURCE / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def close(observed: float, expected: float, tolerance: float = 1e-9) -> None:
    if not math.isclose(observed, expected, abs_tol=tolerance):
        raise AssertionError(f"observed {observed}, expected {expected}")


def main() -> None:
    seed100 = rows("independent_100_seed_level.csv")
    if len(seed100) != 20:
        raise AssertionError("The 100-edit cohort must contain 10 complete pairs")
    seeds_by_arm = {
        arm: {row["seed"] for row in seed100 if row["arm"] == arm}
        for arm in ("LoRA", "LoRA+SSR")
    }
    if len(seeds_by_arm["LoRA"]) != 10 or seeds_by_arm["LoRA"] != seeds_by_arm["LoRA+SSR"]:
        raise AssertionError("The 100-edit arms do not contain the same ten seeds")
    efficacy = {
        arm: statistics.mean(
            float(row["immediate_efficacy_pct"]) for row in seed100 if row["arm"] == arm
        )
        for arm in seeds_by_arm
    }
    locality = {
        arm: statistics.mean(
            float(row["immediate_locality_pct"]) for row in seed100 if row["arm"] == arm
        )
        for arm in seeds_by_arm
    }
    close(efficacy["LoRA"], 69.4)
    close(efficacy["LoRA+SSR"], 81.5)
    close(efficacy["LoRA+SSR"] - efficacy["LoRA"], 12.1)
    close(locality["LoRA+SSR"] - locality["LoRA"], 0.05)

    summaries = rows("rank250_and_replication_summary.csv")
    rank_gain = {
        int(row["rank"]): float(row["delta_mean"])
        for row in summaries
        if row["cohort"] in {"v36_rank_250", "rank32_shared_250"}
    }
    if rank_gain != {8: 14.44, 16: 10.16, 32: 7.52, 64: 6.88}:
        raise AssertionError(f"Unexpected 250-edit rank gains: {rank_gain}")
    replication = [
        float(row["delta_mean"])
        for row in summaries
        if int(row["rank"]) == 8
    ]
    replication.append(efficacy["LoRA+SSR"] - efficacy["LoRA"])
    for observed, expected in zip(sorted(replication), [12.1, 14.44, 17.16]):
        close(observed, expected)

    mechanism = rows("mechanism_250_seed_level.csv")
    if len(mechanism) != 160:
        raise AssertionError("The mechanism cohort must contain 10 pairs at 8 checkpoints")
    for edit_count, metric, expected in (
        (100, "center_surround_contrast", 0.36932729082013144),
        (100, "kernel_alignment", 0.7064130330393386),
        (250, "center_surround_contrast", 0.338066528661823),
        (250, "kernel_alignment", 0.7129789033297182),
    ):
        by_seed: dict[str, dict[str, float]] = {}
        for row in mechanism:
            if int(row["edits"]) == edit_count:
                by_seed.setdefault(row["seed"], {})[row["arm"]] = float(row[metric])
        if len(by_seed) != 10 or any(set(pair) != {"LoRA", "LoRA+SSR"} for pair in by_seed.values()):
            raise AssertionError(f"Incomplete mechanism pairs at {edit_count} edits")
        delta = statistics.mean(pair["LoRA+SSR"] - pair["LoRA"] for pair in by_seed.values())
        close(delta, expected)

    print("Figure 5 low-rank KE source tables verified")


if __name__ == "__main__":
    main()
