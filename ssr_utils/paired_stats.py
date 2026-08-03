"""Paired summaries used by the meeting-revision experiment tables."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class PairedSummary:
    n: int
    control_mean: float
    control_sd: float
    treatment_mean: float
    treatment_sd: float
    raw_difference_mean: float
    raw_difference_sd: float
    difference_mean: float
    difference_sd: float
    ci95_low: float
    ci95_high: float
    favorable_pairs: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "n": self.n,
            "control_mean": self.control_mean,
            "control_sd": self.control_sd,
            "treatment_mean": self.treatment_mean,
            "treatment_sd": self.treatment_sd,
            "raw_difference_mean": self.raw_difference_mean,
            "raw_difference_sd": self.raw_difference_sd,
            "difference_mean": self.difference_mean,
            "difference_sd": self.difference_sd,
            "ci95_low": self.ci95_low,
            "ci95_high": self.ci95_high,
            "favorable_pairs": self.favorable_pairs,
        }


def paired_summary(
    control: Sequence[float] | np.ndarray,
    treatment: Sequence[float] | np.ndarray,
    *,
    higher_is_better: bool = True,
    confidence: float = 0.95,
) -> PairedSummary:
    control_array = np.asarray(control, dtype=float)
    treatment_array = np.asarray(treatment, dtype=float)
    if control_array.ndim != 1 or treatment_array.ndim != 1:
        raise ValueError("paired arrays must be one-dimensional")
    if control_array.size != treatment_array.size:
        raise ValueError("paired arrays must have the same length")
    if control_array.size < 2:
        raise ValueError("at least two complete pairs are required")
    if not np.isfinite(control_array).all() or not np.isfinite(treatment_array).all():
        raise ValueError("paired arrays must contain only finite values")

    raw_difference = treatment_array - control_array
    favorable_difference = raw_difference if higher_is_better else -raw_difference
    mean = float(np.mean(favorable_difference))
    sd = float(np.std(favorable_difference, ddof=1))
    standard_error = sd / math.sqrt(control_array.size)
    critical = float(stats.t.ppf((1.0 + confidence) / 2.0, control_array.size - 1))
    half_width = critical * standard_error
    return PairedSummary(
        n=int(control_array.size),
        control_mean=float(np.mean(control_array)),
        control_sd=float(np.std(control_array, ddof=1)),
        treatment_mean=float(np.mean(treatment_array)),
        treatment_sd=float(np.std(treatment_array, ddof=1)),
        raw_difference_mean=float(np.mean(raw_difference)),
        raw_difference_sd=float(np.std(raw_difference, ddof=1)),
        difference_mean=mean,
        difference_sd=sd,
        ci95_low=mean - half_width,
        ci95_high=mean + half_width,
        favorable_pairs=int(np.sum(favorable_difference > 0)),
    )


def complete_pairs(
    control_rows: Iterable[dict],
    treatment_rows: Iterable[dict],
    *,
    identity_fields: Sequence[str],
) -> tuple[list[tuple[dict, dict]], list[tuple]]:
    def index(rows: Iterable[dict]) -> dict[tuple, dict]:
        output: dict[tuple, dict] = {}
        for row in rows:
            key = tuple(row[field] for field in identity_fields)
            if key in output:
                raise ValueError(f"duplicate result identity: {key}")
            output[key] = row
        return output

    controls = index(control_rows)
    treatments = index(treatment_rows)
    shared = sorted(controls.keys() & treatments.keys())
    missing = sorted(controls.keys() ^ treatments.keys())
    return [(controls[key], treatments[key]) for key in shared], missing
