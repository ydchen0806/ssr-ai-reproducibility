"""Utilities for locked segmentation screen--refine--confirm studies."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F


KERNEL_FAMILIES = ("gaussian", "laplace", "cauchy")


def hwhm_to_scale(family: str, hwhm: float) -> float:
    """Convert half width at half maximum to a kernel-native scale."""
    if not math.isfinite(hwhm) or hwhm <= 0:
        raise ValueError("hwhm must be finite and positive")
    family = family.lower()
    if family == "gaussian":
        return hwhm / math.sqrt(2.0 * math.log(2.0))
    if family == "laplace":
        return hwhm / math.log(2.0)
    if family == "cauchy":
        return hwhm
    raise ValueError(f"Unsupported kernel family: {family!r}")


def scale_to_hwhm(family: str, scale: float) -> float:
    """Convert a kernel-native scale to half width at half maximum."""
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")
    family = family.lower()
    if family == "gaussian":
        return scale * math.sqrt(2.0 * math.log(2.0))
    if family == "laplace":
        return scale * math.log(2.0)
    if family == "cauchy":
        return scale
    raise ValueError(f"Unsupported kernel family: {family!r}")


def _radial_response(distance: torch.Tensor, scale: float, family: str) -> torch.Tensor:
    if family == "gaussian":
        return torch.exp(-(distance.square()) / (2.0 * scale * scale))
    if family == "laplace":
        return torch.exp(-distance / scale)
    if family == "cauchy":
        return 1.0 / (1.0 + (distance / scale).square())
    raise ValueError(f"Unsupported kernel family: {family!r}")


def cross_set_ssr_loss(
    current: torch.Tensor,
    references: torch.Tensor,
    *,
    kernel: str,
    hwhm_exc: float,
    hwhm_inh: float,
    a_exc: float = 1.0,
    a_inh: float = 0.8,
) -> torch.Tensor:
    """SSR interactions from plastic rows to stop-gradient reference rows.

    This is the continual-learning form of the same center--surround kernel:
    only the current-task rows receive gradients from the spatial term.  The
    reference rows remain available to define geometry but cannot be displaced
    by that term.
    """
    if current.ndim != 2 or references.ndim != 2:
        raise ValueError("current and references must be rank-2 matrices")
    if current.shape[1] != references.shape[1]:
        raise ValueError("current and references must have the same feature width")
    if current.shape[0] == 0 or references.shape[0] == 0:
        return current.sum() * 0.0
    if not all(math.isfinite(value) and value >= 0 for value in (a_exc, a_inh)):
        raise ValueError("kernel amplitudes must be finite and non-negative")
    family = kernel.lower()
    sigma_exc = hwhm_to_scale(family, hwhm_exc)
    sigma_inh = hwhm_to_scale(family, hwhm_inh)
    plastic = F.normalize(current.float(), dim=1)
    stable = F.normalize(references.detach().float(), dim=1)
    cosine = torch.clamp(plastic @ stable.T, -1.0, 1.0)
    distance = torch.sqrt(torch.clamp(1.0 - cosine, min=1e-8))
    inhibitory = a_inh * _radial_response(distance, sigma_inh, family)
    excitatory = a_exc * _radial_response(distance, sigma_exc, family)
    return ((inhibitory - excitatory) + (a_exc - a_inh)).mean()


@dataclass(frozen=True)
class Candidate:
    spec_id: str
    kernel: str
    hwhm_exc: float
    hwhm_inh: float
    sigma_exc: float
    sigma_inh: float
    lambda_ssr: float
    target: str
    scope: str
    a_exc: float = 1.0
    a_inh: float = 0.8

    def as_dict(self) -> dict:
        return asdict(self)


def candidate_matrix(
    kernels: Sequence[str],
    widths: Sequence[tuple[float, float]],
    lambdas: Sequence[float],
    target_scopes: Sequence[tuple[str, str]],
) -> tuple[Candidate, ...]:
    candidates = []
    for kernel in kernels:
        family = kernel.lower()
        if family not in KERNEL_FAMILIES:
            raise ValueError(f"Unsupported kernel family: {kernel!r}")
        for width_index, (hwhm_exc, hwhm_inh) in enumerate(widths):
            if hwhm_exc >= hwhm_inh:
                raise ValueError("excitatory HWHM must be narrower than inhibitory HWHM")
            for lambda_ssr in lambdas:
                if not math.isfinite(lambda_ssr) or lambda_ssr <= 0:
                    raise ValueError("lambda_ssr values must be finite and positive")
                for target, scope in target_scopes:
                    if (target, scope) not in {("class", "new_old"), ("channels", "all")}:
                        raise ValueError(f"Unsupported target/scope pair: {(target, scope)!r}")
                    target_tag = "clsnew" if target == "class" else "channel"
                    lambda_tag = f"{lambda_ssr:.6g}".replace(".", "p")
                    spec_id = f"{family}_w{width_index}_{target_tag}_l{lambda_tag}"
                    candidates.append(
                        Candidate(
                            spec_id=spec_id,
                            kernel=family,
                            hwhm_exc=float(hwhm_exc),
                            hwhm_inh=float(hwhm_inh),
                            sigma_exc=hwhm_to_scale(family, hwhm_exc),
                            sigma_inh=hwhm_to_scale(family, hwhm_inh),
                            lambda_ssr=float(lambda_ssr),
                            target=target,
                            scope=scope,
                        )
                    )
    identifiers = [candidate.spec_id for candidate in candidates]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("candidate identifiers are not unique")
    return tuple(candidates)


def shard_candidates(
    candidates: Sequence[Candidate], shard_index: int, shard_count: int
) -> tuple[Candidate, ...]:
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must be in [0, shard_count)")
    return tuple(candidates[shard_index::shard_count])


def select_top(
    rows: Iterable[Mapping],
    *,
    k: int = 4,
    score_key: str = "selection_score",
    stage: str = "refine",
) -> list[dict]:
    if k < 1:
        raise ValueError("k must be positive")
    eligible = [dict(row) for row in rows if row.get("stage", stage) == stage]
    if not eligible:
        raise ValueError(f"No {stage!r} rows are available for selection")
    for row in eligible:
        score = row.get(score_key)
        if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
            raise ValueError(f"Selection row lacks finite {score_key!r}: {row}")
    return sorted(
        eligible,
        key=lambda row: (
            bool(row.get("endpoint_gate", False)),
            bool(row.get("geometry_gate", False)),
            float(row[score_key]),
            float(row.get("sum_score", float("-inf"))),
            str(row.get("spec_id", "")),
        ),
        reverse=True,
    )[:k]


def lock_selection(
    top_rows: Sequence[Mapping], *, confirmation_results: object | None = None
) -> dict:
    if confirmation_results is not None:
        raise ValueError("Selection locks cannot inspect confirmation results")
    if not top_rows:
        raise ValueError("At least one refinement row is required")
    selected = select_top(top_rows, k=1, stage="refine")[0]
    return {
        "selected_spec_id": selected["spec_id"],
        "selection_score": float(selected["selection_score"]),
        "endpoint_gate": bool(selected.get("endpoint_gate", False)),
        "geometry_gate": bool(selected.get("geometry_gate", False)),
    }


def validate_paired_identities(
    controls: Sequence[Mapping],
    treatments: Sequence[Mapping],
    identity_fields: Sequence[str] = (
        "git_commit",
        "task_family",
        "dataset",
        "model",
        "seed",
        "dataset_hash",
        "source_dataset_fingerprint",
        "feature_cache_sha256",
        "encoder_state_sha256",
        "torchvision_version",
        "selection_lock_hash",
        "initial_model_hash",
        "task_schedule_hash",
        "optimizer_steps",
    ),
) -> None:
    if len(controls) != len(treatments):
        raise ValueError("control and treatment inventories differ")
    for index, (control, treatment) in enumerate(zip(controls, treatments)):
        mismatches = {
            field: (control.get(field), treatment.get(field))
            for field in identity_fields
            if control.get(field) != treatment.get(field)
        }
        if mismatches:
            raise ValueError(f"Unmatched pair {index}: {mismatches}")


def stratified_fit_validation_indices(
    labels: torch.Tensor,
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a deterministic per-class fit/validation partition."""
    if labels.ndim != 1 or labels.numel() == 0:
        raise ValueError("labels must be a non-empty rank-1 tensor")
    if not 0.0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be between zero and one half")
    fit: list[torch.Tensor] = []
    validation: list[torch.Tensor] = []
    labels_cpu = labels.detach().cpu()
    for class_id in sorted(int(value) for value in labels_cpu.unique().tolist()):
        indices = torch.where(labels_cpu == class_id)[0]
        if indices.numel() < 2:
            raise ValueError(f"class {class_id} has fewer than two samples")
        generator = torch.Generator().manual_seed(seed + 1_000_003 * class_id)
        shuffled = indices[torch.randperm(indices.numel(), generator=generator)]
        validation_count = max(
            1,
            min(indices.numel() - 1, round(indices.numel() * validation_fraction)),
        )
        validation.append(shuffled[:validation_count])
        fit.append(shuffled[validation_count:])
    return torch.cat(fit).sort().values, torch.cat(validation).sort().values
