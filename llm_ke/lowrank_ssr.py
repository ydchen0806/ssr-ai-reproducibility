"""Center-surround SSR on the plastic output basis of a LoRA adapter.

The update in this module is the low-rank operator used by the sequential
editing experiments. It acts only on two-dimensional ``lora_B`` weights and
does not modify the frozen language-model backbone or ``lora_A`` factors.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import torch


def iter_lora_b_parameters(
    model: torch.nn.Module,
) -> Iterator[tuple[str, torch.nn.Parameter]]:
    """Yield LoRA-B matrices in a deterministic module-name order."""
    for name, parameter in sorted(model.named_parameters()):
        if ".lora_B." in name and name.endswith(".weight") and parameter.ndim == 2:
            yield name, parameter


class LoRABasisSSR:
    """Apply a bounded center-surround proximal step to LoRA-B output rows."""

    def __init__(
        self,
        coefficient: float,
        *,
        near_weight: float = 1.0,
        surround_weight: float = 0.2,
    ) -> None:
        if coefficient < 0:
            raise ValueError("SSR coefficient must be non-negative")
        if near_weight <= 0 or surround_weight < 0:
            raise ValueError("near_weight must be positive and surround_weight non-negative")
        self.coefficient = float(coefficient)
        self.near_weight = float(near_weight)
        self.surround_weight = float(surround_weight)
        self.names: tuple[str, ...] | None = None

    @torch.no_grad()
    def apply(self, model: torch.nn.Module) -> None:
        """Update all LoRA-B matrices in place after one native LoRA edit."""
        if self.coefficient == 0:
            return
        parameters = tuple(iter_lora_b_parameters(model))
        discovered = tuple(name for name, _ in parameters)
        if not discovered:
            raise RuntimeError("No two-dimensional LoRA-B weights were found")
        if self.names is None:
            self.names = discovered
        elif discovered != self.names:
            raise RuntimeError("The LoRA-B parameter set changed across sequential edits")

        for _, parameter in parameters:
            basis = parameter.detach().float()
            near = 0.5 * (torch.roll(basis, 1, 0) + torch.roll(basis, -1, 0))
            surround = 0.5 * (torch.roll(basis, 2, 0) + torch.roll(basis, -2, 0))
            response = (
                self.near_weight * (basis - near)
                - self.surround_weight * (basis - surround)
            )
            parameter.copy_((basis - self.coefficient * response).to(parameter.dtype))


@torch.no_grad()
def summarize_lora_geometry(model: torch.nn.Module) -> dict[str, Any]:
    """Measure effective rank and center-surround alignment on LoRA-B."""
    matrices = [parameter.detach().float() for _, parameter in iter_lora_b_parameters(model)]
    if not matrices:
        raise RuntimeError("LoRA geometry was requested before LoRA-B existed")

    per_module = []
    for matrix in matrices:
        singular = torch.linalg.svdvals(matrix)
        total = singular.sum()
        if total <= 0:
            effective_rank = 0.0
        else:
            probabilities = singular / total
            epsilon = torch.finfo(probabilities.dtype).tiny
            entropy = -(probabilities * probabilities.clamp_min(epsilon).log()).sum()
            effective_rank = float(entropy.exp().item())
        rank_ceiling = max(1, min(matrix.shape))

        epsilon = torch.finfo(matrix.dtype).tiny
        normalized = matrix / matrix.norm(dim=1, keepdim=True).clamp_min(epsilon)
        near = 0.5 * (
            (normalized * torch.roll(normalized, 1, 0)).sum(dim=1)
            + (normalized * torch.roll(normalized, -1, 0)).sum(dim=1)
        )
        surround = 0.5 * (
            (normalized * torch.roll(normalized, 2, 0)).sum(dim=1)
            + (normalized * torch.roll(normalized, -2, 0)).sum(dim=1)
        )
        observed = torch.cat((near, surround))
        target = torch.cat((torch.ones_like(near), -0.2 * torch.ones_like(surround)))
        alignment = torch.dot(observed, target) / (
            observed.norm().clamp_min(epsilon) * target.norm().clamp_min(epsilon)
        )
        per_module.append(
            {
                "effective_rank": effective_rank,
                "effective_rank_fraction": effective_rank / rank_ceiling,
                "near_row_cosine": float(near.mean().item()),
                "surround_row_cosine": float(surround.mean().item()),
                "center_surround_contrast": float((near.mean() - surround.mean()).item()),
                "kernel_alignment": float(alignment.item()),
                "n_rows": int(matrix.shape[0]),
            }
        )

    weights = torch.tensor([row["n_rows"] for row in per_module], dtype=torch.float64)
    weights /= weights.sum()

    def weighted(key: str) -> float:
        values = torch.tensor([row[key] for row in per_module], dtype=torch.float64)
        return float(torch.dot(values, weights).item())

    return {
        "definition": (
            "Entropy effective rank and normalized alignment between cyclic +/-1 and +/-2 "
            "LoRA-B row-cosine profiles and the fixed SSR target weights (+1, -0.2)."
        ),
        "n_modules": len(per_module),
        "effective_rank": weighted("effective_rank"),
        "effective_rank_fraction": weighted("effective_rank_fraction"),
        "near_row_cosine": weighted("near_row_cosine"),
        "surround_row_cosine": weighted("surround_row_cosine"),
        "center_surround_contrast": weighted("center_surround_contrast"),
        "kernel_alignment": weighted("kernel_alignment"),
    }
