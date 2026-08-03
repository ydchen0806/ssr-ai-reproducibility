"""Deterministic helpers for matched segmentation distillation."""

from __future__ import annotations

import torch


def old_class_distillation_ids(
    old_classes: list[int],
    batch_size: int,
    step: int,
    device: torch.device,
) -> torch.Tensor:
    """Cycle deterministically through previously learned class conditions."""
    if not old_classes:
        raise ValueError("old_classes must be non-empty for segmentation KD")
    old = torch.as_tensor(old_classes, dtype=torch.long, device=device)
    positions = (torch.arange(batch_size, device=device) + int(step) * batch_size) % old.numel()
    return old[positions]
