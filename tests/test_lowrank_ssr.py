from __future__ import annotations

import copy

import pytest
import torch

from llm_ke.lowrank_ssr import LoRABasisSSR, summarize_lora_geometry


class TinyAdapter(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = torch.nn.Module()
        self.block.lora_B = torch.nn.ModuleDict(
            {"default": torch.nn.Linear(3, 8, bias=False)}
        )
        self.frozen = torch.nn.Linear(3, 3, bias=False)
        with torch.no_grad():
            values = torch.arange(24, dtype=torch.float32).reshape(8, 3) / 24
            self.block.lora_B["default"].weight.copy_(values)


def test_zero_coefficient_is_an_exact_noop():
    model = TinyAdapter()
    before = copy.deepcopy(model.state_dict())

    LoRABasisSSR(0.0).apply(model)

    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name])


def test_ssr_changes_only_lora_b():
    model = TinyAdapter()
    before_basis = model.block.lora_B["default"].weight.detach().clone()
    before_frozen = model.frozen.weight.detach().clone()

    LoRABasisSSR(0.05).apply(model)

    assert not torch.equal(model.block.lora_B["default"].weight, before_basis)
    assert torch.equal(model.frozen.weight, before_frozen)


def test_geometry_summary_is_finite_and_reports_the_plastic_object():
    summary = summarize_lora_geometry(TinyAdapter())

    assert summary["n_modules"] == 1
    assert 0 < summary["effective_rank"] <= 3
    assert 0 < summary["effective_rank_fraction"] <= 1
    for key in (
        "near_row_cosine",
        "surround_row_cosine",
        "center_surround_contrast",
        "kernel_alignment",
    ):
        assert torch.isfinite(torch.tensor(summary[key]))


def test_missing_lora_b_fails_instead_of_silently_skipping_ssr():
    model = torch.nn.Linear(3, 3)

    with pytest.raises(RuntimeError, match="No two-dimensional LoRA-B"):
        LoRABasisSSR(0.1).apply(model)
    with pytest.raises(RuntimeError, match="before LoRA-B existed"):
        summarize_lora_geometry(model)


@pytest.mark.parametrize("coefficient", [-1.0, -1e-6])
def test_negative_coefficients_are_rejected(coefficient: float):
    with pytest.raises(ValueError, match="non-negative"):
        LoRABasisSSR(coefficient)
