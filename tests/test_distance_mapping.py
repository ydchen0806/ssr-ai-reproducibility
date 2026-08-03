from __future__ import annotations

import pytest
import torch

from llm_ke.biocs_editor import (
    compute_spatial_biocs_llm,
    fixed_random_row_indices,
    pairwise_distance,
)


def _weights() -> torch.Tensor:
    return torch.tensor(
        [
            [1.0, 0.2, -0.1, 0.4],
            [0.3, 0.9, 0.5, -0.2],
            [-0.4, 0.1, 0.8, 0.6],
            [0.7, -0.5, 0.2, 0.3],
        ],
        dtype=torch.float32,
    )


def test_projective_mapping_is_sign_symmetric_but_cosine_is_not():
    weights = _weights()
    sign_flipped = weights.clone()
    sign_flipped[0].mul_(-1)

    projective = compute_spatial_biocs_llm(weights, distance_metric="projective")
    projective_flipped = compute_spatial_biocs_llm(
        sign_flipped,
        distance_metric="projective",
    )
    cosine = compute_spatial_biocs_llm(weights, distance_metric="cosine")
    cosine_flipped = compute_spatial_biocs_llm(sign_flipped, distance_metric="cosine")

    assert projective_flipped == pytest.approx(projective.item(), abs=1e-7)
    assert not torch.isclose(cosine, cosine_flipped, atol=1e-6, rtol=1e-6)


def test_public_pairwise_distance_has_expected_sign_symmetry():
    axis = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
    assert pairwise_distance(axis, "projective")[0, 1] < 1e-3
    assert pairwise_distance(axis, "cosine")[0, 1] > 1.0


@pytest.mark.parametrize("distance_mapping", ["cosine", "projective"])
def test_distance_mapping_has_finite_value_and_gradients(distance_mapping: str):
    weights = _weights().requires_grad_(True)
    penalty = compute_spatial_biocs_llm(
        weights,
        distance_metric=distance_mapping,
    )
    penalty.backward()

    assert torch.isfinite(penalty)
    assert weights.grad is not None
    assert torch.isfinite(weights.grad).all()


def test_single_row_has_zero_penalty_because_diagonal_is_excluded():
    weights = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    penalty = compute_spatial_biocs_llm(weights, distance_metric="cosine")
    penalty.backward()

    assert penalty.item() == 0.0
    assert torch.equal(weights.grad, torch.zeros_like(weights))


def test_fixed_random_sampling_is_deterministic_without_a_model():
    first = fixed_random_row_indices(
        "transformer.h.7.mlp.c_proj",
        128,
        16,
        sampler_seed=31,
        edit_index=4,
    )
    repeated = fixed_random_row_indices(
        "transformer.h.7.mlp.c_proj",
        128,
        16,
        sampler_seed=31,
        edit_index=4,
    )
    next_edit = fixed_random_row_indices(
        "transformer.h.7.mlp.c_proj",
        128,
        16,
        sampler_seed=31,
        edit_index=5,
    )

    assert torch.equal(first, repeated)
    assert not torch.equal(first, next_edit)
    assert first.device.type == "cpu"
    assert first.unique().numel() == 16
