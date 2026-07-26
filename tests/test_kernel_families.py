from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from methods.biocs_plus import BioCsPlus
from methods.bioreg import BioReg, _radial_response, compute_spatial_biocs


FAMILIES = ("gaussian", "laplace", "cauchy", "inverse")


def legacy_gaussian_loss(
    weights: torch.Tensor,
    A_exc: float = 1.0,
    A_inh: float = 0.8,
    sigma_exc: float = 0.2,
    sigma_inh: float = 0.5,
    seen_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    weights = weights.float()
    if seen_mask is not None:
        weights = weights[seen_mask]
    n = weights.size(0)
    if n < 2:
        return torch.tensor(0.0, device=weights.device)
    w_norm = F.normalize(weights, dim=1)
    cos_sim = torch.clamp(w_norm @ w_norm.T, -1.0, 1.0)
    dist = torch.sqrt(torch.clamp(1.0 - cos_sim, min=1e-8))
    exc = A_exc * torch.exp(-(dist ** 2) / (2 * sigma_exc ** 2))
    inh = A_inh * torch.exp(-(dist ** 2) / (2 * sigma_inh ** 2))
    penalty = (inh - exc) + (A_exc - A_inh)
    penalty = penalty - torch.diag(torch.diag(penalty))
    return penalty.sum() / (n * (n - 1) + 1e-8)


def test_default_gaussian_matches_legacy_loss_and_gradient() -> None:
    torch.manual_seed(7)
    new_weights = torch.randn(9, 13, requires_grad=True)
    old_weights = new_weights.detach().clone().requires_grad_(True)

    new_loss = compute_spatial_biocs(new_weights)
    old_loss = legacy_gaussian_loss(old_weights)
    new_grad = torch.autograd.grad(new_loss, new_weights)[0]
    old_grad = torch.autograd.grad(old_loss, old_weights)[0]

    torch.testing.assert_close(new_loss, old_loss, rtol=0.0, atol=0.0)
    torch.testing.assert_close(new_grad, old_grad, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("kernel_family", FAMILIES)
def test_kernel_families_have_finite_loss_and_gradient(kernel_family: str) -> None:
    torch.manual_seed(11)
    weights = torch.randn(12, 17, requires_grad=True)
    loss = compute_spatial_biocs(weights, kernel_family=kernel_family)
    grad = torch.autograd.grad(loss, weights)[0]

    assert torch.isfinite(loss)
    assert torch.isfinite(grad).all()


def _nearly_collinear_weights(near_identical: bool) -> torch.Tensor:
    torch.manual_seed(13)
    base = torch.randn(1, 16)
    weights = base.repeat(6, 1)
    if near_identical:
        direction = torch.randn_like(base)
        direction = direction - (direction * base).sum() / base.square().sum() * base
        offsets = torch.linspace(-1.0, 1.0, steps=6).unsqueeze(1)
        weights = weights + 1e-3 * offsets * direction
    return weights.requires_grad_(True)


@pytest.mark.parametrize("kernel_family", ("laplace", "cauchy", "inverse"))
@pytest.mark.parametrize("near_identical", (False, True))
@pytest.mark.parametrize("use_cpu_autocast", (False, True))
def test_non_gaussian_degenerate_rows_have_finite_gradients(
    kernel_family: str,
    near_identical: bool,
    use_cpu_autocast: bool,
) -> None:
    weights = _nearly_collinear_weights(near_identical)
    if use_cpu_autocast:
        with torch.amp.autocast("cpu", dtype=torch.bfloat16):
            loss = compute_spatial_biocs(weights, kernel_family=kernel_family)
    else:
        loss = compute_spatial_biocs(weights, kernel_family=kernel_family)
    grad = torch.autograd.grad(loss, weights)[0]

    assert torch.isfinite(loss)
    assert torch.isfinite(grad).all()


def test_inverse_response_is_decreasing_and_convex() -> None:
    sigma = 0.7
    dist = torch.linspace(0.0, 1.4, steps=101, dtype=torch.float64)
    response = _radial_response(dist, sigma, "inverse")

    assert response[0] == 1.0
    torch.testing.assert_close(
        _radial_response(torch.tensor(sigma), sigma, "inverse"),
        torch.tensor(0.5),
    )
    assert (torch.diff(response) <= 0).all()
    assert (torch.diff(response, n=2) >= -1e-12).all()


@pytest.mark.parametrize("kernel_family", FAMILIES)
def test_cub_loss_matches_shared_kernel(kernel_family: str) -> None:
    from experiments.cub200_continual_benchmark import biocs_loss

    torch.manual_seed(17)
    weights = torch.randn(7, 11)
    shared = compute_spatial_biocs(
        weights,
        1.1,
        0.7,
        0.3,
        0.8,
        kernel_family=kernel_family,
    )
    cub = biocs_loss(
        weights,
        1.1,
        0.7,
        0.3,
        0.8,
        kernel_family=kernel_family,
    )
    torch.testing.assert_close(cub, shared, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("kernel_family", FAMILIES)
def test_mask_permutation_and_positive_scale_invariance(kernel_family: str) -> None:
    torch.manual_seed(19)
    weights = torch.randn(10, 8)
    mask = torch.tensor([True, False, True, True, False, True, False, True, True, False])
    selected = weights[mask]
    permutation = torch.tensor([3, 0, 5, 2, 1, 4])
    scales = torch.tensor([0.3, 0.7, 1.0, 1.5, 2.0, 4.0]).unsqueeze(1)

    masked_loss = compute_spatial_biocs(
        weights, seen_mask=mask, kernel_family=kernel_family,
    )
    direct_loss = compute_spatial_biocs(selected, kernel_family=kernel_family)
    permuted_loss = compute_spatial_biocs(
        selected[permutation], kernel_family=kernel_family,
    )
    scaled_loss = compute_spatial_biocs(
        selected * scales, kernel_family=kernel_family,
    )

    torch.testing.assert_close(masked_loss, direct_loss)
    torch.testing.assert_close(permuted_loss, direct_loss, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(scaled_loss, direct_loss, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"kernel_family": "triangular"}, ValueError),
        ({"kernel_family": 3}, TypeError),
        ({"A_exc": -0.1}, ValueError),
        ({"A_inh": float("inf")}, ValueError),
        ({"sigma_exc": 0.0}, ValueError),
        ({"sigma_inh": float("nan")}, ValueError),
    ],
)
def test_invalid_kernel_parameters_fail_fast(kwargs: dict, error: type[Exception]) -> None:
    with pytest.raises(error):
        compute_spatial_biocs(torch.randn(4, 5), **kwargs)


def test_bioreg_passes_configured_kernel_family() -> None:
    torch.manual_seed(23)
    model = nn.Sequential(nn.Linear(6, 4, bias=False))
    learner = BioReg(
        model,
        torch.device("cpu"),
        {
            "regularization_mode": "spatial_biocs",
            "lambda_spatial": 1.0,
            "A_exc": 1.1,
            "A_inh": 0.7,
            "sigma_exc": 0.3,
            "sigma_inh": 0.8,
            "kernel_family": "cauchy",
            "biocs_targets": ["classifier"],
            "use_seen_mask": False,
        },
    )

    actual = learner._spatial_penalty()
    expected = compute_spatial_biocs(
        model[0].weight,
        1.1,
        0.7,
        0.3,
        0.8,
        kernel_family="cauchy",
    )
    assert learner.kernel_family == "cauchy"
    torch.testing.assert_close(actual, expected)


def test_biocs_plus_passes_configured_kernel_family() -> None:
    torch.manual_seed(29)
    model = nn.Linear(5, 4, bias=False)
    learner = BioCsPlus(
        model,
        torch.device("cpu"),
        {
            "lambda_spatial": 1.0,
            "lambda_distill": 0.0,
            "lambda_replay": 0.0,
            "buffer_size": 0,
            "A_exc": 1.0,
            "A_inh": 0.6,
            "sigma_exc": 0.4,
            "sigma_inh": 0.9,
            "kernel_family": "laplace",
            "biocs_targets": ["classifier"],
        },
    )
    x = torch.randn(8, 5)
    y = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])

    loss, logits = learner.training_step(x, y, task_id=0)
    expected = F.cross_entropy(logits, y) + compute_spatial_biocs(
        model.weight,
        1.0,
        0.6,
        0.4,
        0.9,
        kernel_family="laplace",
    )
    assert learner.kernel_family == "laplace"
    torch.testing.assert_close(loss, expected)
