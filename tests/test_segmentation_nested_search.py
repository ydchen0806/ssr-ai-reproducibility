from __future__ import annotations

import math

import pytest
import torch

from ssr_utils.segmentation_nested_search import (
    candidate_matrix,
    cross_set_ssr_loss,
    hwhm_to_scale,
    lock_selection,
    select_top,
    shard_candidates,
    validate_paired_identities,
    stratified_fit_validation_indices,
)


KERNELS = ("gaussian", "laplace", "cauchy")
WIDTHS = ((0.12, 0.36), (0.18, 0.50), (0.24, 0.70))
LAMBDAS = (0.003, 0.01, 0.03)
TARGET_SCOPES = (("class", "new_old"), ("channels", "all"))


def _candidates():
    return candidate_matrix(KERNELS, WIDTHS, LAMBDAS, TARGET_SCOPES)


def test_new_old_ssr_only_differentiates_the_plastic_rows():
    current = torch.tensor(
        [[0.9, 0.1, -0.2], [0.2, 0.8, 0.3]],
        dtype=torch.float64,
        requires_grad=True,
    )
    detached_references = torch.tensor(
        [[0.7, -0.4, 0.2], [-0.3, 0.5, 0.9], [0.1, -0.8, 0.6]],
        dtype=torch.float64,
        requires_grad=True,
    )

    loss = cross_set_ssr_loss(
        current,
        detached_references,
        kernel="gaussian",
        hwhm_exc=0.16,
        hwhm_inh=0.45,
        a_exc=1.0,
        a_inh=0.8,
    )
    loss.backward()

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert current.grad is not None
    assert torch.isfinite(current.grad).all()
    assert torch.count_nonzero(current.grad) > 0
    assert detached_references.grad is None


@pytest.mark.parametrize(
    ("family", "expected_scale"),
    [
        ("gaussian", 1.0 / math.sqrt(2.0 * math.log(2.0))),
        ("laplace", 1.0 / math.log(2.0)),
        ("cauchy", 1.0),
    ],
)
def test_hwhm_conversion_matches_each_radial_kernel(family, expected_scale):
    half_width = 0.37
    sigma = hwhm_to_scale(family, half_width)
    assert sigma == pytest.approx(half_width * expected_scale)

    distance = torch.tensor(half_width, dtype=torch.float64)
    if family == "gaussian":
        response = torch.exp(-(distance**2) / (2.0 * sigma**2))
    elif family == "laplace":
        response = torch.exp(-distance / sigma)
    else:
        response = 1.0 / (1.0 + (distance / sigma) ** 2)
    assert response.item() == pytest.approx(0.5)


def test_hwhm_conversion_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="positive"):
        hwhm_to_scale("gaussian", 0.0)
    with pytest.raises(ValueError, match="kernel"):
        hwhm_to_scale("inverse", 0.2)


def test_nested_search_matrix_is_bounded_and_has_no_duplicate_candidates():
    candidates = _candidates()

    assert len(candidates) == 3 * 3 * 3 * 2 == 54
    assert len(candidates) <= 54
    assert len({candidate.spec_id for candidate in candidates}) == len(candidates)
    assert {candidate.kernel for candidate in candidates} == {
        "gaussian",
        "laplace",
        "cauchy",
    }
    assert len({(candidate.hwhm_exc, candidate.hwhm_inh) for candidate in candidates}) == 3
    assert len({candidate.lambda_ssr for candidate in candidates}) == 3
    assert len({(candidate.target, candidate.scope) for candidate in candidates}) == 2


def test_candidate_shards_are_deterministic_complete_and_disjoint():
    candidates = _candidates()
    first = [shard_candidates(candidates, index, 4) for index in range(4)]
    repeated = [shard_candidates(candidates, index, 4) for index in range(4)]

    first_ids = [[candidate.spec_id for candidate in shard] for shard in first]
    repeated_ids = [
        [candidate.spec_id for candidate in shard] for shard in repeated
    ]
    assert first_ids == repeated_ids

    shard_sets = [set(candidate_ids) for candidate_ids in first_ids]
    for left_index, left in enumerate(shard_sets):
        for right in shard_sets[left_index + 1 :]:
            assert left.isdisjoint(right)
    assert set().union(*shard_sets) == {
        candidate.spec_id for candidate in candidates
    }
    assert max(map(len, first)) - min(map(len, first)) <= 1


def _development_rows(candidates):
    return [
        {
            "stage": "refine",
            "spec_id": candidate.spec_id,
            "selection_score": float(index),
            "seed": 9101,
        }
        for index, candidate in enumerate(candidates[:8])
    ]


def test_selection_returns_top_four_using_development_rows_only():
    candidates = _candidates()
    rows = _development_rows(candidates)

    selected = select_top(rows, k=4, stage="refine")

    assert [row["spec_id"] for row in selected] == [
        candidates[index].spec_id for index in (7, 6, 5, 4)
    ]


def test_selection_and_lock_reject_confirmation_information():
    candidates = _candidates()
    development_rows = _development_rows(candidates)
    confirmation_row = {
        "stage": "confirmation",
        "spec_id": candidates[0].spec_id,
        "selection_score": 1e9,
        "seed": 9501,
    }

    selected = select_top(
        [*development_rows, confirmation_row],
        k=4,
        stage="refine",
    )
    assert candidates[0].spec_id not in {row["spec_id"] for row in selected}

    selected = select_top(development_rows, k=4, stage="refine")
    lock = lock_selection(selected)
    assert lock["selected_spec_id"] == selected[0]["spec_id"]
    assert "confirmation" not in repr(lock).lower()

    with pytest.raises(ValueError, match="confirmation|development"):
        lock_selection(
            selected,
            confirmation_results=[confirmation_row],
        )


def _paired_record(*, ssr: bool) -> dict:
    return {
        "git_commit": "a" * 40,
        "task_family": "segmentation",
        "dataset": "oxford_flowers102_masks",
        "model": "resnet18_dense_decoder",
        "seed": 9501,
        "dataset_hash": "b" * 64,
        "initial_model_hash": "c" * 64,
        "task_schedule_hash": "d" * 64,
        "optimizer_steps": 120,
        "selection_lock_hash": "e" * 64,
        "objective": {
            "task": True,
            "kd": True,
            "ssr": ssr,
            "anchor": False,
            "spectral": False,
            "ewc": False,
            "mas": False,
            "si": False,
            "center": False,
            "protodecor": False,
        },
    }


def test_paired_identity_allows_only_the_declared_ssr_delta():
    control = _paired_record(ssr=False)
    treatment = _paired_record(ssr=True)
    validate_paired_identities([control], [treatment])

    for field, mismatch in (
        ("seed", 9503),
        ("optimizer_steps", 121),
        ("initial_model_hash", "f" * 64),
        ("selection_lock_hash", "0" * 64),
    ):
        invalid = dict(treatment)
        invalid[field] = mismatch
        with pytest.raises(ValueError, match=field):
            validate_paired_identities([control], [invalid])


def test_paired_identity_rejects_different_inventory_lengths():
    with pytest.raises(ValueError, match="inventor"):
        validate_paired_identities([_paired_record(ssr=False)], [])


def test_stratified_validation_split_is_deterministic_and_disjoint():
    labels = torch.arange(4).repeat_interleave(10)
    first_fit, first_validation = stratified_fit_validation_indices(
        labels, validation_fraction=0.2, seed=17
    )
    second_fit, second_validation = stratified_fit_validation_indices(
        labels, validation_fraction=0.2, seed=17
    )

    assert torch.equal(first_fit, second_fit)
    assert torch.equal(first_validation, second_validation)
    assert len(first_fit) == 32
    assert len(first_validation) == 8
    assert set(first_fit.tolist()).isdisjoint(first_validation.tolist())
    assert set(first_fit.tolist()) | set(first_validation.tolist()) == set(range(40))
    assert torch.bincount(labels[first_validation]).tolist() == [2, 2, 2, 2]
