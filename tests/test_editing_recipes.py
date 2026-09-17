from __future__ import annotations

from argparse import Namespace

import pytest

from llm_ke.biocs_editor import RECIPES, infer_recipe, resolve_recipe
from scripts.run_llm_ke_easyedit import resolve_biocs_run_config
from ssr_utils.result_schema import KNOWN_RECIPE_COMPONENTS


EXPECTED_RECIPES = {
    "plain": {"ssr": False, "anchor": False, "spectral": False},
    "anchor": {"ssr": False, "anchor": True, "spectral": False},
    "spectral": {"ssr": False, "anchor": False, "spectral": True},
    "stabilized": {"ssr": False, "anchor": True, "spectral": True},
    "ssr_only": {"ssr": True, "anchor": False, "spectral": False},
    "ssr_anchor": {"ssr": True, "anchor": True, "spectral": False},
    "ssr_spectral": {"ssr": True, "anchor": False, "spectral": True},
    "full": {"ssr": True, "anchor": True, "spectral": True},
}


def _args(**overrides) -> Namespace:
    values = {
        "recipe": None,
        "distance_mapping": None,
        "seed": None,
        "lambda_ssr": None,
        "lambda_anchor": None,
        "lambda_spectral": None,
        "biocs_lambda": None,
        "biocs_anchor": None,
    }
    values.update(overrides)
    return Namespace(**values)


def test_recipe_registry_covers_the_factorial_objectives():
    assert RECIPES == EXPECTED_RECIPES
    for recipe, flags in RECIPES.items():
        expected = {"task", *(name for name, enabled in flags.items() if enabled)}
        assert KNOWN_RECIPE_COMPONENTS[recipe] == expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("plain", {"task"}),
        ("stabilized", {"task", "anchor", "spectral"}),
        ("ssr_only", {"task", "ssr"}),
        ("full", {"task", "ssr", "anchor", "spectral"}),
    ],
)
def test_explicit_recipe_records_and_gates_components(name: str, expected: set[str]):
    resolved = resolve_recipe(
        name,
        lambda_ssr=0.7,
        lambda_anchor=0.5,
        lambda_spectral=0.3,
    )

    assert resolved["recipe"] == name
    assert set(resolved["components"]) == expected
    assert {key for key, enabled in resolved["objective"].items() if enabled} == expected
    assert (resolved["lambda_ssr"] > 0) is ("ssr" in expected)
    assert (resolved["lambda_anchor"] > 0) is ("anchor" in expected)
    assert (resolved["lambda_spectral"] > 0) is ("spectral" in expected)


def test_legacy_coefficients_infer_the_same_named_recipe():
    assert infer_recipe(0.2, 0.0, 0.4) == "ssr_spectral"
    resolved = resolve_recipe(
        None,
        lambda_ssr=0.2,
        lambda_anchor=0.0,
        lambda_spectral=0.4,
    )
    assert resolved["recipe"] == "ssr_spectral"
    assert resolved["components"] == ["task", "ssr", "spectral"]


def test_explicit_recipe_rejects_an_inactive_claimed_component():
    with pytest.raises(ValueError, match="coefficients must be positive"):
        resolve_recipe(
            "ssr_only",
            lambda_ssr=0.0,
            lambda_anchor=0.0,
            lambda_spectral=0.0,
        )


def test_cli_values_override_new_and_legacy_environment_names():
    config = resolve_biocs_run_config(
        _args(
            recipe="ssr_anchor",
            distance_mapping="projective",
            seed=17,
            lambda_ssr=0.7,
            lambda_anchor=0.5,
            lambda_spectral=0.3,
        ),
        {
            "BIOCS_RECIPE": "full",
            "BIOCS_DISTANCE_MAPPING": "cosine",
            "BIOCS_LAMBDA_SSR": "0.11",
            "BIOCS_LAMBDA": "0.12",
            "BIOCS_LAMBDA_ANCHOR": "0.13",
            "BIOCS_LAMBDA_SPECTRAL": "0.14",
            "BIOCS_SAMPLER_SEED": "99",
            "KE_SEED": "98",
        },
    )

    assert config["recipe"] == "ssr_anchor"
    assert config["distance_mapping"] == "projective"
    assert config["seed"] == 17
    assert config["sampler_seed"] == 17
    assert config["lambda_ssr"] == pytest.approx(0.7)
    assert config["lambda_anchor"] == pytest.approx(0.5)
    assert config["lambda_spectral"] == 0.0


def test_legacy_environment_remains_supported():
    config = resolve_biocs_run_config(
        _args(),
        {
            "BIOCS_LAMBDA": "0.25",
            "BIOCS_LAMBDA_ANCHOR": "0",
            "BIOCS_LAMBDA_SPECTRAL": "0",
            "BIOCS_DISTANCE_METRIC": "projective",
            "KE_SEED": "23",
        },
    )

    assert config["recipe"] == "ssr_only"
    assert config["lambda_ssr"] == pytest.approx(0.25)
    assert config["distance_mapping"] == "projective"
    assert config["seed"] == 23
    assert config["sampler_seed"] == 23


def test_conflicting_new_and_deprecated_cli_coefficients_fail_fast():
    with pytest.raises(ValueError, match="Conflicting CLI values"):
        resolve_biocs_run_config(
            _args(lambda_ssr=0.2, biocs_lambda=0.1),
            {},
        )
