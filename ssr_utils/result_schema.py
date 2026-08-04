"""Validation and hashing helpers for per-run SSR result records."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "1.0"
TASK_FAMILIES = {"editing", "classification", "segmentation", "adapter"}
DISTANCE_MAPPINGS = {"cosine", "projective", "none"}
OBJECTIVE_COMPONENTS = (
    "task",
    "kd",
    "ssr",
    "anchor",
    "spectral",
    "ewc",
    "mas",
    "si",
    "center",
    "protodecor",
)
KNOWN_RECIPE_COMPONENTS = {
    "plain": {"task"},
    "anchor": {"task", "anchor"},
    "spectral": {"task", "spectral"},
    "stabilized": {"task", "anchor", "spectral"},
    "ssr_only": {"task", "ssr"},
    "ssr_anchor": {"task", "ssr", "anchor"},
    "ssr_spectral": {"task", "ssr", "spectral"},
    "full": {"task", "ssr", "anchor", "spectral"},
    "kd": {"task", "kd"},
    "kd_ewc": {"task", "kd", "ewc"},
    "kd_mas": {"task", "kd", "mas"},
    "kd_si": {"task", "kd", "si"},
    "kd_center": {"task", "kd", "center"},
    "kd_protodecor": {"task", "kd", "protodecor"},
    "kd_spectral": {"task", "kd", "spectral"},
    "kd_ssr": {"task", "kd", "ssr"},
    "ssr_kd": {"task", "kd", "ssr"},
}
KD_MATCHED_REGULARIZERS = {
    "kd": "none",
    "kd_ewc": "ewc",
    "kd_mas": "mas",
    "kd_si": "si",
    "kd_center": "center",
    "kd_protodecor": "protodecor",
    "kd_spectral": "spectral",
    "kd_ssr": "ssr",
}
REQUIRED_FIELDS = {
    "schema_version",
    "git_commit",
    "run_id",
    "task_family",
    "dataset",
    "model",
    "seed",
    "objective",
    "distance_mapping",
    "kernel",
    "metrics",
    "runtime",
    "config_hash",
    "dataset_hash",
    "recipe",
}


class ResultSchemaError(ValueError):
    """Raised when a run record is incomplete or internally inconsistent."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_objective(objective: Mapping[str, Any]) -> dict[str, bool]:
    unknown = set(objective) - set(OBJECTIVE_COMPONENTS)
    if unknown:
        raise ResultSchemaError(f"Unknown objective components: {sorted(unknown)}")
    normalized = {name: bool(objective.get(name, False)) for name in OBJECTIVE_COMPONENTS}
    if not normalized["task"]:
        raise ResultSchemaError("Every training result must include the task objective")
    return normalized


def _active_components(objective: Mapping[str, bool]) -> set[str]:
    return {name for name, enabled in objective.items() if enabled}


def _validate_known_recipe(recipe: Any, objective: Mapping[str, bool]) -> None:
    if not isinstance(recipe, str) or not recipe.strip():
        raise ResultSchemaError("recipe must be a non-empty string")
    expected = KNOWN_RECIPE_COMPONENTS.get(recipe)
    if expected is None:
        return
    actual = _active_components(objective)
    if actual != expected:
        raise ResultSchemaError(
            f"Recipe {recipe!r} requires objective components {sorted(expected)}, "
            f"got {sorted(actual)}"
        )


def _validate_kd_matched_config(recipe: str, config: Mapping[str, Any]) -> None:
    method = config.get("method")
    if not isinstance(method, Mapping):
        return
    method_name = str(method.get("name", "")).lower()
    expected_regularizer = KD_MATCHED_REGULARIZERS.get(method_name)
    if expected_regularizer is None:
        return
    if recipe in KD_MATCHED_REGULARIZERS and recipe != method_name:
        raise ResultSchemaError(
            f"Result recipe {recipe!r} contradicts config method.name={method_name!r}"
        )
    if method.get("regularizer") is not None:
        regularizer = str(method["regularizer"]).lower()
        if regularizer != expected_regularizer:
            raise ResultSchemaError(
                f"Config method.name={method_name!r} requires "
                f"regularizer={expected_regularizer!r}, got {regularizer!r}"
            )


def _validate_editing_completion(
    record: Mapping[str, Any],
    *,
    allow_incomplete: bool,
) -> None:
    for name in ("requested", "attempted", "succeeded", "failed"):
        value = record.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ResultSchemaError(f"Editing field {name!r} must be a non-negative integer")
    requested = record["requested"]
    attempted = record["attempted"]
    succeeded = record["succeeded"]
    failed = record["failed"]
    if requested <= 0:
        raise ResultSchemaError("Editing field 'requested' must be positive")
    if attempted > requested:
        raise ResultSchemaError("Editing attempted count cannot exceed requested count")
    if succeeded + failed != attempted:
        raise ResultSchemaError("Editing succeeded + failed must equal attempted")

    expected_status = "complete" if (
        attempted == requested and succeeded == requested and failed == 0
    ) else ("failed" if attempted > 0 and succeeded == 0 else "incomplete")
    if record.get("status") != expected_status:
        raise ResultSchemaError(
            f"Editing status must be {expected_status!r} for the recorded completion counts"
        )
    for name in (
        "evaluator",
        "evaluator_version",
        "evaluation_protocol_hash",
        "model_hash",
        "pairing_protocol_hash",
    ):
        if not isinstance(record.get(name), str) or not record[name].strip():
            raise ResultSchemaError(f"Editing field {name!r} must be a non-empty string")
    for name in ("evaluation_protocol_hash", "model_hash", "pairing_protocol_hash"):
        if len(record[name]) != 64 or any(character not in "0123456789abcdef" for character in record[name]):
            raise ResultSchemaError(f"Editing field {name!r} must be a lowercase SHA256")
    if not allow_incomplete and expected_status != "complete":
        raise ResultSchemaError(
            "Official editing results require succeeded == requested with no failed edits"
        )


def validate_result_record(
    record: Mapping[str, Any],
    *,
    allow_incomplete: bool = False,
) -> dict[str, Any]:
    missing = REQUIRED_FIELDS - set(record)
    if missing:
        raise ResultSchemaError(f"Missing required result fields: {sorted(missing)}")
    if record["schema_version"] != SCHEMA_VERSION:
        raise ResultSchemaError(
            f"Unsupported schema_version={record['schema_version']!r}; expected {SCHEMA_VERSION!r}"
        )
    if record["task_family"] not in TASK_FAMILIES:
        raise ResultSchemaError(f"Unknown task_family={record['task_family']!r}")
    if record["distance_mapping"] not in DISTANCE_MAPPINGS:
        raise ResultSchemaError(f"Unknown distance_mapping={record['distance_mapping']!r}")
    if not isinstance(record["seed"], int) or isinstance(record["seed"], bool):
        raise ResultSchemaError("seed must be an integer")
    for name in ("git_commit", "run_id", "dataset", "model", "config_hash", "dataset_hash"):
        if not isinstance(record[name], str) or not record[name].strip():
            raise ResultSchemaError(f"{name} must be a non-empty string")
    normalized = dict(record)
    normalized["objective"] = normalize_objective(record["objective"])
    _validate_known_recipe(record["recipe"], normalized["objective"])
    if normalized["objective"]["ssr"] and record["distance_mapping"] == "none":
        raise ResultSchemaError("An SSR objective requires cosine or projective distance mapping")
    if not normalized["objective"]["ssr"] and record["distance_mapping"] != "none":
        raise ResultSchemaError("Non-SSR objectives must use distance_mapping='none'")
    if not isinstance(record["kernel"], Mapping):
        raise ResultSchemaError("kernel must be an object")
    if normalized["objective"]["ssr"]:
        for field in ("family", "A_exc", "A_inh", "sigma_exc", "sigma_inh"):
            if field not in record["kernel"]:
                raise ResultSchemaError(f"SSR kernel is missing {field!r}")
        for field in ("sigma_exc", "sigma_inh"):
            if float(record["kernel"][field]) <= 0:
                raise ResultSchemaError(f"kernel.{field} must be positive")
    if not isinstance(record["metrics"], Mapping) or not record["metrics"]:
        raise ResultSchemaError("metrics must be a non-empty object")
    if not isinstance(record["runtime"], Mapping):
        raise ResultSchemaError("runtime must be an object")
    if record["task_family"] == "editing":
        _validate_editing_completion(record, allow_incomplete=allow_incomplete)
    return normalized


def build_result_record(
    *,
    git_commit: str,
    run_id: str,
    task_family: str,
    dataset: str,
    model: str,
    seed: int,
    objective: Mapping[str, Any],
    distance_mapping: str,
    kernel: Mapping[str, Any],
    metrics: Mapping[str, Any],
    runtime: Mapping[str, Any],
    config: Mapping[str, Any],
    dataset_hash: str,
    allow_incomplete: bool = False,
    **extra: Any,
) -> dict[str, Any]:
    record = {
        "schema_version": SCHEMA_VERSION,
        "git_commit": git_commit,
        "run_id": run_id,
        "task_family": task_family,
        "dataset": dataset,
        "model": model,
        "seed": seed,
        "objective": dict(objective),
        "distance_mapping": distance_mapping,
        "kernel": dict(kernel),
        "metrics": dict(metrics),
        "runtime": dict(runtime),
        "config_hash": sha256_value(config),
        "dataset_hash": dataset_hash,
        **extra,
    }
    recipe = record.get("recipe")
    if isinstance(recipe, str):
        _validate_kd_matched_config(recipe, config)
    return validate_result_record(record, allow_incomplete=allow_incomplete)
