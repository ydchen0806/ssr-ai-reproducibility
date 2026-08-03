"""Versioned locality evaluation shared by the custom FT/SSR editors."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from typing import Any


# Keep this identifier stable for the historical substring metric.  A future
# semantic or token-level evaluator must use a new version rather than silently
# changing the meaning of existing locality numbers.
LOCALITY_EVALUATOR = "custom_substring_any_ground_truth_all_items"
LOCALITY_EVALUATOR_VERSION = "1.0"


def normalize_text(value: Any) -> str:
    """Normalize a KnowEdit text-like value to one display string."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        for key in ("str", "text", "answer", "label", "name"):
            if key in value:
                return normalize_text(value[key])
        return str(dict(value))
    if isinstance(value, (list, tuple)):
        return normalize_text(value[0]) if value else ""
    return str(value)


def normalize_text_list(value: Any) -> list[str]:
    """Flatten nested KnowEdit answers while retaining every non-empty alias."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        texts: list[str] = []
        for item in value:
            texts.extend(normalize_text_list(item))
        return [text for text in texts if text]
    text = normalize_text(value)
    return [text] if text else []


def iter_locality_items(locality: Any) -> Iterator[dict[str, Any]]:
    """Yield valid locality prompts in stable group/item order.

    KnowEdit releases use a mapping of locality group names to lists, while a
    few derived files store a single item directly.  Supporting both shapes is
    useful for auditing and does not change the scoring rule.
    """
    if not isinstance(locality, Mapping):
        return
    for group, raw_items in locality.items():
        items = [raw_items] if isinstance(raw_items, Mapping) else raw_items
        if not isinstance(items, (list, tuple)):
            continue
        for item_index, item in enumerate(items):
            if not isinstance(item, Mapping):
                continue
            if "prompt" not in item or "ground_truth" not in item:
                continue
            prompt = normalize_text(item["prompt"])
            if not prompt:
                continue
            yield {
                "group": str(group),
                "item_index": item_index,
                "prompt": prompt,
                "ground_truths": normalize_text_list(item["ground_truth"]),
            }


def first_locality_item(locality: Any) -> dict[str, Any] | None:
    """Return the first normalized item for APIs limited to one item per edit."""
    return next(iter_locality_items(locality), None)


def evaluate_locality(
    locality: Any,
    generate: Callable[[str], str],
) -> dict[str, Any]:
    """Evaluate all locality items with the historical substring-any rule.

    Empty locality retains the historical score of zero, but ``n_items=0`` is
    recorded so downstream analyses can distinguish missing probes from failed
    probes.
    """
    evaluations = []
    for item in iter_locality_items(locality):
        prediction = normalize_text(generate(item["prompt"])).strip()
        prediction_lower = prediction.lower()
        matched = any(
            ground_truth.lower() in prediction_lower
            for ground_truth in item["ground_truths"]
        )
        evaluations.append(
            {
                **item,
                "prediction": prediction,
                "matched": bool(matched),
            }
        )

    n_items = len(evaluations)
    score = sum(float(item["matched"]) for item in evaluations) / max(n_items, 1)
    return {
        "score": score,
        "n_items": n_items,
        "n_matched": sum(int(item["matched"]) for item in evaluations),
        "evaluator": LOCALITY_EVALUATOR,
        "evaluator_version": LOCALITY_EVALUATOR_VERSION,
        "evaluations": evaluations,
    }
