"""Spatial Synaptic Regularization (SSR) for LLM knowledge editing.

Applies biologically-inspired center-surround regularization to LLM weight
editing. When fine-tuning specific MLP layers to inject new knowledge, SSR
constrains the weight update so that "nearby" neurons suppress each other's
growth — preserving existing knowledge (locality) while allowing targeted edits
(efficacy).

The released target-module discovery supports GPT-2- and LLaMA-style causal
LMs whose editable MLP projections follow the documented layer/module naming
pattern. Other architectures require an explicit target-module regex or a
small discovery adapter. Uses EasyEdit's KnowEdit dataset format for
evaluation.
"""

import json
import logging
import argparse
import hashlib
import os
import re
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

try:
    from .locality import (
        LOCALITY_EVALUATOR,
        LOCALITY_PROTOCOL_HASH,
        LOCALITY_EVALUATOR_VERSION,
        assert_legacy_compatible_locality,
        evaluate_locality,
        normalize_text,
        normalize_text_list,
    )
except ImportError:  # Support ``python llm_ke/biocs_editor.py``.
    from locality import (  # type: ignore
        LOCALITY_EVALUATOR,
        LOCALITY_PROTOCOL_HASH,
        LOCALITY_EVALUATOR_VERSION,
        assert_legacy_compatible_locality,
        evaluate_locality,
        normalize_text,
        normalize_text_list,
    )

logger = logging.getLogger(__name__)


def _to_text(value) -> str:
    """Normalize KnowEdit answer fields to plain strings."""
    return normalize_text(value)


def _to_text_list(value) -> list[str]:
    """Normalize nested KnowEdit labels into a flat list of strings."""
    return normalize_text_list(value)


# The original study reported a post-edit ground-truth substring score as
# ``locality``.  That quantity is retained below for backwards-compatible
# diagnostics, but it cannot measure preservation because it has no pre-edit
# reference.  The primary preservation readout for the long-stream protocol is
# a frozen, pre-edit next-token distribution evaluated at every checkpoint.
# Exact 32-token greedy continuations saturate at zero after a long edit stream
# and cannot distinguish preservation.  This metric keeps the prompt and model
# fixed, then compares the pre/post predictive distributions at the next token.
PRE_EDIT_EVALUATOR = "pre_edit_next_token_topk_js_similarity"
PRE_EDIT_EVALUATOR_VERSION = "2.0"
PRE_EDIT_TOP_K = 32
PRE_EDIT_PROTOCOL_SPEC = {
    "reference": "pre-edit next-token distribution at the final prompt position",
    "compression": "top-32 token probabilities plus one residual-mass bin",
    "probe_selection": "fixed non-overlapping KnowEdit rows from the stream manifest",
    "score": "one minus Jensen-Shannon divergence normalized by log(2); higher is better",
    "scoring": "all non-overlapping probe prompts are scored",
}
PRE_EDIT_PROTOCOL_HASH = hashlib.sha256(
    json.dumps(
        PRE_EDIT_PROTOCOL_SPEC,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
).hexdigest()


@torch.no_grad()
def _next_token_distribution(editor: "BioCsLLMEditor", prompt: str) -> torch.Tensor:
    """Return the causal-LM next-token probabilities after a fixed prompt."""
    tokens = editor._tokenize_text(prompt, padding=False)
    logits = editor.model(**tokens).logits[0, -1].float()
    return torch.softmax(logits, dim=-1)


def _target_match(prediction: str, targets) -> float:
    """Return whether one normalized target alias occurs in a continuation."""
    prediction_lower = _to_text(prediction).casefold()
    aliases = [alias.casefold() for alias in _to_text_list(targets) if alias]
    return float(any(alias in prediction_lower for alias in aliases))


def _iter_target_items(groups: dict):
    """Yield normalized prompt/answer probes from a KnowEdit group mapping."""
    if not isinstance(groups, dict):
        return
    for group, raw_items in groups.items():
        items = [raw_items] if isinstance(raw_items, dict) else raw_items
        if not isinstance(items, (list, tuple)):
            continue
        for item_index, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            prompt = _to_text(item.get("prompt", ""))
            targets = _to_text_list(item.get("ground_truth", ""))
            if prompt and targets:
                yield str(group), item_index, prompt, targets


def _evaluate_target_groups(groups: dict, generate) -> dict:
    """Evaluate KnowEdit portability probes without relabelling them locality."""
    evaluations = []
    for group, item_index, prompt, targets in _iter_target_items(groups):
        prediction = generate(prompt).strip()
        evaluations.append(
            {
                "group": group,
                "item_index": item_index,
                "matched": bool(_target_match(prediction, targets)),
            }
        )
    count = len(evaluations)
    return {
        "score": sum(float(item["matched"]) for item in evaluations) / max(count, 1),
        "n_items": count,
        "n_matched": sum(int(item["matched"]) for item in evaluations),
        "evaluations": evaluations,
    }

# -------------------- SSR core functions ----------------------

KERNEL_FAMILIES = ("gaussian", "laplace", "cauchy", "inverse")
DISTANCE_METRICS = ("cosine", "projective")
ROW_SELECTIONS = ("random_each_step", "fixed_random", "active_top")
SSR_MODES = ("independent_delta_v1", "semantic_history_delta_v1")
HISTORY_NEIGHBOR_POLICIES = ("semantic", "recent")
HISTORY_KERNEL_MODES = ("uniform", "radial_center_surround", "topology_ring")
HISTORY_PROJECTION_MODES = ("none", "semantic_topk")

# Each recipe changes only the listed objective components in addition to the
# task loss.  Keeping this registry explicit prevents a run label from drifting
# away from the loss that was actually optimized.
RECIPES: dict[str, dict[str, bool]] = {
    "plain": {"ssr": False, "anchor": False, "spectral": False},
    "anchor": {"ssr": False, "anchor": True, "spectral": False},
    "spectral": {"ssr": False, "anchor": False, "spectral": True},
    "stabilized": {"ssr": False, "anchor": True, "spectral": True},
    "ssr_only": {"ssr": True, "anchor": False, "spectral": False},
    "ssr_anchor": {"ssr": True, "anchor": True, "spectral": False},
    "ssr_spectral": {"ssr": True, "anchor": False, "spectral": True},
    "full": {"ssr": True, "anchor": True, "spectral": True},
}
OBJECTIVE_COMPONENTS = ("task", "ssr", "anchor", "spectral")


def infer_recipe(
    lambda_ssr: float,
    lambda_anchor: float,
    lambda_spectral: float,
) -> str:
    """Infer the named recipe used by legacy coefficient-only callers."""
    active = {
        name
        for name, value in (
            ("ssr", lambda_ssr),
            ("anchor", lambda_anchor),
            ("spectral", lambda_spectral),
        )
        if value > 0
    }
    for name, component_flags in RECIPES.items():
        registered = {
            component for component, enabled in component_flags.items() if enabled
        }
        if active == registered:
            return name
    raise AssertionError(f"No registered recipe for components {sorted(active)}")


def resolve_recipe(
    recipe: Optional[str] = None,
    *,
    lambda_ssr: float = 0.001,
    lambda_anchor: float = 0.001,
    lambda_spectral: float = 0.01,
) -> dict:
    """Resolve a recipe to the exact coefficients and objective metadata.

    ``recipe=None`` preserves the historical coefficient-driven interface by
    inferring one of the eight complete component combinations.  With an
    explicit recipe, coefficients belonging to disabled components are zeroed.
    Enabled components must have a positive coefficient so the recorded recipe
    cannot claim a loss term that was inactive in the optimizer.
    """
    coefficients = {
        "ssr": float(lambda_ssr),
        "anchor": float(lambda_anchor),
        "spectral": float(lambda_spectral),
    }
    if any(value < 0 for value in coefficients.values()):
        raise ValueError("Recipe coefficients must be non-negative")

    explicit = recipe is not None
    name = (
        recipe.strip().lower()
        if explicit
        else infer_recipe(
            lambda_ssr=coefficients["ssr"],
            lambda_anchor=coefficients["anchor"],
            lambda_spectral=coefficients["spectral"],
        )
    )
    if name not in RECIPES:
        raise ValueError(f"Unknown recipe={recipe!r}; choose from {tuple(RECIPES)}")

    enabled = {
        component for component, is_enabled in RECIPES[name].items() if is_enabled
    }
    if explicit:
        missing = [component for component in enabled if coefficients[component] <= 0]
        if missing:
            raise ValueError(
                f"Recipe {name!r} enables {missing}, so their coefficients must be positive"
            )
    effective = {
        component: coefficients[component] if component in enabled else 0.0
        for component in ("ssr", "anchor", "spectral")
    }
    active_components = (
        "task",
        *(component for component in ("ssr", "anchor", "spectral") if component in enabled),
    )
    return {
        "recipe": name,
        "components": list(active_components),
        "objective": {
            component: component in active_components
            for component in OBJECTIVE_COMPONENTS
        },
        "lambda_ssr": effective["ssr"],
        "lambda_anchor": effective["anchor"],
        "lambda_spectral": effective["spectral"],
    }


def stable_row_seed(module_name: str, sampler_seed: int, edit_index: int) -> int:
    """Return a process-independent seed for fixed spatial-row sampling."""
    digest = hashlib.sha256(module_name.encode("utf-8")).digest()
    module_seed = int.from_bytes(digest[:8], "little")
    return (int(sampler_seed) + 1009 * int(edit_index) + module_seed) % (2**63 - 1)


def fixed_random_row_indices(
    module_name: str,
    n_rows: int,
    max_rows: int,
    *,
    sampler_seed: int = 0,
    edit_index: int = 0,
) -> torch.Tensor:
    """Sample deterministic CPU row indices without constructing an editor."""
    if n_rows < 0:
        raise ValueError("n_rows must be non-negative")
    if max_rows < 0:
        raise ValueError("max_rows must be non-negative")
    count = min(n_rows, max_rows)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(stable_row_seed(module_name, sampler_seed, edit_index))
    return torch.randperm(n_rows, generator=generator)[:count]


def _radial_kernel(dist: torch.Tensor, sigma: float, family: str) -> torch.Tensor:
    """Return the LLM-study radial response, normalized to one at zero.

    The historical ``inverse`` key denotes
    ``sigma / sqrt(distance**2 + sigma**2)``. Its HWHM is
    ``sqrt(3) * sigma``; ``sigma`` itself is not the half-width.
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    if family == "gaussian":
        return torch.exp(-(dist ** 2) / (2 * sigma ** 2))
    if family == "laplace":
        return torch.exp(-dist / sigma)
    if family == "cauchy":
        return 1.0 / (1.0 + (dist / sigma) ** 2)
    if family == "inverse":
        return sigma / torch.sqrt(dist ** 2 + sigma ** 2)
    raise ValueError(f"Unknown kernel_family={family!r}; choose from {KERNEL_FAMILIES}")


def pairwise_distance(
    weights: torch.Tensor,
    mapping: str = "cosine",
    eps: float = 1e-8,
) -> torch.Tensor:
    """Pairwise directional or sign-invariant projective distance."""
    if mapping not in DISTANCE_METRICS:
        raise ValueError(f"Unknown mapping={mapping!r}; choose from {DISTANCE_METRICS}")
    if weights.ndim != 2:
        raise ValueError("weights must be a two-dimensional matrix")
    normalized = F.normalize(weights.float(), dim=1)
    cosine = torch.clamp(normalized @ normalized.T, -1.0, 1.0)
    value = 1.0 - (cosine.square() if mapping == "projective" else cosine)
    return torch.sqrt(torch.clamp(value, min=eps))


def compute_spatial_biocs_llm(
    weights: torch.Tensor,
    A_exc: float = 1.0,
    A_inh: float = 0.8,
    sigma_exc: float = 0.3,
    sigma_inh: float = 0.8,
    kernel_family: str = "gaussian",
    distance_metric: str = "cosine",
    row_indices: Optional[torch.Tensor] = None,
    max_rows: int = 256,
) -> torch.Tensor:
    """Mexican-hat center-surround penalty for LLM weight matrices.

    For large matrices, operates on a random subset of rows to keep
    O(n^2) computation tractable.
    """
    w = weights.float()
    n = w.size(0)

    if row_indices is not None:
        if row_indices.numel() < 2:
            return weights.sum() * 0.0
        w = w.index_select(0, row_indices.to(w.device))
        n = w.size(0)
    elif max_rows > 0 and n > max_rows:
        idx = torch.randperm(n, device=w.device)[:max_rows]
        w = w[idx]
        n = max_rows

    dist = pairwise_distance(w, mapping=distance_metric)

    exc = A_exc * _radial_kernel(dist, sigma_exc, kernel_family)
    inh = A_inh * _radial_kernel(dist, sigma_inh, kernel_family)
    annulus = (inh - exc).clamp_min(0.0)
    offdiag = ~torch.eye(n, dtype=torch.bool, device=annulus.device)
    normalized = F.normalize(w, dim=1)
    cosine = normalized @ normalized.T
    overlap = F.relu(cosine - 0.15).square()
    return (annulus[offdiag] * overlap[offdiag]).sum() / annulus[offdiag].sum().clamp_min(1e-8)


def semantic_history_delta_penalty(
    delta: torch.Tensor,
    prompt_key: torch.Tensor,
    history_prompt_keys: list[torch.Tensor],
    history_update_signatures: list[torch.Tensor],
    *,
    row_indices: torch.Tensor,
    neighbors: int,
    repulsion_margin: float,
    attraction_weight: float,
    repulsion_weight: float,
    neighbor_policy: str = "semantic",
    kernel_mode: str = "uniform",
    kernel_family: str = "gaussian",
    sigma_exc: float = 0.3,
    sigma_inh: float = 0.8,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compare one edit's row update to frozen, semantically keyed history.

    The prompt key comes from the frozen input embedding table before the edit.
    Historical update signatures are detached summaries of prior edits. Only
    the selected rows of the present delta receive gradient.
    """
    if not history_prompt_keys:
        return delta.sum() * 0.0, {
            "history_pairs": 0.0,
            "semantic_neighbor_pairs": 0.0,
            "semantic_attraction": 0.0,
            "semantic_repulsion": 0.0,
        }
    if len(history_prompt_keys) != len(history_update_signatures):
        raise ValueError("Prompt-key and update-signature histories diverged")
    if row_indices.numel() < 2:
        return delta.sum() * 0.0, {
            "history_pairs": 0.0,
            "semantic_neighbor_pairs": 0.0,
            "semantic_attraction": 0.0,
            "semantic_repulsion": 0.0,
        }
    selected = delta.float().index_select(0, row_indices.to(delta.device))
    current_update = selected.mean(dim=0)
    # The first optimization step has zero delta. Smooth normalization keeps
    # its gradient bounded instead of dividing by F.normalize's tiny epsilon.
    current_signature = current_update / torch.sqrt(
        current_update.square().sum() + current_update.new_tensor(1e-4)
    )
    old_updates = torch.stack(history_update_signatures).to(delta.device, dtype=torch.float32)
    if old_updates.ndim != 2 or old_updates.size(1) != current_signature.numel():
        raise ValueError("Historical update signature dimension mismatch")
    old_keys = torch.stack(history_prompt_keys).to(delta.device, dtype=torch.float32)
    key = F.normalize(prompt_key.to(delta.device, dtype=torch.float32), dim=0)
    semantic_similarity = F.normalize(old_keys, dim=1) @ key
    count = min(int(neighbors), semantic_similarity.numel())
    if count <= 0:
        raise ValueError("Semantic history SSR requires at least one previous edit")
    if neighbor_policy == "semantic":
        neighbor_indices = semantic_similarity.topk(count).indices
    elif neighbor_policy == "recent":
        # Mechanism control: keep the identical history bank and objective,
        # but define neighbors by recency rather than prompt similarity.
        neighbor_indices = torch.arange(
            semantic_similarity.numel() - count,
            semantic_similarity.numel(),
            device=semantic_similarity.device,
        )
    else:
        raise ValueError(
            f"Unknown history neighbor policy={neighbor_policy!r}; "
            f"choose from {HISTORY_NEIGHBOR_POLICIES}"
        )
    neighbor_mask = torch.zeros_like(semantic_similarity, dtype=torch.bool)
    neighbor_mask.scatter_(0, neighbor_indices, True)
    update_similarity = F.normalize(old_updates, dim=1) @ current_signature
    # Convert cosine similarity to a bounded half-angle distance.  This makes
    # the history kernel's scale comparable across prompt-key dimensions.
    semantic_distance = torch.sqrt(
        torch.clamp((1.0 - semantic_similarity) * 0.5, min=0.0)
    )

    def weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return (values * weights).sum() / weights.sum().clamp_min(1e-8)

    attraction_values = 1.0 - update_similarity[neighbor_mask]
    if kernel_mode == "uniform":
        attraction = attraction_values.mean()
        attraction_weights = torch.ones_like(attraction_values)
    elif kernel_mode in {"radial_center_surround", "topology_ring"}:
        attraction_weights = _radial_kernel(
            semantic_distance[neighbor_mask], sigma_exc, kernel_family
        )
        attraction = weighted_mean(attraction_values, attraction_weights)
    else:
        raise ValueError(
            f"Unknown history kernel mode={kernel_mode!r}; choose from {HISTORY_KERNEL_MODES}"
        )
    non_neighbor_mask = ~neighbor_mask
    if non_neighbor_mask.any():
        repulsion_values = F.relu(
            update_similarity[non_neighbor_mask] - repulsion_margin
        )
        if kernel_mode == "uniform":
            repulsion = repulsion_values.mean()
            repulsion_weights = torch.ones_like(repulsion_values)
        elif kernel_mode == "radial_center_surround":
            # A far semantic item has a larger surround contribution.  The
            # normalized weighted mean retains the original loss scale.
            repulsion_weights = 1.0 - _radial_kernel(
                semantic_distance[non_neighbor_mask], sigma_inh, kernel_family
            )
            repulsion = weighted_mean(repulsion_values, repulsion_weights)
        else:
            # A genuine center-surround ring is zero at the center and at
            # long range. This preserves the release regime in the manuscript
            # definition rather than imposing maximum repulsion on far edits.
            local = _radial_kernel(
                semantic_distance[non_neighbor_mask], sigma_exc, kernel_family
            )
            broad = _radial_kernel(
                semantic_distance[non_neighbor_mask], sigma_inh, kernel_family
            )
            repulsion_weights = (broad - local).clamp_min(0.0)
            repulsion = weighted_mean(repulsion_values.square(), repulsion_weights)
    else:
        repulsion = attraction * 0.0
        repulsion_weights = attraction_weights.new_zeros((0,))
    loss = attraction_weight * attraction + repulsion_weight * repulsion
    return loss.to(delta.dtype), {
        "history_pairs": float(update_similarity.numel()),
        "semantic_neighbor_pairs": float(neighbor_mask.sum().item()),
        "semantic_attraction": float(attraction.detach()),
        "semantic_repulsion": float(repulsion.detach()),
        "semantic_attraction_weight_mean": float(attraction_weights.detach().mean()),
        "semantic_repulsion_weight_mean": (
            float(repulsion_weights.detach().mean()) if repulsion_weights.numel() else 0.0
        ),
    }


def project_history_gradient(
    gradient: torch.Tensor,
    prompt_key: torch.Tensor,
    history_prompt_keys: list[torch.Tensor],
    history_module_inputs: list[torch.Tensor],
    *,
    neighbors: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Remove gradient components that change semantically nearby old inputs.

    The preserved vectors are frozen inputs to the editable projection from
    earlier edits.  Right-projecting a linear weight gradient onto their
    orthogonal complement makes the first-order update leave those local
    module outputs unchanged.  Prompt similarity chooses the protected local
    set; the existing SSR ring loss still controls update-direction geometry.
    """
    if gradient.ndim != 2 or not history_prompt_keys or not history_module_inputs:
        return gradient, {"protected_history_vectors": 0.0, "gradient_removed_fraction": 0.0}
    if len(history_prompt_keys) != len(history_module_inputs):
        raise ValueError("Semantic history prompt and module-input banks diverged")
    if gradient.size(1) != history_module_inputs[0].numel():
        raise ValueError("Historical module-input dimension does not match editable weight columns")

    old_prompt_keys = torch.stack(history_prompt_keys).to(gradient.device, dtype=torch.float32)
    current_prompt_key = F.normalize(prompt_key.to(gradient.device, dtype=torch.float32), dim=0)
    similarity = F.normalize(old_prompt_keys, dim=1) @ current_prompt_key
    count = min(int(neighbors), similarity.numel())
    if count <= 0:
        return gradient, {"protected_history_vectors": 0.0, "gradient_removed_fraction": 0.0}
    selected = similarity.topk(count).indices
    protected = torch.stack(history_module_inputs).to(gradient.device, dtype=torch.float32)
    protected = protected.index_select(0, selected)
    protected = F.normalize(protected, dim=1)
    basis, _ = torch.linalg.qr(protected.T, mode="reduced")
    gradient_fp32 = gradient.float()
    removed = (gradient_fp32 @ basis) @ basis.T
    projected = gradient_fp32 - removed
    removed_fraction = removed.norm() / gradient_fp32.norm().clamp_min(1e-12)
    return projected.to(gradient.dtype), {
        "protected_history_vectors": float(count),
        "gradient_removed_fraction": float(removed_fraction.detach()),
    }


def compute_spectral_flatness_llm(weights: torch.Tensor) -> torch.Tensor:
    """SVD spectral flattening for LLM weight matrices."""
    w = weights.float()
    if w.size(0) < 2 or w.size(1) < 2:
        return torch.tensor(0.0, device=w.device)
    try:
        sv = torch.linalg.svdvals(w[:512, :512])
    except Exception:
        return torch.tensor(0.0, device=w.device)
    sv_norm = sv / (sv.sum() + 1e-8)
    uniform = torch.ones_like(sv_norm) / sv_norm.numel()
    return F.kl_div((sv_norm + 1e-8).log(), uniform, reduction="sum")


# ──────────────────── Dataset ─────────────────────────────────────

class KnowEditDataset(Dataset):
    """Loads KnowEdit JSON for sequential editing evaluation."""

    def __init__(
        self,
        json_path: str,
        max_samples: int = 500,
        offset: int = 0,
        indices: Optional[list[int]] = None,
    ):
        with open(json_path) as f:
            raw = json.load(f)
        if offset < 0:
            raise ValueError("offset must be non-negative")
        self.offset = offset
        if indices is None:
            self.source_indices = list(range(offset, min(offset + max_samples, len(raw))))
            self.data = raw[offset : offset + max_samples]
        else:
            if len(indices) > max_samples:
                raise ValueError("The index manifest contains more rows than max_samples")
            if len(indices) != len(set(indices)):
                raise ValueError("The index manifest must not repeat edit rows")
            if any(not isinstance(index, int) or index < 0 or index >= len(raw) for index in indices):
                raise ValueError("The index manifest contains an out-of-range edit row")
            self.source_indices = list(indices)
            self.data = [raw[index] for index in self.source_indices]
        self.source_indices_sha256 = hashlib.sha256(
            json.dumps(self.source_indices, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if os.environ.get("KE_REQUIRE_LEGACY_LOCALITY_COMPATIBLE", "0").strip().lower() in {
            "1", "true", "yes", "on",
        }:
            for row_index, item in enumerate(self.data, start=offset):
                try:
                    assert_legacy_compatible_locality(item.get("locality", {}))
                except ValueError as error:
                    raise ValueError(
                        f"KnowEdit row {row_index} is not compatible with the locked "
                        f"locality protocol: {error}"
                    ) from error
        logger.info(
            "Loaded %d editing samples from %s (%s)",
            len(self.data),
            json_path,
            "index manifest" if indices is not None else f"offset {offset}",
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return normalize_knowedit_record(self.data[idx])


def normalize_knowedit_record(item: dict) -> dict:
    """Map factual and WikiBio KnowEdit rows to the custom editor schema."""
    prompt = item.get("prompt", item.get("text"))
    target_new = item.get("target_new", item.get("labels"))
    subject = item.get("subject", item.get("concept", ""))
    if prompt is None or target_new is None:
        raise ValueError(
            "KnowEdit rows require prompt/target_new or WikiBio text/labels fields"
        )
    locality = item.get("locality", {})
    portability = item.get("portability", {})
    if not isinstance(locality, dict) or not isinstance(portability, dict):
        raise ValueError("KnowEdit locality and portability fields must be mappings")
    rephrase = item.get("rephrase_prompt", item.get("rephrase", ""))
    if isinstance(rephrase, (list, tuple)):
        rephrase_prompts = [_to_text(value) for value in rephrase]
    else:
        rephrase_prompts = [_to_text(rephrase)]
    rephrase_prompts = [prompt for prompt in rephrase_prompts if prompt]
    return {
        "prompt": _to_text(prompt),
        "target_new": _to_text(target_new),
        "ground_truth": _to_text(item.get("ground_truth", "")),
        "subject": _to_text(subject),
        "locality": locality,
        "portability": portability,
        "rephrase_prompts": rephrase_prompts,
    }


# -------------------- SSR editor ------------------------------

class BioCsLLMEditor:
    """Fine-tune specific MLP layers with SSR.

    Strategy:
      1. For each edit request, fine-tune target MLP layers to produce
         the new target while constraining weights via SSR
      2. The SSR penalty organizes editable directions to reduce interference,
         preserving locality for unrelated knowledge
      3. Spectral flattening maintains weight matrix health across edits
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        target_layers: Optional[list[int]] = None,
        lambda_biocs: float = 0.001,
        lambda_spectral: float = 0.01,
        lambda_anchor: float = 0.001,
        kernel_family: str = "gaussian",
        a_exc: float = 1.0,
        a_inh: float = 0.8,
        sigma_exc: float = 0.3,
        sigma_inh: float = 0.8,
        biocs_target: str = "weight",
        distance_metric: str = "cosine",
        row_selection: str = "fixed_random",
        max_spatial_rows: int = 256,
        sampler_seed: int = 0,
        lr: float = 1e-4,
        num_steps: int = 25,
        max_length: int = 64,
        max_new_tokens: int = 32,
        recipe: Optional[str] = None,
        lambda_ssr: Optional[float] = None,
        distance_mapping: Optional[str] = None,
    ):
        requested_lambda_ssr = lambda_biocs if lambda_ssr is None else lambda_ssr
        recipe_config = resolve_recipe(
            recipe,
            lambda_ssr=requested_lambda_ssr,
            lambda_anchor=lambda_anchor,
            lambda_spectral=lambda_spectral,
        )
        if distance_mapping is not None:
            distance_metric = distance_mapping

        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
            AutoModelForImageTextToText,
            AutoProcessor,
            AutoTokenizer,
        )

        self.device = torch.device(device)
        self.model_name = model_name
        logger.info(f"Loading model: {model_name}")

        load_kw = {"trust_remote_code": True}
        if os.path.isdir(model_name) or os.environ.get("KE_LOCAL_ONLY") == "1" or os.environ.get("TRANSFORMERS_OFFLINE") == "1" or os.environ.get("HF_HUB_OFFLINE") == "1":
            load_kw["local_files_only"] = True
        model_kw = dict(load_kw)
        dtype_name = os.environ.get("KE_MODEL_DTYPE", "auto").lower()
        if dtype_name == "bf16":
            model_kw["torch_dtype"] = torch.bfloat16
        elif dtype_name == "fp16":
            model_kw["torch_dtype"] = torch.float16
        elif dtype_name == "fp32":
            model_kw["torch_dtype"] = torch.float32
        elif dtype_name != "auto":
            raise ValueError(f"Unsupported KE_MODEL_DTYPE={dtype_name}")
        else:
            model_kw["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        model_kw["low_cpu_mem_usage"] = True
        if os.environ.get("KE_ATTN_IMPLEMENTATION"):
            model_kw["attn_implementation"] = os.environ["KE_ATTN_IMPLEMENTATION"]
        device_map = os.environ.get("KE_DEVICE_MAP", "").strip()
        if device_map:
            model_kw["device_map"] = device_map

        self.config = AutoConfig.from_pretrained(model_name, **load_kw)
        force_text_only = os.environ.get("KE_FORCE_TEXT_ONLY", "0").strip().lower() in {"1", "true", "yes", "on"}
        self.is_multimodal = (not force_text_only) and bool(
            getattr(self.config, "vision_config", None) is not None
            or getattr(self.config, "model_type", "") in {"qwen2_5_vl"}
        )

        self.processor = None
        if self.is_multimodal:
            self.processor = AutoProcessor.from_pretrained(model_name, **load_kw)
            self.tokenizer = getattr(self.processor, "tokenizer", None)
            if self.tokenizer is None:
                raise AttributeError("Multimodal processor does not expose a tokenizer.")
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_name, **model_kw,
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, **load_kw)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, **model_kw,
            )
        if not device_map:
            self.model = self.model.to(self.device)
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.model.eval()
        self.input_device = next(self.model.parameters()).device

        self.target_layers = target_layers or self._auto_select_layers()
        self.recipe = recipe_config["recipe"]
        self.objective_components = recipe_config["components"]
        self.objective = recipe_config["objective"]
        self.use_ssr = self.objective["ssr"]
        self.use_anchor = self.objective["anchor"]
        self.use_spectral = self.objective["spectral"]
        self.lambda_biocs = recipe_config["lambda_ssr"]
        self.lambda_spectral = recipe_config["lambda_spectral"]
        self.lambda_anchor = recipe_config["lambda_anchor"]
        if kernel_family not in KERNEL_FAMILIES:
            raise ValueError(f"Unknown kernel_family={kernel_family!r}; choose from {KERNEL_FAMILIES}")
        if sigma_exc <= 0 or sigma_inh <= 0:
            raise ValueError("sigma_exc and sigma_inh must be positive")
        self.kernel_family = kernel_family
        self.a_exc = a_exc
        self.a_inh = a_inh
        self.sigma_exc = sigma_exc
        self.sigma_inh = sigma_inh
        if biocs_target not in {"weight", "delta"}:
            raise ValueError("biocs_target must be 'weight' or 'delta'")
        self.biocs_target = biocs_target
        if distance_metric not in DISTANCE_METRICS:
            raise ValueError(
                f"distance_metric must be one of {DISTANCE_METRICS}, got {distance_metric!r}"
            )
        self.distance_metric = distance_metric
        if row_selection not in ROW_SELECTIONS:
            raise ValueError(
                f"row_selection must be one of {ROW_SELECTIONS}, got {row_selection!r}"
            )
        if max_spatial_rows < 2:
            raise ValueError("max_spatial_rows must be at least 2")
        self.row_selection = row_selection
        self.max_spatial_rows = max_spatial_rows
        self.sampler_seed = sampler_seed
        self._edit_index = 0
        self._edit_row_indices: dict[str, torch.Tensor] = {}
        # A plain FT control shares the editable object with SSR but must not
        # allocate or update SSR-specific semantic-history state.
        self.ssr_mode = (
            os.environ.get("BIOCS_SSR_MODE", "independent_delta_v1").strip()
            if self.use_ssr
            else "independent_delta_v1"
        )
        if self.ssr_mode not in SSR_MODES:
            raise ValueError(f"Unknown BIOCS_SSR_MODE={self.ssr_mode!r}")
        self.history_neighbors = int(os.environ.get("BIOCS_HISTORY_NEIGHBORS", "3"))
        self.history_margin = float(os.environ.get("BIOCS_HISTORY_MARGIN", "0.15"))
        self.history_limit = int(os.environ.get("BIOCS_HISTORY_LIMIT", "128"))
        self.history_neighbor_policy = os.environ.get(
            "BIOCS_HISTORY_NEIGHBOR_POLICY", "semantic"
        ).strip()
        self.history_kernel_mode = os.environ.get(
            "BIOCS_HISTORY_KERNEL_MODE", "uniform"
        ).strip()
        self.history_projection_mode = os.environ.get(
            "BIOCS_HISTORY_PROJECTION_MODE", "none"
        ).strip()
        self.primary_gradient_mode = os.environ.get(
            "BIOCS_PRIMARY_GRADIENT_MODE", "none"
        ).strip()
        if self.history_neighbors < 1 or self.history_limit < 1:
            raise ValueError("Semantic-history SSR requires positive neighbors and history limit")
        if not -1.0 <= self.history_margin <= 1.0:
            raise ValueError("BIOCS_HISTORY_MARGIN must lie in [-1, 1]")
        if self.history_neighbor_policy not in HISTORY_NEIGHBOR_POLICIES:
            raise ValueError(
                "BIOCS_HISTORY_NEIGHBOR_POLICY must be one of "
                f"{HISTORY_NEIGHBOR_POLICIES}"
            )
        if self.history_kernel_mode not in HISTORY_KERNEL_MODES:
            raise ValueError(
                "BIOCS_HISTORY_KERNEL_MODE must be one of "
                f"{HISTORY_KERNEL_MODES}"
            )
        if self.history_projection_mode not in HISTORY_PROJECTION_MODES:
            raise ValueError(
                "BIOCS_HISTORY_PROJECTION_MODE must be one of "
                f"{HISTORY_PROJECTION_MODES}"
            )
        if self.primary_gradient_mode not in {"none", "pcgrad_v1"}:
            raise ValueError("BIOCS_PRIMARY_GRADIENT_MODE must be 'none' or 'pcgrad_v1'")
        if self.ssr_mode == "semantic_history_delta_v1" and self.biocs_target != "delta":
            raise ValueError("Semantic-history SSR requires BIOCS_TARGET=delta")
        if self.ssr_mode == "semantic_history_delta_v1" and self.row_selection != "fixed_random":
            raise ValueError("Semantic-history SSR requires fixed_random row selection")
        if self.history_projection_mode != "none" and self.ssr_mode != "semantic_history_delta_v1":
            raise ValueError("History projection requires semantic_history_delta_v1 SSR")
        self._history_prompt_keys: list[torch.Tensor] = []
        self._history_update_signatures: dict[str, list[torch.Tensor]] = {}
        self._history_module_inputs: dict[str, list[torch.Tensor]] = {}
        self.lr = lr
        self.num_steps = num_steps
        if max_length <= 0 or max_new_tokens <= 0:
            raise ValueError("max_length and max_new_tokens must be positive")
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens
        self.target_module_regex = os.environ.get("BIOCS_TARGET_MODULE_REGEX", r"(c_proj|down_proj)$").strip()
        self.max_target_modules = int(os.environ.get("BIOCS_MAX_TARGET_MODULES", "0"))

        self._weight_snapshots: dict[str, torch.Tensor] = {}
        self._save_weight_snapshot()

        logger.info(f"Target layers: {self.target_layers}")
        logger.info(
            f"recipe={self.recipe}, components={self.objective_components}; "
            f"λ_ssr={self.lambda_biocs}, λ_spectral={self.lambda_spectral}, "
            f"λ_anchor={self.lambda_anchor}; "
            f"kernel={kernel_family}, A=({a_exc},{a_inh}), sigma=({sigma_exc},{sigma_inh}), "
            f"target={biocs_target}, distance={distance_metric}, "
            f"rows={row_selection}:{max_spatial_rows}"
        )

    def _tokenize_text(self, text: str, padding: bool = False):
        if self.is_multimodal:
            tokens = self.processor(
                text=text,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
                padding=padding,
            )
        else:
            tokens = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
                padding=padding,
            )
        return tokens.to(self.input_device)

    def _auto_select_layers(self) -> list[int]:
        """Select the last 3 MLP layers as edit targets (most knowledge-rich)."""
        n_layers = getattr(self.model.config, "num_hidden_layers", None)
        if n_layers is None:
            n_layers = getattr(self.model.config, "n_layer", None)
        if n_layers is None and hasattr(self.model.config, "text_config"):
            n_layers = getattr(self.model.config.text_config, "num_hidden_layers", None)
        if n_layers is None and hasattr(self.model, "transformer"):
            blocks = getattr(self.model.transformer, "h", None)
            if blocks is not None:
                n_layers = len(blocks)
        if n_layers is None:
            raise AttributeError(
                "Could not infer number of transformer layers from model config; "
                "please pass target_layers explicitly."
            )
        return list(range(max(0, n_layers - 3), n_layers))

    def _get_mlp_modules(self) -> list[tuple[str, nn.Module]]:
        """Get target MLP weight matrices (handles nn.Linear and Conv1D)."""
        modules = []
        for name, mod in self.model.named_modules():
            has_weight = hasattr(mod, "weight") and mod.weight is not None
            is_linear = isinstance(mod, nn.Linear)
            is_conv1d = type(mod).__name__ == "Conv1D"
            if self.target_module_regex and not re.search(self.target_module_regex, name):
                continue
            if has_weight and (is_linear or is_conv1d):
                for layer_idx in self.target_layers:
                    if f".{layer_idx}." in name and "mlp" in name:
                        modules.append((name, mod))
                        if self.max_target_modules > 0 and len(modules) >= self.max_target_modules:
                            return modules
                        break
        return modules

    def metadata(self) -> dict:
        return {
            "run_label": os.environ.get("KE_RUN_LABEL", self.__class__.__name__),
            "editor_class": self.__class__.__name__,
            "model": self.model_name,
            "target_layers": self.target_layers,
            "target_module_regex": self.target_module_regex,
            "max_target_modules": self.max_target_modules,
            "recipe": self.recipe,
            "objective_components": self.objective_components,
            "objective": self.objective,
            "use_ssr": self.use_ssr,
            "use_anchor": self.use_anchor,
            "use_spectral": self.use_spectral,
            "lambda_ssr": self.lambda_biocs,
            "lambda_biocs": self.lambda_biocs,
            "lambda_spectral": self.lambda_spectral,
            "lambda_anchor": self.lambda_anchor,
            "kernel_family": self.kernel_family,
            "a_exc": self.a_exc,
            "a_inh": self.a_inh,
            "sigma_exc": self.sigma_exc,
            "sigma_inh": self.sigma_inh,
            "biocs_target": self.biocs_target,
            "distance_metric": self.distance_metric,
            "distance_mapping": self.distance_metric if self.objective["ssr"] else None,
            "row_selection": self.row_selection,
            "max_spatial_rows": self.max_spatial_rows,
            "ssr_mode": self.ssr_mode,
            "history_neighbors": self.history_neighbors,
            "history_neighbor_policy": self.history_neighbor_policy,
            "history_kernel_mode": self.history_kernel_mode,
            "history_projection_mode": self.history_projection_mode,
            "history_margin": self.history_margin,
            "history_limit": self.history_limit,
            "sampler_seed": self.sampler_seed,
            "lr": self.lr,
            "num_steps": self.num_steps,
            "max_length": self.max_length,
            "max_new_tokens": self.max_new_tokens,
            "seed": os.environ.get("KE_SEED", ""),
            "model_dtype": os.environ.get("KE_MODEL_DTYPE", "auto"),
            "device_map": os.environ.get("KE_DEVICE_MAP", ""),
            "force_text_only": os.environ.get("KE_FORCE_TEXT_ONLY", "0"),
            "locality_evaluator": LOCALITY_EVALUATOR,
            "locality_evaluator_version": LOCALITY_EVALUATOR_VERSION,
            "locality_evaluator_protocol_hash": os.environ.get(
                "KE_EVALUATION_PROTOCOL_HASH", LOCALITY_PROTOCOL_HASH
            ),
        }

    def _save_weight_snapshot(self):
        for name, mod in self._get_mlp_modules():
            self._weight_snapshots[name] = mod.weight.data.clone()

    def _stable_row_seed(self, module_name: str) -> int:
        return stable_row_seed(module_name, self.sampler_seed, self._edit_index)

    def _fixed_random_rows(self, module_name: str, n_rows: int) -> torch.Tensor:
        return fixed_random_row_indices(
            module_name,
            n_rows,
            self.max_spatial_rows,
            sampler_seed=self.sampler_seed,
            edit_index=self._edit_index,
        )

    def _prepare_fixed_rows(self, target_modules: list[tuple[str, nn.Module]]) -> None:
        self._edit_row_indices = {}
        if self.row_selection != "fixed_random":
            return
        for name, mod in target_modules:
            self._edit_row_indices[name] = self._fixed_random_rows(name, mod.weight.shape[0])

    def _capture_active_rows(self, target_modules: list[tuple[str, nn.Module]]) -> None:
        if self.row_selection != "active_top":
            return
        for name, mod in target_modules:
            snapshot = self._weight_snapshots.get(name)
            if snapshot is None:
                continue
            delta = mod.weight.detach().float() - snapshot.to(mod.weight.device).float()
            row_norm = delta.flatten(1).norm(dim=1)
            count = min(row_norm.numel(), self.max_spatial_rows)
            self._edit_row_indices[name] = torch.topk(
                row_norm, k=count, largest=True, sorted=True,
            ).indices.cpu()

    @torch.no_grad()
    def _prompt_semantic_key(self, prompt: str) -> torch.Tensor:
        tokens = self._tokenize_text(prompt, padding=False)
        embeddings = self.model.get_input_embeddings()(tokens["input_ids"])
        attention = tokens.get("attention_mask")
        if attention is None:
            key = embeddings.mean(dim=1).squeeze(0)
        else:
            weight = attention.to(embeddings.dtype).unsqueeze(-1)
            key = (embeddings * weight).sum(dim=1).squeeze(0) / weight.sum(dim=1).squeeze(0).clamp_min(1.0)
        return F.normalize(key.float(), dim=0).detach().cpu()

    @torch.no_grad()
    def _module_input_keys(
        self,
        prompt: str,
        target_modules: list[tuple[str, nn.Module]],
    ) -> dict[str, torch.Tensor]:
        """Capture each target's frozen pre-edit input at the final prompt token."""
        captured: dict[str, torch.Tensor] = {}
        hooks = []
        for name, module in target_modules:
            def capture(_module, args, *, module_name=name):
                if not args or not isinstance(args[0], torch.Tensor):
                    raise ValueError("Editable module did not receive a tensor input")
                activation = args[0]
                vector = activation.reshape(-1, activation.shape[-1])[-1].detach().float()
                captured[module_name] = F.normalize(vector, dim=0).cpu()
            hooks.append(module.register_forward_pre_hook(capture))
        was_training = self.model.training
        self.model.eval()
        try:
            self.model(**self._tokenize_text(prompt, padding=False))
        finally:
            for hook in hooks:
                hook.remove()
            self.model.train(was_training)
        if set(captured) != {name for name, _ in target_modules}:
            raise ValueError("Failed to capture every editable module input")
        return captured

    @torch.no_grad()
    def _record_history_update(
        self,
        target_modules: list[tuple[str, nn.Module]],
        prompt_key: torch.Tensor,
        module_input_keys: dict[str, torch.Tensor] | None = None,
    ) -> None:
        pending: dict[str, torch.Tensor] = {}
        for name, mod in target_modules:
            snapshot = self._weight_snapshots.get(name)
            rows = self._edit_row_indices.get(name)
            if snapshot is None or rows is None:
                raise ValueError("Semantic-history SSR requires a frozen per-edit snapshot and fixed rows")
            delta = mod.weight.detach().float() - snapshot.to(mod.weight.device, dtype=torch.float32)
            signature = F.normalize(delta.index_select(0, rows.to(delta.device)).mean(dim=0), dim=0).cpu()
            pending[name] = signature
            if self.history_projection_mode != "none":
                if module_input_keys is None or name not in module_input_keys:
                    raise ValueError("History projection is missing frozen module inputs")
        for name, signature in pending.items():
            self._history_update_signatures.setdefault(name, []).append(signature)
            if self.history_projection_mode != "none":
                self._history_module_inputs.setdefault(name, []).append(module_input_keys[name])
        self._history_prompt_keys.append(prompt_key.detach().cpu())
        if len(self._history_prompt_keys) > self.history_limit:
            self._history_prompt_keys.pop(0)
            for signatures in self._history_update_signatures.values():
                signatures.pop(0)
            for vectors in self._history_module_inputs.values():
                vectors.pop(0)
        if any(len(signatures) != len(self._history_prompt_keys) for signatures in self._history_update_signatures.values()):
            raise ValueError("Semantic-history update bank is inconsistent")
        if any(len(vectors) != len(self._history_prompt_keys) for vectors in self._history_module_inputs.values()):
            raise ValueError("Semantic-history protection bank is inconsistent")

    def edit(self, prompt: str, target_new: str) -> dict:
        """Apply a single knowledge edit with SSR."""
        self.model.train()
        prompt = _to_text(prompt)
        target_new = _to_text(target_new)
        prompt_key = self._prompt_semantic_key(prompt) if self.ssr_mode == "semantic_history_delta_v1" else None

        target_modules = self._get_mlp_modules()
        if not target_modules:
            logger.warning("No target MLP modules found")
            return {"success": False, "reason": "no target MLP modules found"}

        module_input_keys = (
            self._module_input_keys(prompt, target_modules)
            if self.history_projection_mode != "none"
            else None
        )

        params = []
        for name, mod in target_modules:
            mod.weight.requires_grad_(True)
            params.append(mod.weight)

        optimizer = torch.optim.Adam(params, lr=self.lr)
        self._prepare_fixed_rows(target_modules)

        text = f"{prompt} {target_new}"
        tokens = self._tokenize_text(text, padding=True)
        prompt_tokens = self._tokenize_text(prompt, padding=False)
        prompt_len = prompt_tokens["input_ids"].shape[1]

        labels = tokens["input_ids"].clone()
        labels[:, :prompt_len] = -100

        for step in range(self.num_steps):
            optimizer.zero_grad(set_to_none=True)
            outputs = self.model(**tokens, labels=labels)
            ce_loss = outputs.loss

            biocs_loss = torch.tensor(0.0, device=self.device)
            spectral_loss = torch.tensor(0.0, device=self.device)
            anchor_loss = torch.tensor(0.0, device=self.device)

            with torch.amp.autocast("cuda", enabled=False):
                for name, mod in target_modules:
                    if self.use_ssr:
                        spatial_target = mod.weight.float()
                        if self.biocs_target == "delta" and name in self._weight_snapshots:
                            spatial_target = spatial_target - self._weight_snapshots[name].to(mod.weight.device).float()
                        row_indices = self._edit_row_indices.get(name)
                        if self.ssr_mode == "semantic_history_delta_v1":
                            if row_indices is None or prompt_key is None:
                                raise ValueError("Semantic-history SSR is missing its fixed rows or prompt key")
                            term, _ = semantic_history_delta_penalty(
                                spatial_target, prompt_key, self._history_prompt_keys,
                                self._history_update_signatures.get(name, []),
                                row_indices=row_indices,
                                neighbors=self.history_neighbors,
                                repulsion_margin=self.history_margin,
                                attraction_weight=self.a_exc,
                                repulsion_weight=self.a_inh,
                                neighbor_policy=self.history_neighbor_policy,
                                kernel_mode=self.history_kernel_mode,
                                kernel_family=self.kernel_family,
                                sigma_exc=self.sigma_exc,
                                sigma_inh=self.sigma_inh,
                            )
                            biocs_loss = biocs_loss + term
                        elif self.row_selection != "active_top" or row_indices is not None:
                            biocs_loss = biocs_loss + compute_spatial_biocs_llm(
                                spatial_target,
                                A_exc=self.a_exc,
                                A_inh=self.a_inh,
                                sigma_exc=self.sigma_exc,
                                sigma_inh=self.sigma_inh,
                                kernel_family=self.kernel_family,
                                distance_metric=self.distance_metric,
                                row_indices=row_indices,
                                max_rows=self.max_spatial_rows,
                            )
                    if self.use_spectral:
                        spectral_loss = spectral_loss + compute_spectral_flatness_llm(
                            mod.weight.float()
                        )
                    if self.use_anchor and name in self._weight_snapshots:
                        anchor_loss = anchor_loss + F.mse_loss(
                            mod.weight.float(), self._weight_snapshots[name].to(mod.weight.device).float()
                        )

            auxiliary_loss = (
                self.lambda_biocs * biocs_loss
                + self.lambda_spectral * spectral_loss
                + self.lambda_anchor * anchor_loss
            )
            loss = (
                ce_loss
                + auxiliary_loss
            )
            primary_conflict_removed = 0.0
            if self.primary_gradient_mode == "pcgrad_v1" and auxiliary_loss.requires_grad:
                main_grads = torch.autograd.grad(
                    ce_loss, params, retain_graph=True, allow_unused=True,
                )
                auxiliary_grads = torch.autograd.grad(
                    auxiliary_loss, params, allow_unused=True,
                )
                main_norm_sq = ce_loss.new_zeros(()) + sum(
                    grad.float().square().sum() for grad in main_grads if grad is not None
                )
                auxiliary_norm_sq = ce_loss.new_zeros(()) + sum(
                    grad.float().square().sum() for grad in auxiliary_grads if grad is not None
                )
                if main_norm_sq.item() > 1e-12 and auxiliary_norm_sq.item() > 1e-12:
                    dot = ce_loss.new_zeros(()) + sum(
                        (main.float() * auxiliary.float()).sum()
                        for main, auxiliary in zip(main_grads, auxiliary_grads)
                        if main is not None and auxiliary is not None
                    )
                    coefficient = torch.minimum(dot, dot.new_zeros(())) / main_norm_sq
                    primary_conflict_removed = float((-coefficient).detach().clamp_min(0.0))
                    for parameter, main, auxiliary in zip(params, main_grads, auxiliary_grads):
                        base = torch.zeros_like(parameter) if main is None else main
                        regularizer = torch.zeros_like(parameter) if auxiliary is None else auxiliary
                        parameter.grad = base + regularizer - coefficient * base
                else:
                    ce_loss.backward()
            else:
                loss.backward()
            projection_stats = []
            if self.history_projection_mode == "semantic_topk":
                if prompt_key is None:
                    raise ValueError("History projection requires a semantic prompt key")
                for name, mod in target_modules:
                    if mod.weight.grad is None:
                        continue
                    projected, diagnostics = project_history_gradient(
                        mod.weight.grad,
                        prompt_key,
                        self._history_prompt_keys,
                        self._history_module_inputs.get(name, []),
                        neighbors=self.history_neighbors,
                    )
                    mod.weight.grad.copy_(projected)
                    projection_stats.append(diagnostics)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            if step == 0:
                self._capture_active_rows(target_modules)

        for name, mod in target_modules:
            mod.weight.requires_grad_(False)

        self.model.eval()
        if self.ssr_mode == "semantic_history_delta_v1" and prompt_key is not None:
            self._record_history_update(target_modules, prompt_key, module_input_keys)
        self._save_weight_snapshot()
        self._edit_index += 1
        projection_fraction = (
            sum(item["gradient_removed_fraction"] for item in projection_stats) / len(projection_stats)
            if projection_stats else 0.0
        )
        return {
            "success": True,
            "final_loss": loss.item(),
            "history_projection_mode": self.history_projection_mode,
            "history_projection_removed_fraction": projection_fraction,
            "primary_gradient_mode": self.primary_gradient_mode,
            "primary_gradient_conflict_removed": primary_conflict_removed,
        }

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: Optional[int] = None) -> str:
        tokens = self._tokenize_text(prompt, padding=False)
        out = self.model.generate(
            **tokens, max_new_tokens=max_new_tokens or self.max_new_tokens,
            do_sample=False, pad_token_id=self.tokenizer.pad_token_id,
        )
        return self.tokenizer.decode(out[0][tokens["input_ids"].shape[1]:], skip_special_tokens=True)

    def evaluate_edit(
        self,
        prompt: str,
        target_new: str,
        ground_truth,
        locality: dict,
        rephrase_prompts: Optional[list[str]] = None,
        portability: Optional[dict] = None,
    ) -> dict:
        """Evaluate immediate efficacy plus rephrase and portability probes.

        ``locality`` remains the locked historical post-edit relation score for
        comparability with earlier pilot runs.  It is intentionally reported as
        an auxiliary diagnostic, not as the pre-edit preservation metric.
        """
        prompt = _to_text(prompt)
        target_str = _to_text(target_new)
        prediction = self.generate(prompt).strip()
        efficacy = _target_match(prediction, target_str)

        locality_result = evaluate_locality(locality, self.generate)
        rephrase_evaluations = [
            _target_match(self.generate(rephrase).strip(), target_str)
            for rephrase in (rephrase_prompts or [])
        ]
        portability_result = _evaluate_target_groups(portability or {}, self.generate)

        return {
            "efficacy": efficacy,
            "locality": locality_result["score"],
            "locality_role": "auxiliary_post_edit_relation_consistency",
            "locality_items_evaluated": locality_result["n_items"],
            "locality_items_matched": locality_result["n_matched"],
            "locality_evaluator": locality_result["evaluator"],
            "locality_evaluator_version": locality_result["evaluator_version"],
            "rephrase": sum(rephrase_evaluations) / max(len(rephrase_evaluations), 1),
            "rephrase_items_evaluated": len(rephrase_evaluations),
            "portability": portability_result["score"],
            "portability_items_evaluated": portability_result["n_items"],
            "portability_items_matched": portability_result["n_matched"],
            "prediction": prediction[:100],
        }


# -------------------- FT baseline editor (no SSR) -------------

class FTBaselineEditor(BioCsLLMEditor):
    """Standard fine-tuning editor without SSR."""

    def __init__(self, *args, **kwargs):
        kwargs["recipe"] = "plain"
        kwargs["lambda_ssr"] = 0.0
        kwargs["lambda_biocs"] = 0.0
        kwargs["lambda_spectral"] = 0.0
        kwargs["lambda_anchor"] = 0.0
        super().__init__(*args, **kwargs)


# ──────────────────── Main pipeline ──────────────────────────────

def _history_sample_positions(history_size: int, max_samples: int) -> list[int]:
    """Choose a deterministic, approximately uniform subset of edit history."""
    if history_size <= 0:
        return []
    if max_samples <= 0 or max_samples >= history_size:
        return list(range(history_size))
    if max_samples == 1:
        return [history_size - 1]

    positions = {
        round(index * (history_size - 1) / (max_samples - 1))
        for index in range(max_samples)
    }
    return sorted(positions)


@torch.no_grad()
def evaluate_historical_retention(
    editor: BioCsLLMEditor,
    history: list[dict],
    *,
    after_edits: int,
    max_samples: int = 0,
) -> dict:
    """Re-evaluate earlier edits under the current, cumulatively edited model."""
    positions = _history_sample_positions(len(history), max_samples)
    evaluations = []
    for position in positions:
        entry = history[position]
        sample = entry["sample"]
        retained = editor.evaluate_edit(
            sample["prompt"],
            sample["target_new"],
            sample["ground_truth"],
            sample["locality"],
            sample.get("rephrase_prompts"),
            sample.get("portability"),
        )
        evaluations.append(
            {
                "history_position": position,
                "idx": entry["idx"],
                "edit_age": after_edits - position - 1,
                "immediate_efficacy": entry["immediate_efficacy"],
                "retained_efficacy": retained["efficacy"],
                "immediate_locality": entry["immediate_locality"],
                "retained_locality": retained["locality"],
            }
        )

    count = len(evaluations)
    immediate_efficacy = sum(row["immediate_efficacy"] for row in evaluations) / max(count, 1)
    retained_efficacy = sum(row["retained_efficacy"] for row in evaluations) / max(count, 1)
    immediate_locality = sum(row["immediate_locality"] for row in evaluations) / max(count, 1)
    retained_locality = sum(row["retained_locality"] for row in evaluations) / max(count, 1)
    return {
        "after_edits": after_edits,
        "history_size": len(history),
        "n_evaluated": count,
        "history_positions": positions,
        "efficacy": retained_efficacy * 100.0,
        "locality": retained_locality * 100.0,
        "immediate_efficacy_on_same_items": immediate_efficacy * 100.0,
        "immediate_locality_on_same_items": immediate_locality * 100.0,
        "efficacy_delta_vs_immediate": (retained_efficacy - immediate_efficacy) * 100.0,
        "locality_delta_vs_immediate": (retained_locality - immediate_locality) * 100.0,
        "evaluations": evaluations,
    }


@torch.no_grad()
def capture_pre_edit_responses(editor: BioCsLLMEditor, probes: list[dict]) -> dict:
    """Freeze compact pre-edit next-token references before any edit."""
    references = []
    for probe in probes:
        prompt = _to_text(probe["prompt"])
        probabilities = _next_token_distribution(editor, prompt)
        top_k = min(PRE_EDIT_TOP_K, probabilities.numel())
        values, token_ids = torch.topk(probabilities, k=top_k, largest=True, sorted=True)
        references.append(
            {
                "probe_id": str(probe["probe_id"]),
                "token_ids": token_ids.cpu().tolist(),
                "probabilities": values.cpu().tolist(),
                "residual_probability": float((1.0 - values.sum()).clamp_min(0.0).item()),
            }
        )
    return {
        "references": references,
        "n_probes": len(references),
        "n_eligible": len(references),
        "n_empty_pre_edit": 0,
        "top_k": PRE_EDIT_TOP_K,
        "evaluator": PRE_EDIT_EVALUATOR,
        "evaluator_version": PRE_EDIT_EVALUATOR_VERSION,
        "protocol_hash": PRE_EDIT_PROTOCOL_HASH,
    }


@torch.no_grad()
def evaluate_pre_edit_consistency(
    editor: BioCsLLMEditor,
    probes: list[dict],
    baseline: dict,
    *,
    after_edits: int,
) -> dict:
    """Score next-token distribution preservation against frozen references."""
    references = baseline["references"]
    if len(probes) != len(references):
        raise ValueError("Pre-edit probes and frozen references must stay one-to-one")
    evaluations = []
    for probe, reference in zip(probes, references):
        if str(probe["probe_id"]) != reference["probe_id"]:
            raise ValueError("Pre-edit probe order changed after baseline capture")
        probabilities = _next_token_distribution(editor, _to_text(probe["prompt"]))
        token_ids = torch.tensor(reference["token_ids"], device=probabilities.device, dtype=torch.long)
        before = torch.tensor(
            [*reference["probabilities"], reference["residual_probability"]],
            device=probabilities.device,
            dtype=torch.float32,
        )
        after_top = probabilities.index_select(0, token_ids)
        after = torch.cat((after_top, (1.0 - after_top.sum()).clamp_min(0.0).reshape(1)))
        midpoint = 0.5 * (before + after)
        epsilon = torch.finfo(torch.float32).tiny
        js_divergence = 0.5 * (
            (before * (before.clamp_min(epsilon).log() - midpoint.clamp_min(epsilon).log())).sum()
            + (after * (after.clamp_min(epsilon).log() - midpoint.clamp_min(epsilon).log())).sum()
        )
        similarity = float((1.0 - js_divergence / 0.6931471805599453).clamp(0.0, 1.0).item())
        evaluations.append(
            {
                "probe_id": reference["probe_id"],
                "eligible": True,
                "js_similarity": similarity,
                "js_divergence": float(js_divergence.item()),
            }
        )
    return {
        "after_edits": after_edits,
        "score": sum(item["js_similarity"] for item in evaluations) / max(len(evaluations), 1),
        "n_probes": len(evaluations),
        "n_scored": len(evaluations),
        "n_skipped_empty_pre_edit": 0,
        "evaluator": PRE_EDIT_EVALUATOR,
        "evaluator_version": PRE_EDIT_EVALUATOR_VERSION,
        "protocol_hash": PRE_EDIT_PROTOCOL_HASH,
        "evaluations": evaluations,
    }


def run_sequential_editing(
    editor: BioCsLLMEditor,
    dataset: KnowEditDataset,
    n_edits: int = 100,
    output_path: str = "results.json",
    *,
    evaluate_history: bool = False,
    history_checkpoints: Optional[list[int]] = None,
    history_max_samples: int = 0,
    pre_edit_probes: Optional[list[dict]] = None,
    pre_edit_checkpoints: Optional[list[int]] = None,
):
    """Run a sequential editing stream with pre-edit preservation controls."""
    if n_edits <= 0:
        raise ValueError("n_edits must be positive")
    if history_max_samples < 0:
        raise ValueError("history_max_samples must be non-negative; use 0 for all edits")
    checkpoints = sorted(set(history_checkpoints or []))
    if any(checkpoint <= 0 for checkpoint in checkpoints):
        raise ValueError("history_checkpoints must contain positive edit counts")
    preservation_checkpoints = sorted(set(pre_edit_checkpoints or []))
    if any(checkpoint <= 0 for checkpoint in preservation_checkpoints):
        raise ValueError("pre_edit_checkpoints must contain positive edit counts")

    total_edits = min(n_edits, len(dataset))
    if evaluate_history and any(checkpoint > total_edits for checkpoint in checkpoints):
        raise ValueError(
            f"history checkpoint exceeds the executed stream length ({total_edits}): "
            f"{checkpoints}"
        )
    if any(checkpoint > total_edits for checkpoint in preservation_checkpoints):
        raise ValueError(
            f"pre-edit checkpoint exceeds the executed stream length ({total_edits}): "
            f"{preservation_checkpoints}"
        )

    probes = list(pre_edit_probes or [])
    for probe in probes:
        if not _to_text(probe.get("prompt", "")) or "probe_id" not in probe:
            raise ValueError("Each pre-edit probe needs a non-empty prompt and stable probe_id")
    pre_edit_baseline = capture_pre_edit_responses(editor, probes) if probes else None
    pre_edit_results = []

    results = []
    agg = {"efficacy": [], "locality": [], "rephrase": [], "portability": []}
    history = []
    checkpoint_results = []
    failures = []

    cuda_device = editor.input_device if editor.input_device.type == "cuda" else None
    if cuda_device is not None:
        torch.cuda.synchronize(cuda_device)
        torch.cuda.reset_peak_memory_stats(cuda_device)
    start_time = __import__("time").time()
    for i in tqdm(range(total_edits), desc="Editing"):
        sample = dataset[i]
        edit_result = editor.edit(sample["prompt"], sample["target_new"])

        if edit_result.get("success"):
            eval_result = editor.evaluate_edit(
                sample["prompt"],
                sample["target_new"],
                sample["ground_truth"],
                sample["locality"],
                sample.get("rephrase_prompts"),
                sample.get("portability"),
            )
            results.append({
                "idx": dataset.offset + i,
                "prompt": sample["prompt"],
                "target_new": sample["target_new"],
                **eval_result,
            })
            agg["efficacy"].append(eval_result["efficacy"])
            agg["locality"].append(eval_result["locality"])
            if eval_result["rephrase_items_evaluated"]:
                agg["rephrase"].append(eval_result["rephrase"])
            if eval_result["portability_items_evaluated"]:
                agg["portability"].append(eval_result["portability"])

            if evaluate_history:
                history.append(
                    {
                        "idx": dataset.offset + i,
                        "sample": sample,
                        "immediate_efficacy": eval_result["efficacy"],
                        "immediate_locality": eval_result["locality"],
                    }
                )

        else:
            failures.append(
                {
                    "idx": dataset.offset + i,
                    "reason": str(edit_result.get("reason", "editor returned success=False")),
                }
            )

        if (i + 1) % 10 == 0 and agg["efficacy"]:
            eff = sum(agg["efficacy"]) / len(agg["efficacy"]) * 100
            loc = sum(agg["locality"]) / len(agg["locality"]) * 100
            logger.info(f"Edit {i+1}: Efficacy={eff:.1f}%, Locality={loc:.1f}%")

        if evaluate_history and (i + 1) in checkpoints:
            snapshot = evaluate_historical_retention(
                editor,
                history,
                after_edits=i + 1,
                max_samples=history_max_samples,
            )
            checkpoint_results.append(snapshot)
            logger.info(
                "Historical retention after %d edits: efficacy=%.1f%%, locality=%.1f%% (%d/%d items)",
                i + 1,
                snapshot["efficacy"],
                snapshot["locality"],
                snapshot["n_evaluated"],
                snapshot["history_size"],
            )

        if pre_edit_baseline is not None and (i + 1) in preservation_checkpoints:
            preservation = evaluate_pre_edit_consistency(
                editor,
                probes,
                pre_edit_baseline,
                after_edits=i + 1,
            )
            pre_edit_results.append(preservation)
            logger.info(
                "Pre-edit output preservation after %d edits: %.1f%% (%d probes)",
                i + 1,
                preservation["score"] * 100.0,
                preservation["n_scored"],
            )

    if cuda_device is not None:
        torch.cuda.synchronize(cuda_device)
    succeeded = len(results)
    attempted = total_edits
    failed = attempted - succeeded
    status = "complete" if succeeded == n_edits and failed == 0 else (
        "failed" if attempted > 0 and succeeded == 0 else "incomplete"
    )
    summary = {
        **editor.metadata(),
        # ``n_edits`` retains its historical meaning (successful edits).
        "n_edits": succeeded,
        "requested_n_edits": n_edits,
        "attempted": attempted,
        "succeeded": succeeded,
        "failed": failed,
        "status": status,
        "elapsed_s": round(__import__("time").time() - start_time, 1),
        "efficacy": sum(agg["efficacy"]) / max(len(agg["efficacy"]), 1) * 100,
        "rephrase": sum(agg["rephrase"]) / max(len(agg["rephrase"]), 1) * 100,
        "portability": sum(agg["portability"]) / max(len(agg["portability"]), 1) * 100,
        "locality": sum(agg["locality"]) / max(len(agg["locality"]), 1) * 100,
        "locality_role": "auxiliary_post_edit_relation_consistency",
        "data_offset": dataset.offset,
        "data_indices_sha256": dataset.source_indices_sha256,
        "data_selection": "index_manifest" if dataset.source_indices != list(
            range(dataset.offset, dataset.offset + len(dataset))
        ) else "contiguous_offset",
        "peak_cuda_allocated_mb": (
            torch.cuda.max_memory_allocated(cuda_device) / (1024**2)
            if cuda_device is not None else 0.0
        ),
        "peak_cuda_reserved_mb": (
            torch.cuda.max_memory_reserved(cuda_device) / (1024**2)
            if cuda_device is not None else 0.0
        ),
        "results": results,
        "failures": failures,
    }

    if evaluate_history:
        final_checkpoint = next(
            (
                snapshot
                for snapshot in checkpoint_results
                if snapshot["after_edits"] == total_edits
            ),
            None,
        )
        if final_checkpoint is None:
            final_checkpoint = evaluate_historical_retention(
                editor,
                history,
                after_edits=total_edits,
                max_samples=history_max_samples,
            )
        summary["historical_retention"] = {
            "enabled": True,
            "history_max_samples": history_max_samples,
            "requested_checkpoints": checkpoints,
            "checkpoints": checkpoint_results,
            "final": final_checkpoint,
        }

    if pre_edit_baseline is not None:
        final_preservation = next(
            (
                snapshot
                for snapshot in pre_edit_results
                if snapshot["after_edits"] == total_edits
            ),
            None,
        )
        if final_preservation is None:
            final_preservation = evaluate_pre_edit_consistency(
                editor,
                probes,
                pre_edit_baseline,
                after_edits=total_edits,
            )
        summary["pre_edit_output_consistency"] = {
            "enabled": True,
            "probe_protocol": {
                key: value
                for key, value in pre_edit_baseline.items()
                if key != "references"
            },
            "requested_checkpoints": preservation_checkpoints,
            "checkpoints": pre_edit_results,
            "final": final_preservation,
        }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to {output_path}")
    logger.info(
        "Final: Efficacy=%.1f%%, rephrase=%.1f%%, portability=%.1f%%, "
        "auxiliary locality=%.1f%%",
        summary["efficacy"],
        summary["rephrase"],
        summary["portability"],
        summary["locality"],
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description="SSR LLM knowledge editing")
    parser.add_argument("--model", type=str, default="gpt2",
                        help="HuggingFace model name or path")
    parser.add_argument("--data", type=str,
                        default="dataset/knowedit/benchmark/ZsRE/ZsRE-test-all.json")
    parser.add_argument("--method", type=str, default="biocs", choices=["biocs", "ft"],
                        help="biocs: SSR-regularized FT; ft: vanilla fine-tuning")
    parser.add_argument("--n_edits", type=int, default=100)
    parser.add_argument("--lambda_biocs", type=float, default=0.001)
    parser.add_argument("--lambda_spectral", type=float, default=0.01)
    parser.add_argument("--lambda_anchor", type=float, default=0.001)
    parser.add_argument("--kernel_family", choices=KERNEL_FAMILIES, default="gaussian")
    parser.add_argument("--a_exc", type=float, default=1.0)
    parser.add_argument("--a_inh", type=float, default=0.8)
    parser.add_argument("--sigma_exc", type=float, default=0.3)
    parser.add_argument("--sigma_inh", type=float, default=0.8)
    parser.add_argument("--biocs_target", choices=["weight", "delta"], default="weight")
    parser.add_argument("--distance_metric", choices=DISTANCE_METRICS, default="cosine")
    parser.add_argument("--row_selection", choices=ROW_SELECTIONS, default="fixed_random")
    parser.add_argument("--max_spatial_rows", type=int, default=256)
    parser.add_argument("--sampler_seed", type=int, default=0)
    parser.add_argument("--data_offset", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_steps", type=int, default=25)
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--target_layers", type=int, nargs="*", default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default="results/llm_ke")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

    EditorClass = BioCsLLMEditor if args.method == "biocs" else FTBaselineEditor
    editor = EditorClass(
        model_name=args.model,
        device=args.device,
        target_layers=args.target_layers,
        lambda_biocs=args.lambda_biocs,
        lambda_spectral=args.lambda_spectral,
        lambda_anchor=args.lambda_anchor,
        kernel_family=args.kernel_family,
        a_exc=args.a_exc,
        a_inh=args.a_inh,
        sigma_exc=args.sigma_exc,
        sigma_inh=args.sigma_inh,
        biocs_target=args.biocs_target,
        distance_metric=args.distance_metric,
        row_selection=args.row_selection,
        max_spatial_rows=args.max_spatial_rows,
        sampler_seed=args.sampler_seed,
        lr=args.lr,
        num_steps=args.num_steps,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
    )

    dataset = KnowEditDataset(args.data, max_samples=args.n_edits, offset=args.data_offset)

    out_dir = Path(args.output) / f"{args.method}_{Path(args.model).name}"
    run_sequential_editing(editor, dataset, args.n_edits, str(out_dir / "results.json"))


if __name__ == "__main__":
    main()
