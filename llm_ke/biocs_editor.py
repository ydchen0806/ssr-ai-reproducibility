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

# -------------------- SSR core functions ----------------------

KERNEL_FAMILIES = ("gaussian", "laplace", "cauchy", "inverse")
DISTANCE_METRICS = ("cosine", "projective")
ROW_SELECTIONS = ("random_each_step", "fixed_random", "active_top")

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
    P = (inh - exc) + (A_exc - A_inh)
    P = P - torch.diag(torch.diag(P))
    return P.sum() / (n * (n - 1) + 1e-8)


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

    def __init__(self, json_path: str, max_samples: int = 500, offset: int = 0):
        with open(json_path) as f:
            raw = json.load(f)
        if offset < 0:
            raise ValueError("offset must be non-negative")
        self.offset = offset
        self.data = raw[offset : offset + max_samples]
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
            "Loaded %d editing samples from %s at offset %d",
            len(self.data), json_path, offset,
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
    return {
        "prompt": _to_text(prompt),
        "target_new": _to_text(target_new),
        "ground_truth": _to_text(item.get("ground_truth", "")),
        "subject": _to_text(subject),
        "locality": locality,
        "portability": portability,
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

    def edit(self, prompt: str, target_new: str) -> dict:
        """Apply a single knowledge edit with SSR."""
        self.model.train()
        prompt = _to_text(prompt)
        target_new = _to_text(target_new)

        target_modules = self._get_mlp_modules()
        if not target_modules:
            logger.warning("No target MLP modules found")
            return {"success": False, "reason": "no target MLP modules found"}

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
                        if self.row_selection != "active_top" or row_indices is not None:
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

            loss = (
                ce_loss
                + self.lambda_biocs * biocs_loss
                + self.lambda_spectral * spectral_loss
                + self.lambda_anchor * anchor_loss
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            if step == 0:
                self._capture_active_rows(target_modules)

        for name, mod in target_modules:
            mod.weight.requires_grad_(False)

        self.model.eval()
        self._save_weight_snapshot()
        self._edit_index += 1
        return {"success": True, "final_loss": loss.item()}

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: Optional[int] = None) -> str:
        tokens = self._tokenize_text(prompt, padding=False)
        out = self.model.generate(
            **tokens, max_new_tokens=max_new_tokens or self.max_new_tokens,
            do_sample=False, pad_token_id=self.tokenizer.pad_token_id,
        )
        return self.tokenizer.decode(out[0][tokens["input_ids"].shape[1]:], skip_special_tokens=True)

    def evaluate_edit(self, prompt: str, target_new: str, ground_truth, locality: dict) -> dict:
        """Evaluate a single edit: efficacy, locality, generalization."""
        prompt = _to_text(prompt)
        target_str = _to_text(target_new)
        prediction = self.generate(prompt).strip()
        efficacy = 1.0 if target_str.lower() in prediction.lower() else 0.0

        locality_result = evaluate_locality(locality, self.generate)

        return {
            "efficacy": efficacy,
            "locality": locality_result["score"],
            "locality_items_evaluated": locality_result["n_items"],
            "locality_items_matched": locality_result["n_matched"],
            "locality_evaluator": locality_result["evaluator"],
            "locality_evaluator_version": locality_result["evaluator_version"],
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
        )
        evaluations.append(
            {
                "history_position": position,
                "idx": entry["idx"],
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


def run_sequential_editing(
    editor: BioCsLLMEditor,
    dataset: KnowEditDataset,
    n_edits: int = 100,
    output_path: str = "results.json",
    *,
    evaluate_history: bool = False,
    history_checkpoints: Optional[list[int]] = None,
    history_max_samples: int = 0,
):
    """Run sequential knowledge editing and evaluate."""
    if n_edits <= 0:
        raise ValueError("n_edits must be positive")
    if history_max_samples < 0:
        raise ValueError("history_max_samples must be non-negative; use 0 for all edits")
    checkpoints = sorted(set(history_checkpoints or []))
    if any(checkpoint <= 0 for checkpoint in checkpoints):
        raise ValueError("history_checkpoints must contain positive edit counts")

    total_edits = min(n_edits, len(dataset))
    if evaluate_history and any(checkpoint > total_edits for checkpoint in checkpoints):
        raise ValueError(
            f"history checkpoint exceeds the executed stream length ({total_edits}): "
            f"{checkpoints}"
        )

    results = []
    agg = {"efficacy": [], "locality": []}
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
            )
            results.append({
                "idx": dataset.offset + i,
                "prompt": sample["prompt"],
                "target_new": sample["target_new"],
                **eval_result,
            })
            agg["efficacy"].append(eval_result["efficacy"])
            agg["locality"].append(eval_result["locality"])

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
        "locality": sum(agg["locality"]) / max(len(agg["locality"]), 1) * 100,
        "data_offset": dataset.offset,
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

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to {output_path}")
    logger.info(f"Final: Efficacy={summary['efficacy']:.1f}%, Locality={summary['locality']:.1f}%")
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
