"""SSR Knowledge Editing for LLMs.

Applies biologically-inspired center-surround regularization to LLM weight
editing. When fine-tuning specific MLP layers to inject new knowledge, SSR
constrains the weight update so that "nearby" neurons suppress each other's
growth — preserving existing knowledge (locality) while allowing targeted edits
(efficacy).

Works with any causal LM via HuggingFace transformers. Uses EasyEdit's
KnowEdit dataset format for evaluation.
"""

import sys
import json
import logging
import argparse
import os
import re
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _to_text(value) -> str:
    """Normalize KnowEdit answer fields to plain strings."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("str", "text", "answer", "label", "name"):
            if key in value:
                return _to_text(value[key])
        return str(value)
    if isinstance(value, (list, tuple)):
        if not value:
            return ""
        return _to_text(value[0])
    return str(value)


def _to_text_list(value) -> list[str]:
    """Normalize nested KnowEdit labels into a flat list of strings."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        texts = []
        for item in value:
            texts.extend(_to_text_list(item))
        return [text for text in texts if text]
    text = _to_text(value)
    return [text] if text else []

# ──────────────────── SSR core functions ───────────────────────

def compute_spatial_biocs_llm(
    weights: torch.Tensor,
    A_exc: float = 1.0,
    A_inh: float = 0.8,
    sigma_exc: float = 0.3,
    sigma_inh: float = 0.8,
) -> torch.Tensor:
    """Mexican-hat center-surround penalty for LLM weight matrices.

    For large matrices, operates on a random subset of rows to keep
    O(n^2) computation tractable.
    """
    w = weights.float()
    n = w.size(0)

    MAX_ROWS = 256
    if n > MAX_ROWS:
        idx = torch.randperm(n, device=w.device)[:MAX_ROWS]
        w = w[idx]
        n = MAX_ROWS

    w_norm = F.normalize(w, dim=1)
    cos_sim = torch.clamp(w_norm @ w_norm.T, -1.0, 1.0)
    dist = torch.sqrt(torch.clamp(1.0 - cos_sim, min=1e-8))

    exc = A_exc * torch.exp(-(dist ** 2) / (2 * sigma_exc ** 2))
    inh = A_inh * torch.exp(-(dist ** 2) / (2 * sigma_inh ** 2))
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

    def __init__(self, json_path: str, max_samples: int = 500):
        with open(json_path) as f:
            raw = json.load(f)
        self.data = raw[:max_samples]
        logger.info(f"Loaded {len(self.data)} editing samples from {json_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "prompt": _to_text(item["prompt"]),
            "target_new": _to_text(item.get("target_new", "")),
            "ground_truth": _to_text(item.get("ground_truth", "")),
            "subject": _to_text(item.get("subject", "")),
            "locality": item.get("locality", {}),
            "portability": item.get("portability", {}),
        }


# ──────────────────── SSR Editor ──────────────────────────────

class BioCsLLMEditor:
    """Fine-tune specific MLP layers with SSR regularization.

    Strategy:
      1. For each edit request, fine-tune target MLP layers to produce
         the new target while constraining weights via SSR
      2. SSR penalty prevents excessive weight modification, preserving
         locality (unrelated knowledge stays intact)
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
        lr: float = 1e-4,
        num_steps: int = 25,
        max_length: int = 64,
    ):
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
        self.lambda_biocs = lambda_biocs
        self.lambda_spectral = lambda_spectral
        self.lambda_anchor = lambda_anchor
        self.lr = lr
        self.num_steps = num_steps
        self.max_length = max_length
        self.target_module_regex = os.environ.get("BIOCS_TARGET_MODULE_REGEX", r"(c_proj|down_proj)$").strip()
        self.max_target_modules = int(os.environ.get("BIOCS_MAX_TARGET_MODULES", "0"))

        self._weight_snapshots: dict[str, torch.Tensor] = {}
        self._save_weight_snapshot()

        logger.info(f"Target layers: {self.target_layers}")
        logger.info(f"λ_biocs={lambda_biocs}, λ_spectral={lambda_spectral}, λ_anchor={lambda_anchor}")

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
            "lambda_biocs": self.lambda_biocs,
            "lambda_spectral": self.lambda_spectral,
            "lambda_anchor": self.lambda_anchor,
            "lr": self.lr,
            "num_steps": self.num_steps,
            "max_length": self.max_length,
            "seed": os.environ.get("KE_SEED", ""),
            "model_dtype": os.environ.get("KE_MODEL_DTYPE", "auto"),
            "device_map": os.environ.get("KE_DEVICE_MAP", ""),
            "force_text_only": os.environ.get("KE_FORCE_TEXT_ONLY", "0"),
        }

    def _save_weight_snapshot(self):
        for name, mod in self._get_mlp_modules():
            self._weight_snapshots[name] = mod.weight.data.clone()

    def edit(self, prompt: str, target_new: str) -> dict:
        """Apply a single knowledge edit with SSR regularization."""
        self.model.train()
        prompt = _to_text(prompt)
        target_new = _to_text(target_new)

        target_modules = self._get_mlp_modules()
        if not target_modules:
            logger.warning("No target MLP modules found")
            return {"success": False}

        params = []
        for name, mod in target_modules:
            mod.weight.requires_grad_(True)
            params.append(mod.weight)

        optimizer = torch.optim.Adam(params, lr=self.lr)

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
                    biocs_loss = biocs_loss + compute_spatial_biocs_llm(
                        mod.weight.float()
                    )
                    spectral_loss = spectral_loss + compute_spectral_flatness_llm(
                        mod.weight.float()
                    )
                    if self.lambda_anchor > 0 and name in self._weight_snapshots:
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

        for name, mod in target_modules:
            mod.weight.requires_grad_(False)

        self.model.eval()
        self._save_weight_snapshot()
        return {"success": True, "final_loss": loss.item()}

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 32) -> str:
        tokens = self._tokenize_text(prompt, padding=False)
        out = self.model.generate(
            **tokens, max_new_tokens=max_new_tokens,
            do_sample=False, pad_token_id=self.tokenizer.pad_token_id,
        )
        return self.tokenizer.decode(out[0][tokens["input_ids"].shape[1]:], skip_special_tokens=True)

    def evaluate_edit(self, prompt: str, target_new: str, ground_truth, locality: dict) -> dict:
        """Evaluate a single edit: efficacy, locality, generalization."""
        prompt = _to_text(prompt)
        target_str = _to_text(target_new)
        prediction = self.generate(prompt).strip()
        efficacy = 1.0 if target_str.lower() in prediction.lower() else 0.0

        locality_score = 0.0
        locality_count = 0
        if locality:
            for key, items in locality.items():
                if not isinstance(items, list):
                    continue
                for item in items:
                    if "prompt" not in item or "ground_truth" not in item:
                        continue
                    loc_pred = self.generate(_to_text(item["prompt"])).strip()
                    gt_strs = _to_text_list(item["ground_truth"])
                    match = any(g.lower() in loc_pred.lower() for g in gt_strs)
                    locality_score += float(match)
                    locality_count += 1

        return {
            "efficacy": efficacy,
            "locality": locality_score / max(locality_count, 1),
            "prediction": prediction[:100],
        }


# ──────────────────── FT Baseline Editor (no SSR) ─────────────

class FTBaselineEditor(BioCsLLMEditor):
    """Standard fine-tuning editor without SSR regularization."""

    def __init__(self, *args, **kwargs):
        kwargs["lambda_biocs"] = 0.0
        kwargs["lambda_spectral"] = 0.0
        kwargs["lambda_anchor"] = 0.0
        super().__init__(*args, **kwargs)


# ──────────────────── Main pipeline ──────────────────────────────

def run_sequential_editing(
    editor: BioCsLLMEditor,
    dataset: KnowEditDataset,
    n_edits: int = 100,
    output_path: str = "results.json",
):
    """Run sequential knowledge editing and evaluate."""
    results = []
    agg = {"efficacy": [], "locality": []}

    start_time = __import__("time").time()
    for i in tqdm(range(min(n_edits, len(dataset))), desc="Editing"):
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
                "idx": i,
                "prompt": sample["prompt"],
                "target_new": sample["target_new"],
                **eval_result,
            })
            agg["efficacy"].append(eval_result["efficacy"])
            agg["locality"].append(eval_result["locality"])

            if (i + 1) % 10 == 0:
                eff = sum(agg["efficacy"]) / len(agg["efficacy"]) * 100
                loc = sum(agg["locality"]) / len(agg["locality"]) * 100
                logger.info(f"Edit {i+1}: Efficacy={eff:.1f}%, Locality={loc:.1f}%")

    summary = {
        **editor.metadata(),
        "n_edits": len(results),
        "requested_n_edits": n_edits,
        "elapsed_s": round(__import__("time").time() - start_time, 1),
        "efficacy": sum(agg["efficacy"]) / max(len(agg["efficacy"]), 1) * 100,
        "locality": sum(agg["locality"]) / max(len(agg["locality"]), 1) * 100,
        "results": results,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to {output_path}")
    logger.info(f"Final: Efficacy={summary['efficacy']:.1f}%, Locality={summary['locality']:.1f}%")
    return summary


def main():
    parser = argparse.ArgumentParser(description="SSR LLM Knowledge Editing")
    parser.add_argument("--model", type=str, default="gpt2",
                        help="HuggingFace model name or path")
    parser.add_argument("--data", type=str,
                        default="dataset/knowedit/benchmark/ZsRE/ZsRE-test-all.json")
    parser.add_argument("--method", type=str, default="biocs", choices=["biocs", "ft"],
                        help="biocs: SSR regularized FT; ft: vanilla fine-tuning")
    parser.add_argument("--n_edits", type=int, default=100)
    parser.add_argument("--lambda_biocs", type=float, default=0.001)
    parser.add_argument("--lambda_spectral", type=float, default=0.01)
    parser.add_argument("--lambda_anchor", type=float, default=0.001)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_steps", type=int, default=25)
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
        lr=args.lr,
        num_steps=args.num_steps,
    )

    dataset = KnowEditDataset(args.data, max_samples=args.n_edits)

    out_dir = Path(args.output) / f"{args.method}_{Path(args.model).name}"
    run_sequential_editing(editor, dataset, args.n_edits, str(out_dir / "results.json"))


if __name__ == "__main__":
    main()
