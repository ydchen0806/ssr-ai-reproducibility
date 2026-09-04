#!/usr/bin/env python3
"""LoRA rank/learning-rate SSR development and frozen confirmation.

Each matched rank/learning-rate/coefficient tuple is frozen before confirmation. This runner reports
acquisition, locality, retention and read-only geometry diagnostics at every
predeclared checkpoint. A zero coefficient remains an exact no-op.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(os.environ.get("SSR_REPO_DIR", Path(__file__).resolve().parents[1])).resolve()
EASYEDIT_ROOT = Path(os.environ.get("EASYEDIT_DIR", PROJECT_ROOT / "external" / "EasyEdit")).resolve()
if str(EASYEDIT_ROOT) not in sys.path:
    sys.path.insert(0, str(EASYEDIT_ROOT))
# The repository has a top-level ``datasets`` package.  Keep it after
# site-packages so EasyEdit and sentence-transformers import Hugging Face's
# ``datasets`` dependency rather than this unrelated local package.
for path_entry in list(sys.path):
    if path_entry in {"", "."}:
        sys.path.remove(path_entry)
    elif Path(path_entry).resolve() == PROJECT_ROOT:
        sys.path.remove(path_entry)
sys.path.append(str(PROJECT_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from llm_ke.biocs_editor import normalize_knowedit_record  # noqa: E402
from llm_ke.locality import evaluate_locality, normalize_text, normalize_text_list  # noqa: E402
from llm_ke.lowrank_ssr import LoRABasisSSR, summarize_lora_geometry  # noqa: E402


def _to_text(value: Any) -> str:
    return normalize_text(value)


def _to_text_list(value: Any) -> list[str]:
    return normalize_text_list(value)


def _target_match(prediction: str, target: Any) -> float:
    prediction_lower = normalize_text(prediction).lower()
    targets = normalize_text_list(target)
    return float(any(value.lower() in prediction_lower for value in targets))


def _evaluate_target_groups(groups: dict[str, Any], generate_fn) -> dict[str, Any]:
    return evaluate_locality(groups, generate_fn)


METHOD_CLASSES = {"LoRA": "LoRAHyperParams"}
METHOD_FOLDERS = {"LoRA": "LoRA"}
PROTOCOL = "lowrank_lora_ssr_v1"
AGE_BINS = (
    ("age_0", 0, 0),
    ("age_1_9", 1, 9),
    ("age_10_99", 10, 99),
    ("age_100_249", 100, 249),
    ("age_250_499", 250, 499),
    ("age_500_plus", 500, None),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_layer_list(text: str) -> set[int]:
    layers = {int(value) for value in re.split(r"[ ,]+", text.strip()) if value}
    if not layers:
        raise ValueError("--target-layers must contain at least one layer index")
    return layers


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_model_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def _model_input(model: torch.nn.Module, tokenizer, prompt: str) -> dict[str, torch.Tensor]:
    tokens = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
    return {name: value.to(get_model_device(model)) for name, value in tokens.items()}


@torch.no_grad()
def generate(model: torch.nn.Module, tokenizer, prompt: str) -> str:
    tokens = _model_input(model, tokenizer, prompt)
    generated = model.generate(
        **tokens,
        do_sample=False,
        max_new_tokens=32,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    start = tokens["input_ids"].shape[1]
    return tokenizer.decode(generated[0, start:], skip_special_tokens=True)


@torch.no_grad()
def next_token_distribution(model: torch.nn.Module, tokenizer, prompt: str) -> torch.Tensor:
    tokens = _model_input(model, tokenizer, prompt)
    logits = model(**tokens).logits[0, -1].float()
    return torch.softmax(logits, dim=-1)


def aggregate(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def disable_optional_torchao_dispatcher() -> None:
    """Force PEFT's standard Linear dispatcher for non-quantized base weights.

    Some compute images expose an old optional torchao distribution. PEFT
    raises on that package before it can observe that this experiment uses
    ordinary torch.nn.Linear weights. Disabling only the optional dispatcher
    preserves the standard PEFT LoRA path and does not mutate site packages.
    """
    try:
        import peft.import_utils as peft_import_utils
        import peft.tuners.lora.torchao as peft_lora_torchao
    except ImportError:
        return

    def unavailable() -> bool:
        return False

    peft_import_utils.is_torchao_available = unavailable
    peft_lora_torchao.is_torchao_available = unavailable
    if peft_import_utils.is_torchao_available() or peft_lora_torchao.is_torchao_available():
        raise RuntimeError("Failed to disable the optional PEFT TorchAO dispatcher")


def relation_score(groups: dict[str, Any], model: torch.nn.Module, tokenizer) -> tuple[float | None, int]:
    count = 0

    def generate_for_group(prompt: str) -> str:
        nonlocal count
        count += 1
        return generate(model, tokenizer, prompt)

    score = _evaluate_target_groups(groups, generate_for_group)
    return (score["score"] if score["n_items"] else None, int(score["n_items"]))


def evaluate_one(sample: dict[str, Any], model: torch.nn.Module, tokenizer) -> dict[str, Any]:
    completion = generate(model, tokenizer, sample["prompt"])
    rephrase_scores = [
        _target_match(generate(model, tokenizer, prompt), sample["target_new"])
        for prompt in sample.get("rephrase_prompts", [])
    ]
    portability, portability_n = relation_score(sample.get("portability", {}), model, tokenizer)
    locality, locality_n = relation_score(sample.get("locality", {}), model, tokenizer)
    return {
        "efficacy": _target_match(completion, sample["target_new"]),
        "rephrase": aggregate(rephrase_scores),
        "portability": portability,
        "portability_n": portability_n,
        # This is target consistency on supplied locality probes. It is not
        # relabelled as an untouched-knowledge preservation measure.
        "locality_target_consistency": locality,
        "locality_n": locality_n,
    }


def aggregate_evaluations(rows: list[dict[str, Any]]) -> dict[str, float | int | None]:
    result: dict[str, float | int | None] = {"n_edits": len(rows)}
    for metric in ("efficacy", "rephrase", "portability", "locality_target_consistency"):
        values = [float(row[metric]) for row in rows if row.get(metric) is not None]
        result[metric] = aggregate(values)
        result[f"{metric}_n"] = len(values)
    return result


def sample_positions(size: int, max_samples: int) -> list[int]:
    if max_samples <= 0 or max_samples >= size:
        return list(range(size))
    if max_samples == 1:
        return [size - 1]
    return sorted({round(index * (size - 1) / (max_samples - 1)) for index in range(max_samples)})


def age_label(age: int) -> str:
    for name, lower, upper in AGE_BINS:
        if age >= lower and (upper is None or age <= upper):
            return name
    raise AssertionError(f"Age did not enter a declared bin: {age}")


def evaluate_history(
    history: list[dict[str, Any]], model: torch.nn.Module, tokenizer, after_edits: int, max_samples: int,
) -> dict[str, Any]:
    evaluations = []
    for position in sample_positions(len(history), max_samples):
        sample = history[position]
        age = after_edits - position - 1
        evaluations.append(
            {
                "history_position": position,
                "edit_age": age,
                "age_bin": age_label(age),
                "retained_efficacy": _target_match(generate(model, tokenizer, sample["prompt"]), sample["target_new"]),
            }
        )
    bins: dict[str, list[float]] = {name: [] for name, _, _ in AGE_BINS}
    for row in evaluations:
        bins[row["age_bin"]].append(float(row["retained_efficacy"]))
    return {
        "after_edits": after_edits,
        "n_history": len(history),
        "n_evaluated": len(evaluations),
        "efficacy": 100.0 * aggregate([float(row["retained_efficacy"]) for row in evaluations]),
        "retention_by_age": {
            name: {
                "n": len(values),
                "efficacy": (100.0 * aggregate(values)) if values else None,
            }
            for name, values in bins.items()
        },
        "evaluations": evaluations,
    }


def capture_pre_edit_references(
    probes: list[dict[str, str]], model: torch.nn.Module, tokenizer, top_k: int = 32,
) -> list[dict[str, Any]]:
    records = []
    for probe in probes:
        probabilities = next_token_distribution(model, tokenizer, probe["prompt"])
        values, ids = torch.topk(probabilities, k=min(top_k, probabilities.numel()))
        records.append(
            {
                "probe_id": probe["probe_id"],
                "token_ids": ids.cpu().tolist(),
                "probabilities": values.cpu().tolist(),
                "residual_probability": float((1 - values.sum()).clamp_min(0).item()),
            }
        )
    return records


def evaluate_pre_edit_references(
    probes: list[dict[str, str]], references: list[dict[str, Any]], model: torch.nn.Module, tokenizer, after_edits: int,
) -> dict[str, Any]:
    if len(probes) != len(references):
        raise ValueError("Pre-edit probe/reference counts do not match")
    similarities = []
    for probe, reference in zip(probes, references):
        if probe["probe_id"] != reference["probe_id"]:
            raise ValueError("Pre-edit probe order changed")
        probs = next_token_distribution(model, tokenizer, probe["prompt"])
        token_ids = torch.tensor(reference["token_ids"], device=probs.device, dtype=torch.long)
        before = torch.tensor(
            [*reference["probabilities"], reference["residual_probability"]], device=probs.device, dtype=torch.float32,
        )
        observed = probs.index_select(0, token_ids)
        after = torch.cat((observed, (1 - observed.sum()).clamp_min(0).reshape(1)))
        middle = 0.5 * (before + after)
        epsilon = torch.finfo(torch.float32).tiny
        js = 0.5 * (
            (before * (before.clamp_min(epsilon).log() - middle.clamp_min(epsilon).log())).sum()
            + (after * (after.clamp_min(epsilon).log() - middle.clamp_min(epsilon).log())).sum()
        )
        similarities.append(float((1 - js / 0.6931471805599453).clamp(0, 1).item()))
    return {
        "after_edits": after_edits,
        "n_probes": len(similarities),
        "score": 100.0 * aggregate(similarities),
        "evaluator": "pre_edit_next_token_topk_js_similarity",
        "evaluator_version": "2.0",
    }


def _target_nll(model: torch.nn.Module, tokenizer, sample: dict[str, Any]) -> torch.Tensor:
    prompt = sample["prompt"]
    target = _to_text(sample["target_new"])
    prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)["input_ids"]
    tokens = tokenizer(
        f"{prompt} {target}", return_tensors="pt", truncation=True, max_length=256
    )
    tokens = {name: value.to(get_model_device(model)) for name, value in tokens.items()}
    labels = tokens["input_ids"].clone()
    prompt_length = min(prompt_ids.shape[1], labels.shape[1] - 1)
    labels[:, :prompt_length] = -100
    if not torch.any(labels != -100):
        raise ValueError("The current edit target was truncated completely")
    return model(**tokens, labels=labels).loss


def _reference_kl(
    model: torch.nn.Module,
    tokenizer,
    probe: dict[str, str],
    reference: dict[str, Any],
) -> torch.Tensor:
    tokens = _model_input(model, tokenizer, probe["prompt"])
    probabilities = torch.softmax(model(**tokens).logits[0, -1].float(), dim=-1)
    token_ids = torch.tensor(reference["token_ids"], device=probabilities.device, dtype=torch.long)
    observed = probabilities.index_select(0, token_ids)
    current = torch.cat((observed, (1 - observed.sum()).clamp_min(1e-8).reshape(1)))
    teacher = torch.tensor(
        [*reference["probabilities"], reference["residual_probability"]],
        device=probabilities.device,
        dtype=torch.float32,
    )
    teacher = teacher / teacher.sum().clamp_min(1e-8)
    current = current / current.sum().clamp_min(1e-8)
    return F.kl_div(current.clamp_min(1e-8).log(), teacher, reduction="sum")


def apply_output_preservation_anchor(
    model: torch.nn.Module,
    tokenizer,
    sample: dict[str, Any],
    probes: list[dict[str, str]],
    references: list[dict[str, Any]],
    *,
    edit_index: int,
    weight: float,
    steps: int,
    learning_rate: float,
    batch_size: int,
) -> list[dict[str, float]]:
    """Repair output drift on frozen controls while retaining the current edit."""
    if weight == 0 or steps == 0:
        return []
    parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if ".lora_" in name and parameter.requires_grad
    ]
    if not parameters:
        raise RuntimeError("Output preservation requested but no trainable LoRA parameters were found")
    if len(probes) != len(references):
        raise ValueError("Output-preservation probes and references are not aligned")

    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    diagnostics = []
    model.train()
    for step in range(steps):
        start = ((edit_index - 1) * batch_size + step * batch_size) % len(probes)
        positions = [(start + offset) % len(probes) for offset in range(batch_size)]
        optimizer.zero_grad(set_to_none=True)
        task_loss = _target_nll(model, tokenizer, sample)
        anchor_loss = torch.stack(
            [_reference_kl(model, tokenizer, probes[pos], references[pos]) for pos in positions]
        ).mean()
        loss = task_loss + weight * anchor_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        diagnostics.append(
            {
                "task_loss": float(task_loss.detach().item()),
                "anchor_kl": float(anchor_loss.detach().item()),
                "total_loss": float(loss.detach().item()),
            }
        )
    model.eval()
    return diagnostics


def load_stream(dataset_path: Path, manifest_path: Path, n_edits: int) -> tuple[list[dict[str, Any]], list[dict[str, str]], dict[str, Any]]:
    rows = json.loads(dataset_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset_sha256") != sha256(dataset_path):
        raise ValueError("Stream manifest belongs to a different dataset file")
    edit_indices = manifest.get("edit_indices")
    probe_indices = manifest.get("pre_edit_indices")
    if not isinstance(edit_indices, list) or len(edit_indices) < n_edits:
        raise ValueError("Stream manifest is shorter than --n-edits")
    edit_indices = edit_indices[:n_edits]
    if not isinstance(probe_indices, list) or not probe_indices:
        raise ValueError("Stream manifest must contain non-empty pre-edit controls")
    if len(set(edit_indices)) != len(edit_indices) or len(set(probe_indices)) != len(probe_indices):
        raise ValueError("Stream manifest contains repeated indices")
    if set(edit_indices).intersection(probe_indices):
        raise ValueError("Edit and pre-edit control rows overlap")
    if any(not isinstance(index, int) or index < 0 or index >= len(rows) for index in [*edit_indices, *probe_indices]):
        raise ValueError("Stream manifest contains invalid dataset indices")
    edits = []
    for index in edit_indices:
        raw = rows[index]
        sample = normalize_knowedit_record(raw)
        rephrases = raw.get("rephrase_prompt", raw.get("rephrase", []))
        sample["rephrase_prompts"] = normalize_text_list(rephrases)
        edits.append(sample)
    probes = [
        {"probe_id": f"zsre-row-{index}", "prompt": normalize_knowedit_record(rows[index])["prompt"]}
        for index in probe_indices
    ]
    return edits, probes, manifest


def build_editor(
    method: str,
    hparams_path: Path,
    model_path: str,
    cache_dir: Path,
    stats_cache_dir: Path,
    lora_rank: int,
    lora_lr: float,
    lora_steps: int,
):
    # Reuse the compatibility shim used by the existing EasyEdit launcher;
    # its imports must not allow this repository's datasets package to shadow
    # HuggingFace datasets.
    from run_llm_ke_easyedit import isolated_easyedit_imports, patch_transformers_for_easyedit

    with isolated_easyedit_imports():
        patch_transformers_for_easyedit()
        disable_optional_torchao_dispatcher()
        from easyeditor import BaseEditor
        import easyeditor

        cls_name = METHOD_CLASSES[method]
        hparams_cls = getattr(easyeditor, cls_name)
        hparams = hparams_cls.from_hparams(str(hparams_path))
        hparams.model_name = model_path
        hparams.device = 0
        hparams.rank = lora_rank
        hparams.lora_alpha = 4 * lora_rank
        hparams.lr = lora_lr
        hparams.num_steps = lora_steps
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.chdir(cache_dir)
        editor = BaseEditor.from_hparams(hparams)
    return editor, hparams


def edit_one(editor, sample: dict[str, Any]) -> torch.nn.Module:
    _, edited_model, _ = editor.edit(
        prompts=[sample["prompt"]],
        target_new=[_to_text(sample["target_new"])],
        ground_truth=[_to_text(sample["ground_truth"])],
        subject=[_to_text(sample.get("subject", ""))],
        sequential_edit=True,
        verbose=False,
        test_generation=False,
    )
    if not hasattr(edited_model, "named_parameters"):
        raise TypeError("Selected EasyEdit method did not return a causal-LM parameter module")
    editor.model = edited_model
    return edited_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=sorted(METHOD_CLASSES), required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--hparams-root", type=Path, required=True)
    parser.add_argument("--hparams-stem", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--stream-manifest", type=Path, required=True)
    parser.add_argument("--n-edits", type=int, required=True)
    parser.add_argument("--lora-rank", type=int, required=True)
    parser.add_argument("--lora-lr", type=float, required=True)
    parser.add_argument("--lora-steps", type=int, default=60)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--ssr-lambda", type=float, default=0.0)
    parser.add_argument("--anchor-weight", type=float, default=0.0)
    parser.add_argument("--anchor-steps", type=int, default=0)
    parser.add_argument("--anchor-lr", type=float, default=1e-4)
    parser.add_argument("--anchor-batch-size", type=int, default=4)
    parser.add_argument("--history-checkpoints", type=int, nargs="+", required=True)
    parser.add_argument("--history-max-samples", type=int, default=128)
    parser.add_argument("--pre-edit-max-samples", type=int, default=32)
    parser.add_argument(
        "--method-cache-dir", type=Path, required=True,
        help="Shared, serially written native-editor moment/null-space cache for one method/model pair.",
    )
    parser.add_argument(
        "--stats-cache-dir", type=Path,
        help="Optional model-level shared second-moment cache used across compatible native editors.",
    )
    parser.add_argument(
        "--stats-lock", type=Path,
        help="Optional shared lock serializing the first native edit while moment/projection caches are created.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_edits < 1 or args.history_max_samples < 1 or args.pre_edit_max_samples < 1:
        raise ValueError("n-edits and sample limits must be positive")
    if args.lora_rank < 1 or args.lora_rank > 256:
        raise ValueError("lora-rank must be between 1 and 256")
    if args.lora_lr <= 0 or args.lora_steps < 1:
        raise ValueError("lora-lr and lora-steps must be positive")
    if args.ssr_lambda < 0:
        raise ValueError("ssr-lambda must be non-negative")
    if args.anchor_weight < 0 or args.anchor_steps < 0 or args.anchor_lr <= 0 or args.anchor_batch_size < 1:
        raise ValueError("Invalid output-preservation anchor configuration")
    if (args.anchor_weight == 0) != (args.anchor_steps == 0):
        raise ValueError("Set both --anchor-weight and --anchor-steps, or leave both at zero")
    checkpoints = sorted(set(args.history_checkpoints))
    if not checkpoints or checkpoints[-1] != args.n_edits or any(value <= 0 or value > args.n_edits for value in checkpoints):
        raise ValueError("history checkpoints must be positive, unique, and include n-edits")
    hparams_path = args.hparams_root / METHOD_FOLDERS[args.method] / f"{args.hparams_stem}.yaml"
    for path in (args.dataset, args.stream_manifest, hparams_path, Path(args.model_path) / "config.json"):
        if not path.is_file():
            raise FileNotFoundError(path)
    edits, probes, manifest = load_stream(args.dataset, args.stream_manifest, args.n_edits)
    probe_positions = sample_positions(len(probes), args.pre_edit_max_samples)
    probes = [probes[position] for position in probe_positions]
    disable_optional_torchao_dispatcher()
    config = {
        "protocol": PROTOCOL,
        "method": args.method,
        "model_label": args.model_label,
        "model_path": str(Path(args.model_path).resolve()),
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": sha256(args.dataset),
        "stream_manifest": str(args.stream_manifest.resolve()),
        "stream_manifest_sha256": sha256(args.stream_manifest),
        "stream_seed": manifest.get("random_order_seed"),
        "stream_prefix_policy": "first_n_predeclared_edit_indices",
        "manifest_declared_edits": len(manifest["edit_indices"]),
        "n_edits": args.n_edits,
        "hparams_path": str(hparams_path.resolve()),
        "hparams_sha256": sha256(hparams_path),
        "method_cache_dir": str(args.method_cache_dir.resolve()),
        "stats_cache_dir": str((args.stats_cache_dir or (args.method_cache_dir / "stats")).resolve()),
        "lora": {
            "rank": args.lora_rank,
            "alpha": 4 * args.lora_rank,
            "alpha_over_rank": 4.0,
            "learning_rate": args.lora_lr,
            "num_steps": args.lora_steps,
            "target_modules": ["q_proj", "v_proj"],
            "capacity_variable": "rank_and_learning_rate_with_fixed_alpha_over_rank",
            "hparams_scope": (
                "released EasyEdit Qwen2.5-7B except declared rank/alpha and learning-rate scaling"
            ),
        },
        "peft_dispatch": "standard_linear_with_optional_torchao_disabled",
        "ssr": {
            "enabled": args.ssr_lambda > 0,
            "lambda": args.ssr_lambda,
            "application": "post_lora_update_center_surround_proximal_step",
            "plastic_object": "LoRA-B output rows",
            "topology": "cyclic row adjacency within each LoRA-B matrix",
            "zero_is_exact_noop": True,
        },
        "output_preservation_anchor": {
            "enabled": args.anchor_weight > 0,
            "teacher": "frozen_pre_edit_next_token_topk_distribution",
            "loss": "current_edit_nll_plus_teacher_to_student_kl",
            "weight": args.anchor_weight,
            "steps_per_edit": args.anchor_steps,
            "learning_rate": args.anchor_lr,
            "probe_batch_size": args.anchor_batch_size,
            "application_order": "native_lora_then_anchor_then_ssr",
        },
        "mechanism_diagnostics": {
            "scope": "read_only_on_lora_B_at_predeclared_checkpoints",
            "effective_rank": "entropy_effective_rank_normalized_by_matrix_rank_ceiling",
            "geometry_overlap": "cosine_alignment_with_fixed_center_surround_row_kernel",
        },
        "history_checkpoints": checkpoints,
        "history_max_samples": args.history_max_samples,
        "pre_edit_control_count": len(probes),
        "pre_edit_control_sampling": "fixed_even_positions_from_predeclared_manifest",
    }
    if args.validate_only:
        config["status"] = "validated"
        atomic_json(args.output, config)
        return

    if args.output.exists():
        raise FileExistsError(f"Output must be fresh: {args.output}")
    set_seed(args.seed)
    started = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    # Each arm gets an isolated PEFT working directory so adapters and logs
    # cannot leak across seeds or coefficients.
    cache_dir = args.method_cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(cache_dir)
    stats_cache_dir = (args.stats_cache_dir or (cache_dir / "stats")).resolve()
    stats_cache_dir.mkdir(parents=True, exist_ok=True)
    editor, hparams = build_editor(
        args.method, hparams_path, args.model_path, cache_dir, stats_cache_dir,
        args.lora_rank, args.lora_lr, args.lora_steps,
    )
    model = editor.model
    tokenizer = editor.tok
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    projector = LoRABasisSSR(args.ssr_lambda)
    pre_edit_references = capture_pre_edit_references(probes, model, tokenizer)
    immediate = []
    history: list[dict[str, Any]] = []
    checkpoint_rows = []
    native_update_norms = []
    anchor_diagnostics = []
    for index, sample in enumerate(edits, start=1):
        if index == 1 and args.stats_lock is not None:
            args.stats_lock.parent.mkdir(parents=True, exist_ok=True)
            with args.stats_lock.open("a+", encoding="utf-8") as lock_handle:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
                model = edit_one(editor, sample)
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        else:
            model = edit_one(editor, sample)
        anchor_diagnostics.extend(
            apply_output_preservation_anchor(
                model,
                tokenizer,
                sample,
                probes,
                pre_edit_references,
                edit_index=index,
                weight=args.anchor_weight,
                steps=args.anchor_steps,
                learning_rate=args.anchor_lr,
                batch_size=args.anchor_batch_size,
            )
        )
        projector.apply(model)
        native_update_norm = getattr(model, "_ssr_native_last_update_norm", None)
        if native_update_norm is not None:
            native_update_norms.append(float(native_update_norm))
        model.eval()
        row = evaluate_one(sample, model, tokenizer)
        immediate.append(row)
        history.append(sample)
        if index in checkpoints:
            checkpoint_rows.append(
                {
                    "after_edits": index,
                    "immediate": aggregate_evaluations(immediate),
                    "history": evaluate_history(history, model, tokenizer, index, args.history_max_samples),
                    "pre_edit_output_consistency": evaluate_pre_edit_references(
                        probes, pre_edit_references, model, tokenizer, index
                    ),
                    "lora_geometry": summarize_lora_geometry(model),
                }
            )
    final_checkpoint = checkpoint_rows[-1]
    elapsed_s = round(time.time() - started, 1)
    summary = {
        **config,
        "status": "complete",
        "elapsed_s": elapsed_s,
        "runtime_per_edit_s": elapsed_s / args.n_edits,
        "peak_cuda_memory_mb": (
            float(torch.cuda.max_memory_allocated() / (1024 * 1024))
            if torch.cuda.is_available()
            else None
        ),
        "easyedit_alg_name": getattr(hparams, "alg_name", args.method),
        "immediate": aggregate_evaluations(immediate),
        "checkpoints": checkpoint_rows,
        "final_history_efficacy": final_checkpoint["history"]["efficacy"],
        "final_pre_edit_output_consistency": final_checkpoint["pre_edit_output_consistency"]["score"],
        "native_update_norm": None if not native_update_norms else {
            "minimum": min(native_update_norms),
            "mean": sum(native_update_norms) / len(native_update_norms),
            "maximum": max(native_update_norms),
            "nonzero_fraction": sum(value > 1e-10 for value in native_update_norms) / len(native_update_norms),
            "count": len(native_update_norms),
        },
        "output_preservation_anchor_diagnostics": None if not anchor_diagnostics else {
            "n_steps": len(anchor_diagnostics),
            "mean_task_loss": aggregate([row["task_loss"] for row in anchor_diagnostics]),
            "mean_anchor_kl": aggregate([row["anchor_kl"] for row in anchor_diagnostics]),
            "mean_total_loss": aggregate([row["total_loss"] for row in anchor_diagnostics]),
        },
    }
    atomic_json(args.output, summary)


if __name__ == "__main__":
    main()
