#!/usr/bin/env python3
"""Matched edit-count scaling run for a native EasyEdit method and SSR.

This runner keeps the edit method's own hyperparameters and applies SSR only
after that method has produced an update.  The compositional arm is therefore
reported as ``METHOD + frozen-ring SSR projection`` rather than as a modified
implementation of METHOD.  A zero coefficient is an exact no-op.
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


PROJECT_ROOT = Path(os.environ.get("SSR_REPO_DIR", "/unify/ydchen/unidit/bioreg_meeting_20260803_final")).resolve()
EASYEDIT_ROOT = Path(os.environ.get("EASYEDIT_DIR", "/unify/ydchen/unidit/ssr_teacher_assets_20260818/sources/EasyEdit")).resolve()
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

from llm_ke.biocs_editor_topology_gradalign_v2 import (  # noqa: E402
    _evaluate_target_groups,
    _target_match,
    _to_text,
    _to_text_list,
    normalize_knowedit_record,
)


METHOD_CLASSES = {
    "AlphaEdit": "AlphaEditHyperParams",
    "MEMIT": "MEMITHyperParams",
    "SPHERE": "SPHEREHyperParams",
}
METHOD_FOLDERS = {
    "AlphaEdit": "AlphaEdit",
    "MEMIT": "MEMIT",
    "SPHERE": "SPHERE",
}
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


class FrozenRingSSR:
    """A frozen row-topology center-surround proximal step for native editors."""

    def __init__(
        self,
        model: torch.nn.Module,
        target_layers: set[int],
        coefficient: float,
        *,
        near_weight: float = 1.0,
        surround_weight: float = 0.2,
    ) -> None:
        if coefficient < 0:
            raise ValueError("SSR coefficient must be non-negative")
        self.coefficient = coefficient
        self.near_weight = near_weight
        self.surround_weight = surround_weight
        self.references: dict[str, torch.Tensor] = {}
        layer_re = re.compile(r"(?:layers|h)\.(\d+)\.")
        for name, parameter in model.named_parameters():
            match = layer_re.search(name)
            if match is None or int(match.group(1)) not in target_layers:
                continue
            if not re.search(r"(?:down_proj|c_proj)\.weight$", name) or parameter.ndim != 2:
                continue
            self.references[name] = parameter.detach().clone()
        if not self.references:
            raise ValueError("No editable down_proj/c_proj matrices matched the declared target layers")

    @torch.no_grad()
    def apply(self, model: torch.nn.Module) -> None:
        if self.coefficient == 0:
            return
        current = dict(model.named_parameters())
        for name, reference in self.references.items():
            if name not in current:
                raise RuntimeError(f"Native editor changed the declared SSR parameter set: {name}")
            parameter = current[name]
            if parameter.shape != reference.shape:
                raise RuntimeError(f"Native editor changed the shape of {name}")
            delta = parameter.detach().float() - reference.float()
            near = 0.5 * (torch.roll(delta, 1, 0) + torch.roll(delta, -1, 0))
            surround = 0.5 * (torch.roll(delta, 2, 0) + torch.roll(delta, -2, 0))
            center_surround = self.near_weight * (delta - near) - self.surround_weight * (delta - surround)
            parameter.copy_((reference.float() + delta - self.coefficient * center_surround).to(parameter.dtype))


def load_stream(dataset_path: Path, manifest_path: Path, n_edits: int) -> tuple[list[dict[str, Any]], list[dict[str, str]], dict[str, Any]]:
    rows = json.loads(dataset_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset_sha256") != sha256(dataset_path):
        raise ValueError("Stream manifest belongs to a different dataset file")
    edit_indices = manifest.get("edit_indices")
    probe_indices = manifest.get("pre_edit_indices")
    if not isinstance(edit_indices, list) or len(edit_indices) != n_edits:
        raise ValueError("Stream manifest edit count does not equal --n-edits")
    if not isinstance(probe_indices, list) or not probe_indices:
        raise ValueError("Stream manifest must contain non-empty pre-edit controls")
    if len(set(edit_indices)) != len(edit_indices) or len(set(probe_indices)) != len(probe_indices):
        raise ValueError("Stream manifest contains repeated indices")
    if set(edit_indices).intersection(probe_indices):
        raise ValueError("Edit and pre-edit control rows overlap")
    if any(not isinstance(index, int) or index < 0 or index >= len(rows) for index in [*edit_indices, *probe_indices]):
        raise ValueError("Stream manifest contains invalid dataset indices")
    edits = [normalize_knowedit_record(rows[index]) for index in edit_indices]
    probes = [
        {"probe_id": f"zsre-row-{index}", "prompt": normalize_knowedit_record(rows[index])["prompt"]}
        for index in probe_indices
    ]
    return edits, probes, manifest


def build_editor(
    method: str,
    hparams_path: Path,
    model_path: str,
    target_layers: set[int],
    cache_dir: Path,
    stats_cache_dir: Path,
    projection_basis: Path | None,
    native_projection_rank: int,
):
    # Reuse the compatibility shim used by the existing EasyEdit launcher;
    # its imports must not allow this repository's datasets package to shadow
    # HuggingFace datasets.
    from run_llm_ke_easyedit import isolated_easyedit_imports, patch_transformers_for_easyedit

    with isolated_easyedit_imports():
        patch_transformers_for_easyedit()
        from easyeditor import BaseEditor
        import easyeditor

        cls_name = METHOD_CLASSES[method]
        hparams_cls = getattr(easyeditor, cls_name)
        hparams = hparams_cls.from_hparams(str(hparams_path))
        hparams.model_name = model_path
        hparams.device = 0
        native_layers = {int(layer) for layer in getattr(hparams, "layers", [])}
        if native_layers != target_layers:
            raise ValueError(
                "SSR target layers must exactly equal the native editor layers; "
                f"native={sorted(native_layers)}, requested={sorted(target_layers)}"
            )
        cache_dir.mkdir(parents=True, exist_ok=True)
        hparams.stats_dir = str(stats_cache_dir.resolve())
        Path(hparams.stats_dir).mkdir(parents=True, exist_ok=True)
        if hasattr(hparams, "P_loc"):
            if projection_basis is None:
                raise ValueError(f"{method} requires --projection-basis")
            hparams.P_loc = str(projection_basis.resolve())
            hparams.native_projection_rank = native_projection_rank
        # The imported legacy runner changes cwd to PROJECT_ROOT. AlphaEdit and
        # SPHERE save their projection by a relative filename, so restore the
        # declared method cache immediately before construction.
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
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--target-layers", required=True)
    parser.add_argument("--ssr-lambda", type=float, default=0.0)
    parser.add_argument("--projection-basis", type=Path)
    parser.add_argument("--native-projection-rank", type=int, default=0)
    parser.add_argument("--history-checkpoints", type=int, nargs="+", required=True)
    parser.add_argument("--history-max-samples", type=int, default=128)
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
    if args.n_edits < 1 or args.history_max_samples < 1:
        raise ValueError("n-edits and history-max-samples must be positive")
    if args.ssr_lambda < 0:
        raise ValueError("ssr-lambda must be non-negative")
    if args.method in {"AlphaEdit", "SPHERE"}:
        if args.projection_basis is None:
            raise FileNotFoundError("AlphaEdit/SPHERE require a projection basis")
        if not args.validate_only and not args.projection_basis.is_file():
            raise FileNotFoundError(args.projection_basis)
        if args.native_projection_rank < 1:
            raise ValueError("AlphaEdit/SPHERE require a positive native projection rank")
    elif args.native_projection_rank != 0:
        raise ValueError("MEMIT does not use native projection rank")
    checkpoints = sorted(set(args.history_checkpoints))
    if not checkpoints or checkpoints[-1] != args.n_edits or any(value <= 0 or value > args.n_edits for value in checkpoints):
        raise ValueError("history checkpoints must be positive, unique, and include n-edits")
    hparams_path = args.hparams_root / METHOD_FOLDERS[args.method] / f"{args.hparams_stem}.yaml"
    for path in (args.dataset, args.stream_manifest, hparams_path, Path(args.model_path) / "config.json"):
        if not path.is_file():
            raise FileNotFoundError(path)
    edits, probes, manifest = load_stream(args.dataset, args.stream_manifest, args.n_edits)
    layers = parse_layer_list(args.target_layers)
    config = {
        "protocol": "ke_native_editor_scaling_v29",
        "method": args.method,
        "model_label": args.model_label,
        "model_path": str(Path(args.model_path).resolve()),
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": sha256(args.dataset),
        "stream_manifest": str(args.stream_manifest.resolve()),
        "stream_manifest_sha256": sha256(args.stream_manifest),
        "stream_seed": manifest.get("random_order_seed"),
        "n_edits": args.n_edits,
        "target_layers": sorted(layers),
        "hparams_path": str(hparams_path.resolve()),
        "hparams_sha256": sha256(hparams_path),
        "method_cache_dir": str(args.method_cache_dir.resolve()),
        "stats_cache_dir": str((args.stats_cache_dir or (args.method_cache_dir / "stats")).resolve()),
        "native_projection": {
            "rank": args.native_projection_rank,
            "basis_path": None if args.projection_basis is None else str(args.projection_basis.resolve()),
            "selection_scope": "native rank frozen from v28 development",
        },
        "ssr": {
            "enabled": args.ssr_lambda > 0,
            "lambda": args.ssr_lambda,
            "application": "post_native_update_frozen_ring_center_surround_projection",
            "plastic_object": "declared MLP down_proj/c_proj row deltas",
            "topology": "frozen cyclic row adjacency",
            "zero_is_exact_noop": True,
        },
        "history_checkpoints": checkpoints,
        "history_max_samples": args.history_max_samples,
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
    # AlphaEdit and SPHERE use a relative save path internally when creating
    # their projection. A method/model-specific working directory makes that
    # behavior deterministic and lets the paired SSR arm reuse the same cache.
    cache_dir = args.method_cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(cache_dir)
    stats_cache_dir = (args.stats_cache_dir or (cache_dir / "stats")).resolve()
    stats_cache_dir.mkdir(parents=True, exist_ok=True)
    editor, hparams = build_editor(
        args.method, hparams_path, args.model_path, layers, cache_dir, stats_cache_dir,
        args.projection_basis, args.native_projection_rank,
    )
    model = editor.model
    tokenizer = editor.tok
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    projector = FrozenRingSSR(model, layers, args.ssr_lambda)
    pre_edit_references = capture_pre_edit_references(probes, model, tokenizer)
    immediate = []
    history: list[dict[str, Any]] = []
    checkpoint_rows = []
    native_update_norms = []
    for index, sample in enumerate(edits, start=1):
        if index == 1 and args.stats_lock is not None:
            args.stats_lock.parent.mkdir(parents=True, exist_ok=True)
            with args.stats_lock.open("a+", encoding="utf-8") as lock_handle:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
                model = edit_one(editor, sample)
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        else:
            model = edit_one(editor, sample)
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
    }
    atomic_json(args.output, summary)


if __name__ == "__main__":
    main()
