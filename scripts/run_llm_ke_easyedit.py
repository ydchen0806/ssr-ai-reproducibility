#!/usr/bin/env python3
"""LLM Knowledge Editing experiments using EasyEdit framework.

Compares KE-specific methods (ROME, MEMIT, AlphaEdit) with our SSR
fine-tuning approach on KnowEdit benchmark (ZsRE, WikiCounterfact, WikiRecent).

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/run_llm_ke_easyedit.py \
        --method ROME --dataset zsre --n_edits 100

    Or use the combined launch script:
    bash scripts/run_cluster_8gpu_all.sh
"""
import sys, os, json, argparse, logging, time, random, hashlib, subprocess
from contextlib import contextmanager
from pathlib import Path

PROJECT_DIR = Path(os.environ.get("SSR_REPO_DIR", Path(__file__).resolve().parents[1])).resolve()
EASYEDIT_DIR = Path(os.environ.get("EASYEDIT_DIR", PROJECT_DIR / "external" / "EasyEdit")).resolve()

if EASYEDIT_DIR.exists():
    sys.path.insert(0, str(EASYEDIT_DIR))
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(1 if EASYEDIT_DIR.exists() else 0, str(PROJECT_DIR))
os.chdir(PROJECT_DIR)

from llm_ke.locality import (
    LOCALITY_PROTOCOL_HASH,
    first_locality_item,
    normalize_text,
    normalize_text_list,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

DATASET_MAP = {
    "zsre": str(PROJECT_DIR / "dataset/knowedit/benchmark/ZsRE/ZsRE-test-all.json"),
    "cf": str(PROJECT_DIR / "dataset/knowedit/benchmark/wiki_counterfact/test_cf.json"),
    "recent": str(PROJECT_DIR / "dataset/knowedit/benchmark/wiki_recent/recent_test.json"),
    "wikibio": str(PROJECT_DIR / "dataset/knowedit/benchmark/WikiBio/wikibio-test-all.json"),
}

MODEL_NAME = os.environ.get("KE_MODEL_NAME_OR_PATH", "gpt2-xl")
HPARAMS_MODEL = os.environ.get("KE_HPARAMS_MODEL", "gpt2-xl")
HPARAMS_DIR = os.environ.get("KE_HPARAMS_DIR", str(PROJECT_DIR / "llm_ke/easyedit_hparams"))


def set_seed_from_env() -> None:
    """Make SSR row sampling and optimizer order reproducible when requested."""
    seed_text = os.environ.get("KE_SEED")
    if not seed_text:
        return
    seed = int(seed_text)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    logger.info(f"Set KE_SEED={seed}")


@contextmanager
def isolated_easyedit_imports():
    """Avoid this repo's datasets/ package shadowing HuggingFace datasets."""
    original_path = list(sys.path)
    filtered_path = []
    for entry in sys.path:
        if entry == "":
            continue
        try:
            resolved = Path(entry).resolve()
        except Exception:
            filtered_path.append(entry)
            continue
        if resolved == PROJECT_DIR:
            continue
        filtered_path.append(entry)

    local_datasets = sys.modules.get("datasets")
    local_datasets_path = getattr(local_datasets, "__file__", "") if local_datasets else ""
    removed_local_datasets = None
    if local_datasets_path and str(PROJECT_DIR) in local_datasets_path:
        removed_local_datasets = sys.modules.pop("datasets", None)

    sys.path[:] = filtered_path
    try:
        yield
    finally:
        sys.path[:] = original_path
        if removed_local_datasets is not None and "datasets" not in sys.modules:
            sys.modules["datasets"] = removed_local_datasets


def safe_tag(text: str) -> str:
    return Path(text).name.replace("/", "_").replace(" ", "_")


def current_git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_DIR,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown-uncommitted-environment"


def _parse_bool(text: str) -> bool:
    return text.strip().lower() in {"1", "true", "yes", "y", "on"}


def configure_runtime_from_env() -> dict[str, object]:
    """Apply and report the locked, best-effort PyTorch runtime policy.

    The formal GPT-2 XL cohort fixes seeds and disables cuDNN benchmarking,
    but does not claim bitwise determinism across CUDA, driver, or hardware
    versions.  Keeping this policy explicit prevents a future run from
    silently inheriting a machine-specific default.
    """
    policy = os.environ.get("KE_DETERMINISM_POLICY", "seeded_best_effort")
    requested = {
        "policy": policy,
        "cudnn_deterministic": _parse_bool(
            os.environ.get("KE_CUDNN_DETERMINISTIC", "0")
        ),
        "cudnn_benchmark": _parse_bool(os.environ.get("KE_CUDNN_BENCHMARK", "0")),
        "use_deterministic_algorithms": _parse_bool(
            os.environ.get("KE_USE_DETERMINISTIC_ALGORITHMS", "0")
        ),
        "deterministic_algorithms_warn_only": _parse_bool(
            os.environ.get("KE_DETERMINISTIC_ALGORITHMS_WARN_ONLY", "0")
        ),
    }
    observed = dict(requested)
    try:
        import torch

        torch.backends.cudnn.deterministic = requested["cudnn_deterministic"]
        torch.backends.cudnn.benchmark = requested["cudnn_benchmark"]
        torch.use_deterministic_algorithms(
            requested["use_deterministic_algorithms"],
            warn_only=requested["deterministic_algorithms_warn_only"],
        )
        observed.update(
            {
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "use_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            }
        )
    except Exception as error:  # Runtime metadata must not hide a failed import.
        observed["runtime_configuration_error"] = f"{type(error).__name__}: {error}"
    return {"requested": requested, "observed": observed}


def _parse_int_list(text: str) -> list[int]:
    return [int(part.strip()) for part in text.replace(",", " ").split() if part.strip()]


def resolve_history_run_config(args: argparse.Namespace) -> dict | None:
    """Resolve historical-retention evaluation for the matched custom runners."""
    enabled = bool(getattr(args, "evaluate_history", False))
    checkpoints = list(getattr(args, "history_checkpoints", None) or [])
    max_samples = int(getattr(args, "history_max_samples", 0))
    requested = enabled or bool(checkpoints) or max_samples != 0

    if args.method not in {"biocs", "ft"}:
        if requested:
            raise ValueError(
                "historical-retention CLI options are only supported by the "
                "matched custom biocs/ft evaluators"
            )
        return None
    if checkpoints and not enabled:
        raise ValueError("--history_checkpoints requires --evaluate_history")
    if max_samples < 0:
        raise ValueError("--history_max_samples must be non-negative; use 0 for all edits")
    if max_samples and not enabled:
        raise ValueError("--history_max_samples requires --evaluate_history")
    if any(checkpoint <= 0 for checkpoint in checkpoints):
        raise ValueError("--history_checkpoints must contain positive edit counts")
    if any(checkpoint > args.n_edits for checkpoint in checkpoints):
        raise ValueError("--history_checkpoints cannot exceed --n_edits")
    return {
        "evaluate_history": enabled,
        "history_checkpoints": sorted(set(checkpoints)),
        "history_max_samples": max_samples,
    }


def _compatible_cli_value(args: argparse.Namespace | None, primary: str, legacy: str | None = None):
    """Read a CLI value while rejecting conflicting new and legacy spellings."""
    if args is None:
        return None
    primary_value = getattr(args, primary, None)
    legacy_value = getattr(args, legacy, None) if legacy else None
    if primary_value is not None and legacy_value is not None and primary_value != legacy_value:
        raise ValueError(
            f"Conflicting CLI values: --{primary}={primary_value} and "
            f"--{legacy}={legacy_value}"
        )
    return primary_value if primary_value is not None else legacy_value


def _env_first(environ: dict[str, str], *names: str, default=None):
    for name in names:
        if name in environ and environ[name] != "":
            return environ[name]
    return default


def resolve_biocs_run_config(
    args: argparse.Namespace | None = None,
    environ: dict[str, str] | None = None,
) -> dict:
    """Resolve the SSR editing objective with CLI-over-environment precedence."""
    sys.path.insert(0, str(PROJECT_DIR))
    from llm_ke.biocs_editor import resolve_recipe

    env = os.environ if environ is None else environ

    cli_recipe = getattr(args, "recipe", None) if args is not None else None
    recipe = cli_recipe if cli_recipe is not None else _env_first(env, "BIOCS_RECIPE")

    cli_lambda_ssr = _compatible_cli_value(args, "lambda_ssr", "biocs_lambda")
    cli_lambda_anchor = _compatible_cli_value(args, "lambda_anchor", "biocs_anchor")
    cli_lambda_spectral = getattr(args, "lambda_spectral", None) if args is not None else None
    lambda_ssr = float(
        cli_lambda_ssr
        if cli_lambda_ssr is not None
        else _env_first(env, "BIOCS_LAMBDA_SSR", "BIOCS_LAMBDA", default="0.001")
    )
    lambda_anchor = float(
        cli_lambda_anchor
        if cli_lambda_anchor is not None
        else _env_first(env, "BIOCS_LAMBDA_ANCHOR", default="0.001")
    )
    lambda_spectral = float(
        cli_lambda_spectral
        if cli_lambda_spectral is not None
        else _env_first(env, "BIOCS_LAMBDA_SPECTRAL", default="0.01")
    )

    cli_mapping = getattr(args, "distance_mapping", None) if args is not None else None
    distance_mapping = (
        cli_mapping
        if cli_mapping is not None
        else _env_first(
            env,
            "BIOCS_DISTANCE_MAPPING",
            "BIOCS_DISTANCE_METRIC",
            default="cosine",
        )
    )
    if distance_mapping not in {"cosine", "projective"}:
        raise ValueError(
            f"Unknown distance_mapping={distance_mapping!r}; choose cosine or projective"
        )

    cli_seed = getattr(args, "seed", None) if args is not None else None
    seed = int(cli_seed if cli_seed is not None else _env_first(env, "KE_SEED", default="0"))
    sampler_seed = int(
        cli_seed
        if cli_seed is not None
        else _env_first(env, "BIOCS_SAMPLER_SEED", "KE_SEED", default="0")
    )

    resolved = resolve_recipe(
        recipe,
        lambda_ssr=lambda_ssr,
        lambda_anchor=lambda_anchor,
        lambda_spectral=lambda_spectral,
    )
    return {
        **resolved,
        "distance_mapping": distance_mapping,
        "seed": seed,
        "sampler_seed": sampler_seed,
    }


def apply_biocs_run_config(config: dict) -> None:
    """Expose the resolved config to legacy code and subprocess manifests."""
    canonical = {
        "KE_SEED": config["seed"],
        "BIOCS_RECIPE": config["recipe"],
        "BIOCS_LAMBDA_SSR": config["lambda_ssr"],
        "BIOCS_LAMBDA": config["lambda_ssr"],
        "BIOCS_LAMBDA_ANCHOR": config["lambda_anchor"],
        "BIOCS_LAMBDA_SPECTRAL": config["lambda_spectral"],
        "BIOCS_DISTANCE_MAPPING": config["distance_mapping"],
        "BIOCS_DISTANCE_METRIC": config["distance_mapping"],
        "BIOCS_SAMPLER_SEED": config["sampler_seed"],
    }
    for key, value in canonical.items():
        os.environ[key] = str(value)


def apply_hparam_overrides(hparams):
    """Apply optional env overrides for smoke tests and cluster scheduling."""
    overrides = {
        "KE_MOM2_N_SAMPLES": ("mom2_n_samples", int),
        "KE_MOM2_DATASET": ("mom2_dataset", str),
        "KE_MOM2_DTYPE": ("mom2_dtype", str),
        "KE_V_NUM_GRAD_STEPS": ("v_num_grad_steps", int),
        "KE_V_LR": ("v_lr", float),
        "KE_STATS_DIR": ("stats_dir", str),
        "KE_ALPHAEDIT_P_LOC": ("P_loc", str),
    }
    for env_name, (attr_name, caster) in overrides.items():
        if env_name in os.environ and hasattr(hparams, attr_name):
            value = caster(os.environ[env_name])
            setattr(hparams, attr_name, value)
            logger.info(f"Override hparams.{attr_name}={value!r} from {env_name}")

    if "KE_MOM2_ADJUSTMENT" in os.environ and hasattr(hparams, "mom2_adjustment"):
        value = _parse_bool(os.environ["KE_MOM2_ADJUSTMENT"])
        setattr(hparams, "mom2_adjustment", value)
        logger.info(f"Override hparams.mom2_adjustment={value!r} from KE_MOM2_ADJUSTMENT")

    if "KE_LAYERS" in os.environ and hasattr(hparams, "layers"):
        value = _parse_int_list(os.environ["KE_LAYERS"])
        setattr(hparams, "layers", value)
        logger.info(f"Override hparams.layers={value!r} from KE_LAYERS")

    if hasattr(hparams, "P_loc"):
        Path(hparams.P_loc).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    if hasattr(hparams, "stats_dir"):
        Path(hparams.stats_dir).expanduser().resolve().mkdir(parents=True, exist_ok=True)
    return hparams


def patch_transformers_for_easyedit():
    """Provide small compatibility shims for EasyEdit on newer/older transformers."""
    try:
        import importlib
        import torch
        import transformers.pytorch_utils as pytorch_utils
        import transformers.generation as generation
        import transformers.generation.utils as generation_utils
    except Exception:
        return

    beam_mod_name = "transformers.generation.beam_search"
    stale_beam_module = sys.modules.get(beam_mod_name)
    if stale_beam_module is not None and not hasattr(stale_beam_module, "BeamScorer"):
        sys.modules.pop(beam_mod_name, None)
    try:
        beam_search_module = importlib.import_module(beam_mod_name)
    except ModuleNotFoundError as error:
        # Transformers 5 removed this legacy module. EasyEdit's optional DECO
        # path already handles its absence, and the LoRA editor does not use it.
        if error.name != beam_mod_name:
            raise
        beam_search_module = None
    if beam_search_module is not None:
        for name in ("BeamScorer", "BeamSearchScorer"):
            if not hasattr(beam_search_module, name) and hasattr(generation, name):
                setattr(beam_search_module, name, getattr(generation, name))
            if hasattr(beam_search_module, name) and not hasattr(generation, name):
                setattr(generation, name, getattr(beam_search_module, name))

    if not hasattr(pytorch_utils, "find_pruneable_heads_and_indices"):
        def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
            mask = torch.ones(n_heads, head_size)
            heads = set(heads) - already_pruned_heads
            for head in heads:
                head = head - sum(1 if pruned_head < head else 0 for pruned_head in already_pruned_heads)
                mask[head] = 0
            mask = mask.view(-1).contiguous().eq(1)
            index = torch.arange(len(mask))[mask].long()
            return heads, index

        pytorch_utils.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices

    generation_aliases = {
        "GreedySearchOutput": "GenerateNonBeamOutput",
        "GreedySearchDecoderOnlyOutput": "GenerateDecoderOnlyOutput",
        "GreedySearchEncoderDecoderOutput": "GenerateEncoderDecoderOutput",
        "SampleOutput": "GenerateNonBeamOutput",
        "SampleDecoderOnlyOutput": "GenerateDecoderOnlyOutput",
        "SampleEncoderDecoderOutput": "GenerateEncoderDecoderOutput",
        "BeamSearchOutput": "GenerateBeamOutput",
        "BeamSearchDecoderOnlyOutput": "GenerateBeamDecoderOnlyOutput",
        "BeamSearchEncoderDecoderOutput": "GenerateBeamEncoderDecoderOutput",
        "BeamSampleOutput": "GenerateBeamOutput",
        "BeamSampleDecoderOnlyOutput": "GenerateBeamDecoderOnlyOutput",
        "BeamSampleEncoderDecoderOutput": "GenerateBeamEncoderDecoderOutput",
    }
    for missing_name, fallback_name in generation_aliases.items():
        if not hasattr(generation_utils, missing_name) and hasattr(generation_utils, fallback_name):
            setattr(generation_utils, missing_name, getattr(generation_utils, fallback_name))


def patch_easyedit_dataset_loading():
    """Allow EasyEdit covariance stats to load HF datasets non-interactively."""
    try:
        import datasets

        original_hf_load_dataset = datasets.load_dataset

        def hf_load_dataset_with_trust(*args, **kwargs):
            if args and args[0] == "wikipedia":
                if len(args) > 1 and args[1] == "20200501.en":
                    args = (args[0], "20220301.en", *args[2:])
                kwargs.setdefault("trust_remote_code", True)
            return original_hf_load_dataset(*args, **kwargs)

        datasets.load_dataset = hf_load_dataset_with_trust
    except Exception:
        pass

    try:
        import easyeditor.models.rome.layer_stats as layer_stats
    except Exception:
        layer_stats = None

    modules = []
    if layer_stats is not None:
        modules.append(layer_stats)
    modules.extend(
        module
        for name, module in list(sys.modules.items())
        if name.endswith("layer_stats") and hasattr(module, "load_dataset")
    )

    for module in set(modules):
        original_load_dataset = module.load_dataset

        def load_dataset_with_trust(*args, _original=original_load_dataset, **kwargs):
            if args and args[0] == "wikipedia":
                if len(args) > 1 and args[1] == "20200501.en":
                    args = (args[0], "20220301.en", *args[2:])
                kwargs.setdefault("trust_remote_code", True)
            return _original(*args, **kwargs)

        module.load_dataset = load_dataset_with_trust


def load_dataset(name: str, n_edits: int, offset: int = 0):
    if offset < 0:
        raise ValueError("offset must be non-negative")
    path = DATASET_MAP[name]
    with open(path) as f:
        data = json.load(f)
    data = data[offset : offset + n_edits]
    prompts, targets, grounds, subjects = [], [], [], []
    locality_inputs, locality_labels = [], []
    rephrase_prompts = []

    from llm_ke.biocs_editor import normalize_knowedit_record

    for raw_item in data:
        item = normalize_knowedit_record(raw_item)
        prompts.append(item["prompt"])
        targets.append(item["target_new"])
        grounds.append(item["ground_truth"])
        subjects.append(item["subject"])

        loc = item.get("locality", {})
        loc_item = first_locality_item(loc)
        if loc_item is not None:
            locality_inputs.append(loc_item["prompt"])
            labels = normalize_text_list(loc_item["ground_truths"])
            locality_labels.append(labels[0] if labels else "")
        else:
            locality_inputs.append("")
            locality_labels.append("")

        rp = raw_item.get("rephrase_prompt", raw_item.get("rephrase", ""))
        if isinstance(rp, list):
            rp = rp[0] if rp else ""
        rephrase_prompts.append(rp)

    return {
        "prompts": prompts,
        "target_new": targets,
        "ground_truth": grounds,
        "subject": subjects,
        "locality_inputs": locality_inputs,
        "locality_labels": locality_labels,
        "rephrase_prompts": rephrase_prompts,
    }


def run_easyedit_method(
    method: str,
    dataset_name: str,
    n_edits: int,
    output_dir: str,
    model_name: str,
    hparams_model: str,
    hparams_dir: str,
):
    """Run a KE method via EasyEdit framework."""
    method_upper = method.upper()

    hparams_map = {
        "ROME": ("ROMEHyperParams", os.path.join(hparams_dir, "ROME", f"{hparams_model}.yaml")),
        "MEMIT": ("MEMITHyperParams", os.path.join(hparams_dir, "MEMIT", f"{hparams_model}.yaml")),
        "ALPHAEDIT": ("AlphaEditHyperParams", os.path.join(hparams_dir, "AlphaEdit", f"{hparams_model}.yaml")),
    }

    if method_upper not in hparams_map:
        raise ValueError(f"Unknown method: {method}. Available: {list(hparams_map.keys())}")

    cls_name, yaml_path = hparams_map[method_upper]
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"EasyEdit hparams not found: {yaml_path}. Set KE_HPARAMS_MODEL to a supported file stem.")

    data_offset = int(os.environ.get("KE_DATA_OFFSET", "0"))
    data = load_dataset(dataset_name, n_edits, data_offset)
    logger.info(f"Method={method_upper}, Dataset={dataset_name}, Model={model_name}, N={len(data['prompts'])}")

    locality_data = None
    if any(data["locality_inputs"]):
        if len(data["locality_inputs"]) != len(data["prompts"]):
            raise ValueError("EasyEdit locality prompts are not aligned with edit prompts")
        if len(data["locality_labels"]) != len(data["prompts"]):
            raise ValueError("EasyEdit locality labels are not aligned with edit prompts")
        if not all(data["locality_inputs"]):
            logger.warning(
                "Some edits have no locality prompt; retaining empty placeholders to "
                "preserve one-to-one EasyEdit prompt alignment"
            )
        locality_data = {
            "neighborhood": {
                "prompt": list(data["locality_inputs"]),
                "ground_truth": list(data["locality_labels"]),
            }
        }

    with isolated_easyedit_imports():
        patch_transformers_for_easyedit()
        from easyeditor import BaseEditor
        import easyeditor
        patch_easyedit_dataset_loading()

        HParamsCls = getattr(easyeditor, cls_name)
        hparams = HParamsCls.from_hparams(yaml_path)
        hparams.model_name = model_name
        hparams = apply_hparam_overrides(hparams)

        editor = BaseEditor.from_hparams(hparams)

        t0 = time.time()
        metrics, edited_model, _ = editor.edit(
            prompts=data["prompts"],
            target_new=data["target_new"],
            ground_truth=data["ground_truth"],
            subject=data["subject"],
            locality_inputs=locality_data,
            sequential_edit=True,
        )
        elapsed = time.time() - t0

    agg = {"efficacy": [], "locality": [], "generalization": []}
    succeeded = 0
    for m in metrics:
        if isinstance(m, dict) and isinstance(m.get("post"), dict):
            succeeded += 1
            post = m["post"]
            agg["efficacy"].append(post.get("rewrite_acc", [0])[0] if isinstance(post.get("rewrite_acc"), list) else post.get("rewrite_acc", 0))
            loc_vals = post.get("locality", {})
            if isinstance(loc_vals, dict):
                for v in loc_vals.values():
                    if isinstance(v, (int, float)):
                        agg["locality"].append(v)
                    elif isinstance(v, list) and v:
                        agg["locality"].append(v[0])

    attempted = len(data["prompts"])
    failed = attempted - succeeded
    status = "complete" if succeeded == n_edits and failed == 0 else (
        "failed" if attempted > 0 and succeeded == 0 else "incomplete"
    )
    easyedit_version = str(getattr(easyeditor, "__version__", "unknown"))
    summary = {
        "method": method_upper,
        "dataset": dataset_name,
        "model": model_name,
        "n_edits": len(data["prompts"]),
        "requested_n_edits": n_edits,
        "attempted": attempted,
        "succeeded": succeeded,
        "failed": failed,
        "status": status,
        "data_offset": data_offset,
        "elapsed_s": round(elapsed, 1),
        "efficacy": round(sum(agg["efficacy"]) / max(len(agg["efficacy"]), 1) * 100, 2),
        "locality": round(sum(agg["locality"]) / max(len(agg["locality"]), 1) * 100, 2),
        "raw_metrics": metrics,
        "locality_evaluator": "easyedit_native_locality",
        "locality_evaluator_version": easyedit_version,
    }

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"Saved to {out_path}")
    logger.info(f"Efficacy={summary['efficacy']:.1f}%, Locality={summary['locality']:.1f}%, Time={elapsed:.0f}s")
    return summary


def run_biocs_method(
    dataset_name: str,
    n_edits: int,
    output_dir: str,
    model_name: str,
    run_config: dict | None = None,
    history_run_config: dict | None = None,
):
    """Run our SSR KE method."""
    sys.path.insert(0, str(PROJECT_DIR))
    from llm_ke.biocs_editor import BioCsLLMEditor, KnowEditDataset, run_sequential_editing

    config = run_config or resolve_biocs_run_config()
    data_path = DATASET_MAP[dataset_name]
    target_layers = None
    if os.environ.get("BIOCS_TARGET_LAYERS"):
        target_layers = _parse_int_list(os.environ["BIOCS_TARGET_LAYERS"])
    editor = BioCsLLMEditor(
        model_name=model_name,
        device=os.environ.get("KE_EDITOR_DEVICE", "cuda"),
        target_layers=target_layers,
        recipe=config["recipe"],
        lambda_ssr=config["lambda_ssr"],
        lambda_spectral=config["lambda_spectral"],
        lambda_anchor=config["lambda_anchor"],
        kernel_family=os.environ.get("BIOCS_KERNEL_FAMILY", "gaussian"),
        a_exc=float(os.environ.get("BIOCS_A_EXC", "1.0")),
        a_inh=float(os.environ.get("BIOCS_A_INH", "0.8")),
        sigma_exc=float(os.environ.get("BIOCS_SIGMA_EXC", "0.3")),
        sigma_inh=float(os.environ.get("BIOCS_SIGMA_INH", "0.8")),
        biocs_target=os.environ.get("BIOCS_TARGET", "weight"),
        distance_mapping=config["distance_mapping"],
        row_selection=os.environ.get("BIOCS_ROW_SELECTION", "fixed_random"),
        max_spatial_rows=int(os.environ.get("BIOCS_MAX_SPATIAL_ROWS", "256")),
        sampler_seed=config["sampler_seed"],
        lr=float(os.environ.get("BIOCS_LR", "1e-4")),
        num_steps=int(os.environ.get("BIOCS_NUM_STEPS", "25")),
        optimizer_name=os.environ.get("BIOCS_OPTIMIZER", "adam"),
        adam_beta1=float(os.environ.get("BIOCS_ADAM_BETA1", "0.9")),
        adam_beta2=float(os.environ.get("BIOCS_ADAM_BETA2", "0.999")),
        adam_eps=float(os.environ.get("BIOCS_ADAM_EPS", "1e-8")),
        weight_decay=float(os.environ.get("BIOCS_WEIGHT_DECAY", "0.0")),
        gradient_clip_norm=float(os.environ.get("BIOCS_GRADIENT_CLIP_NORM", "1.0")),
        max_length=int(os.environ.get("KE_MAX_LENGTH", "64")),
        max_new_tokens=int(os.environ.get("KE_MAX_NEW_TOKENS", "32")),
    )
    dataset = KnowEditDataset(
        data_path,
        max_samples=n_edits,
        offset=int(os.environ.get("KE_DATA_OFFSET", "0")),
    )

    os.makedirs(output_dir, exist_ok=True)
    summary = run_sequential_editing(
        editor,
        dataset,
        n_edits,
        os.path.join(output_dir, "results.json"),
        **(history_run_config or {}),
    )
    return summary


def run_ft_method(
    dataset_name: str,
    n_edits: int,
    output_dir: str,
    model_name: str,
    history_run_config: dict | None = None,
):
    """Run vanilla FT KE baseline."""
    sys.path.insert(0, str(PROJECT_DIR))
    from llm_ke.biocs_editor import FTBaselineEditor, KnowEditDataset, run_sequential_editing

    requested_anchor = float(os.environ.get("FT_LAMBDA_ANCHOR", "0.0"))
    if requested_anchor != 0.0:
        raise ValueError(
            "FTBaselineEditor is the unregularized control; FT_LAMBDA_ANCHOR "
            "must be zero. Use the SSR editor with BIOCS_LAMBDA=0 for an "
            "anchor-only control."
        )

    data_path = DATASET_MAP[dataset_name]
    target_layers = None
    if os.environ.get("BIOCS_TARGET_LAYERS"):
        target_layers = _parse_int_list(os.environ["BIOCS_TARGET_LAYERS"])
    editor = FTBaselineEditor(
        model_name=model_name,
        device=os.environ.get("KE_EDITOR_DEVICE", "cuda"),
        target_layers=target_layers,
        lr=float(os.environ.get("FT_LR", os.environ.get("BIOCS_LR", "1e-4"))),
        num_steps=int(os.environ.get("FT_NUM_STEPS", os.environ.get("BIOCS_NUM_STEPS", "25"))),
        optimizer_name=os.environ.get("BIOCS_OPTIMIZER", "adam"),
        adam_beta1=float(os.environ.get("BIOCS_ADAM_BETA1", "0.9")),
        adam_beta2=float(os.environ.get("BIOCS_ADAM_BETA2", "0.999")),
        adam_eps=float(os.environ.get("BIOCS_ADAM_EPS", "1e-8")),
        weight_decay=float(os.environ.get("BIOCS_WEIGHT_DECAY", "0.0")),
        gradient_clip_norm=float(os.environ.get("BIOCS_GRADIENT_CLIP_NORM", "1.0")),
        max_length=int(os.environ.get("KE_MAX_LENGTH", "64")),
        max_new_tokens=int(os.environ.get("KE_MAX_NEW_TOKENS", "32")),
    )
    dataset = KnowEditDataset(
        data_path,
        max_samples=n_edits,
        offset=int(os.environ.get("KE_DATA_OFFSET", "0")),
    )

    os.makedirs(output_dir, exist_ok=True)
    summary = run_sequential_editing(
        editor,
        dataset,
        n_edits,
        os.path.join(output_dir, "results.json"),
        **(history_run_config or {}),
    )
    return summary


def write_run_manifest(
    args: argparse.Namespace,
    resolved_editing_config: dict | None = None,
    history_run_config: dict | None = None,
) -> Path:
    """Persist the inputs that control editor construction for this run."""
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    keys = [
        "KE_SEED",
        "BIOCS_RECIPE",
        "BIOCS_LAMBDA_SSR",
        "BIOCS_LAMBDA",
        "BIOCS_LAMBDA_SPECTRAL",
        "BIOCS_LAMBDA_ANCHOR",
        "BIOCS_KERNEL_FAMILY",
        "BIOCS_A_EXC",
        "BIOCS_A_INH",
        "BIOCS_SIGMA_EXC",
        "BIOCS_SIGMA_INH",
        "BIOCS_TARGET",
        "BIOCS_DISTANCE_MAPPING",
        "BIOCS_DISTANCE_METRIC",
        "BIOCS_ROW_SELECTION",
        "BIOCS_MAX_SPATIAL_ROWS",
        "BIOCS_SAMPLER_SEED",
        "BIOCS_TARGET_MODULE_REGEX",
        "BIOCS_MAX_TARGET_MODULES",
        "BIOCS_EDITABLE_PARAMETER",
        "BIOCS_ADAPTER_MODE",
        "BIOCS_OPTIMIZER",
        "BIOCS_ADAM_BETA1",
        "BIOCS_ADAM_BETA2",
        "BIOCS_ADAM_EPS",
        "BIOCS_WEIGHT_DECAY",
        "BIOCS_GRADIENT_CLIP_NORM",
        "KE_DATA_OFFSET",
        "KE_MODEL_DTYPE",
        "KE_DEVICE_MAP",
        "KE_FORCE_TEXT_ONLY",
        "KE_LOCAL_ONLY",
        "KE_MODEL_FINGERPRINT",
        "KE_PAIRING_PROTOCOL_HASH",
        "KE_EXPECTED_LOCALITY_EVALUATOR",
        "KE_EXPECTED_LOCALITY_EVALUATOR_VERSION",
        "KE_EXPECTED_LOCALITY_PROTOCOL_HASH",
        "KE_REQUIRE_LEGACY_LOCALITY_COMPATIBLE",
        "KE_EVALUATION_PROTOCOL_HASH",
        "KE_MAX_LENGTH",
        "KE_MAX_NEW_TOKENS",
        "KE_ATTN_IMPLEMENTATION",
        "KE_EDITOR_DEVICE",
        "KE_DETERMINISM_POLICY",
        "KE_CUDNN_DETERMINISTIC",
        "KE_CUDNN_BENCHMARK",
        "KE_USE_DETERMINISTIC_ALGORITHMS",
        "KE_DETERMINISTIC_ALGORITHMS_WARN_ONLY",
        "KE_LOCK_NAME",
        "KE_LOCKED_CONFIG_PATH",
        "KE_LOCKED_CONFIG_SHA256",
        "PYTHONHASHSEED",
        "CUDA_VISIBLE_DEVICES",
        "CUBLAS_WORKSPACE_CONFIG",
        "PYTORCH_CUDA_ALLOC_CONF",
        "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
        "KE_MOM2_N_SAMPLES",
        "KE_MOM2_DATASET",
        "KE_MOM2_DTYPE",
        "KE_V_NUM_GRAD_STEPS",
        "KE_V_LR",
        "KE_STATS_DIR",
        "KE_ALPHAEDIT_P_LOC",
        "KE_MOM2_ADJUSTMENT",
        "KE_LAYERS",
        "BIOCS_LR",
        "BIOCS_NUM_STEPS",
        "BIOCS_TARGET_LAYERS",
        "FT_LR",
        "FT_NUM_STEPS",
    ]
    record = {
        "command": vars(args),
        "environment": {key: os.environ[key] for key in keys if key in os.environ},
        "runtime_policy": configure_runtime_from_env(),
    }
    if resolved_editing_config is not None:
        record["editing_objective"] = resolved_editing_config
    if history_run_config is not None:
        record["historical_retention_evaluation"] = history_run_config
    method = args.method.upper()
    if method in {"ROME", "MEMIT", "ALPHAEDIT"}:
        hparams_folder = {"ROME": "ROME", "MEMIT": "MEMIT", "ALPHAEDIT": "AlphaEdit"}[method]
        hparams_path = Path(args.hparams_dir) / hparams_folder / f"{args.hparams_model}.yaml"
        record["hparams"] = {
            "path": str(hparams_path),
            "sha256": (
                hashlib.sha256(hparams_path.read_bytes()).hexdigest()
                if hparams_path.is_file()
                else None
            ),
        }
        try:
            easyedit_commit = subprocess.run(
                ["git", "-C", str(EASYEDIT_DIR), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            easyedit_commit = None
        record["easyedit_git_commit"] = easyedit_commit
    manifest_path = output / "run_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(record, indent=2) + "\n")
    return manifest_path


def write_result_record(
    args: argparse.Namespace,
    summary: dict,
    resolved_editing_config: dict | None,
    run_manifest_path: Path,
) -> Path:
    """Write the task-independent sidecar consumed by paired aggregation."""
    sys.path.insert(0, str(PROJECT_DIR))
    from ssr_utils.result_schema import (
        ResultSchemaError,
        build_result_record,
        sha256_file,
        sha256_value,
    )

    if args.method == "biocs":
        config = resolved_editing_config or resolve_biocs_run_config(args)
        recipe = config["recipe"]
        objective = config["objective"]
        uses_ssr = bool(objective["ssr"])
        distance_mapping = config["distance_mapping"] if uses_ssr else "none"
        seed = config["seed"]
    elif args.method == "ft":
        recipe = "plain"
        objective = {"task": True}
        uses_ssr = False
        distance_mapping = "none"
        seed = int(args.seed if args.seed is not None else os.environ.get("KE_SEED", "0"))
    else:
        recipe = f"{args.method.lower()}_context"
        objective = {"task": True}
        uses_ssr = False
        distance_mapping = "none"
        seed = int(args.seed if args.seed is not None else os.environ.get("KE_SEED", "0"))

    kernel = {}
    if uses_ssr:
        kernel = {
            "family": os.environ.get("BIOCS_KERNEL_FAMILY", "gaussian"),
            "A_exc": float(os.environ.get("BIOCS_A_EXC", "1.0")),
            "A_inh": float(os.environ.get("BIOCS_A_INH", "0.8")),
            "sigma_exc": float(os.environ.get("BIOCS_SIGMA_EXC", "0.3")),
            "sigma_inh": float(os.environ.get("BIOCS_SIGMA_INH", "0.8")),
        }

    missing_metrics = {"efficacy", "locality"} - set(summary)
    if missing_metrics:
        raise ResultSchemaError(
            f"Editing summary is missing official metrics: {sorted(missing_metrics)}"
        )
    metrics = {
        "efficacy": float(summary["efficacy"]),
        "locality": float(summary["locality"]),
    }
    final = summary.get("historical_retention", {}).get("final")
    if isinstance(final, dict):
        metrics.update(
            {
                "final_history_efficacy": float(final.get("efficacy", 0.0)),
                "final_history_locality": float(final.get("locality", 0.0)),
            }
        )
    data_path = Path(DATASET_MAP[args.dataset])
    config_payload = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    # Keep the preflight lock and the values resolved by Transformers in the
    # same manifest.  The latter are only available after model construction.
    resolved_keys = (
        "target_layers",
        "target_module_regex",
        "max_target_modules",
        "editable_parameter",
        "adapter_mode",
        "editable_module_names",
        "optimizer",
        "max_length",
        "max_new_tokens",
        "model_dtype",
        "resolved_model_dtype",
        "device_map",
        "requested_editor_device",
        "resolved_input_device",
        "attention_implementation",
        "resolved_attention_implementation",
    )
    resolved_execution = {
        key: summary[key] for key in resolved_keys if key in summary
    }
    if resolved_execution:
        config_payload["resolved_execution"] = resolved_execution
        run_manifest_path.write_text(
            json.dumps(config_payload, indent=2) + "\n", encoding="utf-8"
        )
    completion = {
        name: summary.get(name)
        for name in ("attempted", "succeeded", "failed", "status")
    }
    requested = summary.get("requested_n_edits")
    if requested is None or any(value is None for value in completion.values()):
        raise ResultSchemaError(
            "Editing summary must report requested_n_edits, attempted, succeeded, failed, and status"
        )
    evaluator = str(summary.get("locality_evaluator", ""))
    evaluator_version = str(summary.get("locality_evaluator_version", ""))
    evaluation_protocol_hash = str(
        summary.get("locality_evaluator_protocol_hash", LOCALITY_PROTOCOL_HASH)
    )
    expected_evaluator = os.environ.get("KE_EXPECTED_LOCALITY_EVALUATOR")
    expected_evaluator_version = os.environ.get(
        "KE_EXPECTED_LOCALITY_EVALUATOR_VERSION"
    )
    expected_protocol_hash = os.environ.get("KE_EXPECTED_LOCALITY_PROTOCOL_HASH")
    observed_protocol = (evaluator, evaluator_version, evaluation_protocol_hash)
    expected_protocol = (
        expected_evaluator or evaluator,
        expected_evaluator_version or evaluator_version,
        expected_protocol_hash or evaluation_protocol_hash,
    )
    if observed_protocol != expected_protocol:
        raise ResultSchemaError(
            "Locality evaluator does not match the locked protocol: "
            f"observed={observed_protocol!r}, expected={expected_protocol!r}"
        )
    model_hash = os.environ.get("KE_MODEL_FINGERPRINT") or sha256_value(
        {"model_reference": str(args.model_name)}
    )
    pairing_protocol_hash = os.environ.get("KE_PAIRING_PROTOCOL_HASH") or sha256_value(
        {
            "run_manifest": config_payload,
            "evaluation_protocol_hash": evaluation_protocol_hash,
        }
    )
    is_complete = completion["status"] == "complete"
    record = build_result_record(
        git_commit=current_git_commit(),
        run_id=str(Path(args.output).resolve()),
        task_family="editing",
        dataset=args.dataset,
        model=safe_tag(getattr(args, "hparams_model", Path(args.model_name).name)),
        seed=seed,
        objective=objective,
        distance_mapping=distance_mapping,
        kernel=kernel,
        metrics=metrics,
        runtime={
            "elapsed_s": float(summary.get("elapsed_s", 0.0)),
            "peak_cuda_allocated_mb": float(summary.get("peak_cuda_allocated_mb", 0.0)),
            "peak_cuda_reserved_mb": float(summary.get("peak_cuda_reserved_mb", 0.0)),
        },
        config=config_payload,
        dataset_hash=sha256_file(data_path),
        allow_incomplete=not is_complete,
        recipe=recipe,
        n_edits=int(summary.get("n_edits", args.n_edits)),
        requested=int(requested),
        attempted=int(completion["attempted"]),
        succeeded=int(completion["succeeded"]),
        failed=int(completion["failed"]),
        status=str(completion["status"]),
        data_offset=int(summary.get("data_offset", os.environ.get("KE_DATA_OFFSET", "0"))),
        metric_directions={key: True for key in metrics},
        evaluator=evaluator,
        evaluator_version=evaluator_version,
        evaluation_protocol_hash=evaluation_protocol_hash,
        model_hash=model_hash,
        pairing_protocol_hash=pairing_protocol_hash,
    )
    result_manifest = getattr(args, "result_manifest", None)
    path = Path(result_manifest) if result_manifest else Path(args.output) / "result_record.json"
    if not is_complete:
        path = (
            path.with_name("incomplete_result_record.json")
            if path.name == "result_record.json"
            else path.with_name(f"{path.stem}.incomplete{path.suffix}")
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not is_complete:
        raise ResultSchemaError(
            f"Editing run is not official ({completion}); diagnostic record written to {path}"
        )
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True,
                        choices=["ROME", "MEMIT", "AlphaEdit", "biocs", "ft"])
    parser.add_argument("--dataset", type=str, default="zsre",
                        choices=["zsre", "cf", "recent", "wikibio"])
    parser.add_argument("--n_edits", type=int, default=200)
    parser.add_argument("--model_name", type=str, default=MODEL_NAME,
                        help="HF model id or local model path. Prefer a local path on offline clusters.")
    parser.add_argument("--hparams_model", type=str, default=HPARAMS_MODEL,
                        help="EasyEdit hparams file stem, e.g. gpt2-xl, llama3.2-3b, qwen2.5-7b.")
    parser.add_argument("--hparams_dir", type=str, default=HPARAMS_DIR,
                        help="Directory containing ROME/MEMIT/AlphaEdit hparams subdirectories.")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None,
                        help="Set KE_SEED for deterministic Python, NumPy, and PyTorch sampling.")
    parser.add_argument(
        "--recipe",
        choices=[
            "plain", "anchor", "spectral", "stabilized", "ssr_only",
            "ssr_anchor", "ssr_spectral", "full",
        ],
        default=None,
        help="Explicit objective recipe. If omitted, infer it from legacy coefficients.",
    )
    parser.add_argument("--distance_mapping", "--distance-mapping",
                        choices=["cosine", "projective"], default=None,
                        help="Distance used by SSR; CLI overrides BIOCS_DISTANCE_MAPPING/METRIC.")
    parser.add_argument("--lambda_ssr", type=float, default=None,
                        help="SSR coefficient; CLI overrides BIOCS_LAMBDA_SSR/BIOCS_LAMBDA.")
    parser.add_argument("--lambda_anchor", type=float, default=None,
                        help="Anchor coefficient; CLI overrides BIOCS_LAMBDA_ANCHOR.")
    parser.add_argument("--lambda_spectral", type=float, default=None,
                        help="Spectral coefficient; CLI overrides BIOCS_LAMBDA_SPECTRAL.")
    parser.add_argument("--result_manifest", type=str, default=None,
                        help="Optional path for the validated per-run result record JSON.")
    parser.add_argument(
        "--evaluate_history",
        action="store_true",
        help="Re-evaluate prior edits after the stream (custom biocs/ft only).",
    )
    parser.add_argument(
        "--history_checkpoints",
        type=int,
        nargs="*",
        default=None,
        metavar="N",
        help="Optional edit counts for interim historical-retention evaluation.",
    )
    parser.add_argument(
        "--history_max_samples",
        type=int,
        default=0,
        help="Maximum retained edits evaluated per checkpoint; 0 evaluates all.",
    )
    parser.add_argument("--biocs_lambda", type=float, default=None,
                        help="Deprecated alias for --lambda_ssr.")
    parser.add_argument("--biocs_anchor", type=float, default=None,
                        help="Deprecated alias for --lambda_anchor.")
    parser.add_argument("--biocs_lr", type=float, default=None,
                        help="Override BIOCS_LR for the SSR editor.")
    parser.add_argument("--biocs_steps", type=int, default=None,
                        help="Override BIOCS_NUM_STEPS for the SSR editor.")
    args = parser.parse_args()

    cli_env = {
        "BIOCS_LR": args.biocs_lr,
        "BIOCS_NUM_STEPS": args.biocs_steps,
    }
    for key, value in cli_env.items():
        if value is not None:
            os.environ[key] = str(value)

    resolved_editing_config = None
    if args.method == "biocs":
        try:
            resolved_editing_config = resolve_biocs_run_config(args)
        except ValueError as exc:
            parser.error(str(exc))
        apply_biocs_run_config(resolved_editing_config)
    else:
        if args.recipe not in {None, "plain"}:
            parser.error("--recipe is only supported by --method biocs (or plain for ft)")
        if args.seed is not None:
            os.environ["KE_SEED"] = str(args.seed)
    configure_runtime_from_env()
    set_seed_from_env()

    try:
        history_run_config = resolve_history_run_config(args)
    except ValueError as exc:
        parser.error(str(exc))

    if args.output is None:
        if args.method == "biocs":
            recipe_tag = resolved_editing_config["recipe"]
            mapping_tag = (
                resolved_editing_config["distance_mapping"]
                if resolved_editing_config["objective"]["ssr"]
                else "none"
            )
            seed_tag = resolved_editing_config["seed"]
        else:
            recipe_tag = "plain" if args.method == "ft" else f"{args.method.lower()}_context"
            mapping_tag = "none"
            seed_tag = int(args.seed if args.seed is not None else os.environ.get("KE_SEED", "0"))
        args.output = str(
            Path("results/llm_ke")
            / safe_tag(args.model_name)
            / args.dataset
            / recipe_tag
            / mapping_tag
            / f"seed_{seed_tag}_n{args.n_edits}"
        )
    run_manifest_path = write_run_manifest(
        args,
        resolved_editing_config,
        history_run_config,
    )

    summary = None
    if args.method in ["ROME", "MEMIT", "AlphaEdit"]:
        summary = run_easyedit_method(
            args.method,
            args.dataset,
            args.n_edits,
            args.output,
            args.model_name,
            args.hparams_model,
            args.hparams_dir,
        )
    elif args.method == "biocs":
        summary = run_biocs_method(
            args.dataset,
            args.n_edits,
            args.output,
            args.model_name,
            resolved_editing_config,
            history_run_config,
        )
    elif args.method == "ft":
        summary = run_ft_method(
            args.dataset,
            args.n_edits,
            args.output,
            args.model_name,
            history_run_config,
        )
    if summary is not None:
        write_result_record(args, summary, resolved_editing_config, run_manifest_path)
