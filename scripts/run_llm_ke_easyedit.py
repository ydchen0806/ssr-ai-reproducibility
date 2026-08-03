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
os.chdir(PROJECT_DIR)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

DATASET_MAP = {
    "zsre": str(PROJECT_DIR / "dataset/knowedit/benchmark/ZsRE/ZsRE-test-all.json"),
    "cf": str(PROJECT_DIR / "dataset/knowedit/benchmark/wiki_counterfact/test_cf.json"),
    "recent": str(PROJECT_DIR / "dataset/knowedit/benchmark/wiki_recent/recent_test.json"),
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


def _parse_bool(text: str) -> bool:
    return text.strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_int_list(text: str) -> list[int]:
    return [int(part.strip()) for part in text.replace(",", " ").split() if part.strip()]


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
    beam_search_module = importlib.import_module(beam_mod_name)
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

    for item in data:
        prompts.append(item["prompt"])
        tn = item.get("target_new", "")
        if isinstance(tn, dict):
            tn = tn.get("str", str(tn))
        targets.append(tn)
        gt = item.get("ground_truth", "")
        if isinstance(gt, dict):
            gt = gt.get("str", str(gt))
        grounds.append(gt)
        subjects.append(item.get("subject", ""))

        loc = item.get("locality", {})
        if loc:
            first_key = next(iter(loc), None)
            if first_key and isinstance(loc[first_key], list) and loc[first_key]:
                loc_item = loc[first_key][0]
                locality_inputs.append(loc_item.get("prompt", ""))
                loc_gt = loc_item.get("ground_truth", "")
                if isinstance(loc_gt, list):
                    loc_gt = loc_gt[0] if loc_gt else ""
                locality_labels.append(str(loc_gt))
            else:
                locality_inputs.append("")
                locality_labels.append("")
        else:
            locality_inputs.append("")
            locality_labels.append("")

        rp = item.get("rephrase_prompt", item.get("rephrase", ""))
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
        valid = [(inp, lbl) for inp, lbl in zip(data["locality_inputs"], data["locality_labels"]) if inp]
        if valid:
            locality_data = {
                "neighborhood": {
                    "prompt": [v[0] for v in valid],
                    "ground_truth": [v[1] for v in valid],
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
    for m in metrics:
        if "post" in m:
            post = m["post"]
            agg["efficacy"].append(post.get("rewrite_acc", [0])[0] if isinstance(post.get("rewrite_acc"), list) else post.get("rewrite_acc", 0))
            loc_vals = post.get("locality", {})
            if isinstance(loc_vals, dict):
                for v in loc_vals.values():
                    if isinstance(v, (int, float)):
                        agg["locality"].append(v)
                    elif isinstance(v, list) and v:
                        agg["locality"].append(v[0])

    summary = {
        "method": method_upper,
        "dataset": dataset_name,
        "model": model_name,
        "n_edits": len(data["prompts"]),
        "data_offset": data_offset,
        "elapsed_s": round(elapsed, 1),
        "efficacy": round(sum(agg["efficacy"]) / max(len(agg["efficacy"]), 1) * 100, 2),
        "locality": round(sum(agg["locality"]) / max(len(agg["locality"]), 1) * 100, 2),
        "raw_metrics": metrics,
    }

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"Saved to {out_path}")
    logger.info(f"Efficacy={summary['efficacy']:.1f}%, Locality={summary['locality']:.1f}%, Time={elapsed:.0f}s")
    return summary


def run_biocs_method(dataset_name: str, n_edits: int, output_dir: str, model_name: str):
    """Run our SSR KE method."""
    sys.path.insert(0, str(PROJECT_DIR))
    from llm_ke.biocs_editor import BioCsLLMEditor, KnowEditDataset, run_sequential_editing

    data_path = DATASET_MAP[dataset_name]
    target_layers = None
    if os.environ.get("BIOCS_TARGET_LAYERS"):
        target_layers = _parse_int_list(os.environ["BIOCS_TARGET_LAYERS"])
    editor = BioCsLLMEditor(
        model_name=model_name,
        device="cuda",
        target_layers=target_layers,
        lambda_biocs=float(os.environ.get("BIOCS_LAMBDA", "0.001")),
        lambda_spectral=float(os.environ.get("BIOCS_LAMBDA_SPECTRAL", "0.01")),
        lambda_anchor=float(os.environ.get("BIOCS_LAMBDA_ANCHOR", "0.001")),
        kernel_family=os.environ.get("BIOCS_KERNEL_FAMILY", "gaussian"),
        a_exc=float(os.environ.get("BIOCS_A_EXC", "1.0")),
        a_inh=float(os.environ.get("BIOCS_A_INH", "0.8")),
        sigma_exc=float(os.environ.get("BIOCS_SIGMA_EXC", "0.3")),
        sigma_inh=float(os.environ.get("BIOCS_SIGMA_INH", "0.8")),
        biocs_target=os.environ.get("BIOCS_TARGET", "weight"),
        distance_metric=os.environ.get("BIOCS_DISTANCE_METRIC", "cosine"),
        row_selection=os.environ.get("BIOCS_ROW_SELECTION", "random_each_step"),
        max_spatial_rows=int(os.environ.get("BIOCS_MAX_SPATIAL_ROWS", "256")),
        sampler_seed=int(os.environ.get("BIOCS_SAMPLER_SEED", os.environ.get("KE_SEED", "0"))),
        lr=float(os.environ.get("BIOCS_LR", "1e-4")),
        num_steps=int(os.environ.get("BIOCS_NUM_STEPS", "25")),
    )
    dataset = KnowEditDataset(
        data_path,
        max_samples=n_edits,
        offset=int(os.environ.get("KE_DATA_OFFSET", "0")),
    )

    os.makedirs(output_dir, exist_ok=True)
    summary = run_sequential_editing(editor, dataset, n_edits, os.path.join(output_dir, "results.json"))
    return summary


def run_ft_method(dataset_name: str, n_edits: int, output_dir: str, model_name: str):
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
        device="cuda",
        target_layers=target_layers,
        lr=float(os.environ.get("FT_LR", "5e-4")),
        num_steps=int(os.environ.get("FT_NUM_STEPS", os.environ.get("BIOCS_NUM_STEPS", "25"))),
    )
    dataset = KnowEditDataset(
        data_path,
        max_samples=n_edits,
        offset=int(os.environ.get("KE_DATA_OFFSET", "0")),
    )

    os.makedirs(output_dir, exist_ok=True)
    summary = run_sequential_editing(editor, dataset, n_edits, os.path.join(output_dir, "results.json"))
    return summary


def write_run_manifest(args: argparse.Namespace) -> None:
    """Persist the inputs that control editor construction for this run."""
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    keys = [
        "KE_SEED",
        "BIOCS_LAMBDA",
        "BIOCS_LAMBDA_SPECTRAL",
        "BIOCS_LAMBDA_ANCHOR",
        "BIOCS_KERNEL_FAMILY",
        "BIOCS_A_EXC",
        "BIOCS_A_INH",
        "BIOCS_SIGMA_EXC",
        "BIOCS_SIGMA_INH",
        "BIOCS_TARGET",
        "BIOCS_DISTANCE_METRIC",
        "BIOCS_ROW_SELECTION",
        "BIOCS_MAX_SPATIAL_ROWS",
        "BIOCS_SAMPLER_SEED",
        "BIOCS_TARGET_MODULE_REGEX",
        "BIOCS_MAX_TARGET_MODULES",
        "KE_DATA_OFFSET",
        "KE_MODEL_DTYPE",
        "KE_DEVICE_MAP",
        "KE_FORCE_TEXT_ONLY",
        "KE_LOCAL_ONLY",
        "KE_ATTN_IMPLEMENTATION",
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
    }
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
    (output / "run_manifest.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True,
                        choices=["ROME", "MEMIT", "AlphaEdit", "biocs", "ft"])
    parser.add_argument("--dataset", type=str, default="zsre",
                        choices=["zsre", "cf", "recent"])
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
    parser.add_argument("--biocs_lambda", type=float, default=None,
                        help="Override BIOCS_LAMBDA for the SSR editor.")
    parser.add_argument("--biocs_anchor", type=float, default=None,
                        help="Override BIOCS_LAMBDA_ANCHOR for the SSR editor.")
    parser.add_argument("--biocs_lr", type=float, default=None,
                        help="Override BIOCS_LR for the SSR editor.")
    parser.add_argument("--biocs_steps", type=int, default=None,
                        help="Override BIOCS_NUM_STEPS for the SSR editor.")
    args = parser.parse_args()
    cli_env = {
        "KE_SEED": args.seed,
        "BIOCS_LAMBDA": args.biocs_lambda,
        "BIOCS_LAMBDA_ANCHOR": args.biocs_anchor,
        "BIOCS_LR": args.biocs_lr,
        "BIOCS_NUM_STEPS": args.biocs_steps,
    }
    for key, value in cli_env.items():
        if value is not None:
            os.environ[key] = str(value)
    set_seed_from_env()

    if args.output is None:
        args.output = f"results/llm_ke/{args.method.lower()}_{safe_tag(args.model_name)}_{args.dataset}_{args.n_edits}"
    write_run_manifest(args)

    if args.method in ["ROME", "MEMIT", "AlphaEdit"]:
        run_easyedit_method(
            args.method,
            args.dataset,
            args.n_edits,
            args.output,
            args.model_name,
            args.hparams_model,
            args.hparams_dir,
        )
    elif args.method == "biocs":
        run_biocs_method(args.dataset, args.n_edits, args.output, args.model_name)
    elif args.method == "ft":
        run_ft_method(args.dataset, args.n_edits, args.output, args.model_name)
