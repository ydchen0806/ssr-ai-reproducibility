# Manuscript Reproduction Map

This is the historical evidence map. The current September 11 Figure 4--6
mapping and checkpoint status are in [manuscript_20260911.md](manuscript_20260911.md).

This document maps every manuscript-facing AI result to a locked configuration,
artifact record and verification command. It separates direct SSR attribution
from observations made with the complete SSR-containing objective.

## Evidence map

| Evidence family | Manuscript role | Public record | Verification |
| --- | --- | --- | --- |
| Continual image classification | Main benchmark and matched geometry analyses | `artifacts/paper_20260803/` | `python3 scripts/verify_paper_artifacts.py` |
| CUB-200 adapter and topology studies | Fine-grained geometry and mapping controls | `artifacts/paper_20260803/raw/adapter/` and `raw/mechanism/` | `python3 scripts/verify_paper_artifacts.py` |
| GPT-2 XL sequential editing | Locked factorial | `artifacts/paper_20260810/counterfact_factorial_summary.json` | `python3 scripts/verify_paper_20260810.py` |
| Segmentation | Complete-objective transfer and direct-SSR boundary audit | `artifacts/paper_20260803/raw/segmentation/`, `artifacts/paper_20260809/` | `python3 scripts/verify_paper_artifacts.py`; `python3 scripts/verify_paper_20260809.py` |
| Raw-image ViT-LoRA | Fresh re-run protocol only | `configs/meeting_20260804/`, `scripts/run_vit_lora_confirmation_8gpu.sh` | `python3 scripts/validate_vit_lora_pairs.py ...` after a new run |

## Environment

Create a Python 3.11 environment and install the base requirements:

```bash
conda create -n ssr-ai python=3.11 -y
conda activate ssr-ai
pip install -r requirements.txt
git clone https://github.com/zjunlp/EasyEdit.git external/EasyEdit
cd external/EasyEdit
git rev-parse HEAD
pip install -e .
cd ../..
export EASYEDIT_DIR="$PWD/external/EasyEdit"
```

The captured completed-run environment is recorded in
`artifacts/paper_20260803/environment.json`. Its EasyEdit revision and the
historical `transformers`-family package versions were not archived, so the
exact historical run cannot honestly be claimed as bit-for-bit replayable.
Install a mutually compatible EasyEdit dependency set, then record
`python --version`, `pip freeze`, `nvidia-smi`, the EasyEdit HEAD, this
repository commit and the model checksum before a new run. Do not treat a
fresh dependency resolution as an exact replay of the historical environment.

## GPT-2 XL Sequential Editing

The manuscript factorial uses `configs/meeting_20260803/locked_editing_gpt2xl.yaml`.
It locks the GPT-2 XL fingerprint, source-order dataset digests, blocks 16--18,
100 edits starting at offset 500, and direct weight updates to every matching
MLP projection `(c_proj|down_proj)$` (`max_target_modules=0`). This cohort does
not insert LoRA or other adapters. Each edit uses 25 Adam updates with
`lr=1e-4`, betas `(0.9, 0.999)`, `eps=1e-8`, zero weight decay and gradient
clip `1.0`; inputs are truncated to 64 tokens and greedy generation is capped
at 32 tokens. The Gaussian SSR tuple is `lambda_ssr=0.003`, `A_exc=1.2`,
`A_inh=0.9`, `sigma_exc=0.22` and `sigma_inh=0.60`; the weight anchor and
spectral coefficients are `0.001` and `0.01`. SSR is applied to the active-top
rows of `Delta W`, capped at 256 rows.

The locked runtime policy is `seeded_best_effort`: each paired run sets Python,
NumPy, PyTorch, CUDA and SSR row-sampling seeds to its listed seed and disables
cuDNN benchmarking, while intentionally not claiming bitwise determinism across
drivers or GPU architectures. `model_dtype=auto` resolves to bf16 when the
active CUDA device supports it and fp16 otherwise; empty `device_map` and
`attention_implementation` retain standard Transformers behavior. The runner
writes the requested values to `locked_runtime.tsv` and each `run_manifest.json`
and appends resolved module names, dtype, input device and optimizer values
after the model is constructed.

Verify the locked model and dataset assets before running:

```bash
python scripts/verify_locked_editing_assets.py \
  --config configs/meeting_20260803/locked_editing_gpt2xl.yaml \
  --model-path "$KE_MODEL_NAME_OR_PATH" \
  --output results/gpt2xl_locked_assets.json
```

Run the ten-seed factorial on one eight-GPU node:

```bash
GPU_LIST="0 1 2 3 4 5 6 7" \
RUN_ID=gpt2xl_ssr_factorial \
RESULT_ROOT="$PWD/results/gpt2xl_ssr_factorial" \
MODEL_NAME="$KE_MODEL_NAME_OR_PATH" \
bash scripts/run_meeting_editing_8gpu.sh --phase confirm
```

The expected recipe matrix is plain task loss, task+anchor, task+spectral,
weight-anchor+spectral, task+SSR, and the full recipe. The last two are run for
cosine and projective distance. Summaries must be generated from the planned
matrix, which checks paired seed identity, data slice and evaluator:

```bash
python scripts/summarize_direct_editing.py \
  --plan results/gpt2xl_ssr_factorial/planned_runs.tsv \
  --output-dir results/gpt2xl_ssr_factorial/paired_summary
```

The manuscript evidence record is
`artifacts/paper_20260810/counterfact_factorial_summary.json`; verify all
reported values with `python scripts/verify_paper_20260810.py`.

### Source datasets

The lock file checks the exact digest of each released JSON file before a new
matrix starts.

| Formal dataset name | Locked path | Original source |
| --- | --- | --- |
| ZsRE | `dataset/knowedit/benchmark/ZsRE/ZsRE-test-all.json` | [Levy et al., CoNLL 2017](https://aclanthology.org/K17-1034/) |
| KnowEdit WikiData_counterfact | `dataset/knowedit/benchmark/wiki_counterfact/test_cf.json` | [KnowEdit release](https://github.com/zjunlp/EasyEdit/blob/main/examples/KnowEdit.md); related original CounterFact/ROME benchmark: [Meng et al., NeurIPS 2022](https://rome.baulab.info/) |
| WikiRecent | `dataset/knowedit/benchmark/wiki_recent/recent_test.json` | KnowEdit release; use the version whose hash matches the lock |

The model is [GPT-2 XL](https://huggingface.co/openai-community/gpt2-xl).
ROME is the editing method introduced by Meng et al.; it is an optional
EasyEdit baseline and is not a substitute for the locked SSR factorial.

## CUB Classification and Adapter Studies

The primary CUB classification result and equal-budget geometry controls are
archived under `artifacts/paper_20260803`. The central direct comparison is the
same KD scaffold with SSR disabled or enabled. The adapter factorization uses
frozen 512-dimensional ResNet-18 features, a rank-16 GELU bottleneck with
scale 0.5, cosine classifier temperature 12, AdamW (`lr=1e-3`, `wd=1e-3`),
batch size 128, clip 5, and KD (`lambda=2`, `T=3`). Prototype and adapter-basis
SSR are explicit targets. The independent rank/mapping validation uses 40
epochs per task and ten paired seeds.

The locked rank-16 all-cosine Gaussian condition can be rerun directly as
follows; the command writes its resolved arguments and per-seed rows next to
the summary:

```bash
python experiments/lowrank_adapter_mapping.py \
  --feature_cache data/cub200/cub200_resnet18_features.pt \
  --output_dir results/cub_adapter_rank16_all_cosine \
  --methods kd biocs_kd \
  --seeds 8411 8413 8415 8417 8419 8421 8423 8425 8427 8429 \
  --num_classes 200 --classes_per_task 10 --class_order semantic \
  --rank 16 --adapter_scale 0.5 --tau 12 \
  --epochs 40 --batch_size 128 --lr 1e-3 --weight_decay 1e-3 \
  --lambda_kd 2 --kd_temperature 3 --lambda_cls 1 --lambda_adapter 0.2 \
  --a-exc 1.0 --a-inh 0.8 --sigma-exc 0.55 --sigma-inh 1.25 \
  --kernel-family gaussian \
  --prototype-distance-metric cosine --adapter-distance-metric cosine \
  --grad_clip 5 --device cuda
```

The archived adapter protocol and the complete radial/mapping configurations
are recorded in `artifacts/paper_20260803/raw/adapter/protocol.json` and
`artifacts/paper_20260803/raw/adapter/configs.json`.

## Segmentation

The historical complete-objective transfer uses binary cross-entropy with
logits plus soft Dice, 192-pixel masks, a 256-dimensional decoder, batch 24,
24 epochs per task and AdamW (`lr=5e-4`, `wd=1e-4`). It is not a direct SSR
attribution because KD and SSR coefficients differ from the unregularized arm.
The direct CUB geometry audit instead compares task-only with task+SSR, uses
a Cauchy kernel (`A_exc=1.0`, `A_inh=0.8`, `sigma_exc=0.188`,
`sigma_inh=0.53`, `lambda_ssr=0.005`), has no KD term, and reads decoder
rank and channel overlap. Its mask endpoints remain unresolved.

The published direct CUB geometry values are recomputed from the 30 per-seed
records by `python3 scripts/verify_paper_20260809.py`: effective-rank gain
`+1.8473e-3` (95% CI `[+2.5946e-4, +3.4352e-3]`) and mean absolute channel-
overlap reduction `+1.0482e-5` (95% CI `[+6.1240e-6, +1.4840e-5]`). They are
geometry evidence only; the same verifier confirms that mIoU, Dice and
forgetting endpoints remain unresolved.

The direct attribution protocol is a four-node launch that selects its
candidate before the confirmation cohort and writes the selection lock and
per-seed pair records. Run the same command once per node of a shared four-node
allocation, setting a distinct `GLOBAL_NODE_RANK` from 0 through 3:

```bash
SSR_DIRECT_RUN_ID=direct_ssr_segmentation_rerun \
EXPECTED_NNODES=4 GPU_COUNT=8 JOBS_PER_GPU=2 \
GLOBAL_NODE_RANK=0 \
bash scripts/submit_direct_ssr_attribution_4node_20260809.sh
```

For the direct CUB arm, the executable protocol fixes a 192-pixel,
256-dimensional prototype-cosine decoder, 24 confirmation epochs, task-only
loss, a 20% SSR warm-up/ramp, and the selected Cauchy kernel above. Node 0
produces `segmentation/CONFIRMATION_SUMMARY.json` after all nodes complete.
The artifact verifier deliberately labels these task endpoints as a boundary
audit rather than a positive SSR-only result.

## Raw-image ViT-LoRA Protocol

The raw-image protocol is retained for a fresh validated cohort but is not a
manuscript evidence source in this revision. It uses a frozen ViT-Tiny
backbone; LoRA rank 8 and alpha 16 in all `qkv` and `proj` projections; and
trains only LoRA A/B and the classifier head. Its two arms are task-only and
task-plus-SSR, with Gaussian kernel `(A_exc, A_inh)=(1.0, 0.8)`,
`(sigma_exc, sigma_inh)=(0.2, 0.5)`, `lambda_spatial=0.01`,
`lambda_adapter=0.002`, no spectral penalty, batch size 64, AdamW
(`lr=5e-4`, no weight decay), one warm-up epoch, cosine schedule, gradient
clip 5 and a 15-epoch budget. The exact executable settings are the two YAML
files in `configs/meeting_20260804/`.

The archived Pet result is invalid because AMP overflow prevented updates;
stage assets with `scripts/fetch_vit_lora_assets.sh`, rerun into a fresh
directory, and validate paired records with the command below before any new
result is interpreted:

```bash
OUTPUT_ROOT="$PWD/results/vit_lora_rerun" \
DATASETS="flowers102 oxfordiiitpet" \
SEEDS="5201 5203 5205 5207 5209" \
GPU_LIST="0 1 2 3 4 5 6 7" \
bash scripts/run_vit_lora_confirmation_8gpu.sh

python3 scripts/validate_vit_lora_pairs.py \
  --results-root "$OUTPUT_ROOT" \
  --datasets flowers102 oxfordiiitpet \
  --seeds 5201 5203 5205 5207 5209 \
  --output-dir "$OUTPUT_ROOT/validated"
```

The validator requires matched training identities, finite optimizer updates,
and distinct final-model hashes. It is not a manuscript evidence cohort until a
new run passes those checks.

## Scope of Comparisons

The repository preserves native continual-learning reference runs for methods
such as LwF, EWC, MAS and SI. Those records were generated under their native
protocols and are not presented as a matched ranking against KD+SSR. No
matched SGD, ER or DER++ matrix is claimed. The manuscript’s direct
comparisons are identified above and the complete CounterFact factorial is
included so that null and unfavorable factorial cells remain inspectable.
