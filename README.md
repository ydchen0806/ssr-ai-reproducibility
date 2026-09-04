# Spatial Synaptic Regularization (SSR): AI Reproducibility

This repository contains the code and compact result records for the artificial-network experiments in the SSR manuscript. SSR stands for **Spatial Synaptic Regularization**. It applies a bounded center-surround interaction to learned directions. Task losses determine what is learned, stabilizers such as knowledge distillation (KD) determine what must be retained, and SSR organizes the geometry of the plastic directions.

The repository is intentionally limited to the AI side of the study:

- continual image classification on Split-CIFAR-10, Split-CIFAR-100, Split-TinyImageNet, and the five-dataset domain stream;
- geometry and mechanism probes, including representation rank, prototype overlap, spectral diagnostics, and attention/localization visualization;
- CUB-200 fine-grained classification, segmentation, and low-rank adapter probes;
- optional sequential LLM editing probes using EasyEdit-compatible baselines and the SSR fine-tuning editor.

Biological connectomics analyses, raw connectomics tables, and manuscript source files are not redistributed here.

## Repository Layout

```text
.
|-- main.py                         # continual-learning training entry point
|-- configs/                        # benchmark and ablation YAML files
|-- datasets/                       # Split-CIFAR, TinyImageNet, five-dataset, KE loaders
|-- methods/                        # CL methods and SSR variants
|-- regularizers/                   # SSR Mexican-hat regularizer implementation
|-- models/                         # ResNet and MLP backbones
|-- utils/                          # metrics and plotting helpers
|-- experiments/                    # CUB, adapter, capacity, and auxiliary probes
|-- llm_ke/                         # SSR fine-tuning editor and EasyEdit hparams
|-- docs/lowrank_ke_reproduction.md # Figure 5 LoRA-B protocol and source map
|-- artifacts/paper_20260803/       # original compact paper audit bundle
|-- artifacts/paper_20260809/       # archived direct-SSR protocols and boundary records
|-- artifacts/paper_20260810/       # current CounterFact factorial and value map
|-- docs/manuscript_reproduction.md # exact manuscript evidence map and commands
|-- scripts/run_smoke_test.sh       # short sanity check
|-- scripts/run_reproducibility_suite.sh
|-- scripts/collect_results.py
`-- scripts/*analysis*.py           # figure and mechanism-analysis utilities
```

Some Python module names and config keys still contain the historical internal token `biocs`. These names are kept for backward compatibility with the recorded runs. In the manuscript and in this README, the method is referred to as **SSR**. In practice:

- `biocs_plus` in a config corresponds to **SSR+KD** when `lambda_distill > 0`;
- `lambda_spatial` is the SSR penalty weight;
- `A_exc`, `A_inh`, `sigma_exc`, and `sigma_inh` parameterize the difference-of-Gaussians kernel.

## Installation

Create a fresh environment with Python 3.10 or 3.11:

```bash
git clone https://github.com/ydchen0806/ssr-ai-reproducibility.git
cd ssr-ai-reproducibility

conda create -n ssr-ai python=3.11 -y
conda activate ssr-ai

pip install --upgrade pip
pip install -r requirements.txt
```

Install the PyTorch build that matches your CUDA driver if the default wheel is not appropriate for your machine. For example, follow the official PyTorch selector and then reinstall the matching `torch` and `torchvision` packages.

For the locked GPT-2 XL editing factorial, follow the exact model, dataset and
command mapping in [`docs/manuscript_reproduction.md`](docs/manuscript_reproduction.md).
The guide states the boundary of the recovered LLM environment rather than
presenting an unrecorded dependency set as a historical lock.

## Datasets

The standard continual-learning datasets are downloaded automatically by `torchvision` or by the repository loader:

- **Split-CIFAR-10** and **Split-CIFAR-100**: downloaded automatically into `./dataset`.
- **Five-dataset stream**: CIFAR-10, MNIST, FashionMNIST, SVHN, and a CIFAR-100 subset are downloaded automatically into `./dataset`.
- **Split-TinyImageNet**: the loader downloads `tiny-imagenet-200.zip` from the Stanford CS231n mirror into `./dataset` and converts the validation folder to `ImageFolder` format.
- **CUB-200-2011**: `experiments/cub200_continual_benchmark.py` downloads the official images and segmentation masks from Caltech data records into `data/cub200`.
- **Locked LLM-editing cohort**: place **ZsRE**, **WikiCounterFact**, and
  **WikiRecent** under the exact paths and verify their SHA-256 digests in
  `configs/meeting_20260803/locked_editing_gpt2xl.yaml`. The original data
  sources and the immutable model/configuration identifiers are listed in
  [the manuscript reproduction guide](docs/manuscript_reproduction.md).

Large downloaded files, model checkpoints, and generated results are ignored by Git.

## Quick Smoke Test

Run a minimal two-task, one-epoch check before launching the full jobs:

```bash
DEVICE=cuda bash scripts/run_smoke_test.sh
```

Use CPU only if needed:

```bash
DEVICE=cpu bash scripts/run_smoke_test.sh
```

The smoke test writes results to `results/smoke` and prints a small summary table from all generated `summary.json` files.

## Main Continual-Learning Reproduction

The main training entry point is:

```bash
python main.py --config <config.yaml> --seed <seed> --device cuda --output_dir results/reproduce_main
```

Each run writes:

- `train.log`: per-task training and evaluation log;
- `config.yaml`: the resolved config used for the run;
- `metrics.json`: full task-by-task accuracy matrix and derived metrics;
- `summary.json`: compact run summary used by figure and table scripts;
- optional `checkpoints/final_model.pt` when `training.save_final_model: true`.

Run the representative manuscript queue:

```bash
SEEDS="42 123 456" DEVICE=cuda bash scripts/run_reproducibility_suite.sh
```

The queue includes matched baseline and SSR+KD configurations for Split-CIFAR-10, Split-CIFAR-100, Split-TinyImageNet, and the five-dataset stream. It also writes `results/reproduce_main/summary_table.csv`.

To run a single SSR+KD Split-CIFAR-100 experiment:

```bash
python main.py \
  --config configs/biocs_plus_lkd2p0_c100.yaml \
  --seed 42 \
  --device cuda \
  --output_dir results/reproduce_main
```

To override a config from the command line:

```bash
python main.py \
  --config configs/sota_sweeps/biocs_plus_c10_sota_lkd3_t3_ls005.yaml \
  --seed 42 \
  --device cuda \
  --output_dir results/debug \
  --overrides training.epochs=5 method.lambda_spatial=0.003
```

## Metrics

The code evaluates every task after each training stage and stores a task-by-task accuracy matrix. The primary reported metrics are:

- **Average accuracy (AA)**: mean task-incremental accuracy across the final evaluation matrix, reported in percent.
- **Last accuracy**: mean accuracy over all tasks after the last task has been trained.
- **Average forgetting (AF)**: average drop from each old task's best historical accuracy to its final accuracy. Lower is better.
- **Backward transfer (BWT)**: final minus historical old-task performance. Less negative is better.
- **Wall-clock time**: total run time in seconds. This is useful for feasibility checks but should not be interpreted as a hardware-normalized benchmark unless all methods are run under the same profiler and hardware allocation.

Aggregate any result folder with:

```bash
python scripts/collect_results.py --results results/reproduce_main --csv results/reproduce_main/summary_table.csv
```

## Geometry and Mechanism Analyses

The paper's AI mechanism analyses use the saved summaries and, where available, final checkpoints:

```bash
python scripts/analyze_representation_geometry.py
python scripts/fourier_update_analysis.py
python scripts/temporal_learning_dynamics_analysis.py
python scripts/attention_map_visualization.py
```

These scripts expect the relevant `results/` folders produced by the training commands above. They generate representation-rank, prototype-overlap, Fourier-spectrum, learning-dynamics, and attention-localization visualizations.

## CUB-200 and Low-Rank Adapter Probes

Run the CUB-200 continual fine-grained benchmark:

```bash
python experiments/cub200_continual_benchmark.py \
  --data_root data/cub200 \
  --output_dir results/cub200_continual \
  --methods baseline kd biocs biocs_kd \
  --seeds 0 1 2 \
  --tasks classification segmentation \
  --device cuda
```

For the same-base CUB classification control used in the paper, run only the
KD and SSR+KD methods with the locked five-seed, 80-epoch protocol:

```bash
python experiments/cub200_continual_benchmark.py \
  --data_root data/cub200 \
  --classification_cache data/cub200/cub200_resnet18_features.pt \
  --output_dir results/cub200_kd_control \
  --methods kd biocs_kd \
  --seeds 3101 3103 3105 3107 3109 \
  --tasks classification \
  --cls_epochs 80 \
  --a-exc 1.2 --a-inh 0.9 \
  --sigma-exc 0.22 --sigma-inh 0.60 \
  --device cuda
```

Run the frozen-feature low-rank adapter probe after creating the CUB ResNet-18 feature cache:

```bash
python experiments/lowrank_adapter_probe.py \
  --feature_cache data/cub200/cub200_resnet18_features.pt \
  --output_dir results/lowrank_adapter_probe \
  --methods kd biocs_cls_kd biocs_adapter_kd biocs_kd \
  --seeds 0 1 2 \
  --epochs 20 \
  --device cuda
```

The internal method names `biocs` and `biocs_kd` in these legacy experiment scripts correspond to SSR-only and SSR+KD. For the adapter factorization, `biocs_cls_kd` applies SSR only to classifier prototypes and `biocs_adapter_kd` applies SSR only to adapter output-basis directions.

The independent rank, distance-mapping, and radial-family validation uses a
separate entry point. The example below reproduces one locked rank-16
all-cosine Gaussian condition and its matched `kd` control.

```bash
python experiments/lowrank_adapter_mapping.py \
  --feature_cache data/cub200/cub200_resnet18_features.pt \
  --output_dir results/adapter_rank16_gaussian_all_cosine \
  --methods kd biocs_kd \
  --seeds 8411 8413 8415 8417 8419 8421 8423 8425 8427 8429 \
  --rank 16 \
  --epochs 40 \
  --kernel-family gaussian \
  --a-exc 1.0 --a-inh 0.8 \
  --sigma-exc 0.55 --sigma-inh 1.25 \
  --prototype-distance-metric cosine \
  --adapter-distance-metric cosine \
  --device cuda
```

`experiments/cub200_mechanism_control.py`,
`experiments/cub200_topology_validation.py`, and
`experiments/cub200_taxonomy_topology.py` contain the matched geometric
controls and semantic-neighborhood diagnostics. They keep the feature cache,
task stream, optimizer, KD scaffold, and training budget fixed while changing
the geometry term. The frozen 200-class taxonomy used by the held-out audit is
included in the paper artifact. For example, one selected SSR validation run is:

```bash
python experiments/cub200_taxonomy_topology.py \
  --method ssr_dynamic \
  --seed 9401 \
  --strength 0.01 \
  --center-hwhm 0.10 \
  --surround-hwhm 0.34 \
  --taxonomy-file artifacts/paper_20260803/raw/mechanism/cub200_gbif_taxonomy_20260730.json \
  --taxonomy-order taxonomy_blocked \
  --output-dir results/cub200_mechanism_seed9401 \
  --device cuda
```

## Optional LLM Editing Probe

The LLM editing code is an optional probe rather than the core continual-vision benchmark. It supports vanilla fine-tuning, SSR fine-tuning, and EasyEdit baselines such as ROME, MEMIT, and AlphaEdit when EasyEdit is installed.

The current low-rank Qwen/ZsRE workflow is documented separately in
[`docs/lowrank_ke_reproduction.md`](docs/lowrank_ke_reproduction.md). It
includes the LoRA-B SSR operator, fixed edit-stream generator, matched
development selector, untouched ten-seed confirmation summarizer and compact
Figure 5 source tables.

Recommended setup:

```bash
mkdir -p external
git clone https://github.com/zjunlp/EasyEdit.git external/EasyEdit
pip install -e external/EasyEdit

export EASYEDIT_DIR="$PWD/external/EasyEdit"
export KE_HPARAMS_DIR="$PWD/llm_ke/easyedit_hparams"
export KE_MODEL_NAME_OR_PATH="gpt2-xl"
```

Run a small ZsRE-style probe:

```bash
python scripts/run_llm_ke_easyedit.py \
  --method biocs \
  --dataset zsre \
  --n_edits 50 \
  --model_name "$KE_MODEL_NAME_OR_PATH" \
  --output results/llm_ke/ssr_ft_zsre50
```

Run an EasyEdit baseline when the corresponding hparams and covariance statistics are available:

```bash
python scripts/run_llm_ke_easyedit.py \
  --method ROME \
  --dataset zsre \
  --n_edits 50 \
  --model_name "$KE_MODEL_NAME_OR_PATH" \
  --hparams_model gpt2-xl \
  --output results/llm_ke/rome_zsre50
```

For larger open-source LLMs, set `KE_MODEL_NAME_OR_PATH` to a local Hugging Face model directory and make sure the model license permits the intended use.

The LLM editor exposes the exact SSR implementation choices through
`BIOCS_KERNEL_FAMILY`, `BIOCS_TARGET` (`weight` or `delta`),
`BIOCS_DISTANCE_METRIC` (`cosine` or `projective`), and the four kernel
parameters. The `BIOCS_` prefix is a historical compatibility key; it does not
denote a second method.

### Locked GPT-2 XL manuscript cohort

The manuscript-facing editing matrix is not the small probe above. It uses the
locked GPT-2 XL configuration in
`configs/meeting_20260803/locked_editing_gpt2xl.yaml`: GPT-2 XL blocks 16--18,
Adam (`lr=1e-4`, 25 steps), active-top selection of at most 256 delta-weight
rows, and a Gaussian SSR kernel (`A_exc=1.2`, `A_inh=0.9`,
`sigma_exc=0.22`, `sigma_inh=0.60`). The configuration also locks the three
source files, model fingerprint, 100-edit contiguous confirmation slice,
paired seeds, anchor and spectral coefficients, and evaluator identity. The
same YAML explicitly locks the `(c_proj|down_proj)$` target-module selector,
direct weight editing (no LoRA insertion), optimizer betas/epsilon/weight
decay/clip, sequence limits and runtime policy. Each run manifest records both
those requested values and the modules, dtype and device resolved at runtime.

Preflight the local model and datasets before starting a new cohort:

```bash
MODEL_NAME=/absolute/path/to/gpt2-xl

python3 scripts/verify_locked_editing_assets.py \
  --config configs/meeting_20260803/locked_editing_gpt2xl.yaml \
  --model-path "$MODEL_NAME"
```

Then run the locked confirmation matrix on one eight-GPU node. It contains all
six recipes; the two SSR-containing recipes are evaluated under both cosine
and projective mappings, while mapping-independent controls are recorded once
with mapping `none` (240 paired cells in total):

```bash
GPU_LIST="0 1 2 3 4 5 6 7"
MODEL_NAME=/absolute/path/to/gpt2-xl
RUN_ID=locked_gpt2xl_confirm
RESULT_ROOT="$PWD/results/locked_gpt2xl_confirm"

GPU_LIST="$GPU_LIST" MODEL_NAME="$MODEL_NAME" RUN_ID="$RUN_ID" \
RESULT_ROOT="$RESULT_ROOT" \
bash scripts/run_meeting_editing_8gpu.sh --phase confirm
```

`artifacts/paper_20260810/counterfact_factorial_summary.json` is the compact
ten-seed result record, and `python3 scripts/verify_paper_20260810.py`
recomputes every value selected for the manuscript. See
[the manuscript reproduction guide](docs/manuscript_reproduction.md) for the
full recipe map, formal dataset names, expected paths, and runtime boundary.

## Paper Artifact Bundle

`artifacts/paper_20260803` contains compact per-seed records for the current
continual-classification, CUB adapter, CUB segmentation, and nested LLM-editing
audits. It also contains the held-out CUB mechanism report and the recorded
runtime subset from the source environment. The original private
workspace commit, full LLM dependency lock, model revision, EasyEdit revision,
and operating-system build were not retained, so this is an audit bundle rather
than a bit-for-bit environment snapshot. Verify file integrity,
seed counts, configurations, manuscript-facing aggregates, and paired
confidence intervals with:

```bash
python scripts/verify_paper_artifacts.py
```

The manifest distinguishes direct SSR attribution (`KD` versus `SSR+KD`) from
complete-objective transfer. The segmentation record compares a base objective
with the complete SSR+KD objective and is retained as an early boundary audit,
not as a positive single-component ablation. The current CounterFact values are
versioned separately under `artifacts/paper_20260810`; use that bundle rather
than the recovered 3 August LLM record for manuscript-facing editing results.
The segmentation outputs combine seeds 0, 1, and 2 from an append-only run
stream; the exact seed-0 launch manifest is not present in the recovered
archive, and this limitation is recorded in `raw/segmentation/protocol.json`.

Absolute cluster paths in the recovered records are normalized to
`source_workspace/...`; content digests of the original source files are kept
where available. See `artifacts/paper_20260803/environment.json` for the exact
boundary of the captured environment metadata.

The historical `inverse` key is implementation-specific: the classification
entry point uses `1 / (1 + d / sigma)`, whereas the adapter and LLM entry
points use `sigma / sqrt(d^2 + sigma^2)`. The selected ZsRE inverse run used a
scale broader than a true HWHM-matched control. Its measurements are retained,
but the artifact verifier explicitly prevents treating it as evidence of
HWHM-matched kernel invariance.

### Direct-SSR protocol update

`artifacts/paper_20260809` retains direct-SSR protocols and their boundary
audits. The archived raw-image ViT-LoRA Pet numbers are intentionally excluded
from manuscript evidence: a later audit found mixed-precision gradient overflow
that prevented parameter updates in both arms. The current implementation
evaluates SSR similarities outside autocast, skips the regularizer when its
coefficient is zero, and rejects a paired run unless every optimizer update is
finite and the final model hashes differ. Rerun the protocol from scratch after
installing this revision; do not reuse the archived Pet results. The same bundle
retains the direct segmentation audit, whose geometry readouts are reproducible
while its task endpoints are unresolved.

Recompute every manuscript-facing aggregate from the per-seed rows with:

```bash
python3 scripts/verify_paper_20260809.py
```

The following four-node launcher is retained for rerunning the direct
attribution protocol after the validation checks; it must use a fresh result
root:

```bash
SSR_DIRECT_RUN_ID=ssr_direct_attribution_4node_20260809_r1 \
EXPECTED_NNODES=4 GPU_COUNT=8 JOBS_PER_GPU=2 \
bash scripts/submit_direct_ssr_attribution_4node_20260809.sh
```

Run that command once on every node of the shared allocation. The launcher
assigns three nodes to the independent segmentation audit and one node to the
Pet ViT-LoRA development-and-confirmation protocol, records the repository
commit, and fails if the expected result summaries are incomplete or a paired
ViT-LoRA run does not demonstrate real optimizer updates.

### Current CounterFact direct and nested comparisons

`artifacts/paper_20260810` contains the complete ten-seed CounterFact factorial
used by the current manuscript. It retains all recipes, both cosine and
projective mappings, and every evaluated endpoint. A small machine-readable
index maps the displayed direct, nested and component-control comparisons to
their exact JSON paths; the matched task endpoint is checked at the same time.

Verify the manuscript-facing values directly from the complete factorial with:

```bash
python3 scripts/verify_paper_20260810.py
```

`artifacts/meeting_20260803` is retained only as a historical meeting workspace.
Its partial manifests and staged checklists are not a current evidence index.

## Manuscript Evidence Map

The exact provenance, datasets, parameterization, commands, verifier entry
points, and known boundaries for every AI result used in the manuscript are in
[`docs/manuscript_reproduction.md`](docs/manuscript_reproduction.md). In
particular, the historical cluster capture fixed Python/PyTorch/CUDA and core
scientific packages but did not retain a bit-for-bit LLM dependency lock or
EasyEdit revision. This limitation is documented rather than hidden; model and
dataset hashes, protocol hashes, and the full locked YAML remain available for
the GPT-2 XL factorial.

## Reproducibility Notes

1. Use the seeds recorded in `artifacts/paper_20260803/manifest.json` for exact paper comparisons; `SEEDS="42 123 456"` is only the representative quick queue.
2. Keep the same task split and class order unless explicitly testing robustness.
3. Run each baseline and SSR variant with comparable hardware, batch size, epoch budget, and tuning budget.
4. Do not compare cached wall-clock times from different machines as an efficiency claim.
5. Regenerate figures from `summary.json`, `metrics.json`, and checkpoints rather than editing figures manually.

## Troubleshooting

- If CUDA is unavailable, `main.py` falls back to CPU when `--device cuda` is requested. CPU smoke tests work, but full benchmarks are slow.
- TinyImageNet download can fail if the CS231n mirror is unavailable. In that case, manually place `tiny-imagenet-200.zip` or the extracted `tiny-imagenet-200/` folder under `./dataset`.
- EasyEdit can be sensitive to `transformers` versions. Use the version recommended by the EasyEdit repository if ROME/MEMIT/AlphaEdit imports fail.
- If local module `datasets/` shadows Hugging Face `datasets`, use `scripts/run_llm_ke_easyedit.py`, which temporarily isolates EasyEdit imports.

## Citation

If you use this code, please cite the SSR manuscript once it is available. Until then, cite the repository URL and include the commit hash used for experiments.
