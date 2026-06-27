# Spatial Synaptic Competition Regularization (SSR): AI Reproducibility

This repository contains the code needed to reproduce the artificial-network experiments from the SSR manuscript. SSR stands for **Spatial Synaptic Competition Regularization**. It implements a Mexican-hat / difference-of-Gaussians regularizer that discourages excessive overlap among plastic directions while retaining local stability through knowledge distillation or task-specific constraints.

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

## Datasets

The standard continual-learning datasets are downloaded automatically by `torchvision` or by the repository loader:

- **Split-CIFAR-10** and **Split-CIFAR-100**: downloaded automatically into `./dataset`.
- **Five-dataset stream**: CIFAR-10, MNIST, FashionMNIST, SVHN, and a CIFAR-100 subset are downloaded automatically into `./dataset`.
- **Split-TinyImageNet**: the loader downloads `tiny-imagenet-200.zip` from the Stanford CS231n mirror into `./dataset` and converts the validation folder to `ImageFolder` format.
- **CUB-200-2011**: `experiments/cub200_continual_benchmark.py` downloads the official images and segmentation masks from Caltech data records into `data/cub200`.
- **KnowEdit-style LLM editing splits**: place the benchmark files under `dataset/knowedit/benchmark/...` using the paths shown in `scripts/run_llm_ke_easyedit.py`.

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
  --methods baseline biocs biocs_kd \
  --seeds 0 1 2 \
  --tasks classification segmentation \
  --device cuda
```

Run the frozen-feature low-rank adapter probe after creating the CUB ResNet-18 feature cache:

```bash
python experiments/lowrank_adapter_probe.py \
  --feature_cache data/cub200/cub200_resnet18_features.pt \
  --output_dir results/lowrank_adapter_probe \
  --methods baseline kd biocs biocs_kd \
  --seeds 0 1 2 \
  --device cuda
```

The internal method names `biocs` and `biocs_kd` in these legacy experiment scripts correspond to SSR-only and SSR+KD.

## Optional LLM Editing Probe

The LLM editing code is an optional probe rather than the core continual-vision benchmark. It supports vanilla fine-tuning, SSR fine-tuning, and EasyEdit baselines such as ROME, MEMIT, and AlphaEdit when EasyEdit is installed.

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
  --method BIOCS \
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

## Reproducibility Notes

1. Use the seeds reported in the manuscript or `SEEDS="42 123 456"` for the representative queue.
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

