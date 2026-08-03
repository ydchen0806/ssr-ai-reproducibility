# Cluster execution for the 2026-08-03 meeting revision

## Locked checkout and data

Run only from the isolated committed checkout:

```bash
cd /unify/ydchen/unidit/bioreg_meeting_20260803_cc3bac0
git rev-parse --short HEAD
test -z "$(git status --porcelain --untracked-files=no)"
```

The launchers use existing assets through the checkout links:

- KnowEdit: `dataset/knowedit/benchmark`
- GPT-2 XL: `hugging_cache/gpt2-xl`
- CIFAR-100 and TinyImageNet: `dataset`
- CUB-200 images and masks: `data/cub200`
- CUB dense feature cache: `data/cub200/cub200_seg_resnet18_dense_192.pt`

Do not launch while `nvidia-smi` reports unrelated active compute. The three workloads below are intended to run serially on the same two GPUs.

## 1. Editing attribution

This executes the locked six-recipe matrix on ZsRE, WikiCounterFact and WikiRecent. Mapping-independent controls run once; only SSR-containing recipes are repeated for cosine and projective mappings. Historical locked records are verified and reused before any new job starts.

```bash
cd /unify/ydchen/unidit/bioreg_meeting_20260803_cc3bac0
RUN_ID=meeting_editing_cc3bac0_confirm \
RESULT_ROOT="$PWD/results/meeting_editing_cc3bac0_confirm" \
REUSE_RESULTS_ROOTS="$PWD/results/meeting_20260803/imported_decisive_editing" \
GPU_LIST="0 1" \
bash scripts/run_meeting_editing_8gpu.sh
```

Expected matrix: 240 conditions, 116 verified reuses and 124 new runs. The 124 new runs are estimated at 22.45 GPU-hours from the imported run times, or about 11.2 hours on two equivalent GPUs.

## 2. Fixed-KD continual learning

This compares `KD`, `KD+EWC`, `KD+MAS`, `KD+SI` and `KD+SSR` under one frozen-teacher scaffold on Split-CIFAR-100 and Split-TinyImageNet.

```bash
cd /unify/ydchen/unidit/bioreg_meeting_20260803_cc3bac0
OUTPUT_ROOT="$PWD/results/meeting_20260803/matched_kd_cl" \
GPU_LIST="0 1" \
bash scripts/run_meeting_matched_kd_cl.sh
```

Expected matrix: 50 jobs. The launcher validates the shared scaffold and writes `fairness_report.json` after all methods complete.

## 3. CUB segmentation attribution

This runs a development-only screen, locks one parameter setting, then confirms `Task`, `KD`, `SSR` and `SSR+KD` on ten untouched seeds. The primary contrast is `SSR+KD - KD`.

```bash
cd /unify/ydchen/unidit/bioreg_meeting_20260803_cc3bac0
python3 scripts/run_meeting_segmentation_ablation.py \
  --result-root "$PWD/results/meeting_20260803/cub_segmentation" \
  --phase all \
  --gpus 0 1
```

Expected matrix: 21 development jobs and 40 confirmation jobs. The driver fingerprints `data/cub200/segmentations.tgz` and the 192-pixel feature cache before dispatch.

## 4. Aggregate and gate the evidence

Run this only after the three workloads finish:

```bash
cd /unify/ydchen/unidit/bioreg_meeting_20260803_cc3bac0
python3 scripts/aggregate_meeting_revision.py \
  --manifest configs/meeting_20260803/manifest.yaml \
  --results-root results \
  --output artifacts/meeting_20260803
python3 scripts/make_meeting_revision_figures.py \
  --tables artifacts/meeting_20260803/tables \
  --output artifacts/meeting_20260803/figures
python3 scripts/check_result_consistency.py \
  --registry artifacts/meeting_20260803/claim_registry.yaml
```

Submission mode must fail if a required pair is missing, a confidence interval is unavailable, or a manuscript claim lacks a source row. Use staged mode only for internal review; staged figures carry a visible incomplete-evidence label.
