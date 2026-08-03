# Cluster execution for the 2026-08-03 meeting revision

## Checkout and preflight

Use only the isolated checkout. Formal launchers reject modified source/config
files while allowing the declared dataset/model-cache links.

```bash
cd /unify/ydchen/unidit/bioreg_meeting_20260803_final
git rev-parse --short HEAD
python3 scripts/check_experiment_worktree.py --project-root "$PWD"
nvidia-smi
```

Do not launch while unrelated compute is active. The three workloads below are
intended to run serially on the currently available two-GPU host.

## 1. Editing attribution

This runs the six required recipes over ZsRE, WikiCounterFact and WikiRecent.
Mapping-independent controls run once; SSR-containing recipes run with cosine
and projective mappings. Before dispatch, each node hashes GPT-2 XL and all
three datasets once. Historical records are reused only after exact protocol
validation.

```bash
cd /unify/ydchen/unidit/bioreg_meeting_20260803_final
RUN_ID=meeting_editing_20260803_confirm \
RESULT_ROOT="$PWD/results/meeting_editing_20260803_confirm" \
REUSE_RESULTS_ROOTS="$PWD/results/meeting_20260803/imported_decisive_editing" \
GPU_LIST="0 1" \
bash scripts/run_meeting_editing_8gpu.sh
```

Expected: 240 cells, 116 verified reuses, 124 new runs. Imported run times imply
about 22.45 GPU-hours, or approximately 11.2 hours on two equivalent GPUs.

## 2. Fixed-KD continual learning

This first completes all KD-only teacher trajectories, then crosses a hard
barrier and launches the four treatments. Each treatment consumes the same
hashed teacher checkpoints for its dataset and seed.

```bash
cd /unify/ydchen/unidit/bioreg_meeting_20260803_final
OUTPUT_ROOT="$PWD/results/meeting_20260803/matched_kd_cl" \
GPU_LIST="0 1" \
bash scripts/run_meeting_matched_kd_cl.sh
```

Expected: 50 jobs. This launcher is single-node/multi-GPU; do not invoke it
concurrently from multiple nodes against one `OUTPUT_ROOT`.

## 3. CUB segmentation attribution

The driver runs 21 development conditions, writes an immutable selection lock,
then confirms `Task`, `KD`, `SSR` and `SSR+KD` on ten untouched seeds.

```bash
cd /unify/ydchen/unidit/bioreg_meeting_20260803_final
python3 scripts/run_meeting_segmentation_ablation.py \
  --result-root "$PWD/results/meeting_20260803/cub_segmentation" \
  --phase all \
  --gpus 0 1
```

Expected: 21 screen jobs and 40 confirmation jobs. The primary attribution is
`SSR+KD - KD`; `SSR - Task` is the independent secondary attribution.

## 4. Aggregate and gate

```bash
cd /unify/ydchen/unidit/bioreg_meeting_20260803_final
python3 scripts/build_result_manifest.py \
  --root results \
  --output artifacts/meeting_20260803/result_manifest.yaml
python3 scripts/aggregate_meeting_revision.py \
  --manifest configs/meeting_20260803/manifest.yaml \
  --results-root results \
  --output artifacts/meeting_20260803
python3 scripts/check_matched_kd_fairness.py \
  --results-root results/meeting_20260803/matched_kd_cl \
  --output results/meeting_20260803/matched_kd_cl/fairness_report.json \
  --expected-datasets split_cifar100 split_tiny_imagenet \
  --expected-seeds 3101 3103 3105 3107 3109
python3 scripts/make_meeting_revision_figures.py \
  --tables artifacts/meeting_20260803/tables \
  --output artifacts/meeting_20260803/figures
python3 scripts/check_result_consistency.py \
  --registry artifacts/meeting_20260803/claim_registry.yaml
```

Submission mode must fail on any missing pair, unresolved required provenance or
manuscript number without an aggregate source row. `--allow-incomplete`/`--staged`
is for internal review only and visibly labels the figures.
