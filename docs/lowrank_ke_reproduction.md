# Low-rank SSR knowledge-editing reproduction

This workflow reproduces the low-rank knowledge-editing analysis used in
Figure 5. It compares EasyEdit LoRA with the same LoRA update followed by SSR
on the output rows of each LoRA-B matrix. The model, edit stream, LoRA rank,
learning rate, optimizer budget and evaluator are identical within every pair.

## What Figure 5d and 5e test

**Figure 5d is a performance replication panel.** It reports the paired change
in immediate efficacy for three independent rank-8 cohorts: a 100-edit cohort,
a 250-edit rank cohort and a separate 250-edit mechanism cohort. Each cohort
contains ten paired edit-order seeds. The panel asks whether the acquisition
gain recurs under new edit orders and a longer horizon; it is not a locality or
mechanism result.

**Figure 5e is a mechanism panel.** On the independent 250-edit mechanism
cohort, it measures the exact plastic object, LoRA-B. Center-surround contrast
is the mean cosine similarity of adjacent rows minus that of two-step rows.
Kernel alignment is the normalized agreement of those row-cosine profiles with
the fixed SSR target `(+1, -0.2)`. The panel therefore tests whether SSR forms
the intended bounded geometry by 100 edits and retains it at 250 edits. It does
not, by itself, establish better efficacy or locality; those endpoints are
reported separately.

## Dependencies

Use Python 3.11, a CUDA-compatible PyTorch build and EasyEdit. The completed
campaign used EasyEdit commit `14cea8245f06715684592ab55184939b99d70784`.
Only its standard LoRA path was used. Place a Qwen2.5-7B-Instruct snapshot and
the KnowEdit ZsRE JSON on local storage. Copy
`configs/lowrank_ke/easyedit_qwen25_lora.yaml` to
`$EASYEDIT_DIR/hparams/LoRA/qwen2.5-7b.yaml` if the checkout does not already
provide an equivalent file.

## One-node campaign

The wrapper first creates fixed non-overlapping edit/control manifests, runs a
matched four-seed development matrix, freezes one eligible rank/rate/SSR tuple,
and only then evaluates it on ten untouched edit-order seeds. A positive claim
requires both primary paired confidence intervals to be above zero.

```bash
export MODEL_PATH=/path/to/Qwen2.5-7B-Instruct
export DATASET_PATH=/path/to/ZsRE-test-all.json
export EASYEDIT_DIR=/path/to/EasyEdit
export HPARAMS_STEM=qwen2.5-7b
export RUN_ID=lowrank_ssr_qwen_zsre_r1
export GPU_COUNT=8

bash scripts/run_lowrank_ke_campaign_8gpu.sh
```

Run the structural preflight without model execution first:

```bash
DRY_RUN=1 bash scripts/run_lowrank_ke_campaign_8gpu.sh
```

The complete protocol and gates are also recorded in
`configs/lowrank_ke/qwen25_zsre_rank_lr.yaml`. The final files are
`results/${RUN_ID}_lowrank_lora_ssr/development_selection.json`,
`confirmation_summary.json` and `confirmation_curves.csv`.

To confirm a second tuple already frozen from the same development matrix,
use `scripts/run_lowrank_ke_frozen_confirmation_8gpu.sh`. This launcher does
not run a new search: it requires the immutable development-selection record,
exactly ten new stream seeds, and the frozen rank, learning rate, step count
and SSR coefficient. It retains every paired outcome and applies the same
two-primary-endpoint confidence-interval gate.

Verify the compact manuscript-facing exports with:

```bash
python scripts/verify_figure5_lowrank_ke.py
```

## Metric interpretation

- `immediate_efficacy`: fraction of applied edits whose new target appears in
  the greedy completion; higher is better.
- `pre_edit_output_consistency`: one minus normalized Jensen-Shannon divergence
  between pre- and post-edit next-token distributions on fixed, disjoint
  controls; higher is better. This is the primary locality endpoint.
- `locality_target_consistency`: target matching on supplied KnowEdit locality
  prompts. It remains a secondary diagnostic and is never relabelled as output
  preservation.
- `history_efficacy`: re-evaluation of earlier edits at fixed checkpoints.
- `center_surround_contrast` and `kernel_alignment`: read-only geometry
  measurements on LoRA-B.

## Next experiments

The next high-value test is not a larger unconstrained sweep. It is a matched
strong-scaffold factorial: `(LoRA + output-preservation anchor)` versus
`(LoRA + the same anchor + SSR)`, with the same development/freeze/ten-seed
confirmation rule. This directly tests whether SSR contributes beyond a
locality-preserving editor. A second test transfers the frozen tuple to a new
dataset or model without retuning. These experiments should be reported only
after the untouched confirmation gate passes.

The optional anchor is already implemented in the runner. A focused
development invocation is:

```bash
export RUN_ID=lowrank_ssr_qwen_zsre_anchor_r1
export DEVELOPMENT_SEEDS_TEXT="17101 17102 17103 17104"
export CONFIRMATION_SEEDS_TEXT="17131 17132 17133 17134 17135 17136 17137 17138 17139 17140"
export ADAPTER_RANKS_TEXT="8 40"
export LORA_LEARNING_RATES_TEXT="0.0025 0.0035"
export SSR_LAMBDAS_TEXT="0.024 0.048 0.064"
export ANCHOR_WEIGHT=0.1 ANCHOR_STEPS=2 ANCHOR_LR=0.0001 ANCHOR_BATCH_SIZE=4

bash scripts/run_lowrank_ke_campaign_8gpu.sh
```

This is a development hypothesis, not a manuscript result. Its complete
design is recorded in
`configs/lowrank_ke/qwen25_zsre_strong_scaffold_development.yaml`.
