# WikiBio direct SSR attribution

## Question

Does adding SSR to the same sequential-editing task loss improve retention on
an external KnowEdit task? The confirmatory pair is exactly:

- `plain`: task loss
- `ssr_only`: task loss + SSR

Anchor and spectral losses are disabled in both conditions. The two conditions
share GPT-2 XL, editable layers 16--18, Adam, learning rate `1e-4`, 25 steps per
edit, source order, 200 edits, seed, context length, decoding budget, and the
versioned substring/locality evaluator.

## Dataset and length lock

The official `zjunlp/KnowEdit` WikiBio test file is downloaded from Hugging
Face and verified as SHA256
`00933f69b47d8281d01f1f4b84f3f266b7482a65a7d74022e660a8a9ca2264a4`.
Its `text`, `labels`, and `concept` fields are normalized to `prompt`,
`target_new`, and `subject`. On the locked GPT-2 XL tokenizer, the longest
prompt-plus-target example is 187 tokens, so `max_length=256` retains every
test target. The longest target is 80 tokens, so deterministic generation is
locked to `max_new_tokens=96`.

## Run

```bash
cd /unify/ydchen/unidit/bioreg_meeting_20260803_final
RUN_ID=wikibio_direct_20260804_r1 \
GPU_LIST="0 1 2 3 4 5 6 7" \
PYTHON=/usr/local/bin/python \
bash scripts/run_wikibio_direct_editing_8gpu.sh
```

The launcher downloads and hash-checks the two small WikiBio files when they
are absent. It creates 20 cells: two recipes by ten paired seeds. Eight cells
run concurrently and the remaining cells run serially on the same node.

## Aggregate

```bash
/usr/local/bin/python scripts/aggregate_meeting_revision.py \
  --manifest configs/meeting_20260804/direct_editing_manifest.yaml \
  --results-root results/wikibio_direct_20260804_r1 \
  --output artifacts/wikibio_direct_20260804_r1
```

The primary memory endpoint is final historical efficacy after 200 edits.
Immediate efficacy and locality are paired guardrails. Results should enter the
manuscript only after all ten seed pairs are complete; a positive mean alone is
not a significance claim.
