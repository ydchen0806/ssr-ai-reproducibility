# Meeting execution status

## What is now executable

- LLM editing has explicit `plain`, `anchor`, `spectral`, `stabilized`,
  `ssr_only`, `ssr_anchor`, `ssr_spectral` and `full` recipes. Cosine and
  projective mappings are first-class recorded conditions, and row selection is
  deterministic for a recorded seed.
- The three KnowEdit files and the exact GPT-2 XL files are checked against
  locked SHA-256 values before a formal launch. The remote asset audit passed.
- The historical and current substring-locality evaluators were proved
  equivalent over all 3,406 available KnowEdit rows (14,761 locality items).
  Only after this zero-disagreement audit were 116 historical runs normalized
  into the current evaluator identity.
- Matched continual learning now creates a KD-only teacher trajectory first.
  `KD+EWC`, `KD+MAS`, `KD+SI` and `KD+SSR` load the same per-task checkpoints;
  manifests and checkpoint bytes are hashed and checked before reuse.
- CUB segmentation implements the `Task`, `KD`, `SSR` and `SSR+KD` 2x2. The
  development selection lock binds the commit, mask archive, dense cache and
  every screen result record; confirmation records carry that lock hash.
- Every official record contains its objective, mapping, seed, data/config
  identity and provenance. Editing additionally records evaluator/model/pairing
  hashes; matched-KD records the common teacher trajectory; segmentation
  records the selection lock.
- Aggregation performs strict paired joins, rejects conflicting copies, requires
  the planned metric contract and reports all missing cells. Figures generated
  before completion are visibly marked `staged_incomplete`.

## Current evidence matrix

| Cohort | Confirmatory cells | Complete | Remaining |
|---|---:|---:|---:|
| GPT-2 XL editing | 240 | 116 | 124 |
| Matched-KD classification | 50 | 0 | 50 |
| CUB segmentation confirmation | 40 | 0 | 40 |

Segmentation also requires 21 development runs to select and lock one setting;
those runs never enter the confirmatory confidence intervals. The machine-readable
inventory is `result_manifest_partial.yaml`.

## Evidence already recovered

- The 116 editing records are complete per-edit records, not reconstructed
  means. The partial audit contains 36 paired summaries and flags 39 incomplete
  pair sets.
- One nested editing endpoint is already resolved: on WikiCounterFact, `full -
  stabilized` improves final historical locality by `+3.165 pp` (paired 95% CI
  `[+1.170, +5.160]`, 9/10 favorable seeds). WikiRecent immediate locality is
  `+1.184 pp` (95% CI `[+0.045, +2.323]`) but has only 8/10 pairs. Other direct
  SSR editing contrasts are not yet statistically resolved, so they are not a
  final manuscript claim.
- The independent ten-seed CUB adapter audit contains 240 records. Both locked
  Gaussian mappings improve accuracy and forgetting relative to same-rank KD at
  ranks 8, 16 and 32. All-cosine is stronger than object-aware at ranks 8 and
  16; the direct difference is unresolved at rank 32. See
  `adapter_evidence_audit.md` for every condition, interval and dual-win count.

## Validation and deployment

- Full local test suite: `91 passed`; the committed revision is also validated
  in the isolated cluster checkout before launch.
- Shell syntax, Python bytecode compilation, manifest validation, result
  aggregation and staged Figure 4/5 generation pass.
- Isolated remote checkout:
  `/unify/ydchen/unidit/bioreg_meeting_20260803_final`.
- Existing data/model trees are linked read-only into that checkout. No new P0
  data download is necessary.
- At the latest check both remote A800 GPUs were at 100% utilization by existing
  work. No new run was launched and no existing process was interrupted.

## Completion gate

Do not freeze manuscript numbers until all 124 editing runs, 50 matched-KD runs,
21 segmentation screen runs and 40 segmentation confirmation runs finish; the
matrix inventory has zero missing cells; the teacher fairness and selection-lock
checks pass; and submission-mode figures/claim consistency pass without the
incomplete override.
