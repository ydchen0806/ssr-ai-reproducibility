# AI manuscript revision instructions

This file governs the AI sections only. Biological Results, Methods and figures are out of scope.

## Evidence hierarchy

1. Use direct attribution first: `SSR-only - Task`, `Full - Stabilized`, `SSR+KD - KD` and `KD+SSR - KD`.
2. Label `Full - Task` as recipe transfer, not the isolated effect of SSR.
3. Keep native LwF/EWC/MAS/SI as standard-method context. Use the fixed-KD table for regularizer attribution.
4. Do not call a result significant when its paired 95% interval includes zero.

## Current locked editing evidence

The imported confirmation cohort contains 116 of 120 locked projective runs. It supports two positive but task-specific observations:

- WikiCounterFact, `Full - Stabilized`, final historical locality: `+3.165 pp`, 95% CI `[+1.170,+5.160]`, 9/10 favorable seeds, with final historical efficacy statistically unresolved.
- WikiRecent, `Full - Stabilized`, immediate locality: `+1.184 pp`, 95% CI `[+0.045,+2.323]`, 6/8 favorable complete pairs. Two seeds are still missing, so this is not manuscript-final.

SSR-only comparisons and the corresponding ZsRE nested comparison are not statistically resolved in the current projective cohort. They must not be summarized as universal direct LLM improvement. Source: `tables/editing_ablation.csv`.

## Abstract and Results

- Keep the broad bridge from connectome-derived center-surround organization to artificial memory stabilization.
- Describe vision classification as the mechanism bridge, CUB mask prediction as structured-output validation, matched continual-learning methods as breadth, the adapter result as direct cross-architecture evidence, and LLM editing as a recipe/mapping transfer test.
- Strengthen the LLM sentence only if the completed `Full - Stabilized` or `SSR-only - Task` paired intervals support it. Otherwise write that the full recipe transfers and that the isolated SSR increment is task dependent.
- Do not introduce segmentation in the main text as an SSR-specific gain until `SSR+KD - KD` is complete under the old-class KD protocol.

## Main figures

- Figure 4: retain the biological-to-representation mapping, CUB classification mechanism and broader vision results. Add segmentation endpoints and paired gains only from `tables/cub_segmentation.csv` after the 2x2 gate passes.
- Figure 5: define the six editing recipes visually; show `SSR-only - Task` and `Full - Stabilized` before `Full - Task`; separate cosine/projective results; retain adapter rank-by-mapping evidence as the strongest direct non-CV attribution.
- Do not use layout, axis truncation or selected seeds to hide intervals crossing zero.

## Tables and Supplement

- Table S3a: `KD`, `KD+EWC`, `KD+MAS`, `KD+SI`, `KD+SSR` from one frozen-teacher scaffold.
- Table S3b: native LwF/EWC/MAS/SI and optional replay methods as contextual baselines.
- Order AI Supplementary results from global to specific: experimental overview; vision classification; structured-output segmentation; fixed-KD regularizer comparison; adapter; editing recipes; distance mapping; kernel sensitivity; computational cost.
- Add a one-paragraph Supplementary overview naming every experiment category and its Figure/Table range after numbering is frozen.

## Conditional claim registry

Do not populate numerical manuscript claims until all confirmation pairs exist. At freeze time, every number in text, caption and table must point to exactly one row in:

- `tables/editing_ablation.csv`
- `tables/editing_mapping.csv`
- `tables/kd_matched_cl.csv`
- `tables/cub_segmentation.csv`

The submission check must run without `--allow-empty`; an empty claim registry is a failure, not a pass.
