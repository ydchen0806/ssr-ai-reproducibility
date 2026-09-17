# Four-node attribution and breadth protocol (2026-08-04)

This launch is a confirmation matrix, not an exploratory screen for favorable
cells. Every reported contrast is paired by seed and holds the task loss,
model, data order, initialization, and training budget fixed.

## Node roles

| Node rank | Workload | Primary comparison | Jobs |
|---|---|---|---:|
| 0--1 | Sequential knowledge editing on ZsRE, CounterFact, Recent and WikiBio | task loss vs task loss + SSR | 80 |
| 2 | Raw-image ViT-LoRA on Flowers-102 and Oxford-IIIT Pet | task loss vs task loss + SSR | 20 |
| 2 | Matched continual-learning controls on Split CIFAR-100 and Split Tiny-ImageNet | KD vs KD + EWC/MAS/SI/center/prototype-decorrelation/spectral/SSR | 80 |
| 3 | CUB-200 mask segmentation | task loss vs task loss + SSR; KD vs KD + SSR | 61 |
| 3 | Oxford-IIIT Pet and Flowers-102 segmentation | task loss vs task loss + SSR | 76 |

The complete matrix contains 317 jobs. Node-local workloads run serially while
each workload uses all eight GPUs as independent workers. The final barrier
does not report success until the expected inventory, result records, paired
summaries, and matched-KD fairness audit all pass.

## Interpretation boundaries

- The direct task-loss comparisons isolate the contribution of SSR. They do
  not contain KD, anchor, spectral regularization, replay, or a second teacher.
- The matched-KD experiment evaluates whether SSR adds value under the same
  distillation protocol and compares it with established continual-learning
  regularizers. It is not interchangeable with the direct comparison.
- The result bundle reports two paired analyses: every additive regularizer
  versus KD, and KD+SSR directly versus each competing regularizer on the same
  seeds. All eight prespecified endpoints must be present; missing endpoints
  cannot disappear through metric intersection.
- Kernel settings are locked before confirmation. Confirmation seeds must not
  be used to select a kernel or tune a coefficient.
- This run is a locked, chosen-setting comparison. It supports a claim of
  superiority over another regularizer only when the direct paired interval is
  resolved in SSR's favor. It is not, by itself, an equal-budget
  hyperparameter-search claim of universal SOTA.
- A claim enters the manuscript only after paired confirmation statistics are
  available. The launcher does not select or suppress unfavorable datasets.

## Output contract

The four-node launcher writes a single result root with one subdirectory per
workload, `MATRIX_PLAN_VALIDATION.json` before real execution (dry run), and
`MATRIX_RESULTS_VALIDATION.json` after real execution. The latter is produced
only when all 317 result cells and their required summaries are present.
Matched-KD records also expose the active regularizer and weight, common KD
weight and temperature, training-batch and successful-update counts, initial
model hash, auxiliary-state hash, and trainable-parameter overhead.
