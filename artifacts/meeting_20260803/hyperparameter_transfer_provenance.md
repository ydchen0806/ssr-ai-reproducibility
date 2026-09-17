# Hyperparameter-transfer provenance

## LLM sequential editing

- Source screen: ZsRE, WikiCounterFact and WikiRecent jointly.
- Development seeds: 9301 and 9303; offset 300; 50 edits.
- Selection rule: maximize the weaker final historical-efficacy/locality gain across all three development cohorts subject to the prespecified immediate and retention guardrails.
- Locked setting: Gaussian kernel, projective distance, `lambda_ssr=0.003`, `sigma_exc=0.22`, `sigma_inh=0.60`, target layers 16--18, delta-weight target and active-row selection.
- Confirmation seeds: 9311--9329 (odd); offset 500; 100 edits. These seeds were not used in selection.
- Provenance: `provenance/editing_lock/SELECTION_LOCK.json`, its source fingerprint, and a representative raw result are archived beside this report.
- Interpretation: this is a joint three-dataset editing screen. It is not a CUB-to-LLM parameter transfer and must not be described that way.

## CUB mask segmentation

- Development seeds: 9101, 9103 and 9105.
- Screen: six prespecified combinations of `lambda_ssr` and the two Gaussian ranges.
- Selection rule: prefer positive mean mIoU, Dice and IoU-forgetting changes for `SSR+KD - KD`, then maximize the weakest change, triple wins and the sum.
- Confirmation seeds: 9201--9219 (odd), hidden until a `SELECTION_LOCK.json` is written.
- Status: no completed development result exists yet. No segmentation parameter may be called locked until the screen finishes.

## Vision classification

- The matched-KD configs currently use `sigma_exc=0.16` and `sigma_inh=0.45` for Split-CIFAR-100 and Split-TinyImageNet.
- The repository does not yet contain an auditable selection artifact proving whether this pair was selected on CUB, CIFAR-100 or another development cohort.
- Until that artifact is recovered, the manuscript may report the exact values but must not claim a directional transfer such as "selected on CUB and transferred without retuning."

## Required manuscript rule

Every transfer statement must name the source dataset, selection seeds, locked configuration, target dataset, evaluation seeds and whether retuning occurred. Table numbering alone is not provenance because Supplementary tables may be reordered.
