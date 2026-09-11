# KE Weight Persistence Audit

Audit date: September 11, 2026. This audit did not retrain, delete or change
any experiment. It distinguishes an evaluated checkpoint from a saved model.

## Current manuscript cohorts

| Cohort | Completed result records | Exported trained-weight files found |
| --- | ---: | ---: |
| WikiRecent rank-8 100-edit trajectory, ten paired orders | 20 | 0 |
| WikiRecent/WikiCounterFact ranks 8/16/32/64, ten paired orders per rank/dataset | 160 | 0 |
| Dense KE geometry, CounterFact/ZsRE, four paired seeds per dataset | 16 | 0 |

The numerical results, evaluation trajectories, stream identities, and geometry
readouts remain present. Their presence is separate from the existence of a
reusable trained model file. An inventory with result-file hashes is in
[`ke_persistence_audit.json`](../artifacts/manuscript_20260911/ke_persistence_audit.json).

## What the historical code saves

In the runner at commit `d85291a`, the `checkpoints` list stores dictionaries
containing `after_edits`, immediate metrics, history metrics, pre-edit output
consistency, and geometry summaries. The completed run is written through
`atomic_json(args.output, summary)`. The runner and its native EasyEdit LoRA
implementation have no `torch.save`, `save_pretrained`, or equivalent adapter
export call. The tensors remain in the process, while the reported summaries
are written to disk. Thus the evidence supports **weights not exported by these
runners**, not a conclusion that the recorded experiments were lost.

The release runner now supports `--checkpoint-dir` and verifies LoRA A/B
tensors when saving. This change cannot retroactively create weights for an
already exited historical process.

## Expanded search

The read-only inventory covered the historical/current bioreg project trees,
SSR worktrees, three EasyEdit source trees, launcher/task directories, SSR
temporary directories, and editor caches. It found 4,138 weight/adapter-format
files overall, most belonging to vision experiments or downloaded base models.
An additional local SSR artifact search found the CUB visualization pair but no
KE adapter exports.

The KE-related cached tensors were null-space projections, a rank basis
derived from second moments, and Wikipedia/WikiText moment statistics. These
are initialization/regularization inputs, not models after a sequence of edits,
and cannot be relabeled as trained KE checkpoints. Base-model cache files were
also excluded from the trained-checkpoint release.

## Reconstructing trained KE weights

Use the original dataset/model fingerprints, stream manifests, rank/scaling,
optimizer settings, and paired seeds, adding only the new export option.
Label such files as **reconstructed reruns**, preserve the historical metrics,
and compare newly evaluated endpoints with the archived records before linking
them to manuscript figures. This audit does not claim an exact numerical replay
before that comparison has been completed.
