# SSR-v23 low-rank segmentation runtime

This runtime derives from SSR-v22 commit `0379fc9724f3fd2862a2b0045b8b577237a5a68b`.

The incremental PASCAL VOC classifier uses three matched arms:

1. unmodified PLOP;
2. PLOP with a frozen historical classifier plus a trainable low-rank residual;
3. the identical low-rank parameterization with SSR applied to its effective historical weights.

The current-task classifier, DeepLabV3 head, ResNet-101 backbone, PLOP losses,
optimizer, schedules, data indices, and training epochs are unchanged.  The
low-rank residual is initialized as `B @ A` with `B = 0`, so the incremental
model initially reproduces the exact shared step-0 classifier.

Development uses seeds 12101 and 12102 and exact pre-existing step-0 checkpoint
hashes declared in `plop_lowrank_v23_development_specs.json`.  Confirmation uses
the fixed seeds 12301 through 12310.  For each confirmation seed, step 0 is
trained once and the resulting checkpoint is shared by PLOP, LowRank, and
LowRank+SSR.  Every declared confirmation seed is included in paired bootstrap
confidence intervals; no confirmation seed is selected or discarded.
