# SSR-v22 derived PLOP runtime

This directory is an isolated derivative of the PLOP SSR cross-task runtime.
`SSR_CROSS_TASK_BASE_COMMIT` records the immutable parent commit.

SSR-v22 changes only the SSR regularizer and its audit fields:

- finite-annulus projective or cosine topology on decoder classifier channels;
- optional `historical_only` targeting, which does not constrain new channels;
- a linear warm-up and a batch-level non-conflicting-gradient gate;
- explicit export of every SSR setting in the per-step metrics.

The PLOP objective, ResNet-101 backbone, optimizer, epochs, data protocol, and
evaluation metrics remain unchanged between paired PLOP and PLOP+SSR arms.
