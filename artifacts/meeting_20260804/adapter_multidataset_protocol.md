# Multi-dataset low-rank adapter attribution protocol

## Question

Does SSR improve sequential low-rank adaptation when the only objective change
is adding SSR to the task loss?

The decisive comparison is therefore:

```text
task cross-entropy  versus  task cross-entropy + SSR
```

There is no KD, anchor loss, spectral penalty, replay, or frozen-teacher term in
either arm. Both arms train the same low-rank residual adapter and classifier
from byte-identical initial weights, with the same class order, minibatch order,
optimizer, learning rate, epoch count, gradient clipping and number of optimizer
steps. Each result record stores hashes of the initial state, feature cache and
training schedule and refuses an unmatched pair.

## Dataset breadth

Four new torchvision datasets occupy separate nodes:

| Node | Dataset | Domain | Classes | Sequential tasks |
|---:|---|---|---:|---:|
| 0 | Flowers-102 | fine-grained flowers | 102 | 10 |
| 1 | Food-101 | fine-grained food | 101 | 10 |
| 2 | Oxford-IIIT Pet | fine-grained animal breeds | 37 | 7 |
| 3 | DTD | material textures | 47 | 8 |

Images are deterministically embedded once with frozen torchvision
ResNet-18/ImageNet-1K-v1. The trainable module is a rank-16 residual bottleneck
adapter plus a normalized classifier. This isolates the low-rank learning
geometry while keeping image decoding and backbone compute outside the paired
comparison.

## Screen, lock, confirm

1. On development seeds 3101, 3103 and 3105, screen SSR scales 0.03, 0.1, 0.3
   and 1.0 with all other SSR parameters fixed.
2. Select one global scale across all four datasets. The prespecified rule first
   maximizes the number of datasets with positive mean accuracy change and
   forgetting reduction, then seed-level dual wins, then the combined endpoint
   change, with the smaller scale as the final tie-breaker.
3. Freeze the selected scale in `selection_lock.json`.
4. Evaluate only the locked scale on untouched confirmation seeds 4101, 4103,
   4105, 4107 and 4109.

The paper-facing result must use the confirmation cohort. Development results
are hyperparameter-selection evidence and must not be pooled with confirmation.

## Download and storage

The launcher uses torchvision's checksum-aware downloaders. A shared persistent
`ADAPTER_DATA_ROOT` needs outbound HTTPS access and roughly 7 GB of free space;
Food-101 dominates the download. Frozen feature caches add less than 0.5 GB.
Flowers-102 requires SciPy, which is already listed in `requirements.txt`.

## Outputs

- Per pair: `pair.json` and `args.json`.
- Development: `development_summary/summary.csv` and `selection_lock.json`.
- Confirmation: `confirmation_summary/pairs.csv`, `summary.csv`, and
  `summary.json`.
- Primary endpoints: final task-incremental average accuracy and average
  forgetting; class-incremental last accuracy and geometry are secondary.

No result is considered manuscript-ready until all four datasets and all five
confirmation seeds are present in the validator output.

## Boundary and raw-image ViT-LoRA confirmation

The four-dataset experiment above is a low-rank residual adapter on frozen
features. It establishes multi-domain breadth efficiently, but it is not by
itself evidence about LoRA modules inside a transformer.

A separate confirmation therefore uses raw Flowers-102 and Oxford-IIIT Pet
images with the same pretrained `vit_tiny_patch16_224` in both arms. Rank-8,
alpha-16 LoRA residuals are registered on all 12 attention blocks' `qkv` and
`proj` matrices (24 projections total); the backbone remains frozen and the
classifier is trainable. The strict pair is again task loss versus task
loss+SSR, with no KD, anchor, spectral or replay objective. The result validator
requires identical initial model hash, dataset hash, pairing hash, rank, alpha,
training config and optimizer-step count.

The cluster cannot reach Hugging Face or the Oxford dataset hosts. Run
`scripts/fetch_vit_lora_assets.sh` on a networked machine and stage its output on
the shared filesystem before launching `scripts/run_vit_lora_confirmation_8gpu.sh`.
The pinned ViT checkpoint SHA-256 is
`fecf81b492bd13ee7a5297cb74d1d417aac8bf7e1b7d96aed89c4691984587ed`.
