# Direct SSR attribution on additional segmentation datasets

This protocol extends the CUB mask experiment without changing its locked
artifacts. It uses two additional fine-grained datasets with image-level class
labels and pixel-level foreground masks:

- **Oxford-IIIT Pet**: 37 breeds, official train/test split and trimap masks.
  The official page licenses the dataset under CC BY-SA 4.0; image copyright
  remains with the original owners. Trimap value 1 is foreground, value 2 is
  background and value 3 is excluded as border/void in loss and metrics.
- **Oxford Flowers 102**: 102 flower categories, official train/validation/test
  split and official foreground segmentations. The Oxford page does not state a
  redistribution licence, so the downloader fetches directly from Oxford and
  the source archives must not be committed or redistributed.

Official sources:

- <https://www.robots.ox.ac.uk/~vgg/data/pets/>
- <https://www.robots.ox.ac.uk/~vgg/data/flowers/102/>

## Matched contrast

The primary comparison is deliberately narrow:

```text
task_only: binary cross-entropy + Dice
task_ssr:  binary cross-entropy + Dice + lambda_ssr * SSR
```

Both arms use the same frozen ImageNet ResNet-18 layer-3 features, decoder,
class order, optimizer, task schedule, epoch budget and seed. No KD, anchor or
spectral term is present. The result therefore estimates the direct increment
from SSR rather than the increment from a composite recipe.

## Selection and confirmation

Five finite Gaussian SSR settings are screened on three development seeds. A
single setting is selected jointly across both datasets, then locked and tested
on ten new paired seeds per dataset. Selection ranks settings by the number of
datasets with positive mean changes in mIoU, Dice and forgetting reduction,
then by the worst individual change, paired triple wins and pooled mean change.

The complete workload is 76 runs:

- 36 development runs: 2 datasets x 3 seeds x (1 task-only + 5 task+SSR)
- 40 confirmation runs: 2 datasets x 10 seeds x 2 matched objectives

The preparer records source URL, exact byte count and SHA256 for every official
archive or mirrored shard. Result records bind the extracted-dataset fingerprint, frozen-feature
cache SHA256, git commit, seed, objective and selection-lock hash. Completed
records are validated and skipped on restart.

For Oxford Pet, `--pet-source auto` first checks
`data/oxford_hf_parquet/data` for the six shards from
`dpdl-benchmark/oxford_iiit_pet`. A complete mirror is verified against the six
Hugging Face LFS SHA256 values at revision
`18515194612b584bf54692372ac035ef773a5b41`, then materialized locally without
re-encoding images or trimaps. A partial or mismatched mirror fails closed.
Use `--pet-source official` to force the Oxford archives, or
`--pet-source hf_parquet` to require the local mirror.

## One eight-GPU node

```bash
cd /unify/ydchen/unidit/bioreg_meeting_20260803_final

python scripts/run_multidataset_segmentation.py \
  --result-root results/segmentation_multidata_20260804_r1 \
  --data-root data \
  --prepare-data \
  --phase all \
  --gpus 0 1 2 3 4 5 6 7
```

The first launch downloads approximately 1.36 GB of official source data and
builds two frozen-feature caches before dispatching training. For a scheduler
retry, use the same command and result root; validated jobs resume. Use a new
result root when changing the protocol or hyperparameter matrix.

To inspect the exact job inventory without downloading data or starting GPUs:

```bash
python scripts/run_multidataset_segmentation.py \
  --result-root results/segmentation_multidata_dryrun \
  --manifest-only
```
