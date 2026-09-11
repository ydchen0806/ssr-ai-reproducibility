# Current AI Figure Reproduction Map

The manuscript follows dense knowledge editing, low-rank knowledge editing,
and visual transfer. Results from different cohorts remain separate. The
September 11 text revision did not change experimental values or training code.

## Evidence and executables

All paths in this table are relative to the repository. The numerical archive
is `artifacts/manuscript_20260911/` (abbreviated `A/` below).

| Figure | Cohort and comparison | Data | Training/analysis entry point |
| --- | --- | --- | --- |
| 4b/e/f | GPT-2 XL dense editing, direct SSR and factorial recipes | `A/source_data/raw/knowledge_editing/meeting_extension_cf/`; `A/figure_source_data/figure4_20260905/` | `scripts/run_meeting_editing_8gpu.sh`, locked GPT-2 XL config |
| 4c | Separate mapping confirmation and radial development | `A/source_data/cosmos_ke_completion_20260901/` | `scripts/run_llm_ke_easyedit.py`; the raw records contain resolved settings |
| 4d/g | Matched editor additions; combined-recipe model transfer | `A/source_data/raw/knowledge_editing/`; `A/artifacts/final_strict_20260829/figure4_source_data.csv` | `scripts/run_llm_ke_easyedit.py`, `llm_ke/biocs_editor.py` |
| 4h | Qwen dense-update geometry, four pairs per dataset | `A/source_data/ke_geometry_cosmos/` | `scripts/ke_geometry/run_ssr_cosmos_ke_geometry.py` |
| 5a/c/d | WikiRecent, ranks 8/16/32/64, 100 edits, ten paired orders | `A/figure5_current/efficacy_locality_seed_source_data.csv` | `scripts/run_lowrank_ke_easyedit.py`, `configs/lowrank_ke/qwen25_all_dataset_rank_transfer_100edit.json` |
| 5b/e/f | Separate frozen WikiRecent rank-8 trajectory and geometry | `A/figure5_current/wikirecent_rank8_100.json` | Same LoRA runner; retain its own seed/configuration identities |
| 6a/b/c | VOC 10-1, rank-8 history decoder, LowRank vs LowRank+SSR | `A/data/lowrank_confirmations/`; `A/figure_source_data/figure6_20260905/` | `external/PLOP_SSR/`, `scripts/voc/run_plop_lowrank_v24_sequential_trial.py` |
| 6d | CUB, fixed KD, GELU bottleneck, 40 epochs/task, ten pairs/rank | `A/figure_source_data/figure6_20260905/` | `experiments/lowrank_adapter_mapping.py` |
| 6e left | Matched fixed-KD CUB geometry families, frozen development budget | Same figure-source directory | `experiments/lowrank_geometry_fixedkd.py` |
| 6e right | Fixed-KD Split-CIFAR-100 rank-4 strength sweep | `A/figure_source_data/figure6_20260907/` | `experiments/lowrank_adapter_functionalspace_regularizers_fig6.py` |
| 6f | Actual CUB rank-32 checkpoint pair, seed 8411, illustrative map | `A/source_data/figure6_20260905/response_maps_rank32_seed8411/` | `scripts/rebuild_cub_lowrank_response_maps.py` |

The archive also retains ZsRE boundaries, unresolved intervals, development
results, and the complete nine-setting strength sweep. A source inventory
records file SHA-256 digests. Raw records retain their original provenance
strings; these are metadata, not filesystem locations that a new user must
recreate. Image datasets are obtained from their original providers.

## Metrics

| Field | Meaning | Display |
| --- | --- | --- |
| Immediate efficacy | Current edit target acquisition | Percent; gain = SSR minus matched control |
| History efficacy | Re-evaluated earlier edit targets | Percent; not the immediate-efficacy field |
| Locality | Target matching on the supplied unrelated relations | Percent; not pre-edit output consistency |
| Pre-edit output consistency | Preservation of a model's original outputs | Separate archived diagnostic |
| AA / AF | Average accuracy / average forgetting | Percent; AA gain = SSR-control, AF reduction = control-SSR |
| VOC trajectory AUC | Arithmetic mean of ten incremental-stage percentages | Normalized percent |
| Row contrast / target alignment | Dimensionless LoRA-B topology diagnostics | Native units, no percentage scaling |

The source records specify the CI procedure for each cohort. Student-t and
bootstrap intervals are not interchangeable. A curve checkpoint is not an
independent seed, and the two VOC seed sets are separate cohorts.

## Environment and upstream code

Use Python 3.11, the repository requirements, and `requirements-release.txt`.
GPU dependencies must match the installed CUDA driver. Historical environment
gaps remain documented in the earlier reproduction map; this packaging pass
does not establish bitwise training reproducibility across hardware.

The low-rank KE source snapshot starts from repository commit
`d85291a`; the release adds optional verified checkpoint saving and resolves
CLI paths before EasyEdit changes the working directory. Evaluation and SSR
math are unchanged. Install EasyEdit at commit
`14cea8245f06715684592ab55184939b99d70784` and apply the recorded editor patch
in `external/EasyEdit_manuscript.patch` for the AlphaEdit/SPHERE extensions.
The patch does not affect the native LoRA editor.

VOC source is vendored from the actual run checkout at
`64dba29a0d5c9b84e98f0487a71d663d5bc0e647`, with its upstream MIT license.
The `external/plop_apex_compat` overlay is also included. This is the
implementation provenance of LowRank vs LowRank+SSR, not a claim that the
panel compares SSR to a newly tuned PLOP baseline.

## Rebuild numerical figures

```bash
pip install -r requirements-release.txt
python scripts/rebuild_manuscript_numerical_figures.py
python scripts/verify_manuscript_source_inventory.py
```

Outputs are written under `artifacts/manuscript_20260911/figures/`; manuscript
files are never overwritten. Figure 4's numerical panels and the full current
Figure 5 can be rebuilt from the archive. The artist-supplied Figure 4a is a
separate vector asset in the manuscript delivery. Figure 6's drawing source is
included; its qualitative panels additionally require the original VOC/CUB
images and masks. CUB scalar maps and their exact checkpoint identities are
included, so colorized images never have to be subtracted to recover a map.

## Retrain and evaluate

Use the locked dense-edit command and CUB 40-epoch command in
`docs/manuscript_reproduction.md`. For current low-rank KE, stage the original
datasets and Qwen2.5-7B-Instruct model, then use the frozen rank transfer config
and `scripts/run_lowrank_ke_all_datasets_ranks_3node.sh`. The cluster wrapper
accepts `ASSET_ROOT` and `PROJECT_ROOT`; its internal directories are stated in
the script. For a single run, use the lower-level Python runner with the same
resolved arguments and add `--checkpoint-dir results/<run>/checkpoints`.

For VOC, download VOC2012/trainaug and the initialization weights specified
by `external/PLOP_SSR/README.md`. The following reproduces the rank-8 SSR arm
of one confirmation seed from its matched stage-0 checkpoint:

```bash
python scripts/voc/run_plop_lowrank_v24_sequential_trial.py \
  --plop-root "$PWD/external/PLOP_SSR" --python "$(command -v python)" \
  --overlay "$PWD/external/plop_apex_compat" --data-root "$VOC_DATA_ROOT" \
  --trial-root "$PWD/results/voc_seed12401_ssr" --task 10-1 \
  --arm lowrank_ssr --seed 12401 --gpus 0 \
  --parent-checkpoint "$VOC_MATCHED_STAGE0" \
  --low-rank-classifier-rank 8 --low-rank-classifier-alpha 8 \
  --ssr-lambda 0.01 --ssr-distance projective --ssr-warmup-epochs 5 \
  --ssr-normalization relative_primary_v1 --ssr-gradient-gate nonconflicting_v1
```

Run the control with `--arm lowrank --ssr-lambda 0` and the same seed and
stage-0 weights. Use seeds 12401--12410 for the main cohort and 13401--13410
for the replication. The wrapper supports `--stage0-only`; use the same
stage-0 artifact for each within-seed pair. Record final file hashes.

The two CUB regularizer runners expose `run`, `select`, and `summarize` CLI
commands. Resolved run records and development selections in the archive are
the configuration source; do not rerank confirmation seeds or tune on them.

## Checkpoints and remaining gaps

See [checkpoint_downloads.md](checkpoint_downloads.md). The 42 exported
visual weights have exact tensor and SHA-256 verification; HF publication is
blocked by the current credential's write permission. Earlier KE runs and
some CUB quantitative cohorts saved results without model weights. Their
training sources and result records are included, but the missing models
require frozen-protocol reruns. This release has unit/smoke checks; it does
not claim that every full GPU experiment has been rerun for this packaging pass.
