# Meeting execution status

## Implemented

- Explicit eight-recipe editing objective with deterministic row selection and cosine/projective mappings.
- Versioned locality evaluator and strict completion accounting; failed edits cannot enter official tables as zero-valued runs.
- Locked GPT-2 XL editing configuration and a resumable, multi-GPU/multi-node launcher.
- Import of 116 complete historical confirmation runs with source and dataset hashes.
- Fixed-KD implementations for KD, KD+EWC, KD+MAS, KD+SI and KD+SSR, including deterministic task loaders and an automatic scaffold-fairness check.
- Corrected segmentation KD over previously learned class conditions and a resumable 21-run screen plus 40-run confirmation driver.
- Unified result schema, strict paired aggregation, mapping comparisons, missing-pair audit and staged/submission consistency modes.
- Figure 4/5 evidence-summary generation directly from aggregated CSVs. Submission mode fails on missing contrasts or paired seeds; staged mode visibly marks incomplete evidence.
- Resume/reuse guards validate schema and exact experiment identity before skipping work. New formal runs also reject a dirty tracked worktree; legacy reuse must carry explicit locked provenance and its original per-edit `results.json`.
- Fixed-KD pairing includes a common-scaffold hash, and the segmentation protocol fingerprints both the mask source archive and the 192-pixel feature cache once before dispatch.

## Experiment state

- Editing projective confirmation: 116/120 complete. Recent `stabilized` and `full` seeds 9327/9329 are missing.
- Full six-recipe/two-mapping editing matrix: 116 records reusable; 124 runs genuinely remain.
- Matched-KD classification: 0/50 new runs complete.
- CUB segmentation: 0/21 development and 0/40 confirmation runs complete under the corrected KD protocol.
- The remote two-GPU host was not used because both GPUs were already at 90--100% utilization by unrelated jobs.

## Compute estimate

- The 116 imported editing runs averaged 651.65 seconds (range 350.70--930.70 seconds). The 124 genuinely missing runs therefore require approximately 22.45 GPU-hours: about 2.8 wall-clock hours on eight equivalent GPUs or 11.2 hours on two.
- Classification and segmentation wall-clock estimates remain intentionally unset until one locked smoke job is profiled on the target server; they are not inferred from unrelated historical jobs.

## Data state

No new P0 dataset download is required. The remote server already contains the three KnowEdit files, GPT-2 XL, CIFAR-100, TinyImageNet, CUB images/masks and the 192-pixel dense segmentation cache. Their authoritative dataset hashes are recorded in the result schema/importer.

## Completion gate

The manuscript is not ready for numerical freeze until the 124 editing runs, 50 matched-KD runs and segmentation screen/confirmation finish; all records aggregate without missing pairs; the fairness check passes; and the populated claim registry passes in submission mode.
