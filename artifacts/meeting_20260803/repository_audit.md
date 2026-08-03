# Repository audit for the 2026-08-03 meeting revision

## Scope and immutable starting point

- Repository: `ydchen0806/ssr-ai-reproducibility`
- Starting commit: `adb8d66` (`Release auditable SSR AI experiments`)
- Working branch: `revision/meeting-20260803`
- Local Python: 3.14.5; local Torch: 2.13.0 without CUDA
- GPU experiments run on the cluster rather than this macOS checkout.

The clean release checkout is used instead of the older working directory, which contains
uncommitted experiment scripts and generated files. Existing artifacts are preserved.

## Existing implementation

- `llm_ke/biocs_editor.py` already supports Gaussian, Laplace, Cauchy and historical inverse
  radial families.
- Cosine and projective distance are already implemented.
- `fixed_random` and `active_top` row selection are deterministic for a recorded seed; the
  legacy `random_each_step` mode remains available for backward compatibility.
- LLM results already retain kernel, distance, row-selection and optimizer metadata.
- `experiments/cub200_continual_benchmark.py` already implements task-only, KD, SSR and
  SSR+KD segmentation conditions through the legacy method names `baseline`, `kd`, `biocs`
  and `biocs_kd`.
- The release contains per-seed classification, adapter, editing, segmentation and CUB
  mechanism artifacts plus `scripts/verify_paper_artifacts.py`.

## Gaps confirmed by the audit

- The editing runner does not expose a named recipe registry, so objective components are
  still selected through environment variables.
- Per-run outputs do not yet share one task-independent result schema.
- The native EWC/MAS/SI classes do not share the same KD scaffold. Existing geometry controls
  do, but they do not implement those three memory penalties.
- The custom editing evaluator and EasyEdit context baselines use different locality
  extraction and aggregation rules. Their numbers must not be mixed as a matched comparison.
- Manuscript source is intentionally outside this code release; revision instructions and
  generated evidence tables are emitted as artifacts.

## Remote run reuse

- `ssr_edit_reg_4node_20260803_r1` is complete and is retained as a negative/mixed editing
  regularizer audit.
- `ssr_decisive_3node_20260803_r1/node_c` is complete and supplies the locked CUB mechanism
  evidence.
- The LLM node has 116/120 complete validation runs and can be resumed without recomputing
  completed jobs.
- The segmentation node created only its manifest and must be rerun.
