# Reproduction

## Source-table and figure reproduction

Use Python 3.11 or newer and `requirements-analysis.txt`. Run `python scripts/verify_release.py` from the repository root. The verifier checks the current Data S1 manifest and the Figure 4/5 source tables delivered with this release. Exact manuscript PDFs are retained under `figures/reference/`. Figure 6 and supplementary artwork are supplied there or in `plot_data/`. The two Figure 5 protocols and reported cohorts are mapped in `Data_S1/figure5_panel_sources.json`.

## Training environments

GPU training was performed under task-specific environments. Install `requirements.txt` for the continual-learning scaffold and `requirements-release.txt` for checkpoint serialization. These are compatibility requirements, not a claim of a universal bitwise-identical environment. Dense C records include `protocols/figure4/C/runtime_versions.json`; per-run commands, dtype, dataset/model fingerprints and training parameters are recorded in `protocols/` and the HF raw records.

Obtain EasyEdit from its official upstream at commit `14cea8245f06715684592ab55184939b99d70784`, retain its license, and apply `external/EasyEdit_manuscript.patch` in that checkout. Set `EASYEDIT_DIR` to it. PLOP and its compatibility overlay are included in `external/`; see `external/SSR_UPSTREAM_SOURCES.md`. Obtain pretrained models and datasets from their original providers, respecting access conditions. No pretrained language-model weights are redistributed here.

## Replaying a recorded run

`protocols/RESULT_INDEX.json` maps the current dense/kernel/transfer/native studies to exact run configurations and HF result hashes. Each run directory provides `invocation.json` or `command.json` with the executed arguments; paths from the original machine are provenance, not portable defaults. Use `python scripts/replay_recorded.py PATH_TO_INVOCATION --path-map paths.json --output RUN_DIRECTORY` to print the resolved command, or add `--execute` to run it. The path-map JSON maps original path prefixes to your local copies. It must cover model, dataset, EasyEdit, cache and stream paths; scientific flags and seeds are preserved. Use the corresponding frozen `implementations/kernel`, `factorial`, `transfer` or `native_history` driver tree. Inspect the printed command and required environment before GPU execution.

Dense B/D/E original runs use `scripts/run_llm_ke_easyedit.py`; the mapping experiment uses the ten original orders. The Figure 5C sweep uses `scripts/run_lowrank_ke_rehearsal.py` with `scripts/run_lowrank_ke_easyedit.py`: learning rate 0.0005, 20 updates/edit, alpha/r=4, tested ranks 8, 16 and 32 and SSR coefficients 0.03, 0.01 and 0.03. The current target is rehearsed with up to eight earlier targets, and every plain/SSR pair shares its stream. `Data_S1/figure5_rehearsal/` contains the 60-run numerical export, source manifest and 60 indexed raw `results.json` records for ranks 8, 16 and 32. The rank-8/32 experiment uses `scripts/run_fig5_dual_endpoints.py` and `scripts/fig5_dual_campaign.py`; its rank-32 cohort supplies A/B/E/F with 32-target rehearsal and coefficient 0.03. Rank 8 also supplies D; the separate rank-16 locality cohort is recorded in Table S20 and displayed in D. This protocol uses complete-phrase matching for immediate/history efficacy and locality. The earlier 60-step immediate-only protocol is preserved in the repository history, not used for the current main figure.

For the broad classification streams, JSON configuration files in `configs/visual/` are accepted as YAML by `main.py`; select the matching seed from the raw record. CUB geometry/taxonomy and low-rank drivers expose `--help`; the corresponding raw configuration/protocol records are in `supporting_records/` and HF. VOC uses the frozen selection recorded in the five-seed `v24_summary.json` and `v31_summary.json` files with `scripts/voc/run_plop_lowrank_v24_sequential_trial.py`.

## Checkpoint coverage

HF contains 54 inference tensor files: paired final-stage VOC checkpoints for the five primary paired seeds, one paired replication illustration (seed 13401), two rank-32 CUB checkpoints for the response illustration, and final LoRA adapters for the rank-8 and rank-32 Figure 5 follow-up (ten seeds, plain and SSR). These are not checkpoints for every experiment or optimizer-resume states. `checkpoint_manifest.json` specifies tensor count, SHA-256, size and role. Use `scripts/load_visual_checkpoint.py` to load safe tensors. Numerical source data for the reported comparisons are in Data S1.

Current Figure 4G shows the original full-recipe examples with one pair per setting; H shows ten paired AlphaEdit orders, with efficacy and locality both evaluated on all 250 historical items at the final checkpoint. These protocols are reported separately.

## Records retained with this release

The five-pair same-base CUB experiment and four general-stream cohorts are supplied. Figure 4G uses the recorded endpoints for each example. Original H field names are clarified in `Data_S1/figure4/COHORT_REGISTRY.md`.
