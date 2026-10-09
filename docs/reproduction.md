# Reproduction

## Source-table and figure reproduction

Use Python 3.11 or newer and `requirements-analysis.txt`. Run `python scripts/verify_release.py` from the repository root. The verifier checks the current Data S1 manifest and the Figure 4/5 source tables delivered with this release. Exact manuscript PDFs are retained under `figures/reference/`. Figure 6 and supplementary artwork are supplied there or in `plot_data/`. The two Figure 5 protocols and reported cohorts are mapped in `Data_S1/figure5_panel_sources.json`.

## Training environments

GPU training was performed under task-specific environments. Install `requirements.txt` for the continual-learning scaffold and `requirements-release.txt` for checkpoint serialization. These are compatibility requirements, not a claim of a universal bitwise-identical environment. Dense C records include `protocols/figure4/C/runtime_versions.json`; per-run commands, dtype, dataset/model fingerprints and training parameters are recorded in `protocols/` and the HF raw records.

Obtain EasyEdit from its official upstream at commit `14cea8245f06715684592ab55184939b99d70784`, retain its license, and apply `external/EasyEdit_manuscript.patch` in that checkout. Set `EASYEDIT_DIR` to it. PLOP and its compatibility overlay are included in `external/`; see `external/SSR_UPSTREAM_SOURCES.md`. Obtain pretrained models and datasets from their original providers, respecting access conditions. No pretrained language-model weights are redistributed here.

## Replaying a recorded run

`protocols/RESULT_INDEX.json` maps the current dense/kernel/transfer/native studies to exact run configurations and HF result hashes. Each run directory provides `invocation.json` or `command.json` with the executed arguments; paths from the original machine are provenance, not portable defaults. Use `python scripts/replay_recorded.py PATH_TO_INVOCATION --path-map paths.json --output RUN_DIRECTORY` to print the resolved command, or add `--execute` to run it. The path-map JSON maps original path prefixes to your local copies. It must cover model, dataset, EasyEdit, cache and stream paths; scientific flags and seeds are preserved. Use the corresponding frozen `implementations/kernel`, `factorial`, `transfer` or `native_history` driver tree. Inspect the printed command and required environment before GPU execution.

Dense B/D/E original runs use `scripts/run_llm_ke_easyedit.py`; the mapping experiment uses the ten original orders. Figure 5C and D are the 32-target comparisons. Ranks 8 and 32 use `scripts/run_fig5_dual_endpoints.py` and `scripts/fig5_dual_campaign.py`: learning rate 0.0005, 20 updates/edit, alpha/r=4, a 32-target rehearsal window, and SSR coefficients 0.01 at rank 8 and 0.03 at rank 32. Rank 32 supplies panels A, B and E and the rank-32 bars in C and D, using orders 26092711--26092719 and 26092721. Rank 8 supplies the rank-8 bars in C and D, using orders 26092711--26092720. Rank 16 is a separate 32-target cohort, coefficient 0.03, orders 26092811--26092820, and supplies the rank-16 bars in C and D. This protocol uses complete-phrase matching for immediate and history efficacy and locality. `Data_S1/figure5_rehearsal/` is the earlier eight-target sweep (up to eight earlier targets; history locality was not recorded) and is not shown in Figure 5. The earlier 60-step immediate-only protocol is preserved in the repository history, not used for the current main figure.

For the broad classification streams, JSON configuration files in `configs/visual/` are accepted as YAML by `main.py`; select the matching seed from the raw record. CUB geometry/taxonomy and low-rank drivers expose `--help`; the corresponding raw configuration/protocol records are in `supporting_records/` and HF. VOC uses the frozen selection recorded in the five-seed `v24_summary.json` and `v31_summary.json` files with `scripts/voc/run_plop_lowrank_v24_sequential_trial.py`.

## Checkpoint coverage

HF contains 54 inference tensor files: paired final-stage VOC checkpoints for the five primary paired seeds, one paired replication illustration (seed 13401), two rank-32 CUB checkpoints for the response illustration, and final LoRA adapters for the rank-8 and rank-32 Figure 5 follow-up (ten seeds, plain and SSR). These are not checkpoints for every experiment or optimizer-resume states. `checkpoint_manifest.json` specifies tensor count, SHA-256, size and role. Use `scripts/load_visual_checkpoint.py` to load safe tensors. Numerical source data for the reported comparisons are in Data S1.

Current Figure 4G shows the original full-recipe examples with one pair per setting; H shows ten paired AlphaEdit orders, with efficacy and locality both evaluated on all 250 historical items at the final checkpoint. These protocols are reported separately.

## Records retained with this release

The five-pair same-base CUB experiment and four general-stream cohorts are supplied. Figure 4G uses the recorded endpoints for each example. Original H field names are clarified in `Data_S1/figure4/COHORT_REGISTRY.md`.
