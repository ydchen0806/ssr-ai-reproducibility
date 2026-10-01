# Spatial synaptic regularization: manuscript reproduction

AI experiments accompanying **A human connectome reveals a synaptic spatial principle for learning while preserving memory**.

This release corresponds to the manuscript and supplementary material identified in `MANUSCRIPT_IDENTITY.json`. It contains the reported dense knowledge editing, WikiRecent low-rank editing, visual learning and supporting sensitivity analyses. Figures 1–3 and biological analyses are maintained in the manuscript's separate biological repository.

- [Experiment and panel index](docs/experiment_index.md)
- [Reproduction instructions](docs/reproduction.md)
- [Metrics and cohort interpretation](docs/metrics_and_cohorts.md)
- [Numerical source data](Data_S1/README.md)
- [Inference checkpoints and raw records](https://huggingface.co/cyd0806/ssr-ai-checkpoints)

Data S1 matches the experiments reported in the current manuscript. Figure 4 preserves the teacher Version 3598 artwork except C, D and H: C/D show cohort summaries without seed points, and H shows ten paired AlphaEdit orders with both endpoints evaluated on the complete final history. G retains one full-recipe pair per setting. Figure 5 uses ten paired orders per comparison; A/B/E/F show the rank-32 cohort, C shows ranks 8/16/32 from the separate eight-target sweep, and D shows immediate/history locality at ranks 8/16/32 under 32-target rehearsal. Rank 16 in D uses a separate cohort. `Data_S1/figure5_panel_sources.json` maps protocols and seeds.

## Quick numerical verification

```bash
python -m pip install -r requirements-analysis.txt
python scripts/verify_release.py
```

The verifier checks the delivered Data S1 tables. The manuscript figure PDFs are in `figures/reference/`. No GPU or model download is needed for this check. Model training requires the separate task environments and licensed datasets described in the reproduction guide.

Code is released under the repository license; bundled upstream code retains its own attribution and license. Dataset-derived illustrations are included only as manuscript source artwork, not as a redistribution of the training datasets.
