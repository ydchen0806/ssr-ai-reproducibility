# Current CounterFact factorial evidence

This directory contains the CounterFact evidence used by the current SSR
manuscript. `counterfact_factorial_summary.json` is the complete paired
factorial record: all six recipes, both distance mappings, every evaluated
metric, and all ten paired seeds are retained, including null and unfavorable
contrasts.

The manuscript reports the prespecified comparisons that isolate the SSR term
or test its addition to a matched stabilizer:

- task loss versus task loss plus SSR;
- anchor plus spectral stabilization versus the same objective plus SSR;
- spectral stabilization versus the same objective plus SSR, under cosine and
  projective mappings.

`reported_values.json` maps each displayed value to its exact location in the
complete factorial record. Recompute and verify those values with:

```bash
python3 scripts/verify_paper_20260810.py
```

The remaining manuscript evidence is organized separately: CUB classification,
mechanism and adapter records are under `../paper_20260803/`; direct
segmentation boundary records and an invalidated raw-image ViT-LoRA archive are
under `../paper_20260809/`. The latter is retained only for auditability and is
not manuscript evidence.
