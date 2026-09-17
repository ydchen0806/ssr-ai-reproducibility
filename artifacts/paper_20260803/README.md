# SSR AI paper artifact (2026-08-03)

This directory is the compact audit bundle for the current AI manuscript
results. `raw/` contains the per-seed result records and locked reports copied
from the completed cluster runs. Large datasets, feature caches, model weights,
checkpoints, and training logs are intentionally excluded.

Run the verifier from the repository root:

```bash
python scripts/verify_paper_artifacts.py
```

The verifier checks every raw-file SHA-256 digest, exact seed cohorts, matched
configuration fields, manuscript-facing rounded values, paired intervals, and
the held-out CUB mechanism report. It prints recomputed, unrounded statistics.

Evidence is separated by what it can establish:

- `classification/*.jsonl`: direct KD versus SSR+KD attribution on matched code paths.
- `adapter/validation_runs.jsonl`: direct KD versus SSR+KD attribution, including rank, mapping, and
  radial-family controls.
- `segmentation/`: transfer of the complete SSR+KD objective relative to the
  base objective; this is not an SSR-only ablation. The three-seed cohort was
  recovered from an append-only result stream, but the exact seed-0 launch
  manifest was not recoverable (`segmentation/protocol.json`).
- `llm/validation_runs.jsonl`: locked nested SSR comparisons. Their confidence intervals
  include zero; they are included to make that null result auditable.
- `llm/descriptive_runs.jsonl`: single-run, cross-checkpoint full-recipe pairs.
  These demonstrate portability but are not inferential replicates.
- `context_baselines/`: the common-seed LwF, EWC, MAS, and SI context table.
- `mechanism/`: the held-out CUB semantic-neighborhood and hard-confusion audit.

Two rows in the exploratory three-seed adapter factorization, prototype-only
SSR and adapter-basis-only SSR, do not have a recoverable raw per-seed artifact
in the available archive. Their manuscript values are therefore not asserted
by this release verifier. The independent ten-seed target/mapping validation is
fully included.

The selected ZsRE inverse response is also retained with an explicit
implementation audit in `llm/protocol.json`. Its recorded scale is broader
than a true HWHM-matched inverse control, so this result cannot be used as
evidence for HWHM-matched kernel invariance.

The recorded environment is intentionally partial. Package versions for
`transformers`, `tokenizers`, `accelerate`, the EasyEdit revision, the model
revision, and the operating-system build were not recoverable, and the source
workspace commit is not part of this public repository. Absolute cluster paths
were normalized to `source_workspace/...`; original source-file content hashes
are retained where available.
