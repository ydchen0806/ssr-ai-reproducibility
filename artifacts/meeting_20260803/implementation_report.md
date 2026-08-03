# 2026-08-03 meeting plan implementation report

This report maps the meeting requests to executable evidence rather than to
manually entered manuscript numbers.

| Meeting requirement | Implementation | Current state |
|---|---|---|
| Isolate SSR from Anchor/Spectral | Eight named editing recipes and four prespecified direct contrasts | Code complete; 116/240 confirmation cells available |
| Compare cosine/projective mapping | Shared distance API, paired mapping contrasts and locked metadata | Code complete; projective historical subset available, full paired matrix pending |
| Keep evaluator/model/optimizer identical | Locality equivalence proof plus evaluator, model and pairing SHA-256 identities | Complete and remotely verified |
| Compare regularizers on one KD scaffold | KD-only teacher trajectory consumed by KD+EWC/MAS/SI/SSR | Code complete; 0/50 runs complete |
| Attribute segmentation gain to SSR | Task/KD/SSR/SSR+KD screen-lock-confirm protocol | Code complete; 0/21 screen and 0/40 confirmation complete |
| Preserve per-seed provenance | Unified records, raw editing trajectories, failure markers and matrix inventory | Complete |
| Generate figures from results | Strict paired aggregator and staged/submission Figure 4/5 scripts | Complete; current figures explicitly incomplete |
| Check adapter direct evidence | Recomputed all 240 rank/mapping/kernel records with paired intervals | Complete; claims bounded in `adapter_evidence_audit.md` |

## Decisions made from the meeting discussion

1. `full - plain` is labelled full-recipe transfer, never direct SSR attribution.
2. Direct editing attribution is `ssr_only - plain`; incremental attribution is
   `full - stabilized`.
3. LwF is represented by the shared KD/LwF-style distillation scaffold. Adding
   a duplicate loss on the same teacher logits would only rescale distillation.
4. Native EWC/MAS/SI remain context results. Only their fixed-KD variants can
   support a controlled comparison with `KD+SSR`.
5. Segmentation's headline comparison is `SSR+KD - KD`, not baseline versus the
   complete objective.
6. Distance mapping is reported as an empirical design variable. Existing
   adapter data support both mappings relative to KD but favor all-cosine in
   the tested rank-8/16 conditions; object-aware is not described as universal.
7. Development screens select parameters only. Their seeds and numbers cannot
   enter confirmation intervals or final figures.

## What remains before manuscript integration

- Complete the three matrices listed in `execution_status.md`.
- Run the strict aggregation and fairness gates without incomplete overrides.
- Populate `claim_registry.yaml` from the final aggregate rows.
- Only then replace manuscript values and regenerate Figure 4/5. No current
  missing value should be inferred, copied from a different protocol or chosen
  seed-by-seed.
