# Adapter evidence audit

## Scope and calculation

This audit recomputes the CUB200 adapter evidence from the ten paired validation seeds `8411, 8413, 8415, 8417, 8419, 8421, 8423, 8425, 8427, 8429`. Every treatment is compared with the KD record at the same adapter rank and seed.

- Paired accuracy change: `Delta AA_s = AA_treatment,s - AA_KD,s`.
- Paired forgetting reduction: `Delta AF red._s = AF_KD,s - AF_treatment,s`; positive values favor the treatment.
- The reported 95% confidence interval is the two-sided paired Student interval `mean +/- t(0.975, 9) * sample_sd / sqrt(10)`, with `t(0.975, 9) = 2.2621571627409915`.
- A dual win is a seed for which both `Delta AA_s > 0` and `Delta AF red._s > 0`.
- `Manifest` means the rank/condition pair is declared under `adapter_validation.conditions` and is checked by the repository verifier. `Raw-only` means that all ten paired records and a normalized configuration exist in the compact artifact, but that pair is not declared as a reported validation condition.

Numeric source: [`validation_runs.jsonl`](../paper_20260803/raw/adapter/validation_runs.jsonl), SHA-256 `6fd5795edc7566a049471b063622a8e641ffbf154579b135205202891900d137`. Configuration source: [`configs.json`](../paper_20260803/raw/adapter/configs.json), SHA-256 `4ed0ecdc4af5668045e20d52cbde79057823be991cb2f5b6faef7dfc4aebfe88`. Cohort declaration: [`manifest.json`](../paper_20260803/manifest.json), SHA-256 `9af2dbacd7288cf83fb09a3a7561c10ceb2be08eff959199ba2e5becddd55cfb`. The calculation follows [`verify_paper_artifacts.py`](../../scripts/verify_paper_artifacts.py), SHA-256 `064c3a023bd27973941ab5df37bd47de25cfcef16f0736d05251b9503638b5f3`.

## Paired results versus same-rank KD

### Rank 8

| Condition | Evidence status | Paired Delta AA (95% CI) | Paired Delta AF reduction (95% CI) | Dual wins |
|---|---|---:|---:|---:|
| Gaussian, object-aware | Manifest | +0.271 [+0.039, +0.503] | +0.350 [+0.049, +0.651] | 7/10 |
| Gaussian, all-cosine | Manifest | +0.593 [+0.402, +0.783] | +0.620 [+0.415, +0.826] | 10/10 |
| Gaussian, prototype only | Raw-only | +0.359 [+0.158, +0.560] | +0.344 [+0.100, +0.587] | 8/10 |
| Gaussian, basis projective | Raw-only | +0.049 [-0.181, +0.279] | +0.221 [-0.014, +0.456] | 6/10 |
| Laplace, object-aware | Raw-only | +0.127 [-0.050, +0.305] | +0.215 [-0.008, +0.439] | 7/10 |
| Cauchy, object-aware | Raw-only | +0.084 [-0.123, +0.290] | +0.096 [-0.162, +0.354] | 5/10 |
| Inverse, object-aware | Raw-only | +0.106 [-0.084, +0.296] | +0.192 [-0.064, +0.449] | 6/10 |

### Rank 16

| Condition | Evidence status | Paired Delta AA (95% CI) | Paired Delta AF reduction (95% CI) | Dual wins |
|---|---|---:|---:|---:|
| Gaussian, object-aware | Manifest | +0.737 [+0.401, +1.073] | +0.795 [+0.462, +1.129] | 9/10 |
| Gaussian, all-cosine | Manifest | +0.953 [+0.603, +1.303] | +0.909 [+0.587, +1.231] | 9/10 |
| Gaussian, prototype only | Manifest | +0.298 [-0.034, +0.630] | +0.191 [-0.153, +0.535] | 6/10 |
| Gaussian, basis projective | Manifest | +0.306 [-0.066, +0.678] | +0.349 [-0.067, +0.764] | 7/10 |
| Laplace, object-aware | Manifest | +0.338 [-0.001, +0.677] | +0.234 [-0.162, +0.629] | 7/10 |
| Cauchy, object-aware | Manifest | +0.361 [+0.045, +0.677] | +0.236 [-0.097, +0.570] | 7/10 |
| Inverse, object-aware | Manifest | +0.087 [-0.223, +0.398] | -0.090 [-0.439, +0.260] | 4/10 |

### Rank 32

| Condition | Evidence status | Paired Delta AA (95% CI) | Paired Delta AF reduction (95% CI) | Dual wins |
|---|---|---:|---:|---:|
| Gaussian, object-aware | Manifest | +0.969 [+0.588, +1.350] | +0.781 [+0.372, +1.191] | 9/10 |
| Gaussian, all-cosine | Manifest | +1.279 [+0.891, +1.667] | +0.915 [+0.487, +1.343] | 10/10 |
| Gaussian, prototype only | Raw-only | +0.341 [+0.012, +0.671] | +0.028 [-0.401, +0.457] | 5/10 |
| Gaussian, basis projective | Raw-only | +0.869 [+0.510, +1.227] | +0.699 [+0.322, +1.077] | 10/10 |
| Laplace, object-aware | Raw-only | +0.639 [+0.189, +1.090] | +0.197 [-0.287, +0.682] | 6/10 |
| Cauchy, object-aware | Raw-only | +0.513 [+0.143, +0.883] | +0.283 [-0.113, +0.678] | 7/10 |
| Inverse, object-aware | Raw-only | +0.453 [+0.113, +0.793] | +0.186 [-0.297, +0.669] | 7/10 |

## Object-aware versus all-cosine

In the locked Gaussian comparison, both conditions use cosine distance for classifier prototypes and the same SSR coefficients. The only configuration difference is the adapter-basis distance: projective for object-aware and cosine for all-cosine. The table below reports `all-cosine - object-aware`; positive values favor all-cosine.

| Rank | Paired Delta AA (95% CI) | Paired Delta AF reduction (95% CI) | All-cosine dual wins |
|---:|---:|---:|---:|
| 8 | +0.321 [+0.155, +0.488] | +0.270 [+0.025, +0.515] | 7/10 |
| 16 | +0.216 [+0.066, +0.366] | +0.114 [+0.003, +0.225] | 6/10 |
| 32 | +0.310 [-0.028, +0.647] | +0.134 [-0.092, +0.360] | 5/10 |

All-cosine has the higher mean AA and lower mean forgetting at all three ranks. Its paired advantage over object-aware is supported by positive 95% intervals for both endpoints at ranks 8 and 16; at rank 32 the direct intervals cross zero. The evidence therefore does **not** support a claim that object-aware is the empirically optimal mapping. It supports the narrower conclusion that object-aware remains effective relative to KD across ranks, while all-cosine is the strongest tested Gaussian mapping in this CUB200 adapter cohort.

## Claims supported for the main text

1. **SSR improves both retained accuracy and forgetting over same-rank KD across adapter capacities.** Both Gaussian mappings have positive paired means and positive 95% intervals for AA and AF reduction at ranks 8, 16 and 32. Gaussian all-cosine records 10/10, 9/10 and 10/10 dual wins; Gaussian object-aware records 7/10, 9/10 and 9/10.
2. **The gain is not confined to a single adapter rank.** The locked rank sweep provides replicated positive evidence at ranks 8, 16 and 32. This is a capacity-robustness result, not proof that increasing rank causes the gain.
3. **Both cosine and projective adapter mappings are viable, but all-cosine is stronger here.** Object-aware projective geometry is consistently positive versus KD; direct paired comparisons favor all-cosine at ranks 8 and 16 and are inconclusive at rank 32.
4. **At rank 16, the locked Gaussian object-aware condition provides the clearest dual-endpoint evidence among the tested projective object-aware radial families.** Cauchy supports AA but not AF reduction at the 95% level; Laplace and inverse do not support both endpoints. This can motivate Gaussian as the primary locked kernel without claiming universal kernel superiority.
5. **The joint rank-16 object-aware condition has stronger evidence than either component alone relative to KD.** Prototype-only and basis-projective have positive means, but both endpoint intervals cross zero; the joint condition has positive intervals for both endpoints. This is evidence of a more reliable joint configuration, not a formal interaction or synergy test.

## Claims not supported for the main text

1. Do not state that object-aware outperforms all-cosine, that projective distance is selected by the adapter object, or that object-aware is the best mapping in this cohort.
2. Do not state that all radial kernels are equivalent or universally beneficial. The rank-16 inverse condition has negative mean AF reduction, and several kernel intervals cross zero.
3. Do not state that every seed or every configuration is a dual win. The dual-win counts range from 4/10 to 10/10.
4. Do not use the rank-8/rank-32 component and non-Gaussian rows as locked confirmatory results unless they are first added to the artifact manifest and verifier. They are complete raw exploratory cohorts but are currently `Raw-only`.
5. Do not infer formal synergy between prototype and basis regularization from separate comparisons with KD. A paired joint-versus-component contrast would be required for that claim.
6. Do not generalize these CUB200 adapter results to other datasets, architectures or tasks without corresponding paired validation evidence.

## Integrity verification

- The raw adapter file contains 240 unique `(rank, condition, seed)` records: 3 ranks, 8 conditions including KD, and 10 seeds per group. No group has a missing or duplicate seed.
- `configs.json` contains 24 normalized configurations. Recomputing each canonical SHA-256 matches `normalized_sha256`, and every raw row points to the matching configuration digest.
- Running `python3 scripts/verify_paper_artifacts.py` completed successfully with: `VERIFICATION PASSED: checksums, cohorts, configurations, and reported values agree.` The verifier checks the 11 manifest-declared adapter treatment groups and reproduces their reported rounded values.
- The additional ten `Raw-only` treatment groups were recomputed from the same seed-level file and passed the extended completeness/configuration checks above, but they are not covered by `manifest.json`'s reported-condition list.
