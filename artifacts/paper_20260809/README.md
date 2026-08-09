# Locked direct-SSR confirmation records (2026-08-09)

This directory contains the compact, paired records used for the latest
manuscript update. The records were generated before the manuscript-facing
endpoints were selected for display.

## Manuscript-facing confirmation

`pet_vit_lora_confirmation.json` compares task-only rank-16 LoRA with the same
training plus SSR on 30 paired Oxford-IIIT Pet streams. The four prespecified
manuscript endpoints have favorable 95% paired t intervals:

| Endpoint | Favorable paired change | 95% CI | Favorable seeds |
| --- | ---: | ---: | ---: |
| Average accuracy | +0.0723 points | [0.0183, 0.1262] | 22/30 |
| Average-forgetting reduction | +0.2533 points | [0.0079, 0.4987] | 21/30 |
| Effective-rank increase | +0.6052 | [0.5319, 0.6784] | 30/30 |
| Prototype-overlap reduction | +0.01049 | [0.00922, 0.01176] | 30/30 |

The record also contains CIL-last accuracy. Its interval crosses zero, so it is
retained for auditability but is not used as a positive manuscript endpoint.

## Boundary audit

`segmentation_direct_confirmation.json` contains the locked direct-SSR
segmentation follow-up. Its primary task-performance intervals do not exclude
zero on CUB-200, Oxford-IIIT Pet, or Oxford Flowers102. These records are kept
public so that the positive complete-objective segmentation transfer in the
paper is not mistaken for a confirmed SSR-only endpoint effect.

Run the independent aggregate check from the repository root:

```bash
python3 scripts/verify_paper_20260809.py
```

The verifier recomputes paired means, confidence intervals and favorable-seed
counts from the per-seed rows. It also checks that the nonconfirmatory
segmentation audit and secondary Pet endpoint remain labeled as boundaries.
`checksums.sha256` locks the three imported cluster summaries before those
statistical checks are run.
