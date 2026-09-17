# Archived direct-SSR protocols and boundary records (2026-08-09)

This directory is retained for auditability. It is not a source of positive
manuscript endpoints.

## Raw-image ViT-LoRA record: invalidated

`pet_vit_lora_confirmation.json` contains a historical task-only versus
task-plus-SSR Oxford-IIIT Pet cohort. A later code audit found float16 AMP
gradient overflow that prevented parameter updates in both arms. The archived
numbers therefore do **not** support an SSR claim and must not be reused in a
manuscript, figure, or aggregate. The current implementation performs the
similarity calculation outside autocast and the pair validator now requires
finite updates and distinct final-model hashes before accepting a new result.

Use `scripts/run_vit_lora_confirmation_8gpu.sh` with a fresh output directory,
then run `scripts/validate_vit_lora_pairs.py`; see
[`docs/manuscript_reproduction.md`](../../docs/manuscript_reproduction.md) for
the exact commands.

## Segmentation boundary audit

`segmentation_direct_confirmation.json` contains a direct task-only versus
task-plus-SSR segmentation follow-up. Its primary task-performance intervals do
not exclude zero on CUB-200, Oxford-IIIT Pet, or Oxford Flowers102. It remains
public so that complete-objective transfer is not mistaken for a confirmed
SSR-only endpoint effect.

Run the aggregate check from the repository root:

```bash
python3 scripts/verify_paper_20260809.py
```

The verifier recomputes paired means and confidence intervals and confirms the
boundary status. `checksums.sha256` locks the imported cluster summaries before
those checks run.
