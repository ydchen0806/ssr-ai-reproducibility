# Checkpoint publication status

The September 11 release has exported and exactly tensor-verified 42 inference
checkpoints (9,308,699,040 bytes): 20 for the VOC main cohort, 20 for the independent
VOC replication, and the two CUB rank-32 visualization models. Both treatment arms
and all final-stage VOC seeds are retained.

**Upload is pending Hugging Face write access.** The authenticated credential
returned HTTP 403 when creating a model repository. No downloadable repository
or successful archive deposition is claimed here.

Provenance and SHA-256 digests are in
[`export_manifest.json`](../artifacts/manuscript_20260911/checkpoints/export_manifest.json).
Exported weights omit optimizer/scheduler state and support inference, not an
interruption-exact training resume. Use `scripts/load_visual_checkpoint.py` to
read the tensor groups and `model.load_state_dict(..., strict=True)` with the
matching architecture.

The historical WikiRecent LoRA run directories audited for this release contain
results and stream manifests but no trained adapter files. Their weights require
a frozen-protocol rerun. The runner now accepts `--checkpoint-dir`; this saves
and verifies LoRA A/B tensors at every declared evaluation checkpoint. It does
not create a retroactive checkpoint for a completed historical run.

After an authorized `hf auth login`, the release operator can run:

```bash
python scripts/publish_checkpoint_release.py \
  --folder "$SSR_CHECKPOINT_EXPORT_DIR" \
  --repo-id "$SSR_HF_MODEL_REPO" \
  --index-output artifacts/manuscript_20260911/checkpoints/hf_index.json \
  --links-output docs/checkpoint_downloads.md
```

The command verifies local hashes, uploads only the listed model files, checks
remote LFS SHA-256 digests, and only then replaces this page with real pinned
download links. Commit the generated index and this page to GitHub afterwards.
