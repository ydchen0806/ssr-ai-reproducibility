# Verified checkpoint downloads

[Hugging Face model repository](https://huggingface.co/cyd0806/ssr-ai-checkpoints/tree/9260ac4062517ad7ee1fb3b4011285247b256514)

Revision: `9260ac4062517ad7ee1fb3b4011285247b256514`. All 42 file sizes and SHA-256 digests verified.

The repository is public. Unauthenticated downloads of a VOC and a CUB file
also passed SHA-256 checks. These are inference weights; optimizer states are
not included. `scripts/load_visual_checkpoint.py` preserves tensor groups and
can remove the uniform VOC DDP prefix when loading an unwrapped model.
The [KE persistence audit](ke_checkpoint_audit.md) explains which historical
runs exported evaluation records without trained weights.

```bash
hf download cyd0806/ssr-ai-checkpoints \
  --revision 9260ac4062517ad7ee1fb3b4011285247b256514 \
  --local-dir checkpoints/ssr-ai
```

| Cohort | Seed | Arm | Checkpoint |
| --- | --- | --- | --- |
| voc_primary | 12401 | lowrank | [safetensors](https://huggingface.co/cyd0806/ssr-ai-checkpoints/resolve/9260ac4062517ad7ee1fb3b4011285247b256514/voc_primary/seed_12401/lowrank_stage10.safetensors) |
| voc_primary | 12402 | lowrank | [safetensors](https://huggingface.co/cyd0806/ssr-ai-checkpoints/resolve/9260ac4062517ad7ee1fb3b4011285247b256514/voc_primary/seed_12402/lowrank_stage10.safetensors) |
| voc_primary | 12403 | lowrank | [safetensors](https://huggingface.co/cyd0806/ssr-ai-checkpoints/resolve/9260ac4062517ad7ee1fb3b4011285247b256514/voc_primary/seed_12403/lowrank_stage10.safetensors) |
| voc_primary | 12404 | lowrank | [safetensors](https://huggingface.co/cyd0806/ssr-ai-checkpoints/resolve/9260ac4062517ad7ee1fb3b4011285247b256514/voc_primary/seed_12404/lowrank_stage10.safetensors) |
| voc_primary | 12405 | lowrank | [safetensors](https://huggingface.co/cyd0806/ssr-ai-checkpoints/resolve/9260ac4062517ad7ee1fb3b4011285247b256514/voc_primary/seed_12405/lowrank_stage10.safetensors) |
| voc_primary | 12406 | lowrank | [safetensors](https://huggingface.co/cyd0806/ssr-ai-checkpoints/resolve/9260ac4062517ad7ee1fb3b4011285247b256514/voc_primary/seed_12406/lowrank_stage10.safetensors) |
| voc_primary | 12407 | lowrank | [safetensors](https://huggingface.co/cyd0806/ssr-ai-checkpoints/resolve/9260ac4062517ad7ee1fb3b4011285247b256514/voc_primary/seed_12407/lowrank_stage10.safetensors) |
| voc_primary | 12408 | lowrank | [safetensors](https://huggingface.co/cyd0806/ssr-ai-checkpoints/resolve/9260ac4062517ad7ee1fb3b4011285247b256514/voc_primary/seed_12408/lowrank_stage10.safetensors) |
| voc_primary | 12409 | lowrank | [safetensors](https://huggingface.co/cyd0806/ssr-ai-checkpoints/resolve/9260ac4062517ad7ee1fb3b4011285247b256514/voc_primary/seed_12409/lowrank_stage10.safetensors) |
| voc_primary | 12410 | lowrank | [safetensors](https://huggingface.co/cyd0806/ssr-ai-checkpoints/resolve/9260ac4062517ad7ee1fb3b4011285247b256514/voc_primary/seed_12410/lowrank_stage10.safetensors) |
| voc_primary | 12401 | lowrank_ssr | [safetensors](https://huggingface.co/cyd0806/ssr-ai-checkpoints/resolve/9260ac4062517ad7ee1fb3b4011285247b256514/voc_primary/seed_12401/lowrank_ssr_stage10.safetensors) |
| voc_primary | 12402 | lowrank_ssr | [safetensors](https://huggingface.co/cyd0806/ssr-ai-checkpoints/resolve/9260ac4062517ad7ee1fb3b4011285247b256514/voc_primary/seed_12402/lowrank_ssr_stage10.safetensors) |
| voc_primary | 12403 | lowrank_ssr | [safetensors](https://huggingface.co/cyd0806/ssr-ai-checkpoints/resolve/9260ac4062517ad7ee1fb3b4011285247b256514/voc_primary/seed_12403/lowrank_ssr_stage10.safetensors) |
| voc_primary | 12404 | lowrank_ssr | [safetensors](https://huggingface.co/cyd0806/ssr-ai-checkpoints/resolve/9260ac4062517ad7ee1fb3b4011285247b256514/voc_primary/seed_12404/lowrank_ssr_stage10.safetensors) |
| voc_primary | 12405 | lowrank_ssr | [safetensors](https://huggingface.co/cyd0806/ssr-ai-checkpoints/resolve/9260ac4062517ad7ee1fb3b4011285247b256514/voc_primary/seed_12405/lowrank_ssr_stage10.safetensors) |
| voc_primary | 12406 | lowrank_ssr | [safetensors](https://huggingface.co/cyd0806/ssr-ai-checkpoints/resolve/9260ac4062517ad7ee1fb3b4011285247b256514/voc_primary/seed_12406/lowrank_ssr_stage10.safetensors) |
| voc_primary | 12407 | lowrank_ssr | [safetensors](https://huggingface.co/cyd0806/ssr-ai-checkpoints/resolve/9260ac4062517ad7ee1fb3b4011285247b256514/voc_primary/seed_12407/lowrank_ssr_stage10.safetensors) |
| voc_primary | 12408 | lowrank_ssr | [safetensors](https://huggingface.co/cyd0806/ssr-ai-checkpoints/resolve/9260ac4062517ad7ee1fb3b4011285247b256514/voc_primary/seed_12408/lowrank_ssr_stage10.safetensors) |
| voc_primary | 12409 | lowrank_ssr | [safetensors](https://huggingface.co/cyd0806/ssr-ai-checkpoints/resolve/9260ac4062517ad7ee1fb3b4011285247b256514/voc_primary/seed_12409/lowrank_ssr_stage10.safetensors) |
| voc_primary | 12410 | lowrank_ssr | [safetensors](https://huggingface.co/cyd0806/ssr-ai-checkpoints/resolve/9260ac4062517ad7ee1fb3b4011285247b256514/voc_primary/seed_12410/lowrank_ssr_stage10.safetensors) |
| voc_replication | 13401 | lowrank | [safetensors](https://huggingface.co/cyd0806/ssr-ai-checkpoints/resolve/9260ac4062517ad7ee1fb3b4011285247b256514/voc_replication/seed_13401/lowrank_stage10.safetensors) |
| voc_replication | 13402 | lowrank | [safetensors](https://huggingface.co/cyd0806/ssr-ai-checkpoints/resolve/9260ac4062517ad7ee1fb3b4011285247b256514/voc_replication/seed_13402/lowrank_stage10.safetensors) |
| voc_replication | 13403 | lowrank | [safetensors](https://huggingface.co/cyd0806/ssr-ai-checkpoints/resolve/9260ac4062517ad7ee1fb3b4011285247b256514/voc_replication/seed_13403/lowrank_stage10.safetensors) |
| voc_replication | 13404 | lowrank | [safetensors](https://huggingface.co/cyd0806/ssr-ai-checkpoints/resolve/9260ac4062517ad7ee1fb3b4011285247b256514/voc_replication/seed_13404/lowrank_stage10.safetensors) |
| voc_replication | 13405 | lowrank | [safetensors](https://huggingface.co/cyd0806/ssr-ai-checkpoints/resolve/9260ac4062517ad7ee1fb3b4011285247b256514/voc_replication/seed_13405/lowrank_stage10.safetensors) |
| voc_replication | 13406 | lowrank | [safetensors](https://huggingface.co/cyd0806/ssr-ai-checkpoints/resolve/9260ac4062517ad7ee1fb3b4011285247b256514/voc_replication/seed_13406/lowrank_stage10.safetensors) |
| voc_replication | 13407 | lowrank | [safetensors](https://huggingface.co/cyd0806/ssr-ai-checkpoints/resolve/9260ac4062517ad7ee1fb3b4011285247b256514/voc_replication/seed_13407/lowrank_stage10.safetensors) |
| voc_replication | 13408 | lowrank | [safetensors](https://huggingface.co/cyd0806/ssr-ai-checkpoints/resolve/9260ac4062517ad7ee1fb3b4011285247b256514/voc_replication/seed_13408/lowrank_stage10.safetensors) |
| voc_replication | 13409 | lowrank | [safetensors](https://huggingface.co/cyd0806/ssr-ai-checkpoints/resolve/9260ac4062517ad7ee1fb3b4011285247b256514/voc_replication/seed_13409/lowrank_stage10.safetensors) |
| voc_replication | 13410 | lowrank | [safetensors](https://huggingface.co/cyd0806/ssr-ai-checkpoints/resolve/9260ac4062517ad7ee1fb3b4011285247b256514/voc_replication/seed_13410/lowrank_stage10.safetensors) |
| voc_replication | 13401 | lowrank_ssr | [safetensors](https://huggingface.co/cyd0806/ssr-ai-checkpoints/resolve/9260ac4062517ad7ee1fb3b4011285247b256514/voc_replication/seed_13401/lowrank_ssr_stage10.safetensors) |
| voc_replication | 13402 | lowrank_ssr | [safetensors](https://huggingface.co/cyd0806/ssr-ai-checkpoints/resolve/9260ac4062517ad7ee1fb3b4011285247b256514/voc_replication/seed_13402/lowrank_ssr_stage10.safetensors) |
| voc_replication | 13403 | lowrank_ssr | [safetensors](https://huggingface.co/cyd0806/ssr-ai-checkpoints/resolve/9260ac4062517ad7ee1fb3b4011285247b256514/voc_replication/seed_13403/lowrank_ssr_stage10.safetensors) |
| voc_replication | 13404 | lowrank_ssr | [safetensors](https://huggingface.co/cyd0806/ssr-ai-checkpoints/resolve/9260ac4062517ad7ee1fb3b4011285247b256514/voc_replication/seed_13404/lowrank_ssr_stage10.safetensors) |
| voc_replication | 13405 | lowrank_ssr | [safetensors](https://huggingface.co/cyd0806/ssr-ai-checkpoints/resolve/9260ac4062517ad7ee1fb3b4011285247b256514/voc_replication/seed_13405/lowrank_ssr_stage10.safetensors) |
| voc_replication | 13406 | lowrank_ssr | [safetensors](https://huggingface.co/cyd0806/ssr-ai-checkpoints/resolve/9260ac4062517ad7ee1fb3b4011285247b256514/voc_replication/seed_13406/lowrank_ssr_stage10.safetensors) |
| voc_replication | 13407 | lowrank_ssr | [safetensors](https://huggingface.co/cyd0806/ssr-ai-checkpoints/resolve/9260ac4062517ad7ee1fb3b4011285247b256514/voc_replication/seed_13407/lowrank_ssr_stage10.safetensors) |
| voc_replication | 13408 | lowrank_ssr | [safetensors](https://huggingface.co/cyd0806/ssr-ai-checkpoints/resolve/9260ac4062517ad7ee1fb3b4011285247b256514/voc_replication/seed_13408/lowrank_ssr_stage10.safetensors) |
| voc_replication | 13409 | lowrank_ssr | [safetensors](https://huggingface.co/cyd0806/ssr-ai-checkpoints/resolve/9260ac4062517ad7ee1fb3b4011285247b256514/voc_replication/seed_13409/lowrank_ssr_stage10.safetensors) |
| voc_replication | 13410 | lowrank_ssr | [safetensors](https://huggingface.co/cyd0806/ssr-ai-checkpoints/resolve/9260ac4062517ad7ee1fb3b4011285247b256514/voc_replication/seed_13410/lowrank_ssr_stage10.safetensors) |
| cub_rank32_response | 8411 | kd | [safetensors](https://huggingface.co/cyd0806/ssr-ai-checkpoints/resolve/9260ac4062517ad7ee1fb3b4011285247b256514/cub_rank32/kd_seed8411.safetensors) |
| cub_rank32_response | 8411 | kd_ssr | [safetensors](https://huggingface.co/cyd0806/ssr-ai-checkpoints/resolve/9260ac4062517ad7ee1fb3b4011285247b256514/cub_rank32/kd_ssr_seed8411.safetensors) |
