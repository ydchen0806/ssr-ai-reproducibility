# Verified checkpoint downloads

Manuscript figures display three to five paired seeds per primary
comparison (see `BEST_SEEDS.md`). The links below are those displayed VOC seeds,
the Figure 6c example, and the CUB response-map pair used in Figure 6f.

[Hugging Face model repository](https://huggingface.co/cyd0806/ssr-ai-checkpoints)

These are inference weights; optimizer states are not included.
`scripts/load_visual_checkpoint.py` preserves tensor groups and can remove
the uniform VOC DDP prefix when loading an unwrapped model.
The [KE persistence audit](ke_checkpoint_audit.md) explains which historical
runs exported evaluation records without trained weights.

```bash
hf download cyd0806/ssr-ai-checkpoints \
  --local-dir checkpoints/ssr-ai
```

Original main-text logs and per-seed result files:
https://huggingface.co/cyd0806/ssr-ai-checkpoints/tree/main/main_text_experiment_records

Displayed VOC seeds: `12401, 12405, 12406, 12409, 12410`.

| Cohort | Seed | Arm | Checkpoint |
| --- | --- | --- | --- |
| voc_primary | 12401 | lowrank | [safetensors](https://huggingface.co/cyd0806/ssr-ai-checkpoints/resolve/main/voc_primary/seed_12401/lowrank_stage10.safetensors) |
| voc_primary | 12405 | lowrank | [safetensors](https://huggingface.co/cyd0806/ssr-ai-checkpoints/resolve/main/voc_primary/seed_12405/lowrank_stage10.safetensors) |
| voc_primary | 12406 | lowrank | [safetensors](https://huggingface.co/cyd0806/ssr-ai-checkpoints/resolve/main/voc_primary/seed_12406/lowrank_stage10.safetensors) |
| voc_primary | 12409 | lowrank | [safetensors](https://huggingface.co/cyd0806/ssr-ai-checkpoints/resolve/main/voc_primary/seed_12409/lowrank_stage10.safetensors) |
| voc_primary | 12410 | lowrank | [safetensors](https://huggingface.co/cyd0806/ssr-ai-checkpoints/resolve/main/voc_primary/seed_12410/lowrank_stage10.safetensors) |
| voc_primary | 12401 | lowrank_ssr | [safetensors](https://huggingface.co/cyd0806/ssr-ai-checkpoints/resolve/main/voc_primary/seed_12401/lowrank_ssr_stage10.safetensors) |
| voc_primary | 12405 | lowrank_ssr | [safetensors](https://huggingface.co/cyd0806/ssr-ai-checkpoints/resolve/main/voc_primary/seed_12405/lowrank_ssr_stage10.safetensors) |
| voc_primary | 12406 | lowrank_ssr | [safetensors](https://huggingface.co/cyd0806/ssr-ai-checkpoints/resolve/main/voc_primary/seed_12406/lowrank_ssr_stage10.safetensors) |
| voc_primary | 12409 | lowrank_ssr | [safetensors](https://huggingface.co/cyd0806/ssr-ai-checkpoints/resolve/main/voc_primary/seed_12409/lowrank_ssr_stage10.safetensors) |
| voc_primary | 12410 | lowrank_ssr | [safetensors](https://huggingface.co/cyd0806/ssr-ai-checkpoints/resolve/main/voc_primary/seed_12410/lowrank_ssr_stage10.safetensors) |
| voc_replication | 13401 | lowrank | [safetensors](https://huggingface.co/cyd0806/ssr-ai-checkpoints/resolve/main/voc_replication/seed_13401/lowrank_stage10.safetensors) |
| voc_replication | 13401 | lowrank_ssr | [safetensors](https://huggingface.co/cyd0806/ssr-ai-checkpoints/resolve/main/voc_replication/seed_13401/lowrank_ssr_stage10.safetensors) |
| cub_rank32_response | 8411 | kd | [safetensors](https://huggingface.co/cyd0806/ssr-ai-checkpoints/resolve/main/cub_rank32/kd_seed8411.safetensors) |
| cub_rank32_response | 8411 | kd_ssr | [safetensors](https://huggingface.co/cyd0806/ssr-ai-checkpoints/resolve/main/cub_rank32/kd_ssr_seed8411.safetensors) |
