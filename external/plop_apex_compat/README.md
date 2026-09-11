# PLOP native-DDP compatibility layer

This overlay is added to `PYTHONPATH` before the pinned PLOP checkout. It
replaces only the Apex interfaces used by PLOP and the unavailable
`inplace-abn` CUDA extension. `amp.scale_loss` is deliberately full precision,
and `delay_allreduce` is ignored because native PyTorch DDP performs the
required synchronization. The normalization fallback uses native
`SyncBatchNorm + LeakyReLU`, retaining the original layer computations without
the fused/in-place memory optimization.

Record this overlay's SHA-256 alongside the PLOP commit. Do not describe a
run using it as mixed precision or as fused InPlace-ABN. Before any full
four-node experiment, run a single-node, one-epoch PASCAL VOC smoke test to
validate the dataloader, checkpointing, and distributed launch arguments.
