"""Native PyTorch DDP adapter for the pinned PLOP source."""

import torch.distributed as distributed
from torch.nn.parallel import DistributedDataParallel as _TorchDDP


class DistributedDataParallel(_TorchDDP):
    """Accept PLOP's Apex-only ``delay_allreduce`` argument.

    PLOP initializes NCCL before constructing this class.  Using the module's
    existing CUDA device keeps the expected one-process-per-GPU topology.
    """

    def __init__(self, module, delay_allreduce=False, **kwargs):
        del delay_allreduce
        if not distributed.is_initialized():
            raise RuntimeError("PLOP must initialize torch.distributed before DDP")
        parameter = next(module.parameters(), None)
        if parameter is not None and parameter.device.type == "cuda":
            device_id = parameter.device.index
            kwargs.setdefault("device_ids", [device_id])
            kwargs.setdefault("output_device", device_id)
        super().__init__(module, **kwargs)
