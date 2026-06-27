"""Dark Experience Replay (DER++).

Buzzega, P. et al. "Dark experience for general continual learning:
a strong, simple baseline." NeurIPS 2020, 33:15920-15930.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .base import BaseContinualLearner


class ReplayBuffer:
    """Fixed-size ring buffer storing (input, label, logits) tuples."""

    def __init__(self, capacity: int, input_shape: tuple, n_classes: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.count = 0
        self.ptr = 0

        self.inputs = torch.zeros(capacity, *input_shape, device="cpu")
        self.labels = torch.zeros(capacity, dtype=torch.long, device="cpu")
        self.logits = torch.zeros(capacity, n_classes, device="cpu")

    def add(self, x: torch.Tensor, y: torch.Tensor, logits: torch.Tensor):
        n_out = logits.shape[-1]
        if n_out > self.logits.shape[-1]:
            old = self.logits
            self.logits = torch.zeros(self.capacity, n_out, device="cpu")
            self.logits[:, :old.shape[-1]] = old
        batch_size = x.size(0)
        for i in range(batch_size):
            self.inputs[self.ptr] = x[i].cpu()
            self.labels[self.ptr] = y[i].cpu()
            self.logits[self.ptr, :n_out] = logits[i].detach().cpu()
            self.ptr = (self.ptr + 1) % self.capacity
            self.count = min(self.count + 1, self.capacity)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = np.random.choice(self.count, size=min(batch_size, self.count), replace=False)
        return (
            self.inputs[idx].to(self.device),
            self.labels[idx].to(self.device),
            self.logits[idx].to(self.device),
        )

    def __len__(self):
        return self.count


class DERPlusPlus(BaseContinualLearner):
    """DER++ combines experience replay with dark knowledge (stored logits) distillation."""

    def __init__(self, model: nn.Module, device: torch.device, config: dict):
        super().__init__(model, device)
        self.alpha = config.get("alpha", 0.5)
        self.beta = config.get("beta", 0.5)
        self.buffer_size = config.get("buffer_size", 2000)
        self.buffer: ReplayBuffer | None = None

    def _ensure_buffer(self, x: torch.Tensor, logits: torch.Tensor):
        if self.buffer is None:
            input_shape = x.shape[1:]
            n_classes = logits.shape[-1]
            self.buffer = ReplayBuffer(
                self.buffer_size, input_shape, n_classes, self.device
            )

    def training_step(self, x, y, task_id):
        logits = self.model(x)
        self._ensure_buffer(x, logits)
        ce_loss = F.cross_entropy(logits, y)

        loss = ce_loss

        if len(self.buffer) > 0:
            buf_x, buf_y, buf_logits = self.buffer.sample(x.size(0))
            buf_pred = self.model(buf_x)

            n_buf = buf_logits.shape[-1]
            loss_mse = self.alpha * F.mse_loss(buf_pred[:, :n_buf], buf_logits)
            loss_ce_buf = self.beta * F.cross_entropy(buf_pred, buf_y)
            loss = ce_loss + loss_mse + loss_ce_buf

        self.buffer.add(x, y, logits)

        return loss, logits

    def after_task(self, task_id: int):
        pass
