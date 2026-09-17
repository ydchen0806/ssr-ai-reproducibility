"""ER-ACE: Experience Replay with Asymmetric Cross-Entropy.

Caccia, L. et al. "New Insights on Reducing Abrupt Representation Change
in Online Continual Learning." ICLR 2022.

Key idea: current-batch CE is computed only over classes present in the
current task (asymmetric masking), while buffer samples use all seen classes.
This reduces representation drift from the output-layer gradient of unseen
classes pulling features toward them.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .base import BaseContinualLearner


class SimpleBuffer:
    """Reservoir sampling buffer for (input, label)."""

    def __init__(self, capacity: int, input_shape: tuple, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.count = 0
        self.ptr = 0
        self.inputs = torch.zeros(capacity, *input_shape, device="cpu")
        self.labels = torch.zeros(capacity, dtype=torch.long, device="cpu")

    def add(self, x: torch.Tensor, y: torch.Tensor):
        bs = x.size(0)
        for i in range(bs):
            if self.count < self.capacity:
                self.inputs[self.ptr] = x[i].cpu()
                self.labels[self.ptr] = y[i].cpu()
                self.ptr += 1
                self.count += 1
            else:
                j = np.random.randint(0, self.count + 1)
                if j < self.capacity:
                    self.inputs[j] = x[i].cpu()
                    self.labels[j] = y[i].cpu()
                self.count += 1

    def sample(self, batch_size: int):
        n = min(batch_size, min(self.count, self.capacity))
        idx = np.random.choice(min(self.count, self.capacity), size=n, replace=False)
        return self.inputs[idx].to(self.device), self.labels[idx].to(self.device)

    def __len__(self):
        return min(self.count, self.capacity)


class ERACE(BaseContinualLearner):
    """ER-ACE: asymmetric CE on current batch, standard CE on buffer."""

    def __init__(self, model: nn.Module, device: torch.device, config: dict):
        super().__init__(model, device)
        self.buffer_size = config.get("buffer_size", 2000)
        self.buffer: SimpleBuffer | None = None
        self._seen_classes: set[int] = set()
        self._current_task_classes: set[int] = set()

    def _ensure_buffer(self, x: torch.Tensor):
        if self.buffer is None:
            self.buffer = SimpleBuffer(self.buffer_size, x.shape[1:], self.device)

    def train_task(self, task_id, train_set, training_config):
        self._current_task_classes = set()
        for _, y in torch.utils.data.DataLoader(train_set, batch_size=512, num_workers=0):
            self._current_task_classes.update(y.tolist())
        self._seen_classes.update(self._current_task_classes)
        super().train_task(task_id, train_set, training_config)

    def training_step(self, x, y, task_id):
        self._ensure_buffer(x)
        logits = self.model(x)

        # Asymmetric CE: mask logits to only current-task classes
        mask = torch.full_like(logits, float("-inf"))
        for c in self._current_task_classes:
            if c < mask.size(1):
                mask[:, c] = 0.0
        masked_logits = logits + mask
        loss_ce = F.cross_entropy(masked_logits, y)

        loss = loss_ce

        if len(self.buffer) > 0:
            buf_x, buf_y = self.buffer.sample(x.size(0))
            buf_logits = self.model(buf_x)
            # Standard CE on buffer (all seen classes visible)
            loss_buf = F.cross_entropy(buf_logits, buf_y)
            loss = loss_ce + loss_buf

        self.buffer.add(x, y)
        return loss, logits

    def after_task(self, task_id: int):
        pass
