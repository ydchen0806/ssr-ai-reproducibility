"""GDumb: Greedy Sampler and Dumb Learner.

Prabhu, A. et al. "GDumb: A Simple Approach that Questions Our Progress
in Continual Learning." ECCV 2020, pp. 524-540.

Key idea: greedily store balanced samples in a fixed buffer, then train
from scratch on the buffer alone at evaluation time. Surprisingly strong
baseline that questions the value of complex CL algorithms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

from .base import BaseContinualLearner


class BalancedBuffer:
    """Class-balanced greedy buffer via reservoir sampling per class."""

    def __init__(self, capacity: int, input_shape: tuple, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.input_shape = input_shape
        self.per_class: dict[int, list] = defaultdict(list)

    def add_batch(self, x: torch.Tensor, y: torch.Tensor):
        bs = x.size(0)
        for i in range(bs):
            label = y[i].item()
            self.per_class[label].append(x[i].cpu())
        self._rebalance()

    def _rebalance(self):
        n_cls = len(self.per_class)
        if n_cls == 0:
            return
        per = self.capacity // n_cls
        for cls_id in self.per_class:
            if len(self.per_class[cls_id]) > per:
                idx = np.random.choice(len(self.per_class[cls_id]), size=per, replace=False)
                self.per_class[cls_id] = [self.per_class[cls_id][i] for i in idx]

    def get_all(self):
        xs, ys = [], []
        for cls_id, samples in self.per_class.items():
            for s in samples:
                xs.append(s)
                ys.append(cls_id)
        if not xs:
            return None, None
        return torch.stack(xs), torch.tensor(ys, dtype=torch.long)

    def __len__(self):
        return sum(len(v) for v in self.per_class.values())


class GDumb(BaseContinualLearner):
    """GDumb: store balanced samples, retrain from scratch at eval."""

    def __init__(self, model: nn.Module, device: torch.device, config: dict):
        super().__init__(model, device)
        self.buffer_size = config.get("buffer_size", 2000)
        self.gdumb_epochs = config.get("gdumb_epochs", 50)
        self.buffer: BalancedBuffer | None = None
        self._initial_state = None

    def training_step(self, x, y, task_id):
        self._ensure_buffer(x)
        self.buffer.add_batch(x, y)
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        return loss, logits

    def _ensure_buffer(self, x):
        if self.buffer is None:
            self.buffer = BalancedBuffer(self.buffer_size, x.shape[1:], self.device)
        if self._initial_state is None:
            self._initial_state = {k: v.clone() for k, v in self.model.state_dict().items()}

    def train_task(self, task_id, train_set, training_config):
        """GDumb: only accumulate data, defer training to after_task."""
        self.current_task = task_id
        loader = torch.utils.data.DataLoader(
            train_set, batch_size=training_config.get("batch_size", 256),
            shuffle=True, num_workers=0, pin_memory=True,
        )
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            self._ensure_buffer(x)
            self.buffer.add_batch(x, y)

    def after_task(self, task_id: int):
        """Retrain model from scratch on buffer contents."""
        if self.buffer is None or len(self.buffer) == 0:
            return
        if self._initial_state is not None:
            self.model.load_state_dict(self._initial_state)

        buf_x, buf_y = self.buffer.get_all()
        if buf_x is None:
            return

        dataset = torch.utils.data.TensorDataset(buf_x, buf_y)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=128, shuffle=True, drop_last=False,
        )

        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.gdumb_epochs)

        self.model.train()
        for epoch in range(self.gdumb_epochs):
            for bx, by in loader:
                bx, by = bx.to(self.device), by.to(self.device)
                logits = self.model(bx)
                loss = F.cross_entropy(logits, by)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                optimizer.step()
            scheduler.step()
