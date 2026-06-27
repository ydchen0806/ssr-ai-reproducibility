"""CLS-ER: Complementary Learning System - Experience Replay.

Arani, E. et al. "Learning Fast, Learning Slow: A General Continual Learning
Method based on Complementary Learning System." ICLR 2022.

Key idea: maintains two EMA (exponential moving average) model copies —
a "plastic" (fast) model and a "stable" (slow) model — inspired by the
hippocampal/neocortical systems in CLS theory. The loss combines current CE
with consistency losses toward both EMA models plus buffer replay.
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .base import BaseContinualLearner


class CLSERBuffer:
    """Ring buffer for (input, label)."""

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
            self.inputs[self.ptr] = x[i].cpu()
            self.labels[self.ptr] = y[i].cpu()
            self.ptr = (self.ptr + 1) % self.capacity
            self.count = min(self.count + 1, self.capacity)

    def sample(self, batch_size: int):
        n = min(batch_size, self.count)
        idx = np.random.choice(self.count, size=n, replace=False)
        return self.inputs[idx].to(self.device), self.labels[idx].to(self.device)

    def __len__(self):
        return self.count


class CLSER(BaseContinualLearner):
    """CLS-ER: dual EMA models (plastic + stable) + replay."""

    def __init__(self, model: nn.Module, device: torch.device, config: dict):
        super().__init__(model, device)
        self.buffer_size = config.get("buffer_size", 2000)
        self.ema_alpha_plastic = config.get("ema_alpha_plastic", 0.999)
        self.ema_alpha_stable = config.get("ema_alpha_stable", 0.9999)
        self.reg_weight = config.get("reg_weight", 0.5)

        self.plastic_model = copy.deepcopy(model).to(device)
        self.stable_model = copy.deepcopy(model).to(device)
        for p in self.plastic_model.parameters():
            p.requires_grad_(False)
        for p in self.stable_model.parameters():
            p.requires_grad_(False)

        self.buffer: CLSERBuffer | None = None
        self._step_count = 0

    def _ensure_buffer(self, x: torch.Tensor):
        if self.buffer is None:
            self.buffer = CLSERBuffer(self.buffer_size, x.shape[1:], self.device)

    @torch.no_grad()
    def _update_ema(self):
        for p_main, p_plastic in zip(self.model.parameters(), self.plastic_model.parameters()):
            p_plastic.data.mul_(self.ema_alpha_plastic).add_(p_main.data, alpha=1 - self.ema_alpha_plastic)
        for p_main, p_stable in zip(self.model.parameters(), self.stable_model.parameters()):
            p_stable.data.mul_(self.ema_alpha_stable).add_(p_main.data, alpha=1 - self.ema_alpha_stable)

    def training_step(self, x, y, task_id):
        self._ensure_buffer(x)
        logits = self.model(x)
        loss_ce = F.cross_entropy(logits, y)

        loss = loss_ce

        if len(self.buffer) > 0:
            buf_x, buf_y = self.buffer.sample(x.size(0))

            buf_logits = self.model(buf_x)
            loss_buf_ce = F.cross_entropy(buf_logits, buf_y)

            with torch.no_grad():
                plastic_logits = self.plastic_model(buf_x)
                stable_logits = self.stable_model(buf_x)

            T = 2.0
            p_soft = F.softmax(plastic_logits / T, dim=1)
            s_soft = F.softmax(stable_logits / T, dim=1)
            buf_log_soft = F.log_softmax(buf_logits / T, dim=1)

            loss_plastic = T * T * F.kl_div(buf_log_soft, p_soft, reduction="batchmean")
            loss_stable = T * T * F.kl_div(buf_log_soft, s_soft, reduction="batchmean")

            loss = loss_ce + loss_buf_ce + self.reg_weight * (loss_plastic + loss_stable)

        self.buffer.add(x, y)
        self._update_ema()
        self._step_count += 1

        return loss, logits

    def after_task(self, task_id: int):
        pass
