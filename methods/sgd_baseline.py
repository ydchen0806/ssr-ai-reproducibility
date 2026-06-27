"""SGD Baseline (Fine-tuning / Naive sequential training).

Lower bound for continual learning — no anti-forgetting mechanism at all.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseContinualLearner


class SGDBaseline(BaseContinualLearner):

    def __init__(self, model: nn.Module, device: torch.device, config: dict):
        super().__init__(model, device)

    def training_step(self, x, y, task_id):
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        return loss, logits

    def after_task(self, task_id: int):
        pass
