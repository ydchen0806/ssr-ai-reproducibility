"""Learning without Forgetting (LwF).

Li, Z. & Hoiem, D. "Learning without forgetting."
IEEE TPAMI, 40(12):2935-2947, 2018.
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseContinualLearner


class LwF(BaseContinualLearner):
    """Knowledge distillation-based continual learning.

    After each task, a frozen copy of the model is stored as the teacher.
    During new task training, a KL divergence loss constrains the current
    model's outputs on new data to remain close to the teacher's outputs.
    """

    def __init__(self, model: nn.Module, device: torch.device, config: dict):
        super().__init__(model, device)
        self.lambda_lwf = config.get("lambda_lwf", 1.0)
        self.temperature = config.get("temperature", 2.0)

        self.teacher: nn.Module | None = None

    def training_step(self, x, y, task_id):
        logits = self.model(x)
        ce_loss = F.cross_entropy(logits, y)

        distill_loss = torch.tensor(0.0, device=self.device)
        if self.teacher is not None:
            with torch.no_grad():
                teacher_logits = self.teacher(x)
            distill_loss = self._distillation_loss(logits, teacher_logits)

        loss = ce_loss + self.lambda_lwf * distill_loss
        return loss, logits

    def after_task(self, task_id: int):
        self.teacher = copy.deepcopy(self.model)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

    def _distillation_loss(self, student_logits: torch.Tensor,
                           teacher_logits: torch.Tensor) -> torch.Tensor:
        T = self.temperature
        student_soft = F.log_softmax(student_logits / T, dim=1)
        teacher_soft = F.softmax(teacher_logits / T, dim=1)
        return F.kl_div(student_soft, teacher_soft, reduction="batchmean") * (T * T)
