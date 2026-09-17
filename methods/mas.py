"""Memory Aware Synapses (MAS).

Aljundi, R. et al. "Memory Aware Synapses: Learning what (not) to forget."
ECCV 2018, pp.144-161. DOI:10.1007/978-3-030-01219-9_9
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .base import BaseContinualLearner


class MAS(BaseContinualLearner):
    """MAS computes importance via sensitivity of the learned function to parameter changes.

    Key property: does NOT require labels — importance is estimated from the
    gradient of the L2 norm of the network output w.r.t. parameters.
    """

    def __init__(self, model: nn.Module, device: torch.device, config: dict):
        super().__init__(model, device)
        self.lambda_mas = config.get("lambda_mas", 1.0)
        self.n_importance_samples = config.get("n_importance_samples", 2000)

        self.prev_params: dict[str, torch.Tensor] = {}
        self.omega: dict[str, torch.Tensor] = {}
        self._task_train_set = None

    def training_step(self, x, y, task_id):
        logits = self.model(x)
        ce_loss = F.cross_entropy(logits, y)
        mas_loss = self._mas_penalty()
        loss = ce_loss + self.lambda_mas * mas_loss
        return loss, logits

    def train_task(self, task_id: int, train_set, training_config: dict):
        self._task_train_set = train_set
        super().train_task(task_id, train_set, training_config)

    def after_task(self, task_id: int):
        new_omega = self._compute_importance(self._task_train_set)

        if self.omega:
            for name in new_omega:
                if name in self.omega:
                    self.omega[name] += new_omega[name]
                else:
                    self.omega[name] = new_omega[name]
        else:
            self.omega = new_omega

        self.prev_params = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

    def _mas_penalty(self) -> torch.Tensor:
        if not self.prev_params:
            return torch.tensor(0.0, device=self.device)

        penalty = torch.tensor(0.0, device=self.device)
        for name, param in self.model.named_parameters():
            if name in self.omega and name in self.prev_params:
                penalty += (
                    self.omega[name] * (param - self.prev_params[name]).pow(2)
                ).sum()
        return penalty

    def _compute_importance(self, train_set) -> dict[str, torch.Tensor]:
        """Estimate parameter importance via output sensitivity (no labels needed)."""
        omega = {
            name: torch.zeros_like(param)
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        self.model.eval()
        loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=2)
        count = 0

        for x, _ in loader:
            if count >= self.n_importance_samples:
                break
            x = x.to(self.device)
            batch_size = x.size(0)

            self.model.zero_grad()
            output = self.model(x)

            # MAS uses L2 norm of the output as the sensitivity measure
            output_norm = output.pow(2).mean()
            output_norm.backward()

            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    omega[name] += param.grad.data.abs() * batch_size

            count += batch_size

        for name in omega:
            omega[name] /= max(count, 1)

        self.model.train()
        return omega
