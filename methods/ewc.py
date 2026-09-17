"""Elastic Weight Consolidation (EWC).

Kirkpatrick, J. et al. "Overcoming catastrophic forgetting in neural networks."
PNAS, 114(13):3521-3526, 2017. DOI:10.1073/pnas.1611835114
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .base import BaseContinualLearner


class EWC(BaseContinualLearner):

    def __init__(self, model: nn.Module, device: torch.device, config: dict):
        super().__init__(model, device)
        self.lambda_ewc = config.get("lambda_ewc", 400.0)
        self.online = config.get("online", False)
        self.gamma = config.get("gamma", 1.0)
        self.fisher_samples = config.get("fisher_samples", 2000)

        self.prev_params: dict[str, torch.Tensor] = {}
        self.fisher: dict[str, torch.Tensor] = {}
        self._task_train_set = None

    def training_step(self, x, y, task_id):
        logits = self.model(x)
        ce_loss = F.cross_entropy(logits, y)
        ewc_loss = self._ewc_penalty()
        loss = ce_loss + self.lambda_ewc * ewc_loss
        return loss, logits

    def train_task(self, task_id: int, train_set, training_config: dict):
        self._task_train_set = train_set
        super().train_task(task_id, train_set, training_config)

    def after_task(self, task_id: int):
        new_fisher = self._compute_fisher(self._task_train_set)

        if self.online and self.fisher:
            for name in new_fisher:
                if name in self.fisher:
                    self.fisher[name] = (
                        self.gamma * self.fisher[name] + new_fisher[name]
                    )
                else:
                    self.fisher[name] = new_fisher[name]
        else:
            if self.fisher:
                for name in new_fisher:
                    if name in self.fisher:
                        self.fisher[name] = self.fisher[name] + new_fisher[name]
                    else:
                        self.fisher[name] = new_fisher[name]
            else:
                self.fisher = new_fisher

        self.prev_params = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

    def _ewc_penalty(self) -> torch.Tensor:
        if not self.prev_params:
            return torch.tensor(0.0, device=self.device)

        penalty = torch.tensor(0.0, device=self.device)
        for name, param in self.model.named_parameters():
            if name in self.fisher and name in self.prev_params:
                penalty += (
                    self.fisher[name] * (param - self.prev_params[name]).pow(2)
                ).sum()
        return penalty

    def _compute_fisher(self, train_set, n_samples: int = None) -> dict[str, torch.Tensor]:
        """Compute diagonal Fisher information via empirical Fisher (log-likelihood gradients)."""
        n_samples = n_samples or self.fisher_samples
        fisher = {
            name: torch.zeros_like(param)
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        self.model.eval()
        loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=2)
        count = 0

        for x, y in loader:
            if count >= n_samples:
                break
            x, y = x.to(self.device), y.to(self.device)
            batch_size = x.size(0)

            self.model.zero_grad()
            logits = self.model(x)
            log_probs = F.log_softmax(logits, dim=1)

            # Empirical Fisher: use true labels
            nll = F.nll_loss(log_probs, y)
            nll.backward()

            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.data.pow(2) * batch_size

            count += batch_size

        for name in fisher:
            fisher[name] /= max(count, 1)

        self.model.train()
        return fisher
