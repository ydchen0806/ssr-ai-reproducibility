"""Synaptic Intelligence (SI).

Zenke, F., Poole, B., Ganguli, S. "Continual Learning Through Synaptic Intelligence."
ICML 2017, PMLR vol.70, pp.3987-3995.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseContinualLearner


class SI(BaseContinualLearner):

    def __init__(self, model: nn.Module, device: torch.device, config: dict):
        super().__init__(model, device)
        self.lambda_si = config.get("lambda_si", 1.0)
        self.xi = config.get("xi", 0.1)

        self.prev_params: dict[str, torch.Tensor] = {}
        self.omega: dict[str, torch.Tensor] = {}

        # Running accumulation within a task
        self._w: dict[str, torch.Tensor] = {}
        self._init_params: dict[str, torch.Tensor] = {}

        self._initialized = False

    def _init_tracking(self):
        """Snapshot parameters at the start of a task for path integral computation."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self._init_params[name] = param.data.clone()
                if name not in self._w:
                    self._w[name] = torch.zeros_like(param)
        self._initialized = True

    def training_step(self, x, y, task_id):
        if not self._initialized:
            self._init_tracking()

        logits = self.model(x)
        ce_loss = F.cross_entropy(logits, y)
        si_loss = self._si_penalty()
        loss = ce_loss + self.lambda_si * si_loss

        return loss, logits

    def train_task(self, task_id: int, train_set, training_config: dict):
        self._initialized = False
        self._init_tracking()

        # Override to accumulate gradient * delta after each optimizer step
        self._setup_grad_hooks()
        super().train_task(task_id, train_set, training_config)
        self._remove_grad_hooks()

    def _setup_grad_hooks(self):
        """Register hooks to track parameter changes for path integral."""
        self._hooks = []
        self._prev_step_params = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

    def _remove_grad_hooks(self):
        self._hooks = []

    def _accumulate_w(self):
        """Called after each optimizer step to accumulate contribution scores."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self._prev_step_params:
                delta = param.data - self._prev_step_params[name]
                if param.grad is not None:
                    # w_k += -grad * delta (gradient points towards loss increase)
                    self._w[name] += (-param.grad.data * delta)
                self._prev_step_params[name] = param.data.clone()

    def after_task(self, task_id: int):
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            delta = param.data - self._init_params.get(name, param.data)
            importance = self._w.get(name, torch.zeros_like(param)) / (delta.pow(2) + self.xi)
            importance = torch.clamp(importance, min=0.0)

            if name in self.omega:
                self.omega[name] += importance
            else:
                self.omega[name] = importance

        self.prev_params = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        # Reset per-task accumulators
        self._w = {
            name: torch.zeros_like(param)
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

    def _si_penalty(self) -> torch.Tensor:
        if not self.prev_params:
            return torch.tensor(0.0, device=self.device)

        penalty = torch.tensor(0.0, device=self.device)
        for name, param in self.model.named_parameters():
            if name in self.omega and name in self.prev_params:
                penalty += (
                    self.omega[name] * (param - self.prev_params[name]).pow(2)
                ).sum()
        return penalty
