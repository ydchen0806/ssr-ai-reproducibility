"""Fixed-KD scaffold with one explicitly selected additive regularizer."""

from __future__ import annotations

import copy
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .base import BaseContinualLearner
from .bioreg import compute_spatial_biocs


logger = logging.getLogger(__name__)
REGULARIZERS = {"none", "ewc", "mas", "si", "ssr"}


class KDMatched(BaseContinualLearner):
    """One LwF-style KD scaffold shared by KD/EWC/MAS/SI/SSR conditions."""

    def __init__(self, model: nn.Module, device: torch.device, config: dict):
        super().__init__(model, device)
        method_name = str(config.get("name", "kd")).lower()
        inferred = {
            "kd": "none",
            "kd_ewc": "ewc",
            "kd_mas": "mas",
            "kd_si": "si",
            "kd_ssr": "ssr",
        }.get(method_name)
        explicit_regularizer = config.get("regularizer")
        if inferred is not None and explicit_regularizer is not None:
            explicit_regularizer = str(explicit_regularizer).lower()
            if explicit_regularizer != inferred:
                raise ValueError(
                    f"Contradictory matched-KD config: method name {method_name!r} "
                    f"requires regularizer={inferred!r}, got {explicit_regularizer!r}"
                )
        self.regularizer = str(explicit_regularizer or inferred or "none").lower()
        if self.regularizer not in REGULARIZERS:
            raise ValueError(
                f"Unknown matched-KD regularizer={self.regularizer!r}; "
                f"choose from {sorted(REGULARIZERS)}"
            )

        self.lambda_kd = float(config.get("lambda_kd", config.get("lambda_distill", 5.0)))
        self.temperature = float(config.get("temperature", 2.0))
        self.lambda_ewc = float(config.get("lambda_ewc", 400.0))
        self.lambda_mas = float(config.get("lambda_mas", 1.0))
        self.lambda_si = float(config.get("lambda_si", 1.0))
        self.lambda_ssr = float(config.get("lambda_ssr", config.get("lambda_spatial", 0.005)))
        self.fisher_samples = int(config.get("fisher_samples", 2000))
        self.importance_samples = int(config.get("importance_samples", 2000))
        self.xi = float(config.get("xi", 0.1))

        self.a_exc = float(config.get("A_exc", 1.0))
        self.a_inh = float(config.get("A_inh", 0.8))
        self.sigma_exc = float(config.get("sigma_exc", 0.16))
        self.sigma_inh = float(config.get("sigma_inh", 0.45))
        if self.sigma_exc <= 0 or self.sigma_inh <= 0:
            raise ValueError("SSR sigma values must be positive")

        self.teacher: nn.Module | None = None
        self.prev_params: dict[str, torch.Tensor] = {}
        self.importance: dict[str, torch.Tensor] = {}
        self._task_train_set = None
        self._seen_classes: set[int] = set()
        self._w: dict[str, torch.Tensor] = {}
        self._init_params: dict[str, torch.Tensor] = {}
        self._prev_step_params: dict[str, torch.Tensor] = {}
        self._run_seed = 0
        self._task_id = 0

        linears = [(name, module) for name, module in model.named_modules() if isinstance(module, nn.Linear)]
        targets = config.get("ssr_targets", ["classifier"])
        self._ssr_layers = linears[-1:] if "classifier" in set(targets) else linears
        logger.info(
            "KDMatched | regularizer=%s lambda_kd=%s T=%s",
            self.regularizer,
            self.lambda_kd,
            self.temperature,
        )

    def _kd_loss(self, logits: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if self.teacher is None or self.lambda_kd <= 0:
            return logits.sum() * 0.0
        with torch.no_grad():
            teacher_logits = self.teacher(x)
        temperature = self.temperature
        return F.kl_div(
            F.log_softmax(logits / temperature, dim=1),
            F.softmax(teacher_logits / temperature, dim=1),
            reduction="batchmean",
        ) * (temperature * temperature)

    def _quadratic_penalty(self) -> torch.Tensor:
        if not self.prev_params or not self.importance:
            return torch.tensor(0.0, device=self.device)
        penalty = torch.tensor(0.0, device=self.device)
        for name, parameter in self.model.named_parameters():
            if name in self.prev_params and name in self.importance:
                penalty = penalty + (
                    self.importance[name] * (parameter - self.prev_params[name]).pow(2)
                ).sum()
        return penalty

    def _ssr_penalty(self) -> torch.Tensor:
        penalty = torch.tensor(0.0, device=self.device)
        for _, layer in self._ssr_layers:
            mask = torch.zeros(layer.weight.size(0), dtype=torch.bool, device=self.device)
            for class_id in self._seen_classes:
                if class_id < mask.numel():
                    mask[class_id] = True
            seen_mask = mask if int(mask.sum()) > 1 else None
            penalty = penalty + compute_spatial_biocs(
                layer.weight.float(),
                self.a_exc,
                self.a_inh,
                self.sigma_exc,
                self.sigma_inh,
                seen_mask,
            )
        return penalty

    def training_step(self, x, y, task_id):
        self._seen_classes.update(int(value) for value in y.detach().cpu().tolist())
        logits = self.model(x)
        loss = F.cross_entropy(logits, y) + self.lambda_kd * self._kd_loss(logits, x)
        if self.regularizer == "ewc":
            loss = loss + self.lambda_ewc * self._quadratic_penalty()
        elif self.regularizer == "mas":
            loss = loss + self.lambda_mas * self._quadratic_penalty()
        elif self.regularizer == "si":
            loss = loss + self.lambda_si * self._quadratic_penalty()
        elif self.regularizer == "ssr":
            with torch.amp.autocast("cuda", enabled=False):
                loss = loss + self.lambda_ssr * self._ssr_penalty()
        return loss, logits

    def train_task(self, task_id: int, train_set, training_config: dict):
        self._task_train_set = train_set
        self._run_seed = int(training_config.get("run_seed", 0))
        self._task_id = int(task_id)
        if self.regularizer == "si":
            self._init_params = {
                name: parameter.detach().clone()
                for name, parameter in self.model.named_parameters()
                if parameter.requires_grad
            }
            self._prev_step_params = {
                name: parameter.detach().clone()
                for name, parameter in self.model.named_parameters()
                if parameter.requires_grad
            }
            for name, parameter in self.model.named_parameters():
                if parameter.requires_grad:
                    self._w.setdefault(name, torch.zeros_like(parameter))
        super().train_task(task_id, train_set, training_config)

    def _accumulate_w(self) -> None:
        if self.regularizer != "si":
            return
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad or name not in self._prev_step_params:
                continue
            delta = parameter.detach() - self._prev_step_params[name]
            if parameter.grad is not None:
                self._w[name] = self._w[name] + (-parameter.grad.detach() * delta)
            self._prev_step_params[name] = parameter.detach().clone()

    def _compute_fisher(self) -> dict[str, torch.Tensor]:
        importance = {
            name: torch.zeros_like(parameter)
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }
        self.model.eval()
        count = 0
        generator = torch.Generator()
        generator.manual_seed(self._run_seed + 1_000_003 + 100_003 * self._task_id)
        for x, y in DataLoader(
            self._task_train_set,
            batch_size=32,
            shuffle=True,
            num_workers=2,
            generator=generator,
        ):
            if count >= self.fisher_samples:
                break
            x, y = x.to(self.device), y.to(self.device)
            self.model.zero_grad(set_to_none=True)
            F.cross_entropy(self.model(x), y).backward()
            for name, parameter in self.model.named_parameters():
                if parameter.requires_grad and parameter.grad is not None:
                    importance[name] += parameter.grad.detach().pow(2) * x.size(0)
            count += x.size(0)
        for name in importance:
            importance[name] /= max(count, 1)
        self.model.train()
        return importance

    def _compute_mas(self) -> dict[str, torch.Tensor]:
        importance = {
            name: torch.zeros_like(parameter)
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }
        self.model.eval()
        count = 0
        generator = torch.Generator()
        generator.manual_seed(self._run_seed + 2_000_003 + 100_003 * self._task_id)
        for x, _ in DataLoader(
            self._task_train_set,
            batch_size=32,
            shuffle=True,
            num_workers=2,
            generator=generator,
        ):
            if count >= self.importance_samples:
                break
            x = x.to(self.device)
            self.model.zero_grad(set_to_none=True)
            self.model(x).pow(2).mean().backward()
            for name, parameter in self.model.named_parameters():
                if parameter.requires_grad and parameter.grad is not None:
                    importance[name] += parameter.grad.detach().abs() * x.size(0)
            count += x.size(0)
        for name in importance:
            importance[name] /= max(count, 1)
        self.model.train()
        return importance

    def _consolidate_si(self) -> dict[str, torch.Tensor]:
        output = {}
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            delta = parameter.detach() - self._init_params[name]
            output[name] = torch.clamp(self._w[name] / (delta.pow(2) + self.xi), min=0.0)
            self._w[name] = torch.zeros_like(parameter)
        return output

    def _merge_importance(self, new_importance: dict[str, torch.Tensor]) -> None:
        for name, value in new_importance.items():
            if name in self.importance:
                self.importance[name] = self.importance[name] + value
            else:
                self.importance[name] = value

    def after_task(self, task_id: int):
        if self.regularizer == "ewc":
            self._merge_importance(self._compute_fisher())
        elif self.regularizer == "mas":
            self._merge_importance(self._compute_mas())
        elif self.regularizer == "si":
            self._merge_importance(self._consolidate_si())
        if self.regularizer in {"ewc", "mas", "si"}:
            self.prev_params = {
                name: parameter.detach().clone()
                for name, parameter in self.model.named_parameters()
                if parameter.requires_grad
            }
        self.teacher = copy.deepcopy(self.model)
        self.teacher.eval()
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)
