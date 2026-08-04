"""SSR LoRA: SSR regularization for parameter-efficient continual learning.

Applies SSR spatial regularization to LoRA adapter weights during sequential
fine-tuning of pre-trained ViT models. Inspired by EWC-LoRA (ICLR 2026) but
replaces the global Fisher computation with the local, biologically plausible
SSR spatial constraint.

Biological rationale:
  In the brain, pre-trained low-level features (V1) are left unconstrained,
  while higher-order representations (temporal cortex) are regulated by
  center-surround suppression. LoRA + SSR mirrors this: the frozen backbone
  acts as V1, while the trainable adapters are the H01-like higher-order
  adaptation layer.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseContinualLearner
from .bioreg import compute_spatial_biocs, compute_spectral_flatness


class LoRALayer(nn.Module):
    """Low-Rank Adaptation layer."""

    def __init__(self, in_features: int, out_features: int, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.rank = rank
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x @ self.lora_A.T @ self.lora_B.T) * self.scaling


class LoRAInjectedLinear(nn.Module):
    """Registered LoRA residual around a frozen linear projection."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        self.lora = LoRALayer(
            base.in_features,
            base.out_features,
            rank=rank,
            alpha=alpha,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora(x)


class BioCsLoRA(BaseContinualLearner):
    """Pre-trained model + LoRA adapters + SSR regularization.

    Freezes the backbone and only trains LoRA adapters + classifier head.
    SSR is applied to the classifier weight matrix and optionally to
    the concatenated LoRA_B matrices (the "output projection" of adapters).
    """

    def __init__(self, model: nn.Module, device: torch.device, config: dict):
        super().__init__(model, device)
        self.lambda_spatial = config.get("lambda_spatial", 0.01)
        self.lambda_adapter = config.get("lambda_adapter", 0.0)
        self.lambda_spectral = config.get("lambda_spectral", 0.0)
        self.A_exc = config.get("A_exc", 1.0)
        self.A_inh = config.get("A_inh", 0.8)
        self.sigma_exc = config.get("sigma_exc", 0.2)
        self.sigma_inh = config.get("sigma_inh", 0.5)
        self.use_seen_mask = config.get("use_seen_mask", True)
        self.lora_rank = config.get("lora_rank", 8)
        self.lora_alpha = config.get("lora_alpha", 16.0)
        self.training_batches = 0
        self.optimizer_steps = 0

        self._seen_classes: set[int] = set()
        self._classifier = self._resolve_classifier()
        if self._classifier is None:
            raise ValueError(
                "BioCsLoRA requires an explicit linear classifier exposed by "
                "model.get_classifier() or a top-level head/fc/classifier attribute."
            )
        self._freeze_backbone()
        self._inject_lora(config)

    def _resolve_classifier(self) -> nn.Linear | None:
        getter = getattr(self.model, "get_classifier", None)
        if callable(getter):
            classifier = getter()
            if isinstance(classifier, nn.Linear):
                return classifier
        for attribute in ("head", "fc", "classifier"):
            classifier = getattr(self.model, attribute, None)
            if isinstance(classifier, nn.Linear):
                return classifier
        return None

    def _freeze_backbone(self):
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self._classifier.parameters():
            param.requires_grad = True

    def _inject_lora(self, config: dict):
        """Replace target projections with registered LoRA residual modules."""
        target_modules = config.get("lora_targets", ["qkv", "proj"])

        targets = [
            (name, module)
            for name, module in self.model.named_modules()
            if name
            and isinstance(module, nn.Linear)
            and any(target.lower() in name.lower() for target in target_modules)
        ]
        self._lora_module_names: list[str] = []
        for name, module in targets:
            parent_name, _, child_name = name.rpartition(".")
            parent = self.model.get_submodule(parent_name) if parent_name else self.model
            parent._modules[child_name] = LoRAInjectedLinear(
                module,
                rank=self.lora_rank,
                alpha=self.lora_alpha,
            ).to(self.device)
            self._lora_module_names.append(name)

        if not self._lora_module_names:
            raise ValueError(
                "No LoRA target modules matched. Check method.lora_targets against "
                "the model's named linear modules."
            )

        n_lora_params = sum(p.numel() for p in self._lora_parameters())
        n_head_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        import logging
        logging.getLogger(__name__).info(
            f"LoRA params: {n_lora_params:,}, Head params: {n_head_params:,}"
        )

    def _lora_modules(self):
        for module in self.model.modules():
            if isinstance(module, LoRAInjectedLinear):
                yield module

    def _lora_parameters(self):
        for module in self._lora_modules():
            yield from module.lora.parameters()

    def _build_optimizer(self, config: dict):
        optimizer = super()._build_optimizer(config)
        optimized = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        missing = [
            name
            for name, parameter in self.model.named_parameters()
            if ".lora." in name and id(parameter) not in optimized
        ]
        if missing:
            raise RuntimeError(f"LoRA parameters missing from optimizer: {missing}")
        return optimizer

    def _get_seen_mask(self, layer: nn.Linear) -> torch.Tensor | None:
        if not self.use_seen_mask or not self._seen_classes:
            return None
        n_out = layer.weight.size(0)
        mask = torch.zeros(n_out, dtype=torch.bool, device=self.device)
        for c in self._seen_classes:
            if c < n_out:
                mask[c] = True
        return mask if mask.sum() > 1 else None

    def _get_classifier(self) -> nn.Linear | None:
        return self._classifier

    def training_step(self, x, y, task_id):
        self.training_batches += 1
        self._seen_classes.update(y.cpu().tolist())
        logits = self.model(x)
        ce_loss = F.cross_entropy(logits, y)

        reg_loss = torch.tensor(0.0, device=self.device)

        classifier = self._get_classifier()
        if classifier is not None:
            mask = self._get_seen_mask(classifier)
            reg_loss = reg_loss + self.lambda_spatial * compute_spatial_biocs(
                classifier.weight, self.A_exc, self.A_inh,
                self.sigma_exc, self.sigma_inh, mask,
            )
            if self.lambda_spectral > 0:
                reg_loss = reg_loss + self.lambda_spectral * compute_spectral_flatness(
                    classifier.weight, mask,
                )

        if self.lambda_adapter > 0:
            adapter_penalties = [
                compute_spatial_biocs(
                    module.lora.lora_B.T,
                    self.A_exc,
                    self.A_inh,
                    self.sigma_exc,
                    self.sigma_inh,
                )
                for module in self._lora_modules()
            ]
            if adapter_penalties:
                reg_loss = reg_loss + self.lambda_adapter * torch.stack(
                    adapter_penalties
                ).mean()

        loss = ce_loss + reg_loss
        return loss, logits

    def _after_optimizer_step(self) -> None:
        self.optimizer_steps += 1

    def after_task(self, task_id: int):
        pass
