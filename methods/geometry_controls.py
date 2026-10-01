"""Matched geometry-control baselines for SSR+ ablations.

These controls keep the same continual-learning scaffold as SSR+ where
possible: cross-entropy on the current task, optional logit distillation from a
teacher snapshot, and one additional geometric or norm-based control. They are
intended to answer whether the SSR+ gain is explained by generic
orthogonalization, decorrelation, spectral flattening, center compactness,
supervised contrastive structure, or optimizer weight decay.
"""

from __future__ import annotations

import copy
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseContinualLearner
from .bioreg import compute_spectral_flatness

logger = logging.getLogger(__name__)


def _final_linear(model: nn.Module) -> nn.Linear | None:
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    return linears[-1] if linears else None


def _offdiag(mat: torch.Tensor) -> torch.Tensor:
    n = mat.size(0)
    if n < 2:
        return mat.new_zeros(0)
    return mat[~torch.eye(n, dtype=torch.bool, device=mat.device)]


def cosine_orthogonal_loss(
    weights: torch.Tensor,
    seen_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean squared off-diagonal cosine between classifier prototypes."""
    weights = weights.float()
    if seen_mask is not None:
        weights = weights[seen_mask]
    if weights.size(0) < 2:
        return weights.new_tensor(0.0)
    w = F.normalize(weights, dim=1)
    cos = w @ w.T
    vals = _offdiag(cos)
    return vals.pow(2).mean() if vals.numel() else weights.new_tensor(0.0)


def prototype_decorrelation_loss(
    weights: torch.Tensor,
    seen_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Correlation-style decorrelation after centering each prototype vector."""
    weights = weights.float()
    if seen_mask is not None:
        weights = weights[seen_mask]
    if weights.size(0) < 2 or weights.size(1) < 2:
        return weights.new_tensor(0.0)
    centered = weights - weights.mean(dim=1, keepdim=True)
    centered = F.normalize(centered, dim=1)
    corr = centered @ centered.T
    vals = _offdiag(corr)
    return vals.pow(2).mean() if vals.numel() else weights.new_tensor(0.0)


def supervised_contrastive_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.2,
) -> torch.Tensor:
    """Batch supervised contrastive loss with no replay memory."""
    if features.size(0) < 2:
        return features.new_tensor(0.0)
    z = F.normalize(features.float(), dim=1)
    logits = (z @ z.T) / max(temperature, 1e-6)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    eye = torch.eye(labels.size(0), dtype=torch.bool, device=labels.device)
    positive = labels[:, None].eq(labels[None, :]) & ~eye
    if not positive.any():
        return features.new_tensor(0.0)

    exp_logits = torch.exp(logits).masked_fill(eye, 0.0)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)
    pos_count = positive.sum(dim=1).clamp_min(1)
    per_sample = -(log_prob * positive.float()).sum(dim=1) / pos_count
    valid = positive.any(dim=1)
    return per_sample[valid].mean() if valid.any() else features.new_tensor(0.0)


class GeometryControls(BaseContinualLearner):
    """Matched control family for SSR+.

    Config keys:
        control_type:
            "cosine_orthogonal", "prototype_decorrelation", "spectral",
            "center_loss", "supervised_contrastive", or "none".
        lambda_control: weight of the selected control loss.
        lambda_distill: logit distillation weight.
        temperature: distillation temperature.
        biocs_targets: kept for config compatibility; classifier is used.
        center_lr_scale: optimizer LR multiplier for trainable class centers.
    """

    VALID_CONTROLS = {
        "none",
        "cosine_orthogonal",
        "prototype_decorrelation",
        "spectral",
        "center_loss",
        "supervised_contrastive",
    }

    def __init__(self, model: nn.Module, device: torch.device, config: dict):
        super().__init__(model, device)
        self.control_type = config.get("control_type", "none").lower()
        if self.control_type not in self.VALID_CONTROLS:
            raise ValueError(
                f"Unknown control_type '{self.control_type}'. "
                f"Available: {sorted(self.VALID_CONTROLS)}"
            )

        self.lambda_control = float(config.get("lambda_control", 0.0))
        self.lambda_distill = float(config.get("lambda_distill", 5.0))
        self.temperature = float(config.get("temperature", 3.0))
        self.supcon_temperature = float(config.get("supcon_temperature", 0.2))
        self.center_lr_scale = float(config.get("center_lr_scale", 0.5))
        self.use_seen_mask = bool(config.get("use_seen_mask", True))

        self.teacher: nn.Module | None = None
        self._seen_classes: set[int] = set()
        self._features: torch.Tensor | None = None

        self.classifier = _final_linear(model)
        if self.classifier is None:
            raise ValueError("GeometryControls requires a model with an nn.Linear classifier")

        self._feature_hook = None
        self._register_feature_hook()

        self.centers: nn.Parameter | None = None
        if self.control_type == "center_loss":
            self.centers = nn.Parameter(
                torch.zeros(self.classifier.out_features, self.classifier.in_features, device=device)
            )

        logger.info(
            "GeometryControls | control=%s lambda=%s distill=%s T=%s",
            self.control_type,
            self.lambda_control,
            self.lambda_distill,
            self.temperature,
        )

    def _capture_classifier_input(self, module, inputs, output):
        if inputs:
            feat = inputs[0]
            self._features = feat.view(feat.size(0), -1)

    def _register_feature_hook(self):
        if self._feature_hook is None:
            self._feature_hook = self.classifier.register_forward_hook(
                self._capture_classifier_input
            )

    def _remove_feature_hook(self):
        if self._feature_hook is not None:
            self._feature_hook.remove()
            self._feature_hook = None

    def _get_seen_mask(self) -> torch.Tensor | None:
        if not self.use_seen_mask or not self._seen_classes:
            return None
        n_out = self.classifier.weight.size(0)
        mask = torch.zeros(n_out, dtype=torch.bool, device=self.device)
        for c in self._seen_classes:
            if c < n_out:
                mask[c] = True
        return mask if mask.sum() > 1 else None

    def _distillation_loss(self, logits: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if self.teacher is None or self.lambda_distill <= 0:
            return logits.new_tensor(0.0)
        with torch.no_grad():
            teacher_logits = self.teacher(x)
        t = self.temperature
        student_soft = F.log_softmax(logits / t, dim=1)
        teacher_soft = F.softmax(teacher_logits / t, dim=1)
        return F.kl_div(student_soft, teacher_soft, reduction="batchmean") * (t * t)

    def _control_loss(self, y: torch.Tensor) -> torch.Tensor:
        if self.lambda_control <= 0 or self.control_type == "none":
            return self.classifier.weight.new_tensor(0.0)

        seen_mask = self._get_seen_mask()
        with torch.amp.autocast("cuda", enabled=False):
            if self.control_type == "cosine_orthogonal":
                return cosine_orthogonal_loss(self.classifier.weight, seen_mask)
            if self.control_type == "prototype_decorrelation":
                return prototype_decorrelation_loss(self.classifier.weight, seen_mask)
            if self.control_type == "spectral":
                return compute_spectral_flatness(self.classifier.weight.float(), seen_mask)

        if self._features is None:
            return self.classifier.weight.new_tensor(0.0)
        features = self._features

        if self.control_type == "center_loss":
            assert self.centers is not None
            centers = self.centers[y].to(features.device)
            return (features.float() - centers.float()).pow(2).sum(dim=1).mean()

        if self.control_type == "supervised_contrastive":
            return supervised_contrastive_loss(features, y, self.supcon_temperature)

        return self.classifier.weight.new_tensor(0.0)

    def training_step(self, x, y, task_id):
        self._seen_classes.update(y.detach().cpu().tolist())
        self._features = None
        logits = self.model(x)
        ce_loss = F.cross_entropy(logits, y)
        kd_loss = self._distillation_loss(logits, x)
        ctrl_loss = self._control_loss(y)
        loss = ce_loss + self.lambda_distill * kd_loss + self.lambda_control * ctrl_loss
        return loss, logits

    def _build_optimizer(self, config: dict):
        if self.centers is None:
            return super()._build_optimizer(config)

        name = config.get("optimizer", "sgd").lower()
        lr = config.get("lr", 0.01)
        wd = config.get("weight_decay", 0.0)
        params = [
            {"params": self.model.parameters(), "weight_decay": wd},
            {"params": [self.centers], "lr": lr * self.center_lr_scale, "weight_decay": 0.0},
        ]
        if name == "sgd":
            return torch.optim.SGD(params, lr=lr, momentum=config.get("momentum", 0.9))
        if name == "adam":
            return torch.optim.Adam(params, lr=lr)
        if name == "adamw":
            return torch.optim.AdamW(params, lr=lr)
        raise ValueError(f"Unknown optimizer: {name}")

    def after_task(self, task_id: int):
        self._features = None
        self._remove_feature_hook()
        self.teacher = copy.deepcopy(self.model)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
        self._register_feature_hook()
        logger.info("  [GeometryControls] Teacher snapshot saved after task %d", task_id + 1)
