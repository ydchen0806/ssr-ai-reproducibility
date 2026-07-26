"""BioReg: Biologically-inspired Regularization for Continual Learning.

Modes controlled by `regularization_mode`:

  (A) "importance": Parameter importance-weighted quadratic penalty (EWC/SI/MAS family).
  (B) "spatial_biocs": Mexican-hat center-surround spatial suppression.
  (C) "hybrid": Combines (A) + (B) via two independent λ weights.
  (D) "adaptive_biocs": SSR with SVD-health-driven adaptive λ scheduling.
  (E) "spectral_biocs": SSR + explicit SVD spectral flattening regularizer.
  (F) "layerwise_biocs": SSR on higher-order layers only (mimicking H01 vs V1).

Biological basis (from H01 connectome paper):
  - dw_i/dt = η·Hebb_i − λ·Σ_j K(d_ij)·w_j
  - K is a Mexican-hat center-surround kernel
  - Exponential spatial suppression creates a "glass ceiling" for synaptic growth
  - Maintains high-rank SVD spectrum (prevents representational collapse)
  - Temporal update spectra require separate task-resolution controls
  - Stronger in analyzed human higher-order cortex than in mouse V1 (MICrONS)
"""

import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .base import BaseContinualLearner

logger = logging.getLogger(__name__)

KERNEL_FAMILIES = frozenset({"gaussian", "laplace", "cauchy", "inverse"})


def validate_spatial_kernel_parameters(
    A_exc: float,
    A_inh: float,
    sigma_exc: float,
    sigma_inh: float,
    kernel_family: str,
) -> str:
    """Validate a center-surround kernel and return its normalized family."""
    if not isinstance(kernel_family, str):
        raise TypeError("kernel_family must be a string")
    family = kernel_family.lower()
    if family not in KERNEL_FAMILIES:
        choices = ", ".join(sorted(KERNEL_FAMILIES))
        raise ValueError(f"Unknown kernel_family '{kernel_family}'. Expected one of: {choices}")

    for name, value in (("A_exc", A_exc), ("A_inh", A_inh)):
        try:
            finite = math.isfinite(value)
        except TypeError as exc:
            raise TypeError(f"{name} must be a finite number") from exc
        if not finite or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    for name, value in (("sigma_exc", sigma_exc), ("sigma_inh", sigma_inh)):
        try:
            finite = math.isfinite(value)
        except TypeError as exc:
            raise TypeError(f"{name} must be a finite number") from exc
        if not finite or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    return family


def _radial_response(dist: torch.Tensor, sigma: float, kernel_family: str) -> torch.Tensor:
    """Evaluate a unit-height radial response using its native scale parameter."""
    if kernel_family == "gaussian":
        return torch.exp(-(dist ** 2) / (2 * sigma ** 2))
    if kernel_family == "laplace":
        return torch.exp(-dist / sigma)
    if kernel_family == "cauchy":
        return 1.0 / (1.0 + (dist / sigma) ** 2)
    return 1.0 / (1.0 + dist / sigma)


def compute_spatial_biocs(
    weights: torch.Tensor,
    A_exc: float = 1.0,
    A_inh: float = 0.8,
    sigma_exc: float = 0.2,
    sigma_inh: float = 0.5,
    seen_mask: torch.Tensor | None = None,
    kernel_family: str = "gaussian",
) -> torch.Tensor:
    """Mexican-hat center-surround penalty on weight vectors.

    Implements a center-surround geometry inspired by the H01 connectome:
    strong weight directions are penalized for nearby co-amplification in an
    intermediate annulus, without claiming direct plasticity dynamics.

    Forced float32: the normalize→cosine→sqrt→exp chain produces NaN
    gradients under float16 AMP autocast.
    """
    family = validate_spatial_kernel_parameters(
        A_exc, A_inh, sigma_exc, sigma_inh, kernel_family,
    )
    weights = weights.float()
    if seen_mask is not None:
        weights = weights[seen_mask]
    n = weights.size(0)
    if n < 2:
        return torch.tensor(0.0, device=weights.device)

    w_norm = F.normalize(weights, dim=1)
    cos_sim = torch.clamp(w_norm @ w_norm.T, -1.0, 1.0)
    dist = torch.sqrt(torch.clamp(1.0 - cos_sim, min=1e-8))

    exc = A_exc * _radial_response(dist, sigma_exc, family)
    inh = A_inh * _radial_response(dist, sigma_inh, family)
    P = (inh - exc) + (A_exc - A_inh)
    P = P - torch.diag(torch.diag(P))

    return P.sum() / (n * (n - 1) + 1e-8)


def compute_spectral_flatness(weights: torch.Tensor, seen_mask: torch.Tensor | None = None) -> torch.Tensor:
    """SVD spectral flattening loss (forced float32 for numerical stability)."""
    weights = weights.float()
    if seen_mask is not None:
        weights = weights[seen_mask]
    if weights.size(0) < 2:
        return torch.tensor(0.0, device=weights.device)

    try:
        sv = torch.linalg.svdvals(weights)
    except Exception:
        return torch.tensor(0.0, device=weights.device)

    if sv.numel() < 2:
        return torch.tensor(0.0, device=weights.device)

    sv_normalized = sv / (sv.sum() + 1e-8)
    uniform = torch.ones_like(sv_normalized) / sv_normalized.numel()
    spectral_loss = F.kl_div(
        (sv_normalized + 1e-8).log(), uniform, reduction="sum"
    )
    return spectral_loss


def compute_weight_smoothness(
    current_weights: torch.Tensor,
    prev_weights: torch.Tensor | None,
) -> torch.Tensor:
    """High-frequency weight dynamics suppression.

    Penalizes large task-to-task weight changes. This is retained as an
    engineering control; temporal smoothness must be evaluated from update
    trajectories rather than inferred from this penalty alone.
    """
    if prev_weights is None:
        return torch.tensor(0.0, device=current_weights.device)

    delta = current_weights - prev_weights
    smoothness_loss = delta.pow(2).mean()
    return smoothness_loss


class BioReg(BaseContinualLearner):
    """Biologically-inspired regularization with six operational modes.

    Modes A-C: original.
    Modes D-F: new, derived from deeper analysis of the H01 connectome paper.

    Mode D "adaptive_biocs":
        Monitors SVD spectral health of target layers. When the spectrum
        becomes skewed (spectral collapse beginning), λ_spatial is dynamically
        increased. Implements the biological insight that spatial suppression
        is a homeostatic response to representational monopoly.

    Mode E "spectral_biocs":
        Adds explicit SVD flattening loss alongside spatial SSR. This tests
        whether a simpler spectral objective can explain the high-rank geometry.

    Mode F "layerwise_biocs":
        Applies SSR only to the final (higher-order) layers while leaving
        early feature extraction layers unconstrained. This is a computational
        analogy to the H01--MICrONS boundary, not a biological equivalence claim.
    """

    VALID_MODES = {
        "importance", "spatial_biocs", "hybrid",
        "adaptive_biocs", "spectral_biocs", "layerwise_biocs",
    }

    def __init__(self, model: nn.Module, device: torch.device, config: dict):
        super().__init__(model, device)
        self.mode = config.get("regularization_mode", "importance")
        assert self.mode in self.VALID_MODES, f"Unknown mode: {self.mode}. Valid: {self.VALID_MODES}"

        # --- Mode A: importance-based ---
        self.lambda_reg = config.get("lambda_reg", 1.0)
        self.importance_mode = config.get("importance_mode", "fisher")
        self.use_homeostatic = config.get("homeostatic_scaling", False)
        self.use_metaplasticity = config.get("metaplasticity", False)
        self.meta_decay = config.get("meta_decay", 0.9)
        self.fisher_samples = config.get("fisher_samples", 2000)
        self.xi = config.get("xi", 0.1)

        self.prev_params: dict[str, torch.Tensor] = {}
        self.importance: dict[str, torch.Tensor] = {}
        self.consolidation_count: dict[str, torch.Tensor] = {}
        self._w: dict[str, torch.Tensor] = {}
        self._init_params: dict[str, torch.Tensor] = {}
        self._task_train_set = None

        # --- Mode B/D/E/F: spatial SSR ---
        self.lambda_spatial = config.get("lambda_spatial", 0.001)
        self.A_exc = config.get("A_exc", 1.0)
        self.A_inh = config.get("A_inh", 0.8)
        self.sigma_exc = config.get("sigma_exc", 0.2)
        self.sigma_inh = config.get("sigma_inh", 0.5)
        self.kernel_family = validate_spatial_kernel_parameters(
            self.A_exc,
            self.A_inh,
            self.sigma_exc,
            self.sigma_inh,
            config.get("kernel_family", "gaussian"),
        )
        self.biocs_targets = config.get("biocs_targets", ["classifier"])
        self.use_seen_mask = config.get("use_seen_mask", True)

        # --- Mode D: adaptive λ scheduling ---
        self.adaptive_base_lambda = config.get("adaptive_base_lambda", 0.001)
        self.adaptive_max_lambda = config.get("adaptive_max_lambda", 0.1)
        self.spectral_threshold = config.get("spectral_threshold", 0.5)
        self._current_adaptive_lambda = self.adaptive_base_lambda

        # --- Mode E: spectral flattening ---
        self.lambda_spectral = config.get("lambda_spectral", 0.01)

        # --- Mode F: layerwise (higher-order only) ---
        self.layerwise_fraction = config.get("layerwise_fraction", 0.5)

        # --- Weight smoothness (Modes D/E/F can optionally include) ---
        self.lambda_smoothness = config.get("lambda_smoothness", 0.0)
        self._prev_task_weights: dict[str, torch.Tensor] = {}

        self._seen_classes: set[int] = set()
        self._target_layers = self._resolve_target_layers()
        self._step_count = 0

    def _resolve_target_layers(self) -> list[tuple[str, nn.Linear]]:
        linears = [
            (n, m) for n, m in self.model.named_modules()
            if isinstance(m, nn.Linear)
        ]
        if not linears:
            return []

        if self.mode == "layerwise_biocs":
            n_skip = max(1, int(len(linears) * (1 - self.layerwise_fraction)))
            return linears[n_skip:]

        targets_set = set(self.biocs_targets)
        if "all" in targets_set:
            return linears

        result = []
        if "classifier" in targets_set:
            result.append(linears[-1])
        if "hidden" in targets_set:
            result.extend(linears[:-1])
        return result

    def _get_seen_mask(self, layer: nn.Linear) -> torch.Tensor | None:
        if not self.use_seen_mask or not self._seen_classes:
            return None
        n_out = layer.weight.size(0)
        mask = torch.zeros(n_out, dtype=torch.bool, device=self.device)
        for c in self._seen_classes:
            if c < n_out:
                mask[c] = True
        return mask if mask.sum() > 1 else None

    # ── SVD health monitoring (for adaptive mode) ─────────────────────

    @torch.no_grad()
    def _compute_spectral_skewness(self) -> float:
        """Measure how "collapsed" the SVD spectrum is. Returns 0-1 (1 = fully collapsed)."""
        total_ratio = 0.0
        count = 0
        for name, layer in self._target_layers:
            w = layer.weight.data
            mask = self._get_seen_mask(layer)
            if mask is not None:
                w = w[mask]
            if w.size(0) < 2:
                continue
            try:
                sv = torch.linalg.svdvals(w)
                top1_ratio = sv[0].item() / (sv.sum().item() + 1e-8)
                total_ratio += top1_ratio
                count += 1
            except Exception:
                continue
        return total_ratio / max(count, 1)

    def _update_adaptive_lambda(self):
        skewness = self._compute_spectral_skewness()
        if skewness > self.spectral_threshold:
            boost = (skewness - self.spectral_threshold) / (1.0 - self.spectral_threshold + 1e-8)
            self._current_adaptive_lambda = min(
                self.adaptive_base_lambda + boost * (self.adaptive_max_lambda - self.adaptive_base_lambda),
                self.adaptive_max_lambda,
            )
        else:
            self._current_adaptive_lambda = self.adaptive_base_lambda

    # ── Training ──────────────────────────────────────────────────────

    def training_step(self, x, y, task_id):
        self._seen_classes.update(y.cpu().tolist())
        self._step_count += 1

        logits = self.model(x)
        ce_loss = F.cross_entropy(logits, y)
        reg_loss = torch.tensor(0.0, device=self.device)

        if self.mode in ("importance", "hybrid"):
            reg_loss = reg_loss + self.lambda_reg * self._importance_penalty()

        if self.mode == "spatial_biocs":
            reg_loss = reg_loss + self.lambda_spatial * self._spatial_penalty()

        elif self.mode == "hybrid":
            reg_loss = reg_loss + self.lambda_spatial * self._spatial_penalty()

        elif self.mode == "adaptive_biocs":
            if self._step_count % 100 == 0:
                self._update_adaptive_lambda()
            reg_loss = reg_loss + self._current_adaptive_lambda * self._spatial_penalty()

        elif self.mode == "spectral_biocs":
            reg_loss = reg_loss + self.lambda_spatial * self._spatial_penalty()
            reg_loss = reg_loss + self.lambda_spectral * self._spectral_penalty()

        elif self.mode == "layerwise_biocs":
            reg_loss = reg_loss + self.lambda_spatial * self._spatial_penalty()

        if self.lambda_smoothness > 0:
            reg_loss = reg_loss + self.lambda_smoothness * self._smoothness_penalty()

        loss = ce_loss + reg_loss
        return loss, logits

    def train_task(self, task_id: int, train_set, training_config: dict):
        self._task_train_set = train_set
        self._step_count = 0

        if self.importance_mode == "path_integral" and self.mode in ("importance", "hybrid"):
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self._init_params[name] = param.data.clone()
                    self._w[name] = torch.zeros_like(param)

        if self.mode == "adaptive_biocs":
            self._current_adaptive_lambda = self.adaptive_base_lambda

        super().train_task(task_id, train_set, training_config)

    def after_task(self, task_id: int):
        if self.mode in ("importance", "hybrid"):
            self._consolidate_importance()

        self.prev_params = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        for name, layer in self._target_layers:
            self._prev_task_weights[name] = layer.weight.data.clone()

        if self.mode == "adaptive_biocs":
            skewness = self._compute_spectral_skewness()
            logger.info(f"  [Adaptive SSR] Post-task SVD skewness: {skewness:.4f}, "
                        f"λ_spatial: {self._current_adaptive_lambda:.6f}")

    # ── Spatial SSR penalty ────────────────────────────────────────

    def _spatial_penalty(self) -> torch.Tensor:
        with torch.amp.autocast("cuda", enabled=False):
            total = torch.tensor(0.0, device=self.device)
            for name, layer in self._target_layers:
                mask = self._get_seen_mask(layer)
                total = total + compute_spatial_biocs(
                    layer.weight.float(), self.A_exc, self.A_inh,
                    self.sigma_exc, self.sigma_inh, mask,
                    kernel_family=self.kernel_family,
                )
            return total

    # ── Spectral flattening penalty (Mode E) ──────────────────────────

    def _spectral_penalty(self) -> torch.Tensor:
        with torch.amp.autocast("cuda", enabled=False):
            total = torch.tensor(0.0, device=self.device)
            for name, layer in self._target_layers:
                mask = self._get_seen_mask(layer)
                total = total + compute_spectral_flatness(layer.weight.float(), mask)
            return total

    # ── Weight smoothness penalty ─────────────────────────────────────

    def _smoothness_penalty(self) -> torch.Tensor:
        total = torch.tensor(0.0, device=self.device)
        for name, layer in self._target_layers:
            prev_w = self._prev_task_weights.get(name)
            total = total + compute_weight_smoothness(layer.weight, prev_w)
        return total

    # ── Importance-based penalty (Mode A) ─────────────────────────────

    def _importance_penalty(self) -> torch.Tensor:
        if not self.prev_params:
            return torch.tensor(0.0, device=self.device)
        reg = torch.tensor(0.0, device=self.device)
        for name, param in self.model.named_parameters():
            if name not in self.prev_params or name not in self.importance:
                continue
            imp = self.importance[name]
            if self.use_metaplasticity and name in self.consolidation_count:
                imp = imp * (1.0 + self.consolidation_count[name])
            reg += (imp * (param - self.prev_params[name]).pow(2)).sum()
        return reg

    def _consolidate_importance(self):
        new_importance = self._estimate_importance()
        if self.use_homeostatic:
            new_importance = self._apply_homeostatic_scaling(new_importance)
        for name in new_importance:
            if name in self.importance:
                self.importance[name] = self.importance[name] + new_importance[name]
            else:
                self.importance[name] = new_importance[name]
        if self.use_metaplasticity:
            for name in self.importance:
                if name not in self.consolidation_count:
                    self.consolidation_count[name] = torch.zeros_like(self.importance[name])
                self.consolidation_count[name] = (
                    self.meta_decay * self.consolidation_count[name]
                    + (self.importance[name] > self.importance[name].mean()).float()
                )

    def _estimate_importance(self) -> dict[str, torch.Tensor]:
        dispatch = {
            "fisher": self._importance_fisher,
            "path_integral": self._importance_path_integral,
            "sensitivity": self._importance_sensitivity,
        }
        fn = dispatch.get(self.importance_mode)
        if fn is None:
            raise ValueError(f"Unknown importance_mode '{self.importance_mode}'")
        return fn()

    def _importance_fisher(self) -> dict[str, torch.Tensor]:
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters() if p.requires_grad}
        self.model.eval()
        loader = DataLoader(self._task_train_set, batch_size=32, shuffle=True, num_workers=2)
        count = 0
        for x, y in loader:
            if count >= self.fisher_samples:
                break
            x, y = x.to(self.device), y.to(self.device)
            bs = x.size(0)
            self.model.zero_grad()
            loss = F.cross_entropy(self.model(x), y)
            loss.backward()
            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.data.pow(2) * bs
            count += bs
        for n in fisher:
            fisher[n] /= max(count, 1)
        self.model.train()
        return fisher

    def _importance_path_integral(self) -> dict[str, torch.Tensor]:
        importance = {}
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            delta = param.data - self._init_params.get(name, param.data)
            w = self._w.get(name, torch.zeros_like(param))
            imp = w / (delta.pow(2) + self.xi)
            importance[name] = torch.clamp(imp, min=0.0)
        return importance

    def _importance_sensitivity(self) -> dict[str, torch.Tensor]:
        omega = {n: torch.zeros_like(p) for n, p in self.model.named_parameters() if p.requires_grad}
        self.model.eval()
        loader = DataLoader(self._task_train_set, batch_size=32, shuffle=True, num_workers=2)
        count = 0
        for x, _ in loader:
            if count >= self.fisher_samples:
                break
            x = x.to(self.device)
            bs = x.size(0)
            self.model.zero_grad()
            self.model(x).pow(2).mean().backward()
            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    omega[n] += p.grad.data.abs() * bs
            count += bs
        for n in omega:
            omega[n] /= max(count, 1)
        self.model.train()
        return omega

    @staticmethod
    def _apply_homeostatic_scaling(importance: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {name: imp / (imp.norm() + 1e-8) for name, imp in importance.items()}
