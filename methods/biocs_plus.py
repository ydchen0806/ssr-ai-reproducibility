"""SSR+: Biologically-inspired CL with Knowledge Distillation and optional Replay.

Combines complementary anti-forgetting mechanisms:
  1. SSR spatial regularization (biological: weight geometry preservation)
  2. Knowledge distillation — logit-level + optional feature-level
  3. Experience replay (optional: direct rehearsal of old samples)
  4. Label smoothing (optional: softer training targets)
"""

import copy
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .base import BaseContinualLearner
from .bioreg import compute_spatial_biocs, compute_spectral_flatness

logger = logging.getLogger(__name__)


class SmallReplayBuffer:
    """Lightweight replay buffer for SSR+."""

    def __init__(self, capacity: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.data: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.ptr = 0

    def add_batch(self, x: torch.Tensor, y: torch.Tensor, max_per_batch: int = 16):
        idx = torch.randperm(x.size(0))[:max_per_batch]
        for i in idx:
            if len(self.data) < self.capacity:
                self.data.append((x[i].cpu(), y[i].cpu()))
            else:
                self.data[self.ptr] = (x[i].cpu(), y[i].cpu())
            self.ptr = (self.ptr + 1) % self.capacity

    def sample(self, batch_size: int):
        if not self.data:
            return None, None
        n = min(batch_size, len(self.data))
        indices = np.random.choice(len(self.data), size=n, replace=False)
        xs = torch.stack([self.data[i][0] for i in indices]).to(self.device)
        ys = torch.stack([self.data[i][1] for i in indices]).to(self.device)
        return xs, ys

    def __len__(self):
        return len(self.data)


def _hook_features(module, input, output, store: dict, key: str):
    store[key] = output


class BioCsPlus(BaseContinualLearner):
    """SSR+ = SSR spatial + Knowledge Distillation + optional Replay.

    Config keys:
        lambda_spatial:  SSR spatial penalty weight (default 0.01)
        lambda_distill:  Knowledge distillation weight (default 2.0)
        lambda_feat:     Feature distillation weight (default 0.0, >0 to enable)
        lambda_replay:   Replay CE weight (default 1.0, set 0 to disable)
        temperature:     Distillation temperature (default 2.0)
        buffer_size:     Replay buffer capacity (default 500)
        biocs_targets:   Which layers to apply SSR to (default ["all"])
        lambda_spectral: SVD spectral penalty (default 0.0, set >0 to enable)
        label_smoothing: CE label smoothing (default 0.0)
    """

    def __init__(self, model: nn.Module, device: torch.device, config: dict):
        super().__init__(model, device)

        self.lambda_spatial = config.get("lambda_spatial", 0.01)
        self.lambda_distill = config.get("lambda_distill", 2.0)
        self.lambda_feat = config.get("lambda_feat", 0.0)
        self.lambda_replay = config.get("lambda_replay", 1.0)
        self.lambda_spectral = config.get("lambda_spectral", 0.0)
        self.temperature = config.get("temperature", 2.0)
        self.buffer_size = config.get("buffer_size", 500)
        self.label_smoothing = config.get("label_smoothing", 0.0)

        self.A_exc = config.get("A_exc", 1.0)
        self.A_inh = config.get("A_inh", 0.8)
        self.sigma_exc = config.get("sigma_exc", 0.2)
        self.sigma_inh = config.get("sigma_inh", 0.5)
        biocs_targets = config.get("biocs_targets", ["all"])

        self.teacher: nn.Module | None = None
        self.buffer = SmallReplayBuffer(self.buffer_size, device) if self.buffer_size > 0 else None

        self._seen_classes: set[int] = set()

        targets_set = set(biocs_targets)
        linears = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear)]
        if "all" in targets_set:
            self._target_layers = linears
        elif "classifier" in targets_set:
            self._target_layers = [linears[-1]] if linears else []
        else:
            self._target_layers = linears

        self._feat_hook_handles = []
        self._student_feats: dict[str, torch.Tensor] = {}
        self._teacher_feats: dict[str, torch.Tensor] = {}
        self._feat_layer_name: str | None = None
        if self.lambda_feat > 0:
            self._setup_feat_hooks(model)

        logger.info(f"SSR+ | spatial={self.lambda_spatial} distill={self.lambda_distill} "
                     f"feat={self.lambda_feat} replay={self.lambda_replay} "
                     f"spectral={self.lambda_spectral} T={self.temperature} "
                     f"buf={self.buffer_size} smooth={self.label_smoothing} "
                     f"targets={len(self._target_layers)} layers")

    def _setup_feat_hooks(self, model: nn.Module):
        feat_layer = None
        for name, mod in model.named_modules():
            if isinstance(mod, nn.AdaptiveAvgPool2d):
                feat_layer = (name, mod)
            elif isinstance(mod, (nn.AvgPool2d,)) and feat_layer is None:
                feat_layer = (name, mod)
        if feat_layer is None:
            for name, mod in model.named_modules():
                if isinstance(mod, nn.Linear):
                    break
                feat_layer = (name, mod)
        if feat_layer is not None:
            self._feat_layer_name = feat_layer[0]
            h = feat_layer[1].register_forward_hook(
                lambda m, i, o: _hook_features(m, i, o, self._student_feats, "feat")
            )
            self._feat_hook_handles.append(h)
            logger.info(f"  [SSR+] Feature distillation on layer: {feat_layer[0]}")

    def _get_seen_mask(self, layer: nn.Linear):
        if not self._seen_classes:
            return None
        n_out = layer.weight.size(0)
        mask = torch.zeros(n_out, dtype=torch.bool, device=self.device)
        for c in self._seen_classes:
            if c < n_out:
                mask[c] = True
        return mask if mask.sum() > 1 else None

    def training_step(self, x, y, task_id):
        self._seen_classes.update(y.cpu().tolist())

        logits = self.model(x)
        ce_loss = F.cross_entropy(logits, y, label_smoothing=self.label_smoothing)
        loss = ce_loss

        # --- Knowledge Distillation (logit-level) ---
        if self.teacher is not None and self.lambda_distill > 0:
            with torch.no_grad():
                teacher_logits = self.teacher(x)
            T = self.temperature
            student_soft = F.log_softmax(logits / T, dim=1)
            teacher_soft = F.softmax(teacher_logits / T, dim=1)
            distill_loss = F.kl_div(student_soft, teacher_soft, reduction="batchmean") * (T * T)
            loss = loss + self.lambda_distill * distill_loss

        # --- Feature Distillation (representation-level) ---
        if (self.teacher is not None and self.lambda_feat > 0
                and "feat" in self._student_feats and "feat" in self._teacher_feats):
            s_feat = self._student_feats["feat"]
            t_feat = self._teacher_feats["feat"]
            if s_feat.shape == t_feat.shape:
                s_flat = s_feat.view(s_feat.size(0), -1)
                t_flat = t_feat.view(t_feat.size(0), -1)
                feat_loss = F.mse_loss(s_flat, t_flat)
                loss = loss + self.lambda_feat * feat_loss

        # --- Experience Replay ---
        if self.buffer is not None and len(self.buffer) > 0 and self.lambda_replay > 0:
            buf_x, buf_y = self.buffer.sample(min(x.size(0), 64))
            if buf_x is not None:
                buf_logits = self.model(buf_x)
                replay_loss = F.cross_entropy(buf_logits, buf_y)
                loss = loss + self.lambda_replay * replay_loss

        # --- SSR Spatial Regularization ---
        if self.lambda_spatial > 0:
            with torch.amp.autocast("cuda", enabled=False):
                spatial_loss = torch.tensor(0.0, device=self.device)
                for name, layer in self._target_layers:
                    mask = self._get_seen_mask(layer)
                    spatial_loss = spatial_loss + compute_spatial_biocs(
                        layer.weight.float(), self.A_exc, self.A_inh,
                        self.sigma_exc, self.sigma_inh, mask,
                    )
                loss = loss + self.lambda_spatial * spatial_loss

        # --- SVD Spectral Regularization ---
        if self.lambda_spectral > 0:
            with torch.amp.autocast("cuda", enabled=False):
                spec_loss = torch.tensor(0.0, device=self.device)
                for name, layer in self._target_layers:
                    mask = self._get_seen_mask(layer)
                    spec_loss = spec_loss + compute_spectral_flatness(
                        layer.weight.float(), mask,
                    )
                loss = loss + self.lambda_spectral * spec_loss

        if self.buffer is not None:
            self.buffer.add_batch(x.detach(), y.detach())

        return loss, logits

    def after_task(self, task_id: int):
        self.teacher = copy.deepcopy(self.model)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
        if self.lambda_feat > 0 and self._feat_layer_name:
            for name, mod in self.teacher.named_modules():
                if name == self._feat_layer_name:
                    mod.register_forward_hook(
                        lambda m, i, o: _hook_features(m, i, o, self._teacher_feats, "feat")
                    )
                    break
        logger.info(f"  [SSR+] Teacher snapshot saved after task {task_id+1}")
