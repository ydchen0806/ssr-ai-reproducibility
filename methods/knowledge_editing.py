"""SSR for Knowledge Editing / Sequential Model Refinement.

Simulates the knowledge editing scenario: a trained model must sequentially
absorb corrections (label flips, new associations) without catastrophic
forgetting of correct knowledge. This is a critical real-world scenario
for deployed models.

Biological motivation:
  The H01 connectome shows that the temporal cortex (which handles semantic
  knowledge) uses spatial exclusion zones to isolate memory engrams. During
  knowledge editing (analogous to synaptic reconsolidation), SSR ensures
  that updating one "fact" does not destructively interfere with neighboring
  memory representations.

Scenarios implemented:
  1. Sequential Correction: Fix mislabeled subsets while retaining accuracy
  2. Class Relabeling: Swap class identities sequentially
  3. Incremental Refinement: Iteratively improve accuracy on hard examples
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset
import numpy as np

from .base import BaseContinualLearner
from .bioreg import compute_spatial_biocs, compute_spectral_flatness


class KnowledgeEditingBioCS(BaseContinualLearner):
    """SSR regularization for sequential knowledge editing.

    Instead of learning new tasks, this method sequentially edits the model's
    knowledge: correcting specific class predictions while maintaining overall
    accuracy. SSR prevents the "ripple effect" where editing one fact
    corrupts related facts.
    """

    def __init__(self, model: nn.Module, device: torch.device, config: dict):
        super().__init__(model, device)
        self.lambda_spatial = config.get("lambda_spatial", 0.01)
        self.lambda_spectral = config.get("lambda_spectral", 0.0)
        self.lambda_locality = config.get("lambda_locality", 1.0)
        self.A_exc = config.get("A_exc", 1.0)
        self.A_inh = config.get("A_inh", 0.8)
        self.sigma_exc = config.get("sigma_exc", 0.2)
        self.sigma_inh = config.get("sigma_inh", 0.5)

        self._pre_edit_params: dict[str, torch.Tensor] = {}
        self._edit_classes: set[int] = set()

    def begin_edit(self):
        """Snapshot model before editing."""
        self._pre_edit_params = {
            n: p.data.clone()
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }

    def training_step(self, x, y, task_id):
        self._edit_classes.update(y.cpu().tolist())
        logits = self.model(x)
        ce_loss = F.cross_entropy(logits, y)

        reg_loss = torch.tensor(0.0, device=self.device)

        with torch.amp.autocast("cuda", enabled=False):
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Linear) and ("fc" in name or "head" in name or "classifier" in name):
                    n_out = module.weight.size(0)
                    mask = torch.ones(n_out, dtype=torch.bool, device=self.device)
                    reg_loss = reg_loss + self.lambda_spatial * compute_spatial_biocs(
                        module.weight.float(), self.A_exc, self.A_inh,
                        self.sigma_exc, self.sigma_inh, mask,
                    )
                    if self.lambda_spectral > 0:
                        reg_loss = reg_loss + self.lambda_spectral * compute_spectral_flatness(
                            module.weight.float(), mask,
                        )

        if self._pre_edit_params and self.lambda_locality > 0:
            locality_loss = torch.tensor(0.0, device=self.device)
            for name, param in self.model.named_parameters():
                if name in self._pre_edit_params:
                    locality_loss += (param - self._pre_edit_params[name]).pow(2).sum()
            reg_loss += self.lambda_locality * locality_loss

        loss = ce_loss + reg_loss
        return loss, logits

    def train_task(self, task_id: int, train_set, training_config: dict):
        self.begin_edit()
        super().train_task(task_id, train_set, training_config)

    def after_task(self, task_id: int):
        self._pre_edit_params = {
            n: p.data.clone()
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }
