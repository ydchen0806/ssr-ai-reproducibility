"""Spatial and spectral penalties used by the reported classifiers."""
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .base import BaseContinualLearner
logger = logging.getLogger(__name__)

def compute_spatial_biocs(weights: torch.Tensor, A_exc: float=1.0, A_inh: float=0.8, sigma_exc: float=0.2, sigma_inh: float=0.5, seen_mask: torch.Tensor | None=None) -> torch.Tensor:
    """Mexican-hat center-surround penalty on weight vectors.

    Implements a center-surround geometry inspired by the H01 connectome:
    strong weight directions are penalized for nearby co-amplification in an
    intermediate annulus, without claiming direct plasticity dynamics.

    Forced float32: the normalize→cosine→sqrt→exp chain produces NaN
    gradients under float16 AMP autocast.
    """
    with torch.autocast(device_type=weights.device.type, enabled=False):
        weights = weights.float()
        if seen_mask is not None:
            weights = weights[seen_mask]
        n = weights.size(0)
        if n < 2:
            return torch.tensor(0.0, device=weights.device)
        w_norm = F.normalize(weights, dim=1)
        cos_sim = torch.clamp(w_norm @ w_norm.T, -1.0, 1.0)
        dist = torch.sqrt(torch.clamp(1.0 - cos_sim, min=1e-08))
        exc = A_exc * torch.exp(-dist ** 2 / (2 * sigma_exc ** 2))
        inh = A_inh * torch.exp(-dist ** 2 / (2 * sigma_inh ** 2))
        P = inh - exc + (A_exc - A_inh)
        P = P - torch.diag(torch.diag(P))
        return P.sum() / (n * (n - 1) + 1e-08)

def compute_spectral_flatness(weights: torch.Tensor, seen_mask: torch.Tensor | None=None) -> torch.Tensor:
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
    sv_normalized = sv / (sv.sum() + 1e-08)
    uniform = torch.ones_like(sv_normalized) / sv_normalized.numel()
    spectral_loss = F.kl_div((sv_normalized + 1e-08).log(), uniform, reduction='sum')
    return spectral_loss
