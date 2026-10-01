"""SSR penalties for incremental-segmentation decoder channels."""

import torch
from torch.nn import functional as F


def _distance_squared(cosine, distance):
    if distance == "projective":
        return (1.0 - cosine.square()).clamp_min(0.0)
    if distance == "cosine":
        return (1.0 - cosine).clamp_min(0.0)
    raise ValueError("Unsupported SSR distance: %s" % distance)


def _finite_annulus(cosine, *, a_exc, a_inh, sigma_exc, sigma_inh,
                    distance, repulsion_margin):
    distance_squared = _distance_squared(cosine, distance)
    pairwise = torch.sqrt(
        distance_squared + torch.finfo(distance_squared.dtype).eps
    )
    excitation = a_exc * torch.exp(-0.5 * (pairwise / sigma_exc).square())
    inhibition = a_inh * torch.exp(-0.5 * (pairwise / sigma_inh).square())
    annulus = (inhibition - excitation).clamp_min(0.0)
    overlap = F.relu(cosine - repulsion_margin).square()
    return annulus * overlap, annulus


def classifier_channel_ssr_penalty(classifier_heads, *, a_exc, a_inh,
                                   sigma_exc, sigma_inh, distance,
                                   target_scope, repulsion_margin):
    """Apply bounded SSR to either new-to-old or historical channel geometry.

    ``current_to_old`` preserves the earlier cross-task implementation: old
    rows are anchors and only the current head receives gradients.
    ``historical_only`` applies the same finite-annulus topology among the
    trainable historical rows.  It therefore does not directly constrain the
    newly introduced class channels, addressing the plasticity loss observed
    in the preceding PASCAL development studies.
    """
    rows = [
        (head.effective_weight() if hasattr(head, "effective_weight") else head.weight)
        .float().flatten(start_dim=1)
        for head in classifier_heads
    ]
    if not rows:
        raise ValueError("SSR requires at least one classifier head")
    if len(rows) < 2:
        return rows[-1].sum() * 0.0
    historical = torch.cat(rows[:-1], dim=0)
    current = rows[-1]
    if historical.shape[1] != current.shape[1]:
        raise ValueError("Old and current decoder channels have incompatible dimensions")

    if target_scope == "current_to_old":
        old_unit = F.normalize(historical.detach(), p=2, dim=1, eps=1e-12)
        current_unit = F.normalize(current, p=2, dim=1, eps=1e-12)
        cosine = (current_unit @ old_unit.T).clamp(min=-1.0, max=1.0)
        weighted, annulus = _finite_annulus(
            cosine, a_exc=a_exc, a_inh=a_inh, sigma_exc=sigma_exc,
            sigma_inh=sigma_inh, distance=distance,
            repulsion_margin=repulsion_margin,
        )
        return weighted.sum() / annulus.sum().clamp_min(1e-8)

    if target_scope == "historical_only":
        if historical.shape[0] < 2:
            return historical.sum() * 0.0
        old_unit = F.normalize(historical, p=2, dim=1, eps=1e-12)
        cosine = (old_unit @ old_unit.T).clamp(min=-1.0, max=1.0)
        mask = ~torch.eye(
            cosine.shape[0], dtype=torch.bool, device=cosine.device
        )
        weighted, annulus = _finite_annulus(
            cosine[mask], a_exc=a_exc, a_inh=a_inh,
            sigma_exc=sigma_exc, sigma_inh=sigma_inh,
            distance=distance, repulsion_margin=repulsion_margin,
        )
        return weighted.sum() / annulus.sum().clamp_min(1e-8)

    raise ValueError("Unsupported SSR target scope: %s" % target_scope)
