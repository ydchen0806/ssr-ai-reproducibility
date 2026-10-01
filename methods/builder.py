"""Factory for building continual learning methods."""

import torch
import torch.nn as nn

from .base import BaseContinualLearner
from .biocs_plus import BioCsPlus
from .geometry_controls import GeometryControls
from .kd_matched import KDMatched

METHOD_REGISTRY = {
    "biocs_plus": BioCsPlus,
    "biocs+": BioCsPlus,
    "geometry_controls": GeometryControls,
    "geometry_control": GeometryControls,
    "kd": KDMatched,
    "kd_center": KDMatched,
    "kd_protodecor": KDMatched,
    "kd_spectral": KDMatched,
    "kd_ssr": KDMatched,
}


def build_method(
    config: dict, model: nn.Module, device: torch.device
) -> BaseContinualLearner:
    name = config["name"].lower()
    if name not in METHOD_REGISTRY:
        raise ValueError(
            f"Unknown method '{name}'. Available: {list(METHOD_REGISTRY.keys())}"
        )
    return METHOD_REGISTRY[name](model=model, device=device, config=config)
