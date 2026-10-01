"""Numerically equivalent native-PyTorch fallback for PLOP normalization.

The historical ``inplace-abn`` CUDA extension cannot be built reproducibly
against this stack's CUDA 12.9 compiler and PyTorch 1.13 CUDA 11.7 binaries.
PLOP needs its BatchNorm-plus-activation semantics, not the fused/in-place
implementation.  This module preserves those operations while giving up only
the extension's memory optimization.
"""

import torch
from torch import nn
from torch.nn import functional as F


def _activation(x, name, parameter):
    if name in {None, "identity", "none"}:
        return x
    if name == "leaky_relu":
        return F.leaky_relu(x, negative_slope=parameter, inplace=False)
    if name == "relu":
        return F.relu(x, inplace=False)
    if name == "elu":
        return F.elu(x, alpha=parameter, inplace=False)
    raise ValueError(f"Unsupported PLOP InPlaceABN activation: {name!r}")


class _ActivationBatchNorm:
    def _configure_activation(self, activation, activation_param):
        self.activation = activation
        self.activation_param = activation_param

    def forward(self, x):
        return _activation(super().forward(x), self.activation, self.activation_param)


class ABN(_ActivationBatchNorm, nn.BatchNorm2d):
    def __init__(self, num_features, activation="leaky_relu", activation_param=0.01, **kwargs):
        self._configure_activation(activation, activation_param)
        super().__init__(num_features, **kwargs)


class InPlaceABN(ABN):
    """Non-in-place fallback retaining the public InPlaceABN constructor."""


class InPlaceABNSync(_ActivationBatchNorm, nn.SyncBatchNorm):
    def __init__(self, num_features, activation="leaky_relu", activation_param=0.01, **kwargs):
        self._configure_activation(activation, activation_param)
        super().__init__(num_features, **kwargs)
