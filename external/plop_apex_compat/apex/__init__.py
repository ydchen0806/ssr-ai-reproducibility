"""Minimal Apex compatibility layer for the pinned PLOP source.

PLOP only uses ``amp.initialize`` and ``amp.scale_loss``.  The cluster's
PyTorch 1.13 runtime is paired with CUDA 11.7 while the host compiler is
CUDA 12.9, so building the historical Apex extension would be neither
reproducible nor necessary.  This layer keeps PLOP's optimizer semantics in
full precision and is intentionally not a replacement for Apex in general.
"""

from contextlib import contextmanager


class _FullPrecisionAmp:
    @staticmethod
    def initialize(models, optimizer, opt_level=None):
        """Match Apex's return convention without changing model or optimizer."""
        return models, optimizer

    @staticmethod
    @contextmanager
    def scale_loss(loss, optimizer):
        """Yield the unscaled loss for an explicit FP32 PLOP execution."""
        del optimizer
        yield loss


amp = _FullPrecisionAmp()
