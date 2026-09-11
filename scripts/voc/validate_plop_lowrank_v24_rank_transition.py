#!/usr/bin/env python3
"""Validate the PASCAL 10-1 low-rank classifier transition before launch."""

import argparse
import importlib
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn


def build_model(module, classes):
    opts = SimpleNamespace(
        dataset="voc",
        low_rank_classifier_rank=8,
        low_rank_classifier_alpha=8.0,
    )
    return module.IncrementalSegmentationModule(
        nn.Identity(),
        nn.Identity(),
        256,
        classes=list(classes),
        opts=opts,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plop-root", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.plop_root.resolve()))
    segmentation_module = importlib.import_module("segmentation_module")

    step1 = build_model(segmentation_module, [11, 1])
    step2_old = build_model(segmentation_module, [11, 1])
    step2 = build_model(segmentation_module, [11, 1, 1])

    step2_old.load_state_dict(step1.state_dict(), strict=True)
    incompatible = step2.load_state_dict(step1.state_dict(), strict=False)
    assert not incompatible.unexpected_keys
    assert set(incompatible.missing_keys) == {
        "cls.1.lora_A",
        "cls.1.lora_B",
        "cls.2.weight",
        "cls.2.bias",
    }

    wide_head, narrow_head = step2.cls[:2]
    assert wide_head.low_rank_requested_rank == 8
    assert wide_head.low_rank_rank == 8
    assert narrow_head.low_rank_requested_rank == 8
    assert narrow_head.low_rank_rank == 1
    assert narrow_head.lora_A.shape == (1, 256)
    assert narrow_head.lora_B.shape == (1, 1)
    assert torch.equal(narrow_head.effective_weight(), step1.cls[1].weight)
    assert torch.equal(narrow_head.bias, step1.cls[1].bias)
    assert math.isclose(
        wide_head.low_rank_alpha / wide_head.low_rank_rank,
        narrow_head.low_rank_alpha / narrow_head.low_rank_rank,
    )
    print("PLOP 10-1 low-rank transition contract OK")


if __name__ == "__main__":
    main()
