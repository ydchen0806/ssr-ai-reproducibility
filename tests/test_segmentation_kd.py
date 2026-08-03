from __future__ import annotations

import pytest
import torch

from ssr_utils.segmentation_kd import old_class_distillation_ids


def test_old_class_distillation_cycles_only_over_previous_classes():
    first = old_class_distillation_ids([2, 5, 9], 5, 0, torch.device("cpu"))
    second = old_class_distillation_ids([2, 5, 9], 5, 1, torch.device("cpu"))
    assert first.tolist() == [2, 5, 9, 2, 5]
    assert second.tolist() == [9, 2, 5, 9, 2]
    assert set(first.tolist() + second.tolist()) <= {2, 5, 9}


def test_old_class_distillation_rejects_task_zero():
    with pytest.raises(ValueError, match="old_classes"):
        old_class_distillation_ids([], 2, 0, torch.device("cpu"))
