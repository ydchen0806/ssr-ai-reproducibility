"""Dependency-light mask and task helpers for segmentation experiments."""

from __future__ import annotations

import random

import numpy as np
from PIL import Image


def flower_foreground(mask: Image.Image) -> np.ndarray:
    rgb = np.asarray(mask.convert("RGB"), dtype=np.int32)
    nominal_background = np.asarray([0, 0, 255], dtype=np.int32)
    distance = np.sqrt(np.square(rgb - nominal_background).sum(axis=2))
    return distance > 40.0


def pet_foreground_and_valid(mask: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    trimap = np.asarray(mask.convert("L"), dtype=np.uint8)
    values = set(int(value) for value in np.unique(trimap))
    if not values <= {1, 2, 3}:
        raise RuntimeError(f"Unexpected Oxford Pet trimap values: {sorted(values)}")
    return trimap == 1, trimap != 3


def make_tasks(num_classes: int, classes_per_task: int, seed: int) -> list[list[int]]:
    if num_classes <= 0 or classes_per_task <= 0:
        raise ValueError("num_classes and classes_per_task must be positive")
    order = list(range(num_classes))
    random.Random(seed).shuffle(order)
    return [
        order[index : index + classes_per_task]
        for index in range(0, num_classes, classes_per_task)
    ]
