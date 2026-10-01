"""Raw-image class-incremental torchvision datasets for ViT-LoRA checks."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from torch.utils.data import ConcatDataset, Subset
from torchvision import datasets, transforms

from .materialized_oxford_pet import (
    MaterializedOxfordIIITPet,
    find_materialized_root,
)


logger = logging.getLogger(__name__)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _targets(dataset) -> list[int]:
    if isinstance(dataset, ConcatDataset):
        return [target for child in dataset.datasets for target in _targets(child)]
    for attribute in ("targets", "_labels", "labels"):
        value = getattr(dataset, attribute, None)
        if value is not None:
            return [int(target) for target in value]
    raise TypeError(f"Cannot obtain targets from {type(dataset).__name__}")


class TorchvisionFineGrainedCL:
    """Balanced class stream over Flowers-102 or Oxford-IIIT Pet raw images."""

    SPECS = {
        "flowers102": {"classes": 102, "tasks": 10},
        "oxfordiiitpet": {"classes": 37, "tasks": 7},
    }

    def __init__(
        self,
        dataset: str,
        data_root: str = "./data/torchvision",
        n_tasks: int | None = None,
        class_order_seed: int | None = 7301,
        image_size: int = 224,
        download: bool = True,
        deterministic_transform: bool = True,
        **_: object,
    ):
        dataset = dataset.lower()
        if dataset not in self.SPECS:
            raise ValueError(f"Unsupported fine-grained dataset: {dataset}")
        spec = self.SPECS[dataset]
        self.dataset_name = dataset
        self.n_classes = int(spec["classes"])
        self.n_tasks = int(n_tasks or spec["tasks"])
        if not 1 <= self.n_tasks <= self.n_classes:
            raise ValueError("n_tasks must be between 1 and the number of classes")
        self.classes_per_task = int(np.ceil(self.n_classes / self.n_tasks))

        resize = int(round(image_size / 0.875))
        evaluation_transform = transforms.Compose(
            [
                transforms.Resize(resize),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
        if deterministic_transform:
            training_transform = evaluation_transform
        else:
            training_transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                ]
            )

        root = Path(data_root)
        if dataset == "flowers102":
            self.train_data = ConcatDataset(
                [
                    datasets.Flowers102(
                        root,
                        split="train",
                        transform=training_transform,
                        download=download,
                    ),
                    datasets.Flowers102(
                        root,
                        split="val",
                        transform=training_transform,
                        download=download,
                    ),
                ]
            )
            self.test_data = datasets.Flowers102(
                root,
                split="test",
                transform=evaluation_transform,
                download=download,
            )
        else:
            if find_materialized_root(root) is not None:
                self.train_data = MaterializedOxfordIIITPet(
                    root, split="trainval", transform=training_transform
                )
                self.test_data = MaterializedOxfordIIITPet(
                    root, split="test", transform=evaluation_transform
                )
            else:
                self.train_data = datasets.OxfordIIITPet(
                    root,
                    split="trainval",
                    target_types="category",
                    transform=training_transform,
                    download=download,
                )
                self.test_data = datasets.OxfordIIITPet(
                    root,
                    split="test",
                    target_types="category",
                    transform=evaluation_transform,
                    download=download,
                )

        self.train_targets = np.asarray(_targets(self.train_data), dtype=np.int64)
        self.test_targets = np.asarray(_targets(self.test_data), dtype=np.int64)
        observed = sorted(set(self.train_targets) | set(self.test_targets))
        if observed != list(range(self.n_classes)):
            raise RuntimeError(
                f"{dataset} labels are not contiguous 0..{self.n_classes - 1}: {observed}"
            )
        if set(self.train_targets) != set(self.test_targets):
            raise RuntimeError(f"{dataset} train and test class sets differ")

        self.class_order = list(range(self.n_classes))
        if class_order_seed is not None:
            np.random.RandomState(class_order_seed).shuffle(self.class_order)
        self._task_classes = [
            [int(value) for value in chunk]
            for chunk in np.array_split(self.class_order, self.n_tasks)
        ]
        self._current_task = 0
        logger.info(
            "%s raw-image CL: %s classes, %s tasks, train=%s, test=%s",
            dataset,
            self.n_classes,
            self.n_tasks,
            len(self.train_data),
            len(self.test_data),
        )

    def __iter__(self):
        self._current_task = 0
        return self

    def __next__(self):
        if self._current_task >= self.n_tasks:
            raise StopIteration
        task_id = self._current_task
        self._current_task += 1
        classes = self._get_task_classes(task_id)
        return (
            self._subset(self.train_data, self.train_targets, classes),
            self._subset(self.test_data, self.test_targets, classes),
        )

    def _get_task_classes(self, task_id: int) -> list[int]:
        return self._task_classes[task_id]

    def get_test_set(self, task_id: int):
        classes = self._get_task_classes(task_id)
        return self._subset(self.test_data, self.test_targets, classes)

    def get_cumulative_test_set(self, up_to_task: int):
        classes = [
            value
            for task in self._task_classes[: up_to_task + 1]
            for value in task
        ]
        return self._subset(self.test_data, self.test_targets, classes)

    @staticmethod
    def _subset(dataset, targets: np.ndarray, classes: list[int]):
        indices = np.where(np.isin(targets, classes))[0].tolist()
        return Subset(dataset, indices)


class Flowers102CL(TorchvisionFineGrainedCL):
    def __init__(self, **kwargs):
        super().__init__(dataset="flowers102", **kwargs)


class OxfordIIITPetCL(TorchvisionFineGrainedCL):
    def __init__(self, **kwargs):
        super().__init__(dataset="oxfordiiitpet", **kwargs)
