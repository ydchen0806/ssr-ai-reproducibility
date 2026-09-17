"""Split-CIFAR benchmarks for continual learning."""

import logging
import numpy as np
from torch.utils.data import Subset, ConcatDataset
from torchvision import datasets, transforms

logger = logging.getLogger(__name__)


class SplitCIFAR100:
    """Split-CIFAR100: 100 classes split into sequential tasks.

    Supports configurable class ordering via seed, and cumulative test set
    construction for proper CIL evaluation.
    """

    def __init__(self, n_tasks: int = 10, data_root: str = "./data",
                 class_order_seed: int | None = None, **kwargs):
        self.n_tasks = n_tasks
        self.n_classes = 100
        self.classes_per_task = self.n_classes // n_tasks

        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        self.train_data = datasets.CIFAR100(
            data_root, train=True, download=True, transform=train_transform
        )
        self.test_data = datasets.CIFAR100(
            data_root, train=False, download=True, transform=test_transform
        )

        self.train_targets = np.array(self.train_data.targets)
        self.test_targets = np.array(self.test_data.targets)

        self.class_order = list(range(self.n_classes))
        if class_order_seed is not None:
            rng = np.random.RandomState(class_order_seed)
            rng.shuffle(self.class_order)

        self._current_task = 0
        self._log_stats()

    def _log_stats(self):
        logger.info(
            f"SplitCIFAR100: {self.n_classes} classes, {self.n_tasks} tasks, "
            f"{self.classes_per_task} classes/task, "
            f"train={len(self.train_data)}, test={len(self.test_data)}"
        )
        logger.info(f"Class order (first 20): {self.class_order[:20]}...")

    def __iter__(self):
        self._current_task = 0
        return self

    def __next__(self):
        if self._current_task >= self.n_tasks:
            raise StopIteration
        task_id = self._current_task
        self._current_task += 1

        classes = self._get_task_classes(task_id)
        train_subset = self._get_subset(self.train_data, self.train_targets, classes)
        test_subset = self._get_subset(self.test_data, self.test_targets, classes)
        logger.info(
            f"Task {task_id+1}: classes={classes}, "
            f"train={len(train_subset)}, test={len(test_subset)}"
        )
        return train_subset, test_subset

    def get_test_set(self, task_id: int):
        """Get test set for a single task."""
        classes = self._get_task_classes(task_id)
        return self._get_subset(self.test_data, self.test_targets, classes)

    def get_cumulative_test_set(self, up_to_task: int):
        """Get combined test set for all tasks up to (inclusive) the given task.

        This is the proper evaluation protocol for CIL: the model must
        classify among ALL classes seen so far.
        """
        all_classes = []
        for t in range(up_to_task + 1):
            all_classes.extend(self._get_task_classes(t))
        return self._get_subset(self.test_data, self.test_targets, all_classes)

    def _get_task_classes(self, task_id: int) -> list[int]:
        start = task_id * self.classes_per_task
        end = start + self.classes_per_task
        return self.class_order[start:end]

    @staticmethod
    def _get_subset(dataset, targets, classes):
        mask = np.isin(targets, classes)
        indices = np.where(mask)[0].tolist()
        return Subset(dataset, indices)


class SplitCIFAR10:
    """Split-CIFAR10: 10 classes split into sequential tasks."""

    def __init__(self, n_tasks: int = 5, data_root: str = "./data",
                 class_order_seed: int | None = None, **kwargs):
        self.n_tasks = n_tasks
        self.n_classes = 10
        self.classes_per_task = self.n_classes // n_tasks

        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])

        self.train_data = datasets.CIFAR10(
            data_root, train=True, download=True, transform=train_transform
        )
        self.test_data = datasets.CIFAR10(
            data_root, train=False, download=True, transform=test_transform
        )

        self.train_targets = np.array(self.train_data.targets)
        self.test_targets = np.array(self.test_data.targets)

        self.class_order = list(range(self.n_classes))
        if class_order_seed is not None:
            rng = np.random.RandomState(class_order_seed)
            rng.shuffle(self.class_order)

        self._current_task = 0

    def __iter__(self):
        self._current_task = 0
        return self

    def __next__(self):
        if self._current_task >= self.n_tasks:
            raise StopIteration
        task_id = self._current_task
        self._current_task += 1

        classes = self._get_task_classes(task_id)
        train_subset = SplitCIFAR100._get_subset(self.train_data, self.train_targets, classes)
        test_subset = SplitCIFAR100._get_subset(self.test_data, self.test_targets, classes)
        return train_subset, test_subset

    def get_test_set(self, task_id: int):
        classes = self._get_task_classes(task_id)
        return SplitCIFAR100._get_subset(self.test_data, self.test_targets, classes)

    def get_cumulative_test_set(self, up_to_task: int):
        all_classes = []
        for t in range(up_to_task + 1):
            all_classes.extend(self._get_task_classes(t))
        return SplitCIFAR100._get_subset(self.test_data, self.test_targets, all_classes)

    def _get_task_classes(self, task_id: int) -> list[int]:
        start = task_id * self.classes_per_task
        end = start + self.classes_per_task
        return self.class_order[start:end]
