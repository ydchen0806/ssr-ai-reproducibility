"""5-Datasets benchmark for continual learning (Domain-Incremental).

Sequential training on 5 completely different datasets:
  Task 1: CIFAR-10       (10 classes, natural images)
  Task 2: MNIST          (10 classes, handwritten digits)
  Task 3: FashionMNIST   (10 classes, clothing items)
  Task 4: SVHN           (10 classes, street view house numbers)
  Task 5: CIFAR-100 (first 10 superclasses → 10 classes)

This is a domain-incremental benchmark: each task has a different data
distribution. Tests whether SSR can maintain cross-domain representations
without catastrophic interference across very different visual domains.
"""

import logging
import numpy as np
import torch
from torch.utils.data import Subset, Dataset
from torchvision import datasets, transforms

logger = logging.getLogger(__name__)


class GrayscaleToRGB:
    """Convert 1-channel grayscale to 3-channel RGB."""
    def __call__(self, x):
        if x.shape[0] == 1:
            return x.repeat(3, 1, 1)
        return x


class ResizeTo32:
    """Resize to 32x32 if needed."""
    def __call__(self, x):
        if x.shape[1] != 32 or x.shape[2] != 32:
            return torch.nn.functional.interpolate(x.unsqueeze(0), size=(32, 32), mode='bilinear', align_corners=False).squeeze(0)
        return x


class OffsetLabelDataset(Dataset):
    """Adds a class offset so each task's labels don't overlap."""
    def __init__(self, base_dataset, offset: int):
        self.base = base_dataset
        self.offset = offset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        return x, y + self.offset


class FiveDatasets:
    """5-Datasets: 5 tasks × 10 classes = 50 total classes.

    Each task is a completely different dataset/domain.
    """

    DATASET_NAMES = ["CIFAR-10", "MNIST", "FashionMNIST", "SVHN", "CIFAR-100-sub"]

    def __init__(self, data_root: str = "./dataset", **kwargs):
        self.n_tasks = 5
        self.n_classes = 50
        self.classes_per_task = 10
        self.data_root = data_root

        self._datasets = self._build_datasets()
        self._current_task = 0
        logger.info(f"FiveDatasets: 5 tasks, 50 classes total")
        for i, name in enumerate(self.DATASET_NAMES):
            tr, te = self._datasets[i]
            logger.info(f"  Task {i+1} ({name}): train={len(tr)}, test={len(te)}")

    def _build_datasets(self):
        result = []

        cifar10_tr_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        cifar10_te_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])

        gray_tr_tf = transforms.Compose([
            transforms.Resize(32),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            GrayscaleToRGB(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        gray_te_tf = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            GrayscaleToRGB(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        svhn_tr_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
        ])
        svhn_te_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
        ])

        # Task 1: CIFAR-10
        tr = datasets.CIFAR10(self.data_root, train=True, download=True, transform=cifar10_tr_tf)
        te = datasets.CIFAR10(self.data_root, train=False, download=True, transform=cifar10_te_tf)
        result.append((OffsetLabelDataset(tr, 0), OffsetLabelDataset(te, 0)))

        # Task 2: MNIST
        tr = datasets.MNIST(self.data_root, train=True, download=True, transform=gray_tr_tf)
        te = datasets.MNIST(self.data_root, train=False, download=True, transform=gray_te_tf)
        result.append((OffsetLabelDataset(tr, 10), OffsetLabelDataset(te, 10)))

        # Task 3: FashionMNIST
        tr = datasets.FashionMNIST(self.data_root, train=True, download=True, transform=gray_tr_tf)
        te = datasets.FashionMNIST(self.data_root, train=False, download=True, transform=gray_te_tf)
        result.append((OffsetLabelDataset(tr, 20), OffsetLabelDataset(te, 20)))

        # Task 4: SVHN
        tr = datasets.SVHN(self.data_root, split='train', download=True, transform=svhn_tr_tf)
        te = datasets.SVHN(self.data_root, split='test', download=True, transform=svhn_te_tf)
        result.append((OffsetLabelDataset(tr, 30), OffsetLabelDataset(te, 30)))

        # Task 5: CIFAR-100 subset (first 10 classes)
        c100_tr = datasets.CIFAR100(self.data_root, train=True, download=True, transform=cifar10_tr_tf)
        c100_te = datasets.CIFAR100(self.data_root, train=False, download=True, transform=cifar10_te_tf)
        c100_tr_tgt = np.array(c100_tr.targets)
        c100_te_tgt = np.array(c100_te.targets)
        mask_tr = np.isin(c100_tr_tgt, list(range(10)))
        mask_te = np.isin(c100_te_tgt, list(range(10)))
        c100_tr_sub = Subset(c100_tr, np.where(mask_tr)[0].tolist())
        c100_te_sub = Subset(c100_te, np.where(mask_te)[0].tolist())
        result.append((OffsetLabelDataset(c100_tr_sub, 40), OffsetLabelDataset(c100_te_sub, 40)))

        return result

    def __iter__(self):
        self._current_task = 0
        return self

    def __next__(self):
        if self._current_task >= self.n_tasks:
            raise StopIteration
        tid = self._current_task
        self._current_task += 1
        tr, te = self._datasets[tid]
        logger.info(f"Task {tid+1} ({self.DATASET_NAMES[tid]}): train={len(tr)}, test={len(te)}")
        return tr, te

    def get_test_set(self, task_id):
        return self._datasets[task_id][1]

    def get_cumulative_test_set(self, up_to_task):
        from torch.utils.data import ConcatDataset
        return ConcatDataset([self._datasets[t][1] for t in range(up_to_task + 1)])

    def _get_task_classes(self, task_id: int) -> list[int]:
        start = task_id * self.classes_per_task
        return list(range(start, start + self.classes_per_task))
