"""Split-TinyImageNet benchmark for continual learning.

TinyImageNet: 200 classes, 64x64 images, 500 train / 50 val per class.
Split into sequential tasks for CIL evaluation.
"""

import logging
import os
import zipfile
import numpy as np
from pathlib import Path
from torch.utils.data import Subset
from torchvision import datasets, transforms

logger = logging.getLogger(__name__)

TINYIMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"


def _download_tiny_imagenet(data_root: str):
    """Download and extract TinyImageNet if not present."""
    root = Path(data_root)
    tinydir = root / "tiny-imagenet-200"
    if tinydir.exists() and (tinydir / "train").exists():
        return str(tinydir)

    root.mkdir(parents=True, exist_ok=True)
    zip_path = root / "tiny-imagenet-200.zip"

    if not zip_path.exists():
        import urllib.request
        logger.info(f"Downloading TinyImageNet to {zip_path} ...")
        urllib.request.urlretrieve(TINYIMAGENET_URL, str(zip_path))
        logger.info("Download complete.")

    if not tinydir.exists():
        logger.info("Extracting TinyImageNet...")
        with zipfile.ZipFile(str(zip_path), 'r') as z:
            z.extractall(str(root))
        logger.info("Extraction complete.")

    _reorganize_val(tinydir)
    return str(tinydir)


def _reorganize_val(tinydir: Path):
    """Reorganize val folder into class-subfolder structure for ImageFolder."""
    val_dir = tinydir / "val"
    val_annotations = val_dir / "val_annotations.txt"
    if not val_annotations.exists():
        return

    images_dir = val_dir / "images"
    if not images_dir.exists():
        return

    with open(val_annotations) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
            fname, class_id = parts[0], parts[1]
            class_dir = val_dir / class_id / "images"
            class_dir.mkdir(parents=True, exist_ok=True)
            src = images_dir / fname
            dst = class_dir / fname
            if src.exists() and not dst.exists():
                src.rename(dst)

    if images_dir.exists() and not any(images_dir.iterdir()):
        images_dir.rmdir()
    if val_annotations.exists():
        val_annotations.unlink(missing_ok=True)


class SplitTinyImageNet:
    """Split-TinyImageNet: 200 classes → n_tasks sequential tasks."""

    def __init__(self, n_tasks: int = 10, data_root: str = "./dataset", **kwargs):
        self.n_tasks = n_tasks
        self.n_classes = 200
        self.classes_per_task = self.n_classes // n_tasks

        tinydir = _download_tiny_imagenet(data_root)

        train_transform = transforms.Compose([
            transforms.RandomCrop(64, padding=8),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)),
        ])

        self.train_data = datasets.ImageFolder(os.path.join(tinydir, "train"), transform=train_transform)
        self.test_data = datasets.ImageFolder(os.path.join(tinydir, "val"), transform=test_transform)

        self.train_targets = np.array([s[1] for s in self.train_data.samples])
        self.test_targets = np.array([s[1] for s in self.test_data.samples])

        self.class_order = list(range(self.n_classes))
        self._current_task = 0
        logger.info(f"SplitTinyImageNet: {self.n_classes} classes, {n_tasks} tasks, "
                     f"{self.classes_per_task} cls/task, train={len(self.train_data)}, test={len(self.test_data)}")

    def __iter__(self):
        self._current_task = 0
        return self

    def __next__(self):
        if self._current_task >= self.n_tasks:
            raise StopIteration
        tid = self._current_task
        self._current_task += 1
        classes = self._get_task_classes(tid)
        train_sub = self._get_subset(self.train_data, self.train_targets, classes)
        test_sub = self._get_subset(self.test_data, self.test_targets, classes)
        logger.info(f"Task {tid+1}: classes={classes[:5]}..., train={len(train_sub)}, test={len(test_sub)}")
        return train_sub, test_sub

    def get_test_set(self, task_id):
        return self._get_subset(self.test_data, self.test_targets, self._get_task_classes(task_id))

    def get_cumulative_test_set(self, up_to_task):
        all_c = []
        for t in range(up_to_task + 1):
            all_c.extend(self._get_task_classes(t))
        return self._get_subset(self.test_data, self.test_targets, all_c)

    def _get_task_classes(self, tid):
        s = tid * self.classes_per_task
        return self.class_order[s:s + self.classes_per_task]

    @staticmethod
    def _get_subset(dataset, targets, classes):
        mask = np.isin(targets, classes)
        return Subset(dataset, np.where(mask)[0].tolist())
