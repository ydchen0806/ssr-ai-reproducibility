"""Knowledge Editing benchmark for continual learning.

Proper KE evaluation following the framework from:
  - ROME (Meng et al., NeurIPS 2022)
  - MEMIT (Meng et al., ICLR 2023)
  - KnowEdit (Zhang et al., 2024)

Adapted for vision models: sequential class-mapping edits on CIFAR-100.

Metrics:
  - Efficacy: accuracy on edited classes with new labels
  - Locality: accuracy retention on non-edited classes
  - Edit Success Rate: fraction of edits that achieve >50% efficacy
"""

import logging
import numpy as np
import torch
from torch.utils.data import Subset, Dataset
from torchvision import datasets, transforms

logger = logging.getLogger(__name__)


class LabelFlipDataset(Dataset):
    """Wraps a subset and remaps labels according to a mapping."""

    def __init__(self, base_dataset, label_map: dict):
        self.base = base_dataset
        self.label_map = label_map

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        if y in self.label_map:
            y = self.label_map[y]
        return x, y


class KnowledgeEditingBenchmark:
    """Sequential knowledge editing on CIFAR-100.

    Protocol:
      Task 0: "Pre-train" on all 100 classes (standard training)
      Task 1..N: Edit n_edit_classes per round (relabel → fine-tune)

    Each edit round:
      1. Pick n_edit_classes, create a label remapping
      2. Fine-tune on the remapped subset
      3. Evaluate:
         - Efficacy: accuracy on edited-class samples using new labels
         - Locality: accuracy on non-edited classes (should stay high)
    """

    def __init__(
        self,
        n_edit_rounds: int = 5,
        n_edit_classes: int = 10,
        data_root: str = "./dataset",
        **kwargs,
    ):
        self.n_tasks = 1 + n_edit_rounds
        self.n_edit_rounds = n_edit_rounds
        self.n_edit_classes = n_edit_classes
        self.n_classes = 100
        self.classes_per_task = n_edit_classes

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

        self.train_data = datasets.CIFAR100(data_root, train=True, download=True, transform=train_transform)
        self.test_data = datasets.CIFAR100(data_root, train=False, download=True, transform=test_transform)

        self.train_targets = np.array(self.train_data.targets)
        self.test_targets = np.array(self.test_data.targets)

        rng = np.random.RandomState(42)
        all_classes = list(range(100))
        rng.shuffle(all_classes)

        self._edit_schedule = []
        self._cumulative_label_map = {}
        for t in range(n_edit_rounds):
            src_start = (t * n_edit_classes) % 100
            tgt_start = (src_start + 50) % 100
            src_classes = all_classes[src_start:src_start + n_edit_classes]
            tgt_classes = all_classes[tgt_start:tgt_start + n_edit_classes]
            if len(tgt_classes) < n_edit_classes:
                tgt_classes = tgt_classes + all_classes[:n_edit_classes - len(tgt_classes)]
            label_map = {}
            for i in range(min(len(src_classes), len(tgt_classes))):
                label_map[src_classes[i]] = tgt_classes[i]
            self._edit_schedule.append((src_classes, label_map))

        self._current_task = 0
        logger.info(f"KnowledgeEditing: 1 pretrain + {n_edit_rounds} edit rounds, "
                     f"{n_edit_classes} classes/round")

    def __iter__(self):
        self._current_task = 0
        return self

    def __next__(self):
        if self._current_task >= self.n_tasks:
            raise StopIteration
        task_id = self._current_task
        self._current_task += 1

        if task_id == 0:
            logger.info(f"Task 0 (pretrain): full CIFAR-100, {len(self.train_data)} samples")
            return self.train_data, self.test_data

        edit_idx = task_id - 1
        edit_classes, label_map = self._edit_schedule[edit_idx]
        self._cumulative_label_map.update(label_map)

        mask = np.isin(self.train_targets, edit_classes)
        indices = np.where(mask)[0].tolist()
        edit_subset = Subset(self.train_data, indices)
        edit_dataset = LabelFlipDataset(edit_subset, label_map)

        logger.info(f"Edit round {edit_idx+1}: editing classes {edit_classes[:5]}..., "
                     f"train={len(edit_dataset)}, label_map has {len(label_map)} entries")
        return edit_dataset, self.test_data

    def get_test_set(self, task_id: int):
        return self.test_data

    def get_cumulative_test_set(self, up_to_task: int):
        return self.test_data
