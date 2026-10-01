"""CUB feature and classification utilities for reported topology experiments."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import random
import subprocess
import sys
import tarfile
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import models
PROJECT_ROOT = Path(__file__).resolve().parents[1]
from ssr_utils.result_schema import build_result_record
from ssr_utils.dataset_fingerprint import cub_dataset_fingerprint
CUB_IMAGES_URL = 'https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1'
CUB_SEG_URL = 'https://data.caltech.edu/records/w9d68-gec53/files/segmentations.tgz?download=1'
KERNEL_FAMILIES = ('gaussian', 'laplace', 'cauchy', 'inverse')

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def download_file(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 1024:
        return
    tmp = path.with_suffix(path.suffix + '.tmp')
    print(f'[download] {url} -> {path}', flush=True)
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(path)

def ensure_cub(data_root: Path, download: bool=True) -> Path:
    data_root.mkdir(parents=True, exist_ok=True)
    cub_dir = data_root / 'CUB_200_2011'
    seg_dir = data_root / 'segmentations'
    if cub_dir.exists() and seg_dir.exists():
        return cub_dir
    if not download:
        raise FileNotFoundError(f'CUB files not found under {data_root}')
    images_tgz = data_root / 'CUB_200_2011.tgz'
    seg_tgz = data_root / 'segmentations.tgz'
    download_file(CUB_IMAGES_URL, images_tgz)
    download_file(CUB_SEG_URL, seg_tgz)
    if not cub_dir.exists():
        print(f'[extract] {images_tgz}', flush=True)
        with tarfile.open(images_tgz, 'r:gz') as tar:
            tar.extractall(data_root)
    if not seg_dir.exists():
        print(f'[extract] {seg_tgz}', flush=True)
        with tarfile.open(seg_tgz, 'r:gz') as tar:
            tar.extractall(data_root)
    return cub_dir

def read_cub_metadata(cub_dir: Path, max_classes: int | None=None) -> list[dict]:
    images = {}
    with (cub_dir / 'images.txt').open('r', encoding='utf-8') as f:
        for line in f:
            idx, rel = line.strip().split()
            images[int(idx)] = rel
    labels = {}
    with (cub_dir / 'image_class_labels.txt').open('r', encoding='utf-8') as f:
        for line in f:
            idx, cls = line.strip().split()
            labels[int(idx)] = int(cls) - 1
    splits = {}
    with (cub_dir / 'train_test_split.txt').open('r', encoding='utf-8') as f:
        for line in f:
            idx, is_train = line.strip().split()
            splits[int(idx)] = int(is_train)
    rows = []
    for idx in sorted(images):
        y = labels[idx]
        if max_classes is not None and y >= max_classes:
            continue
        rows.append({'id': idx, 'relpath': images[idx], 'label': y, 'is_train': bool(splits[idx])})
    return rows

def _validate_kernel_parameters(a_exc, a_inh, sigma_exc, sigma_inh, kernel_family: str) -> str:
    if not isinstance(kernel_family, str):
        raise TypeError('kernel_family must be a string')
    family = kernel_family.lower()
    if family not in KERNEL_FAMILIES:
        choices = ', '.join(KERNEL_FAMILIES)
        raise ValueError(f"Unknown kernel_family '{kernel_family}'. Expected one of: {choices}")
    for name, value in (('a_exc', a_exc), ('a_inh', a_inh)):
        try:
            finite = math.isfinite(value)
        except TypeError as exc:
            raise TypeError(f'{name} must be a finite number') from exc
        if not finite or value < 0:
            raise ValueError(f'{name} must be finite and non-negative')
    for name, value in (('sigma_exc', sigma_exc), ('sigma_inh', sigma_inh)):
        try:
            finite = math.isfinite(value)
        except TypeError as exc:
            raise TypeError(f'{name} must be a finite number') from exc
        if not finite or value <= 0:
            raise ValueError(f'{name} must be finite and positive')
    return family

def _radial_response(dist: torch.Tensor, sigma: float, kernel_family: str) -> torch.Tensor:
    """Evaluate the classification-study radial response.

    For backward compatibility with the recorded classification runs, the
    ``inverse`` key here denotes ``1 / (1 + distance / sigma)``. The adapter
    and LLM studies use a separately documented inverse-square-root response.
    """
    if kernel_family == 'gaussian':
        return torch.exp(-dist ** 2 / (2 * sigma ** 2))
    if kernel_family == 'laplace':
        return torch.exp(-dist / sigma)
    if kernel_family == 'cauchy':
        return 1.0 / (1.0 + (dist / sigma) ** 2)
    if kernel_family == 'inverse':
        return 1.0 / (1.0 + dist / sigma)
    raise ValueError(f'Unknown kernel_family={kernel_family!r}')

def biocs_loss(weights: torch.Tensor, a_exc=1.0, a_inh=0.8, sigma_exc=0.2, sigma_inh=0.5, kernel_family: str='gaussian') -> torch.Tensor:
    family = _validate_kernel_parameters(a_exc, a_inh, sigma_exc, sigma_inh, kernel_family)
    if weights.shape[0] < 2:
        return weights.sum() * 0.0
    w = F.normalize(weights, dim=1)
    cos = torch.clamp(w @ w.T, -1.0, 1.0)
    dist = torch.sqrt(torch.clamp(1.0 - cos, min=1e-08))
    inh = a_inh * _radial_response(dist, sigma_inh, family)
    exc = a_exc * _radial_response(dist, sigma_exc, family)
    penalty = inh - exc + (a_exc - a_inh)
    penalty = penalty - torch.diag(torch.diag(penalty))
    n = weights.shape[0]
    return penalty.sum() / (n * (n - 1) + 1e-08)

def geometry(weight: torch.Tensor) -> dict[str, float]:
    w = weight.detach().float().cpu()
    sv = torch.linalg.svdvals(w)
    p = sv / (sv.sum() + 1e-12)
    eff_rank = float(torch.exp(-(p * torch.log(p + 1e-12)).sum()).item())
    wn = F.normalize(w, dim=1)
    cos = wn @ wn.T
    offdiag = cos[~torch.eye(cos.shape[0], dtype=torch.bool)]
    return {'effective_rank': eff_rank, 'mean_abs_offdiag_cosine': float(offdiag.abs().mean().item())}

class CubImageDataset(Dataset):

    def __init__(self, cub_dir: Path, rows: list[dict], transform):
        self.cub_dir = cub_dir
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        img = Image.open(self.cub_dir / 'images' / row['relpath']).convert('RGB')
        return (self.transform(img), row['label'], row['relpath'])

def extract_classification_features(args: argparse.Namespace, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    cache = Path(args.classification_cache)
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        payload = torch.load(cache, map_location='cpu')
        return (payload['x_train'], payload['y_train'], payload['x_test'], payload['y_test'])
    cub_dir = ensure_cub(Path(args.data_root), download=not args.no_download)
    rows = read_cub_metadata(cub_dir, args.max_classes)
    train_rows = [r for r in rows if r['is_train']]
    test_rows = [r for r in rows if not r['is_train']]
    weights = models.ResNet18_Weights.DEFAULT
    transform = weights.transforms()
    backbone = models.resnet18(weights=weights)
    backbone.fc = nn.Identity()
    backbone = backbone.to(device).eval()

    def collect(split_rows: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
        loader = DataLoader(CubImageDataset(cub_dir, split_rows, transform), batch_size=args.feature_batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
        xs, ys = ([], [])
        with torch.no_grad():
            for xb, yb, _ in loader:
                xs.append(backbone(xb.to(device)).cpu())
                ys.append(torch.as_tensor(yb, dtype=torch.long))
        return (torch.cat(xs), torch.cat(ys))
    x_train, y_train = collect(train_rows)
    x_test, y_test = collect(test_rows)
    torch.save({'x_train': x_train, 'y_train': y_train, 'x_test': x_test, 'y_test': y_test}, cache)
    return (x_train, y_train, x_test, y_test)

def make_tasks(num_classes: int, classes_per_task: int, seed: int, class_order: str) -> list[list[int]]:
    order = list(range(num_classes))
    if class_order == 'random':
        random.Random(seed).shuffle(order)
    return [order[i:i + classes_per_task] for i in range(0, num_classes, classes_per_task) if len(order[i:i + classes_per_task]) == classes_per_task]

def subset_loader(x: torch.Tensor, y: torch.Tensor, classes: list[int], batch_size: int, shuffle: bool) -> DataLoader:
    mask = torch.zeros_like(y, dtype=torch.bool)
    for cls in classes:
        mask |= y == cls
    idx = torch.where(mask)[0]
    return DataLoader(TensorDataset(x[idx], y[idx]), batch_size=batch_size, shuffle=shuffle)

class LinearHead(nn.Module):

    def __init__(self, dim: int, num_classes: int, tau: float=12.0):
        super().__init__()
        self.classifier = nn.Linear(dim, num_classes, bias=False)
        self.tau = tau

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tau * F.linear(F.normalize(x, dim=1), F.normalize(self.classifier.weight, dim=1))

def masked_argmax(logits: torch.Tensor, allowed: list[int]) -> torch.Tensor:
    mask = torch.full_like(logits, -1000000000.0)
    mask[:, allowed] = logits[:, allowed]
    return mask.argmax(dim=1)

def evaluate_classification_til(model: nn.Module, x_test: torch.Tensor, y_test: torch.Tensor, tasks: list[list[int]], seen: int, batch_size: int, device: torch.device) -> list[float]:
    model.eval()
    accs = []
    with torch.no_grad():
        for tid in range(seen):
            loader = subset_loader(x_test, y_test, tasks[tid], batch_size, False)
            correct = total = 0
            for xb, yb in loader:
                pred = masked_argmax(model(xb.to(device)), tasks[tid]).cpu()
                correct += (pred == yb).sum().item()
                total += yb.numel()
            accs.append(100.0 * correct / max(total, 1))
    return accs

def evaluate_classification_cil(model: nn.Module, x_test: torch.Tensor, y_test: torch.Tensor, seen_classes: list[int], batch_size: int, device: torch.device) -> float:
    model.eval()
    loader = subset_loader(x_test, y_test, seen_classes, batch_size, False)
    correct = total = 0
    with torch.no_grad():
        for xb, yb in loader:
            pred = masked_argmax(model(xb.to(device)), seen_classes).cpu()
            correct += (pred == yb).sum().item()
            total += yb.numel()
    return 100.0 * correct / max(total, 1)
SEGMENTATION_HEADS = ('additive', 'prototype_cosine')
