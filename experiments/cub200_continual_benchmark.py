#!/usr/bin/env python3
"""CUB-200-2011 continual fine-grained classification and segmentation.

The script is intentionally self-contained so it can run outside the main CL
trainer. It downloads the official CUB-200-2011 images/annotations and
segmentation masks, then evaluates Spatial Synaptic Regularization (SSR) in two settings:

1. Frozen ImageNet ResNet-18 features + sequential classifier head.
2. Frozen ImageNet ResNet-18 dense features + class-conditioned mask decoder.

The segmentation task is not camouflaged-object detection, but it is a genuine
fine-grained segmentation benchmark with per-pixel masks over 200 bird species.
"""

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
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ssr_utils.result_schema import build_result_record
from ssr_utils.segmentation_kd import old_class_distillation_ids
from ssr_utils.dataset_fingerprint import cub_dataset_fingerprint
from ssr_utils.segmentation_nested_search import (
    cross_set_ssr_loss,
    scale_to_hwhm,
    stratified_fit_validation_indices,
)


CUB_IMAGES_URL = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1"
CUB_SEG_URL = "https://data.caltech.edu/records/w9d68-gec53/files/segmentations.tgz?download=1"
KERNEL_FAMILIES = ("gaussian", "laplace", "cauchy", "inverse")


def current_git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown-uncommitted-environment"


def state_dict_sha256(module: nn.Module) -> str:
    """Hash initialized trainable state without relying on serialization bytes."""
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def task_schedule_sha256(tasks: list[list[int]]) -> str:
    return hashlib.sha256(
        json.dumps(tasks, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    ).hexdigest()


def objective_for_method(method: str) -> tuple[str, dict[str, bool]]:
    mapping = {
        "baseline": ("plain", {"task": True}),
        "kd": ("kd", {"task": True, "kd": True}),
        "biocs": ("ssr_only", {"task": True, "ssr": True}),
        "biocs_kd": ("ssr_kd", {"task": True, "ssr": True, "kd": True}),
    }
    return mapping.get(method, (f"{method}_context", {"task": True}))


def write_result_record(
    *,
    outdir: Path,
    row: dict,
    args: argparse.Namespace,
    elapsed_s: float,
    dataset_hash: str,
) -> None:
    recipe, objective = objective_for_method(row["method"])
    uses_ssr = bool(objective.get("ssr"))
    kernel = {}
    if uses_ssr:
        kernel = {
            "family": args.kernel_family,
            "A_exc": args.a_exc,
            "A_inh": args.a_inh,
            "sigma_exc": args.sigma_exc,
            "sigma_inh": args.sigma_inh,
        }
    audit_fields = {
        "initial_model_hash",
        "task_schedule_hash",
        "optimizer_steps",
    }
    metrics = {
        key: float(value)
        for key, value in row.items()
        if key not in {"task", "method", "seed", *audit_fields}
        and isinstance(value, (int, float))
    }
    config_payload = dict(vars(args))
    if row["task"] == "segmentation":
        config_payload["segmentation_kd_protocol"] = (
            "old_class_conditions_on_current_features_v1"
        )
    model_name = "resnet18_classifier"
    model_config = None
    if row["task"] == "segmentation":
        model_name = (
            "resnet18_dense_prototype_cosine_decoder"
            if args.segmentation_head == "prototype_cosine"
            else "resnet18_dense_decoder"
        )
        model_config = segmentation_head_config(
            args.segmentation_head,
            args.seg_hidden_dim,
            prototype_logit_scale=args.prototype_logit_scale,
        )
    record = build_result_record(
        git_commit=current_git_commit(),
        run_id=str((outdir / row["task"] / row["method"] / f"seed_{row['seed']}").resolve()),
        task_family=row["task"],
        dataset="cub200_masks" if row["task"] == "segmentation" else "cub200",
        model=model_name,
        seed=int(row["seed"]),
        objective=objective,
        distance_mapping="cosine" if uses_ssr else "none",
        kernel=kernel,
        metrics=metrics,
        runtime={"elapsed_s": elapsed_s},
        config=config_payload,
        dataset_hash=dataset_hash,
        recipe=recipe,
        selection_lock_hash=(
            args.selection_lock_sha256
            if row["task"] == "segmentation"
            else "not_applicable"
        ),
        lambda_ssr=float(args.lambda_sp) if uses_ssr else 0.0,
        metric_directions={
            "mean_iou": True,
            "mean_dice": True,
            "avg_forgetting_iou": False,
            "avg_accuracy": True,
            "avg_forgetting": False,
            "effective_rank": True,
            "mean_abs_offdiag_cosine": False,
        },
        **({"model_config": model_config} if model_config is not None else {}),
        **{
            key: row[key]
            for key in audit_fields
            if key in row
        },
    )
    path = outdir / "result_records" / row["task"] / row["method"] / f"seed_{row['seed']}" / "result_record.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")


@dataclass
class ClassificationRun:
    task: str
    method: str
    seed: int
    avg_accuracy: float
    avg_cil_accuracy: float
    avg_forgetting: float
    final_til_accuracy_seen: float
    final_cil_accuracy_seen: float
    til_capacity_at_80: int
    til_capacity_at_75: int
    til_capacity_at_70: int
    cil_capacity_at_20: int
    effective_rank: float
    mean_abs_offdiag_cosine: float


@dataclass
class SegmentationRun:
    task: str
    method: str
    seed: int
    mean_iou: float
    mean_dice: float
    avg_forgetting_iou: float
    effective_rank: float
    mean_abs_offdiag_cosine: float
    initial_model_hash: str
    task_schedule_hash: str
    optimizer_steps: int


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
    tmp = path.with_suffix(path.suffix + ".tmp")
    print(f"[download] {url} -> {path}", flush=True)
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(path)


def ensure_cub(data_root: Path, download: bool = True) -> Path:
    data_root.mkdir(parents=True, exist_ok=True)
    cub_dir = data_root / "CUB_200_2011"
    seg_dir = data_root / "segmentations"
    if cub_dir.exists() and seg_dir.exists():
        return cub_dir
    if not download:
        raise FileNotFoundError(f"CUB files not found under {data_root}")
    images_tgz = data_root / "CUB_200_2011.tgz"
    seg_tgz = data_root / "segmentations.tgz"
    download_file(CUB_IMAGES_URL, images_tgz)
    download_file(CUB_SEG_URL, seg_tgz)
    if not cub_dir.exists():
        print(f"[extract] {images_tgz}", flush=True)
        with tarfile.open(images_tgz, "r:gz") as tar:
            tar.extractall(data_root)
    if not seg_dir.exists():
        print(f"[extract] {seg_tgz}", flush=True)
        with tarfile.open(seg_tgz, "r:gz") as tar:
            tar.extractall(data_root)
    return cub_dir


def read_cub_metadata(cub_dir: Path, max_classes: int | None = None) -> list[dict]:
    images = {}
    with (cub_dir / "images.txt").open("r", encoding="utf-8") as f:
        for line in f:
            idx, rel = line.strip().split()
            images[int(idx)] = rel
    labels = {}
    with (cub_dir / "image_class_labels.txt").open("r", encoding="utf-8") as f:
        for line in f:
            idx, cls = line.strip().split()
            labels[int(idx)] = int(cls) - 1
    splits = {}
    with (cub_dir / "train_test_split.txt").open("r", encoding="utf-8") as f:
        for line in f:
            idx, is_train = line.strip().split()
            splits[int(idx)] = int(is_train)
    rows = []
    for idx in sorted(images):
        y = labels[idx]
        if max_classes is not None and y >= max_classes:
            continue
        rows.append({"id": idx, "relpath": images[idx], "label": y, "is_train": bool(splits[idx])})
    return rows


def _validate_kernel_parameters(a_exc, a_inh, sigma_exc, sigma_inh, kernel_family: str) -> str:
    if not isinstance(kernel_family, str):
        raise TypeError("kernel_family must be a string")
    family = kernel_family.lower()
    if family not in KERNEL_FAMILIES:
        choices = ", ".join(KERNEL_FAMILIES)
        raise ValueError(f"Unknown kernel_family '{kernel_family}'. Expected one of: {choices}")
    for name, value in (("a_exc", a_exc), ("a_inh", a_inh)):
        try:
            finite = math.isfinite(value)
        except TypeError as exc:
            raise TypeError(f"{name} must be a finite number") from exc
        if not finite or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    for name, value in (("sigma_exc", sigma_exc), ("sigma_inh", sigma_inh)):
        try:
            finite = math.isfinite(value)
        except TypeError as exc:
            raise TypeError(f"{name} must be a finite number") from exc
        if not finite or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    return family


def _radial_response(dist: torch.Tensor, sigma: float, kernel_family: str) -> torch.Tensor:
    """Evaluate the classification-study radial response.

    For backward compatibility with the recorded classification runs, the
    ``inverse`` key here denotes ``1 / (1 + distance / sigma)``. The adapter
    and LLM studies use a separately documented inverse-square-root response.
    """
    if kernel_family == "gaussian":
        return torch.exp(-(dist ** 2) / (2 * sigma ** 2))
    if kernel_family == "laplace":
        return torch.exp(-dist / sigma)
    if kernel_family == "cauchy":
        return 1.0 / (1.0 + (dist / sigma) ** 2)
    if kernel_family == "inverse":
        return 1.0 / (1.0 + dist / sigma)
    raise ValueError(f"Unknown kernel_family={kernel_family!r}")


def biocs_loss(
    weights: torch.Tensor,
    a_exc=1.0,
    a_inh=0.8,
    sigma_exc=0.2,
    sigma_inh=0.5,
    kernel_family: str = "gaussian",
) -> torch.Tensor:
    family = _validate_kernel_parameters(
        a_exc, a_inh, sigma_exc, sigma_inh, kernel_family,
    )
    if weights.shape[0] < 2:
        return weights.sum() * 0.0
    w = F.normalize(weights, dim=1)
    cos = torch.clamp(w @ w.T, -1.0, 1.0)
    dist = torch.sqrt(torch.clamp(1.0 - cos, min=1e-8))
    inh = a_inh * _radial_response(dist, sigma_inh, family)
    exc = a_exc * _radial_response(dist, sigma_exc, family)
    penalty = (inh - exc) + (a_exc - a_inh)
    penalty = penalty - torch.diag(torch.diag(penalty))
    n = weights.shape[0]
    return penalty.sum() / (n * (n - 1) + 1e-8)


def geometry(weight: torch.Tensor) -> dict[str, float]:
    w = weight.detach().float().cpu()
    sv = torch.linalg.svdvals(w)
    p = sv / (sv.sum() + 1e-12)
    eff_rank = float(torch.exp(-(p * torch.log(p + 1e-12)).sum()).item())
    wn = F.normalize(w, dim=1)
    cos = wn @ wn.T
    offdiag = cos[~torch.eye(cos.shape[0], dtype=torch.bool)]
    return {"effective_rank": eff_rank, "mean_abs_offdiag_cosine": float(offdiag.abs().mean().item())}


class CubImageDataset(Dataset):
    def __init__(self, cub_dir: Path, rows: list[dict], transform):
        self.cub_dir = cub_dir
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        img = Image.open(self.cub_dir / "images" / row["relpath"]).convert("RGB")
        return self.transform(img), row["label"], row["relpath"]


def extract_classification_features(args: argparse.Namespace, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    cache = Path(args.classification_cache)
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        payload = torch.load(cache, map_location="cpu")
        return payload["x_train"], payload["y_train"], payload["x_test"], payload["y_test"]

    cub_dir = ensure_cub(Path(args.data_root), download=not args.no_download)
    rows = read_cub_metadata(cub_dir, args.max_classes)
    train_rows = [r for r in rows if r["is_train"]]
    test_rows = [r for r in rows if not r["is_train"]]

    weights = models.ResNet18_Weights.DEFAULT
    transform = weights.transforms()
    backbone = models.resnet18(weights=weights)
    backbone.fc = nn.Identity()
    backbone = backbone.to(device).eval()

    def collect(split_rows: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
        loader = DataLoader(
            CubImageDataset(cub_dir, split_rows, transform),
            batch_size=args.feature_batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
        )
        xs, ys = [], []
        with torch.no_grad():
            for xb, yb, _ in loader:
                xs.append(backbone(xb.to(device)).cpu())
                ys.append(torch.as_tensor(yb, dtype=torch.long))
        return torch.cat(xs), torch.cat(ys)

    x_train, y_train = collect(train_rows)
    x_test, y_test = collect(test_rows)
    torch.save({"x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test}, cache)
    return x_train, y_train, x_test, y_test


def make_tasks(num_classes: int, classes_per_task: int, seed: int, class_order: str) -> list[list[int]]:
    order = list(range(num_classes))
    if class_order == "random":
        random.Random(seed).shuffle(order)
    return [order[i : i + classes_per_task] for i in range(0, num_classes, classes_per_task) if len(order[i : i + classes_per_task]) == classes_per_task]


def subset_loader(x: torch.Tensor, y: torch.Tensor, classes: list[int], batch_size: int, shuffle: bool) -> DataLoader:
    mask = torch.zeros_like(y, dtype=torch.bool)
    for cls in classes:
        mask |= y == cls
    idx = torch.where(mask)[0]
    return DataLoader(TensorDataset(x[idx], y[idx]), batch_size=batch_size, shuffle=shuffle)


class LinearHead(nn.Module):
    def __init__(self, dim: int, num_classes: int, tau: float = 12.0):
        super().__init__()
        self.classifier = nn.Linear(dim, num_classes, bias=False)
        self.tau = tau

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tau * F.linear(F.normalize(x, dim=1), F.normalize(self.classifier.weight, dim=1))


def masked_argmax(logits: torch.Tensor, allowed: list[int]) -> torch.Tensor:
    mask = torch.full_like(logits, -1e9)
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


def run_classification(args: argparse.Namespace, method: str, seed: int, device: torch.device) -> ClassificationRun:
    set_seed(seed)
    x_train, y_train, x_test, y_test = extract_classification_features(args, device)
    num_classes = int(args.max_classes or args.num_classes)
    tasks = make_tasks(num_classes, args.classes_per_task, seed, args.class_order)
    model = LinearHead(x_train.shape[1], num_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.cls_lr, weight_decay=args.cls_weight_decay if method == "l2" else 0.0)
    teacher: LinearHead | None = None
    uses_ssr = method in {"biocs", "biocs_kd"}
    uses_kd = method in {"kd", "biocs_kd"}
    acc_matrix = []
    cil_curve = []
    for tid, task_classes in enumerate(tasks):
        loader = subset_loader(x_train, y_train, task_classes, args.cls_batch_size, True)
        for _ in range(args.cls_epochs):
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = F.cross_entropy(logits, yb)
                if uses_ssr:
                    loss = loss + args.lambda_sp * biocs_loss(
                        model.classifier.weight,
                        args.a_exc,
                        args.a_inh,
                        args.sigma_exc,
                        args.sigma_inh,
                        args.kernel_family,
                    )
                if uses_kd and teacher is not None:
                    with torch.no_grad():
                        target = teacher(xb)
                    t = args.kd_temperature
                    loss = loss + args.lambda_kd * (t * t) * F.kl_div(
                        F.log_softmax(logits / t, dim=1),
                        F.softmax(target / t, dim=1),
                        reduction="batchmean",
                    )
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        acc_matrix.append(evaluate_classification_til(model, x_test, y_test, tasks, tid + 1, args.cls_batch_size, device))
        seen_classes = [c for task in tasks[: tid + 1] for c in task]
        cil_curve.append(evaluate_classification_cil(model, x_test, y_test, seen_classes, args.cls_batch_size, device))
        if uses_kd:
            teacher = LinearHead(x_train.shape[1], num_classes).to(device)
            teacher.load_state_dict(model.state_dict())
            teacher.eval()
    final = acc_matrix[-1]
    forgetting = []
    for task_idx in range(len(tasks) - 1):
        best = max(row[task_idx] for row in acc_matrix[task_idx:])
        forgetting.append(best - final[task_idx])

    def threshold_capacity(values: list[float], threshold: float) -> int:
        cap = 0
        for idx, value in enumerate(values):
            if value >= threshold:
                cap = (idx + 1) * args.classes_per_task
        return cap

    til_curve = [float(np.mean(row)) for row in acc_matrix]
    return ClassificationRun(
        task="classification",
        method=method,
        seed=seed,
        avg_accuracy=float(np.mean(final)),
        avg_cil_accuracy=float(np.mean(cil_curve)),
        avg_forgetting=float(np.mean(forgetting)) if forgetting else 0.0,
        final_til_accuracy_seen=float(np.mean(final)),
        final_cil_accuracy_seen=float(cil_curve[-1]),
        til_capacity_at_80=threshold_capacity(til_curve, 80.0),
        til_capacity_at_75=threshold_capacity(til_curve, 75.0),
        til_capacity_at_70=threshold_capacity(til_curve, 70.0),
        cil_capacity_at_20=threshold_capacity(cil_curve, 20.0),
        **geometry(model.classifier.weight),
    )


class CubSegDataset(Dataset):
    def __init__(self, cub_dir: Path, rows: list[dict], image_size: int, classes: list[int] | None = None):
        self.cub_dir = cub_dir
        self.seg_dir = cub_dir.parent / "segmentations"
        if classes is not None:
            classes_set = set(classes)
            rows = [r for r in rows if r["label"] in classes_set]
        self.rows = rows
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        img = Image.open(self.cub_dir / "images" / row["relpath"]).convert("RGB")
        mask_rel = Path(row["relpath"]).with_suffix(".png")
        mask = Image.open(self.seg_dir / mask_rel).convert("L")
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr - np.asarray([0.485, 0.456, 0.406], dtype=np.float32)) / np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
        x = torch.from_numpy(arr.transpose(2, 0, 1)).float()
        y = torch.from_numpy((np.asarray(mask) > 0).astype(np.float32))[None, ...]
        return x, torch.tensor(row["label"], dtype=torch.long), y


class CubSegFeatureDataset(Dataset):
    def __init__(self, feats: torch.Tensor, labels: torch.Tensor, masks: torch.Tensor, classes: list[int] | None = None):
        if classes is not None:
            class_tensor = torch.as_tensor(classes, dtype=labels.dtype)
            keep = (labels[:, None] == class_tensor[None, :]).any(dim=1)
            idx = torch.where(keep)[0]
            feats = feats[idx]
            labels = labels[idx]
            masks = masks[idx]
        self.feats = feats
        self.labels = labels
        self.masks = masks

    def __len__(self) -> int:
        return int(self.labels.numel())

    def __getitem__(self, idx: int):
        return self.feats[idx].float(), self.labels[idx], self.masks[idx].float()


class DenseResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self.stem(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        return x


class ClassConditionedMaskDecoder(nn.Module):
    def __init__(self, num_classes: int, hidden_dim: int = 128, feature_dim: int = 256):
        super().__init__()
        self.classifier = nn.Embedding(num_classes, hidden_dim)
        self.proj = nn.Conv2d(feature_dim, hidden_dim, kernel_size=1)
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim // 2),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 2, 1, kernel_size=1),
        )

    def forward(self, feats: torch.Tensor, labels: torch.Tensor, out_size: tuple[int, int]) -> torch.Tensor:
        x = self.proj(feats)
        cond = self.classifier(labels).unsqueeze(-1).unsqueeze(-1)
        x = x + cond
        logits = self.decoder(x)
        return F.interpolate(logits, size=out_size, mode="bilinear", align_corners=False)

    def channel_prototypes(self) -> torch.Tensor:
        conv = self.decoder[0]
        return conv.weight.flatten(1)


class PrototypeCosineMaskDecoder(nn.Module):
    """Predict masks by comparing every pixel embedding to its class prototype."""

    def __init__(
        self,
        num_classes: int,
        hidden_dim: int = 128,
        feature_dim: int = 256,
        logit_scale: float = 10.0,
    ):
        super().__init__()
        if logit_scale <= 0.0:
            raise ValueError("prototype cosine logit scale must be positive")
        self.classifier = nn.Embedding(num_classes, hidden_dim)
        self.class_bias = nn.Embedding(num_classes, 1)
        nn.init.zeros_(self.class_bias.weight)
        self.background_prototype = nn.Parameter(torch.empty(hidden_dim))
        nn.init.normal_(self.background_prototype)
        self.proj = nn.Conv2d(feature_dim, hidden_dim, kernel_size=1)
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
        )
        self.logit_scale = nn.Parameter(torch.tensor(float(logit_scale)))

    def forward(
        self,
        feats: torch.Tensor,
        labels: torch.Tensor,
        out_size: tuple[int, int],
    ) -> torch.Tensor:
        pixel_embeddings = F.normalize(self.decoder(self.proj(feats)), dim=1)
        class_prototypes = F.normalize(self.classifier(labels), dim=1)
        foreground_similarity = (
            pixel_embeddings
            * class_prototypes.unsqueeze(-1).unsqueeze(-1)
        ).sum(dim=1, keepdim=True)
        background_prototype = F.normalize(
            self.background_prototype, dim=0
        ).view(1, -1, 1, 1)
        background_similarity = (pixel_embeddings * background_prototype).sum(
            dim=1, keepdim=True
        )
        scale = self.logit_scale.clamp(1.0, 30.0).to(
            dtype=foreground_similarity.dtype
        )
        bias = self.class_bias(labels).view(-1, 1, 1, 1)
        logits = scale * (foreground_similarity - background_similarity) + bias
        return F.interpolate(logits, size=out_size, mode="bilinear", align_corners=False)

    def channel_prototypes(self) -> torch.Tensor:
        conv = self.decoder[0]
        return conv.weight.flatten(1)


SEGMENTATION_HEADS = ("additive", "prototype_cosine")


def build_segmentation_decoder(
    num_classes: int,
    hidden_dim: int,
    *,
    head: str = "additive",
    prototype_logit_scale: float = 10.0,
) -> nn.Module:
    if head == "additive":
        return ClassConditionedMaskDecoder(num_classes, hidden_dim)
    if head == "prototype_cosine":
        return PrototypeCosineMaskDecoder(
            num_classes,
            hidden_dim,
            logit_scale=prototype_logit_scale,
        )
    raise ValueError(f"Unknown segmentation head: {head!r}")


def segmentation_head_config(
    head: str,
    hidden_dim: int,
    *,
    prototype_logit_scale: float,
) -> dict[str, object]:
    if head not in SEGMENTATION_HEADS:
        raise ValueError(f"Unknown segmentation head: {head!r}")
    return {
        "head": head,
        "hidden_dim": int(hidden_dim),
        "feature_dim": 256,
        "class_conditioning": (
            "pre_decoder_addition"
            if head == "additive"
            else "normalized_foreground_minus_background_prototype_cosine_mask_logit"
        ),
        "prototype_logit_scale": (
            float(prototype_logit_scale) if head == "prototype_cosine" else None
        ),
        "prototype_logit_scale_learnable": head == "prototype_cosine",
        "prototype_logit_scale_clamp": (
            [1.0, 30.0] if head == "prototype_cosine" else None
        ),
        "class_bias": head == "prototype_cosine",
    }


def ssr_warmup_ramp_multiplier(
    step: int,
    total_steps: int,
    *,
    warmup_fraction: float,
    ramp_fraction: float,
) -> float:
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if step < 0 or step >= total_steps:
        raise ValueError("step must be within the current task training budget")
    if warmup_fraction < 0.0 or ramp_fraction < 0.0:
        raise ValueError("SSR warmup and ramp fractions cannot be negative")
    if warmup_fraction + ramp_fraction > 1.0:
        raise ValueError("SSR warmup and ramp fractions cannot sum above one")
    warmup_steps = int(np.ceil(total_steps * warmup_fraction))
    ramp_steps = int(np.ceil(total_steps * ramp_fraction))
    if step < warmup_steps:
        return 0.0
    if ramp_steps == 0:
        return 1.0
    return min(1.0, float(step - warmup_steps + 1) / float(ramp_steps))


def dice_loss(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    inter = (probs * mask).sum(dim=(1, 2, 3))
    denom = probs.sum(dim=(1, 2, 3)) + mask.sum(dim=(1, 2, 3))
    return 1.0 - ((2.0 * inter + 1.0) / (denom + 1.0)).mean()


def segmentation_metrics(logits: torch.Tensor, mask: torch.Tensor) -> tuple[float, float]:
    pred = (torch.sigmoid(logits) > 0.5).float()
    inter = (pred * mask).sum(dim=(1, 2, 3))
    union = ((pred + mask) > 0).float().sum(dim=(1, 2, 3))
    denom = pred.sum(dim=(1, 2, 3)) + mask.sum(dim=(1, 2, 3))
    iou = ((inter + 1.0) / (union + 1.0)).mean().item()
    dice = ((2.0 * inter + 1.0) / (denom + 1.0)).mean().item()
    return iou, dice


def extract_segmentation_features(args: argparse.Namespace, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if args.no_segmentation_cache:
        raise RuntimeError("segmentation feature cache disabled")
    cache = Path(args.segmentation_cache or f"data/cub200/cub200_seg_resnet18_dense_{args.seg_image_size}.pt")
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        payload = torch.load(cache, map_location="cpu")
        return (
            payload["train_feats"],
            payload["train_labels"],
            payload["train_masks"],
            payload["test_feats"],
            payload["test_labels"],
            payload["test_masks"],
        )

    cub_dir = ensure_cub(Path(args.data_root), download=not args.no_download)
    rows = read_cub_metadata(cub_dir, args.max_classes)
    train_rows = [r for r in rows if r["is_train"]]
    test_rows = [r for r in rows if not r["is_train"]]
    encoder = DenseResNet18().to(device).eval()

    def collect(split_rows: list[dict]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        loader = DataLoader(
            CubSegDataset(cub_dir, split_rows, args.seg_image_size),
            batch_size=args.seg_batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
        )
        feats_out, labels_out, masks_out = [], [], []
        with torch.no_grad():
            for xb, cls, mask in loader:
                feats = encoder(xb.to(device)).cpu().half()
                feats_out.append(feats)
                labels_out.append(cls.cpu().long())
                masks_out.append(mask.cpu().to(torch.bool))
        return torch.cat(feats_out), torch.cat(labels_out), torch.cat(masks_out)

    print(f"[seg-cache] building {cache}", flush=True)
    train_feats, train_labels, train_masks = collect(train_rows)
    test_feats, test_labels, test_masks = collect(test_rows)
    torch.save(
        {
            "train_feats": train_feats,
            "train_labels": train_labels,
            "train_masks": train_masks,
            "test_feats": test_feats,
            "test_labels": test_labels,
            "test_masks": test_masks,
            "image_size": args.seg_image_size,
        },
        cache,
    )
    print(f"[seg-cache] saved {cache}", flush=True)
    return train_feats, train_labels, train_masks, test_feats, test_labels, test_masks


def evaluate_segmentation(encoder: nn.Module, model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    encoder.eval()
    model.eval()
    iou_sum = dice_sum = 0.0
    sample_count = 0
    with torch.no_grad():
        for xb, cls, mask in loader:
            xb, cls, mask = xb.to(device), cls.to(device), mask.to(device)
            logits = model(encoder(xb), cls, mask.shape[-2:])
            iou, dice = segmentation_metrics(logits, mask)
            batch_size = int(mask.shape[0])
            iou_sum += iou * batch_size
            dice_sum += dice * batch_size
            sample_count += batch_size
    return iou_sum / max(sample_count, 1), dice_sum / max(sample_count, 1)


def evaluate_segmentation_features(model: nn.Module, loader: DataLoader, device: torch.device, out_size: tuple[int, int]) -> tuple[float, float]:
    model.eval()
    iou_sum = dice_sum = 0.0
    sample_count = 0
    with torch.no_grad():
        for feats, cls, mask in loader:
            feats, cls, mask = feats.to(device), cls.to(device), mask.to(device)
            logits = model(feats, cls, out_size)
            iou, dice = segmentation_metrics(logits, mask)
            batch_size = int(mask.shape[0])
            iou_sum += iou * batch_size
            dice_sum += dice * batch_size
            sample_count += batch_size
    return iou_sum / max(sample_count, 1), dice_sum / max(sample_count, 1)


def run_segmentation(args: argparse.Namespace, method: str, seed: int, device: torch.device) -> SegmentationRun:
    set_seed(seed)
    cub_dir = ensure_cub(Path(args.data_root), download=not args.no_download)
    rows = read_cub_metadata(cub_dir, args.max_classes)
    train_rows = [r for r in rows if r["is_train"]]
    test_rows = [r for r in rows if not r["is_train"]]
    num_classes = int(args.max_classes or args.num_classes)
    tasks = make_tasks(num_classes, args.classes_per_task, seed, args.class_order)
    schedule_hash = task_schedule_sha256(tasks)
    use_feature_cache = not args.no_segmentation_cache
    if use_feature_cache:
        train_feats, train_labels, train_masks, test_feats, test_labels, test_masks = extract_segmentation_features(args, device)
        if args.seg_evaluation_split == "validation":
            fit_indices, validation_indices = stratified_fit_validation_indices(
                train_labels,
                validation_fraction=args.seg_validation_fraction,
                seed=args.seg_validation_seed,
            )
            test_feats = train_feats[validation_indices]
            test_labels = train_labels[validation_indices]
            test_masks = train_masks[validation_indices]
            train_feats = train_feats[fit_indices]
            train_labels = train_labels[fit_indices]
            train_masks = train_masks[fit_indices]
        encoder = None
    else:
        if args.seg_evaluation_split == "validation":
            labels = torch.as_tensor([row["label"] for row in train_rows])
            fit_indices, validation_indices = stratified_fit_validation_indices(
                labels,
                validation_fraction=args.seg_validation_fraction,
                seed=args.seg_validation_seed,
            )
            validation_rows = [train_rows[index] for index in validation_indices.tolist()]
            train_rows = [train_rows[index] for index in fit_indices.tolist()]
            test_rows = validation_rows
        encoder = DenseResNet18().to(device).eval()
    model = build_segmentation_decoder(
        num_classes,
        args.seg_hidden_dim,
        head=args.segmentation_head,
        prototype_logit_scale=args.prototype_logit_scale,
    ).to(device)
    initial_hash = state_dict_sha256(model)
    opt = torch.optim.AdamW(model.parameters(), lr=args.seg_lr, weight_decay=args.seg_weight_decay if method == "l2" else 0.0)
    teacher: nn.Module | None = None
    uses_ssr = method in {"biocs", "biocs_kd"}
    uses_kd = method in {"kd", "biocs_kd"}
    iou_matrix = []
    optimizer_steps = 0
    for tid, task_classes in enumerate(tasks):
        seen_class_ids = [c for task in tasks[: tid + 1] for c in task]
        old_class_ids = [c for task in tasks[:tid] for c in task]
        if use_feature_cache:
            train_dataset = CubSegFeatureDataset(train_feats, train_labels, train_masks, task_classes)
        else:
            train_dataset = CubSegDataset(cub_dir, train_rows, args.seg_image_size, task_classes)
        train_generator = torch.Generator()
        train_generator.manual_seed(seed + 100_003 * tid)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.seg_batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True,
            generator=train_generator,
        )
        task_training_step = 0
        total_task_training_steps = args.seg_epochs * len(train_loader)
        distillation_step = 0
        for _ in range(args.seg_epochs):
            model.train()
            for feats_or_xb, cls, mask in train_loader:
                cls, mask = cls.to(device), mask.to(device)
                if use_feature_cache:
                    feats = feats_or_xb.to(device)
                else:
                    xb = feats_or_xb.to(device)
                    feats = encoder(xb)
                logits = model(feats, cls, mask.shape[-2:])
                loss = F.binary_cross_entropy_with_logits(logits, mask) + dice_loss(logits, mask)
                if uses_ssr and tid >= args.seg_biocs_start_task:
                    if args.seg_biocs_target == "channels":
                        spatial_loss = biocs_loss(
                            model.channel_prototypes(),
                            args.a_exc,
                            args.a_inh,
                            args.sigma_exc,
                            args.sigma_inh,
                            args.kernel_family,
                        )
                    else:
                        if args.seg_biocs_scope == "new_old":
                            if not old_class_ids:
                                spatial_loss = model.classifier.weight.sum() * 0.0
                            else:
                                spatial_loss = cross_set_ssr_loss(
                                    model.classifier.weight[
                                        torch.as_tensor(task_classes, device=device)
                                    ],
                                    model.classifier.weight[
                                        torch.as_tensor(old_class_ids, device=device)
                                    ],
                                    kernel=args.kernel_family,
                                    hwhm_exc=scale_to_hwhm(
                                        args.kernel_family, args.sigma_exc
                                    ),
                                    hwhm_inh=scale_to_hwhm(
                                        args.kernel_family, args.sigma_inh
                                    ),
                                    a_exc=args.a_exc,
                                    a_inh=args.a_inh,
                                )
                        elif args.seg_biocs_scope == "seen":
                            reg_weight = model.classifier.weight[torch.as_tensor(seen_class_ids, device=device)]
                        elif args.seg_biocs_scope == "batch":
                            reg_weight = model.classifier.weight[torch.unique(cls)]
                        else:
                            reg_weight = model.classifier.weight
                        if args.seg_biocs_scope != "new_old":
                            spatial_loss = biocs_loss(
                                reg_weight,
                                args.a_exc,
                                args.a_inh,
                                args.sigma_exc,
                                args.sigma_inh,
                                args.kernel_family,
                            )
                    ssr_multiplier = ssr_warmup_ramp_multiplier(
                        task_training_step,
                        total_task_training_steps,
                        warmup_fraction=args.ssr_warmup_fraction,
                        ramp_fraction=args.ssr_ramp_fraction,
                    )
                    loss = loss + args.lambda_sp * ssr_multiplier * spatial_loss
                if uses_kd and teacher is not None:
                    distill_cls = old_class_distillation_ids(
                        old_class_ids,
                        feats.size(0),
                        distillation_step,
                        device,
                    )
                    student_old = model(feats, distill_cls, mask.shape[-2:])
                    with torch.no_grad():
                        target_old = teacher(feats, distill_cls, mask.shape[-2:])
                    loss = loss + args.lambda_kd_seg * F.binary_cross_entropy_with_logits(
                        student_old,
                        torch.sigmoid(target_old),
                    )
                    distillation_step += 1
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                optimizer_steps += 1
                task_training_step += 1
        seen_ious = []
        for seen_classes in tasks[: tid + 1]:
            if use_feature_cache:
                test_dataset = CubSegFeatureDataset(test_feats, test_labels, test_masks, seen_classes)
                test_loader = DataLoader(test_dataset, batch_size=args.seg_batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
                seen_ious.append(evaluate_segmentation_features(model, test_loader, device, (args.seg_image_size, args.seg_image_size))[0] * 100.0)
            else:
                test_loader = DataLoader(
                    CubSegDataset(cub_dir, test_rows, args.seg_image_size, seen_classes),
                    batch_size=args.seg_batch_size,
                    shuffle=False,
                    num_workers=args.workers,
                    pin_memory=True,
                )
                seen_ious.append(evaluate_segmentation(encoder, model, test_loader, device)[0] * 100.0)
        iou_matrix.append(seen_ious)
        if uses_kd:
            teacher = build_segmentation_decoder(
                num_classes,
                args.seg_hidden_dim,
                head=args.segmentation_head,
                prototype_logit_scale=args.prototype_logit_scale,
            ).to(device)
            teacher.load_state_dict(model.state_dict())
            teacher.eval()
    final_ious = iou_matrix[-1]
    forgetting = []
    for task_idx in range(len(tasks) - 1):
        best = max(row[task_idx] for row in iou_matrix[task_idx:])
        forgetting.append(best - final_ious[task_idx])
    if use_feature_cache:
        full_dataset = CubSegFeatureDataset(test_feats, test_labels, test_masks, [c for task in tasks for c in task])
        full_loader = DataLoader(full_dataset, batch_size=args.seg_batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
        mean_iou, mean_dice = evaluate_segmentation_features(model, full_loader, device, (args.seg_image_size, args.seg_image_size))
    else:
        full_loader = DataLoader(
            CubSegDataset(cub_dir, test_rows, args.seg_image_size, [c for task in tasks for c in task]),
            batch_size=args.seg_batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
        )
        mean_iou, mean_dice = evaluate_segmentation(encoder, model, full_loader, device)
    if args.seg_biocs_target == "channels":
        geo = geometry(model.channel_prototypes())
    else:
        geo = geometry(model.classifier.weight)
    return SegmentationRun(
        task="segmentation",
        method=method,
        seed=seed,
        mean_iou=100.0 * mean_iou,
        mean_dice=100.0 * mean_dice,
        avg_forgetting_iou=float(np.mean(forgetting)) if forgetting else 0.0,
        **geo,
        initial_model_hash=initial_hash,
        task_schedule_hash=schedule_hash,
        optimizer_steps=optimizer_steps,
    )


def summarize(rows: list[dict]) -> dict:
    out: dict[str, dict] = {}
    for task in sorted({r["task"] for r in rows}):
        out[task] = {}
        for method in sorted({r["method"] for r in rows if r["task"] == task}):
            items = [r for r in rows if r["task"] == task and r["method"] == method]
            out[task][method] = {}
            keys = [
                key
                for key in items[0]
                if key
                not in {
                    "task",
                    "method",
                    "seed",
                    "initial_model_hash",
                    "task_schedule_hash",
                    "optimizer_steps",
                }
            ]
            for key in keys:
                vals = np.asarray([float(item[key]) for item in items], dtype=float)
                out[task][method][key] = {"mean": float(vals.mean()), "std": float(vals.std()), "n": int(vals.size)}
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="data/cub200")
    p.add_argument("--output_dir", default="results/cub200_continual_20260429")
    p.add_argument("--classification_cache", default="data/cub200/cub200_resnet18_features.pt")
    p.add_argument("--segmentation_cache", default=None)
    p.add_argument("--segmentation_source_sha256", default=None)
    p.add_argument("--segmentation_cache_sha256", default=None)
    p.add_argument("--selection_lock_sha256", default="development_screen")
    p.add_argument("--tasks", nargs="+", choices=["classification", "segmentation"], default=["classification", "segmentation"])
    p.add_argument("--methods", nargs="+", default=["baseline", "l2", "biocs", "biocs_kd"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--num_classes", type=int, default=200)
    p.add_argument("--max_classes", type=int, default=None)
    p.add_argument("--classes_per_task", type=int, default=10)
    p.add_argument("--class_order", choices=["semantic", "random"], default="semantic")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--feature_batch_size", type=int, default=128)
    p.add_argument("--cls_epochs", type=int, default=80)
    p.add_argument("--cls_batch_size", type=int, default=128)
    p.add_argument("--cls_lr", type=float, default=1e-3)
    p.add_argument("--cls_weight_decay", type=float, default=1e-3)
    p.add_argument("--seg_epochs", type=int, default=12)
    p.add_argument("--seg_batch_size", type=int, default=32)
    p.add_argument("--seg_lr", type=float, default=5e-4)
    p.add_argument("--seg_weight_decay", type=float, default=1e-4)
    p.add_argument("--seg_image_size", type=int, default=128)
    p.add_argument("--seg_hidden_dim", type=int, default=128)
    p.add_argument(
        "--segmentation-head",
        choices=SEGMENTATION_HEADS,
        default="additive",
        help="Opt-in mask head; the default preserves the historical additive decoder.",
    )
    p.add_argument(
        "--prototype-logit-scale",
        type=float,
        default=10.0,
        help="Initial learnable cosine-logit scale for the prototype head (clamped to [1, 30]).",
    )
    p.add_argument("--seg_biocs_target", choices=["class", "channels"], default="class")
    p.add_argument(
        "--seg_biocs_scope",
        choices=["all", "seen", "batch", "new_old"],
        default="all",
    )
    p.add_argument("--seg_biocs_start_task", type=int, default=0)
    p.add_argument(
        "--seg_evaluation_split", choices=["test", "validation"], default="test"
    )
    p.add_argument("--seg_validation_fraction", type=float, default=0.2)
    p.add_argument("--seg_validation_seed", type=int, default=20260809)
    p.add_argument("--lambda_sp", type=float, default=2.0)
    p.add_argument(
        "--ssr-warmup-fraction",
        type=float,
        default=0.0,
        help="Per-task zero-SSR fraction; use 0.2 for the prototype-head protocol.",
    )
    p.add_argument(
        "--ssr-ramp-fraction",
        type=float,
        default=0.0,
        help="Per-task linear SSR ramp fraction; use 0.2 for the prototype-head protocol.",
    )
    p.add_argument("--lambda_kd", type=float, default=2.0)
    p.add_argument("--lambda_kd_seg", type=float, default=0.5)
    p.add_argument("--a-exc", type=float, default=1.0)
    p.add_argument("--a-inh", type=float, default=0.8)
    p.add_argument("--sigma-exc", type=float, default=0.2)
    p.add_argument("--sigma-inh", type=float, default=0.5)
    p.add_argument("--kernel-family", choices=KERNEL_FAMILIES, default="gaussian")
    p.add_argument("--kd_temperature", type=float, default=3.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no_download", action="store_true")
    p.add_argument("--no_segmentation_cache", action="store_true")
    args = p.parse_args()

    if not 0.0 < args.seg_validation_fraction < 0.5:
        raise ValueError("--seg_validation_fraction must be between zero and one half")
    if args.prototype_logit_scale <= 0.0:
        raise ValueError("--prototype-logit-scale must be positive")
    if args.ssr_warmup_fraction < 0.0 or args.ssr_ramp_fraction < 0.0:
        raise ValueError("SSR warmup and ramp fractions cannot be negative")
    if args.ssr_warmup_fraction + args.ssr_ramp_fraction > 1.0:
        raise ValueError("SSR warmup and ramp fractions cannot sum above one")

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ensure_cub(Path(args.data_root), download=not args.no_download)
    if "segmentation" in args.tasks and (
        not args.segmentation_source_sha256 or not args.segmentation_cache_sha256
    ):
        raise ValueError(
            "Segmentation runs require --segmentation_source_sha256 and "
            "--segmentation_cache_sha256 for auditable data identity"
        )
    dataset_hash = cub_dataset_fingerprint(
        Path(args.data_root),
        segmentation_source_sha256=args.segmentation_source_sha256,
        segmentation_cache_sha256=args.segmentation_cache_sha256,
    )

    rows = []
    runs_path = outdir / "runs.jsonl"
    for seed in args.seeds:
        for task in args.tasks:
            for method in args.methods:
                print(f"[cub200] task={task} method={method} seed={seed}", flush=True)
                started = time.time()
                if task == "classification":
                    row = asdict(run_classification(args, method, seed, device))
                else:
                    row = asdict(run_segmentation(args, method, seed, device))
                write_result_record(
                    outdir=outdir,
                    row=row,
                    args=args,
                    elapsed_s=time.time() - started,
                    dataset_hash=dataset_hash,
                )
                rows.append(row)
                with runs_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(json.dumps(row, ensure_ascii=False), flush=True)
                (outdir / "summary.json").write_text(json.dumps(summarize(rows), indent=2), encoding="utf-8")

    (outdir / "summary.json").write_text(json.dumps(summarize(rows), indent=2), encoding="utf-8")
    print(f"Saved to {outdir}", flush=True)


if __name__ == "__main__":
    main()
