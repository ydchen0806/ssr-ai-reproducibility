#!/usr/bin/env python3
"""Matched task-only versus task+SSR continual segmentation experiments.

Oxford-IIIT Pet and Oxford Flowers 102 are represented as class-conditioned
binary foreground-mask tasks. A frozen ImageNet ResNet-18 dense encoder keeps
the trainable objective and compute identical across the matched pair.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import loadmat

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.cub200_continual_benchmark import (
    ClassConditionedMaskDecoder,
    DenseResNet18,
    biocs_loss,
    geometry,
)
from ssr_utils.result_schema import build_result_record, sha256_file, sha256_value
from ssr_utils.segmentation_kd import old_class_distillation_ids
from ssr_utils.segmentation_nested_search import (
    cross_set_ssr_loss,
    scale_to_hwhm,
    stratified_fit_validation_indices,
)
from ssr_utils.multidataset_segmentation import (
    flower_foreground,
    make_tasks,
    pet_foreground_and_valid,
)


DATASET_CONFIGS = {
    "oxford_iiit_pet": {"num_classes": 37, "classes_per_task": 5},
    "oxford_flowers102": {"num_classes": 102, "classes_per_task": 10},
}
METHODS = ("task_only", "task_ssr", "kd", "kd_ssr")


@dataclass(frozen=True)
class Sample:
    sample_id: int
    image: Path
    mask: Path
    label: int
    train: bool


@dataclass(frozen=True)
class SegmentationResult:
    task: str
    method: str
    seed: int
    mean_iou: float
    mean_dice: float
    avg_forgetting_iou: float
    effective_rank: float
    mean_abs_offdiag_cosine: float
    num_tasks: int
    num_classes: int
    initial_model_hash: str
    task_schedule_hash: str
    optimizer_steps: int


def current_git_commit() -> str:
    return subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def module_state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_dataset_manifest(root: Path, dataset: str) -> dict:
    path = root / "DATASET_MANIFEST.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"Prepare {dataset} with scripts/prepare_multidataset_segmentation.py: {error}"
        ) from error
    if manifest.get("dataset") != dataset:
        raise RuntimeError(f"Dataset manifest identity mismatch: {path}")
    fingerprint = manifest.get("dataset_fingerprint")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise RuntimeError(f"Dataset manifest lacks a valid fingerprint: {path}")
    return manifest


def read_pet_samples(root: Path) -> list[Sample]:
    rows: list[Sample] = []
    sample_id = 0
    for split_name, train in (("trainval.txt", True), ("test.txt", False)):
        split = root / "annotations" / split_name
        with split.open("r", encoding="utf-8") as handle:
            for raw in handle:
                fields = raw.strip().split()
                if not fields:
                    continue
                if len(fields) != 4:
                    raise RuntimeError(f"Malformed Oxford Pet split row: {raw!r}")
                stem, class_id, _species_id, _breed_id = fields
                image_candidates = [
                    root / "images" / f"{stem}{suffix}"
                    for suffix in (".jpg", ".jpeg", ".png")
                ]
                matches = [path for path in image_candidates if path.is_file()]
                if len(matches) != 1:
                    raise RuntimeError(
                        f"Expected one image payload for Pet stem {stem!r}, got {matches}"
                    )
                image = matches[0]
                mask = root / "annotations" / "trimaps" / f"{stem}.png"
                if not image.is_file() or not mask.is_file():
                    raise FileNotFoundError(f"Missing paired Pet sample: {image}, {mask}")
                rows.append(
                    Sample(sample_id, image, mask, int(class_id) - 1, train)
                )
                sample_id += 1
    return rows


def read_flower_samples(root: Path) -> list[Sample]:
    labels = np.asarray(loadmat(root / "imagelabels.mat")["labels"]).reshape(-1)
    split = loadmat(root / "setid.mat")
    training_ids = {
        int(value)
        for key in ("trnid", "valid")
        for value in np.asarray(split[key]).reshape(-1)
    }
    test_ids = {int(value) for value in np.asarray(split["tstid"]).reshape(-1)}
    if training_ids & test_ids or training_ids | test_ids != set(range(1, 8190)):
        raise RuntimeError("Oxford Flowers split does not partition image ids 1..8189")
    rows = []
    for image_id, raw_label in enumerate(labels, start=1):
        image = root / "jpg" / f"image_{image_id:05d}.jpg"
        mask = root / "segmim" / f"segmim_{image_id:05d}.jpg"
        if not image.is_file() or not mask.is_file():
            raise FileNotFoundError(f"Missing paired Flowers sample: {image}, {mask}")
        rows.append(
            Sample(image_id, image, mask, int(raw_label) - 1, image_id in training_ids)
        )
    return rows


def read_samples(root: Path, dataset: str) -> list[Sample]:
    rows = read_pet_samples(root) if dataset == "oxford_iiit_pet" else read_flower_samples(root)
    labels = {row.label for row in rows}
    expected = set(range(int(DATASET_CONFIGS[dataset]["num_classes"])))
    if labels != expected:
        raise RuntimeError(
            f"{dataset} class inventory mismatch: missing={sorted(expected - labels)}, "
            f"unexpected={sorted(labels - expected)}"
        )
    return rows


class RawMaskDataset(Dataset):
    def __init__(self, rows: list[Sample], dataset: str, image_size: int):
        self.rows = rows
        self.dataset = dataset
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image = Image.open(row.image).convert("RGB").resize(
            (self.image_size, self.image_size), Image.Resampling.BILINEAR
        )
        mask_image = Image.open(row.mask).resize(
            (self.image_size, self.image_size), Image.Resampling.NEAREST
        )
        if self.dataset == "oxford_iiit_pet":
            mask, valid = pet_foreground_and_valid(mask_image)
        else:
            mask = flower_foreground(mask_image)
            valid = np.ones_like(mask, dtype=np.bool_)
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        image_array = (
            image_array - np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        ) / np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
        return (
            torch.from_numpy(image_array.transpose(2, 0, 1)).float(),
            torch.tensor(row.label, dtype=torch.long),
            torch.from_numpy(mask.astype(np.bool_))[None, ...],
            torch.from_numpy(valid.astype(np.bool_))[None, ...],
        )


class FeatureMaskDataset(Dataset):
    def __init__(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        masks: torch.Tensor,
        valid_masks: torch.Tensor,
        classes: list[int] | None = None,
    ):
        if classes is not None:
            keep = torch.zeros_like(labels, dtype=torch.bool)
            for class_id in classes:
                keep |= labels == class_id
            indices = torch.where(keep)[0]
            features = features[indices]
            labels = labels[indices]
            masks = masks[indices]
            valid_masks = valid_masks[indices]
        self.features = features
        self.labels = labels
        self.masks = masks
        self.valid_masks = valid_masks

    def __len__(self) -> int:
        return int(self.labels.numel())

    def __getitem__(self, index: int):
        return (
            self.features[index].float(),
            self.labels[index],
            self.masks[index].float(),
            self.valid_masks[index].float(),
        )


def cache_manifest_path(cache: Path) -> Path:
    return cache.with_name(cache.name + ".manifest.json")


def validate_cache_manifest(
    cache: Path,
    *,
    dataset: str,
    dataset_fingerprint: str,
    image_size: int,
    expected_sha256: str | None = None,
) -> dict:
    path = cache_manifest_path(cache)
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Feature-cache manifest is unavailable: {path}: {error}") from error
    expected = {
        "dataset": dataset,
        "dataset_fingerprint": dataset_fingerprint,
        "image_size": image_size,
        "encoder": "torchvision_resnet18_default_layer3",
        "cache_format": "binary_mask_with_valid_region_v1",
    }
    mismatch = {
        key: {"expected": value, "observed": manifest.get(key)}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatch:
        raise RuntimeError(f"Feature-cache identity mismatch: {json.dumps(mismatch)}")
    for field in ("torchvision_version", "encoder_weights", "encoder_state_sha256"):
        if not isinstance(manifest.get(field), str) or not manifest[field]:
            raise RuntimeError(f"Feature-cache manifest is missing {field}: {path}")
    encoder_hash = manifest["encoder_state_sha256"]
    if len(encoder_hash) != 64 or any(
        character not in "0123456789abcdef" for character in encoder_hash
    ):
        raise RuntimeError(f"Invalid encoder_state_sha256 in {path}")
    observed_sha256 = expected_sha256 or sha256_file(cache)
    if manifest.get("cache_sha256") != observed_sha256:
        raise RuntimeError(f"Feature cache SHA256 mismatch: {cache}")
    return manifest


def prepare_feature_cache(args: argparse.Namespace, device: torch.device) -> dict:
    dataset_root = args.dataset_root or args.data_root / args.dataset
    dataset_manifest = load_dataset_manifest(dataset_root, args.dataset)
    fingerprint = dataset_manifest["dataset_fingerprint"]
    cache = args.feature_cache
    if cache.is_file() and cache_manifest_path(cache).is_file():
        return validate_cache_manifest(
            cache,
            dataset=args.dataset,
            dataset_fingerprint=fingerprint,
            image_size=args.image_size,
            expected_sha256=args.feature_cache_sha256,
        )

    rows = read_samples(dataset_root, args.dataset)
    encoder = DenseResNet18().to(device).eval()
    encoder_state_sha256 = module_state_sha256(encoder)
    import torchvision
    from torchvision import models

    def collect(split_rows: list[Sample]):
        loader = DataLoader(
            RawMaskDataset(split_rows, args.dataset, args.image_size),
            batch_size=args.feature_batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
        )
        features_out, labels_out, masks_out, valid_out = [], [], [], []
        with torch.no_grad():
            for images, labels, masks, valid_masks in loader:
                features_out.append(encoder(images.to(device)).cpu().half())
                labels_out.append(labels.long())
                masks_out.append(masks.bool())
                valid_out.append(valid_masks.bool())
        return (
            torch.cat(features_out),
            torch.cat(labels_out),
            torch.cat(masks_out),
            torch.cat(valid_out),
        )

    print(f"[cache] building {args.dataset} -> {cache}", flush=True)
    train_features, train_labels, train_masks, train_valid_masks = collect(
        [row for row in rows if row.train]
    )
    test_features, test_labels, test_masks, test_valid_masks = collect(
        [row for row in rows if not row.train]
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_name(f".{cache.name}.tmp.{os.getpid()}")
    torch.save(
        {
            "train_features": train_features,
            "train_labels": train_labels,
            "train_masks": train_masks,
            "train_valid_masks": train_valid_masks,
            "test_features": test_features,
            "test_labels": test_labels,
            "test_masks": test_masks,
            "test_valid_masks": test_valid_masks,
        },
        temporary,
    )
    temporary.replace(cache)
    manifest = {
        "schema_version": 1,
        "dataset": args.dataset,
        "dataset_fingerprint": fingerprint,
        "image_size": args.image_size,
        "encoder": "torchvision_resnet18_default_layer3",
        "torchvision_version": torchvision.__version__,
        "encoder_weights": f"ResNet18_Weights.{models.ResNet18_Weights.DEFAULT.name}",
        "encoder_state_sha256": encoder_state_sha256,
        "cache_format": "binary_mask_with_valid_region_v1",
        "cache_sha256": sha256_file(cache),
        "train_samples": int(train_labels.numel()),
        "test_samples": int(test_labels.numel()),
    }
    write_json_atomic(cache_manifest_path(cache), manifest)
    return manifest


def masked_task_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    valid = valid.to(logits.dtype)
    target = target.to(logits.dtype)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    bce = (bce * valid).sum() / valid.sum().clamp_min(1.0)
    probability = torch.sigmoid(logits)
    intersection = (probability * target * valid).sum(dim=(1, 2, 3))
    denominator = ((probability + target) * valid).sum(dim=(1, 2, 3))
    dice = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
    return bce + dice


def masked_segmentation_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[float, float]:
    valid = valid.bool()
    target = target.bool()
    prediction = torch.sigmoid(logits) > 0.5
    intersection = (prediction & target & valid).sum(dim=(1, 2, 3)).float()
    union = ((prediction | target) & valid).sum(dim=(1, 2, 3)).float()
    denominator = ((prediction & valid).sum(dim=(1, 2, 3)) + (target & valid).sum(dim=(1, 2, 3))).float()
    iou = ((intersection + 1.0) / (union + 1.0)).mean().item()
    dice = ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean().item()
    return iou, dice


def evaluate(
    model: ClassConditionedMaskDecoder,
    dataset: FeatureMaskDataset,
    batch_size: int,
    image_size: int,
    workers: int,
    device: torch.device,
) -> tuple[float, float]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )
    model.eval()
    iou_sum = dice_sum = 0.0
    sample_count = 0
    with torch.no_grad():
        for features, labels, masks, valid_masks in loader:
            features = features.to(device)
            labels = labels.to(device)
            masks = masks.to(device)
            valid_masks = valid_masks.to(device)
            logits = model(features, labels, (image_size, image_size))
            iou, dice = masked_segmentation_metrics(logits, masks, valid_masks)
            count = int(labels.numel())
            iou_sum += iou * count
            dice_sum += dice * count
            sample_count += count
    return iou_sum / sample_count, dice_sum / sample_count


def train(args: argparse.Namespace, device: torch.device) -> SegmentationResult:
    set_seed(args.seed)
    payload = torch.load(args.feature_cache, map_location="cpu")
    train_features = payload["train_features"]
    train_labels = payload["train_labels"].long()
    train_masks = payload["train_masks"]
    train_valid_masks = payload["train_valid_masks"]
    test_features = payload["test_features"]
    test_labels = payload["test_labels"].long()
    test_masks = payload["test_masks"]
    test_valid_masks = payload["test_valid_masks"]

    if args.evaluation_split == "validation":
        fit_indices, validation_indices = stratified_fit_validation_indices(
            train_labels,
            validation_fraction=args.validation_fraction,
            seed=args.validation_seed,
        )
        test_features = train_features[validation_indices]
        test_labels = train_labels[validation_indices]
        test_masks = train_masks[validation_indices]
        test_valid_masks = train_valid_masks[validation_indices]
        train_features = train_features[fit_indices]
        train_labels = train_labels[fit_indices]
        train_masks = train_masks[fit_indices]
        train_valid_masks = train_valid_masks[fit_indices]

    config = DATASET_CONFIGS[args.dataset]
    num_classes = int(config["num_classes"])
    tasks = make_tasks(num_classes, int(config["classes_per_task"]), args.seed)
    model = ClassConditionedMaskDecoder(num_classes, args.hidden_dim).to(device)
    initial_model_hash = module_state_sha256(model)
    task_schedule_hash = sha256_value(tasks)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    uses_ssr = args.method in {"task_ssr", "kd_ssr"}
    uses_kd = args.method in {"kd", "kd_ssr"}
    teacher: ClassConditionedMaskDecoder | None = None
    optimizer_steps = 0
    iou_matrix: list[list[float]] = []
    for task_index, task_classes in enumerate(tasks):
        seen_classes = [class_id for task in tasks[: task_index + 1] for class_id in task]
        old_classes = [class_id for task in tasks[:task_index] for class_id in task]
        train_dataset = FeatureMaskDataset(
            train_features,
            train_labels,
            train_masks,
            train_valid_masks,
            task_classes,
        )
        generator = torch.Generator().manual_seed(args.seed + 100_003 * task_index)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True,
            generator=generator,
        )
        distillation_step = 0
        for _epoch in range(args.epochs):
            model.train()
            for features, labels, masks, valid_masks in train_loader:
                features = features.to(device)
                labels = labels.to(device)
                masks = masks.to(device)
                valid_masks = valid_masks.to(device)
                logits = model(features, labels, (args.image_size, args.image_size))
                loss = masked_task_loss(logits, masks, valid_masks)
                if uses_ssr and task_index >= args.ssr_start_task:
                    if args.ssr_target == "channels":
                        spatial_loss = biocs_loss(
                            model.channel_prototypes(),
                            args.a_exc,
                            args.a_inh,
                            args.sigma_exc,
                            args.sigma_inh,
                            args.kernel_family,
                        )
                    elif args.ssr_scope == "new_old":
                        if not old_classes:
                            spatial_loss = model.classifier.weight.sum() * 0.0
                        else:
                            spatial_loss = cross_set_ssr_loss(
                                model.classifier.weight[
                                    torch.as_tensor(task_classes, device=device)
                                ],
                                model.classifier.weight[
                                    torch.as_tensor(old_classes, device=device)
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
                    else:
                        indices = torch.as_tensor(seen_classes, device=device)
                        spatial_loss = biocs_loss(
                            model.classifier.weight[indices],
                            args.a_exc,
                            args.a_inh,
                            args.sigma_exc,
                            args.sigma_inh,
                            args.kernel_family,
                        )
                    loss = loss + args.lambda_ssr * spatial_loss
                if uses_kd and teacher is not None:
                    distill_classes = old_class_distillation_ids(
                        old_classes,
                        features.size(0),
                        distillation_step,
                        device,
                    )
                    student_old = model(
                        features,
                        distill_classes,
                        (args.image_size, args.image_size),
                    )
                    with torch.no_grad():
                        teacher_old = teacher(
                            features,
                            distill_classes,
                            (args.image_size, args.image_size),
                        )
                    loss = loss + args.lambda_kd * F.binary_cross_entropy_with_logits(
                        student_old,
                        torch.sigmoid(teacher_old),
                    )
                    distillation_step += 1
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                optimizer_steps += 1

        task_ious = []
        for seen_task in tasks[: task_index + 1]:
            dataset = FeatureMaskDataset(
                test_features,
                test_labels,
                test_masks,
                test_valid_masks,
                seen_task,
            )
            task_ious.append(
                100.0
                * evaluate(
                    model,
                    dataset,
                    args.batch_size,
                    args.image_size,
                    args.workers,
                    device,
                )[0]
            )
        iou_matrix.append(task_ious)
        if uses_kd:
            teacher = copy.deepcopy(model).eval()
            for parameter in teacher.parameters():
                parameter.requires_grad_(False)

    final_ious = iou_matrix[-1]
    forgetting = []
    for task_index in range(len(tasks) - 1):
        best = max(row[task_index] for row in iou_matrix[task_index:])
        forgetting.append(best - final_ious[task_index])
    all_classes = [class_id for task in tasks for class_id in task]
    mean_iou, mean_dice = evaluate(
        model,
        FeatureMaskDataset(
            test_features,
            test_labels,
            test_masks,
            test_valid_masks,
            all_classes,
        ),
        args.batch_size,
        args.image_size,
        args.workers,
        device,
    )
    geometry_weight = (
        model.channel_prototypes()
        if args.ssr_target == "channels"
        else model.classifier.weight
    )
    return SegmentationResult(
        task="segmentation",
        method=args.method,
        seed=args.seed,
        mean_iou=100.0 * mean_iou,
        mean_dice=100.0 * mean_dice,
        avg_forgetting_iou=float(np.mean(forgetting)) if forgetting else 0.0,
        num_tasks=len(tasks),
        num_classes=num_classes,
        initial_model_hash=initial_model_hash,
        task_schedule_hash=task_schedule_hash,
        optimizer_steps=optimizer_steps,
        **geometry(geometry_weight),
    )


def result_record(
    args: argparse.Namespace,
    result: SegmentationResult,
    *,
    dataset_manifest: dict,
    cache_manifest: dict,
    elapsed_s: float,
) -> dict:
    uses_ssr = args.method in {"task_ssr", "kd_ssr"}
    uses_kd = args.method in {"kd", "kd_ssr"}
    objective = {"task": True, "kd": uses_kd, "ssr": uses_ssr}
    kernel = (
        {
            "family": args.kernel_family,
            "A_exc": args.a_exc,
            "A_inh": args.a_inh,
            "sigma_exc": args.sigma_exc,
            "sigma_inh": args.sigma_inh,
        }
        if uses_ssr
        else {}
    )
    metrics = {
        key: float(value)
        for key, value in asdict(result).items()
        if key
        not in {
            "task",
            "method",
            "seed",
            "initial_model_hash",
            "task_schedule_hash",
            "optimizer_steps",
        }
    }
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    dataset_hash = sha256_value(
        {
            "dataset_fingerprint": dataset_manifest["dataset_fingerprint"],
            "feature_cache_sha256": cache_manifest["cache_sha256"],
            "mask_policy": dataset_manifest["mask_policy"],
            "image_size": args.image_size,
        }
    )
    return build_result_record(
        git_commit=current_git_commit(),
        run_id=str(args.output_dir.resolve()),
        task_family="segmentation",
        dataset=f"{args.dataset}_masks",
        model="resnet18_dense_class_conditioned_decoder",
        seed=args.seed,
        objective=objective,
        distance_mapping="cosine" if uses_ssr else "none",
        kernel=kernel,
        metrics=metrics,
        runtime={"elapsed_s": elapsed_s},
        config=config,
        dataset_hash=dataset_hash,
        recipe={
            "task_only": "plain",
            "task_ssr": "ssr_only",
            "kd": "kd",
            "kd_ssr": "kd_ssr",
        }[args.method],
        selection_lock_hash=args.selection_lock_sha256,
        lambda_ssr=args.lambda_ssr if uses_ssr else 0.0,
        lambda_kd=args.lambda_kd if uses_kd else 0.0,
        source_dataset_fingerprint=dataset_manifest["dataset_fingerprint"],
        feature_cache_sha256=cache_manifest["cache_sha256"],
        encoder_state_sha256=cache_manifest["encoder_state_sha256"],
        torchvision_version=cache_manifest["torchvision_version"],
        initial_model_hash=result.initial_model_hash,
        task_schedule_hash=result.task_schedule_hash,
        optimizer_steps=result.optimizer_steps,
        metric_directions={
            "mean_iou": True,
            "mean_dice": True,
            "avg_forgetting_iou": False,
            "effective_rank": True,
            "mean_abs_offdiag_cosine": False,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=tuple(DATASET_CONFIGS), required=True)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--feature-cache-sha256", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS, default="task_only")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--feature-batch-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lambda-ssr", type=float, default=0.05)
    parser.add_argument("--lambda-kd", type=float, default=0.5)
    parser.add_argument("--ssr-target", choices=("class", "channels"), default="class")
    parser.add_argument("--ssr-scope", choices=("seen", "new_old", "all"), default="seen")
    parser.add_argument("--ssr-start-task", type=int, default=0)
    parser.add_argument(
        "--evaluation-split", choices=("test", "validation"), default="test"
    )
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--validation-seed", type=int, default=20260809)
    parser.add_argument("--a-exc", type=float, default=1.0)
    parser.add_argument("--a-inh", type=float, default=0.8)
    parser.add_argument("--sigma-exc", type=float, default=0.16)
    parser.add_argument("--sigma-inh", type=float, default=0.45)
    parser.add_argument(
        "--kernel-family", choices=("gaussian", "laplace", "cauchy", "inverse"), default="gaussian"
    )
    parser.add_argument("--selection-lock-sha256", default="development_screen")
    parser.add_argument("--prepare-cache-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.data_root = args.data_root.resolve()
    if args.dataset_root is not None:
        args.dataset_root = args.dataset_root.resolve()
    args.feature_cache = args.feature_cache.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.ssr_start_task < 0:
        raise ValueError("--ssr-start-task cannot be negative")
    if args.ssr_target == "channels" and args.ssr_scope != "all":
        raise ValueError("channel-target SSR requires --ssr-scope all")
    if not 0.0 < args.validation_fraction < 0.5:
        raise ValueError("--validation-fraction must be between zero and one half")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cache_manifest = prepare_feature_cache(args, device)
    if args.prepare_cache_only:
        print(json.dumps(cache_manifest, indent=2))
        return
    dataset_manifest = load_dataset_manifest(
        args.dataset_root or args.data_root / args.dataset,
        args.dataset,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.output_dir / "args.json", dict(vars(args)))
    started = time.time()
    result = train(args, device)
    record = result_record(
        args,
        result,
        dataset_manifest=dataset_manifest,
        cache_manifest=cache_manifest,
        elapsed_s=time.time() - started,
    )
    write_json_atomic(args.output_dir / "result_record.json", record)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
