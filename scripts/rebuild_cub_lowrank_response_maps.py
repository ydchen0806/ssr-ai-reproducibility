#!/usr/bin/env python3
"""Rebuild a low-rank CUB response-map example from exact model checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import sys
import tarfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.lowrank_adapter_mapping import (
    AdapterHead,
    biocs_loss,
    evaluate_til,
    make_tasks,
    set_seed,
    subset_loader,
)


EXPECTED = {
    "kd": {"avg_accuracy": 79.08096840855016, "avg_forgetting": 3.2330310536631046},
    "biocs_kd": {"avg_accuracy": 80.79545729609853, "avg_forgetting": 2.1646723277966102},
}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def train_model(args: argparse.Namespace, method: str, device: torch.device):
    set_seed(args.seed)
    payload = torch.load(args.feature_cache, map_location="cpu")
    x_train = payload["x_train"].float()
    y_train = payload["y_train"].long()
    x_test = payload["x_test"].float()
    y_test = payload["y_test"].long()
    tasks = make_tasks(200, 10, args.seed, "semantic")
    model = AdapterHead(x_train.shape[1], 200, 32, 12.0, 0.5).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    teacher = None
    matrix = []
    for task_id, classes in enumerate(tasks):
        loader = subset_loader(x_train, y_train, classes, 128, True)
        for _ in range(40):
            model.train()
            for features, labels in loader:
                features, labels = features.to(device), labels.to(device)
                logits = model(features)
                loss = F.cross_entropy(logits, labels)
                if method == "biocs_kd":
                    loss = loss + biocs_loss(
                        model.classifier.weight,
                        1.0,
                        0.8,
                        0.55,
                        1.25,
                        "gaussian",
                        distance_metric="cosine",
                    )
                    loss = loss + 0.2 * biocs_loss(
                        model.adapter_basis(),
                        1.0,
                        0.8,
                        0.55,
                        1.25,
                        "gaussian",
                        distance_metric="cosine",
                    )
                if teacher is not None:
                    with torch.no_grad():
                        target = teacher(features)
                    temperature = 3.0
                    loss = loss + 2.0 * temperature * temperature * F.kl_div(
                        F.log_softmax(logits / temperature, dim=1),
                        F.softmax(target / temperature, dim=1),
                        reduction="batchmean",
                    )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        matrix.append(evaluate_til(model, x_test, y_test, tasks, task_id + 1, 128, device))
        # Preserve the historical RNG trajectory: the archived runner creates
        # a fresh module before loading each teacher snapshot.
        teacher = AdapterHead(x_train.shape[1], 200, 32, 12.0, 0.5).to(device)
        teacher.load_state_dict(model.state_dict())
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)

    final = matrix[-1]
    forgetting = []
    for task_id in range(19):
        forgetting.append(max(row[task_id] for row in matrix[task_id:]) - final[task_id])
    metrics = {
        "avg_accuracy": float(np.mean(final)),
        "avg_forgetting": float(np.mean(forgetting)),
    }
    for key, expected in EXPECTED[method].items():
        if abs(metrics[key] - expected) > 1e-8:
            raise RuntimeError(f"{method} {key} mismatch: {metrics[key]} != {expected}")
    return model.eval(), metrics


def archive_text(archive: tarfile.TarFile, name: str) -> str:
    handle = archive.extractfile(name)
    if handle is None:
        raise FileNotFoundError(name)
    return handle.read().decode("utf-8")


def cub_test_rows(archive: tarfile.TarFile) -> list[dict]:
    images = {int(line.split()[0]): line.split()[1] for line in archive_text(archive, "CUB_200_2011/images.txt").splitlines()}
    labels = {int(line.split()[0]): int(line.split()[1]) - 1 for line in archive_text(archive, "CUB_200_2011/image_class_labels.txt").splitlines()}
    splits = {int(line.split()[0]): int(line.split()[1]) for line in archive_text(archive, "CUB_200_2011/train_test_split.txt").splitlines()}
    return [
        {"id": image_id, "relpath": images[image_id], "label": labels[image_id]}
        for image_id in sorted(images)
        if splits[image_id] == 0
    ]


def spatial_backbone(device: torch.device):
    backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(device).eval()

    def forward(image: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            value = backbone.conv1(image)
            value = backbone.bn1(value)
            value = backbone.relu(value)
            value = backbone.maxpool(value)
            value = backbone.layer1(value)
            value = backbone.layer2(value)
            value = backbone.layer3(value)
            return backbone.layer4(value)

    return forward


def margin_map(
    dense: torch.Tensor,
    model: AdapterHead,
    target: int,
    confuser: int,
    size: tuple[int, int],
) -> np.ndarray:
    channels = dense.shape[1]
    positions = dense.permute(0, 2, 3, 1).reshape(-1, channels)
    encoded = F.normalize(model.encode(positions), dim=1)
    prototypes = F.normalize(model.classifier.weight, dim=1)
    margin = 12.0 * (encoded @ (prototypes[target] - prototypes[confuser]))
    margin = margin.reshape(1, 1, dense.shape[2], dense.shape[3])
    margin = F.interpolate(margin, size=size, mode="bilinear", align_corners=False)
    return margin[0, 0].detach().cpu().numpy()


def top_fraction_precision(values: np.ndarray, mask: np.ndarray, fraction: float = 0.10) -> float:
    count = max(1, int(values.size * fraction))
    selected = np.argpartition(values.ravel(), -count)[-count:]
    return float(mask.ravel()[selected].mean())


def normalize_uint8(values: np.ndarray, low: float, high: float, cmap: str) -> np.ndarray:
    normalized = np.clip((values - low) / max(high - low, 1e-12), 0.0, 1.0)
    return (plt.get_cmap(cmap)(normalized)[..., :3] * 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-cache", required=True)
    parser.add_argument("--image-archive", required=True)
    parser.add_argument("--segmentation-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=8411)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--candidate-pairs", type=int, default=20)
    parser.add_argument("--checkpoint-dir", type=Path,
                        help="Use the exported cub_rank32 safetensors pair instead of retraining.")
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)

    if args.checkpoint_dir is None:
        kd_model, kd_metrics = train_model(args, "kd", device)
        ssr_model, ssr_metrics = train_model(args, "biocs_kd", device)
    else:
        from safetensors.torch import load_file

        if args.seed != 8411:
            raise ValueError("The released visualization pair belongs to seed 8411")
        loaded_models = []
        for arm, expected in (
            ("kd", "013b02634334e4f76648e34e28ef16d9ac8cf1b28a739ee7f06dc21c9ad39d00"),
            ("kd_ssr", "1a4b8859a148f31d0e72127641e7c5c486c2683732768d449c2befcd6a9c9e6e"),
        ):
            model = AdapterHead(512, 200, 32, 12.0, 0.5).to(device)
            model.load_state_dict(load_file(str(args.checkpoint_dir / f"{arm}_seed8411.safetensors")), strict=True)
            if state_sha256(model) != expected:
                raise ValueError(f"{arm} checkpoint is not the Figure 6f source model")
            loaded_models.append(model.eval())
        kd_model, ssr_model = loaded_models
        # AF requires the full training trajectory; retain its archived source.
        kd_metrics, ssr_metrics = EXPECTED["kd"], EXPECTED["biocs_kd"]
    torch.save(kd_model.state_dict(), output / "kd_rank32_seed8411.pt")
    torch.save(ssr_model.state_dict(), output / "kd_ssr_rank32_seed8411.pt")

    with torch.no_grad():
        prototypes = F.normalize(kd_model.classifier.weight, dim=1)
        cosine = prototypes @ prototypes.T
        cosine.fill_diagonal_(-float("inf"))
        flat = torch.topk(cosine.flatten(), k=2 * args.candidate_pairs).indices.tolist()
    pairs = []
    for index in flat:
        first, second = divmod(index, 200)
        pair = tuple(sorted((first, second)))
        if pair not in pairs:
            pairs.append(pair)
        if len(pairs) == args.candidate_pairs:
            break

    image_transform = models.ResNet18_Weights.DEFAULT.transforms()
    display_transform = transforms.Compose([
        transforms.Resize(232, interpolation=InterpolationMode.BILINEAR),
        transforms.CenterCrop(224),
    ])
    mask_transform = transforms.Compose([
        transforms.Resize(232, interpolation=InterpolationMode.NEAREST),
        transforms.CenterCrop(224),
    ])
    dense_forward = spatial_backbone(device)
    segmentation_root = Path(args.segmentation_root)
    candidates = []
    with tarfile.open(args.image_archive, "r:gz") as archive:
        test_rows = cub_test_rows(archive)
        for row in test_rows:
            pair = next((pair for pair in pairs if row["label"] in pair), None)
            if pair is None:
                continue
            confuser = pair[1] if row["label"] == pair[0] else pair[0]
            member = archive.getmember("CUB_200_2011/images/" + row["relpath"])
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            image_bytes = extracted.read()
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            mask_path = segmentation_root / Path(row["relpath"]).with_suffix(".png")
            if not mask_path.exists():
                continue
            mask_image = Image.open(mask_path).convert("L")
            mask = np.asarray(mask_transform(mask_image), dtype=np.uint8) > 0
            coverage = float(mask.mean())
            if coverage < 0.08 or coverage > 0.82:
                continue
            tensor = image_transform(image).unsqueeze(0).to(device)
            dense = dense_forward(tensor)
            with torch.no_grad():
                kd_map = margin_map(dense, kd_model, row["label"], confuser, mask.shape)
                ssr_map = margin_map(dense, ssr_model, row["label"], confuser, mask.shape)
            kd_contrast = float(kd_map[mask].mean() - kd_map[~mask].mean())
            ssr_contrast = float(ssr_map[mask].mean() - ssr_map[~mask].mean())
            kd_precision = top_fraction_precision(kd_map, mask)
            ssr_precision = top_fraction_precision(ssr_map, mask)
            selection_score = (ssr_contrast - kd_contrast) + 2.0 * (ssr_precision - kd_precision)
            candidates.append(
                {
                    "row": row,
                    "confuser": confuser,
                    "image_bytes": image_bytes,
                    "image": np.asarray(display_transform(image)),
                    "mask": mask,
                    "kd_map": kd_map,
                    "ssr_map": ssr_map,
                    "kd_contrast": kd_contrast,
                    "ssr_contrast": ssr_contrast,
                    "kd_precision": kd_precision,
                    "ssr_precision": ssr_precision,
                    "selection_score": selection_score,
                }
            )

    if not candidates:
        raise RuntimeError("no eligible CUB response-map candidates")
    selected = max(candidates, key=lambda item: item["selection_score"])
    difference = selected["ssr_map"] - selected["kd_map"]
    response_bound = float(max(abs(selected["kd_map"]).max(), abs(selected["ssr_map"]).max()))
    difference_bound = float(max(abs(difference).max(), 1e-8))

    np.save(output / "kd_response.npy", selected["kd_map"])
    np.save(output / "kd_ssr_response.npy", selected["ssr_map"])
    np.save(output / "signed_difference.npy", difference)
    Image.fromarray(selected["image"]).save(output / "input.png")
    Image.fromarray((selected["mask"] * 255).astype(np.uint8)).save(output / "gt_mask.png")
    Image.fromarray(normalize_uint8(selected["kd_map"], -response_bound, response_bound, "coolwarm")).save(output / "kd_response.png")
    Image.fromarray(normalize_uint8(selected["ssr_map"], -response_bound, response_bound, "coolwarm")).save(output / "kd_ssr_response.png")
    Image.fromarray(normalize_uint8(difference, -difference_bound, difference_bound, "PiYG")).save(output / "signed_difference.png")

    manifest = {
        "model_source": "verified_export" if args.checkpoint_dir else "retrained_and_endpoint_checked",
        "metrics_source": "archived_training_trajectory" if args.checkpoint_dir else "recomputed_training_trajectory",
        "protocol": "cub_lowrank_response_map_rebuild_v1",
        "seed": args.seed,
        "rank": 32,
        "dataset": "CUB-200-2011",
        "model": "frozen ResNet-18 layer-4 features + GELU bottleneck adapter",
        "control": "KD+LowRank",
        "treatment": "KD+LowRank+SSR",
        "response": "signed target-minus-hard-confuser cosine-logit margin at each layer-4 location",
        "shared_resize_crop": "ResNet18_Weights.DEFAULT resize and center crop; masks use nearest-neighbor",
        "selection": "outcome-aware illustrative screen over test images from the 20 highest-cosine KD prototype pairs",
        "candidate_count": len(candidates),
        "image_id": selected["row"]["id"],
        "image_relpath": selected["row"]["relpath"],
        "target_class_zero_based": selected["row"]["label"],
        "confuser_class_zero_based": selected["confuser"],
        "kd_foreground_contrast": selected["kd_contrast"],
        "kd_ssr_foreground_contrast": selected["ssr_contrast"],
        "kd_top10_precision": selected["kd_precision"],
        "kd_ssr_top10_precision": selected["ssr_precision"],
        "response_shared_abs_bound": response_bound,
        "difference_zero_center_abs_bound": difference_bound,
        "control_metrics": kd_metrics,
        "treatment_metrics": ssr_metrics,
        "control_checkpoint_sha256": state_sha256(kd_model),
        "treatment_checkpoint_sha256": state_sha256(ssr_model),
        "input_jpeg_sha256": sha256_bytes(selected["image_bytes"]),
        "raw_scalar_maps_saved": True,
        "use_for_population_inference": False,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
