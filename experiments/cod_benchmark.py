#!/usr/bin/env python3
"""COD benchmark trainer for camouflaged object segmentation.

The script targets the standard SINet data layout:

data/cod/
  TrainDataset/{Image,GT}/...
  TestDataset/{CAMO,CHAMELEON,COD10K}/{Imgs,GT}/...

It trains a compact ResNet-FPN binary mask model with optional SSR channel
regularization and mask-query prototype regularization.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import timm


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def biocs_loss(weights: torch.Tensor, a_exc=1.0, a_inh=0.8, sigma_exc=0.2, sigma_inh=0.5) -> torch.Tensor:
    w = F.normalize(weights.flatten(1), dim=1)
    dist2 = torch.cdist(w, w, p=2).pow(2)
    eye = torch.eye(dist2.size(0), device=dist2.device, dtype=torch.bool)
    exc = -a_exc * torch.exp(-dist2 / (2 * sigma_exc**2))
    inh = a_inh * torch.exp(-dist2 / (2 * sigma_inh**2))
    return (exc + inh)[~eye].mean()


def diversity_loss(weights: torch.Tensor) -> torch.Tensor:
    w = F.normalize(weights.flatten(1), dim=1)
    gram = w @ w.t()
    eye = torch.eye(gram.size(0), device=gram.device, dtype=torch.bool)
    return gram[~eye].abs().mean()


def find_dir(root: Path, names: list[str]) -> Path | None:
    lower = {p.name.lower(): p for p in root.iterdir() if p.is_dir()} if root.exists() else {}
    for name in names:
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def collect_pairs(image_dir: Path, mask_dir: Path, max_items: int | None = None) -> list[tuple[Path, Path]]:
    images = [p for p in image_dir.rglob("*") if p.suffix.lower() in IMG_EXTS]
    masks = [p for p in mask_dir.rglob("*") if p.suffix.lower() in IMG_EXTS]
    mask_by_stem: dict[str, Path] = {}
    for m in masks:
        mask_by_stem.setdefault(m.stem, m)
    pairs = []
    for img in sorted(images):
        mask = mask_by_stem.get(img.stem)
        if mask is not None:
            pairs.append((img, mask))
    if max_items:
        pairs = pairs[:max_items]
    return pairs


def discover_train_pairs(data_root: Path, max_items: int | None = None) -> list[tuple[Path, Path]]:
    train = data_root / "TrainDataset"
    image_dir = find_dir(train, ["Image", "Imgs", "Imgs_train", "JPEGImages", "images"])
    mask_dir = find_dir(train, ["GT", "Mask", "Masks", "gts", "Annotation"])
    if image_dir is None or mask_dir is None:
        raise FileNotFoundError(f"Cannot find TrainDataset image/GT folders under {train}")
    pairs = collect_pairs(image_dir, mask_dir, max_items=max_items)
    if not pairs:
        raise FileNotFoundError(f"No train image/mask pairs found under {train}")
    return pairs


def discover_test_sets(data_root: Path, max_items: int | None = None) -> dict[str, list[tuple[Path, Path]]]:
    test = data_root / "TestDataset"
    if not test.exists():
        raise FileNotFoundError(f"Missing {test}")
    out = {}
    for ds in sorted(p for p in test.iterdir() if p.is_dir()):
        image_dir = find_dir(ds, ["Imgs", "Image", "images", "JPEGImages"])
        mask_dir = find_dir(ds, ["GT", "Mask", "Masks", "gts"])
        if image_dir and mask_dir:
            pairs = collect_pairs(image_dir, mask_dir, max_items=max_items)
            if pairs:
                out[ds.name] = pairs
    if not out:
        raise FileNotFoundError(f"No COD test sets found under {test}")
    return out


class CODDataset(Dataset):
    def __init__(self, pairs: list[tuple[Path, Path]], image_size: int, train: bool):
        self.pairs = pairs
        self.image_size = image_size
        self.train = train
        self.color = transforms.ColorJitter(0.12, 0.12, 0.08, 0.03)
        self.normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path, mask_path = self.pairs[idx]
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        if self.train and random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        img = img.resize((self.image_size, self.image_size), Image.BICUBIC)
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)
        if self.train:
            img = self.color(img)
        x = transforms.functional.to_tensor(img)
        x = self.normalize(x)
        y = transforms.functional.to_tensor(mask)
        y = (y > 0.5).float()
        return x, y


class BioMaskQueryNet(nn.Module):
    def __init__(self, backbone: str = "resnet50", hidden_dim: int = 128, num_queries: int = 8, pretrained: bool = True):
        super().__init__()
        self.encoder = timm.create_model(backbone, features_only=True, pretrained=pretrained, out_indices=(1, 2, 3, 4))
        channels = self.encoder.feature_info.channels()
        self.lateral = nn.ModuleList([nn.Conv2d(c, hidden_dim, 1) for c in channels])
        self.smooth = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.mask_queries = nn.Parameter(torch.randn(num_queries, hidden_dim) * 0.02)
        self.query_score = nn.Parameter(torch.ones(num_queries) / max(1, num_queries))
        self.pred = nn.Conv2d(hidden_dim, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        target_size = feats[0].shape[-2:]
        fused = 0
        for feat, lat in zip(feats, self.lateral):
            item = lat(feat)
            item = F.interpolate(item, size=target_size, mode="bilinear", align_corners=False)
            fused = fused + item
        pix = self.smooth(fused)
        q = F.normalize(self.mask_queries, dim=1)
        pix_norm = F.normalize(pix, dim=1)
        sim = torch.einsum("bchw,qc->bqhw", pix_norm, q)
        query_logit = torch.einsum("bqhw,q->bhw", sim, self.query_score).unsqueeze(1)
        logits = self.pred(pix) + query_logit
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)

    def channel_weights(self) -> torch.Tensor:
        weights = []
        for module in self.smooth:
            if isinstance(module, nn.Conv2d):
                weights.append(module.weight)
        return torch.cat([w.flatten(1) for w in weights], dim=0)

    def query_weights(self) -> torch.Tensor:
        return self.mask_queries


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum(dim=(2, 3))
    denom = prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    return (1.0 - (2.0 * inter + 1.0) / (denom + 1.0)).mean()


def structure_measure(pred: np.ndarray, gt: np.ndarray) -> float:
    gt = (gt > 0.5).astype(np.float32)
    pred = pred.astype(np.float32)
    y = float(gt.mean())
    if y == 0:
        return 1.0 - float(pred.mean())
    if y == 1:
        return float(pred.mean())
    fg = pred[gt == 1]
    bg = pred[gt == 0]
    o_fg = (2 * fg.mean()) / (fg.mean() ** 2 + 1.0 + 1e-8)
    o_bg = (2 * (1 - bg.mean())) / ((1 - bg.mean()) ** 2 + 1.0 + 1e-8)
    s_object = y * o_fg + (1 - y) * o_bg
    h, w = gt.shape
    xs = np.arange(w, dtype=np.float32)[None, :]
    ys = np.arange(h, dtype=np.float32)[:, None]
    area = gt.sum() + 1e-8
    cx = int(np.clip(np.round((gt * xs).sum() / area), 1, w - 1))
    cy = int(np.clip(np.round((gt * ys).sum() / area), 1, h - 1))

    def ssim_region(p: np.ndarray, g: np.ndarray) -> float:
        if p.size == 0:
            return 0.0
        mux, muy = p.mean(), g.mean()
        sigx, sigy = p.var(), g.var()
        sigxy = ((p - mux) * (g - muy)).mean()
        alpha = 4 * mux * muy * sigxy
        beta = (mux * mux + muy * muy) * (sigx + sigy)
        if alpha != 0:
            return float(alpha / (beta + 1e-8))
        return float(1.0 if alpha == beta else 0.0)

    regions = [
        (slice(0, cy), slice(0, cx)),
        (slice(0, cy), slice(cx, w)),
        (slice(cy, h), slice(0, cx)),
        (slice(cy, h), slice(cx, w)),
    ]
    weights = [cx * cy, (w - cx) * cy, cx * (h - cy), (w - cx) * (h - cy)]
    s_region = 0.0
    for reg, weight in zip(regions, weights):
        s_region += weight / (h * w) * ssim_region(pred[reg], gt[reg])
    return float(max(0.0, 0.5 * s_object + 0.5 * s_region))


@dataclass
class Metrics:
    mae: float
    dice: float
    miou: float
    f_beta: float
    s_measure: float


def batch_metrics(logits: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    prob = torch.sigmoid(logits).detach().cpu().numpy()
    gt = target.detach().cpu().numpy()
    pred_bin = prob >= 0.5
    gt_bin = gt >= 0.5
    eps = 1e-8
    maes, dices, ious, fbs, sms = [], [], [], [], []
    beta2 = 0.3
    for p, g, pb, gb in zip(prob[:, 0], gt[:, 0], pred_bin[:, 0], gt_bin[:, 0]):
        tp = float(np.logical_and(pb, gb).sum())
        pred_sum = float(pb.sum())
        gt_sum = float(gb.sum())
        precision = tp / (pred_sum + eps)
        recall = tp / (gt_sum + eps)
        dice = (2 * tp + eps) / (pred_sum + gt_sum + eps)
        iou = (tp + eps) / (pred_sum + gt_sum - tp + eps)
        fb = (1 + beta2) * precision * recall / (beta2 * precision + recall + eps)
        maes.append(float(np.abs(p - g).mean()))
        dices.append(dice)
        ious.append(iou)
        fbs.append(fb)
        sms.append(structure_measure(p, g))
    return {
        "mae": float(np.mean(maes)),
        "dice": float(np.mean(dices)),
        "miou": float(np.mean(ious)),
        "f_beta": float(np.mean(fbs)),
        "s_measure": float(np.mean(sms)),
    }


@torch.no_grad()
def evaluate(model: nn.Module, loaders: dict[str, DataLoader], device: torch.device) -> dict[str, Metrics]:
    model.eval()
    out = {}
    for name, loader in loaders.items():
        sums: dict[str, float] = {}
        count = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            metrics = batch_metrics(model(x), y)
            bsz = x.size(0)
            for key, value in metrics.items():
                sums[key] = sums.get(key, 0.0) + value * bsz
            count += bsz
        out[name] = Metrics(**{key: value / max(1, count) for key, value in sums.items()})
    return out


def train_one(args: argparse.Namespace, method: str, seed: int, device: torch.device) -> dict:
    set_seed(seed)
    data_root = Path(args.data_root)
    train_pairs = discover_train_pairs(data_root, args.max_train)
    test_pairs = discover_test_sets(data_root, args.max_test)
    train_loader = DataLoader(
        CODDataset(train_pairs, args.image_size, train=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )
    test_loaders = {
        name: DataLoader(
            CODDataset(pairs, args.image_size, train=False),
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
        )
        for name, pairs in test_pairs.items()
    }
    model = BioMaskQueryNet(args.backbone, args.hidden_dim, args.num_queries, pretrained=not args.no_pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    best_score = -math.inf
    best_metrics: dict[str, Metrics] | None = None
    outdir = Path(args.output_dir)
    ckpt_dir = outdir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    history = []
    lambda_channel = args.lambda_channel if method in {"biocs_channel", "biocs_both"} else 0.0
    lambda_query = args.lambda_query if method in {"mask_query", "biocs_both"} else 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                logits = model(x)
                loss = F.binary_cross_entropy_with_logits(logits, y) + dice_loss(logits, y)
                if lambda_channel:
                    loss = loss + lambda_channel * biocs_loss(model.channel_weights())
                if lambda_query:
                    loss = loss + lambda_query * (biocs_loss(model.query_weights()) + diversity_loss(model.query_weights()))
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        if epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs:
            metrics = evaluate(model, test_loaders, device)
            score = float(np.mean([m.s_measure - m.mae for m in metrics.values()]))
            flat = {
                "epoch": epoch,
                "method": method,
                "seed": seed,
                "loss": float(np.mean(losses)),
                "score": score,
                "metrics": {k: asdict(v) for k, v in metrics.items()},
            }
            history.append(flat)
            with (outdir / f"{method}_seed{seed}_history.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(flat, ensure_ascii=False) + "\n")
            print(json.dumps(flat, ensure_ascii=False), flush=True)
            if score > best_score:
                best_score = score
                best_metrics = metrics
                if args.save_checkpoints:
                    torch.save(model.state_dict(), ckpt_dir / f"{method}_seed{seed}_best.pt")
    assert best_metrics is not None
    row = {
        "task": "cod",
        "method": method,
        "seed": seed,
        "best_score": best_score,
        "train_items": len(train_pairs),
        "test_items": {k: len(v) for k, v in test_pairs.items()},
        "metrics": {k: asdict(v) for k, v in best_metrics.items()},
    }
    return row


def summarize(rows: list[dict]) -> dict:
    out: dict[str, dict] = {}
    for method in sorted({r["method"] for r in rows}):
        items = [r for r in rows if r["method"] == method]
        datasets = sorted({ds for r in items for ds in r["metrics"]})
        out[method] = {}
        for ds in datasets:
            out[method][ds] = {}
            for key in ["s_measure", "f_beta", "mae", "miou", "dice"]:
                vals = [float(r["metrics"][ds][key]) for r in items if ds in r["metrics"]]
                out[method][ds][key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}
        scores = [float(r["best_score"]) for r in items]
        out[method]["macro_score"] = {"mean": float(np.mean(scores)), "std": float(np.std(scores)), "n": len(scores)}
    return out


def write_summary_md(summary: dict, path: Path) -> None:
    lines = ["# COD Benchmark Summary", ""]
    for method, block in summary.items():
        lines += [f"## {method}", "", "| Dataset | S_alpha | F_beta | MAE | mIoU | Dice | n |", "|---|---:|---:|---:|---:|---:|---:|"]
        for ds, metrics in block.items():
            if ds == "macro_score":
                continue
            lines.append(
                f"| {ds} | {metrics['s_measure']['mean']:.4f}±{metrics['s_measure']['std']:.4f} | "
                f"{metrics['f_beta']['mean']:.4f}±{metrics['f_beta']['std']:.4f} | "
                f"{metrics['mae']['mean']:.4f}±{metrics['mae']['std']:.4f} | "
                f"{metrics['miou']['mean']:.4f}±{metrics['miou']['std']:.4f} | "
                f"{metrics['dice']['mean']:.4f}±{metrics['dice']['std']:.4f} | "
                f"{metrics['mae']['n']} |"
            )
        lines.append("")
        macro = block["macro_score"]
        lines.append(f"Macro score S_alpha-MAE: {macro['mean']:.4f}±{macro['std']:.4f}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="data/cod")
    p.add_argument("--output_dir", default="results/cod_benchmark_20260429")
    p.add_argument("--methods", nargs="+", default=["baseline", "biocs_channel", "mask_query", "biocs_both"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--backbone", default="resnet50")
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--num_queries", type=int, default=8)
    p.add_argument("--image_size", type=int, default=352)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--eval_batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--eval_every", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--lambda_channel", type=float, default=0.02)
    p.add_argument("--lambda_query", type=float, default=0.05)
    p.add_argument("--max_train", type=int, default=None)
    p.add_argument("--max_test", type=int, default=None)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no_pretrained", action="store_true")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--save_checkpoints", action="store_true")
    args = p.parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rows = []
    runs_path = outdir / "runs.jsonl"
    for seed in args.seeds:
        for method in args.methods:
            print(f"[cod] method={method} seed={seed}", flush=True)
            row = train_one(args, method, seed, device)
            rows.append(row)
            with runs_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            summary = summarize(rows)
            (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            write_summary_md(summary, outdir / "summary.md")
    print(f"Saved to {outdir}", flush=True)


if __name__ == "__main__":
    main()
