"""Base class for continual learning methods.

Supports:
  - AMP (Automatic Mixed Precision) via training config `use_amp: true`
  - torch.compile via training config `compile_model: true`
  - Gradient clipping via `grad_clip`
  - Configurable num_workers, batch_size, etc.
"""

import logging
import time
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


class BaseContinualLearner(ABC):
    """Abstract base class for all CL methods."""

    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.current_task = -1
        self._compiled = False
        self._scaler = None

    def train_task(self, task_id: int, train_set, training_config: dict):
        self.current_task = task_id
        self.model.train()

        use_amp = training_config.get("use_amp", False)
        if use_amp and self._scaler is None:
            self._scaler = torch.amp.GradScaler("cuda")
            logger.info("AMP enabled (float16 mixed precision)")

        if training_config.get("compile_model", False) and not self._compiled:
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                self._compiled = True
                logger.info("Model compiled with torch.compile (reduce-overhead)")
            except Exception as e:
                logger.warning(f"torch.compile failed, falling back: {e}")

        n_workers = training_config.get("num_workers", 4)
        run_seed = int(training_config.get("run_seed", 0))
        loader_generator = torch.Generator()
        loader_generator.manual_seed(run_seed + 100_003 * int(task_id))
        loader = DataLoader(
            train_set,
            batch_size=training_config.get("batch_size", 256),
            shuffle=True,
            num_workers=n_workers,
            pin_memory=True,
            persistent_workers=n_workers > 0,
            prefetch_factor=4 if n_workers > 0 else None,
            drop_last=False,
            generator=loader_generator,
        )

        optimizer = self._build_optimizer(training_config)
        scheduler = self._build_scheduler(optimizer, training_config)
        grad_clip = training_config.get("grad_clip", 0.0)
        n_epochs = training_config.get("epochs", 50)

        t_start = time.time()

        for epoch in range(n_epochs):
            epoch_loss = 0.0
            correct, total = 0, 0

            for x, y in tqdm(loader, desc=f"Epoch {epoch+1}/{n_epochs}", leave=False):
                x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

                if use_amp:
                    with torch.amp.autocast("cuda"):
                        loss, logits = self.training_step(x, y, task_id)
                    optimizer.zero_grad(set_to_none=True)
                    self._scaler.scale(loss).backward()
                    if grad_clip > 0:
                        self._scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                    scale_before_step = self._scaler.get_scale()
                    self._scaler.step(optimizer)
                    self._scaler.update()
                    if self._scaler.get_scale() >= scale_before_step:
                        self._after_optimizer_step()
                else:
                    loss, logits = self.training_step(x, y, task_id)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    if grad_clip > 0:
                        nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                    optimizer.step()
                    self._after_optimizer_step()

                if hasattr(self, '_accumulate_w'):
                    self._accumulate_w()

                epoch_loss += loss.item() * x.size(0)
                with torch.no_grad():
                    _, predicted = logits.max(1)
                    correct += predicted.eq(y).sum().item()
                total += y.size(0)

            if scheduler is not None:
                scheduler.step()

            avg_loss = epoch_loss / total
            acc = 100.0 * correct / total
            logger.info(f"Task {task_id+1} Epoch {epoch+1}: loss={avg_loss:.4f}, acc={acc:.1f}%")

        elapsed = time.time() - t_start
        logger.info(f"Task {task_id+1} training completed in {elapsed:.1f}s")

    @abstractmethod
    def training_step(self, x, y, task_id) -> tuple[torch.Tensor, torch.Tensor]:
        ...

    @abstractmethod
    def after_task(self, task_id: int):
        ...

    def _after_optimizer_step(self) -> None:
        """Record method-specific state only after an optimizer update succeeds."""

    @torch.no_grad()
    def evaluate(self, task_id: int, test_set, task_classes: list[int] | None = None) -> float:
        """Evaluate accuracy on a test set.

        Args:
            task_id: Current task being evaluated.
            test_set: Test dataset.
            task_classes: If provided, mask logits to these classes (TIL mode).
        """
        self.model.eval()
        loader = DataLoader(test_set, batch_size=512, shuffle=False, num_workers=2, pin_memory=True)
        correct, total = 0, 0
        for x, y in loader:
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            logits = self.model(x)
            if task_classes is not None:
                mask = torch.full_like(logits, float('-inf'))
                for c in task_classes:
                    if c < mask.size(1):
                        mask[:, c] = 0.0
                logits = logits + mask
            _, predicted = logits.max(1)
            correct += predicted.eq(y).sum().item()
            total += y.size(0)
        self.model.train()
        return 100.0 * correct / total if total > 0 else 0.0

    def _build_optimizer(self, config: dict):
        name = config.get("optimizer", "sgd").lower()
        lr = config.get("lr", 0.01)
        wd = config.get("weight_decay", 0.0)
        if name == "sgd":
            return torch.optim.SGD(
                self.model.parameters(), lr=lr,
                momentum=config.get("momentum", 0.9), weight_decay=wd,
            )
        elif name == "adam":
            return torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        elif name == "adamw":
            return torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        raise ValueError(f"Unknown optimizer: {name}")

    def _build_scheduler(self, optimizer, config: dict):
        name = config.get("scheduler", None)
        if name is None:
            return None
        epochs = config.get("epochs", 50)
        warmup = config.get("warmup_epochs", 0)
        if name == "cosine":
            base = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs - warmup, 1))
        elif name == "step":
            base = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.get("step_size", 30), gamma=config.get("gamma", 0.1))
        elif name == "multistep":
            base = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.get("milestones", [35, 45]), gamma=config.get("gamma", 0.1))
        else:
            raise ValueError(f"Unknown scheduler: {name}")
        if warmup > 0:
            warmup_sched = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup)
            return torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup_sched, base], milestones=[warmup])
        return base
