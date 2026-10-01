"""Fixed-KD scaffold with one explicitly selected additive regularizer."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .base import BaseContinualLearner
from .bioreg import compute_spatial_biocs, compute_spectral_flatness
from .geometry_controls import prototype_decorrelation_loss
from ssr_utils.teacher_trajectory import TEACHER_PROTOCOL


logger = logging.getLogger(__name__)
REGULARIZERS = {
    "none",
    "ewc",
    "mas",
    "si",
    "center",
    "protodecor",
    "spectral",
    "ssr",
}


class KDMatched(BaseContinualLearner):
    """One KD scaffold shared by memory, geometric, and SSR regularizers."""

    def __init__(self, model: nn.Module, device: torch.device, config: dict):
        super().__init__(model, device)
        method_name = str(config.get("name", "kd")).lower()
        inferred = {
            "kd": "none",
            "kd_ewc": "ewc",
            "kd_mas": "mas",
            "kd_si": "si",
            "kd_center": "center",
            "kd_protodecor": "protodecor",
            "kd_spectral": "spectral",
            "kd_ssr": "ssr",
        }.get(method_name)
        explicit_regularizer = config.get("regularizer")
        if inferred is not None and explicit_regularizer is not None:
            explicit_regularizer = str(explicit_regularizer).lower()
            if explicit_regularizer != inferred:
                raise ValueError(
                    f"Contradictory matched-KD config: method name {method_name!r} "
                    f"requires regularizer={inferred!r}, got {explicit_regularizer!r}"
                )
        self.regularizer = str(explicit_regularizer or inferred or "none").lower()
        if self.regularizer not in REGULARIZERS:
            raise ValueError(
                f"Unknown matched-KD regularizer={self.regularizer!r}; "
                f"choose from {sorted(REGULARIZERS)}"
            )

        self.lambda_kd = float(config.get("lambda_kd", config.get("lambda_distill", 5.0)))
        self.temperature = float(config.get("temperature", 2.0))
        self.lambda_ewc = float(config.get("lambda_ewc", 400.0))
        self.lambda_mas = float(config.get("lambda_mas", 1.0))
        self.lambda_si = float(config.get("lambda_si", 1.0))
        self.lambda_center = float(config.get("lambda_center", 0.0005))
        self.lambda_protodecor = float(config.get("lambda_protodecor", 0.005))
        self.lambda_spectral = float(config.get("lambda_spectral", 0.01))
        self.lambda_ssr = float(config.get("lambda_ssr", config.get("lambda_spatial", 0.005)))
        self.center_lr_scale = float(config.get("center_lr_scale", 0.5))
        self.fisher_samples = int(config.get("fisher_samples", 2000))
        self.importance_samples = int(config.get("importance_samples", 2000))
        self.xi = float(config.get("xi", 0.1))
        self.teacher_mode = str(config.get("teacher_mode", "self_previous")).lower()
        if self.teacher_mode not in {"self_previous", "locked_kd_trajectory"}:
            raise ValueError(
                "teacher_mode must be 'self_previous' or 'locked_kd_trajectory'"
            )
        teacher_root = config.get("teacher_checkpoint_root")
        if self.teacher_mode == "locked_kd_trajectory" and not teacher_root:
            raise ValueError(
                "locked_kd_trajectory requires method.teacher_checkpoint_root"
            )
        self.teacher_checkpoint_root = Path(teacher_root).resolve() if teacher_root else None
        self.teacher_dataset = str(config.get("teacher_dataset", "unknown"))
        self.teacher_model = str(config.get("teacher_model", "resnet18"))

        self.a_exc = float(config.get("A_exc", 1.0))
        self.a_inh = float(config.get("A_inh", 0.8))
        self.sigma_exc = float(config.get("sigma_exc", 0.16))
        self.sigma_inh = float(config.get("sigma_inh", 0.45))
        if self.lambda_kd <= 0 or self.temperature <= 0:
            raise ValueError("Matched-KD requires lambda_kd > 0 and temperature > 0")
        active_weights = {
            "ewc": self.lambda_ewc,
            "mas": self.lambda_mas,
            "si": self.lambda_si,
            "center": self.lambda_center,
            "protodecor": self.lambda_protodecor,
            "spectral": self.lambda_spectral,
            "ssr": self.lambda_ssr,
        }
        if self.regularizer != "none" and active_weights[self.regularizer] <= 0:
            raise ValueError(
                f"Matched-KD regularizer {self.regularizer!r} requires a positive weight"
            )
        if self.regularizer == "center" and self.center_lr_scale <= 0:
            raise ValueError("Center regularization requires center_lr_scale > 0")
        if self.fisher_samples <= 0 or self.importance_samples <= 0 or self.xi <= 0:
            raise ValueError("Importance sample counts and xi must be positive")
        if self.sigma_exc <= 0 or self.sigma_inh <= 0:
            raise ValueError("SSR sigma values must be positive")

        self.teacher: nn.Module | None = None
        self.prev_params: dict[str, torch.Tensor] = {}
        self.importance: dict[str, torch.Tensor] = {}
        self._task_train_set = None
        self._seen_classes: set[int] = set()
        self._w: dict[str, torch.Tensor] = {}
        self._init_params: dict[str, torch.Tensor] = {}
        self._prev_step_params: dict[str, torch.Tensor] = {}
        self._run_seed = 0
        self._task_id = 0
        self.training_batches = 0
        self.optimizer_steps = 0
        self._features: torch.Tensor | None = None
        self._feature_hook = None

        linears = [(name, module) for name, module in model.named_modules() if isinstance(module, nn.Linear)]
        self._classifier = linears[-1][1]
        targets = config.get("ssr_targets", ["classifier"])
        self._ssr_layers = linears[-1:] if "classifier" in set(targets) else linears
        self.centers: nn.Parameter | None = None
        if self.regularizer == "center":
            self.centers = nn.Parameter(
                torch.zeros(
                    self._classifier.out_features,
                    self._classifier.in_features,
                    device=device,
                )
            )
            self._register_feature_hook()
        logger.info(
            "KDMatched | regularizer=%s lambda_kd=%s T=%s teacher_mode=%s",
            self.regularizer,
            self.lambda_kd,
            self.temperature,
            self.teacher_mode,
        )

    def _capture_classifier_input(self, _module, inputs, _output) -> None:
        if inputs:
            features = inputs[0]
            self._features = features.view(features.size(0), -1)

    def _register_feature_hook(self) -> None:
        if self.regularizer == "center" and self._feature_hook is None:
            self._feature_hook = self._classifier.register_forward_hook(
                self._capture_classifier_input
            )

    def _remove_feature_hook(self) -> None:
        if self._feature_hook is not None:
            self._feature_hook.remove()
            self._feature_hook = None

    def _copy_teacher_without_feature_hook(self) -> nn.Module:
        had_hook = self._feature_hook is not None
        if had_hook:
            self._remove_feature_hook()
        try:
            teacher = copy.deepcopy(self.model)
        finally:
            if had_hook:
                self._register_feature_hook()
        return teacher

    def _build_optimizer(self, config: dict):
        if self.centers is None:
            return super()._build_optimizer(config)
        name = config.get("optimizer", "sgd").lower()
        learning_rate = float(config.get("lr", 0.01))
        weight_decay = float(config.get("weight_decay", 0.0))
        parameters = [
            {"params": self.model.parameters(), "weight_decay": weight_decay},
            {
                "params": [self.centers],
                "lr": learning_rate * self.center_lr_scale,
                "weight_decay": 0.0,
            },
        ]
        if name == "sgd":
            return torch.optim.SGD(
                parameters,
                lr=learning_rate,
                momentum=float(config.get("momentum", 0.9)),
            )
        if name == "adam":
            return torch.optim.Adam(parameters, lr=learning_rate)
        if name == "adamw":
            return torch.optim.AdamW(parameters, lr=learning_rate)
        raise ValueError(f"Unknown optimizer: {name}")

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()

    def _teacher_manifest_path(self) -> Path:
        if self.teacher_checkpoint_root is None:
            raise RuntimeError("teacher checkpoint root is not configured")
        return self.teacher_checkpoint_root / "manifest.json"

    def _load_teacher_manifest(self) -> dict:
        path = self._teacher_manifest_path()
        if not path.is_file():
            raise FileNotFoundError(f"Locked KD teacher manifest is missing: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        expected = {
            "protocol": TEACHER_PROTOCOL,
            "seed": self._run_seed,
            "dataset": self.teacher_dataset,
            "model": self.teacher_model,
        }
        mismatches = {
            key: {"expected": value, "observed": payload.get(key)}
            for key, value in expected.items()
            if payload.get(key) != value
        }
        if mismatches:
            raise RuntimeError(
                f"Locked KD teacher manifest identity mismatch: {mismatches}"
            )
        return payload

    def _load_locked_teacher(self, previous_task: int) -> None:
        manifest = self._load_teacher_manifest()
        entries = {
            int(entry["task_id"]): entry for entry in manifest.get("checkpoints", [])
        }
        if previous_task not in entries:
            raise RuntimeError(
                f"Locked KD teacher manifest lacks task {previous_task}: "
                f"{self._teacher_manifest_path()}"
            )
        entry = entries[previous_task]
        checkpoint = self.teacher_checkpoint_root / entry["file"]
        observed_hash = self._sha256_file(checkpoint)
        if observed_hash != entry["sha256"]:
            raise RuntimeError(
                f"Locked KD teacher checkpoint hash mismatch for {checkpoint}: "
                f"expected {entry['sha256']}, observed {observed_hash}"
            )
        payload = torch.load(checkpoint, map_location=self.device, weights_only=True)
        teacher = self._copy_teacher_without_feature_hook()
        teacher.load_state_dict(payload["model_state_dict"], strict=True)
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        self.teacher = teacher

    def _save_locked_teacher(self, task_id: int) -> None:
        if self.teacher_checkpoint_root is None:
            raise RuntimeError("teacher checkpoint root is not configured")
        self.teacher_checkpoint_root.mkdir(parents=True, exist_ok=True)
        checkpoint = self.teacher_checkpoint_root / f"task_{task_id:02d}.pt"
        checkpoint_tmp = checkpoint.with_suffix(checkpoint.suffix + ".tmp")
        torch.save(
            {
                "protocol": TEACHER_PROTOCOL,
                "task_id": int(task_id),
                "seed": self._run_seed,
                "dataset": self.teacher_dataset,
                "model": self.teacher_model,
                "model_state_dict": self.model.state_dict(),
            },
            checkpoint_tmp,
        )
        checkpoint_tmp.replace(checkpoint)
        entry = {
            "task_id": int(task_id),
            "file": checkpoint.name,
            "sha256": self._sha256_file(checkpoint),
        }
        manifest_path = self._teacher_manifest_path()
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        else:
            manifest = {
                "protocol": TEACHER_PROTOCOL,
                "seed": self._run_seed,
                "dataset": self.teacher_dataset,
                "model": self.teacher_model,
                "checkpoints": [],
            }
        entries = {
            int(item["task_id"]): item for item in manifest.get("checkpoints", [])
        }
        entries[int(task_id)] = entry
        manifest["checkpoints"] = [entries[index] for index in sorted(entries)]
        manifest_tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
        manifest_tmp.write_text(
            json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        manifest_tmp.replace(manifest_path)

    def teacher_trajectory_identity(self) -> dict[str, str]:
        if self.teacher_mode == "self_previous":
            return {"protocol": "self_previous", "sha256": "self_previous"}
        manifest = self._load_teacher_manifest()
        canonical = json.dumps(
            manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        return {
            "protocol": TEACHER_PROTOCOL,
            "sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        }

    def _kd_loss(self, logits: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if self.teacher is None or self.lambda_kd <= 0:
            return logits.sum() * 0.0
        with torch.no_grad():
            teacher_logits = self.teacher(x)
        temperature = self.temperature
        return F.kl_div(
            F.log_softmax(logits / temperature, dim=1),
            F.softmax(teacher_logits / temperature, dim=1),
            reduction="batchmean",
        ) * (temperature * temperature)

    def _quadratic_penalty(self) -> torch.Tensor:
        if not self.prev_params or not self.importance:
            return torch.tensor(0.0, device=self.device)
        penalty = torch.tensor(0.0, device=self.device)
        for name, parameter in self.model.named_parameters():
            if name in self.prev_params and name in self.importance:
                penalty = penalty + (
                    self.importance[name] * (parameter - self.prev_params[name]).pow(2)
                ).sum()
        return penalty

    def _ssr_penalty(self) -> torch.Tensor:
        penalty = torch.tensor(0.0, device=self.device)
        for _, layer in self._ssr_layers:
            mask = torch.zeros(layer.weight.size(0), dtype=torch.bool, device=self.device)
            for class_id in self._seen_classes:
                if class_id < mask.numel():
                    mask[class_id] = True
            seen_mask = mask if int(mask.sum()) > 1 else None
            penalty = penalty + compute_spatial_biocs(
                layer.weight.float(),
                self.a_exc,
                self.a_inh,
                self.sigma_exc,
                self.sigma_inh,
                seen_mask,
            )
        return penalty

    def _seen_mask(self) -> torch.Tensor | None:
        mask = torch.zeros(
            self._classifier.weight.size(0), dtype=torch.bool, device=self.device
        )
        for class_id in self._seen_classes:
            if class_id < mask.numel():
                mask[class_id] = True
        return mask if int(mask.sum()) > 1 else None

    def _geometry_penalty(
        self,
        labels: torch.Tensor,
        features: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.regularizer == "center":
            if features is None or self.centers is None:
                raise RuntimeError("Center regularization requires classifier features")
            targets = self.centers[labels].to(features.device)
            return (features.float() - targets.float()).pow(2).sum(dim=1).mean()
        mask = self._seen_mask()
        if self.regularizer == "protodecor":
            return prototype_decorrelation_loss(self._classifier.weight, mask)
        if self.regularizer == "spectral":
            return compute_spectral_flatness(self._classifier.weight.float(), mask)
        return self._classifier.weight.new_tensor(0.0)

    def training_step(self, x, y, task_id):
        self.training_batches += 1
        self._seen_classes.update(int(value) for value in y.detach().cpu().tolist())
        self._features = None
        logits = self.model(x)
        student_features = self._features
        loss = F.cross_entropy(logits, y) + self.lambda_kd * self._kd_loss(logits, x)
        if self.regularizer == "ewc":
            loss = loss + self.lambda_ewc * self._quadratic_penalty()
        elif self.regularizer == "mas":
            loss = loss + self.lambda_mas * self._quadratic_penalty()
        elif self.regularizer == "si":
            loss = loss + self.lambda_si * self._quadratic_penalty()
        elif self.regularizer == "center":
            loss = loss + self.lambda_center * self._geometry_penalty(y, student_features)
        elif self.regularizer == "protodecor":
            loss = loss + self.lambda_protodecor * self._geometry_penalty(y, student_features)
        elif self.regularizer == "spectral":
            loss = loss + self.lambda_spectral * self._geometry_penalty(y, student_features)
        elif self.regularizer == "ssr":
            with torch.amp.autocast("cuda", enabled=False):
                loss = loss + self.lambda_ssr * self._ssr_penalty()
        return loss, logits

    def _after_optimizer_step(self) -> None:
        self.optimizer_steps += 1

    def active_regularizer_weight(self) -> float:
        return float(
            {
                "none": 0.0,
                "ewc": self.lambda_ewc,
                "mas": self.lambda_mas,
                "si": self.lambda_si,
                "center": self.lambda_center,
                "protodecor": self.lambda_protodecor,
                "spectral": self.lambda_spectral,
                "ssr": self.lambda_ssr,
            }[self.regularizer]
        )

    def train_task(self, task_id: int, train_set, training_config: dict):
        self._task_train_set = train_set
        self._run_seed = int(training_config.get("run_seed", 0))
        self._task_id = int(task_id)
        if self.teacher_mode == "locked_kd_trajectory":
            self.teacher = None
            if task_id > 0:
                self._load_locked_teacher(task_id - 1)
        if self.regularizer == "si":
            self._init_params = {
                name: parameter.detach().clone()
                for name, parameter in self.model.named_parameters()
                if parameter.requires_grad
            }
            self._prev_step_params = {
                name: parameter.detach().clone()
                for name, parameter in self.model.named_parameters()
                if parameter.requires_grad
            }
            for name, parameter in self.model.named_parameters():
                if parameter.requires_grad:
                    self._w.setdefault(name, torch.zeros_like(parameter))
        super().train_task(task_id, train_set, training_config)

    def _accumulate_w(self) -> None:
        if self.regularizer != "si":
            return
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad or name not in self._prev_step_params:
                continue
            delta = parameter.detach() - self._prev_step_params[name]
            if parameter.grad is not None:
                self._w[name] = self._w[name] + (-parameter.grad.detach() * delta)
            self._prev_step_params[name] = parameter.detach().clone()

    def _compute_fisher(self) -> dict[str, torch.Tensor]:
        importance = {
            name: torch.zeros_like(parameter)
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }
        self.model.eval()
        count = 0
        generator = torch.Generator()
        generator.manual_seed(self._run_seed + 1_000_003 + 100_003 * self._task_id)
        for x, y in DataLoader(
            self._task_train_set,
            batch_size=32,
            shuffle=True,
            num_workers=2,
            generator=generator,
        ):
            if count >= self.fisher_samples:
                break
            x, y = x.to(self.device), y.to(self.device)
            self.model.zero_grad(set_to_none=True)
            F.cross_entropy(self.model(x), y).backward()
            for name, parameter in self.model.named_parameters():
                if parameter.requires_grad and parameter.grad is not None:
                    importance[name] += parameter.grad.detach().pow(2) * x.size(0)
            count += x.size(0)
        for name in importance:
            importance[name] /= max(count, 1)
        self.model.train()
        return importance

    def _compute_mas(self) -> dict[str, torch.Tensor]:
        importance = {
            name: torch.zeros_like(parameter)
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }
        self.model.eval()
        count = 0
        generator = torch.Generator()
        generator.manual_seed(self._run_seed + 2_000_003 + 100_003 * self._task_id)
        for x, _ in DataLoader(
            self._task_train_set,
            batch_size=32,
            shuffle=True,
            num_workers=2,
            generator=generator,
        ):
            if count >= self.importance_samples:
                break
            x = x.to(self.device)
            self.model.zero_grad(set_to_none=True)
            self.model(x).pow(2).mean().backward()
            for name, parameter in self.model.named_parameters():
                if parameter.requires_grad and parameter.grad is not None:
                    importance[name] += parameter.grad.detach().abs() * x.size(0)
            count += x.size(0)
        for name in importance:
            importance[name] /= max(count, 1)
        self.model.train()
        return importance

    def _consolidate_si(self) -> dict[str, torch.Tensor]:
        output = {}
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            delta = parameter.detach() - self._init_params[name]
            output[name] = torch.clamp(self._w[name] / (delta.pow(2) + self.xi), min=0.0)
            self._w[name] = torch.zeros_like(parameter)
        return output

    def _merge_importance(self, new_importance: dict[str, torch.Tensor]) -> None:
        for name, value in new_importance.items():
            if name in self.importance:
                self.importance[name] = self.importance[name] + value
            else:
                self.importance[name] = value

    def after_task(self, task_id: int):
        if self.regularizer == "ewc":
            self._merge_importance(self._compute_fisher())
        elif self.regularizer == "mas":
            self._merge_importance(self._compute_mas())
        elif self.regularizer == "si":
            self._merge_importance(self._consolidate_si())
        if self.regularizer in {"ewc", "mas", "si"}:
            self.prev_params = {
                name: parameter.detach().clone()
                for name, parameter in self.model.named_parameters()
                if parameter.requires_grad
            }
        if self.teacher_mode == "locked_kd_trajectory":
            if self.regularizer == "none":
                self._save_locked_teacher(task_id)
            self.teacher = None
        else:
            self.teacher = self._copy_teacher_without_feature_hook()
            self.teacher.eval()
            for parameter in self.teacher.parameters():
                parameter.requires_grad_(False)
