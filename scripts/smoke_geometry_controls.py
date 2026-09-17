#!/usr/bin/env python3
"""CPU smoke test for matched geometry-control ablations."""

from __future__ import annotations

from pathlib import Path
import json
import sys
import subprocess
import tempfile

import torch
import torch.nn as nn
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from methods.builder import build_method
from models.builder import build_model


CONTROL_CONFIGS = [
    "configs/geomctrl_cosorth_c100.yaml",
    "configs/geomctrl_proto_decor_c100.yaml",
    "configs/geomctrl_spectral_c100.yaml",
    "configs/geomctrl_center_loss_c100.yaml",
    "configs/geomctrl_supcon_c100.yaml",
    "configs/geomctrl_kd_weight_decay_c100.yaml",
]


class TinyNet(nn.Module):
    def __init__(self, n_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 8 * 8, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def run_loss_smoke(control_type: str) -> None:
    torch.manual_seed(7)
    model = TinyNet()
    method = build_method(
        {
            "name": "geometry_controls",
            "control_type": control_type,
            "lambda_control": 0.01,
            "lambda_distill": 1.0,
            "temperature": 2.0,
        },
        model=model,
        device=torch.device("cpu"),
    )
    x = torch.randn(16, 3, 8, 8)
    y = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7])
    loss, logits = method.training_step(x, y, task_id=0)
    assert torch.isfinite(loss), f"{control_type}: non-finite loss"
    assert logits.shape == (16, 10), f"{control_type}: wrong logits shape {logits.shape}"
    loss.backward()
    method.after_task(0)
    loss2, _ = method.training_step(x, y, task_id=1)
    assert torch.isfinite(loss2), f"{control_type}: non-finite KD loss"
    loss2.backward()


def run_config_build_smoke() -> None:
    for cfg_path in CONTROL_CONFIGS:
        cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
        model = build_model(cfg["model"], num_classes=100)
        method = build_method(cfg["method"], model=model, device=torch.device("cpu"))
        assert method.__class__.__name__ == "GeometryControls", cfg_path


def write_summary(path: Path, aa: float, af: float, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"avg_accuracy": aa, "avg_forgetting": af, "seed": seed}),
        encoding="utf-8",
    )


def run_summarizer_smoke() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        result_root = tmpdir / "results"
        write_summary(result_root / "lwf_split_cifar100" / "seed_42" / "summary.json", 72.0, 8.0, 42)
        write_summary(result_root / "biocs_plus_s05_c100" / "seed_42" / "summary.json", 72.5, 1.5, 42)
        md_out = tmpdir / "summary.md"
        json_out = tmpdir / "summary.json"
        subprocess.run(
            [
                "python",
                "scripts/summarize_geometry_control_ablations.py",
                "--result-root",
                str(result_root),
                "--out",
                str(md_out),
                "--json-out",
                str(json_out),
            ],
            check=True,
        )
        summary = json.loads(json_out.read_text(encoding="utf-8"))
        assert summary["broad_ai_method_claim_gate"] == "not_ready"
        assert any("geomctrl_cosorth_c100" in item for item in summary["missing"])
        text = md_out.read_text(encoding="utf-8")
        assert "Broad AI-method claim gate" in text
        assert "not_ready" in text


def main() -> None:
    for control_type in [
        "cosine_orthogonal",
        "prototype_decorrelation",
        "spectral",
        "center_loss",
        "supervised_contrastive",
        "none",
    ]:
        run_loss_smoke(control_type)
    run_config_build_smoke()
    run_summarizer_smoke()
    print("geometry_controls smoke passed")


if __name__ == "__main__":
    main()
