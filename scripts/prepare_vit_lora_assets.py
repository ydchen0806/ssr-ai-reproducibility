#!/usr/bin/env python3
"""Serially download raw datasets and pretrained ViT weights before workers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--pretrained-checkpoint", required=True, type=Path)
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()

    from datasets.builder import build_benchmark
    from models.builder import build_model

    prepared = []
    for path in args.configs:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        dataset_config = dict(config["dataset"])
        dataset_config["data_root"] = str(args.data_root)
        dataset_config["download"] = args.allow_download
        config["model"]["pretrained_checkpoint"] = str(args.pretrained_checkpoint)
        benchmark = build_benchmark(dataset_config)
        model = build_model(config["model"], num_classes=benchmark.n_classes)
        prepared.append(
            {
                "config": str(path),
                "dataset": dataset_config["name"],
                "train": len(benchmark.train_data),
                "test": len(benchmark.test_data),
                "model": config["model"]["name"],
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
            }
        )
    print(json.dumps(prepared, indent=2))


if __name__ == "__main__":
    main()
