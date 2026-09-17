"""Exact, inference-only LoRA tensor snapshots without pickled objects."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch


def adapter_tensors(model):
    return {name: tensor for name, tensor in model.state_dict().items()
            if ".lora_A." in name or ".lora_B." in name}


def save_lora_snapshot(model, path: Path, metadata: dict) -> dict:
    from safetensors.torch import load_file, save_file

    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    state = {name: tensor.detach().cpu().contiguous().clone()
             for name, tensor in adapter_tensors(model).items()}
    if not state:
        raise ValueError("No LoRA A/B tensors found")
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(state, str(path), metadata={"format": "pt", "provenance": json.dumps(metadata)})
    restored = load_file(str(path))
    if state.keys() != restored.keys() or any(not torch.equal(v, restored[k]) for k, v in state.items()):
        raise RuntimeError("Saved LoRA checkpoint failed exact tensor verification")
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "tensor_count": len(state), "bytes": path.stat().st_size}


def load_lora_snapshot(model, path: Path) -> None:
    """Load into an already matching base model and PEFT configuration."""
    from safetensors.torch import load_file

    restored = load_file(str(path))
    destination = adapter_tensors(model)
    if destination.keys() != restored.keys():
        raise ValueError("LoRA module names do not match the snapshot")
    for key, value in restored.items():
        if destination[key].shape != value.shape or destination[key].dtype != value.dtype:
            raise ValueError(f"Shape or dtype mismatch: {key}")
    with torch.no_grad():
        for key, value in restored.items():
            destination[key].copy_(value)
