import pytest
import torch
from llm_ke.checkpoint_io import save_lora_snapshot, load_lora_snapshot


def model(rank=2):
    outer = torch.nn.Module()
    outer.block = torch.nn.Module()
    outer.block.lora_A = torch.nn.ModuleDict({"default": torch.nn.Linear(3, rank, bias=False)})
    outer.block.lora_B = torch.nn.ModuleDict({"default": torch.nn.Linear(rank, 3, bias=False)})
    return outer


def test_roundtrip(tmp_path):
    original, restored = model(), model()
    rng = torch.get_rng_state().clone()
    path = tmp_path / "adapter.safetensors"
    record = save_lora_snapshot(original, path, {"seed": 1, "rank": 2})
    assert torch.equal(rng, torch.get_rng_state())
    load_lora_snapshot(restored, path)
    assert record["tensor_count"] == 2
    assert all(torch.equal(value, restored.state_dict()[key]) for key, value in original.state_dict().items())
    with pytest.raises(FileExistsError):
        save_lora_snapshot(original, path, {})
    with pytest.raises(ValueError, match="mismatch"):
        load_lora_snapshot(model(3), path)


def test_no_adapters(tmp_path):
    with pytest.raises(ValueError, match="No LoRA"):
        save_lora_snapshot(torch.nn.Linear(2, 3), tmp_path / "invalid.safetensors", {})
