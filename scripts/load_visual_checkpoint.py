"""Load exported inference tensors without unpickling Python objects."""
import json
from safetensors import safe_open
from safetensors.torch import load_file


def load_export(path, strip_ddp_prefix=False):
    state = load_file(str(path))
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        groups = json.loads((handle.metadata() or {}).get("groups", "{}"))
    if groups:
        return {key: ({name[len(key) + 1:]: value for name, value in state.items()
                       if name.startswith(key + ".")} if kind == "state_dict" else state[key])
                for key, kind in groups.items()}
    if strip_ddp_prefix:
        if not all(key.startswith("module.") for key in state):
            raise ValueError("Checkpoint does not have a uniform DDP prefix")
        return {key[len("module."):]: value for key, value in state.items()}
    return state
