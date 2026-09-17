from __future__ import annotations

from pathlib import Path

import yaml

from llm_ke.locality import LOCALITY_PROTOCOL_HASH
from ssr_utils.editing_protocol import (
    EDITING_PAIRING_PROTOCOL_HASH,
    GPT2_XL_FINGERPRINT_FILES,
    GPT2_XL_MODEL_HASH,
    KNOWEDIT_DATASET_FILES,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_locked_yaml_matches_code_derived_protocol_hashes():
    config = yaml.safe_load(
        (PROJECT_ROOT / "configs/meeting_20260803/locked_editing_gpt2xl.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert config["model"]["fingerprint_files"] == GPT2_XL_FINGERPRINT_FILES
    assert config["model"]["fingerprint_sha256"] == GPT2_XL_MODEL_HASH
    assert config["datasets"] == KNOWEDIT_DATASET_FILES
    assert config["evaluation"]["protocol_sha256"] == LOCALITY_PROTOCOL_HASH
    assert config["pairing"]["protocol_sha256"] == EDITING_PAIRING_PROTOCOL_HASH


def test_locked_yaml_contains_all_replay_critical_editor_settings():
    config = yaml.safe_load(
        (PROJECT_ROOT / "configs/meeting_20260803/locked_editing_gpt2xl.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert config["editor"] == {
        "target_module_regex": "(c_proj|down_proj)$",
        "max_target_modules": 0,
        "editable_parameter": "weight",
        "adapter_mode": "none",
    }
    assert config["sequence"] == {"max_length": 64, "max_new_tokens": 32}
    assert config["optimizer"] == {
        "name": "adam",
        "lr": 0.0001,
        "num_steps": 25,
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": 1.0e-8,
        "weight_decay": 0.0,
        "gradient_clip_norm": 1.0,
    }
    assert config["runtime"] == {
        "editor_device": "cuda",
        "model_dtype": "auto",
        "device_map": "",
        "attention_implementation": "",
        "cuda_visible_devices": "inherited_from_scheduler",
        "determinism_policy": "seeded_best_effort",
        "cudnn_deterministic": False,
        "cudnn_benchmark": False,
        "use_deterministic_algorithms": False,
        "deterministic_algorithms_warn_only": False,
    }
