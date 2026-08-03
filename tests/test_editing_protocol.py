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
