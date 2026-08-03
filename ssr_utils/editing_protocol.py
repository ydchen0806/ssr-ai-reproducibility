"""Locked shared identities for the meeting editing attribution cohort."""

from __future__ import annotations

from llm_ke.locality import LOCALITY_PROTOCOL_HASH
from ssr_utils.result_schema import sha256_value


GPT2_XL_FINGERPRINT_FILES = {
    "config.json": "be48115d52314bc27327e160bf10197760db02ea689d10adb24db4102edf7a8a",
    "generation_config.json": "b90eadacf585a743a30ea51e8b5c88b8d282a2a34dc0c7e556d0987cdbd68805",
    "merges.txt": "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5",
    "model.safetensors": "0f8b28eb05a8075f48b61b6f35332978c74fc7763fa9fb4051a1c30511736a6a",
    "tokenizer.json": "8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6",
    "tokenizer_config.json": "5e04eb606e3a1583530a42e36c2a6b6615c86f34fe77e44d9ddeb43ff940931f",
    "vocab.json": "196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783",
}
GPT2_XL_MODEL_HASH = sha256_value(GPT2_XL_FINGERPRINT_FILES)

KNOWEDIT_DATASET_FILES = {
    "zsre": {
        "path": "dataset/knowedit/benchmark/ZsRE/ZsRE-test-all.json",
        "sha256": "0e0214dda853a906ef02bef58c7ce2978af5a962e91fc4a10827e0125459905c",
    },
    "cf": {
        "path": "dataset/knowedit/benchmark/wiki_counterfact/test_cf.json",
        "sha256": "efc01ba398dfe04975939f729aab45523addb3c6b3cf41d7429ba5de1d45225b",
    },
    "recent": {
        "path": "dataset/knowedit/benchmark/wiki_recent/recent_test.json",
        "sha256": "e5c4ce6de1e8e2d10d892164ad6f4cbd28c976ed509e29e78ae853b01de1797b",
    },
}

EDITING_PAIRING_PROTOCOL_SPEC = {
    "protocol": "meeting_20260803_editing_matched_v2",
    "model": "gpt2-xl",
    "model_hash": GPT2_XL_MODEL_HASH,
    "target_layers": [16, 17, 18],
    "target_module_regex": "(c_proj|down_proj)$",
    "max_target_modules": 0,
    "optimizer": "adam",
    "lr": 0.0001,
    "num_steps": 25,
    "max_length": 64,
    "model_dtype": "auto",
    "device_map": "",
    "force_text_only": True,
    "local_files_only": True,
    "data_order": "contiguous_source_order",
    "row_sampling_seed": "run_seed",
    "evaluate_history": True,
    "history_checkpoints": [50],
    "history_max_samples": 0,
    "locality_protocol_hash": LOCALITY_PROTOCOL_HASH,
}
EDITING_PAIRING_PROTOCOL_HASH = sha256_value(EDITING_PAIRING_PROTOCOL_SPEC)
