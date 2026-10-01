#!/usr/bin/env python3
"""Fig. 5 fixed-window rehearsal runner.

This wrapper keeps the released EasyEdit/LoRA optimizer and SSR operator, but
trains each edit on the current target plus a deterministic, evenly spaced
rehearsal subset of earlier targets. Qwen's turn-ending token is supervised so
greedy completions have an explicit stopping target. It is deliberately a
separate protocol from the archived Fig. 5 runner.
"""

from __future__ import annotations

import json
import hashlib
import os
import re
import subprocess
from pathlib import Path

import run_lowrank_ke_easyedit as runner


WINDOW = int(os.environ.get("SSR_REHEARSAL_WINDOW", "8"))
TARGET_EOS = os.environ.get("SSR_TARGET_EOS", "<|im_end|>")
if WINDOW < 1:
    raise ValueError("SSR_REHEARSAL_WINDOW must be positive")
if TARGET_EOS != "<|im_end|>":
    raise ValueError("This Qwen protocol requires the explicit <|im_end|> token")


def boundary_target_match(prediction: str, target) -> float:
    """Match complete normalized target phrases, avoiding substring false positives."""
    prediction_text = runner._to_text(prediction).strip().lower()
    for value in runner._to_text_list(target):
        phrase = value.strip().lower()
        if phrase and re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", prediction_text):
            return 1.0
    return 0.0


runner._target_match = boundary_target_match
history: list[dict] = []


def replay_positions(n_previous: int, window: int) -> list[int]:
    if n_previous <= window:
        return list(range(n_previous))
    if window == 1:
        return [n_previous - 1]
    return sorted({round(i * (n_previous - 1) / (window - 1)) for i in range(window)})


def edit_one_with_rehearsal(editor, sample: dict):
    history.append(sample)
    previous = history[:-1]
    positions = replay_positions(len(previous), WINDOW)
    requests = [previous[position] for position in positions] + [sample]
    token_ids = editor.tok(TARGET_EOS, add_special_tokens=False)["input_ids"]
    if token_ids != [151645] or editor.tok.pad_token_id == token_ids[0]:
        raise RuntimeError("Qwen <|im_end|> is unavailable or aliases the padding token")

    payload = []
    for item in requests:
        payload.append(
            {
                "prompt": item["prompt"],
                "target_new": runner._to_text(item["target_new"]) + TARGET_EOS,
                "ground_truth": runner._to_text(item["ground_truth"]),
                "subject": runner._to_text(item.get("subject", "")),
            }
        )

    # One optimizer update contains the complete current-plus-rehearsal batch.
    # This keeps the declared optimizer-update budget matched to the archived
    # protocol while exposing the old targets on every edit.
    editor.hparams.batch_size = len(payload)
    edited_model, _ = editor.apply_algo(
        editor.model,
        editor.tok,
        payload,
        editor.hparams,
        copy=False,
        return_orig_weights=True,
        keep_original_weight=False,
    )
    # EasyEdit does not expose its internal per-step loss stream.  A finite
    # parameter check immediately after every optimizer call catches the
    # NaN/Inf failure mode that otherwise produces plausible-looking JSON.
    for name, parameter in edited_model.named_parameters():
        if parameter.requires_grad and not parameter.detach().isfinite().all().item():
            raise FloatingPointError(f"non-finite trainable parameter after edit: {name}")
    editor.model = edited_model
    return edited_model


runner.edit_one = edit_one_with_rehearsal


if __name__ == "__main__":
    runner.main()
    output = Path(os.environ["SSR_RESULT_PATH"]).resolve()
    record = json.loads(output.read_text(encoding="utf-8"))
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = None
    record["fig5_rehearsal_protocol"] = {
        "protocol": "fig5_lowrank_rehearsal_v2",
        "status": "new_rehearsal_protocol_not_an_original_fig5_replication",
        "rehearsal_window": WINDOW,
        "rehearsal_selection": "evenly_spaced_previous_targets_plus_current",
        "target_suffix": TARGET_EOS,
        "target_suffix_token_id": 151645,
        "target_match": "complete_normalized_phrase_boundary_match",
        "optimizer_updates_per_edit": int(record["lora"]["num_steps"]),
        "source_git_commit": git_commit,
        "runner_sha256": hashlib.sha256(
            Path(runner.__file__).resolve().read_bytes()
        ).hexdigest(),
        "wrapper_sha256": hashlib.sha256(
            Path(__file__).resolve().read_bytes()
        ).hexdigest(),
    }
    runner.atomic_json(output, record)
