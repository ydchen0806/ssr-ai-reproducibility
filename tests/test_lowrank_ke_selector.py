from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from scripts.select_lowrank_ke_candidate import main


def _record(rank: int, learning_rate: float, coefficient: float, seed: int) -> dict:
    selected = rank == 8 and learning_rate == 0.0025 and coefficient == 0.03
    return {
        "protocol": "lowrank_lora_ssr_v1",
        "status": "complete",
        "n_edits": 100,
        "stream_seed": seed,
        "lora": {"rank": rank, "learning_rate": learning_rate, "num_steps": 60},
        "ssr": {"lambda": coefficient},
        "immediate": {
            "efficacy": 0.56 if selected else 0.50,
            "locality_target_consistency": 0.50,
            "rephrase": 0.50,
            "portability": 0.50,
        },
        "final_pre_edit_output_consistency": 51.0 if selected else 50.0,
        "final_history_efficacy": 20.0,
        "output_preservation_anchor": {"enabled": False},
    }


def test_selector_freezes_one_complete_eligible_cell(tmp_path: Path, monkeypatch):
    result_root = tmp_path / "results"
    seeds = [1, 2, 3, 4]
    for rank in (8, 40):
        for learning_rate in (0.0025, 0.0035):
            for coefficient in (0.0, 0.03):
                for seed in seeds:
                    path = result_root / str(rank) / str(learning_rate) / str(coefficient) / str(seed)
                    path.mkdir(parents=True)
                    (path / "results.json").write_text(
                        json.dumps(_record(rank, learning_rate, coefficient, seed)),
                        encoding="utf-8",
                    )

    output = tmp_path / "selection.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "select_lowrank_ke_candidate.py",
            "--result-root", str(result_root),
            "--ranks", "8", "40",
            "--learning-rates", "0.0025", "0.0035",
            "--lora-steps", "60",
            "--seeds", *map(str, seeds),
            "--lambdas", "0.03",
            "--output", str(output),
        ],
    )

    main()

    selection = json.loads(output.read_text(encoding="utf-8"))
    assert selection["status"] == "selected"
    assert selection["selected"]["rank"] == 8
    assert selection["selected"]["learning_rate"] == 0.0025
    assert selection["selected"]["lambda"] == 0.03
    assert selection["selected"]["mean_delta_pp"]["efficacy"] == pytest.approx(6.0)
    assert selection["selected"]["mean_delta_pp"]["locality"] == pytest.approx(1.0)
    assert selection["confirmation_was_not_read"] is True
