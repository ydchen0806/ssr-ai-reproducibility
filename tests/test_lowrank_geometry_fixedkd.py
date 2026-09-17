import importlib.util
import json
from argparse import Namespace
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).parents[1] / "experiments" / "lowrank_geometry_fixedkd.py"
SPEC = importlib.util.spec_from_file_location("lowrank_geometry_fixedkd", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_geometry_losses_are_finite_and_differentiable():
    weights = torch.randn(8, 16, requires_grad=True)
    losses = [
        MODULE.geometry_loss(
            method,
            weights,
            a_exc=1.0,
            a_inh=0.8,
            sigma_exc=0.55,
            sigma_inh=1.25,
        )
        for method in MODULE.TUNED_METHODS
    ]
    assert all(torch.isfinite(loss) for loss in losses)
    sum(losses).backward()
    assert weights.grad is not None
    assert torch.isfinite(weights.grad).all()


def _write_record(root, stage, method, seed, coefficient, aa, af):
    path = root / stage / f"{method}_coefficient_{coefficient}" / f"seed_{seed}" / "result.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "method": method,
                "seed": seed,
                "coefficient": coefficient,
                "rank": 16,
                "avg_accuracy": aa,
                "avg_forgetting": af,
            }
        )
    )


def test_selector_prefers_dual_positive_candidate(tmp_path):
    seeds = [1, 2, 3]
    for seed in seeds:
        _write_record(tmp_path, "development", "kd", seed, 0.0, 70.0, 10.0)
    for method in MODULE.TUNED_METHODS:
        for seed in seeds:
            _write_record(tmp_path, "development", method, seed, 0.1, 71.0, 9.0)
            _write_record(tmp_path, "development", method, seed, 1.0, 73.0, 11.0)
    output = tmp_path / "selection.json"
    result = MODULE.select(Namespace(result_root=str(tmp_path), seeds=seeds, output=str(output)))
    assert all(row["coefficient"] == 0.1 for row in result["selected"].values())
    assert all(row["paired_nonregression_gate"] for row in result["selected"].values())
    assert result["confirmation_gate"]["passed"] is False


def test_selector_opens_confirmation_only_when_ssr_leads(tmp_path):
    seeds = [1, 2, 3]
    for seed in seeds:
        _write_record(tmp_path, "development", "kd", seed, 0.0, 70.0, 10.0)
    for method in ("kd_orthogonal", "kd_protodecor", "kd_spectral"):
        for seed in seeds:
            _write_record(tmp_path, "development", method, seed, 0.1, 70.5, 9.5)
            _write_record(tmp_path, "development", method, seed, 1.0, 69.0, 11.0)
    for seed in seeds:
        _write_record(tmp_path, "development", "kd_ssr", seed, 0.1, 72.0, 8.0)
        _write_record(tmp_path, "development", "kd_ssr", seed, 1.0, 69.0, 11.0)
    result = MODULE.select(
        Namespace(result_root=str(tmp_path), seeds=seeds, output=str(tmp_path / "selection.json"))
    )
    assert result["confirmation_gate"]["passed"] is True
    assert all(
        margin["aa_gain_pp"] > 0 and margin["af_reduction_pp"] > 0
        for margin in result["confirmation_gate"]["ssr_vs_selected_geometry_controls"].values()
    )


def test_summary_gate_requires_ssr_to_beat_every_control(tmp_path):
    seeds = [11, 13, 15]
    values = {
        "kd": (70.0, 10.0),
        "kd_orthogonal": (70.5, 9.5),
        "kd_protodecor": (70.7, 9.3),
        "kd_spectral": (70.9, 9.1),
        "kd_ssr": (72.0, 8.0),
    }
    for method, (aa, af) in values.items():
        for seed in seeds:
            _write_record(tmp_path, "confirmation", method, seed, 0.1, aa, af)
    result = MODULE.summarize(
        Namespace(
            result_root=str(tmp_path),
            seeds=seeds,
            output=str(tmp_path / "summary.json"),
            csv_output=str(tmp_path / "seed_level.csv"),
        )
    )
    assert result["main_text_gate"]["passed"] is True
    assert result["contrasts"]["kd_ssr_vs_kd_spectral"]["aa_gain_pp"]["mean"] > 0
