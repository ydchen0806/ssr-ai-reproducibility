from __future__ import annotations

import csv
import json
from collections import defaultdict

import matplotlib.pyplot as plt
import pytest

from scripts.make_meeting_revision_figures import (
    EvidenceError,
    build_figure5,
    generate_figures,
    load_evidence,
    required_specs,
)


FIELDS = [
    "cohort",
    "task_family",
    "dataset",
    "model",
    "contrast",
    "mapping",
    "metric",
    "recipe",
    "higher_is_better",
    "n",
    "control_mean",
    "control_sd",
    "treatment_mean",
    "treatment_sd",
    "raw_difference_mean",
    "raw_difference_sd",
    "difference_mean",
    "difference_sd",
    "ci95_low",
    "ci95_high",
    "favorable_pairs",
]


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _complete_tables(root, *, n=2, short_spec=None):
    figure4, figure5 = required_specs(
        expected_cl_n=n,
        expected_seg_n=n,
        expected_editing_n=n,
    )
    grouped = defaultdict(list)
    for index, spec in enumerate((*figure4, *figure5), start=1):
        observed_n = n - 1 if spec == short_spec else n
        difference = index / 10.0
        grouped[spec.table].append(
            {
                "cohort": "synthetic",
                "task_family": (
                    "classification"
                    if spec.table == "matched_kd"
                    else "segmentation"
                    if spec.table == "segmentation"
                    else "editing"
                ),
                "dataset": spec.dataset,
                "model": "synthetic-model",
                "contrast": spec.contrast,
                "mapping": spec.mapping,
                "metric": spec.metric,
                "recipe": spec.recipe,
                "higher_is_better": "True",
                "n": observed_n,
                "control_mean": 20.0 + index,
                "control_sd": 0.5,
                "treatment_mean": 20.0 + index + difference,
                "treatment_sd": 0.6,
                "raw_difference_mean": difference,
                "raw_difference_sd": 0.4,
                "difference_mean": difference,
                "difference_sd": 0.4,
                "ci95_low": difference - 0.2,
                "ci95_high": difference + 0.2,
                "favorable_pairs": observed_n,
            }
        )
    filenames = {
        "matched_kd": "kd_matched_cl.csv",
        "segmentation": "cub_segmentation.csv",
        "editing": "editing_ablation.csv",
        "mapping": "editing_mapping.csv",
    }
    for table, filename in filenames.items():
        _write(root / filename, grouped[table])


def test_submission_mode_fails_clearly_when_generated_tables_are_missing(tmp_path):
    with pytest.raises(EvidenceError, match="missing required generated table"):
        load_evidence(tmp_path)


def test_fixed_kd_ssr_figure_gate_uses_cosine_mapping():
    figure4, _ = required_specs()
    kd_ssr = [spec for spec in figure4 if spec.contrast == "kd_ssr_minus_kd"]

    assert kd_ssr
    assert {spec.mapping for spec in kd_ssr} == {"cosine"}


def test_submission_mode_rejects_a_short_paired_cohort(tmp_path):
    figure4, _ = required_specs(expected_cl_n=2, expected_seg_n=2, expected_editing_n=2)
    short_spec = figure4[0]
    _complete_tables(tmp_path, n=2, short_spec=short_spec)

    with pytest.raises(EvidenceError, match=r"incomplete contrast: .*observed n=1"):
        load_evidence(
            tmp_path,
            expected_cl_n=2,
            expected_seg_n=2,
            expected_editing_n=2,
        )


def test_staged_mode_generates_marked_pdf_and_png_from_empty_csvs(tmp_path):
    tables = tmp_path / "tables"
    output = tmp_path / "figures"
    for filename in (
        "kd_matched_cl.csv",
        "cub_segmentation.csv",
        "editing_ablation.csv",
        "editing_mapping.csv",
    ):
        _write(tables / filename, [])

    report = generate_figures(
        tables,
        output,
        allow_incomplete=True,
        expected_cl_n=2,
        expected_seg_n=2,
        expected_editing_n=2,
        dpi=72,
    )

    assert report["status"] == "staged_incomplete"
    assert report["issues"]
    for stem in ("figure4_evidence_summary", "figure5_evidence_summary"):
        assert (output / f"{stem}.pdf").stat().st_size > 1000
        assert (output / f"{stem}.png").stat().st_size > 1000


def test_complete_tables_generate_submission_figures_and_provenance(tmp_path):
    tables = tmp_path / "tables"
    output = tmp_path / "figures"
    _complete_tables(tables, n=2)

    report = generate_figures(
        tables,
        output,
        expected_cl_n=2,
        expected_seg_n=2,
        expected_editing_n=2,
        dpi=72,
    )

    assert report["status"] == "complete"
    assert not report["issues"]
    assert set(report["source_sha256"]) == {
        "kd_matched_cl.csv",
        "cub_segmentation.csv",
        "editing_ablation.csv",
        "editing_mapping.csv",
    }
    saved = json.loads((output / "figure_evidence_provenance.json").read_text())
    assert saved["claim_labels"]["full_minus_plain"] == "recipe transfer"
    assert saved["claim_labels"]["full_minus_stabilized"] == "direct attribution"


def test_figure5_visibly_distinguishes_direct_attribution_from_transfer(tmp_path):
    _complete_tables(tmp_path, n=2)
    bundle = load_evidence(
        tmp_path,
        expected_cl_n=2,
        expected_seg_n=2,
        expected_editing_n=2,
    )

    figure = build_figure5(bundle)
    titles = [axis.get_title(loc="left") for axis in figure.axes]
    footer = " ".join(text.get_text() for text in figure.texts)
    plt.close(figure)

    assert any("Direct SSR attribution" in title for title in titles)
    assert any("Recipe transfer" in title and "not isolated SSR" in title for title in titles)
    assert "Direct attribution holds" in footer
