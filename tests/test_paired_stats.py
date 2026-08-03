from __future__ import annotations

import pytest

from ssr_utils.paired_stats import complete_pairs, paired_summary


def test_paired_summary_uses_favorable_direction():
    summary = paired_summary([5.0, 4.0, 3.0], [4.0, 3.0, 2.0], higher_is_better=False)
    assert summary.difference_mean == pytest.approx(1.0)
    assert summary.raw_difference_mean == pytest.approx(-1.0)
    assert summary.control_sd == pytest.approx(1.0)
    assert summary.treatment_sd == pytest.approx(1.0)
    assert summary.favorable_pairs == 3


def test_complete_pairs_reports_missing_identities():
    pairs, missing = complete_pairs(
        [{"dataset": "a", "seed": 1}, {"dataset": "a", "seed": 2}],
        [{"dataset": "a", "seed": 2}, {"dataset": "a", "seed": 3}],
        identity_fields=("dataset", "seed"),
    )
    assert len(pairs) == 1
    assert missing == [("a", 1), ("a", 3)]


def test_paired_summary_rejects_unmatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        paired_summary([1.0, 2.0], [2.0])
