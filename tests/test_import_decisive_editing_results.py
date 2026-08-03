from __future__ import annotations

import json

from scripts.import_decisive_editing_results import import_one


def test_locked_full_result_is_imported_with_direct_objective(tmp_path):
    raw_root = tmp_path / "raw"
    raw_path = raw_root / "zsre" / "zsre_full_x_seed9311" / "results.json"
    raw_path.parent.mkdir(parents=True)
    raw_path.write_text(
        json.dumps(
            {
                "seed": 9311,
                "requested_n_edits": 2,
                "n_edits": 2,
                "results": [{}, {}],
                "efficacy": 99.0,
                "locality": 12.0,
                "elapsed_s": 1.0,
                "kernel_family": "gaussian",
                "a_exc": 1.2,
                "a_inh": 0.9,
                "sigma_exc": 0.22,
                "sigma_inh": 0.6,
                "distance_metric": "projective",
                "data_offset": 500,
                "historical_retention": {
                    "final": {"efficacy": 60.0, "locality": 15.0}
                },
            }
        ),
        encoding="utf-8",
    )
    output = import_one(
        raw_path,
        raw_root=raw_root,
        output_root=tmp_path / "out",
        source_fingerprint="abc123",
    )

    record = json.loads(output.read_text(encoding="utf-8"))
    assert (output.parent / "results.json").is_file()
    assert record["recipe"] == "full"
    assert record["objective"]["ssr"] is True
    assert record["objective"]["anchor"] is True
    assert record["distance_mapping"] == "projective"
    assert record["status"] == "complete"
    assert record["metrics"]["final_history_locality"] == 15.0
