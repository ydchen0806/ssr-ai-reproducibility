from __future__ import annotations

import copy

import yaml

from scripts.verify_locked_editing_assets import verify_assets
from ssr_utils.result_schema import sha256_file, sha256_value


def test_locked_asset_verifier_detects_dataset_tampering(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    model = root / "model"
    model.mkdir(parents=True)
    (model / "config.json").write_text("model", encoding="utf-8")
    dataset = root / "dataset.json"
    dataset.write_text("data", encoding="utf-8")
    model_files = {"config.json": sha256_file(model / "config.json")}
    config = {
        "model": {
            "fingerprint_files": model_files,
            "fingerprint_sha256": sha256_value(model_files),
        },
        "datasets": {
            "toy": {"path": "dataset.json", "sha256": sha256_file(dataset)}
        },
    }
    config_path = root / "locked.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(
        "scripts.verify_locked_editing_assets.PROJECT_ROOT", root
    )

    assert verify_assets(config_path, model)["status"] == "PASS"
    dataset.write_text("changed", encoding="utf-8")
    report = verify_assets(config_path, model)
    assert report["status"] == "FAIL"
    assert any("dataset hash mismatch" in error for error in report["errors"])
