from __future__ import annotations

import subprocess
import sys


def test_empty_registry_requires_explicit_staged_mode(tmp_path):
    registry = tmp_path / "claims.yaml"
    registry.write_text("claims: []\n", encoding="utf-8")
    command = [
        sys.executable,
        "scripts/check_result_consistency.py",
        "--registry",
        str(registry),
    ]
    failed = subprocess.run(command, capture_output=True, text=True)
    assert failed.returncode != 0
    staged = subprocess.run(command + ["--allow-empty"], capture_output=True, text=True)
    assert staged.returncode == 0
    assert "STAGED" in (tmp_path / "consistency_report.md").read_text(encoding="utf-8")
