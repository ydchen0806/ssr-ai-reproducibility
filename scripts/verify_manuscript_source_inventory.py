#!/usr/bin/env python3
"""Check numerical-source inventory hashes before rebuilding a figure."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1] / "artifacts/manuscript_20260911"
if __name__ == "__main__":
    records = json.loads((ROOT / "source_inventory.json").read_text())
    for row in records:
        path = (ROOT / row["path"]).resolve()
        if not path.is_relative_to(ROOT.resolve()):
            raise ValueError("Inventory path leaves release directory")
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"Hash mismatch: {row['path']}")
    print(f"Verified {len(records)} source files")
