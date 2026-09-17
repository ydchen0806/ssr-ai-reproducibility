#!/usr/bin/env python3
"""Rebuild numerical figures into the release, without changing a manuscript."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/manuscript_20260911/scripts"
sys.path.insert(0, str(SOURCE))

if __name__ == "__main__":
    import build_final_strict_figures as figure4
    import build_figure5_current as figure5

    figure4.build_figure4()
    figure5.configure_style()
    figure5.build_main_figure()
    figure5.build_extended_data()
    print("Rebuilt Figure 4 numerical panels and current Figure 5 + Extended Data")
