import importlib.util
from pathlib import Path


def test_current_counterfact_bundle_matches_reported_values():
    script = Path(__file__).resolve().parents[1] / "scripts" / "verify_paper_20260810.py"
    spec = importlib.util.spec_from_file_location("verify_paper_20260810", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    verified = module.verify()
    assert len(verified) == 4
