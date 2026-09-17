import hashlib
import json
import pytest
from scripts.publish_checkpoint_release import verify_local


def test_release_integrity(tmp_path):
    payload = b"test checkpoint bytes"
    path = tmp_path / "test.safetensors"
    path.write_bytes(payload)
    manifest = {"checkpoints": [{"path": path.name, "bytes": len(payload),
                                "sha256": hashlib.sha256(payload).hexdigest()}]}
    (tmp_path / "checkpoint_manifest.json").write_text(json.dumps(manifest))
    assert verify_local(tmp_path) == manifest
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="mismatch"):
        verify_local(tmp_path)


def test_reject_traversal(tmp_path):
    (tmp_path / "checkpoint_manifest.json").write_text(json.dumps(
        {"checkpoints": [{"path": "../outside", "bytes": 0, "sha256": ""}]}))
    with pytest.raises(ValueError, match="Invalid"):
        verify_local(tmp_path)
