from __future__ import annotations

import subprocess

from scripts.check_experiment_worktree import worktree_violations


def _git(root, *arguments):
    subprocess.run(["git", "-C", str(root), *arguments], check=True, capture_output=True)


def test_worktree_check_allows_data_assets_but_rejects_untracked_source(tmp_path):
    _git(tmp_path, "init")
    (tmp_path / "tracked.py").write_text("x = 1\n", encoding="utf-8")
    _git(tmp_path, "add", "tracked.py")
    _git(
        tmp_path,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-m",
        "initial",
    )
    (tmp_path / "dataset").mkdir()
    (tmp_path / "dataset" / "asset.bin").write_bytes(b"data")
    assert worktree_violations(tmp_path) == []

    (tmp_path / "new_runner.py").write_text("print('untracked')\n", encoding="utf-8")
    assert worktree_violations(tmp_path) == ["?? new_runner.py"]


def test_worktree_check_rejects_tracked_modification(tmp_path):
    _git(tmp_path, "init")
    path = tmp_path / "tracked.py"
    path.write_text("x = 1\n", encoding="utf-8")
    _git(tmp_path, "add", "tracked.py")
    _git(
        tmp_path,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-m",
        "initial",
    )
    path.write_text("x = 2\n", encoding="utf-8")
    assert worktree_violations(tmp_path) == [" M tracked.py"]
