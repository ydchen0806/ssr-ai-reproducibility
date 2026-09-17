#!/usr/bin/env python3
"""Reject source drift while allowing explicitly ignored runtime asset roots."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


ALLOWED_UNTRACKED_ROOTS = {
    "checkpoints",
    "data",
    "dataset",
    "external",
    "hugging_cache",
    "logs",
    "results",
    "runs",
    "wandb",
}


def worktree_violations(project_root: Path) -> list[str]:
    output = subprocess.run(
        [
            "git",
            "-C",
            str(project_root),
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
        ],
        check=True,
        capture_output=True,
    ).stdout
    fields = [field.decode("utf-8", errors="replace") for field in output.split(b"\0") if field]
    violations = []
    index = 0
    while index < len(fields):
        item = fields[index]
        status, path = item[:2], item[3:]
        if status == "??" and Path(path).parts[0] in ALLOWED_UNTRACKED_ROOTS:
            index += 1
            continue
        violations.append(item)
        index += 2 if "R" in status or "C" in status else 1
    return violations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    violations = worktree_violations(args.project_root.resolve())
    if violations:
        for violation in violations:
            print(violation)
        raise SystemExit("Formal experiments require committed source and config files")


if __name__ == "__main__":
    main()
