#!/usr/bin/env python3
"""Validate a KD-only teacher manifest, checkpoints, and result identity."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ssr_utils.teacher_trajectory import validate_teacher_trajectory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--expected-checkpoints", type=int, required=True)
    parser.add_argument("--result-record", type=Path, required=True)
    args = parser.parse_args()
    record = json.loads(args.result_record.read_text(encoding="utf-8"))
    report = validate_teacher_trajectory(
        args.root.resolve(),
        dataset=args.dataset,
        model=args.model,
        seed=args.seed,
        expected_checkpoints=args.expected_checkpoints,
        expected_trajectory_hash=record.get("teacher_trajectory_hash"),
    )
    print(json.dumps({"status": "PASS", **report}, indent=2))


if __name__ == "__main__":
    main()
