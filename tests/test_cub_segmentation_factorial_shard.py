from __future__ import annotations

from argparse import Namespace

import scripts.run_cub_segmentation_factorial_shard as factorial_shard
from scripts.run_meeting_segmentation_ablation import (
    FACTORIAL_CONFIRMATION_SEEDS,
    METHODS,
    PROTOCOL_FACTORIAL,
    SPECS,
)


def test_factorial_confirmation_is_complete_and_evenly_partitioned_over_four_nodes(
    tmp_path,
):
    shards = []
    complete = None
    for shard_index in range(4):
        args = Namespace(
            result_root=tmp_path,
            shard_index=shard_index,
            shard_count=4,
        )
        all_jobs, assigned = factorial_shard.selected_jobs(
            args,
            SPECS[0],
            FACTORIAL_CONFIRMATION_SEEDS,
        )
        complete = all_jobs
        shards.append(assigned)

    assert complete is not None
    assert len(complete) == len(FACTORIAL_CONFIRMATION_SEEDS) * len(METHODS) == 80
    assert {job.method for job in complete} == set(METHODS)
    assert [len(shard) for shard in shards] == [20, 20, 20, 20]
    assert [len({job.seed for job in shard}) for shard in shards] == [5, 5, 5, 5]
    assert len({job.output for shard in shards for job in shard}) == 80
    assert {job.output for shard in shards for job in shard} == {
        job.output for job in complete
    }


def test_summary_only_never_rewrites_a_rank_zero_shard_manifest(monkeypatch, tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    (data_root / "segmentations.tgz").write_bytes(b"source")
    cache = tmp_path / "cache.pt"
    cache.write_bytes(b"cache")
    lock = tmp_path / "SELECTION_LOCK.json"
    lock.write_text('{"lock_sha256": "selection-lock"}', encoding="utf-8")
    args = Namespace(
        result_root=tmp_path / "results",
        selection_lock=lock,
        data_root=data_root,
        segmentation_cache=cache,
        shard_index=0,
        shard_count=1,
        gpus=["0"],
        jobs_per_gpu=1,
        workers=0,
        seg_batch_size=24,
        python="python3",
        summary_only=True,
    )
    writes = []
    summaries = []

    monkeypatch.setattr(factorial_shard, "parse_args", lambda: args)
    monkeypatch.setattr(factorial_shard, "sha256_file", lambda path: "a" * 64)
    monkeypatch.setattr(factorial_shard, "current_git_commit", lambda: "commit-a")
    monkeypatch.setattr(
        factorial_shard,
        "validate_selection_lock",
        lambda path, parsed_args, commit: SPECS[0],
    )
    monkeypatch.setattr(
        factorial_shard,
        "write_confirmation_summary",
        lambda jobs, result_root, parsed_args, commit: summaries.append(jobs),
    )
    monkeypatch.setattr(
        factorial_shard,
        "write_json_atomic",
        lambda path, payload: writes.append((path, payload)),
    )
    monkeypatch.setattr(
        factorial_shard,
        "run_jobs",
        lambda *unused: (_ for _ in ()).throw(AssertionError("must not run jobs")),
    )

    factorial_shard.main()

    assert len(summaries) == 1
    assert len(summaries[0]) == 80
    assert writes == []
    assert args.protocol == PROTOCOL_FACTORIAL
