from argparse import Namespace

from scripts.run_cub_segmentation_kd_shard import METHODS, selected_jobs
from scripts.run_meeting_segmentation_ablation import screen_jobs


def test_locked_cub_confirmation_is_complete_and_evenly_sharded(tmp_path):
    seeds = tuple(range(9201, 9240, 2))
    spec = screen_jobs(tmp_path)[1].spec
    shards = []
    complete = None
    for shard_index in range(2):
        args = Namespace(
            result_root=tmp_path,
            shard_index=shard_index,
            shard_count=2,
        )
        all_jobs, assigned = selected_jobs(args, spec, seeds)
        complete = all_jobs
        shards.append(assigned)

    assert complete is not None
    assert len(complete) == len(seeds) * len(METHODS) == 40
    assert {job.method for job in complete} == set(METHODS)
    assert len(shards[0]) == len(shards[1]) == 20
    assert {job.output for job in shards[0]}.isdisjoint(
        {job.output for job in shards[1]}
    )
    assert {job.output for jobs in shards for job in jobs} == {
        job.output for job in complete
    }
