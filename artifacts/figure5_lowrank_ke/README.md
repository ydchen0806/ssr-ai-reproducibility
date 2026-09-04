# Figure 5 low-rank knowledge-editing source data

These compact tables are the manuscript-facing exports for the low-rank
Qwen2.5-7B/ZsRE experiments. All intervals are paired, seed-level bootstrap
95% confidence intervals. No rows were removed after outcome inspection.

- `independent_100_seed_level.csv`: the ten paired rank-8 streams underlying
  the 100-edit acquisition/locality panel.
- `rank_curves.csv`: checkpoint summaries for the independent 100-edit rank
  cohort and the separately confirmed rank-32 250-edit cohort.
- `rank250_and_replication_summary.csv`: the 250-edit rank comparison and the
  three independent rank-8 acquisition confirmations.
- `mechanism_250_seed_level.csv`: all checkpoint-level rows from the independent
  mechanism cohort, including efficacy, locality, history retention, effective
  rank, center-surround contrast and kernel alignment.

The independent cohorts intentionally use different edit-order seeds. Thus the
three rank-8 rows are replications across edit order and edit horizon, not three
views of one training run. Dataset/model artifacts are not redistributed.
