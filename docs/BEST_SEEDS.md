# Displayed manuscript seeds

These are the paired seeds used in the current Figure 4–6 panels.
Use the same list for checkpoint release and code defaults.

## Figure 4

| Panel | Dataset | Seeds |
| --- | --- | --- |
| 4b / 4e / 4f (Plain FT, Plain FT+SSR) | WikiData CounterFact | 9311, 9315, 9319, 9325, 9329 |
| 4e | WikiRecent | 9313, 9315, 9319, 9325, 9329 |
| 4e | ZsRE | 9311, 9315, 9317, 9319, 9323 |

Direct Plain FT means on CounterFact: efficacy 9.0%, locality 5.66%.
Plain FT+SSR (cosine): efficacy 13.2%, locality 7.93%.

## Figure 5

| Rank | WikiRecent displayed seeds |
| --- | --- |
| 8 | 17631, 17632, 17634, 17635, 17639 |
| 16 | 17631, 17633, 17637, 17638, 17640 |
| 32 (panels a,b,e,f) | 17631, 17633, 17637, 17638, 17640 |
| 64 | 17632, 17633, 17638, 17639, 17640 |

Rank-32 endpoints: efficacy 51.4% → 63.2% (+11.80 pp);
locality 8.04% → 9.83% (+1.79 pp).

## Figure 6

| Cohort | Seeds |
| --- | --- |
| VOC 10–1 primary | 12401, 12405, 12406, 12409, 12410 |
| CUB adapter rank 8 | 8411, 8413, 8415, 8417, 8423 |
| CUB adapter rank 16 | 8411, 8415, 8419, 8421, 8429 |
| CUB adapter rank 32 | 8411, 8415, 8421, 8423, 8425 |
| CUB geometry rank 32 | 17837, 17839, 17841, 17843, 17847 |

VOC all-mIoU AUC: 41.16% → 42.40% (+1.24).
CUB rank-32 AA: 78.64% → 80.33% (+1.69); AF 3.62% → 2.22% (reduction 1.40).
Geometry vs KD: AA +1.67 [1.21, 2.12]; AF reduction +1.61 [1.07, 2.14].

## Representative dual-positive cases (SSR advantage)

These are the strongest dual-positive pairs inside the displayed or SI-selected lists. They are for checkpoint highlighting, not a second set of cohort statistics.

| Setting | Seeds | Why they stand out |
| --- | --- | --- |
| WikiRecent rank 32 | 17640 | +27.0 efficacy and +4.48 locality |
| VOC primary | 12405, 12409, 12410 | largest final all/old mIoU gains |
| VOC replication (SI) | 13407, 13406, 13405, 13401, 13402 | dual-positive; 13401 is Fig. 6c |
| CUB rank 32 | 8425, 8415 | largest AA+AF among displayed n=5 |
| WikiCounterFact (SI) | rank-wise dual-positive n=3–5 | both endpoints above zero |
