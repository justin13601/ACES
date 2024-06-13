# Computational Profile

To establish an overview of the computational profile of ACES, a collection of various common tasks was queried on the MIMIC-IV dataset in MEDS format.

The MIMIC-IV MEDS schema has approximately 50,000 patients per shard with an average of approximately 80,500,000 total event rows per shard over five shards.

All tests were executed on a single MEDS shard, which provides a bounded computational overview of ACES. For instance, if one shard costs $M$ memory and $T$ time, then $N$ shards may be executed in parallel with $N*M$ memory and $T$ time, or in series with $M$ memory and $T*N$ time.

| Task                                  | # Patients | # Samples | Total Time (secs) | Max Mem (MB) |
| ------------------------------------- | ---------- | --------- | ----------------- | ------------ |
| First 24h in-hospital mortality       | 20,971     | 58,823    | 363.09            | -            |
| First 48h in-hospital mortality       | 18,847     | 60,471    | 364.62            | -            |
| First 24h in-ICU mortality            | 4,768      | 7,156     | 216.81            | -            |
| First 48h in-ICU mortality            | 4,093      | 7,112     | 217.98            | -            |
| 30d post-hospital-discharge mortality | 28,416     | 68,547    | 182.91            | -            |
| 30d re-admission                      | 18,908     | 464,821   | 367.41            | -            |
