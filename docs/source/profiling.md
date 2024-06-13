# Computational Profile

To establish an overview of the computational profile of ACES, a collection of various common tasks was queried on the MIMIC-IV dataset in MEDS format.

The MIMIC-IV MEDS schema has approximately 50,000 patients per shard with an average of approximately 80,500,000 total event rows per shard over five shards.

All tests were executed on a Linux server with 36 cores and 340 GBs of RAM available. A single MEDS shard was used, which provides a bounded computational overview of ACES. For instance, if one shard costs $M$ memory and $T$ time, then $N$ shards may be executed in parallel with $N*M$ memory and $T$ time, or in series with $M$ memory and $T*N$ time.

| Task                                                                                                                                                                                  | # Patients | # Samples | Total Time (secs) | Max Memory (MiBs) |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- | --------- | ----------------- | ----------------- |
| [First 24h in-hospital mortality](https://github.com/mmcdermott/PIE_MD/blob/e94189864080f957fcf2b7416c1dde401dfe4c15/tasks/MIMIC-IV/mortality/in_hospital/first_24h.yaml)             | 20,971     | 58,823    | 363.09            | 106,367.14        |
| [First 48h in-hospital mortality](https://github.com/mmcdermott/PIE_MD/blob/e94189864080f957fcf2b7416c1dde401dfe4c15/tasks/MIMIC-IV/mortality/in_hospital/first_48h.yaml)             | 18,847     | 60,471    | 364.62            | 108,913.95        |
| [First 24h in-ICU mortality](https://github.com/mmcdermott/PIE_MD/blob/e94189864080f957fcf2b7416c1dde401dfe4c15/tasks/MIMIC-IV/mortality/in_icu/first_24h.yaml)                       | 4,768      | 7,156     | 216.81            | 39,594.37         |
| [First 48h in-ICU mortality](https://github.com/mmcdermott/PIE_MD/blob/e94189864080f957fcf2b7416c1dde401dfe4c15/tasks/MIMIC-IV/mortality/in_icu/first_48h.yaml)                       | 4,093      | 7,112     | 217.98            | 39,451.86         |
| [30d post-hospital-discharge mortality](https://github.com/mmcdermott/PIE_MD/blob/e94189864080f957fcf2b7416c1dde401dfe4c15/tasks/MIMIC-IV/mortality/post_hospital_discharge/30d.yaml) | 28,416     | 68,547    | 182.91            | 30,434.86         |
| [30d re-admission](https://github.com/mmcdermott/PIE_MD/blob/e94189864080f957fcf2b7416c1dde401dfe4c15/tasks/MIMIC-IV/readmission/30d.yaml)                                            | 18,908     | 464,821   | 367.41            | 106,064.04        |
