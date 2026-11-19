"""PROTOTYPE benchmark: ACES's generate_plain_predicates_from_meds vs aces.native.plain_predicates.

usage: python benchmarks/native/bench_predicates.py <MEDS data dir> <shard step> <task.yaml> [task.yaml ...]

For each task and prune setting, checks the outputs are identical on every shard (frame.equals) and prints the
time of ACES (sequential), native (sequential) and native on 8 threads.
"""

import sys
import time
from pathlib import Path

from aces import config, native, predicates

data, step = Path(sys.argv[1]), int(sys.argv[2])
shards = sorted(data.rglob("*.parquet"))[::step]
print(f"{len(shards)} shards")
for cfg_path in map(Path, sys.argv[3:]):
    plain = config.TaskExtractorConfig.load(config_path=cfg_path).plain_predicates
    for prune in (True, False):
        t = time.perf_counter()
        ref = [predicates.generate_plain_predicates_from_meds(s, plain, prune=prune) for s in shards]
        t_ref = time.perf_counter() - t
        t = time.perf_counter()
        got = [native.plain_predicates(s, plain, prune=prune) for s in shards]
        t_seq = time.perf_counter() - t
        t = time.perf_counter()
        par = native.plain_predicates_many(shards, plain, prune=prune, workers=8)
        t_par = time.perf_counter() - t
        same = all(a.equals(b) for a, b in zip(ref, got, strict=True)) and all(
            a.equals(b) for a, b in zip(ref, par, strict=True)
        )
        print(
            f"{cfg_path.stem:34s} prune={prune!s:5s} identical={same} | ACES {t_ref:6.2f} s"
            f" | native {t_seq:5.2f} s (x{t_ref / t_seq:4.1f})"
            f" | native 8 threads {t_par:5.2f} s (x{t_ref / t_par:4.1f})"
        )
