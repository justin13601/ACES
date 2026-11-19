"""PROTOTYPE benchmark: end-to-end ACES extraction (get_predicates_df + query) over every shard of a dataset.

usage: python benchmarks/native/bench_extraction.py <mode> <MEDS data dir> <task.yaml> <labels.parquet>

modes: dev-seq (ACES, one shard at a time), dev-proc8 (ACES, 8 spawned processes), native-thr8 (native plain
predicates, 8 threads). Writes every label (with a _shard column) and <labels.parquet>.time.txt with the wall
time; compare the label files of two modes after sorting their rows.
"""

import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import polars as pl


def extract(cfg_path: str, shard: str, use_native: bool = False) -> pl.DataFrame:
    from omegaconf import DictConfig

    from aces import config, native, predicates, query

    if use_native:
        predicates.generate_plain_predicates_from_meds = native.plain_predicates
    task = config.TaskExtractorConfig.load(config_path=Path(cfg_path))
    frame = predicates.get_predicates_df(task, DictConfig({"path": shard, "standard": "meds"}))
    out = query.query(task, frame)
    return out.with_columns(pl.lit(f"{Path(shard).parent.name}/{Path(shard).name}").alias("_shard"))


def main():
    mode, data, cfg, out_path = sys.argv[1], Path(sys.argv[2]), sys.argv[3], sys.argv[4]
    logging.disable(logging.INFO)
    shards = [str(s) for s in sorted(data.rglob("*.parquet"))]
    t = time.perf_counter()
    if mode == "dev-seq":
        frames = [extract(cfg, s) for s in shards]
    elif mode == "dev-proc8":
        with ProcessPoolExecutor(8, mp_context=get_context("spawn")) as pool:
            frames = list(pool.map(extract, [cfg] * len(shards), shards))
    elif mode == "native-thr8":
        with ThreadPoolExecutor(8) as pool:
            frames = list(pool.map(lambda s: extract(cfg, s, True), shards))
    else:
        raise SystemExit(f"unknown mode {mode}")
    wall = time.perf_counter() - t
    labels = pl.concat([f for f in frames if f.height], how="diagonal_relaxed")
    labels.write_parquet(out_path)
    line = f"{mode} {Path(cfg).stem}: {len(shards)} shards, {wall:.1f} s, {labels.height:,} labels"
    Path(out_path + ".time.txt").write_text(line + "\n")
    print(line)


if __name__ == "__main__":
    main()
