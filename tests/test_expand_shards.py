"""Tests the extract_shards CLI process."""

import tempfile
from pathlib import Path

from .utils import run_command


def test_e2e():
    es_stderr, es_stdout = run_command("expand_shards train/3 tuning/1", {}, "expand_shards")
    assert es_stdout == "train/0,train/1,train/2,tuning/0\n", (
        f"Expected 'train/0,train/1,train/2,tuning/0' but got '{es_stdout}'"
    )

    with tempfile.TemporaryDirectory() as d:
        data_dir = Path(d) / "sample_data"

        want_shards = ["train/0", "train/1", "train_2", "tuning/0/1"]
        for shard in want_shards:
            shard_fp = data_dir / f"{shard}.parquet"
            shard_fp.mkdir(parents=True)
            shard_fp.touch()

        es_stderr, es_stdout = run_command(f"expand_shards {data_dir}", {}, "expand_shards")
        got_shards = es_stdout.strip().split(",")
        assert sorted(got_shards) == sorted(want_shards), f"Expected {want_shards} but got {got_shards}"
