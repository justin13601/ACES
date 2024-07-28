#!/usr/bin/env python

import glob
import os
import re
import sys


def expand_shards(*shards: str) -> str:
    """This function expands a set of shard prefixes and number of shards into a list of all shards or expands
    a directory into a list of all files within it.

    This can be useful with Hydra applications where you wish to expand a list of options for the sweeper to
    sweep over but can't use an OmegaConf resolver as those are evaluated after the sweep has been
    initialized.

    Args:
        shards: A list of shard prefixes and number of shards to expand, or a directory to list all files.

    Returns: A comma-separated list of all shards, expanded to the specified number, or all files in the
        directory.

    Examples:
        >>> import polars as pl
        >>> import tempfile
        >>> from pathlib import Path

        >>> expand_shards("train/4", "val/IID/1", "val/prospective/1")
        'train/0,train/1,train/2,train/3,val/IID/0,val/prospective/0'
        >>> expand_shards("data/data_4", "data/test_4")
        'data/data_0,data/data_1,data/data_2,data/data_3,data/test_0,data/test_1,data/test_2,data/test_3'

        >>> parquet_data = pl.DataFrame({
        ...     "patient_id": [1, 1, 1, 2, 3],
        ...     "timestamp": ["1/1/1989 00:00", "1/1/1989 01:00", "1/1/1989 01:00", "1/1/1989 02:00", None],
        ...     "code": ['admission', 'discharge', 'discharge', 'admission', "gender"],
        ... }).with_columns(pl.col("timestamp").str.strptime(pl.Datetime, format="%m/%d/%Y %H:%M"))

        >>> with tempfile.TemporaryDirectory() as tmpdirname:
        ...     for i in range(4):
        ...         with tempfile.NamedTemporaryFile(mode="w", suffix=".parquet") as f:
        ...             data_path = Path(tmpdirname + f"/file_{i}")
        ...             parquet_data.write_parquet(data_path)
        ...     result = expand_shards(tmpdirname)
        ...     ','.join(sorted(os.path.basename(f) for f in result.split(',')))
        'file_0,file_1,file_2,file_3'
    """

    result = []
    for arg in shards:
        if os.path.isdir(arg):
            # If the argument is a directory, list all files in the directory
            files = glob.glob(os.path.join(arg, "*"))
            result.extend(files)
        else:
            # Otherwise, treat it as a shard prefix and number of shards
            match = re.match(r"(.+)([/_])(\d+)$", arg)
            if match:
                prefix = match.group(1)
                delimiter = match.group(2)
                num = int(match.group(3))
                result.extend(f"{prefix}{delimiter}{i}" for i in range(num))
            else:
                raise ValueError(f"Invalid shard format: {arg}")

    return ",".join(result)


def main():
    print(expand_shards(*sys.argv[1:]))


if __name__ == "__main__":
    main()
