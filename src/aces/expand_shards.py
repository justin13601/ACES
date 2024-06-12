#!/usr/bin/env python

import sys


def expand_shards(*shards: str) -> str:
    """This function expands a set of shard prefixes and number of shards into a list of all shards.

    This can be useful with Hydra applications where you wish to expand a list of options for the sweeper to
    sweep over but can't use an OmegaConf resolver as those are evaluated after the sweep has been
    initialized.

    Args:
        shards: A list of shard prefixes and number of shards to expand.

    Returns: A comma-separated list of all shards, expanded to the specified number.

    Examples:
        >>> expand_shards("train/4", "val/IID/1", "val/prospective/1")
        'train/0,train/1,train/2,train/3,val/IID/0,val/prospective/0'
    """

    result = []
    for arg in shards:
        prefix = arg[: arg.rfind("/")]
        num = int(arg[arg.rfind("/") + 1 :])
        result.extend(f"{prefix}/{i}" for i in range(num))

    return ",".join(result)


def main():
    print(expand_shards(*sys.argv[1:]))


if __name__ == "__main__":
    main()
