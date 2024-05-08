"""This module contains functions for loading and parsing the configuration file and subsequently building a
tree structure from the configuration."""

from datetime import timedelta
from typing import Any

import polars as pl
import ruamel.yaml
from bigtree import Node
from pytimeparse import parse


def load_config(config_path: str) -> dict[str, Any]:
    """Load a configuration file from the given path and return it as a dict.

    Args:
        config_path (str): The path to the configuration file.
    """
    yaml = ruamel.yaml.YAML(typ="safe", pure=True)
    with open(config_path) as file:
        return yaml.load(file)


def get_config(cfg: dict, key: str, default: Any) -> Any:
    if key not in cfg or cfg[key] is None:
        return default
    return cfg[key]


def parse_timedelta(time_str: str) -> timedelta:
    """Parse a time string and return a timedelta object.

    Using time expression parser: https://github.com/wroberts/pytimeparse

    Args:
        time_str: The time string to parse.

    Returns:
        timedelta: The parsed timedelta object.

    Examples:
        >>> parse_timedelta("1 days")
        datetime.timedelta(days=1)
        >>> parse_timedelta("1 day")
        datetime.timedelta(days=1)
        >>> parse_timedelta("1 days 2 hours 3 minutes 4 seconds")
        datetime.timedelta(days=1, seconds=7384)
        >>> parse_timedelta('1 day, 14:20:16')
        datetime.timedelta(days=1, seconds=51616)
        >>> parse_timedelta('365 days')
        datetime.timedelta(days=365)
    """
    if not time_str:
        return timedelta(days=0)

    return timedelta(seconds=parse(time_str))


def get_max_duration(data: pl.DataFrame) -> timedelta:
    """Get the maximum duration for each subject timestamps.

    Args:
        data: The data containing the events for each subject. The timestamps of the events are in the
        column ``"timestamp"`` and the subject IDs are in the column ``"subject_id"``.

    Returns:
        The maximum duration between the latest timestamp and the earliest timestamp over all subjects.

    Examples:
        >>> from datetime import datetime
        >>> data = pl.DataFrame({
        ...     "subject_id": [1, 1, 2, 2],
        ...     "timestamp": [
        ...         datetime(2021, 1, 1), datetime(2021, 1, 2), datetime(2021, 1, 1), datetime(2021, 1, 3)
        ...     ]
        ... })
        >>> get_max_duration(data)
        datetime.timedelta(days=2)
    """

    return (
        data.group_by("subject_id")
        .agg((pl.col("timestamp").max() - pl.col("timestamp").min()).alias("duration"))
        .get_column("duration")
        .max()
    )


def build_tree_from_config(cfg: dict[str, Any]) -> Node:
    """Build a tree structure from the given configuration. Note: the parse_timedelta function handles
    negative durations already if duration is specified with "-".

    Args:
        cfg: The configuration object.

    Returns:
        Node: The root node of the built tree.

    Examples:
        >>> cfg = {
        ...     "windows": {
        ...         "window1": {
        ...             "start": 'event1',
        ...             "duration": "24 hours",
        ...             "offset": None,
        ...             "excludes": [],
        ...             "includes": [],
        ...             "st_inclusive": False,
        ...             "end_inclusive": True,
        ...         },
        ...         "window2": {
        ...             "start": "window1.end",
        ...             "duration": "24 hours",
        ...             "end": None,
        ...             "excludes": [],
        ...             "st_inclusive": False,
        ...             "end_inclusive": True,
        ...         },
        ...     }
        ... }
        >>> build_tree_from_config(cfg) # doctest: +NORMALIZE_WHITESPACE
        Node(
            /window1,
            constraints={},
            endpoint_expr=(False, datetime.timedelta(days=1), True, datetime.timedelta(0))
        )
    """
    nodes = {}
    windows = [name for name, _ in cfg["windows"].items()]
    for window_name, window_info in cfg["windows"].items():
        defined_keys = [
            key for key in ["start", "end", "duration"] if get_config(window_info, key, None) is not None
        ]
        if len(defined_keys) != 2:
            error_keys = [f"{key}: {window_info[key]}" for key in defined_keys]

            error_lines = [
                f"Invalid window specification for '{window_name}'",
                (
                    "Must specify non-None values for exactly two fields of ['start', 'end', 'duration'], "
                    "and if 'start' is not one of those two, then 'end' must be specified. Got:"
                ),
                ", ".join(error_keys),
            ]
            raise ValueError("\n".join(error_lines))

        node = Node(window_name)

        # set node end_event
        duration = get_config(window_info, "duration", None)
        match duration:
            case timedelta():
                end_event = window_info["duration"]
            case str():
                end_event = parse_timedelta(window_info["duration"])
            case False | None:
                end_event = f"is_{window_info['end']}"
            case _:
                raise ValueError(f"Invalid duration in '{window_name}': {window_info['duration']}")

        # set node st_inclusive and end_inclusive
        st_inclusive = get_config(window_info, "st_inclusive", False)
        end_inclusive = get_config(window_info, "end_inclusive", False)

        # set node offset
        offset = get_config(window_info, "offset", None)
        offset = parse_timedelta(offset)

        node.endpoint_expr = (st_inclusive, end_event, end_inclusive, offset)

        # set node exclude constraints
        constraints = {}
        for each_exclusion in get_config(window_info, "excludes", []):
            if each_exclusion["predicate"]:
                constraints[f"is_{each_exclusion['predicate']}"] = (None, 0)

        # set node include constraints, using min and max values and None if not specified
        for each_inclusion in get_config(window_info, "includes", []):
            if each_inclusion["predicate"]:
                constraints[f"is_{each_inclusion['predicate']}"] = (
                    (
                        int(each_inclusion["min"])
                        if "min" in each_inclusion and each_inclusion["min"] is not None
                        else None
                    ),
                    (
                        int(each_inclusion["max"])
                        if "max" in each_inclusion and each_inclusion["max"] is not None
                        else None
                    ),
                )

        node.constraints = constraints

        # search for the parent node in tree
        if get_config(window_info, "start", None):
            root_name = window_info["start"].split(".")[0]
            node_root = next(
                (substring for substring in windows if substring == root_name),
                None,
            )
        elif get_config(window_info, "end", None):
            root_name = window_info["end"].split(".")[0]
            node_root = next(
                (substring for substring in windows if substring == root_name),
                None,
            )

        if node_root:
            node.parent = nodes[node_root]

        nodes[window_name] = node

    for node in nodes.values():
        if node.parent:
            node.parent = nodes[node.parent.name]

    root = next(iter(nodes.values())).root

    return root
