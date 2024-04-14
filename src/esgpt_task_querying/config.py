"""This module contains functions for loading and parsing the configuration file into a dot accessible
dictionary and subsequently building a tree structure from the configuration."""

import re
from datetime import timedelta

import polars as pl
import ruamel.yaml
from bigtree import Node, preorder_iter
from typing import Any

def load_config(config_path: str) -> dict[str, Any]:
    """Load a configuration file from the given path and return it as a dict.

    Args:
        config_path (str): The path to the configuration file.
    """
    yaml = ruamel.yaml.YAML()
    with open(config_path) as file:
        return yaml.load(file)


def parse_timedelta(time_str: str) -> timedelta:
    """Parse a time string and return a timedelta object.

    Args:
        time_str (str): The time string to parse.

    Returns:
        timedelta: The parsed timedelta object.
    """
    if not time_str:
        return timedelta(days=0)

    units = {"days": 0, "hours": 0, "minutes": 0, "seconds": 0}
    pattern = r"(\d+)\s*(seconds|minutes|hours|days)"
    matches = re.findall(pattern, time_str.lower())
    for value, unit in matches:
        units[unit] = int(value)

    return timedelta(
        days=units["days"],
        hours=units["hours"],
        minutes=units["minutes"],
        seconds=units["seconds"],
    )


def get_max_duration(data: pl.DataFrame) -> pl.DataFrame:
    """Get the maximum duration data for each subject by calculating the delta between the min and max
    timestamps."""

    # get the start and end timestamps for each subject
    data = data.groupby("subject_id").agg(
        [
            pl.col("timestamp").min().alias("min_timestamp"),
            pl.col("timestamp").max().alias("max_timestamp"),
        ]
    )

    # calculate the duration for each subject
    data = data.with_columns((pl.col("max_timestamp") - pl.col("min_timestamp")).alias("duration"))

    # get the maximum duration
    max_duration = data["duration"].max()
    return max_duration


def build_tree_from_config(cfg: dict[str, Any]) -> Node:
    """Build a tree structure from the given configuration.

    Args:
        cfg: The configuration object.

    Returns:
        Node: The root node of the built tree.
    """
    nodes = {}
    for window_name, window_info in cfg['windows'].items():
        node = Node(window_name)

        # set node end_event
        match window_info['duration']:
            case timedelta():
                end_event = window_info['duration']
            case str() if "-" in window_info['duration']:
                end_event = -parse_timedelta(window_info['duration'])
            case str():
                end_event = parse_timedelta(window_info['duration'])
            case False | None:
                end_event = f"is_{window_info['end']}"
            case _:
                raise ValueError(f"Invalid duration: {window_info['duration']}")

        # set node st_inclusive and end_inclusive
        st_inclusive = window_info.get("st_inclusive", False)
        end_inclusive = window_info.get("end_inclusive", True)

        # set node offset
        offset = parse_timedelta(window_info["offset"])
        node.endpoint_expr = (st_inclusive, end_event, end_inclusive, offset)

        # set node exclude constraints
        constraints = {}
        for each_exclusion in window_info.get("excludes", []):
            if each_exclusion["predicate"]:
                constraints[f"is_{each_exclusion['predicate']}"] = (None, 0)

        # set node include constraints, using min and max values and None if not specified
        for each_inclusion in window_info.get("includes", []):
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
        if window_info["start"]:
            root_name = window_info["start"].split(".")[0]
            node_root = next(
                (substring for substring in windows if substring == root_name),
                None,
            )
        elif window_info["end"]:
            root_name = window_info["end"].split(".")[0]
            node_root = next(
                (substring for substring in windows if substring == root_name),
                None,
            )

        if node_root:
            node.parent = node_root

        nodes[window_name] = node

    for node in nodes.values():
        if node.parent:
            node.parent = nodes[node.parent]

    root = next(iter(nodes.values())).root

    return root
