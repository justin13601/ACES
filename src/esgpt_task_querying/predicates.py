"""This module contains functions for generating predicate columns for event sequences."""

from functools import reduce
from typing import Any

import polars as pl
from loguru import logger


def get_config(cfg: dict, key: str, default: Any) -> Any:
    if key not in cfg or cfg[key] is None:
        return default
    return cfg[key]


def has_event_type(type_str: str) -> pl.Expr:
    """Check if the event type contains the specified string.

    Args:
        type_str (str): The string to search for in the event type.

    Returns:
        pl.Expr: A Polars expression representing the check for the event type.

    Examples:
    >>> import polars as pl
    >>> data = pl.DataFrame({"event_type": ["A&B&C", "A&B", "C"]})
    >>> data.with_columns(has_event_type("A").alias("has_A"))
    shape: (3, 2)
    ┌────────────┬───────┐
    │ event_type ┆ has_A │
    │ ---        ┆ ---   │
    │ str        ┆ bool  │
    ╞════════════╪═══════╡
    │ A&B&C      ┆ true  │
    │ A&B        ┆ true  │
    │ C          ┆ false │
    └────────────┴───────┘
    """
    has_event_type = pl.col("event_type").cast(pl.Utf8).str.contains(type_str)
    return has_event_type


def generate_simple_predicates(predicate_name: str, predicate_info: dict, df: pl.DataFrame) -> pl.DataFrame:
    """Generate simple predicate columns based on the configuration.

    Args:
        predicate_name: The name of the predicate.
        predicate_info: The information about the predicate.
        df: The Polars DataFrame to add the predicate columns to.

    Returns:
        df: The Polars DataFrame with the added predicate columns.

    Raises:
        ValueError: If an invalid value is specified for the predicate.

    Examples:
    >>> predicate_name = "A"
    >>> predicate_info = {"column": "event_type", "value": "A", "system": "boolean"}
    >>> data = pl.DataFrame(
    ...     {
    ...         "subject_id": [1, 1, 1, 2, 2, 3, 3],
    ...         "timestamp": [1, 1, 3, 1, 2, 1, 3],
    ...         "event_type": ["A", "B", "C", "A", "A&C&D", "B", "C"],
    ...     }
    ... )
    >>> generate_simple_predicates(predicate_name, predicate_info, data)
    shape: (7, 4)
    ┌────────────┬───────────┬────────────┬──────┐
    │ subject_id ┆ timestamp ┆ event_type ┆ is_A │
    │ ---        ┆ ---       ┆ ---        ┆ ---  │
    │ i64        ┆ i64       ┆ str        ┆ i64  │
    ╞════════════╪═══════════╪════════════╪══════╡
    │ 1          ┆ 1         ┆ A          ┆ 1    │
    │ 1          ┆ 1         ┆ B          ┆ 0    │
    │ 1          ┆ 3         ┆ C          ┆ 0    │
    │ 2          ┆ 1         ┆ A          ┆ 1    │
    │ 2          ┆ 2         ┆ A&C&D      ┆ 1    │
    │ 3          ┆ 1         ┆ B          ┆ 0    │
    │ 3          ┆ 3         ┆ C          ┆ 0    │
    └────────────┴───────────┴────────────┴──────┘
    """
    value = predicate_info["value"]
    match value:
        case dict() if {"min", "max"}.issuperset(value.keys()):
            # if value of predicate is specified with min and max
            predicate_col = (
                pl.when(
                    (pl.col(predicate_info["column"]) >= float(get_config(value, "min", float("-inf"))))
                    & (pl.col(predicate_info["column"]) <= float(get_config(value, "max", float("inf"))))
                )
                .then(1)
                .otherwise(0)
            )
        case str() if predicate_info["column"] == "event_type":
            predicate_col = has_event_type(str(predicate_info["value"]))
        case str() if value:
            predicate_col = pl.when(pl.col(predicate_info["column"]) == value).then(1).otherwise(0)
        case _:
            raise ValueError(f"Invalid value '{value}' for '{predicate_name}'.")

    df = df.with_columns(predicate_col.alias(f"is_{predicate_name}").cast(pl.Int64))
    logger.debug(f"Added predicate column 'is_{predicate_name}'.")
    return df


def generate_predicate_columns(cfg: dict, data: list | pl.DataFrame) -> pl.DataFrame:
    """Generate predicate columns based on the configuration.

    Args:
        cfg: The configuration object containing the predicate information.
        ESD: The Polars DataFrame to add the predicate columns to.

    Returns:
        ESD: The Polars DataFrame with the added predicate columns.

    Raises:
        ValueError: If an invalid predicate type is specified in the configuration.

    Examples:
    >>> cfg = {
    ...     "predicates": {
    ...         "A": {"column": "event_type", "value": "A", "system": "boolean"},
    ...         "B": {"column": "event_type", "value": "B", "system": "boolean"},
    ...         "C": {"column": "event_type", "value": "C", "system": "boolean"},
    ...         "D": {"column": "event_type", "value": "D", "system": "boolean"},
    ...         "A_or_B": {"type": "ANY", "predicates": ["A", "B"], "system": "boolean"},
    ...         "A_and_C_and_D": {"type": "ALL", "predicates": ["A", "C", "D"], "system": "boolean"},
    ...         "any": {"type": "special"},
    ...     }
    ... }
    >>> data = pl.DataFrame(
    ...     {
    ...         "subject_id": [1, 1, 1, 2, 2, 3, 3],
    ...         "timestamp": [1, 1, 3, 1, 2, 1, 3],
    ...         "event_type": ["A", "B", "C", "A", "A&C&D", "B", "C"],
    ...     }
    ... )
    >>> generate_predicate_columns(cfg, data)
    shape: (6, 9)
    ┌────────────┬───────────┬──────┬──────┬───┬──────┬───────────┬──────────────────┬────────┐
    │ subject_id ┆ timestamp ┆ is_A ┆ is_B ┆ … ┆ is_D ┆ is_A_or_B ┆ is_A_and_C_and_D ┆ is_any │
    │ ---        ┆ ---       ┆ ---  ┆ ---  ┆   ┆ ---  ┆ ---       ┆ ---              ┆ ---    │
    │ i64        ┆ i64       ┆ i64  ┆ i64  ┆   ┆ i64  ┆ i64       ┆ i64              ┆ i64    │
    ╞════════════╪═══════════╪══════╪══════╪═══╪══════╪═══════════╪══════════════════╪════════╡
    │ 1          ┆ 1         ┆ 1    ┆ 1    ┆ … ┆ 0    ┆ 1         ┆ 0                ┆ 1      │
    │ 1          ┆ 3         ┆ 0    ┆ 0    ┆ … ┆ 0    ┆ 0         ┆ 0                ┆ 1      │
    │ 2          ┆ 1         ┆ 1    ┆ 0    ┆ … ┆ 0    ┆ 1         ┆ 0                ┆ 1      │
    │ 2          ┆ 2         ┆ 1    ┆ 0    ┆ … ┆ 1    ┆ 1         ┆ 1                ┆ 1      │
    │ 3          ┆ 1         ┆ 0    ┆ 1    ┆ … ┆ 0    ┆ 1         ┆ 0                ┆ 1      │
    │ 3          ┆ 3         ┆ 0    ┆ 0    ┆ … ┆ 0    ┆ 0         ┆ 0                ┆ 1      │
    └────────────┴───────────┴──────┴──────┴───┴──────┴───────────┴──────────────────┴────────┘
    """
    logger.debug("Generating predicate columns...")
    boolean_cols = []
    count_cols = []
    simple_predicates = []
    complex_predicates = []
    for predicate_name, predicate_info in cfg["predicates"].items():
        if predicate_name == "any":
            continue

        # 1 or 0 for boolean predicates or count for count predicates
        match predicate_info["system"]:
            case "boolean":
                boolean_cols.append(f"is_{predicate_name}")
            case "count":
                count_cols.append(f"is_{predicate_name}")
            case _ as invalid:
                raise ValueError(f"Invalid predicate system '{invalid}' for '{predicate_name}'.")

        value = get_config(predicate_info, "value", None)
        predicate_type = get_config(predicate_info, "type", None)
        if not (value or predicate_type):
            raise ValueError(
                f"Invalid predicate specification for '{predicate_name}': must specify value or type. "
                f"Got value={value} and type={predicate_type}."
            )
        elif value and predicate_type:
            raise ValueError(
                f"Invalid predicate specification for '{predicate_name}': can't specify both value and type. "
                f"Got value={value} and type={predicate_type}."
            )
        elif value:
            simple_predicates.append(predicate_name)
        elif predicate_type:
            complex_predicates.append(predicate_name)

    match data:
        case pl.DataFrame():
            # populate event_id column
            data = data.with_row_index("event_id").select(
                *data.columns,
                pl.first("event_id").over(["subject_id", "timestamp"]).rank("dense") - 1,
            )
            events = data.select(
                "subject_id",
                "timestamp",
                "event_id",
                "event_type",
            )
            measurements = data.select(
                "event_id",
                *[pl.col(c) for c in data.columns if c not in events.columns],
            )

            events = events.group_by(["subject_id", "timestamp"]).agg(
                pl.col("event_id").first(),
                *[pl.col("event_type").flatten()],
            )
            events = events.with_columns(event_type=events["event_type"].list.join("&"))

            data = [events, measurements]

    for predicate_name in simple_predicates:
        if cfg["predicates"][predicate_name]["column"] == "event_type":
            data[0] = generate_simple_predicates(predicate_name, cfg["predicates"][predicate_name], data[0])
        else:
            data[1] = generate_simple_predicates(predicate_name, cfg["predicates"][predicate_name], data[1])

    # aggregate measurements (data[1]) by summing columns that are in count_cols, and taking the max for
    # columns in boolean_cols
    data[1] = (
        data[1]
        .group_by(["event_id"])
        .agg(
            *[pl.col(c).sum().cast(pl.Int64) for c in data[1].columns if c in count_cols],
            *[pl.col(c).max().cast(pl.Int64) for c in data[1].columns if c in boolean_cols],
        )
    )

    data = data[0].join(data[1], on="event_id", how="left")
    data = data.select(
        "subject_id",
        "timestamp",
        *[pl.col(c) for c in data.columns if c.startswith("is_")],
    )

    # if complex predicates specified with ALL (AND) or ANY (OR)
    for predicate_name in complex_predicates:
        predicate_info = cfg["predicates"][predicate_name]
        predicate_type = predicate_info["type"]
        sub_predicate_cols = [pl.col(f"is_{predicate}") for predicate in predicate_info["predicates"]]
        match predicate_type:
            case "ANY":
                predicate_expr = reduce(lambda x, y: x | y, sub_predicate_cols)
            case "ALL":
                predicate_expr = reduce(lambda x, y: x & y, sub_predicate_cols)
            case _:
                raise ValueError(f"Invalid predicate type '{predicate_type}' for '{predicate_name}'.")

        data = data.with_columns(predicate_expr.alias(f"is_{predicate_name}").cast(pl.Int64))
        logger.debug(f"Added predicate column 'is_{predicate_name}'.")

    # add a column of 1s representing any predicate
    if "any" in cfg["predicates"]:
        data = data.with_columns(pl.lit(1).alias("is_any").cast(pl.Int64))
        logger.debug("Added predicate column 'is_any'.")

    data = data.sort(by=["subject_id", "timestamp"])

    return data
