"""This module contains functions for generating predicate columns for event sequences."""

import numpy as np
import polars as pl
from loguru import logger
from functools import reduce


def has_event_type(type_str: str) -> pl.Expr:
    """Check if the event type contains the specified string.

    Args:
        type_str (str): The string to search for in the event type.

    Returns:
        pl.Expr: A Polars expression representing the check for the event type.

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


def generate_predicate_columns(cfg: dict, ESD: pl.DataFrame) -> pl.DataFrame:
    """Generate predicate columns based on the configuration.

    Args:
        cfg: The configuration object containing the predicate information.
        ESD: The Polars DataFrame to add the predicate columns to.

    Returns:
        ESD: The Polars DataFrame with the added predicate columns.

    Raises:
        ValueError: If an invalid predicate type is specified in the configuration.

    Example:
        >>> cfg = ...
        >>> ESD = ...
        >>> generate_predicate_columns(cfg, ESD)
        Added predicate column is_predicate1.
        Added predicate column is_predicate2.
        ...
        Added predicate column is_any.
        <Polars DataFrame>
        ...
    """
    boolean_cols = []
    count_cols = []
    for predicate_name, predicate_info in cfg["predicates"].items():
        if predicate_name == "any":
            continue

        predicate_col = f"is_{predicate_name}"

        # 1 or 0 for boolean predicates or count for count predicates
        match predicate_info["system"]:
            case "boolean":
                boolean_cols.append(predicate_col)
            case "count":
                count_cols.append(predicate_col)
            case _ as invalid:
                raise ValueError(f"Invalid predicate system {invalid} for {predicate_name}.")

        value = predicate_info.get("value", None)
        predicate_type = predicate_info.get("type", None)
        if not (value or predicate_type):
            raise ValueError(
                f"Invalid predicate specification for {predicate_name}: Must specify value or type. "
                f"Got value={value} and type={predicate_type}."
            )
        elif value and predicate_type:
            raise ValueError(
                f"Invalid predicate specification for {predicate_name}: Can't specify both value and type. "
                f"Got value={value} and type={predicate_type}."
            )

        match value:
            case dict() if {"min", "max"}.issuperset(value.keys()):
                # if value of predicate is specified with min and max
                predicate_col = (
                    pl.when(
                        (pl.col(predicate_info.column) >= float(value.get("min", -float('inf')))) &
                        (pl.col(predicate_info.column) <= float(value.get("max", float('inf'))))
                    )
                    .then(1)
                    .otherwise(0)
                )
            case str() if predicate_info["column"] == "event_type":
                predicate_col = has_event_type(value)
            case str() if value:
                predicate_col = pl.when(pl.col(predicate_info.column) == value).then(1).otherwise(0)
            case None | "":
                pass
            case _:
                raise ValueError(f"Invalid value {value} for {predicate_name}.")

        # complex predicates specified with AND/ALL or OR/ANY
        subpredicate_cols = [pl.col(f"is_{predicate}") for predicate in predicate_info["predicates"]]
        match predicate_type:
            case "ANY":
                predicate_col = reduce(lambda x, y: x | y, subpredicate_cols)
            case "ALL":
                predicate_col = reduce(lambda x, y: x & y, subpredicate_cols)
            case None | "":
                pass
            case _:
                raise ValueError(f"Invalid predicate type {predicate_type} for {predicate_name}.")

        ESD = ESD.with_columns(predicate_col.alias(predicate_col).cast(pl.Int32))
        logger.debug(f"Added predicate column {predicate_col}")

    # add a column for any predicate
    ESD = ESD.with_columns(pl.when(pl.col("event_type").is_not_null()).then(1).otherwise(0).alias("is_any"))

    # aggregate for unique subject_id and timestamp
    # TODO(justin): this is not the cleanest:
    ESD = ESD.groupby(["subject_id", "timestamp"]).agg(
        *[pl.col(c).sum().alias(f"{c}_count").cast(pl.Int32) for c in ESD.columns if c.startswith("is_")],
        *[pl.col(c).max().alias(f"{c}_boolean").cast(pl.Int32) for c in ESD.columns if c.startswith("is_")],
    )

    # select and rename columns
    ESD = ESD.select(
        "subject_id",
        "timestamp",
        *[
            pl.col(c).alias(c.replace("_boolean", ""))
            for c in ESD.columns
            if c.replace("_boolean", "") in boolean_cols
        ],
        *[
            pl.col(c).alias(c.replace("_count", ""))
            for c in ESD.columns
            if c.replace("_count", "") in count_cols
        ],
        pl.col("is_any_boolean").alias("is_any"),
    )

    ESD = ESD.sort(by=["subject_id", "timestamp"])

    return ESD
