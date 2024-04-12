"""This module contains functions for generating predicate columns for event sequences."""

import numpy as np
import polars as pl
from loguru import logger


def has_event_type(type_str: str) -> pl.Expr:
    """Check if the event type contains the specified string.

    Args:
        type_str (str): The string to search for in the event type.

    Returns:
        pl.Expr: A Polars expression representing the check for the event type.

    >>> import polars as pl
    >>> has_event_type("A").evaluate(pl.DataFrame({"event_type": ["A", "B", "C"]}))
    shape: (3,)
    Series: 'event_type' [str]
    [
        "A"
        "A&B"
        "C"
    ]
    0    True
    1    True
    2   False
    dtype: bool
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
    for predicate_name, predicate_info in cfg.predicates.items():
        if predicate_name == "any":
            continue

        if predicate_info.system == "boolean":
            boolean_cols.append(f"is_{predicate_name}")
        elif predicate_info.system == "count":
            count_cols.append(f"is_{predicate_name}")
        else:
            raise ValueError(
                f"Invalid predicate system {predicate_info.system} for {predicate_name}."
            )

        if "value" in predicate_info:
            if isinstance(predicate_info["value"], list):
                ESD = ESD.with_columns(
                    pl.when(
                        (
                            pl.col(predicate_info.column)
                            >= (
                                float(predicate_info["value"][0]["min"] or -np.inf)
                                if "min" in predicate_info["value"][0]
                                else float(-np.inf)
                            )
                        )
                        & (
                            pl.col(predicate_info.column)
                            <= (
                                float(predicate_info["value"][0]["max"] or np.inf)
                                if "max" in predicate_info["value"][0]
                                else float(np.inf)
                            )
                        )
                    )
                    .then(1)
                    .otherwise(0)
                    .alias(f"is_{predicate_name}")
                    .cast(pl.Int32)
                )
                logger.debug(f"Added predicate column is_{predicate_name}.")
            else:
                if predicate_info.column == "event_type":
                    ESD = ESD.with_columns(
                        has_event_type(predicate_info["value"])
                        .alias(f"is_{predicate_name}")
                        .cast(pl.Int32)
                    )
                else:
                    ESD = ESD.with_columns(
                        pl.when(
                            pl.col(predicate_info.column) == predicate_info["value"]
                        )
                        .then(1)
                        .otherwise(0)
                        .alias(f"is_{predicate_name}")
                        .cast(pl.Int32)
                    )
                logger.debug(f"Added predicate column is_{predicate_name}.")
        elif "type" in predicate_info:
            if predicate_info.type == "ANY":
                any_expr = pl.col(f"is_{predicate_info.predicates[0]}")
                for predicate in predicate_info.predicates[1:]:
                    any_expr = any_expr | pl.col(f"is_{predicate}")
                ESD = ESD.with_columns(any_expr.alias(f"is_{predicate_name}"))
                logger.debug(f"Added predicate column is_{predicate_name}.")
            elif predicate_info.type == "ALL":
                all_expr = pl.col(f"is_{predicate_info.predicates[0]}")
                for predicate in predicate_info.predicates[1:]:
                    all_expr = all_expr & pl.col(f"is_{predicate}")
                ESD = ESD.with_columns(all_expr.alias(f"is_{predicate_name}"))
                logger.debug(f"Added predicate column is_{predicate_name}.")
            else:
                raise ValueError(
                    f"Invalid predicate type {predicate_info.type} for {predicate_name}."
                )
        else:
            raise ValueError(f"Invalid predicate specification for {predicate_name}.")

    ESD = ESD.with_columns(
        pl.when(pl.col("event_type").is_not_null()).then(1).otherwise(0).alias("is_any")
    )

    ESD = ESD.groupby(["subject_id", "timestamp"]).agg(
        *[
            pl.col(c).sum().alias(f"{c}_count").cast(pl.Int32)
            for c in ESD.columns
            if c.startswith("is_")
        ],
        *[
            pl.col(c).max().alias(f"{c}_boolean").cast(pl.Int32)
            for c in ESD.columns
            if c.startswith("is_")
        ],
    )

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
