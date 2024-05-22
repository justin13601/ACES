"""This module contains functions for generating predicate columns for event sequences."""


import polars as pl
from loguru import logger

from .config import TaskExtractorConfig
from .types import ANY_EVENT_COLUMN


def generate_predicates_df(cfg: TaskExtractorConfig, data: pl.DataFrame, standard: str) -> pl.DataFrame:
    """Generate predicate columns based on the configuration.

    Args:
        predicates: The object containing the predicates information.
        df_data: The Polars DataFrame containing the original external data.
        standard: The data standard, either 'MEDS' or 'ESGPT'.

    Returns:
        predicates_df: The Polars DataFrame with the added predicate columns.

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
    predicate_cols = []

    # plain predicates
    for name, code in cfg.plain_predicates.items():
        match standard:
            case "MEDS":
                data = data.with_columns(code.MEDS_eval_expr().alias(name))
            case "ESGPT":
                data = data.with_columns(code.ESGPT_eval_expr().cast(pl.UInt16).alias(name))
        logger.debug(f"Added predicate column '{name}'.")
        predicate_cols.append(name)

    # derived predicates
    for name, code in cfg.derived_predicates.items():
        data = data.with_columns(code.eval_expr().cast(pl.UInt16).alias(name))
        logger.debug(f"Added predicate column '{name}'.")
        predicate_cols.append(name)

    # add a column of 1s representing any predicate
    data = data.with_columns(pl.lit(1).alias(ANY_EVENT_COLUMN).cast(pl.UInt16))
    logger.debug(f"Added predicate column '{ANY_EVENT_COLUMN}'.")
    predicate_cols.append(ANY_EVENT_COLUMN)

    data = data.sort(by=["subject_id", "timestamp"]).select(["subject_id", "timestamp"] + predicate_cols)

    return data
