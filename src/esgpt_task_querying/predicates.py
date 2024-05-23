"""This module contains functions for generating predicate columns for event sequences."""


import polars as pl
from loguru import logger

from .config import TaskExtractorConfig
from .types import ANY_EVENT_COLUMN


def generate_predicates_df(cfg: TaskExtractorConfig, data: pl.DataFrame, standard: str) -> pl.DataFrame:
    """Generate predicate columns based on the configuration.

    Args:
        cfg: The TaskExtractorConfig object containing the predicates information.
        df_data: The Polars DataFrame containing the original external data.
        standard: The data standard, either 'CSV, 'MEDS' or 'ESGPT'.

    Returns:
        predicates_df: The Polars DataFrame with the added predicate columns.

    Raises:
        ValueError: If an invalid predicate type is specified in the configuration.

    Examples: TODO
    """
    logger.info("Generating predicate columns...")
    predicate_cols = []

    # plain predicates
    match standard:
        case "CSV":
            for name, plain_predicate in cfg.plain_predicates.items():
                data = data.with_columns(
                    plain_predicate.ESGPT_eval_expr(plain_predicate.values_column).cast(pl.UInt16).alias(name)
                )
                logger.info(f"Added predicate column '{name}'.")
                predicate_cols.append(name)
        case "MEDS":
            for name, plain_predicate in cfg.plain_predicates.items():
                data = data.with_columns(plain_predicate.MEDS_eval_expr().alias(name))
                logger.info(f"Added predicate column '{name}'.")
                predicate_cols.append(name)
        case "ESGPT":
            for name, plain_predicate in cfg.plain_predicates.items():
                if "event_type" in plain_predicate.code:
                    data[0] = data[0].with_columns(
                        plain_predicate.ESGPT_eval_expr(plain_predicate.values_column)
                        .cast(pl.UInt16)
                        .alias(name)
                    )
                else:
                    data[1] = data[1].with_columns(
                        plain_predicate.ESGPT_eval_expr(plain_predicate.values_column)
                        .cast(pl.UInt16)
                        .alias(name)
                    )
                logger.info(f"Added predicate column '{name}'.")
                predicate_cols.append(name)

            # aggregate measurements (data[1]) by summing columns that are in count_cols, and taking the max
            # for columns in boolean_cols -> new ver only maxing
            data[1] = (
                data[1]
                .group_by(["event_id"])
                .agg(
                    *[pl.col(c).max().cast(pl.Int64) for c in data[1].columns if c in predicate_cols],
                )
            )

            data = data[0].join(data[1], on="event_id", how="left")
            data = data.select(
                "subject_id",
                "timestamp",
                *[pl.col(c) for c in data.columns if c in predicate_cols],
            )

    # derived predicates
    for name, code in cfg.derived_predicates.items():
        data = data.with_columns(code.eval_expr().cast(pl.UInt16).alias(name))
        logger.info(f"Added predicate column '{name}'.")
        predicate_cols.append(name)

    # add a column of 1s representing any predicate
    data = data.with_columns(pl.lit(1).alias(ANY_EVENT_COLUMN).cast(pl.UInt16))
    logger.info(f"Added predicate column '{ANY_EVENT_COLUMN}'.")
    predicate_cols.append(ANY_EVENT_COLUMN)

    data = data.sort(by=["subject_id", "timestamp"]).select(["subject_id", "timestamp"] + predicate_cols)

    return data
