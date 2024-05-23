"""This module contains functions for generating predicate columns for event sequences."""

from pathlib import Path

import polars as pl
from EventStream.data.dataset_polars import Dataset
from loguru import logger

from .config import TaskExtractorConfig
from .types import ANY_EVENT_COLUMN


def verify_plain_predicates_from_csv(data_path: Path, predicates: dict) -> pl.DataFrame:
    logger.info("Loading CSV data...")
    data = pl.read_csv(data_path).with_columns(
        pl.col("timestamp").str.strptime(pl.Datetime, format="%m/%d/%Y %H:%M").cast(pl.Datetime)
    )

    # check if data has necessary plain predicate columns
    logger.info("Verifying plain predicate columns...")
    predicate_cols = list(predicates.keys())
    for name in predicate_cols:
        if name not in data.columns:
            raise ValueError(f"Column '{name}' not found in the provided data!")
        logger.info(f"Predicate column '{name}' found.")

    # clean up predicates_df
    logger.info("Cleaning up predicates DataFrame...")
    return data.select(["subject_id", "timestamp"] + predicate_cols)


def generate_plain_predicates_from_meds(data_path: Path, predicates: dict) -> pl.DataFrame:
    logger.info("Loading MEDS data...")
    data = pl.read_parquet(data_path).with_columns(
        pl.col("timestamp").str.strptime(pl.Datetime, format="%m/%d/%Y %H:%M").cast(pl.Datetime)
    )

    # generate plain predicate columns
    logger.info("Generating plain predicate columns...")
    for name, plain_predicate in predicates.items():
        data = data.with_columns(plain_predicate.MEDS_eval_expr().alias(name))
        logger.info(f"Added predicate column '{name}'.")

    # clean up predicates_df
    logger.info("Cleaning up predicates DataFrame...")
    predicate_cols = list(predicates.keys())
    return data.select(["subject_id", "timestamp"] + predicate_cols)


def generate_plain_predicates_from_esgpt(data_path: Path, predicates: dict) -> pl.DataFrame:
    try:
        ESD = Dataset.load(data_path)
    except Exception as e:
        raise ValueError(
            f"Error loading data using ESGPT: {e}. "
            "Please ensure the path provided is a valid ESGPT dataset directory."
        ) from e

    events_df = ESD.events_df
    dynamic_measurements_df = ESD.dynamic_measurements_df
    config = ESD.config

    logger.info("Generating plain predicate columns...")
    for name, plain_predicate in predicates.items():
        if "event_type" in plain_predicate.code:
            events_df = events_df.with_columns(plain_predicate.ESGPT_eval_expr().cast(pl.UInt16).alias(name))
        else:
            values_column = config.measurement_configs[plain_predicate.code.split("//")[0]].values_column
            dynamic_measurements_df = dynamic_measurements_df.with_columns(
                plain_predicate.ESGPT_eval_expr(values_column).cast(pl.UInt16).alias(name)
            )
        logger.info(f"Added predicate column '{name}'.")

    predicate_cols = list(predicates.keys())

    # aggregate dynamic_measurements_df by summing predicates (counts)
    dynamic_measurements_df = (
        dynamic_measurements_df.group_by(["event_id"])
        .agg(
            *[pl.col(c).sum().cast(pl.Int64) for c in dynamic_measurements_df.columns if c in predicate_cols],
        )
        .select(["event_id"] + [c for c in dynamic_measurements_df.columns if c in predicate_cols])
    )

    # join events_df and dynamic_measurements_df for the final predicates_df
    data = events_df.join(dynamic_measurements_df, on="event_id", how="left")

    # clean up predicates_df
    logger.info("Cleaning up predicates DataFrame...")
    return data.select(["subject_id", "timestamp"] + predicate_cols)


def generate_predicates_df(cfg: TaskExtractorConfig, data_path: str | Path, standard: str) -> pl.DataFrame:
    """Generate predicate columns based on the configuration.

    Args:
        cfg: The TaskExtractorConfig object containing the predicates information.
        data_path: Path to external data (file path to .csv or .parquet, or ESGPT directory) as
            string or Path.
        standard: The data standard, either 'CSV, 'MEDS' or 'ESGPT'.

    Returns:
        predicates_df: The Polars DataFrame with the added predicate columns.

    Raises:
        ValueError: If an invalid predicate type is specified in the configuration.
    """
    if isinstance(data_path, str):
        data_path = Path(data_path)

    # plain predicates
    plain_predicates = cfg.plain_predicates
    match standard.lower():
        case "csv":
            data = verify_plain_predicates_from_csv(data_path, plain_predicates)
        case "meds":
            data = generate_plain_predicates_from_meds(data_path, plain_predicates)
        case "esgpt":
            data = generate_plain_predicates_from_esgpt(data_path, plain_predicates)
        case _:
            raise ValueError(f"Invalid data standard: {standard}")
    predicate_cols = list(plain_predicates.keys())

    # derived predicates
    logger.info("Generating derived predicate columns...")
    for name, code in cfg.derived_predicates.items():
        data = data.with_columns(code.eval_expr().cast(pl.UInt16).alias(name))
        logger.info(f"Added predicate column '{name}'.")
        predicate_cols.append(name)

    # add a column of 1s representing any predicate
    logger.info("Generating '_ANY_EVENT' predicate column...")
    data = data.with_columns(pl.lit(1).alias(ANY_EVENT_COLUMN).cast(pl.UInt16))
    logger.info(f"Added predicate column '{ANY_EVENT_COLUMN}'.")
    predicate_cols.append(ANY_EVENT_COLUMN)

    return data.sort(by=["subject_id", "timestamp"])
