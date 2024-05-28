"""This module contains functions for generating predicate columns for event sequences."""

from pathlib import Path

import polars as pl
from loguru import logger

from .config import TaskExtractorConfig
from .types import ANY_EVENT_COLUMN, PRED_CNT_TYPE

CSV_TIMESTAMP_FORMAT = "%m/%d/%Y %H:%M"


def verify_plain_predicates_from_csv(data_path: Path, predicates: list[str]) -> pl.DataFrame:
    """Loads a CSV file from disk and verifies that the necessary plain predicate columns are present.

    This CSV file must have the following columns:
        - subject_id: The subject identifier.
        - timestamp: The timestamp of the event, in the format "MM/DD/YYYY HH:MM".
        - Any additional columns specified in the set of desired plain predicates.

    Args:
        data_path: The path to the CSV file.
        predicates: The list of columns to read from the CSV file.

    Returns:
        The Polars DataFrame containing the specified columns.

    Example:
        >>> import tempfile
        >>> CSV_data = pl.DataFrame({
        ...     "subject_id": [1, 1, 2],
        ...     "timestamp": ["01/01/2021 00:00", "01/01/2021 12:00", "01/02/2021 00:00"],
        ...     "is_admission": [1, 0, 1],
        ...     "is_discharge": [0, 1, 0],
        ... })
        >>> with tempfile.NamedTemporaryFile(mode="w", suffix=".csv") as f:
        ...     data_path = Path(f.name)
        ...     CSV_data.write_csv(data_path)
        ...     verify_plain_predicates_from_csv(data_path, ["is_admission", "is_discharge"])
        shape: (3, 4)
        ┌────────────┬─────────────────────┬──────────────┬──────────────┐
        │ subject_id ┆ timestamp           ┆ is_admission ┆ is_discharge │
        │ ---        ┆ ---                 ┆ ---          ┆ ---          │
        │ i64        ┆ datetime[μs]        ┆ i64          ┆ i64          │
        ╞════════════╪═════════════════════╪══════════════╪══════════════╡
        │ 1          ┆ 2021-01-01 00:00:00 ┆ 1            ┆ 0            │
        │ 1          ┆ 2021-01-01 12:00:00 ┆ 0            ┆ 1            │
        │ 2          ┆ 2021-01-02 00:00:00 ┆ 1            ┆ 0            │
        └────────────┴─────────────────────┴──────────────┴──────────────┘
        >>> with tempfile.NamedTemporaryFile(mode="w", suffix=".csv") as f:
        ...     data_path = Path(f.name)
        ...     CSV_data.write_csv(data_path)
        ...     verify_plain_predicates_from_csv(data_path, ["is_discharge"])
        shape: (3, 3)
        ┌────────────┬─────────────────────┬──────────────┐
        │ subject_id ┆ timestamp           ┆ is_discharge │
        │ ---        ┆ ---                 ┆ ---          │
        │ i64        ┆ datetime[μs]        ┆ i64          │
        ╞════════════╪═════════════════════╪══════════════╡
        │ 1          ┆ 2021-01-01 00:00:00 ┆ 0            │
        │ 1          ┆ 2021-01-01 12:00:00 ┆ 1            │
        │ 2          ┆ 2021-01-02 00:00:00 ┆ 0            │
        └────────────┴─────────────────────┴──────────────┘
        >>> with tempfile.NamedTemporaryFile(mode="w", suffix=".csv") as f:
        ...     data_path = Path(f.name)
        ...     CSV_data.write_csv(data_path)
        ...     verify_plain_predicates_from_csv(data_path, ["is_foobar"])
        Traceback (most recent call last):
            ...
        polars.exceptions.ColumnNotFoundError: unable to find column "is_foobar"...
    """

    columns = ["subject_id", "timestamp"] + predicates
    logger.info(f"Attempting to load {columns} from CSV file {str(data_path.resolve())}")
    data = pl.read_csv(data_path, columns=columns).drop_nulls(subset=["subject_id", "timestamp"])
    return (
        data.select(
            "subject_id",
            pl.col("timestamp").str.strptime(pl.Datetime, format=CSV_TIMESTAMP_FORMAT),
            *predicates,
        )
        .group_by(["subject_id", "timestamp"], maintain_order=True)
        .agg(*(pl.col(c).sum().alias(c) for c in predicates))
    )


def generate_plain_predicates_from_meds(data_path: Path, predicates: dict) -> pl.DataFrame:
    """Generate plain predicate columns from a MEDS dataset.

    To learn more about the MEDS format, please visit https://github.com/Medical-Event-Data-Standard/meds

    Args:
        data_path: The path to the MEDS dataset file.
        predicates: The dictionary of plain predicate configurations.

    Returns:
        The Polars DataFrame containing the extracted predicates per subject per timestamp across the entire
        MEDS dataset.

    Example:
        >>> import tempfile
        >>> from .config import PlainPredicateConfig
        >>> parquet_data = pl.DataFrame({
        ...     "patient_id": [1, 1, 1, 2, 3],
        ...     "timestamp": ["1/1/1989 00:00", "1/1/1989 01:00", "1/1/1989 01:00", "1/1/1989 02:00", None],
        ...     "code": ['admission', 'discharge', 'discharge', 'admission', "gender"],
        ... }).with_columns(pl.col("timestamp").str.strptime(pl.Datetime, format="%m/%d/%Y %H:%M"))
        >>> with tempfile.NamedTemporaryFile(mode="w", suffix=".parquet") as f:
        ...     data_path = Path(f.name)
        ...     parquet_data.write_parquet(data_path)
        ...     generate_plain_predicates_from_meds(data_path, {"discharge":
        ...                                                         PlainPredicateConfig("discharge")})
        shape: (3, 3)
        ┌────────────┬─────────────────────┬───────────┐
        │ subject_id ┆ timestamp           ┆ discharge │
        │ ---        ┆ ---                 ┆ ---       │
        │ i64        ┆ datetime[μs]        ┆ i64       │
        ╞════════════╪═════════════════════╪═══════════╡
        │ 1          ┆ 1989-01-01 00:00:00 ┆ 0         │
        │ 1          ┆ 1989-01-01 01:00:00 ┆ 2         │
        │ 2          ┆ 1989-01-01 02:00:00 ┆ 0         │
        └────────────┴─────────────────────┴───────────┘
    """

    logger.info("Loading MEDS data...")
    data = (
        pl.read_parquet(data_path)
        .rename({"patient_id": "subject_id"})
        .drop_nulls(subset=["subject_id", "timestamp"])
    )

    # generate plain predicate columns
    logger.info("Generating plain predicate columns...")
    for name, plain_predicate in predicates.items():
        data = data.with_columns(plain_predicate.MEDS_eval_expr().cast(PRED_CNT_TYPE).alias(name))
        logger.info(f"Added predicate column '{name}'.")

    # clean up predicates_df
    logger.info("Cleaning up predicates DataFrame...")
    predicate_cols = list(predicates.keys())
    return (
        data.select(["subject_id", "timestamp"] + predicate_cols)
        .group_by(["subject_id", "timestamp"], maintain_order=True)
        .agg(*(pl.col(c).sum().alias(c) for c in predicate_cols))
    )


def generate_plain_predicates_from_esgpt(data_path: Path, predicates: dict) -> pl.DataFrame:
    """Generate plain predicate columns from an ESGPT dataset.

    To learn more about the ESGPT format, please visit https://eventstreamml.readthedocs.io/en/latest/

    Args:
        data_path: The path to the ESGPT dataset directory.
        predicates: The dictionary of plain predicate configurations.

    Returns:
        The Polars DataFrame containing the extracted predicates per subject per timestamp across the entire
        ESGPT dataset.
    """

    try:
        from EventStream.data.dataset_polars import Dataset
    except ImportError as e:
        raise ImportError(
            "The 'EventStream' package is required to load ESGPT datasets. "
            "If you mean to use a MEDS dataset, please specify the 'MEDS' standard. "
            "Otherwise, please install the package from https://github.com/mmcdermott/EventStreamGPT and add "
            "the package to your PYTHONPATH."
        ) from e

    try:
        ESD = Dataset.load(data_path)
    except Exception as e:
        raise ValueError(
            f"Error loading data using ESGPT: {e}. "
            "Please ensure the path provided is a valid ESGPT dataset directory. "
            "If you mean to use a MEDS dataset, please specify the 'MEDS' standard."
        ) from e

    events_df = ESD.events_df
    dynamic_measurements_df = ESD.dynamic_measurements_df
    config = ESD.config

    logger.info("Generating plain predicate columns...")
    for name, plain_predicate in predicates.items():
        if "event_type" in plain_predicate.code:
            events_df = events_df.with_columns(
                plain_predicate.ESGPT_eval_expr().cast(PRED_CNT_TYPE).alias(name)
            )
        else:
            values_column = config.measurement_configs[plain_predicate.code.split("//")[0]].values_column
            dynamic_measurements_df = dynamic_measurements_df.with_columns(
                plain_predicate.ESGPT_eval_expr(values_column).cast(PRED_CNT_TYPE).alias(name)
            )
        logger.info(f"Added predicate column '{name}'.")

    predicate_cols = list(predicates.keys())

    # aggregate dynamic_measurements_df by summing predicates (counts)
    dynamic_measurements_df = (
        dynamic_measurements_df.group_by(["event_id"])
        .agg(
            *[
                pl.col(c).sum().cast(PRED_CNT_TYPE)
                for c in dynamic_measurements_df.columns
                if c in predicate_cols
            ],
        )
        .select(["event_id"] + [c for c in dynamic_measurements_df.columns if c in predicate_cols])
    )

    # join events_df and dynamic_measurements_df for the final predicates_df
    data = events_df.join(dynamic_measurements_df, on="event_id", how="left")

    # clean up predicates_df
    logger.info("Cleaning up predicates DataFrame...")
    return data.select(["subject_id", "timestamp"] + predicate_cols)


def get_predicates_df(cfg: TaskExtractorConfig, data_path: str | Path, standard: str) -> pl.DataFrame:
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

    Example:
        >>> import tempfile
        >>> from .config import PlainPredicateConfig, DerivedPredicateConfig, EventConfig, WindowConfig
        >>> CSV_data = pl.DataFrame({
        ...     "subject_id": [1, 1, 2, 2],
        ...     "timestamp": ["01/01/2021 00:00", "01/01/2021 12:00", "01/02/2021 00:00", "01/02/2021 12:00"],
        ...     "admission": [1, 0, 1, 0],
        ...     "discharge": [0, 1, 0, 0],
        ...     "death":     [0, 0, 0, 1],
        ... })
        >>> predicates = {
        ...     "admission": PlainPredicateConfig("admission"),
        ...     "discharge": PlainPredicateConfig("discharge"),
        ...     "death": PlainPredicateConfig("death"),
        ...     "death_or_discharge": DerivedPredicateConfig("or(death, discharge)"),
        ... }
        >>> trigger = EventConfig("admission")
        >>> windows = {
        ...     "input": WindowConfig(
        ...         start=None,
        ...         end="trigger + 24h",
        ...         start_inclusive=True,
        ...         end_inclusive=True,
        ...         has={"_ANY_EVENT": "(32, None)"},
        ...     ),
        ...     "gap": WindowConfig(
        ...         start="input.end",
        ...         end="start + 24h",
        ...         start_inclusive=False,
        ...         end_inclusive=True,
        ...         has={"death_or_discharge": "(None, 0)", "admission": "(None, 0)"},
        ...     ),
        ...     "target": WindowConfig(
        ...         start="gap.end",
        ...         end="start -> death_or_discharge",
        ...         start_inclusive=False,
        ...         end_inclusive=True,
        ...         has={},
        ...     ),
        ... }
        >>> config = TaskExtractorConfig(predicates=predicates, trigger=trigger, windows=windows)
        >>> with tempfile.NamedTemporaryFile(mode="w", suffix=".csv") as f:
        ...     data_path = Path(f.name)
        ...     CSV_data.write_csv(data_path)
        ...     get_predicates_df(config, data_path, standard="csv")
        shape: (4, 7)
        ┌────────────┬─────────────────────┬───────────┬───────────┬───────┬────────────────────┬────────────┐
        │ subject_id ┆ timestamp           ┆ admission ┆ discharge ┆ death ┆ death_or_discharge ┆ _ANY_EVENT │
        │ ---        ┆ ---                 ┆ ---       ┆ ---       ┆ ---   ┆ ---                ┆ ---        │
        │ i64        ┆ datetime[μs]        ┆ i64       ┆ i64       ┆ i64   ┆ i64                ┆ i64        │
        ╞════════════╪═════════════════════╪═══════════╪═══════════╪═══════╪════════════════════╪════════════╡
        │ 1          ┆ 2021-01-01 00:00:00 ┆ 1         ┆ 0         ┆ 0     ┆ 0                  ┆ 1          │
        │ 1          ┆ 2021-01-01 12:00:00 ┆ 0         ┆ 1         ┆ 0     ┆ 1                  ┆ 1          │
        │ 2          ┆ 2021-01-02 00:00:00 ┆ 1         ┆ 0         ┆ 0     ┆ 0                  ┆ 1          │
        │ 2          ┆ 2021-01-02 12:00:00 ┆ 0         ┆ 0         ┆ 1     ┆ 1                  ┆ 1          │
        └────────────┴─────────────────────┴───────────┴───────────┴───────┴────────────────────┴────────────┘
        >>> with tempfile.NamedTemporaryFile(mode="w", suffix=".csv") as f:
        ...     data_path = Path(f.name)
        ...     CSV_data.write_csv(data_path)
        ...     get_predicates_df(config, data_path, standard="buzz")
        Traceback (most recent call last):
            ...
        ValueError: Invalid data standard: buzz. Options are 'CSV', 'MEDS', 'ESGPT'.
    """
    if isinstance(data_path, str):
        data_path = Path(data_path)

    # plain predicates
    plain_predicates = cfg.plain_predicates
    match standard.lower():
        case "csv":
            data = verify_plain_predicates_from_csv(data_path, list(plain_predicates.keys()))
        case "meds":
            data = generate_plain_predicates_from_meds(data_path, plain_predicates)
        case "esgpt":
            data = generate_plain_predicates_from_esgpt(data_path, plain_predicates)
        case _:
            raise ValueError(f"Invalid data standard: {standard}. Options are 'CSV', 'MEDS', 'ESGPT'.")
    predicate_cols = list(plain_predicates.keys())

    # derived predicates
    logger.info("Loaded plain predicates. Generating derived predicate columns...")
    for name, code in cfg.derived_predicates.items():
        data = data.with_columns(code.eval_expr().cast(PRED_CNT_TYPE).alias(name))
        logger.info(f"Added predicate column '{name}'.")
        predicate_cols.append(name)

    # add a column of 1s representing any predicate
    logger.info("Generating '_ANY_EVENT' predicate column...")
    data = data.with_columns(pl.lit(1).alias(ANY_EVENT_COLUMN).cast(PRED_CNT_TYPE))
    logger.info(f"Added predicate column '{ANY_EVENT_COLUMN}'.")
    predicate_cols.append(ANY_EVENT_COLUMN)

    return data.sort(by=["subject_id", "timestamp"])
