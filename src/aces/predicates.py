"""This module contains functions for generating predicate columns for event sequences."""

import logging
from pathlib import Path

import polars as pl
from omegaconf import DictConfig
from polars.exceptions import ColumnNotFoundError

from .config import PlainPredicateConfig, TaskExtractorConfig
from .types import (
    ANY_EVENT_COLUMN,
    END_OF_RECORD_KEY,
    PRED_CNT_TYPE,
    START_OF_RECORD_KEY,
)

logger = logging.getLogger(__name__)


def direct_load_plain_predicates(
    data_path: Path, predicates: list[str], ts_format: str | None
) -> pl.DataFrame:
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
        >>> CSV_data = pl.DataFrame({
        ...     "subject_id": [1, 1, 1, 1, 2, 2],
        ...     "timestamp": [None, "01/01/2021 00:00", None, "01/01/2021 12:00", "01/02/2021 00:00", None],
        ...     "is_admission": [0, 1, 0, 0, 1, 0],
        ...     "is_discharge": [0, 0, 0, 1, 0, 0],
        ...     "is_male": [1, 0, 0, 0, 0, 0],
        ...     "is_female": [0, 0, 0, 0, 0, 1],
        ...     "brown_eyes": [0, 0, 1, 0, 0, 0],
        ... })
        >>> with tempfile.NamedTemporaryFile(mode="w", suffix=".parquet") as f:
        ...     data_path = Path(f.name)
        ...     CSV_data.write_parquet(data_path)
        ...     direct_load_plain_predicates(data_path, ["is_admission", "is_discharge", "is_male",
        ...          "is_female", "brown_eyes"], "%m/%d/%Y %H:%M")
        shape: (5, 7)
        ┌────────────┬─────────────────────┬──────────────┬──────────────┬─────────┬───────────┬────────────┐
        │ subject_id ┆ timestamp           ┆ is_admission ┆ is_discharge ┆ is_male ┆ is_female ┆ brown_eyes │
        │ ---        ┆ ---                 ┆ ---          ┆ ---          ┆ ---     ┆ ---       ┆ ---        │
        │ i64        ┆ datetime[μs]        ┆ i64          ┆ i64          ┆ i64     ┆ i64       ┆ i64        │
        ╞════════════╪═════════════════════╪══════════════╪══════════════╪═════════╪═══════════╪════════════╡
        │ 1          ┆ null                ┆ 0            ┆ 0            ┆ 1       ┆ 0         ┆ 1          │
        │ 1          ┆ 2021-01-01 00:00:00 ┆ 1            ┆ 0            ┆ 0       ┆ 0         ┆ 0          │
        │ 1          ┆ 2021-01-01 12:00:00 ┆ 0            ┆ 1            ┆ 0       ┆ 0         ┆ 0          │
        │ 2          ┆ 2021-01-02 00:00:00 ┆ 1            ┆ 0            ┆ 0       ┆ 0         ┆ 0          │
        │ 2          ┆ null                ┆ 0            ┆ 0            ┆ 0       ┆ 1         ┆ 0          │
        └────────────┴─────────────────────┴──────────────┴──────────────┴─────────┴───────────┴────────────┘

    If the timestamp column is already a timestamp, then the `ts_format` argument id not needed, but can be
    used without an error.
        >>> with tempfile.NamedTemporaryFile(mode="w", suffix=".parquet") as f:
        ...     data_path = Path(f.name)
        ...     (
        ...         CSV_data
        ...         .with_columns(pl.col("timestamp").str.strptime(pl.Datetime, format="%m/%d/%Y %H:%M"))
        ...         .write_parquet(data_path)
        ...     )
        ...     direct_load_plain_predicates(data_path, ["is_admission", "is_discharge", "is_male",
        ...          "is_female", "brown_eyes"], "%m/%d/%Y %H:%M")
        shape: (5, 7)
        ┌────────────┬─────────────────────┬──────────────┬──────────────┬─────────┬───────────┬────────────┐
        │ subject_id ┆ timestamp           ┆ is_admission ┆ is_discharge ┆ is_male ┆ is_female ┆ brown_eyes │
        │ ---        ┆ ---                 ┆ ---          ┆ ---          ┆ ---     ┆ ---       ┆ ---        │
        │ i64        ┆ datetime[μs]        ┆ i64          ┆ i64          ┆ i64     ┆ i64       ┆ i64        │
        ╞════════════╪═════════════════════╪══════════════╪══════════════╪═════════╪═══════════╪════════════╡
        │ 1          ┆ null                ┆ 0            ┆ 0            ┆ 1       ┆ 0         ┆ 1          │
        │ 1          ┆ 2021-01-01 00:00:00 ┆ 1            ┆ 0            ┆ 0       ┆ 0         ┆ 0          │
        │ 1          ┆ 2021-01-01 12:00:00 ┆ 0            ┆ 1            ┆ 0       ┆ 0         ┆ 0          │
        │ 2          ┆ 2021-01-02 00:00:00 ┆ 1            ┆ 0            ┆ 0       ┆ 0         ┆ 0          │
        │ 2          ┆ null                ┆ 0            ┆ 0            ┆ 0       ┆ 1         ┆ 0          │
        └────────────┴─────────────────────┴──────────────┴──────────────┴─────────┴───────────┴────────────┘
        >>> with tempfile.NamedTemporaryFile(mode="w", suffix=".parquet") as f:
        ...     data_path = Path(f.name)
        ...     (
        ...         CSV_data
        ...         .with_columns(pl.col("timestamp").str.strptime(pl.Datetime, format="%m/%d/%Y %H:%M"))
        ...         .write_parquet(data_path)
        ...     )
        ...     direct_load_plain_predicates(data_path, ["is_admission", "is_discharge", "is_male",
        ...          "is_female", "brown_eyes"], None)
        shape: (5, 7)
        ┌────────────┬─────────────────────┬──────────────┬──────────────┬─────────┬───────────┬────────────┐
        │ subject_id ┆ timestamp           ┆ is_admission ┆ is_discharge ┆ is_male ┆ is_female ┆ brown_eyes │
        │ ---        ┆ ---                 ┆ ---          ┆ ---          ┆ ---     ┆ ---       ┆ ---        │
        │ i64        ┆ datetime[μs]        ┆ i64          ┆ i64          ┆ i64     ┆ i64       ┆ i64        │
        ╞════════════╪═════════════════════╪══════════════╪══════════════╪═════════╪═══════════╪════════════╡
        │ 1          ┆ null                ┆ 0            ┆ 0            ┆ 1       ┆ 0         ┆ 1          │
        │ 1          ┆ 2021-01-01 00:00:00 ┆ 1            ┆ 0            ┆ 0       ┆ 0         ┆ 0          │
        │ 1          ┆ 2021-01-01 12:00:00 ┆ 0            ┆ 1            ┆ 0       ┆ 0         ┆ 0          │
        │ 2          ┆ 2021-01-02 00:00:00 ┆ 1            ┆ 0            ┆ 0       ┆ 0         ┆ 0          │
        │ 2          ┆ null                ┆ 0            ┆ 0            ┆ 0       ┆ 1         ┆ 0          │
        └────────────┴─────────────────────┴──────────────┴──────────────┴─────────┴───────────┴────────────┘
        >>> with tempfile.NamedTemporaryFile(mode="w", suffix=".csv") as f:
        ...     data_path = Path(f.name)
        ...     CSV_data.write_csv(data_path)
        ...     direct_load_plain_predicates(data_path, ["is_admission", "is_discharge", "is_male",
        ...          "is_female", "brown_eyes"], "%m/%d/%Y %H:%M")
        shape: (5, 7)
        ┌────────────┬─────────────────────┬──────────────┬──────────────┬─────────┬───────────┬────────────┐
        │ subject_id ┆ timestamp           ┆ is_admission ┆ is_discharge ┆ is_male ┆ is_female ┆ brown_eyes │
        │ ---        ┆ ---                 ┆ ---          ┆ ---          ┆ ---     ┆ ---       ┆ ---        │
        │ i64        ┆ datetime[μs]        ┆ i64          ┆ i64          ┆ i64     ┆ i64       ┆ i64        │
        ╞════════════╪═════════════════════╪══════════════╪══════════════╪═════════╪═══════════╪════════════╡
        │ 1          ┆ null                ┆ 0            ┆ 0            ┆ 1       ┆ 0         ┆ 1          │
        │ 1          ┆ 2021-01-01 00:00:00 ┆ 1            ┆ 0            ┆ 0       ┆ 0         ┆ 0          │
        │ 1          ┆ 2021-01-01 12:00:00 ┆ 0            ┆ 1            ┆ 0       ┆ 0         ┆ 0          │
        │ 2          ┆ 2021-01-02 00:00:00 ┆ 1            ┆ 0            ┆ 0       ┆ 0         ┆ 0          │
        │ 2          ┆ null                ┆ 0            ┆ 0            ┆ 0       ┆ 1         ┆ 0          │
        └────────────┴─────────────────────┴──────────────┴──────────────┴─────────┴───────────┴────────────┘
        >>> with tempfile.NamedTemporaryFile(mode="w", suffix=".csv") as f:
        ...     data_path = Path(f.name)
        ...     CSV_data.write_csv(data_path)
        ...     direct_load_plain_predicates(data_path, ["is_discharge", "brown_eyes"], "%m/%d/%Y %H:%M")
        shape: (5, 4)
        ┌────────────┬─────────────────────┬──────────────┬────────────┐
        │ subject_id ┆ timestamp           ┆ is_discharge ┆ brown_eyes │
        │ ---        ┆ ---                 ┆ ---          ┆ ---        │
        │ i64        ┆ datetime[μs]        ┆ i64          ┆ i64        │
        ╞════════════╪═════════════════════╪══════════════╪════════════╡
        │ 1          ┆ null                ┆ 0            ┆ 1          │
        │ 1          ┆ 2021-01-01 00:00:00 ┆ 0            ┆ 0          │
        │ 1          ┆ 2021-01-01 12:00:00 ┆ 1            ┆ 0          │
        │ 2          ┆ 2021-01-02 00:00:00 ┆ 0            ┆ 0          │
        │ 2          ┆ null                ┆ 0            ┆ 0          │
        └────────────┴─────────────────────┴──────────────┴────────────┘
        >>> with tempfile.NamedTemporaryFile(mode="w", suffix=".csv") as f:
        ...     data_path = Path(f.name)
        ...     CSV_data.write_csv(data_path)
        ...     direct_load_plain_predicates(data_path, ["is_foobar"], "%m/%d/%Y %H:%M")
        Traceback (most recent call last):
            ...
        polars.exceptions.ColumnNotFoundError: ['is_foobar']
        >>> with tempfile.NamedTemporaryFile(mode="w", suffix=".foo") as f:
        ...     data_path = Path(f.name)
        ...     CSV_data.write_csv(data_path)
        ...     direct_load_plain_predicates(data_path, ["is_discharge"], "%m/%d/%Y %H:%M")
        Traceback (most recent call last):
            ...
        ValueError: Unsupported file format: .foo
        >>> with tempfile.TemporaryDirectory() as d:
        ...     data_path = Path(d) / "data.csv"
        ...     assert not data_path.exists()
        ...     direct_load_plain_predicates(data_path, ["is_admission", "is_discharge"], "%m/%d/%Y %H:%M")
        Traceback (most recent call last):
            ...
        FileNotFoundError: Direct predicates file ... does not exist!
        >>> with tempfile.NamedTemporaryFile(mode="w", suffix=".parquet") as f:
        ...     data_path = Path(f.name)
        ...     CSV_data.write_parquet(data_path)
        ...     direct_load_plain_predicates(data_path, ["is_admission", "is_discharge"], None)
        Traceback (most recent call last):
            ...
        ValueError: Must provide a timestamp format for direct predicates with str timestamps.
        >>> with tempfile.NamedTemporaryFile(mode="w", suffix=".parquet") as f:
        ...     data_path = Path(f.name)
        ...     (
        ...         CSV_data
        ...         .with_columns(
        ...             pl.col("timestamp").str.strptime(pl.Datetime, format="%m/%d/%Y %H:%M")
        ...             .dt.timestamp()
        ...         )
        ...         .write_parquet(data_path)
        ...     )
        ...     direct_load_plain_predicates(data_path, ["is_admission", "is_discharge"], None)
        Traceback (most recent call last):
            ...
        TypeError: Passed predicates have timestamps of invalid type Int64.
    """

    columns = ["subject_id", "timestamp", *predicates]
    logger.info(f"Attempting to load {columns} from file {data_path.resolve()!s}")

    if not data_path.is_file():
        raise FileNotFoundError(f"Direct predicates file {data_path} does not exist!")

    match data_path.suffix:
        case ".csv":
            data = pl.scan_csv(data_path)
        case ".parquet":
            data = pl.scan_parquet(data_path)
        case _:
            raise ValueError(f"Unsupported file format: {data_path.suffix}")

    missing_columns = [col for col in columns if col not in data.collect_schema().names()]
    if missing_columns:
        raise ColumnNotFoundError(missing_columns)

    data = data.select(columns)
    ts_type = data.collect_schema()["timestamp"]
    if ts_type == pl.Utf8:
        if ts_format is None:
            raise ValueError("Must provide a timestamp format for direct predicates with str timestamps.")
        data = data.with_columns(pl.col("timestamp").str.strptime(pl.Datetime, format=ts_format))
    elif ts_type.is_temporal():
        if ts_format is not None:
            logger.info(
                f"Ignoring specified timestamp format of {ts_format} as timestamps are already {ts_type}"
            )
    else:
        raise TypeError(f"Passed predicates have timestamps of invalid type {ts_type}.")

    logger.info("Cleaning up predicates dataframe...")
    return (
        data.select("subject_id", "timestamp", *predicates)
        .group_by(["subject_id", "timestamp"], maintain_order=True)
        .agg(pl.all().sum().cast(PRED_CNT_TYPE))
        .collect()
    )


def generate_plain_predicates_from_meds(
    data_path: Path, predicates: dict[str, PlainPredicateConfig]
) -> pl.DataFrame:
    """Generate plain predicate columns from a MEDS dataset.

    To learn more about the MEDS format, please visit https://github.com/Medical-Event-Data-Standard/meds

    Args:
        data_path: The path to the MEDS dataset file.
        predicates: The dictionary of plain predicate configurations.

    Returns:
        The Polars DataFrame containing the extracted predicates per subject per timestamp across the entire
        MEDS dataset.

    Example:
        >>> parquet_data = pl.DataFrame({
        ...     "subject_id": [1, 1, 1, 2, 3],
        ...     "time": ["1/1/1989 00:00", "1/1/1989 01:00", "1/1/1989 01:00", "1/1/1989 02:00", None],
        ...     "code": ['admission', 'discharge', 'discharge', 'admission', "gender//male"],
        ... }).with_columns(pl.col("time").str.strptime(pl.Datetime, format="%m/%d/%Y %H:%M"))
        >>> with tempfile.NamedTemporaryFile(mode="w", suffix=".parquet") as f:
        ...     data_path = Path(f.name)
        ...     parquet_data.write_parquet(data_path)
        ...     generate_plain_predicates_from_meds(
        ...         data_path,
        ...         {"discharge": PlainPredicateConfig("discharge"),
        ...             "male": PlainPredicateConfig("gender//male", static=True)}
        ...     )
        shape: (4, 4)
        ┌────────────┬─────────────────────┬───────────┬──────┐
        │ subject_id ┆ timestamp           ┆ discharge ┆ male │
        │ ---        ┆ ---                 ┆ ---       ┆ ---  │
        │ i64        ┆ datetime[μs]        ┆ i64       ┆ i64  │
        ╞════════════╪═════════════════════╪═══════════╪══════╡
        │ 1          ┆ 1989-01-01 00:00:00 ┆ 0         ┆ 0    │
        │ 1          ┆ 1989-01-01 01:00:00 ┆ 2         ┆ 0    │
        │ 2          ┆ 1989-01-01 02:00:00 ┆ 0         ┆ 0    │
        │ 3          ┆ null                ┆ 0         ┆ 1    │
        └────────────┴─────────────────────┴───────────┴──────┘
    """

    logger.info("Loading MEDS data...")
    data = pl.read_parquet(data_path, use_pyarrow=True).rename({"time": "timestamp"})

    # generate plain predicate columns
    logger.info("Generating plain predicate columns...")
    for name, plain_predicate in predicates.items():
        data = data.with_columns(data["code"].cast(pl.Utf8).alias("code"))  # may remove after MEDS v0.3
        data = data.with_columns(plain_predicate.MEDS_eval_expr().cast(PRED_CNT_TYPE).alias(name))
        logger.info(f"Added predicate column '{name}'.")

    # clean up predicates_df
    logger.info("Cleaning up predicates dataframe...")
    predicate_cols = list(predicates.keys())
    return (
        data.select(["subject_id", "timestamp", *predicate_cols])
        .group_by(["subject_id", "timestamp"], maintain_order=True)
        .agg(*(pl.col(c).sum().cast(PRED_CNT_TYPE).alias(c) for c in predicate_cols))
    )


def process_esgpt_data(
    subjects_df: pl.DataFrame,
    events_df: pl.DataFrame,
    dynamic_measurements_df: pl.DataFrame,
    value_columns: dict[str, str],
    predicates: dict,
) -> pl.DataFrame:
    """Process ESGPT data to generate plain predicate columns.

    Args:
        events_df: The Polars DataFrame containing the events data.
        dynamic_measurements_df: The Polars DataFrame containing the dynamic measurements data.

    Returns:
        The Polars DataFrame containing the extracted predicates per subject per timestamp across the entire
        ESGPT dataset.

    Examples:
        >>> subjects_df = pl.DataFrame({
        ...    "subject_id": [1, 2],
        ...    "MRN": ["A123", "B456"],
        ...    "eye_colour": ["brown", "blue"],
        ...    "dob": [datetime(1980, 1, 1), datetime(1990, 1, 1)],
        ... })
        >>> events_df = pl.DataFrame({
        ...    "event_id": [1, 2, 3, 4],
        ...    "subject_id": [1, 1, 2, 2],
        ...    "timestamp": [
        ...         datetime(2021, 1, 1, 0, 0),
        ...         datetime(2021, 1, 1, 12, 0),
        ...         datetime(2021, 1, 2, 0, 0),
        ...         datetime(2021, 1, 2, 12, 0),
        ...    ],
        ...    "event_type": ["adm", "dis", "adm", "obs"],
        ...    "age": [30, 30, 40, 40],
        ... })
        >>> dynamic_measurements_df = pl.DataFrame({
        ...    "event_id": [1,     1,    1,    2,    2,    2,    3,     4,    5],
        ...    "adm_loc":  ["foo", None, None, None, None, None, "bar", None, None],
        ...    "dis_loc":  [None,  None, None, None, None, "H",  None,  None, None],
        ...    "HR":       [None,  150,  None, 120,  None, None, None,  177,  89],
        ...    "lab":      [None,  None, "K",  None, "K",  None, None,  None, "SpO2"],
        ...    "lab_val":  [None,  None, 5.1,  None, 3.8,  None, None,  None, 99],
        ... })
        >>> value_columns = {
        ...    "is_admission": None,
        ...    "is_discharge": None,
        ...    "high_HR": "HR",
        ...    "high_Potassium": "lab_val",
        ... }
        >>> predicates = {
        ...    "is_adm": PlainPredicateConfig(code="event_type//adm"),
        ...    "is_dis": PlainPredicateConfig(code="event_type//dis"),
        ...    "high_HR": PlainPredicateConfig(code="HR", value_min=140),
        ...    "high_Potassium": PlainPredicateConfig(code="lab//K", value_min=5.0),
        ...    "eye_colour": PlainPredicateConfig(code="eye_colour//brown", static=True),
        ... }
        >>> process_esgpt_data(subjects_df, events_df, dynamic_measurements_df, value_columns, predicates)
        shape: (6, 7)
        ┌────────────┬─────────────────────┬────────┬────────┬─────────┬────────────────┬────────────┐
        │ subject_id ┆ timestamp           ┆ is_adm ┆ is_dis ┆ high_HR ┆ high_Potassium ┆ eye_colour │
        │ ---        ┆ ---                 ┆ ---    ┆ ---    ┆ ---     ┆ ---            ┆ ---        │
        │ i64        ┆ datetime[μs]        ┆ i64    ┆ i64    ┆ i64     ┆ i64            ┆ i64        │
        ╞════════════╪═════════════════════╪════════╪════════╪═════════╪════════════════╪════════════╡
        │ 1          ┆ null                ┆ 0      ┆ 0      ┆ 0       ┆ 0              ┆ 1          │
        │ 2          ┆ null                ┆ 0      ┆ 0      ┆ 0       ┆ 0              ┆ 0          │
        │ 1          ┆ 2021-01-01 00:00:00 ┆ 1      ┆ 0      ┆ 1       ┆ 1              ┆ 0          │
        │ 1          ┆ 2021-01-01 12:00:00 ┆ 0      ┆ 1      ┆ 0       ┆ 0              ┆ 0          │
        │ 2          ┆ 2021-01-02 00:00:00 ┆ 1      ┆ 0      ┆ 0       ┆ 0              ┆ 0          │
        │ 2          ┆ 2021-01-02 12:00:00 ┆ 0      ┆ 0      ┆ 1       ┆ 0              ┆ 0          │
        └────────────┴─────────────────────┴────────┴────────┴─────────┴────────────────┴────────────┘
    """

    logger.info("Generating plain predicate columns...")
    for name, plain_predicate in predicates.items():
        if "event_type" in plain_predicate.code:
            events_df = events_df.with_columns(
                plain_predicate.ESGPT_eval_expr().cast(PRED_CNT_TYPE).alias(name)
            )
        elif plain_predicate.static:
            subjects_df = subjects_df.with_columns(
                plain_predicate.ESGPT_eval_expr().cast(PRED_CNT_TYPE).alias(name),
            )
        else:
            values_column = value_columns[name]
            dynamic_measurements_df = dynamic_measurements_df.with_columns(
                plain_predicate.ESGPT_eval_expr(values_column).cast(PRED_CNT_TYPE).alias(name)
            )
        logger.info(f"Added predicate column '{name}'.")

    # clean up predicates_df
    logger.info("Cleaning up predicates dataframe...")
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

    # return concatenated subjects_df and data
    static_rows = subjects_df.select(
        "subject_id",
        pl.lit(None).alias("timestamp").cast(pl.Datetime),
        *[pl.lit(0).alias(c).cast(PRED_CNT_TYPE) for c in predicate_cols if not predicates[c].static],
        *[pl.col(c) for c in predicate_cols if predicates[c].static],
    )
    data = data.select(
        "subject_id",
        "timestamp",
        *[pl.col(c) for c in predicate_cols if not predicates[c].static],
        *[pl.lit(0).alias(c).cast(PRED_CNT_TYPE) for c in predicate_cols if predicates[c].static],
    )
    return pl.concat([static_rows, data])


def generate_plain_predicates_from_esgpt(data_path: Path, predicates: dict) -> pl.DataFrame:
    """Generate plain predicate columns from an ESGPT dataset.

    To learn more about the ESGPT format, please visit https://eventstreamml.readthedocs.io/en/latest/

    Args:
        data_path: The path to the ESGPT dataset directory.
        predicates: The dictionary of plain predicate configurations.

    Returns:
        The Polars DataFrame containing the extracted predicates per subject per timestamp across the entire
        ESGPT dataset.

        >>> generate_plain_predicates_from_esgpt(Path("/fake/path"), {})
        Traceback (most recent call last):
            ...
        ImportError: The 'EventStream' package is required to load ESGPT datasets. If you mean to use a
        MEDS dataset, please specify the 'MEDS' standard. Otherwise, please install the package from
        https://github.com/mmcdermott/EventStreamGPT and add the package to your PYTHONPATH.
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

    subjects_df = ESD.subjects_df
    events_df = ESD.events_df
    dynamic_measurements_df = ESD.dynamic_measurements_df
    config = ESD.config

    value_columns = {}
    for name, plain_predicate in predicates.items():
        if "event_type" in plain_predicate.code:
            value_columns[name] = None
        else:
            measurement_name = plain_predicate.code.split("//")[0]
            value_columns[name] = config.measurement_configs[measurement_name].values_column

    return process_esgpt_data(subjects_df, events_df, dynamic_measurements_df, value_columns, predicates)


def get_predicates_df(cfg: TaskExtractorConfig, data_config: DictConfig) -> pl.DataFrame:
    """Generate predicate columns based on the configuration.

    Args:
        cfg: The TaskExtractorConfig object containing the predicates information.
        data_path: Path to external data (file path to .csv or .parquet, or ESGPT directory) as
            string or Path.
        standard: The data standard, either 'CSV, 'MEDS' or 'ESGPT'.

    Returns:
        pl.DataFrame: The Polars DataFrame with the added predicate columns.

    Raises:
        ValueError: If an invalid predicate type is specified in the configuration.

    Example:
        >>> from .config import DerivedPredicateConfig, EventConfig, WindowConfig
        >>> data = pl.DataFrame({
        ...     "subject_id": [1, 1, 1, 2, 2, 2],
        ...     "timestamp": [
        ...         None,
        ...         "01/01/2021 00:00",
        ...         "01/01/2021 12:00",
        ...         None,
        ...         "01/02/2021 00:00",
        ...         "01/02/2021 12:00"],
        ...     "adm":       [0, 1, 0, 0, 1, 0],
        ...     "dis":       [0, 0, 1, 0, 0, 0],
        ...     "death":     [0, 0, 0, 0, 0, 1],
        ...     "male":      [1, 0, 0, 0, 0, 0],
        ...     "female":    [0, 0, 0, 1, 0, 0],
        ... })
        >>> predicates = {
        ...     "adm": PlainPredicateConfig("adm"),
        ...     "dis": PlainPredicateConfig("dis"),
        ...     "death": PlainPredicateConfig("death"),
        ...     "male": PlainPredicateConfig("male", static=True), # predicate match based on name for direct
        ...     "death_or_dis": DerivedPredicateConfig("or(death, dis)"),
        ... }
        >>> trigger = EventConfig("adm")
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
        ...         has={
        ...             "death_or_dis": "(None, 0)",
        ...             "adm": "(None, 0)",
        ...         },
        ...     ),
        ...     "target": WindowConfig(
        ...         start="gap.end",
        ...         end="start -> death_or_dis",
        ...         start_inclusive=False,
        ...         end_inclusive=True,
        ...         has={},
        ...     ),
        ... }
        >>> config = TaskExtractorConfig(predicates=predicates, trigger=trigger, windows=windows)
        >>> with tempfile.NamedTemporaryFile(mode="w", suffix=".csv") as f:
        ...     data_path = Path(f.name)
        ...     data.write_csv(data_path)
        ...     data_config = DictConfig({
        ...         "path": str(data_path), "standard": "direct", "ts_format": "%m/%d/%Y %H:%M"
        ...     })
        ...     get_predicates_df(config, data_config)
        shape: (6, 8)
        ┌────────────┬─────────────────────┬─────┬─────┬───────┬──────┬──────────────┬────────────┐
        │ subject_id ┆ timestamp           ┆ adm ┆ dis ┆ death ┆ male ┆ death_or_dis ┆ _ANY_EVENT │
        │ ---        ┆ ---                 ┆ --- ┆ --- ┆ ---   ┆ ---  ┆ ---          ┆ ---        │
        │ i64        ┆ datetime[μs]        ┆ i64 ┆ i64 ┆ i64   ┆ i64  ┆ i64          ┆ i64        │
        ╞════════════╪═════════════════════╪═════╪═════╪═══════╪══════╪══════════════╪════════════╡
        │ 1          ┆ null                ┆ 0   ┆ 0   ┆ 0     ┆ 1    ┆ 0            ┆ null       │
        │ 1          ┆ 2021-01-01 00:00:00 ┆ 1   ┆ 0   ┆ 0     ┆ 0    ┆ 0            ┆ 1          │
        │ 1          ┆ 2021-01-01 12:00:00 ┆ 0   ┆ 1   ┆ 0     ┆ 0    ┆ 1            ┆ 1          │
        │ 2          ┆ null                ┆ 0   ┆ 0   ┆ 0     ┆ 0    ┆ 0            ┆ null       │
        │ 2          ┆ 2021-01-02 00:00:00 ┆ 1   ┆ 0   ┆ 0     ┆ 0    ┆ 0            ┆ 1          │
        │ 2          ┆ 2021-01-02 12:00:00 ┆ 0   ┆ 0   ┆ 1     ┆ 0    ┆ 1            ┆ 1          │
        └────────────┴─────────────────────┴─────┴─────┴───────┴──────┴──────────────┴────────────┘
        >>> with tempfile.NamedTemporaryFile(mode="w", suffix=".parquet") as f:
        ...     data_path = Path(f.name)
        ...     (
        ...         data
        ...         .with_columns(pl.col("timestamp").str.strptime(pl.Datetime, format="%m/%d/%Y %H:%M"))
        ...         .write_parquet(data_path)
        ...     )
        ...     data_config = DictConfig({"path": str(data_path), "standard": "direct", "ts_format": None})
        ...     get_predicates_df(config, data_config)
        shape: (6, 8)
        ┌────────────┬─────────────────────┬─────┬─────┬───────┬──────┬──────────────┬────────────┐
        │ subject_id ┆ timestamp           ┆ adm ┆ dis ┆ death ┆ male ┆ death_or_dis ┆ _ANY_EVENT │
        │ ---        ┆ ---                 ┆ --- ┆ --- ┆ ---   ┆ ---  ┆ ---          ┆ ---        │
        │ i64        ┆ datetime[μs]        ┆ i64 ┆ i64 ┆ i64   ┆ i64  ┆ i64          ┆ i64        │
        ╞════════════╪═════════════════════╪═════╪═════╪═══════╪══════╪══════════════╪════════════╡
        │ 1          ┆ null                ┆ 0   ┆ 0   ┆ 0     ┆ 1    ┆ 0            ┆ null       │
        │ 1          ┆ 2021-01-01 00:00:00 ┆ 1   ┆ 0   ┆ 0     ┆ 0    ┆ 0            ┆ 1          │
        │ 1          ┆ 2021-01-01 12:00:00 ┆ 0   ┆ 1   ┆ 0     ┆ 0    ┆ 1            ┆ 1          │
        │ 2          ┆ null                ┆ 0   ┆ 0   ┆ 0     ┆ 0    ┆ 0            ┆ null       │
        │ 2          ┆ 2021-01-02 00:00:00 ┆ 1   ┆ 0   ┆ 0     ┆ 0    ┆ 0            ┆ 1          │
        │ 2          ┆ 2021-01-02 12:00:00 ┆ 0   ┆ 0   ┆ 1     ┆ 0    ┆ 1            ┆ 1          │
        └────────────┴─────────────────────┴─────┴─────┴───────┴──────┴──────────────┴────────────┘
        >>> any_event_trigger = EventConfig("_ANY_EVENT")
        >>> adm_only_predicates = {"adm": PlainPredicateConfig("adm"), "male": PlainPredicateConfig("male")}
        >>> st_end_windows = {
        ...     "input": WindowConfig(
        ...         start="end - 365d",
        ...         end="trigger + 24h",
        ...         start_inclusive=True,
        ...         end_inclusive=True,
        ...         has={
        ...             "_RECORD_END": "(None, 0)",   # These are added just to show start/end predicates
        ...             "_RECORD_START": "(None, 0)", # These are added just to show start/end predicates
        ...         },
        ...     ),
        ... }
        >>> st_end_config = TaskExtractorConfig(
        ...     predicates=adm_only_predicates, trigger=any_event_trigger, windows=st_end_windows
        ... )
        >>> with tempfile.NamedTemporaryFile(mode="w", suffix=".csv") as f:
        ...     data_path = Path(f.name)
        ...     data.write_csv(data_path)
        ...     data_config = DictConfig({
        ...         "path": str(data_path), "standard": "direct", "ts_format": "%m/%d/%Y %H:%M"
        ...     })
        ...     get_predicates_df(st_end_config, data_config)
        shape: (6, 7)
        ┌────────────┬─────────────────────┬─────┬──────┬────────────┬───────────────┬─────────────┐
        │ subject_id ┆ timestamp           ┆ adm ┆ male ┆ _ANY_EVENT ┆ _RECORD_START ┆ _RECORD_END │
        │ ---        ┆ ---                 ┆ --- ┆ ---  ┆ ---        ┆ ---           ┆ ---         │
        │ i64        ┆ datetime[μs]        ┆ i64 ┆ i64  ┆ i64        ┆ i64           ┆ i64         │
        ╞════════════╪═════════════════════╪═════╪══════╪════════════╪═══════════════╪═════════════╡
        │ 1          ┆ null                ┆ 0   ┆ 1    ┆ null       ┆ null          ┆ null        │
        │ 1          ┆ 2021-01-01 00:00:00 ┆ 1   ┆ 0    ┆ 1          ┆ 1             ┆ 0           │
        │ 1          ┆ 2021-01-01 12:00:00 ┆ 0   ┆ 0    ┆ 1          ┆ 0             ┆ 1           │
        │ 2          ┆ null                ┆ 0   ┆ 0    ┆ null       ┆ null          ┆ null        │
        │ 2          ┆ 2021-01-02 00:00:00 ┆ 1   ┆ 0    ┆ 1          ┆ 1             ┆ 0           │
        │ 2          ┆ 2021-01-02 12:00:00 ┆ 0   ┆ 0    ┆ 1          ┆ 0             ┆ 1           │
        └────────────┴─────────────────────┴─────┴──────┴────────────┴───────────────┴─────────────┘

        >>> data = pl.DataFrame({
        ...     "subject_id": [1, 1, 1, 2, 2],
        ...     "timestamp": [
        ...         None,
        ...         "01/01/2021 00:00",
        ...         "01/01/2021 12:00",
        ...         "01/02/2021 00:00",
        ...         "01/02/2021 12:00"],
        ...     "adm":       [0, 1, 0, 1, 0],
        ...     "male":      [1, 0, 0, 0, 0],
        ... })
        >>> predicates = {
        ...     "adm": PlainPredicateConfig("adm"),
        ...     "male": PlainPredicateConfig("male", static=True), # predicate match based on name for direct
        ...     "male_adm": DerivedPredicateConfig("and(male, adm)", static=['male']),
        ... }
        >>> trigger = EventConfig("adm")
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
        ...         has={
        ...             "adm": "(None, 0)",
        ...             "male_adm": "(None, 0)",
        ...         },
        ...     ),
        ... }
        >>> config = TaskExtractorConfig(predicates=predicates, trigger=trigger, windows=windows)
        >>> with tempfile.NamedTemporaryFile(mode="w", suffix=".csv") as f:
        ...     data_path = Path(f.name)
        ...     data.write_csv(data_path)
        ...     data_config = DictConfig({
        ...         "path": str(data_path), "standard": "direct", "ts_format": "%m/%d/%Y %H:%M"
        ...     })
        ...     get_predicates_df(config, data_config)
        shape: (5, 6)
        ┌────────────┬─────────────────────┬─────┬──────┬──────────┬────────────┐
        │ subject_id ┆ timestamp           ┆ adm ┆ male ┆ male_adm ┆ _ANY_EVENT │
        │ ---        ┆ ---                 ┆ --- ┆ ---  ┆ ---      ┆ ---        │
        │ i64        ┆ datetime[μs]        ┆ i64 ┆ i64  ┆ i64      ┆ i64        │
        ╞════════════╪═════════════════════╪═════╪══════╪══════════╪════════════╡
        │ 1          ┆ null                ┆ 0   ┆ 1    ┆ 0        ┆ null       │
        │ 1          ┆ 2021-01-01 00:00:00 ┆ 1   ┆ 1    ┆ 1        ┆ 1          │
        │ 1          ┆ 2021-01-01 12:00:00 ┆ 0   ┆ 1    ┆ 0        ┆ 1          │
        │ 2          ┆ 2021-01-02 00:00:00 ┆ 1   ┆ 0    ┆ 0        ┆ 1          │
        │ 2          ┆ 2021-01-02 12:00:00 ┆ 0   ┆ 0    ┆ 0        ┆ 1          │
        └────────────┴─────────────────────┴─────┴──────┴──────────┴────────────┘

        >>> with tempfile.NamedTemporaryFile(mode="w", suffix=".csv") as f:
        ...     data_path = Path(f.name)
        ...     data.write_csv(data_path)
        ...     data_config = DictConfig({
        ...         "path": str(data_path), "standard": "buzz", "ts_format": "%m/%d/%Y %H:%M"
        ...     })
        ...     get_predicates_df(config, data_config)
        Traceback (most recent call last):
            ...
        ValueError: Invalid data standard: buzz. Options are 'direct', 'MEDS', 'ESGPT'.
    """

    standard = data_config.standard
    data_path = Path(data_config.path)

    expand_shards_enabled = getattr(data_config, "shard", False)
    if not expand_shards_enabled and data_path.is_dir():  # pragma: no cover
        logger.warning(
            "Expand shards is not enabled but your data path is a directory. "
            "If you are working with sharded datasets or large-scale queries, using `expand_shards` and"
            "`data=sharded` will improve efficiency and completeness."
        )

    # plain predicates
    plain_predicates = cfg.plain_predicates
    match standard.lower():
        case "direct":
            ts_format = data_config.ts_format
            data = direct_load_plain_predicates(data_path, list(plain_predicates.keys()), ts_format)
        case "meds":
            data = generate_plain_predicates_from_meds(data_path, plain_predicates)
        case "esgpt":
            data = generate_plain_predicates_from_esgpt(data_path, plain_predicates)
        case _:
            raise ValueError(f"Invalid data standard: {standard}. Options are 'direct', 'MEDS', 'ESGPT'.")
    predicate_cols = list(plain_predicates.keys())

    data = data.sort(by=["subject_id", "timestamp"], nulls_last=False)

    # derived predicates
    logger.info("Loaded plain predicates. Generating derived predicate columns...")
    static_variables = [pred for pred in cfg.plain_predicates if cfg.plain_predicates[pred].static]
    for name, code in cfg.derived_predicates.items():
        if any(x in static_variables for x in code.input_predicates):
            data = data.with_columns(
                [
                    pl.col(static_var)
                    .first()
                    .over("subject_id")  # take the first value in each subject_id group and propagate it
                    .alias(static_var)
                    for static_var in static_variables
                ]
            )
        data = data.with_columns(code.eval_expr().cast(PRED_CNT_TYPE).alias(name))
        logger.info(f"Added predicate column '{name}'.")
        predicate_cols.append(name)

    # add special predicates:
    # a column of 1s representing any predicate
    # a column of 0s with 1 in the first event of each subject_id representing the start of record
    # a column of 0s with 1 in the last event of each subject_id representing the end of record
    logger.info("Generating special predicate columns...")
    special_predicates = []
    for window in cfg.windows.values():
        if ANY_EVENT_COLUMN in window.referenced_predicates and ANY_EVENT_COLUMN not in special_predicates:
            special_predicates.append(ANY_EVENT_COLUMN)
        if (
            START_OF_RECORD_KEY in window.constraint_predicates
            and START_OF_RECORD_KEY not in special_predicates
        ):
            special_predicates.append(START_OF_RECORD_KEY)
        if END_OF_RECORD_KEY in window.constraint_predicates and END_OF_RECORD_KEY not in special_predicates:
            special_predicates.append(END_OF_RECORD_KEY)

    if (
        cfg.trigger.predicate in [ANY_EVENT_COLUMN, START_OF_RECORD_KEY, END_OF_RECORD_KEY]
        and cfg.trigger.predicate not in special_predicates
    ):
        special_predicates.append(cfg.trigger.predicate)

    if ANY_EVENT_COLUMN in special_predicates:
        data = data.with_columns(
            pl.when(pl.col("timestamp").is_not_null())
            .then(pl.lit(1))
            .otherwise(pl.lit(None))
            .alias(ANY_EVENT_COLUMN)
            .cast(PRED_CNT_TYPE)
        )
        logger.info(f"Added predicate column '{ANY_EVENT_COLUMN}'.")
        predicate_cols.append(ANY_EVENT_COLUMN)
    if START_OF_RECORD_KEY in special_predicates:
        data = data.with_columns(
            [
                (pl.col("timestamp") == pl.col("timestamp").min().over("subject_id"))
                .cast(PRED_CNT_TYPE)
                .alias(START_OF_RECORD_KEY)
            ]
        )
        logger.info(f"Added predicate column '{START_OF_RECORD_KEY}'.")
    if END_OF_RECORD_KEY in special_predicates:
        data = data.with_columns(
            [
                (pl.col("timestamp") == pl.col("timestamp").max().over("subject_id"))
                .cast(PRED_CNT_TYPE)
                .alias(END_OF_RECORD_KEY)
            ]
        )
        logger.info(f"Added predicate column '{END_OF_RECORD_KEY}'.")
    predicate_cols += special_predicates

    return data
