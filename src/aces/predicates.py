"""This module contains functions for generating predicate columns for event sequences."""

import dataclasses
import logging
from pathlib import Path

import meds
import polars as pl
from omegaconf import DictConfig
from polars.exceptions import ColumnNotFoundError

from .config import DerivedPredicateConfig, NestedPredicateConfig, PlainPredicateConfig, TaskExtractorConfig
from .query import query
from .types import (
    ANY_EVENT_COLUMN,
    END_OF_RECORD_KEY,
    PRED_CNT_TYPE,
    START_OF_RECORD_KEY,
)

logger = logging.getLogger(__name__)

SPECIAL_PREDICATES = (ANY_EVENT_COLUMN, START_OF_RECORD_KEY, END_OF_RECORD_KEY)


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


def expand_descendant_codes(
    predicates: dict[str, PlainPredicateConfig], data_path: Path
) -> dict[str, PlainPredicateConfig]:
    """Rewrites predicates with ``code: {descendants_of: [...]}`` to ``code: {any: [...]}``.

    The descendants of the listed codes (which are included themselves) are found by following the
    ``parent_codes`` links in the MEDS code metadata file, ``metadata/codes.parquet``, of the first parent
    directory of ``data_path`` that has one.

    Args:
        predicates: The dictionary of plain predicate configurations.
        data_path: The path to the MEDS data file (or shard) the predicates will be evaluated on.

    Returns:
        The predicates, with every ``descendants_of`` code replaced by the equivalent ``any`` list.

    Raises:
        ValueError: If ``descendants_of`` is not a non-empty list of codes.
        FileNotFoundError: If a predicate uses ``descendants_of`` but no code metadata file is found.

    Examples:
        >>> codes = pl.DataFrame({
        ...     "code": ["ICD10//I21", "ICD10//I21.0", "ICD10//I21.01", "ICD9//410", "ICD10//I50"],
        ...     "parent_codes": [None, ["ICD10//I21"], ["ICD10//I21.0"], ["ICD10//I21"], None],
        ... })
        >>> predicates = {
        ...     "mi": PlainPredicateConfig({"descendants_of": ["ICD10//I21"]}),
        ...     "hf": PlainPredicateConfig("ICD10//I50"),
        ... }
        >>> with tempfile.TemporaryDirectory() as root:
        ...     (Path(root) / "metadata").mkdir()
        ...     codes.write_parquet(Path(root) / "metadata" / "codes.parquet")
        ...     expanded = expand_descendant_codes(predicates, Path(root) / "data" / "train" / "0.parquet")
        >>> expanded["mi"].code
        {'any': ['ICD10//I21', 'ICD10//I21.0', 'ICD10//I21.01', 'ICD9//410']}
        >>> expanded["hf"].code
        'ICD10//I50'
        >>> with tempfile.TemporaryDirectory() as root:
        ...     expand_descendant_codes(predicates, Path(root) / "data" / "0.parquet")
        Traceback (most recent call last):
            ...
        FileNotFoundError: Predicate(s) 'mi' use 'descendants_of', which needs MEDS code metadata, but
        no 'metadata/codes.parquet' was found in any parent directory of .../data/0.parquet.
        >>> expand_descendant_codes({"mi": PlainPredicateConfig({"descendants_of": "ICD10//I21"})}, Path("."))
        Traceback (most recent call last):
            ...
        ValueError: Invalid specification in the code field! Got: {'descendants_of': 'ICD10//I21'}.
        Expected a list of strings for 'descendants_of'.
    """
    to_expand = {
        name: pred
        for name, pred in predicates.items()
        if isinstance(pred.code, dict) and "descendants_of" in pred.code
    }
    if not to_expand:
        return predicates

    for pred in to_expand.values():
        if not pred.code["descendants_of"] or not isinstance(pred.code["descendants_of"], list):
            raise ValueError(
                f"Invalid specification in the code field! Got: {pred.code}. "
                "Expected a list of strings for 'descendants_of'."
            )

    codes_fp = next(
        (
            d / "metadata" / "codes.parquet"
            for d in data_path.resolve().parents
            if (d / "metadata" / "codes.parquet").is_file()
        ),
        None,
    )
    if codes_fp is None:
        raise FileNotFoundError(
            f"Predicate(s) '{', '.join(to_expand)}' use 'descendants_of', which needs MEDS code metadata, "
            f"but no 'metadata/codes.parquet' was found in any parent directory of {data_path}."
        )

    edges = pl.read_parquet(codes_fp, columns=["code", "parent_codes"]).explode("parent_codes")
    expanded = dict(predicates)
    for name, pred in to_expand.items():
        codes = set(pred.code["descendants_of"])
        new_codes = codes
        while new_codes:
            children = edges.filter(pl.col("parent_codes").is_in(list(new_codes)))["code"]
            new_codes = set(children) - codes
            codes |= new_codes
        logger.info(f"Expanded predicate '{name}' to {len(codes)} codes using {codes_fp}.")
        expanded[name] = dataclasses.replace(pred, code={"any": sorted(codes)})
    return expanded


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
    predicates = expand_descendant_codes(predicates, data_path)

    # Cast `code` to Utf8 once up front. Older MEDS schemas sometimes emitted `code` as
    # a categorical / enum dtype, and both the birth-row filter below and the per-predicate
    # expression loop assume a string column. (Previously the cast was repeated once per
    # predicate inside the loop; hoisting it avoids that redundant work.)
    data = data.with_columns(pl.col("code").cast(pl.Utf8))

    # Drop MEDS birth rows before predicate evaluation. The birth code is a
    # quasi-static demographic marker anchored at the patient's birth timestamp;
    # it shouldn't count toward predicate matches or the implicit _ANY_EVENT
    # column. Static rows (null timestamp) are already excluded; this extends
    # the same treatment to MEDS_BIRTH. See issue #168.
    data = data.filter(pl.col("code") != meds.birth_code)

    # generate plain predicate columns
    logger.info("Generating plain predicate columns...")
    for name, plain_predicate in predicates.items():
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


def get_special_predicates(cfg: TaskExtractorConfig) -> list[str]:
    """Returns the special predicates that need a column to evaluate ``cfg``, including its nested predicates.

    Windows bounded by the start or end of the record do not need a column, as those bounds are found from
    the timestamps directly.

    Examples:
        >>> from .config import EventConfig, WindowConfig
        >>> windows = {"input": WindowConfig(None, "trigger", True, True, has={"_RECORD_END": "(None, 0)"})}
        >>> get_special_predicates(TaskExtractorConfig(
        ...     predicates={}, trigger=EventConfig("_ANY_EVENT"), windows=windows
        ... ))
        ['_ANY_EVENT', '_RECORD_END']
        >>> nested = NestedPredicateConfig(
        ...     trigger=EventConfig("_RECORD_START"),
        ...     windows={"w": WindowConfig("trigger", "start -> _ANY_EVENT", False, True)},
        ... )
        >>> get_special_predicates(TaskExtractorConfig(
        ...     predicates={"first_event": nested}, trigger=EventConfig("first_event"), windows={}
        ... ))
        ['_ANY_EVENT', '_RECORD_START']
    """
    tasks = [cfg, *(p.task for p in cfg.predicates.values() if isinstance(p, NestedPredicateConfig))]

    special_predicates = set()
    for task in tasks:
        for window in task.windows.values():
            if ANY_EVENT_COLUMN in window.referenced_predicates:
                special_predicates.add(ANY_EVENT_COLUMN)
            special_predicates |= {START_OF_RECORD_KEY, END_OF_RECORD_KEY} & window.constraint_predicates
        if task.trigger.predicate in SPECIAL_PREDICATES:
            special_predicates.add(task.trigger.predicate)

    return [p for p in SPECIAL_PREDICATES if p in special_predicates]


def add_special_predicates(data: pl.DataFrame, special_predicates: list[str]) -> pl.DataFrame:
    """Adds columns for the given special predicates to ``data``.

    These are a column of 1s representing any event (null for static rows), a column of 0s with 1 in the first
    event of each subject representing the start of record, and a column of 0s with 1 in the last event of
    each subject representing the end of record.
    """
    if ANY_EVENT_COLUMN in special_predicates:
        data = data.with_columns(
            pl.when(pl.col("timestamp").is_not_null())
            .then(pl.lit(1))
            .otherwise(pl.lit(None))
            .alias(ANY_EVENT_COLUMN)
            .cast(PRED_CNT_TYPE)
        )
        logger.info(f"Added predicate column '{ANY_EVENT_COLUMN}'.")
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
    return data


def add_nested_predicate(data: pl.DataFrame, name: str, predicate: NestedPredicateConfig) -> pl.DataFrame:
    """Adds the column of a nested predicate to ``data``, by extracting the nested predicate's task from it.

    The nested predicate occurs at the index timestamp of each valid realization of its task (or at its
    trigger, if no window sets an ``index_timestamp``). Its count at a timestamp is the number of realizations
    indexed there or, if the task has a ``label``, the sum of their labels. An index timestamp that is not the
    time of an existing event is added as a new event with all other predicate counts zero, so that it does
    not change any window's counts, unless it is outside the subject's record, in which case it is dropped.

    Args:
        data: The predicates dataframe, sorted by subject and timestamp, containing (at least) the columns of
            the predicates and special predicates the nested predicate uses.
        name: The name of the nested predicate.
        predicate: The nested predicate configuration.

    Returns:
        ``data`` with the column ``name`` added, sorted by subject and timestamp.

    Examples:
        >>> from .config import EventConfig, WindowConfig
        >>> data = pl.DataFrame({
        ...     "subject_id": [1, 1, 1, 1, 1, 1, 2, 2],
        ...     "timestamp": [None, *(datetime(2020, 1, d) for d in (1, 2, 3, 10, 11)),
        ...                   datetime(2020, 1, 1), datetime(2020, 1, 2)],
        ...     "adm": [0, 1, 0, 0, 1, 0, 1, 0],
        ...     "hf":  [0, 0, 1, 0, 0, 0, 1, 0],
        ...     "dis": [0, 0, 0, 1, 0, 1, 0, 1],
        ... })
        >>> hf_hospitalization = NestedPredicateConfig(
        ...     trigger=EventConfig("adm"),
        ...     windows={"stay": WindowConfig(
        ...         "trigger", "start -> dis", True, True, has={"hf": "(1, None)"}, index_timestamp="start",
        ...     )},
        ... )
        >>> add_nested_predicate(data, "hf_hosp", hf_hospitalization)
        shape: (8, 6)
        ┌────────────┬─────────────────────┬─────┬─────┬─────┬─────────┐
        │ subject_id ┆ timestamp           ┆ adm ┆ hf  ┆ dis ┆ hf_hosp │
        │ ---        ┆ ---                 ┆ --- ┆ --- ┆ --- ┆ ---     │
        │ i64        ┆ datetime[μs]        ┆ i64 ┆ i64 ┆ i64 ┆ i64     │
        ╞════════════╪═════════════════════╪═════╪═════╪═════╪═════════╡
        │ 1          ┆ null                ┆ 0   ┆ 0   ┆ 0   ┆ 0       │
        │ 1          ┆ 2020-01-01 00:00:00 ┆ 1   ┆ 0   ┆ 0   ┆ 1       │
        │ 1          ┆ 2020-01-02 00:00:00 ┆ 0   ┆ 1   ┆ 0   ┆ 0       │
        │ 1          ┆ 2020-01-03 00:00:00 ┆ 0   ┆ 0   ┆ 1   ┆ 0       │
        │ 1          ┆ 2020-01-10 00:00:00 ┆ 1   ┆ 0   ┆ 0   ┆ 0       │
        │ 1          ┆ 2020-01-11 00:00:00 ┆ 0   ┆ 0   ┆ 1   ┆ 0       │
        │ 2          ┆ 2020-01-01 00:00:00 ┆ 1   ┆ 1   ┆ 0   ┆ 1       │
        │ 2          ┆ 2020-01-02 00:00:00 ┆ 0   ┆ 0   ┆ 1   ┆ 0       │
        └────────────┴─────────────────────┴─────┴─────┴─────┴─────────┘

        With a label, the nested predicate counts the label instead. Here, it occurs 60 hours after each
        admission, with its count being the number of discharges in that time. For subject 1's first
        admission, that time is not an existing event, so it is added as a new one; the other two are after
        the end of their subject's record, so they are dropped:

        >>> discharges_60h_after_adm = NestedPredicateConfig(
        ...     trigger=EventConfig("adm"),
        ...     windows={"after": WindowConfig(
        ...         "trigger", "start + 60h", False, True, label="dis", index_timestamp="end",
        ...     )},
        ... )
        >>> with caplog.at_level(logging.WARNING):
        ...     add_nested_predicate(data, "dis_60h", discharges_60h_after_adm)
        shape: (9, 6)
        ┌────────────┬─────────────────────┬─────┬─────┬─────┬─────────┐
        │ subject_id ┆ timestamp           ┆ adm ┆ hf  ┆ dis ┆ dis_60h │
        │ ---        ┆ ---                 ┆ --- ┆ --- ┆ --- ┆ ---     │
        │ i64        ┆ datetime[μs]        ┆ i64 ┆ i64 ┆ i64 ┆ i64     │
        ╞════════════╪═════════════════════╪═════╪═════╪═════╪═════════╡
        │ 1          ┆ null                ┆ 0   ┆ 0   ┆ 0   ┆ 0       │
        │ 1          ┆ 2020-01-01 00:00:00 ┆ 1   ┆ 0   ┆ 0   ┆ 0       │
        │ 1          ┆ 2020-01-02 00:00:00 ┆ 0   ┆ 1   ┆ 0   ┆ 0       │
        │ 1          ┆ 2020-01-03 00:00:00 ┆ 0   ┆ 0   ┆ 1   ┆ 0       │
        │ 1          ┆ 2020-01-03 12:00:00 ┆ 0   ┆ 0   ┆ 0   ┆ 1       │
        │ 1          ┆ 2020-01-10 00:00:00 ┆ 1   ┆ 0   ┆ 0   ┆ 0       │
        │ 1          ┆ 2020-01-11 00:00:00 ┆ 0   ┆ 0   ┆ 1   ┆ 0       │
        │ 2          ┆ 2020-01-01 00:00:00 ┆ 1   ┆ 1   ┆ 0   ┆ 0       │
        │ 2          ┆ 2020-01-02 00:00:00 ┆ 0   ┆ 0   ┆ 1   ┆ 0       │
        └────────────┴─────────────────────┴─────┴─────┴─────┴─────────┘
        >>> "Dropped 2 occurrences of nested predicate 'dis_60h'" in caplog.text
        True
    """
    task = predicate.task
    input_cols = [*predicate.input_predicates, *(c for c in SPECIAL_PREDICATES if c in data.columns)]
    realizations = query(task, data.select("subject_id", "timestamp", *input_cols))

    if realizations.is_empty():
        logger.info(f"Added predicate column '{name}', which never occurs.")
        return data.with_columns(pl.lit(0, dtype=PRED_CNT_TYPE).alias(name))

    timestamp_col = "index_timestamp" if task.index_timestamp_window else "trigger"
    count = pl.col("label").sum() if task.label_window else pl.len()
    counts = realizations.group_by("subject_id", pl.col(timestamp_col).alias("timestamp")).agg(
        count.cast(PRED_CNT_TYPE).alias(name)
    )

    record_bounds = data.group_by("subject_id").agg(
        pl.col("timestamp").min().alias("record_start"), pl.col("timestamp").max().alias("record_end")
    )
    counts = counts.join(record_bounds, on="subject_id")
    in_record = pl.col("timestamp").is_between("record_start", "record_end")
    n_outside_record = counts.filter(~in_record).height
    if n_outside_record:
        logger.warning(
            f"Dropped {n_outside_record:,} occurrences of nested predicate '{name}' that fall outside their "
            "subject's record."
        )
    counts = counts.filter(in_record).drop("record_start", "record_end")

    keys = ["subject_id", "timestamp"]
    data = data.join(counts, on=keys, how="left").with_columns(pl.col(name).fill_null(0))
    new_events = counts.join(data, on=keys, how="anti").select(
        *(pl.col(c) if c in counts.columns else pl.lit(0).cast(t).alias(c) for c, t in data.schema.items())
    )
    logger.info(f"Added predicate column '{name}', with {new_events.height:,} new events at its timestamps.")
    return pl.concat([data, new_events], how="vertical_relaxed").sort(
        by=["subject_id", "timestamp"], nulls_last=False
    )


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

    # Special predicates are added first, as nested predicates may use them, but are kept as the last columns.
    # They must reflect only real events, so they cannot be added after nested predicates add new events.
    logger.info("Generating special predicate columns...")
    special_predicates = get_special_predicates(cfg)
    data = add_special_predicates(data, special_predicates)

    # derived and nested predicates
    logger.info("Loaded plain predicates. Generating derived predicate columns...")
    static_variables = [pred for pred in cfg.plain_predicates if cfg.plain_predicates[pred].static]
    for name, code in cfg.derived_predicates.items():
        match code:
            case DerivedPredicateConfig():
                if any(x in static_variables for x in code.input_predicates):
                    data = data.with_columns(
                        [
                            pl.col(static_var)
                            .first()
                            .over("subject_id")  # take the first value in each subject_id group and propagate
                            .alias(static_var)
                            for static_var in static_variables
                        ]
                    )
                data = data.with_columns(code.eval_expr().cast(PRED_CNT_TYPE).alias(name))
                logger.info(f"Added predicate column '{name}'.")
            case NestedPredicateConfig():
                data = add_nested_predicate(data, name, code)
        predicate_cols.append(name)

    return data.select("subject_id", "timestamp", *predicate_cols, *special_predicates)
