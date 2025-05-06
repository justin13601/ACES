"""Main script for end-to-end cohort extraction."""

import logging
import os
import sys
from datetime import datetime
from importlib.resources import files
from pathlib import Path

import hydra
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from meds import LabelSchema
from omegaconf import DictConfig, OmegaConf

from . import config, predicates, query

logger = logging.getLogger(__name__)
config_yaml = files("aces").joinpath("configs/_aces.yaml")

MEDS_LABEL_MANDATORY_TYPES = {
    LabelSchema.subject_id_name: pl.Int64,
}

MEDS_LABEL_OPTIONAL_TYPES = {
    LabelSchema.prediction_time_name: pl.Datetime("us"),
    LabelSchema.boolean_value_name: pl.Boolean,
    LabelSchema.integer_value_name: pl.Int64,
    LabelSchema.float_value_name: pl.Float64,
    LabelSchema.categorical_value_name: pl.String,
}


def get_and_validate_label_schema(df: pl.DataFrame) -> pa.Table:
    """Validates the schema of a MEDS data DataFrame.

    This function validates the schema of a MEDS label DataFrame, ensuring that it has the correct columns
    and that the columns are of the correct type. This function will:
      1. Re-type any of the mandator MEDS column to the appropriate type.
      2. Attempt to add the ``numeric_value`` or ``time`` columns if either are missing, and set it to `None`.
         It will not attempt to add any other missing columns even if ``do_retype`` is `True` as the other
         columns cannot be set to `None`.

    Args:
        df: The MEDS label DataFrame to validate.

    Returns:
        pa.Table: The validated MEDS data DataFrame, with columns re-typed as needed.

    Raises:
        ValueError: if do_retype is False and the MEDS data DataFrame is not schema compliant.

    Examples:
        >>> df = pl.DataFrame({})
        >>> get_and_validate_label_schema(df)
        Traceback (most recent call last):
            ...
        ValueError: MEDS Label DataFrame must have a 'subject_id' column of type Int64.
        >>> df = pl.DataFrame({
        ...     "subject_id": pl.Series([1, 3, 2], dtype=pl.UInt32),
        ...     "time": [datetime(2021, 1, 1), datetime(2021, 1, 2), datetime(2021, 1, 3)],
        ...     "boolean_value": [1, 0, 100],
        ... })
        >>> get_and_validate_label_schema(df)
        pyarrow.Table
        subject_id: int64
        prediction_time: timestamp[us]
        boolean_value: bool
        integer_value: int64
        float_value: float
        categorical_value: string
        ----
        subject_id: [[1,3,2]]
        prediction_time: [[null,null,null]]
        boolean_value: [[true,false,true]]
        integer_value: [[null,null,null]]
        float_value: [[null,null,null]]
        categorical_value: [[null,null,null]]
    """

    schema = df.schema
    if LabelSchema.prediction_time_name not in schema:
        logger.warning(
            "Output DataFrame is missing a 'prediction_time' column. If this is not intentional, add a "
            "'index_timestamp' (yes, it should be different) key to the task configuration identifying "
            "which window's start or end time to use as the prediction time."
        )

    errors = []
    for col, dtype in MEDS_LABEL_MANDATORY_TYPES.items():
        if col in schema and schema[col] != dtype:
            df = df.with_columns(pl.col(col).cast(dtype, strict=False))
        elif col not in schema:
            errors.append(f"MEDS Label DataFrame must have a '{col}' column of type {dtype}.")

    if errors:
        raise ValueError("\n".join(errors))

    for col, dtype in MEDS_LABEL_OPTIONAL_TYPES.items():
        if col in schema and schema[col] != dtype:
            df = df.with_columns(pl.col(col).cast(dtype, strict=False))
        elif col not in schema:
            df = df.with_columns(pl.lit(None, dtype=dtype).alias(col))

    extra_cols = [
        c for c in schema if c not in MEDS_LABEL_MANDATORY_TYPES and c not in MEDS_LABEL_OPTIONAL_TYPES
    ]
    if extra_cols:
        err_cols_str = "\n".join(f"  - {c}" for c in extra_cols)
        logger.warning(
            "Output contains columns that are not valid MEDS label columns. For now, we are dropping them.\n"
            "If you need these columns, please comment on https://github.com/justin13601/ACES/issues/97\n"
            f"Columns:\n{err_cols_str}"
        )
        df = df.drop(extra_cols)

    return LabelSchema.align(df.to_arrow())


@hydra.main(version_base=None, config_path=str(config_yaml.parent.resolve()), config_name=config_yaml.stem)
def main(cfg: DictConfig) -> None:  # pragma: no cover
    st = datetime.now()

    # load configuration
    logger.info(f"Loading config from '{cfg.config_path}'")
    if cfg.predicates_path:
        logger.info(f"Overriding predicates and/or demographics from '{cfg.predicates_path}'")
        predicates_path = Path(cfg.predicates_path)
    else:
        predicates_path = None
    task_cfg = config.TaskExtractorConfig.load(
        config_path=Path(cfg.config_path), predicates_path=predicates_path
    )

    logger.info(f"Attempting to get predicates dataframe given:\n{OmegaConf.to_yaml(cfg.data)}")
    predicates_df = predicates.get_predicates_df(task_cfg, cfg.data)

    # query results
    result = query.query(task_cfg, predicates_df)
    result_is_empty = len(result) == 0

    # save results to parquet
    os.makedirs(os.path.dirname(cfg.output_filepath), exist_ok=True)

    if cfg.data.standard.lower() == "meds":
        for in_col, out_col in [
            ("subject_id", LabelSchema.subject_id_name),
            ("index_timestamp", LabelSchema.prediction_time_name),
            ("label", LabelSchema.boolean_value_name),
        ]:
            if in_col in result.columns:
                result = result.rename({in_col: out_col})
        if LabelSchema.subject_id_name not in result.columns:
            if not result_is_empty:
                raise ValueError("Output dataframe is missing a 'subject_id' column.")
            else:
                logger.warning("Output dataframe is empty; adding an empty patient ID column.")
                result = result.with_columns(pl.lit(None, dtype=pl.Int64).alias(LabelSchema.subject_id_name))
                result = result.head(0)
        if cfg.window_stats_dir:
            Path(cfg.window_stats_filepath).parent.mkdir(exist_ok=True, parents=True)
            result.write_parquet(cfg.window_stats_filepath)
        result = get_and_validate_label_schema(result)
        pq.write_table(result, cfg.output_filepath)
    else:
        result.write_parquet(cfg.output_filepath, use_pyarrow=True)
    logger.info(f"Completed in {datetime.now() - st}. Results saved to '{cfg.output_filepath}'.")


def cli():
    """Main entry point for the script, allowing for no-arg help messages."""

    if len(sys.argv) == 1:
        print("Usage: aces-cli [OPTIONS]")
        print("Try 'aces-cli --help' for more information.")
        print("For more information, visit: https://eventstreamaces.readthedocs.io/en/latest/usage.html")
        sys.exit(1)

    main()
