"""Main script for end-to-end cohort extraction."""

import sys
from importlib.resources import files

import hydra
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger
from meds import label_schema
from omegaconf import DictConfig

config_yaml = files("aces").joinpath("configs/aces.yaml")

if len(sys.argv) == 1:
    print("Usage: aces-cli [OPTIONS]")
    print("Try 'aces-cli --help' for more information.")
    print("For more information, visit: https://eventstreamaces.readthedocs.io/en/latest/usage.html")
    sys.exit(1)

MEDS_LABEL_MANDATORY_TYPES = {
    "patient_id": pl.Int64,
}

MEDS_LABEL_OPTIONAL_TYPES = {
    "boolean_value": pl.Boolean,
    "integer_value": pl.Int64,
    "float_value": pl.Float64,
    "categorical_value": pl.String,
    "prediction_time": pl.Datetime("us"),
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
        >>> get_and_validate_label_schema(df) # doctest: +NORMALIZE_WHITESPACE
        Traceback (most recent call last):
            ...
        ValueError: MEDS Data DataFrame must have a 'patient_id' column of type Int64.
                    MEDS Data DataFrame must have a 'prediction_time' column of type String.
                        Datetime(time_unit='us', time_zone=None).
        >>> from datetime import datetime
        >>> df = pl.DataFrame({
        ...     "patient_id": pl.Series([1, 3, 2], dtype=pl.UInt32),
        ...     "time": [datetime(2021, 1, 1), datetime(2021, 1, 2), datetime(2021, 1, 3)],
        ...     "boolean_value": [1, 0, 100],
        ... })
        >>> get_and_validate_label_schema(df)
        pyarrow.Table
        patient_id: int64
        time: timestamp[us]
        boolean_value: bool
        integer_value: int64
        float_value: float
        categorical_value: string
        ----
        patient_id: [[1,3,2]]
        time: [[2021-01-01 00:00:00.000000,2021-01-02 00:00:00.000000,2021-01-03 00:00:00.000000]]
        boolean_value: [[true,false,true]]
        integer_value: [[null,null,null]]
        float_value: [[null,null,null]]
        categorical_value: [[null,null,null]]
    """

    schema = df.schema
    if "prediction_time" not in schema:
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
            errors.append(f"MEDS Data DataFrame must have a '{col}' column of type {dtype}.")

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

    df = df.select(
        "patient_id", "prediction_time", "boolean_value", "integer_value", "float_value", "categorical_value"
    )

    return df.to_arrow().cast(label_schema)


@hydra.main(version_base=None, config_path=str(config_yaml.parent.resolve()), config_name=config_yaml.stem)
def main(cfg: DictConfig):
    import os
    from datetime import datetime
    from pathlib import Path

    from hydra.core.hydra_config import HydraConfig
    from omegaconf import OmegaConf

    from . import config, predicates, query, utils

    utils.hydra_loguru_init(f"{HydraConfig.get().job.name}.log")

    st = datetime.now()

    # Set output path
    cohort_dir = Path(cfg.cohort_dir)
    cohort_dir.mkdir(exist_ok=True, parents=True)

    # load configuration
    logger.info(f"Loading config from '{cfg.config_path}'")
    task_cfg = config.TaskExtractorConfig.load(Path(cfg.config_path))

    logger.info(f"Attempting to get predicates dataframe given:\n{OmegaConf.to_yaml(cfg.data)}")
    predicates_df = predicates.get_predicates_df(task_cfg, cfg.data)

    # query results
    result = query.query(task_cfg, predicates_df)
    result_is_empty = len(result) == 0

    # save results to parquet
    os.makedirs(os.path.dirname(cfg.output_filepath), exist_ok=True)

    if cfg.data.standard.lower() == "meds":
        for in_col, out_col in [
            ("subject_id", "patient_id"),
            ("index_timestamp", "prediction_time"),
            ("label", "boolean_value"),
        ]:
            if in_col in result.columns:
                result = result.rename({in_col: out_col})
        if "patient_id" not in result.columns:
            if not result_is_empty:
                raise ValueError("Output dataframe is missing a 'patient_id' column.")
            else:
                logger.warning("Output dataframe is empty; adding an empty patient ID column.")
                result = result.with_columns(pl.lit(None, dtype=pl.Int64).alias("patient_id"))
                result = result.head(0)

        result = get_and_validate_label_schema(result)
        pq.write_table(result, cfg.output_filepath)
    else:
        result.write_parquet(cfg.output_filepath, use_pyarrow=True)
    logger.info(f"Completed in {datetime.now() - st}. Results saved to '{cfg.output_filepath}'.")


if __name__ == "__main__":
    main()
