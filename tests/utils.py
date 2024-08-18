"""Test utilities."""

import rootutils

root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=True)

import subprocess

import polars as pl
import pyarrow as pa
from loguru import logger
from meds import label_schema
from polars.testing import assert_frame_equal

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


def run_command(script: str, hydra_kwargs: dict[str, str], test_name: str, expected_returncode: int = 0):
    command_parts = [script] + [f"{k}={v}" for k, v in hydra_kwargs.items()]
    cmd = " ".join(command_parts)
    logger.info(f"Running {test_name}: {cmd}")
    command_out = subprocess.run(cmd, shell=True, capture_output=True)
    stderr = command_out.stderr.decode()
    stdout = command_out.stdout.decode()
    if command_out.returncode != expected_returncode:
        raise AssertionError(
            f"{test_name} returned {command_out.returncode} (expected {expected_returncode})!\n"
            f"stdout:\n{stdout}\nstderr:\n{stderr}"
        )
    return stderr, stdout


def assert_df_equal(want: pl.DataFrame, got: pl.DataFrame, msg: str = None, **kwargs):
    try:
        assert_frame_equal(want, got, **kwargs)
    except AssertionError as e:
        pl.Config.set_tbl_rows(-1)
        print(f"DFs are not equal: {msg}\nWant:")
        print(want)
        print("Got:")
        print(got)
        raise AssertionError(f"{msg}\n{e}") from e


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
