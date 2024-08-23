"""Tests the full end-to-end extraction process."""


import rootutils

root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=True)

from io import StringIO

import polars as pl
import pyarrow as pa
from loguru import logger
from meds import label_schema
from yaml import load as load_yaml

from .utils import cli_test

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

pl.enable_string_cache()

TS_FORMAT = "%m/%d/%Y %H:%M"
PRED_CNT_TYPE = pl.Int64
EVENT_INDEX_TYPE = pl.UInt64
ANY_EVENT_COLUMN = "_ANY_EVENT"
LAST_EVENT_INDEX_COLUMN = "_LAST_EVENT_INDEX"

DEFAULT_CSV_TS_FORMAT = "%m/%d/%Y %H:%M"

# TODO: Make use meds library
MEDS_PL_SCHEMA = {
    "patient_id": pl.UInt32,
    "time": pl.Datetime("us"),
    "code": pl.Utf8,
    "numeric_value": pl.Float32,
    "numeric_value/is_inlier": pl.Boolean,
}


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


def parse_meds_csvs(
    csvs: str | dict[str, str], schema: dict[str, pl.DataType] = MEDS_PL_SCHEMA
) -> pl.DataFrame | dict[str, pl.DataFrame]:
    """Converts a string or dict of named strings to a MEDS DataFrame by interpreting them as CSVs.

    TODO: doctests.
    """

    default_read_schema = {**schema}
    default_read_schema["time"] = pl.Utf8

    def reader(csv_str: str) -> pl.DataFrame:
        cols = csv_str.strip().split("\n")[0].split(",")
        read_schema = {k: v for k, v in default_read_schema.items() if k in cols}
        return pl.read_csv(StringIO(csv_str), schema=read_schema).with_columns(
            pl.col("time").str.strptime(MEDS_PL_SCHEMA["time"], DEFAULT_CSV_TS_FORMAT)
        )

    if isinstance(csvs, str):
        return reader(csvs)
    else:
        return {k: reader(v) for k, v in csvs.items()}


def parse_shards_yaml(yaml_str: str, **schema_updates) -> dict[str, pl.DataFrame]:
    schema = {**MEDS_PL_SCHEMA, **schema_updates}
    return parse_meds_csvs(load_yaml(yaml_str, Loader=Loader), schema=schema)


def parse_labels_yaml(yaml_str: str) -> dict[str, pl.DataFrame]:
    dfs = {}
    for k, v in load_yaml(yaml_str, Loader=Loader).items():
        dfs[k] = pl.from_arrow(
            get_and_validate_label_schema(
                pl.read_csv(StringIO(v)).with_columns(
                    pl.col("prediction_time").str.strptime(pl.Datetime("us"), "%m/%d/%Y %H:%M")
                )
            )
        )
    return dfs


# Data (input)
MEDS_SHARDS = parse_shards_yaml(
    """
  "train/0": |-
    patient_id,time,code,numeric_value
    2,,SNP//rs234567,
    2,,SNP//rs345678,
    2,,GENDER//FEMALE,
    2,3/8/1996 2:22,ED_VISIT,
    2,3/8/1996 2:24,ADMISSION//SURGICAL,
    2,3/8/1996 2:24,TEMP//F,98.6
    2,3/8/1996 2:24,AOx3,
    2,3/8/1996 2:35,LAB//HR,110
    2,3/8/1996 2:35,LAB//HR,102
    2,3/8/1996 4:00,diagnosis//unk,
    2,3/8/1996 10:00,LAB//RBC,3.2
    2,3/8/1996 16:00,DISCHARGE//HOME,
    2,6/5/1996 0:32,ADMISSION//ED,
    2,6/5/1996 0:48,LAB//HR,9999
    2,6/5/1996 1:59,LAB//HR,102
    2,6/7/1996 6:00,LAB//HR,89
    2,6/7/1996 9:00,LAB//RR,25
    2,6/7/1996 12:00,VENTILATION_START,
    2,6/7/1996 12:00,LAB//SpO2,79.1
    2,6/7/1996 15:00,LAB//RR,40
    2,6/7/1996 15:00,LAB//HR,60
    2,6/8/1996 3:00,DEATH,

  "train/1": |-2
    patient_id,time,code,numeric_value
    4,,GENDER//MALE,
    4,,SNP//rs123456,
    4,12/1/1989 12:03,ADMISSION//CARDIAC,
    4,12/1/1989 12:03,diagnosis//ICD10CM//K79.8,
    4,12/1/1989 13:14,LAB//SPO2,98.6
    4,12/1/1989 15:17,LAB//SPO2,99.6
    4,12/1/1989 16:17,LAB//SPO2,98.9
    4,12/1/1989 20:17,LAB//SPO2,99.2
    4,12/2/1989 3:00,LAB//SPO2,99.1
    4,12/2/1989 9:00,LAB//HR,60
    4,12/2/1989 10:00,diagnosis//ICD9CM//403.2,
    4,12/2/1989 10:00,LAB//BUN,
    4,12/2/1989 14:22,CXR,
    4,12/2/1989 14:22,LAB//RR,40.2
    4,12/2/1989 15:00,DISCHARGE//HOME,
    4,1/21/1991 11:59,CLINIC_VISIT,
    4,1/27/1991 23:32,ADMISSION//ORTHO,
    4,1/27/1991 23:46,LAB//HR,60
    4,1/28/1991 3:18,LAB//HR,60
    4,1/28/1991 3:18,LAB//HR,62
    4,1/28/1991 3:28,LAB//HR,68
    4,1/28/1991 4:36,LAB//HR,70
    4,1/28/1991 4:36,LAB//SpO2,99.2
    4,1/29/1991 23:32,LAB//HR,60
    4,1/30/1991 5:00,diagnosis//ICD9CM//403.2,
    4,1/30/1991 8:00,LAB//HR,62
    4,1/30/1991 11:00,LAB//HR,59
    4,1/30/1991 14:00,LAB//HR,60
    4,1/30/1991 14:15,LAB//HR,60
    4,1/31/1991 1:00,LAB//HR,60
    4,1/31/1991 2:15,DISCHARGE//SNF,
    4,2/8/1991 8:15,OUTPATIENT_VISIT,
    4,3/3/1991 19:33,ADMISSION//ED,
    4,3/3/1991 20:33,LAB//HR,42
    4,3/3/1991 21:38,DEATH,
    6,,GENDER//MALE,
    6,,SNP//rs234567,
    6,,SNP//rs345678,
    6,3/8/1996 2:22,ED_VISIT,
    6,3/8/1996 2:24,ADMISSION//MEDICAL,
    6,3/8/1996 2:37,LAB//HR,60
    6,3/9/1996 8:00,LAB//HR,60
    6,3/9/1996 11:00,LAB//SpO2,99.2
    6,3/9/1996 19:00,LAB//RR,43
    6,3/9/1996 22:00,LAB//RR,40
    6,3/11/1996 21:00,LAB//HR,60
    6,3/12/1996 0:00,DEATH,

  "held_out/0/0": |-2
    patient_id,time,code,numeric_value
    3,,GENDER//FEMALE,
    3,,SNP//rs234567,
    3,,SNP//rs345678,
    3,3/8/1996 2:22,ED_VISIT,
    3,3/8/1996 2:24,ADMISSION//MEDICAL,
    3,3/8/1996 2:37,LAB//HR,60
    3,3/9/1996 8:00,LAB//HR,60
    3,3/9/1996 11:00,LAB//SpO2,99.2
    3,3/9/1996 19:00,LAB//RR,43
    3,3/9/1996 22:00,LAB//RR,40
    3,3/11/1996 21:00,LAB//HR,60
    3,3/12/1996 0:00,DEATH,

  "empty_shard": |-2
    patient_id,time,code,numeric_value

  "held_out": |-2
    patient_id,time,code,numeric_value
    1,,GENDER//MALE,
    1,,SNP//rs123456,
    1,12/1/1989 12:03,ADMISSION//CARDIAC,
    1,12/1/1989 12:03,diagnosis//ICD10CM//K79.8,
    1,12/1/1989 13:14,LAB//SPO2,98.6
    1,12/1/1989 15:17,LAB//SPO2,99.6
    1,12/1/1989 16:17,LAB//SPO2,98.9
    1,12/1/1989 20:17,LAB//SPO2,99.2
    1,12/2/1989 3:00,LAB//SPO2,99.1
    1,12/2/1989 9:00,LAB//HR,60
    1,12/2/1989 10:00,diagnosis//ICD9CM//403.2,
    1,12/2/1989 10:00,LAB//BUN,
    1,12/2/1989 14:22,CXR,
    1,12/2/1989 14:22,LAB//RR,40.2
    1,12/2/1989 15:00,DISCHARGE//HOME,
    1,1/21/1991 11:59,CLINIC_VISIT,
    1,1/27/1991 23:32,ADMISSION//ORTHO,
    1,1/27/1991 23:46,LAB//HR,60
    1,1/28/1991 3:18,LAB//HR,60
    1,1/28/1991 3:18,LAB//HR,62
    1,1/28/1991 3:28,LAB//HR,68
    1,1/28/1991 4:36,LAB//HR,70
    1,1/28/1991 4:36,LAB//SpO2,99.2
    1,1/29/1991 23:32,LAB//HR,60
    1,1/30/1991 5:00,diagnosis//ICD9CM//403.2,
    1,1/30/1991 8:00,LAB//HR,62
    1,1/30/1991 11:00,LAB//HR,59
    1,1/30/1991 14:00,LAB//HR,60
    1,1/30/1991 14:15,LAB//HR,60
    1,1/31/1991 1:00,LAB//HR,60
    1,1/31/1991 2:15,DISCHARGE//SNF,
    1,2/8/1991 8:15,OUTPATIENT_VISIT,
    1,3/3/1991 19:33,ADMISSION//ED,
    1,3/3/1991 20:33,LAB//HR,42
    1,3/3/1991 21:38,DEATH,
    """
)

# Tasks (input)
TASK_NAME = "inhospital_mortality"
TASK_CFG = """
# Task: 24-hour In-hospital Mortality Prediction
predicates:
  admission:
    code: {regex: ADMISSION.*}
  discharge:
    code: {regex: DISCHARGE.*}
  death:
    code: DEATH
  discharge_or_death:
    expr: or(discharge, death)

patient_demographics:
  male:
    code: GENDER//MALE

trigger: admission

windows:
  input:
    start: NULL
    end: trigger + 24h
    start_inclusive: True
    end_inclusive: True
    has:
      _ANY_EVENT: (5, None)
    index_timestamp: end
  gap:
    start: trigger
    end: start + 48h
    start_inclusive: False
    end_inclusive: True
    has:
      admission: (None, 0)
      discharge_or_death: (None, 0)
  target:
    start: gap.end
    end: start -> discharge_or_death
    start_inclusive: False
    end_inclusive: True
    label: death
"""

WANT_SHARDS = parse_labels_yaml(
    """
  "train/0": |-2
    patient_id,prediction_time,boolean_value,integer_value,float_value,categorical_value

  "train/1": |-2
    patient_id,prediction_time,boolean_value,integer_value,float_value,categorical_value
    4,1/28/1991 23:32,False,,,,

  "held_out/0/0": |-2
    patient_id,prediction_time,boolean_value,integer_value,float_value,categorical_value

  "empty_shard": |-2
    patient_id,prediction_time,boolean_value,integer_value,float_value,categorical_value

  "held_out": |-2
    patient_id,prediction_time,boolean_value,integer_value,float_value,categorical_value
    1,1/28/1991 23:32,False,,,,
    """
)


def test_meds():
    cli_test(
        input_files=MEDS_SHARDS,
        task_configs={TASK_NAME: TASK_CFG},
        want_outputs_by_task={TASK_NAME: WANT_SHARDS},
        data_standard="meds",
    )
