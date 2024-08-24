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
  "0": |-2
    patient_id,time,code,numeric_value,text_value
    1,,GENDER//MALE,,
    1,,SNP//rs234567,,
    1,12/18/1960 11:03,MEDS_BIRTH,,
    1,08/02/1972 10:00,CLINIC_VISIT,,
    1,08/02/1972 10:00,ICD9CM//493.90,,
    1,08/02/1972 10:00,LOINC//8310-5,0.65,
    1,08/02/1972 10:00,VITALS//BP//SYSTOLIC,108,
    1,01/14/2020 15:14,ADMISSION//MEDICAL,,
    1,01/14/2020 15:18,VITALS//BP//SYSTOLIC,132,
    1,01/14/2020 15:18,VITALS//BP//DIASTOLIC,90,
    1,01/14/2020 15:18,VITALS//HR//BPM,121,
    1,01/14/2020 15:18,VITALS//WEIGHT//LBS,233.2,
    1,01/15/2020 10:04,VITALS//BP//SYSTOLIC,126,
    1,01/15/2020 10:04,VITALS//BP//DIASTOLIC,91,
    1,01/15/2020 10:04,VITALS//HR//BPM,85,
    1,01/16/2020 10:11,VITALS//BP//SYSTOLIC,135,
    1,01/16/2020 10:11,VITALS//BP//DIASTOLIC,88,
    1,01/16/2020 10:11,VITALS//HR//BPM,79,
    1,01/16/2020 13:02,LVEF//ECHO,0.24,
    1,01/17/2020 10:00,ICD9CM//428.9,,
    1,01/17/2020 10:00,DISCHARGE//HOME,,
    1,01/18/2022 04:46,ADMISSION//MEDICAL,,
    1,01/20/2022 08:00,DISCHARGE//HOME_AMA,,
    1,01/20/2022 08:00,ICD9CM//428.41,,
    1,01/20/2022 08:00,ICD9CM//451.1,,
    1,01/24/2022 08:11,ADMISSION//ED,,
    1,01/25/2022 10:04,VITALS//BP//SYSTOLIC,168,
    1,01/25/2022 10:04,VITALS//BP//DIASTOLIC,100,
    1,01/25/2022 10:04,VITALS//HR//BPM,56,
    1,02/27/2022 01:13,ICD9CM//428.41,,
    1,02/27/2022 01:13,ICD9CM//410.1,,
    1,02/27/2022 01:13,DEATH,,

  "1": |-2
    patient_id,time,code,numeric_value,text_value
    3,,GENDER//FEMALE,,
    3,,SNP//rs2345291,,
    3,,SNP//rs228192,,
    3,02/28/1982 00:00,MEDS_BIRTH,,
    3,01/14/2020 15:14,ADMISSION//MEDICAL,,
    3,01/14/2020 15:18,VITALS//BP//SYSTOLIC,132,
    3,01/14/2020 15:18,VITALS//BP//DIASTOLIC,90,
    3,01/14/2020 15:18,VITALS//HR//BPM,121,
    3,01/17/2020 10:00,ICD9CM//V30.00,,
    3,01/17/2020 10:00,DISCHARGE//HOME,,
    3,01/18/2020 18:18,ADMISSION//MEDICAL,,
    3,01/20/2020 15:18,DISCHARGE//HOME,,
    3,03/18/2024 16:54,ICD9CM//428.9,,
    3,03/18/2024 17:11,ADMISSION//SURGICAL,,
    3,03/28/2024 10:00,DISCHARGE//HOME,,
    3,03/29/2024 11:00,ADMISSION//SURGICAL,,
    3,04/19/2024 13:32,DISCHARGE//HOME,,
    3,05/22/2024 00:00,ICD9CM//428.9,,
    """,
    text_value=pl.Utf8,
)

# Tasks (input)
TASKS = {
    "inhospital_mortality": """
        predicates:
          admission:
            code: {regex: ADMISSION//.*}
          discharge:
            code: {regex: DISCHARGE//.*}
          death:
            code: DEATH
          discharge_or_death:
            expr: or(discharge, death)

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
        """,
    "HF_derived_readmission": """
        predicates:
          admission:
            code: {regex: ADMISSION//.*}
          discharge:
            code: {regex: DISCHARGE//.*}
          HF_dx:
            code: {regex: ICD9CM//428.*}

        trigger: discharge

        windows:
          data_within_5yr_of_admit:
            start: end - 1825d
            end: admission_is_HF.start
            start_inclusive: True
            end_inclusive: False
            has:
              _ANY_EVENT: (1, None)
          admission_is_HF:
            start: end <- admission
            end: trigger
            start_inclusive: True
            end_inclusive: True
            has:
              HF_dx: (1, None)
          input:
            start: NULL
            end: trigger
            start_inclusive: True
            end_inclusive: True
            index_timestamp: end
          target:
            start: input.end
            end: start + 30d
            start_inclusive: False
            end_inclusive: True
            label: admission
          censor_protection:
            start: target.end
            end: null
            start_inclusive: False
            end_inclusive: True
            has:
              _ANY_EVENT: (1, None)
    """,
}

WANT_SHARDS = {
    "inhospital_mortality": parse_labels_yaml(
        """
  "0": |-2
    patient_id,prediction_time,boolean_value,integer_value,float_value,categorical_value
    1,01/15/2020 15:14,0,,,
    1,01/19/2022 04:46,0,,,
    1,01/25/2022 08:11,1,,,

  "1": |-2
    patient_id,prediction_time,boolean_value,integer_value,float_value,categorical_value
    3,03/19/2024 17:11,0,,,
    3,03/30/2024 11:00,0,,,
    """
    ),
    "HF_derived_readmission": parse_labels_yaml(
        """
  "0": |-2
    patient_id,prediction_time,boolean_value,integer_value,float_value,categorical_value
    1,01/20/2022 08:00,1,,,

  "1": |-2
    patient_id,prediction_time,boolean_value,integer_value,float_value,categorical_value
    """
    ),
}


def test_meds():
    cli_test(
        input_files=MEDS_SHARDS,
        task_configs=TASKS,
        want_outputs_by_task=WANT_SHARDS,
        data_standard="meds",
    )
