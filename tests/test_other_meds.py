"""Tests the full end-to-end extraction process."""


import rootutils

root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=True)


import polars as pl

from .test_meds import parse_labels_yaml, parse_shards_yaml
from .utils import cli_test

pl.enable_string_cache()

# Data (input)
MEDS_SHARDS = parse_shards_yaml(
    """
  "0": |-2
    subject_id,time,code,numeric_value,text_value
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
    subject_id,time,code,numeric_value,text_value
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
    subject_id,prediction_time,boolean_value,integer_value,float_value,categorical_value
    1,01/15/2020 15:14,0,,,
    1,01/19/2022 04:46,0,,,
    1,01/25/2022 08:11,1,,,

  "1": |-2
    subject_id,prediction_time,boolean_value,integer_value,float_value,categorical_value
    3,03/19/2024 17:11,0,,,
    3,03/30/2024 11:00,0,,,
    """
    ),
    "HF_derived_readmission": parse_labels_yaml(
        """
  "0": |-2
    subject_id,prediction_time,boolean_value,integer_value,float_value,categorical_value
    1,01/20/2022 08:00,1,,,

  "1": |-2
    subject_id,prediction_time,boolean_value,integer_value,float_value,categorical_value
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
