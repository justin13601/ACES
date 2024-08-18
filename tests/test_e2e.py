"""Tests the full end-to-end extraction process."""

import rootutils

root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=True)

import tempfile
from pathlib import Path

import polars as pl
from loguru import logger

from .utils import assert_df_equal, run_command

pl.enable_string_cache()

TS_FORMAT = "%m/%d/%Y %H:%M"
PRED_CNT_TYPE = pl.Int64
EVENT_INDEX_TYPE = pl.UInt64
ANY_EVENT_COLUMN = "_ANY_EVENT"
LAST_EVENT_INDEX_COLUMN = "_LAST_EVENT_INDEX"


# Data (input)
PREDICATES_CSV = """
subject_id,timestamp,male,female,admission,death,discharge,lab,spo2,normal_spo2,abnormally_low_spo2,abnormally_high_spo2,procedure_start,procedure_end,ventilation,diagnosis_ICD9CM_41071,diagnosis_ICD10CM_I214
1,,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0
1,12/1/1989 12:03,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0
1,12/1/1989 13:14,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0
1,12/1/1989 15:17,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0
1,12/1/1989 16:17,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0
1,12/1/1989 20:17,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0
1,12/2/1989 3:00,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0
1,12/2/1989 9:00,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
1,12/2/1989 10:00,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0
1,12/2/1989 14:22,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0
1,12/2/1989 15:00,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0
1,1/21/1991 11:59,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0
1,1/27/1991 23:32,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0
1,1/27/1991 23:46,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0
1,1/28/1991 3:18,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0
1,1/28/1991 3:28,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0
1,1/28/1991 4:36,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0
1,1/29/1991 23:32,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0
1,1/30/1991 5:00,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0
1,1/30/1991 8:00,0,0,0,0,0,1,1,0,0,1,0,0,0,0,0
1,1/30/1991 11:00,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0
1,1/30/1991 14:00,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
1,1/30/1991 14:15,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0
1,1/31/1991 1:00,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0
1,1/31/1991 2:15,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0
1,2/8/1991 8:15,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0
1,3/3/1991 19:33,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0
1,3/3/1991 20:33,0,0,0,0,0,1,1,0,1,0,0,0,0,0,0
1,3/3/1991 21:38,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0
2,,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0
2,3/8/1996 2:24,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0
2,3/8/1996 2:35,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0
2,3/8/1996 4:00,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0
2,3/8/1996 10:00,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
2,3/8/1996 16:00,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0
2,6/5/1996 0:32,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0
2,6/5/1996 0:48,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1
2,6/5/1996 1:59,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0
2,6/7/1996 6:00,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0
2,6/7/1996 9:00,0,0,0,0,0,1,1,0,1,0,0,0,0,0,0
2,6/7/1996 12:00,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0
2,6/7/1996 15:00,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0
2,6/7/1996 15:00,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
2,6/8/1996 3:00,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0
3,,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0
3,3/8/1996 2:22,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0
3,3/8/1996 2:24,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0
3,3/8/1996 2:37,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0
3,3/9/1996 8:00,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0
3,3/9/1996 11:00,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0
3,3/9/1996 19:00,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0
3,3/9/1996 22:00,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
3,3/11/1996 21:00,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0
3,3/12/1996 0:00,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0
"""

# Tasks (input)
TASKS_CFGS = {
    "inhospital_mortality": """
      # Task: 24-hour In-hospital Mortality Prediction
      predicates:
        admission:
          code: event_type//ADMISSION
        discharge:
          code: event_type//DISCHARGE
        death:
          code: event_type//DEATH
        discharge_or_death:
          expr: or(discharge, death)

      patient_demographics:
        male:
          code: SEX//male

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
            discharge: (None, 0)
            death: (None, 0)
        target:
          start: gap.end
          end: start -> discharge_or_death
          start_inclusive: False
          end_inclusive: True
          label: death
      """
}

# Expected output
EXPECTED_OUTPUT = {
    "inhospital_mortality": {
        "subject_id": [1],
        "index_timestamp": ["01/28/1991 23:32"],
        "label": [0],
        "trigger": ["01/27/1991 23:32"],
        "input.end_summary": [
            {
                "window_name": "input.end",
                "timestamp_at_start": "01/27/1991 23:32",
                "timestamp_at_end": "01/28/1991 23:32",
                "admission": 0,
                "discharge": 0,
                "death": 0,
                "discharge_or_death": 0,
                "_ANY_EVENT": 4,
            },
        ],
        "input.start_summary": [
            {
                "window_name": "input.start",
                "timestamp_at_start": "12/01/1989 12:03",
                "timestamp_at_end": "01/28/1991 23:32",
                "admission": 2,
                "discharge": 1,
                "death": 0,
                "discharge_or_death": 1,
                "_ANY_EVENT": 16,
            },
        ],
        "gap.end_summary": [
            {
                "window_name": "gap.end",
                "timestamp_at_start": "01/27/1991 23:32",
                "timestamp_at_end": "01/29/1991 23:32",
                "admission": 0,
                "discharge": 0,
                "death": 0,
                "discharge_or_death": 0,
                "_ANY_EVENT": 5,
            },
        ],
        "target.end_summary": [
            {
                "window_name": "target.end",
                "timestamp_at_start": "01/29/1991 23:32",
                "timestamp_at_end": "01/31/1991 02:15",
                "admission": 0,
                "discharge": 1,
                "death": 0,
                "discharge_or_death": 1,
                "_ANY_EVENT": 7,
            },
        ],
    }
}


def test_e2e():
    with tempfile.TemporaryDirectory() as d:
        data_dir = Path(d) / "sample_data"
        configs_dir = Path(d) / "sample_configs"
        output_dir = Path(d) / "sample_output"

        # Create the directories
        data_dir.mkdir()
        configs_dir.mkdir()
        output_dir.mkdir()

        # Write the predicates CSV file
        predicates_csv = data_dir / "sample_data.csv"
        predicates_csv.write_text(PREDICATES_CSV.strip())

        # Run script and check the outputs
        all_stderrs = []
        all_stdouts = []
        full_stderr = ""
        full_stdout = ""
        try:
            for task_name, task_cfg in TASKS_CFGS.items():
                logger.info(f"Running task '{task_name}'...")

                # Write the task config YAMLs
                task_cfg_path = configs_dir / f"{task_name}.yaml"
                task_cfg_path.write_text(task_cfg)

                output_path = output_dir / f"{task_name}.parquet"

                extraction_config_kwargs = {
                    "data.path": str(predicates_csv.resolve()),
                    "data.standard": "direct",
                    "cohort_dir": str(configs_dir.resolve()),
                    "cohort_name": task_name,
                    "output_filepath": str(output_path.resolve()),
                    "hydra.verbose": True,
                }

                stderr, stdout = run_command("aces-cli", extraction_config_kwargs, task_name)
                stderr, stdout = run_command("aces-cli", extraction_config_kwargs, task_name)

                all_stderrs.append(stderr)
                all_stdouts.append(stdout)

                full_stderr = "\n".join(all_stderrs)
                full_stdout = "\n".join(all_stdouts)

                fp = output_dir / f"{task_name}.parquet"
                assert fp.is_file(), f"Expected {fp} to exist."
                got_df = pl.read_parquet(fp)

                # Check the columns
                expected_columns = EXPECTED_OUTPUT[task_name].keys()
                assert got_df.columns == list(expected_columns), f"Columns mismatch for task '{task_name}'"

                # Check the data
                for col_name, expected_data in EXPECTED_OUTPUT[task_name].items():
                    if col_name in ["index_timestamp", "trigger"]:
                        want = pl.DataFrame({col_name: expected_data}).with_columns(
                            pl.col(col_name).str.strptime(pl.Datetime, format=TS_FORMAT)
                        )
                    elif col_name.endswith("_summary"):
                        df_struct = pl.DataFrame(expected_data)
                        df_struct = df_struct.with_columns(
                            pl.col("timestamp_at_start").str.strptime(pl.Datetime, format=TS_FORMAT),
                            pl.col("timestamp_at_end").str.strptime(pl.Datetime, format=TS_FORMAT),
                        )
                        want = df_struct.select(
                            pl.struct(*[col for col in df_struct.columns]).alias(col_name)
                        )
                    else:
                        want = pl.DataFrame({col_name: expected_data}).with_columns(
                            *[
                                pl.col(col_name).cast(PRED_CNT_TYPE)
                                if col_name != LAST_EVENT_INDEX_COLUMN
                                else pl.col(col_name).cast(EVENT_INDEX_TYPE)
                            ]
                        )
                    got = got_df.select(col_name)
                    assert_df_equal(want, got, f"Data mismatch for task '{task_name}', column '{col_name}'")

        except AssertionError as e:
            print(f"Failed on task '{task_name}'")
            print(f"stderr:\n{full_stderr}")
            print(f"stdout:\n{full_stdout}")
            raise e
