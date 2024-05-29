"""Tests the full end-to-end extraction process."""

import rootutils

root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=True)

import subprocess
import tempfile
from io import StringIO
from pathlib import Path

import polars as pl
from loguru import logger
from polars.testing import assert_frame_equal

pl.enable_string_cache()

# Data (input)
PREDICATES_CSV = """
subject_id,timestamp,admission,death,discharge,lab,spo2,normal_spo2,abnormally_low_spo2,abnormally_high_spo2,procedure_start,procedure_end,ventilation,diagnosis_ICD9CM/41071,diagnosis_ICD10CM/I214
1,12/1/1989 12:03,1,0,0,0,0,0,0,0,0,0,0,0,0
1,12/1/1989 13:14,0,0,0,1,1,1,0,0,0,0,0,0,0
1,12/1/1989 15:17,0,0,0,1,1,1,0,0,0,0,0,0,0
1,12/1/1989 16:17,0,0,0,1,1,1,0,0,0,0,0,0,0
1,12/1/1989 20:17,0,0,0,1,1,1,0,0,0,0,0,0,0
1,12/2/1989 3:00,0,0,0,1,1,1,0,0,0,0,0,0,0
1,12/2/1989 9:00,0,0,0,0,0,0,0,0,0,0,0,0,0
1,12/2/1989 10:00,0,0,0,0,0,0,0,0,1,0,1,0,0
1,12/2/1989 14:22,0,0,0,0,0,0,0,0,0,1,1,0,0
1,12/2/1989 15:00,0,0,1,0,0,0,0,0,0,0,0,0,0
1,1/21/1991 11:59,0,0,0,0,0,0,0,0,0,0,0,1,0
1,1/27/1991 23:32,1,0,0,0,0,0,0,0,0,0,0,0,0
1,1/27/1991 23:46,0,0,0,1,1,1,0,0,0,0,0,0,0
1,1/28/1991 3:18,0,0,0,1,0,0,0,0,0,0,0,0,0
1,1/28/1991 3:28,0,0,0,0,0,0,0,0,1,0,1,0,0
1,1/28/1991 4:36,0,0,0,1,1,1,0,0,0,0,0,0,0
1,1/29/1991 23:32,0,0,0,1,1,1,0,0,0,0,0,0,0
1,1/30/1991 5:00,0,0,0,1,0,0,0,0,0,0,0,0,0
1,1/30/1991 8:00,0,0,0,1,1,0,0,1,0,0,0,0,0
1,1/30/1991 11:00,0,0,0,1,1,1,0,0,0,0,0,0,0
1,1/30/1991 14:00,0,0,0,0,0,0,0,0,0,0,0,0,0
1,1/30/1991 14:15,0,0,0,1,0,0,0,0,0,0,0,0,0
1,1/31/1991 1:00,0,0,0,0,0,0,0,0,0,1,1,0,0
1,1/31/1991 2:15,0,0,1,0,0,0,0,0,0,0,0,0,0
1,2/8/1991 8:15,0,0,0,1,1,1,0,0,0,0,0,0,0
1,3/3/1991 19:33,1,0,0,0,0,0,0,0,0,0,0,0,0
1,3/3/1991 20:33,0,0,0,1,1,0,1,0,0,0,0,0,0
1,3/3/1991 21:38,0,1,0,0,0,0,0,0,0,0,0,0,0
2,3/8/1996 2:24,1,0,0,0,0,0,0,0,0,0,0,0,0
2,3/8/1996 2:35,0,0,0,1,1,1,0,0,0,0,0,0,0
2,3/8/1996 4:00,0,0,0,1,1,1,0,0,0,0,0,0,0
2,3/8/1996 10:00,0,0,0,0,0,0,0,0,0,0,0,0,0
2,3/8/1996 16:00,0,0,1,0,0,0,0,0,0,0,0,0,0
2,6/5/1996 0:32,1,0,0,0,0,0,0,0,0,0,0,0,0
2,6/5/1996 0:48,0,0,0,0,0,0,0,0,0,0,0,0,1
2,6/5/1996 1:59,0,0,0,0,0,0,0,0,1,0,1,0,0
2,6/7/1996 6:00,0,0,0,1,0,0,0,0,0,0,0,0,0
2,6/7/1996 9:00,0,0,0,1,1,0,1,0,0,0,0,0,0
2,6/7/1996 12:00,0,0,0,1,1,1,0,0,0,0,0,0,0
2,6/7/1996 15:00,0,0,0,0,0,0,0,0,0,1,1,0,0
2,6/7/1996 15:00,0,0,0,0,0,0,0,0,0,0,0,0,0
2,6/8/1996 3:00,0,1,0,0,0,0,0,0,0,0,0,0,0
3,3/8/1996 2:22,0,0,0,0,0,0,0,0,1,0,1,0,0
3,3/8/1996 2:24,1,0,0,0,0,0,0,0,0,0,0,0,0
3,3/8/1996 2:37,0,0,0,1,1,1,0,0,0,0,0,0,0
3,3/9/1996 8:00,0,0,0,1,0,0,0,0,0,0,0,0,0
3,3/9/1996 11:00,0,0,0,1,1,1,0,0,0,0,0,0,0
3,3/9/1996 19:00,0,0,0,1,1,1,0,0,0,0,0,0,0
3,3/9/1996 22:00,0,0,0,0,0,0,0,0,0,0,0,0,0
3,3/11/1996 21:00,0,0,0,0,0,0,0,0,0,1,1,0,0
3,3/12/1996 0:00,0,1,0,0,0,0,0,0,0,0,0,0,0
"""

# Tasks (input)
TASKS_CFGS = {
    "inhospital-mortality": """
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
          start: input.end
          end: start + 24h
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
expected_columns = {
    "inhospital-mortality": [
        "subject_id",
        "index_timestamp",
        "label",
        "trigger",
        "input.start_summary",
        "target.end_summary",
        "gap.end_summary",
        "input.end_summary",
    ]
}

expected_data = {"inhospital-mortality": ""}


def get_expected_output(df: str) -> pl.DataFrame:
    return pl.read_parquet(source=StringIO(f"{df}.parquet"))


def run_command(script: Path, hydra_kwargs: dict[str, str], test_name: str):
    script = str(script.resolve())
    command_parts = ["python", script] + [f"{k}={v}" for k, v in hydra_kwargs.items()]
    command_out = subprocess.run(" ".join(command_parts), shell=True, capture_output=True)
    stderr = command_out.stderr.decode()
    stdout = command_out.stdout.decode()
    if command_out.returncode != 0:
        raise AssertionError(f"{test_name} failed!\nstdout:\n{stdout}\nstderr:\n{stderr}")
    return stderr, stdout


def assert_df_equal(want: pl.DataFrame, got: pl.DataFrame, msg: str = None, **kwargs):
    try:
        assert_frame_equal(want, got, **kwargs)
    except AssertionError as e:
        pl.Config.set_tbl_rows(-1)
        print(f"DFs are not equal: {msg}\nwant:")
        print(want)
        print("got:")
        print(got)
        raise AssertionError(f"{msg}\n{e}") from e


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

                extraction_config_kwargs = {
                    "config_path": str(task_cfg_path.resolve()),
                    "data.path": str(predicates_csv.resolve()),
                    "data.standard": "csv",
                    "output_dir": str(output_dir.resolve()),
                    "hydra.verbose": True,
                }

                stderr, stdout = run_command("aces-cli", extraction_config_kwargs, task_name)

                all_stderrs.append(stderr)
                all_stdouts.append(stdout)

                full_stderr = "\n".join(all_stderrs)
                full_stdout = "\n".join(all_stdouts)

                expected_df = get_expected_output(task_name)

                fp = output_dir / f"results_{task_name}.parquet"
                assert fp.is_file(), f"Expected {fp} to exist."
                got_df = pl.read_parquet(fp, glob=False)

                assert_df_equal(
                    expected_df,
                    got_df,
                    f"Expected output for task '{task_name}' to be equal to the expected output.",
                    check_column_order=False,
                    check_row_order=False,
                )

        except AssertionError as e:
            print(f"Failed on task '{task_name}'")
            print(f"stderr:\n{full_stderr}")
            print(f"stdout:\n{full_stdout}")
            raise e
