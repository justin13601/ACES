"""Tests the full end-to-end extraction process."""

import rootutils

root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=True)

import json
import subprocess
import tempfile
from io import StringIO
from pathlib import Path

import polars as pl
from loguru import logger
from polars.testing import assert_frame_equal

pl.enable_string_cache()

# Test data (inputs)

SUBJECTS_CSV = """
MRN,dob,eye_color,height
1195293,06/20/1978,BLUE,164.6868838269085
239684,12/28/1980,BROWN,175.271115221764
1500733,07/20/1986,BROWN,158.60131573580904
814703,03/28/1976,HAZEL,156.48559093209357
754281,12/19/1988,BROWN,166.22261567137025
68729,03/09/1978,HAZEL,160.3953106166676
"""

ADMIT_VITALS_CSV = """
patient_id,admit_date,disch_date,department,vitals_date,HR,temp
239684,"05/11/2010, 17:41:51","05/11/2010, 19:27:19",CARDIAC,"05/11/2010, 18:57:18",112.6,95.5
754281,"01/03/2010, 06:27:59","01/03/2010, 08:22:13",PULMONARY,"01/03/2010, 06:27:59",142.0,99.8
814703,"02/05/2010, 05:55:39","02/05/2010, 07:02:30",ORTHOPEDIC,"02/05/2010, 05:55:39",170.2,100.1
239684,"05/11/2010, 17:41:51","05/11/2010, 19:27:19",CARDIAC,"05/11/2010, 18:25:35",113.4,95.8
68729,"05/26/2010, 02:30:56","05/26/2010, 04:51:52",PULMONARY,"05/26/2010, 02:30:56",86.0,97.8
1195293,"06/20/2010, 19:23:52","06/20/2010, 20:50:04",CARDIAC,"06/20/2010, 20:12:31",112.5,99.8
1500733,"06/03/2010, 14:54:38","06/03/2010, 16:44:26",ORTHOPEDIC,"06/03/2010, 16:20:49",90.1,100.1
239684,"05/11/2010, 17:41:51","05/11/2010, 19:27:19",CARDIAC,"05/11/2010, 17:48:48",105.1,96.2
239684,"05/11/2010, 17:41:51","05/11/2010, 19:27:19",CARDIAC,"05/11/2010, 17:41:51",102.6,96.0
1195293,"06/20/2010, 19:23:52","06/20/2010, 20:50:04",CARDIAC,"06/20/2010, 19:25:32",114.1,100.0
1500733,"06/03/2010, 14:54:38","06/03/2010, 16:44:26",ORTHOPEDIC,"06/03/2010, 14:54:38",91.4,100.0
1195293,"06/20/2010, 19:23:52","06/20/2010, 20:50:04",CARDIAC,"06/20/2010, 20:41:33",107.5,100.4
1195293,"06/20/2010, 19:23:52","06/20/2010, 20:50:04",CARDIAC,"06/20/2010, 20:24:44",107.7,100.0
1195293,"06/20/2010, 19:23:52","06/20/2010, 20:50:04",CARDIAC,"06/20/2010, 19:45:19",119.8,99.9
1195293,"06/20/2010, 19:23:52","06/20/2010, 20:50:04",CARDIAC,"06/20/2010, 19:23:52",109.0,100.0
1500733,"06/03/2010, 14:54:38","06/03/2010, 16:44:26",ORTHOPEDIC,"06/03/2010, 15:39:49",84.4,100.3
"""

EVENT_CFGS_YAML = """
subjects:
  patient_id_col: MRN
  eye_color:
    code:
      - EYE_COLOR
      - col(eye_color)
    timestamp: null
  height:
    code: HEIGHT
    timestamp: null
    numerical_value: height
  dob:
    code: DOB
    timestamp: col(dob)
    timestamp_format: "%m/%d/%Y"
admit_vitals:
  admissions:
    code:
      - ADMISSION
      - col(department)
    timestamp: col(admit_date)
    timestamp_format: "%m/%d/%Y, %H:%M:%S"
  discharge:
    code: DISCHARGE
    timestamp: col(disch_date)
    timestamp_format: "%m/%d/%Y, %H:%M:%S"
  HR:
    code: HR
    timestamp: col(vitals_date)
    timestamp_format: "%m/%d/%Y, %H:%M:%S"
    numerical_value: HR
  temp:
    code: TEMP
    timestamp: col(vitals_date)
    timestamp_format: "%m/%d/%Y, %H:%M:%S"
    numerical_value: temp
"""

# Test data (expected outputs) -- ALL OF THIS MAY CHANGE IF THE SEED OR DATA CHANGES
EXPECTED_SPLITS = {
    "train/0": [239684, 1195293],
    "train/1": [68729, 814703],
    "tuning/0": [754281],
    "held_out/0": [1500733],
}


def get_expected_output(df: str) -> pl.DataFrame:
    return (
        pl.read_csv(source=StringIO(df))
        .select(
            "patient_id",
            pl.col("timestamp").str.strptime(pl.Datetime, "%m/%d/%Y, %H:%M:%S").alias("timestamp"),
            pl.col("code").cast(pl.Categorical),
            "numerical_value",
        )
        .sort(by=["patient_id", "timestamp"])
    )


MEDS_OUTPUT_TRAIN_0_SUBJECTS = """
patient_id,timestamp,code,numerical_value
239684,,EYE_COLOR//BROWN,
239684,,HEIGHT,175.271115221764
239684,"12/28/1980, 00:00:00",DOB,
1195293,,EYE_COLOR//BLUE,
1195293,,HEIGHT,164.6868838269085
1195293,"06/20/1978, 00:00:00",DOB,
"""

MEDS_OUTPUT_TRAIN_0_ADMIT_VITALS = """
patient_id,timestamp,code,numerical_value
239684,"05/11/2010, 17:41:51",ADMISSION//CARDIAC,
239684,"05/11/2010, 17:41:51",HR,102.6
239684,"05/11/2010, 17:41:51",TEMP,96.0
239684,"05/11/2010, 17:48:48",HR,105.1
239684,"05/11/2010, 17:48:48",TEMP,96.2
239684,"05/11/2010, 18:25:35",HR,113.4
239684,"05/11/2010, 18:25:35",TEMP,95.8
239684,"05/11/2010, 18:57:18",HR,112.6
239684,"05/11/2010, 18:57:18",TEMP,95.5
239684,"05/11/2010, 19:27:19",DISCHARGE,
1195293,"06/20/2010, 19:23:52",ADMISSION//CARDIAC,
1195293,"06/20/2010, 19:23:52",HR,109.0
1195293,"06/20/2010, 19:23:52",TEMP,100.0
1195293,"06/20/2010, 19:25:32",HR,114.1
1195293,"06/20/2010, 19:25:32",TEMP,100.0
1195293,"06/20/2010, 19:45:19",HR,119.8
1195293,"06/20/2010, 19:45:19",TEMP,99.9
1195293,"06/20/2010, 20:12:31",HR,112.5
1195293,"06/20/2010, 20:12:31",TEMP,99.8
1195293,"06/20/2010, 20:24:44",HR,107.7
1195293,"06/20/2010, 20:24:44",TEMP,100.0
1195293,"06/20/2010, 20:41:33",HR,107.5
1195293,"06/20/2010, 20:41:33",TEMP,100.4
1195293,"06/20/2010, 20:50:04",DISCHARGE,
"""

SUB_SHARDED_OUTPUTS = {
    "train/0": {
        "subjects": MEDS_OUTPUT_TRAIN_0_SUBJECTS,
        "admit_vitals": MEDS_OUTPUT_TRAIN_0_ADMIT_VITALS,
    },
    "train/1": {
        "subjects": None,
        "admit_vitals": None,
    },
    "tuning/0": {
        "subjects": None,
        "admit_vitals": None,
    },
    "held_out/0": {
        "subjects": None,
        "admit_vitals": None,
    },
}


MEDS_OUTPUTS = {
    "train/0": [MEDS_OUTPUT_TRAIN_0_SUBJECTS, MEDS_OUTPUT_TRAIN_0_ADMIT_VITALS],
    "train/1": None,
    "tuning/0": None,
    "held_out/0": None,
}


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


def test_extraction():
    with tempfile.TemporaryDirectory() as d:
        raw_cohort_dir = Path(d) / "raw_cohort"
        MEDS_cohort_dir = Path(d) / "MEDS_cohort"

        # Create the directories
        raw_cohort_dir.mkdir()
        MEDS_cohort_dir.mkdir()

        subjects_csv = raw_cohort_dir / "subjects.csv"
        admit_vitals_csv = raw_cohort_dir / "admit_vitals.csv"
        event_cfgs_yaml = raw_cohort_dir / "event_cfgs.yaml"

        # Write the CSV files
        subjects_csv.write_text(SUBJECTS_CSV.strip())
        admit_vitals_csv.write_text(ADMIT_VITALS_CSV.strip())

        # Mix things up -- have one CSV be also in parquet format.
        admit_vitals_parquet = raw_cohort_dir / "admit_vitals.parquet"
        df = pl.read_csv(admit_vitals_csv)

        old_shape = df.shape
        assert old_shape[0] > 0, "Should have some rows"

        df.write_parquet(admit_vitals_parquet, use_pyarrow=True)

        df = pl.scan_parquet(admit_vitals_parquet)
        df = df.select(list(df.columns))
        assert (
            df.select(pl.len()).collect().item() == old_shape[0]
        ), "Should have the same number of rows after select."
        assert old_shape == df.collect().shape, "Shapes should be the same after selecting all columns."

        df = pl.scan_parquet(admit_vitals_parquet)
        assert old_shape == df.collect().shape, "Shapes should be the same after scanning the parquet file."

        # Write the event config YAML
        event_cfgs_yaml.write_text(EVENT_CFGS_YAML)

        # Run the extraction script
        #   1. Sub-shard the data (this will be a null operation in this case, but it is worth doing just in
        #      case.
        #   2. Collect the patient splits.
        #   3. Extract the events and sub-shard by patient.
        #   4. Merge to the final output.

        extraction_config_kwargs = {
            "raw_cohort_dir": str(raw_cohort_dir.resolve()),
            "MEDS_cohort_dir": str(MEDS_cohort_dir.resolve()),
            "event_conversion_config_fp": str(event_cfgs_yaml.resolve()),
            "split_fracs.train": 4 / 6,
            "split_fracs.tuning": 1 / 6,
            "split_fracs.held_out": 1 / 6,
            "row_chunksize": 10,
            "n_patients_per_shard": 2,
            "hydra.verbose": True,
        }

        extraction_root = root / "scripts" / "extraction"

        all_stderrs = []
        all_stdouts = []

        # Step 1: Sub-shard the data
        stderr, stdout = run_command(
            extraction_root / "shard_events.py", extraction_config_kwargs, "shard_events"
        )

        all_stderrs.append(stderr)
        all_stdouts.append(stdout)

        subsharded_dir = MEDS_cohort_dir / "sub_sharded"

        try:
            out_files = list(subsharded_dir.glob("**/*.parquet"))
            assert len(out_files) == 3, f"Expected 3 output files, got {len(out_files)}."

            # Checking specific out files:
            #   1. subjects.parquet
            subjects_out = subsharded_dir / "subjects" / "[0-6).parquet"
            assert subjects_out.is_file(), f"Expected {subjects_out} to exist. Files include {out_files}."

            assert_df_equal(
                pl.read_csv(subjects_csv),
                pl.read_parquet(subjects_out, glob=False),
                "Subjects should be equal after sub-sharding",
                check_column_order=False,
                check_row_order=False,
            )
        except AssertionError as e:
            full_stderr = "\n".join(all_stderrs)
            print("Sub-sharding failed")
            print(f"stderr:\n{full_stderr}")
            raise e

        #   2. admit_vitals.parquet
        df_chunks = []
        for chunk in ["[0-10)", "[10-16)"]:
            admit_vitals_chunk_fp = subsharded_dir / "admit_vitals" / f"{chunk}.parquet"
            assert admit_vitals_chunk_fp.is_file(), f"Expected {admit_vitals_chunk_fp} to exist."

            df_chunks.append(pl.read_parquet(admit_vitals_chunk_fp, glob=False))

        assert_df_equal(
            pl.read_csv(admit_vitals_csv),
            pl.concat(df_chunks),
            "Admit vitals should be equal after sub-sharding",
            check_column_order=False,
            check_row_order=False,
        )

        # Step 2: Collect the patient splits
        stderr, stdout = run_command(
            extraction_root / "split_and_shard_patients.py",
            extraction_config_kwargs,
            "split_and_shard_patients",
        )

        all_stderrs.append(stderr)
        all_stdouts.append(stdout)

        splits_fp = MEDS_cohort_dir / "splits.json"
        assert splits_fp.is_file(), f"Expected splits @ {str(splits_fp.resolve())} to exist."

        splits = json.loads(splits_fp.read_text())
        expected_keys = ["train/0", "train/1", "tuning/0", "held_out/0"]

        expected_keys_str = ", ".join(f"'{k}'" for k in expected_keys)
        got_keys_str = ", ".join(f"'{k}'" for k in splits.keys())

        assert set(splits.keys()) == set(expected_keys), (
            f"Expected splits to have keys {expected_keys_str}.\n" f"Got keys: {got_keys_str}"
        )

        assert splits == EXPECTED_SPLITS, (
            f"Expected splits to be {EXPECTED_SPLITS}, got {splits}. NOTE THIS MAY CHANGE IF THE SEED OR "
            "DATA CHANGES -- FAILURE HERE MAY BE JUST DUE TO A NON-DETERMINISTIC SPLIT AND THE TEST NEEDING "
            "TO BE UPDATED."
        )

        # Step 3: Extract the events and sub-shard by patient
        stderr, stdout = run_command(
            extraction_root / "convert_to_sharded_events.py",
            extraction_config_kwargs,
            "convert_events",
        )
        all_stderrs.append(stderr)
        all_stdouts.append(stdout)

        patient_subsharded_folder = MEDS_cohort_dir / "patient_sub_sharded_events"
        assert patient_subsharded_folder.is_dir(), f"Expected {patient_subsharded_folder} to be a directory."

        for split, expected_outputs in SUB_SHARDED_OUTPUTS.items():
            for prefix, expected_df_L in expected_outputs.items():
                if expected_df_L is None:
                    continue

                if not isinstance(expected_df_L, list):
                    expected_df_L = [expected_df_L]

                expected_df = pl.concat([get_expected_output(df) for df in expected_df_L])

                fps = list((patient_subsharded_folder / split / prefix).glob("*.parquet"))
                assert len(fps) > 0

                # We add a "unique" here as there may be some duplicates across the row-group sub-shards.
                got_df = pl.concat([pl.read_parquet(fp, glob=False) for fp in fps]).unique()
                try:
                    assert_df_equal(
                        expected_df,
                        got_df,
                        f"Expected output for split {split}/{prefix} to be equal to the expected output.",
                        check_column_order=False,
                        check_row_order=False,
                    )
                except AssertionError as e:
                    print(f"Failed on split {split}/{prefix}")
                    print(f"stderr:\n{stderr}")
                    print(f"stdout:\n{stdout}")
                    raise e

        # Step 4: Merge to the final output
        stderr, stdout = run_command(
            extraction_root / "merge_to_MEDS_cohort.py",
            extraction_config_kwargs,
            "merge_sharded_events",
        )
        all_stderrs.append(stderr)
        all_stdouts.append(stdout)

        full_stderr = "\n".join(all_stderrs)
        full_stdout = "\n".join(all_stdouts)

        # Check the final output
        output_folder = MEDS_cohort_dir / "final_cohort"
        try:
            for split, expected_df_L in MEDS_OUTPUTS.items():
                if expected_df_L is None:
                    continue

                if not isinstance(expected_df_L, list):
                    expected_df_L = [expected_df_L]

                expected_df = pl.concat([get_expected_output(df) for df in expected_df_L])

                fp = output_folder / f"{split}.parquet"
                assert fp.is_file(), f"Expected {fp} to exist."

                got_df = pl.read_parquet(fp, glob=False)
                assert_df_equal(
                    expected_df,
                    got_df,
                    f"Expected output for split {split} to be equal to the expected output.",
                    check_column_order=False,
                    check_row_order=False,
                )

                assert got_df["patient_id"].is_sorted(), f"Patient IDs should be sorted for split {split}."
                for subj in splits[split]:
                    got_df_subj = got_df.filter(pl.col("patient_id") == subj)
                    assert got_df_subj[
                        "timestamp"
                    ].is_sorted(), f"Timestamps should be sorted for patient {subj} in split {split}."

        except AssertionError as e:
            print(f"Failed on split {split}")
            print(f"stderr:\n{full_stderr}")
            print(f"stdout:\n{full_stdout}")
            raise e

    logger.warning("Only checked the train/0 split for now. TODO: add the rest of the splits.")
