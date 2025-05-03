"""Test utilities."""

import logging
import subprocess
import tempfile
from pathlib import Path

import polars as pl
from polars.testing import assert_frame_equal

logger = logging.getLogger(__name__)


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


def assert_df_equal(want: pl.DataFrame, got: pl.DataFrame, msg: str | None = None, **kwargs):
    try:
        assert_frame_equal(want, got, **kwargs)
    except AssertionError as e:
        pl.Config.set_tbl_rows(-1)
        print(f"DFs are not equal: {msg}\nWant:")
        print(want)
        print("Got:")
        print(got)
        raise AssertionError(f"{msg}\n{e}") from e


def write_input_files(data_dir: Path, input_files: dict[str, pl.DataFrame | str]) -> dict[str, Path]:
    wrote_files = {}
    for name, df in input_files.items():
        if isinstance(df, str):
            data_fp = data_dir / f"{name}.csv"
            data_fp.parent.mkdir(parents=True, exist_ok=True)
            data_fp.write_text(df)
        elif isinstance(df, pl.DataFrame):
            data_fp = data_dir / f"{name}.parquet"
            data_fp.parent.mkdir(parents=True, exist_ok=True)
            df.write_parquet(data_fp)
        else:
            raise ValueError(f"Invalid input type: {type(df)}")
        wrote_files[name] = data_fp
    return wrote_files


def write_task_configs(
    cohort_dir: Path, task_configs: dict[str, str], predicate_files: dict[str, str] | None = None
) -> dict[str, Path]:
    wrote_files = {}
    for name, cfg in task_configs.items():
        fp = cohort_dir / f"{name}.yaml"
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(cfg)
        wrote_files[name] = fp
    if predicate_files is not None:
        for name, cfg in predicate_files.items():
            fp = cohort_dir / f"{name}_predicates.yaml"
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_text(cfg)
            wrote_files[f"{name}_predicates"] = fp
    return wrote_files


def cli_test(
    input_files: dict[str, pl.DataFrame | str],
    task_configs: dict[str, str],
    want_outputs_by_task: dict[str, dict[str, pl.DataFrame]],
    data_standard: str,
    predicate_files: str | None = None,
):
    if data_standard not in ["meds", "direct"]:
        raise ValueError(f"Invalid data standard: {data_standard}")

    with tempfile.TemporaryDirectory() as root_dir:
        root_dir = Path(root_dir)
        data_dir = root_dir / "sample_data" / "data"
        cohort_dir = root_dir / "sample_cohort"

        wrote_files = write_input_files(data_dir, input_files)
        if len(wrote_files) == 0:
            raise ValueError("No input files were written.")
        elif len(wrote_files) > 1:
            sharded = True
            command = "aces-cli --multirun"
        else:
            sharded = False
            command = "aces-cli"

        wrote_configs = write_task_configs(cohort_dir, task_configs, predicate_files)
        if len(wrote_configs) == 0:
            raise ValueError("No task configs were written.")

        for task in task_configs:
            if sharded:
                want_outputs = {
                    cohort_dir / task / f"{n}.parquet": df for n, df in want_outputs_by_task[task].items()
                }
            else:
                want_outputs = {cohort_dir / f"{task}.parquet": want_outputs_by_task[task]}

            extraction_config_kwargs = {
                "cohort_dir": str(cohort_dir.resolve()),
                "cohort_name": task,
                "hydra.verbose": True,
                "data.standard": data_standard,
            }

            if len(wrote_files) > 1:
                extraction_config_kwargs["data"] = "sharded"
                extraction_config_kwargs["data.root"] = str(data_dir.resolve())
                extraction_config_kwargs['"data.shard'] = f'$(expand_shards {data_dir.resolve()!s})"'
            else:
                extraction_config_kwargs["data.path"] = str(next(iter(wrote_files.values())).resolve())

            if predicate_files is not None:
                extraction_config_kwargs["predicates_path"] = str(
                    wrote_configs[f"{task}_predicates"].resolve()
                )

            stderr, stdout = run_command(command, extraction_config_kwargs, f"CLI should run for {task}")

            try:
                if sharded:
                    out_dir = cohort_dir / task
                    all_out_fps = list(out_dir.glob("**/*.parquet"))
                    all_out_fps_str = ", ".join(str(x.relative_to(out_dir)) for x in all_out_fps)
                    if len(all_out_fps) == 0 and len(want_outputs) > 0:
                        all_directory_contents = ", ".join(
                            str(x.relative_to(cohort_dir)) for x in cohort_dir.glob("**/*")
                        )

                        raise AssertionError(
                            f"No output files found for task '{task}'. Found files: {all_directory_contents}"
                        )

                    assert len(all_out_fps) == len(want_outputs), (
                        f"Expected {len(want_outputs)} outputs, got {len(all_out_fps)}: {all_out_fps_str}"
                    )

                for want_fp, want_df in want_outputs.items():
                    out_shard = want_fp.relative_to(cohort_dir)
                    assert want_fp.is_file(), f"Expected {out_shard} to exist."

                    got_df = pl.read_parquet(want_fp)
                    assert_df_equal(
                        want_df, got_df, f"Data mismatch for shard '{out_shard}':\n{want_df}\n{got_df}"
                    )
            except AssertionError as e:
                logger.error(f"{stderr}\n{stdout}")
                raise AssertionError(f"Error running task '{task}': {e}") from e
