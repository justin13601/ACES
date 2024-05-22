"""Main script for end-to-end task querying."""

import os
from pathlib import Path

import hydra
import polars as pl
from EventStream.data.dataset_polars import Dataset
from loguru import logger
from omegaconf import DictConfig

from esgpt_task_querying import config, predicates, query


def load_using_directory(cfg, path):
    # currently only supports ESGPT dataset when loading from directory
    try:
        ESD = Dataset.load(path)
    except Exception as e:
        raise ValueError(
            f"Error loading data using ESGPT: {e}. "
            "Please ensure the path provided is a valid ESGPT dataset directory."
        ) from e

    events_df = ESD.events_df
    dynamic_measurements_df = ESD.dynamic_measurements_df

    try:
        predicates_df = predicates.generate_predicates_df(
            cfg, [events_df, dynamic_measurements_df], standard="ESGPT"
        )
    except Exception as e:
        raise ValueError(
            "Error generating predicate columns from configuration file! "
            "Check to make sure the format of the configuration file is valid."
        ) from e

    return predicates_df


def load_using_file(cfg, path):
    # .csv file
    if path.suffix.lower() == ".csv":
        df_data = pl.read_csv(path).with_columns(
            pl.col("timestamp").str.strptime(pl.Datetime, format="%m/%d/%Y %H:%M").cast(pl.Datetime)
        )
    # .parquet file (for MEDS)
    elif path.suffix.lower() == ".parquet":
        df_data = pl.read_parquet(path).with_columns(
            pl.col("timestamp").str.strptime(pl.Datetime, format="%m/%d/%Y %H:%M").cast(pl.Datetime)
        )

    # check if data is in correct format
    if df_data.shape[0] == 0:
        raise ValueError("Provided dataset is empty!")
    if "subject_id" not in df_data.columns:
        raise ValueError("Provided dataset does not have subject_id column!")
    if "timestamp" not in df_data.columns:
        raise ValueError("Provided dataset does not have timestamp column!")

    # .csv file
    if path.suffix.lower() == ".csv":
        try:
            predicates_df = predicates.generate_predicates_df(cfg, df_data, standard="CSV")
        except Exception as e:
            raise ValueError(
                "Error generating predicate columns from configuration file! "
                "Check to make sure the format of the configuration file is valid."
            ) from e
    # .parquet file (for MEDS)
    elif path.suffix.lower() == ".parquet":
        try:
            predicates_df = predicates.generate_predicates_df(cfg, df_data, standard="MEDS")
        except Exception as e:
            raise ValueError(
                "Error generating predicate columns from configuration file! "
                "Check to make sure the format of the configuration file is valid."
            ) from e
    return predicates_df


@hydra.main(version_base=None, config_path="", config_name="run")
def run(cfg: DictConfig) -> None:
    cfg = hydra.utils.instantiate(cfg, _convert_="all")

    # Set output path
    if "output_path" not in cfg or not cfg["output_path"]:
        output_path = Path.cwd()
    else:
        output_path = Path(cfg["output_path"])

    # load configuration
    logger.debug("Loading config...")
    task_cfg_path = Path(cfg["config_path"])
    task_cfg = config.TaskExtractorConfig.load(task_cfg_path)

    # load data
    logger.debug("Loading data...")
    data_path = Path(cfg["data_path"])
    if not data_path.exists():
        logger.error(f"{data_path} does not exist.")
        return

    if data_path.is_dir():
        # load directory
        logger.debug("Directory provided, checking directory...")
        predicates_df = load_using_directory(task_cfg, data_path)
    else:
        # load file
        logger.debug("File path provided, checking file...")
        predicates_df = load_using_file(task_cfg, data_path)

    result = query.query(task_cfg, predicates_df)
    result = result.with_columns(
        *[pl.col(col).struct.json_encode() for col in result.columns if "summary" in col]
    )
    result.write_csv(
        os.path.join(output_path, "result.csv"),
        include_header=True,
        date_format="%m/%d/%Y %H:%M",
    )


if __name__ == "__main__":
    run()
