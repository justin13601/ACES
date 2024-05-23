"""Main script for end-to-end task querying."""

import os
from pathlib import Path

import hydra
import polars as pl
from loguru import logger
from omegaconf import DictConfig

from esgpt_task_querying import config, predicates, query


def load_using_directory(cfg: config.TaskExtractorConfig, path: str | Path) -> pl.DataFrame:
    # currently only supports ESGPT dataset when loading from directory
    return predicates.generate_predicates_df(cfg, path, standard="ESGPT")


def load_using_file(cfg: config.TaskExtractorConfig, path: str | Path) -> pl.DataFrame:
    # .csv file
    if path.suffix.lower() == ".csv":
        return predicates.generate_predicates_df(cfg, path, standard="CSV")
    # .parquet file (for MEDS)
    elif path.suffix.lower() == ".parquet":
        return predicates.generate_predicates_df(cfg, path, standard="MEDS")
    else:
        logger.error(f"Unsupported file type: {path.suffix}")
        return


@hydra.main(version_base=None, config_path="", config_name="run")
def run(cfg: DictConfig) -> None:
    cfg = hydra.utils.instantiate(cfg, _convert_="all")

    # Set output path
    if "output_path" not in cfg or not cfg["output_path"]:
        output_path = Path.cwd()
    else:
        output_path = Path(cfg["output_path"])

    # load configuration
    logger.info("Loading config...")
    task_cfg_path = Path(cfg["config_path"])
    task_cfg = config.TaskExtractorConfig.load(task_cfg_path)

    # get predicates_df
    logger.info("Loading data...")
    data_path = Path(cfg["data_path"])
    if not data_path.exists():
        logger.error(f"{data_path} does not exist.")
        return

    if data_path.is_dir():
        # load directory
        logger.info("Directory provided, checking directory...")
        predicates_df = load_using_directory(task_cfg, data_path)
    else:
        # load file
        logger.info("File path provided, checking file...")
        predicates_df = load_using_file(task_cfg, data_path)

    # query results
    result = query.query(task_cfg, predicates_df)

    # save results
    result = result.with_columns(
        *[pl.col(col).struct.json_encode() for col in result.columns if "summary" in col]
    ).write_csv(
        os.path.join(output_path, "result.csv"),
        include_header=True,
        date_format="%m/%d/%Y %H:%M",
    )


if __name__ == "__main__":
    run()
