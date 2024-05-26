"""Main script for end-to-end task querying."""

import os
from datetime import datetime
from pathlib import Path

import hydra
from loguru import logger
from omegaconf import DictConfig

from . import config, predicates, query


@hydra.main(version_base=None, config_path="", config_name="")
def main(cfg: DictConfig) -> None:
    cfg = hydra.utils.instantiate(cfg, _convert_="all")

    # Set output path
    if "output_dir" not in cfg or not cfg["output_dir"]:
        output_dir = Path.cwd()
    else:
        output_dir = Path(cfg["output_dir"])
    os.makedirs(output_dir, exist_ok=True)

    # load configuration
    logger.info("Loading config...")
    task_cfg_path = Path(cfg["config_path"])
    task_cfg = config.TaskExtractorConfig.load(task_cfg_path)

    # get predicates_df
    logger.info("Loading data...")
    data_path = Path(cfg["data"]["path"])
    if not data_path.exists():
        logger.error(f"{data_path} does not exist.")
        return

    data_standard = cfg["data"]["standard"]
    if data_standard.lower() not in ["csv", "meds", "esgpt"]:
        logger.error(f"Unsupported data standard: {data_standard}")
        return

    predicates_df = predicates.generate_predicates_df(task_cfg, data_path, standard=data_standard.lower())

    # query results
    result = query.query(task_cfg, predicates_df)

    # save results to parquet
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result_path = output_dir / f"results_{timestamp}.parquet"
    result.write_parquet(result_path)
    logger.info(f"Results saved to {result_path}.")


if __name__ == "__main__":
    main()
