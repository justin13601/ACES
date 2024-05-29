"""Main script for end-to-end task querying."""

import os
from datetime import datetime
from pathlib import Path

import hydra
from loguru import logger
from omegaconf import DictConfig

from . import config, predicates, query


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    cfg = hydra.utils.instantiate(cfg, _convert_="all")

    # Set output path
    output_dir = Path(cfg.get("output_dir", Path.cwd()))
    os.makedirs(output_dir, exist_ok=True)

    # load configuration
    logger.info("Loading config...")
    task_cfg_path = Path(cfg["config_path"])
    task_cfg = config.TaskExtractorConfig.load(task_cfg_path)

    # get predicates_df
    logger.info("Loading data...")
    data_path = Path(cfg["data"]["path"])
    data_standard = cfg["data"]["standard"]
    try:
        assert data_path.exists(), f"{data_path} does not exist."
        assert data_standard.lower() in [
            "csv",
            "meds",
            "esgpt",
        ], f"Unsupported data standard: {data_standard}"
    except AssertionError as e:
        logger.error(str(e))
        return
    predicates_df = predicates.get_predicates_df(task_cfg, data_path, standard=data_standard.lower())

    # query results
    result = query.query(task_cfg, predicates_df)

    # save results to parquet
    name = cfg.get("name", "")
    if name:
        result_path = output_dir / f"results_{name}.parquet"
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        result_path = output_dir / f"results_{timestamp}.parquet"
    result.write_parquet(result_path)
    logger.info(f"Results saved to {result_path}.")


if __name__ == "__main__":
    main()
