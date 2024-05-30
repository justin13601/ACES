"""Main script for end-to-end task querying."""

from datetime import datetime
from importlib.resources import files
from pathlib import Path

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from . import config, predicates, query

config_yaml = files("aces").joinpath("config.yaml")
if not config_yaml.is_file():
    raise FileNotFoundError("Core configuration not successfully installed!")


@hydra.main(version_base=None, config_path=str(config_yaml.parent.resolve()), config_name=config_yaml.stem)
def main(cfg: DictConfig):
    st = datetime.now()

    # Set output path
    cohort_dir = Path(cfg.cohort_dir)
    cohort_dir.mkdir(exist_ok=True, parents=True)

    # load configuration
    logger.info(f"Loading config from {cfg.config_path}")
    task_cfg = config.TaskExtractorConfig.load(Path(cfg.config_path))

    logger.info(f"Attempting to get predicates dataframe given:\n{OmegaConf.to_yaml(cfg.data)}")
    predicates_df = predicates.get_predicates_df(task_cfg, cfg.data)

    # query results
    result = query.query(task_cfg, predicates_df)

    # save results to parquet
    result.write_parquet(cfg.output_filepath, use_pyarrow=True)
    logger.info(f"Completed in {datetime.now() - st}. Results saved to {cfg.output_filepath}.")


if __name__ == "__main__":
    main()
