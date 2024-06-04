"""Main script for end-to-end task querying."""

import sys
from importlib.resources import files

import hydra
import hydra.core
import hydra.core.hydra_config
from loguru import logger
from omegaconf import DictConfig, OmegaConf

config_yaml = files("aces").joinpath("configs/aces.yaml")
if not config_yaml.is_file():
    raise FileNotFoundError("Core configuration not successfully installed!")

if len(sys.argv) == 1:
    print("Usage: aces-cli [OPTIONS]")
    print("Try 'aces-cli --help' for more information.")
    print("For more information, visit: https://eventstreamaces.readthedocs.io/en/latest/usage.html")
    sys.exit(1)


@hydra.main(version_base=None, config_path=str(config_yaml.parent.resolve()), config_name=config_yaml.stem)
def main(cfg: DictConfig):
    import os
    from datetime import datetime
    from pathlib import Path

    from . import config, predicates, query, utils

    utils.hydra_loguru_init(f"{hydra.core.hydra_config.HydraConfig.get().job.name}.log")

    st = datetime.now()

    # Set output path
    cohort_dir = Path(cfg.cohort_dir)
    cohort_dir.mkdir(exist_ok=True, parents=True)

    # load configuration
    logger.info(f"Loading config from '{cfg.config_path}'")
    task_cfg = config.TaskExtractorConfig.load(Path(cfg.config_path))

    logger.info(f"Attempting to get predicates dataframe given:\n{OmegaConf.to_yaml(cfg.data)}")
    predicates_df = predicates.get_predicates_df(task_cfg, cfg.data)

    # query results
    result = query.query(task_cfg, predicates_df)

    # save results to parquet
    os.makedirs(os.path.dirname(cfg.output_filepath), exist_ok=True)
    result.write_parquet(cfg.output_filepath, use_pyarrow=True)
    logger.info(f"Completed in {datetime.now() - st}. Results saved to '{cfg.output_filepath}'.")


if __name__ == "__main__":
    main()
