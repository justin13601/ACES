"""Main script for end-to-end task querying."""

from datetime import datetime
from importlib.resources import files
from pathlib import Path

import hydra
from loguru import logger
from omegaconf import DictConfig

from . import config, predicates, query

config_yaml = files("aces").joinpath("config.yaml")
if not config_yaml.is_file():
    raise FileNotFoundError("Core configuration not successfully installed!")


@hydra.main(version_base=None, config_path=str(config_yaml.parent.resolve()), config_name=config_yaml.stem)
def main(cfg: DictConfig):
    cfg = hydra.utils.instantiate(cfg, _convert_="all")

    st = datetime.now()

    # Set output path
    cohort_dir = Path(cfg.cohort_dir)
    cohort_dir.mkdir(exist_ok=True, parents=True)

    # load configuration
    logger.info(f"Loading config from {cfg.config_path}")
    task_cfg = config.TaskExtractorConfig.load(Path(cfg.config_path))

    # get predicates_df
    data_path = Path(cfg.data.path)
    data_standard = cfg.data.standard.lower()

    if not data_path.exists():
        raise FileNotFoundError(f"Requested data path {data_path} does not exist!")
    if data_standard != {"csv", "meds", "esgpt"}:
        raise ValueError(
            f"Data standard {cfg.data.standard} not supported. Must be one of 'csv', 'meds', 'esgpt'"
        )

    logger.info("Loading data from {data_path} in format {data_standard}")
    predicates_df = predicates.get_predicates_df(task_cfg, data_path, standard=data_standard.lower())

    # query results
    result = query.query(task_cfg, predicates_df)

    # save results to parquet
    result.write_parquet(cfg.output_filepath, use_pyarrow=True)
    logger.info(f"Completed in {datetime.now() - st}. Results saved to {cfg.output_filepath}.")


if __name__ == "__main__":
    main()
