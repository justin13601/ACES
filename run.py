import os
from pathlib import Path

import polars as pl
from loguru import logger
from EventStream.data.dataset_polars import Dataset
import hydra
from omegaconf import DictConfig, OmegaConf

from esgpt_task_querying import main


@hydra.main(version_base=None, config_path="", config_name="run")
def run(cfg: DictConfig) -> None:
    cfg = hydra.utils.instantiate(cfg, _convert_="all")

    config_path = Path(cfg["config_path"])
    data_path = Path(cfg["data_path"])
    if not cfg["output_path"] or "output_path" not in cfg:
        output_path = Path.cwd()
    else:
        output_path = Path(cfg["output_path"])

    if data_path.is_dir():
        result = main.query_task(config_path, data_path.absolute().as_posix())
    else:
        if not data_path.exists():
            logger.error(f"File {data_path} does not exist.")
            return
        if "csv" not in data_path.suffix:
            logger.error(f"File {data_path} is not a csv file.")
            return
        df_data = pl.read_csv(data_path).with_columns(
            pl.col("timestamp")
            .str.strptime(pl.Datetime, format="%m/%d/%Y %H:%M")
            .cast(pl.Datetime)
        )
        result = main.query_task(cfg, df_data)
    
    result = result.with_columns(
        *[
            pl.col(col).struct.json_encode() for col in result.columns if 'window_summary' in col
        ]
    )

    result.write_csv(os.path.join(output_path, "result.csv"), include_header=True, date_format="%m/%d/%Y %H:%M")


if __name__ == "__main__":
    run()
