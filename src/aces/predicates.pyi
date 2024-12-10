from pathlib import Path

import polars as pl
from omegaconf import DictConfig

from .config import TaskExtractorConfig

def direct_load_plain_predicates(
    data_path: Path, predicates: list[str], ts_format: str | None
) -> pl.DataFrame: ...
def generate_plain_predicates_from_meds(data_path: Path, predicates: dict) -> pl.DataFrame: ...
def process_esgpt_data(
    subjects_df: pl.DataFrame,
    events_df: pl.DataFrame,
    dynamic_measurements_df: pl.DataFrame,
    value_columns: dict[str, str],
    predicates: dict,
) -> pl.DataFrame: ...
def generate_plain_predicates_from_esgpt(data_path: Path, predicates: dict) -> pl.DataFrame: ...
def get_predicates_df(cfg: TaskExtractorConfig, data_config: DictConfig) -> pl.DataFrame: ...
