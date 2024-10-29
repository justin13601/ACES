from typing import Optional

import polars as pl
from bigtree import Node

from .config import TaskExtractorConfig

def query(cfg: TaskExtractorConfig, predicates_df: pl.DataFrame) -> pl.DataFrame: ...
