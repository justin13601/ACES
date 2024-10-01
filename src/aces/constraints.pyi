import polars as pl
from loguru import logger

from .types import ANY_EVENT_COLUMN

def check_constraints(
    window_constraints: dict[str, tuple[int | None, int | None]], summary_df: pl.DataFrame
) -> pl.DataFrame: ...
def check_static_variables(patient_demographics: list[str], predicates_df: pl.DataFrame) -> pl.DataFrame: ...
