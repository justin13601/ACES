from datetime import timedelta

import polars as pl

from .types import PRED_CNT_TYPE, TemporalWindowBounds, ToEventWindowBounds

def aggregate_temporal_window(
    predicates_df: pl.DataFrame,
    endpoint_expr: TemporalWindowBounds | tuple[bool, timedelta, bool, timedelta | None],
) -> pl.DataFrame: ...
def aggregate_event_bound_window(
    predicates_df: pl.DataFrame,
    endpoint_expr: ToEventWindowBounds | tuple[bool, str, bool, timedelta | None],
) -> pl.DataFrame: ...
def boolean_expr_bound_sum(
    df: pl.DataFrame,
    boundary_expr: pl.Expr,
    mode: str,
    closed: str,
    offset: timedelta = timedelta(0),
) -> pl.DataFrame: ...
