from datetime import timedelta
from typing import Any, Iterator

import polars as pl

PRED_CNT_TYPE: pl.Int64

START_OF_RECORD_KEY: str
END_OF_RECORD_KEY: str
ANY_EVENT_COLUMN: str

@dataclasses.dataclass(order=True)
class TemporalWindowBounds:
    left_inclusive: bool
    window_size: timedelta
    right_inclusive: bool
    offset: timedelta | None = None

    def __iter__(self) -> Iterator[Any]: ...
    def __getitem__(self, key: int) -> Any: ...
    def __post_init__(self) -> None: ...
    @property
    def polars_gp_rolling_kwargs(self) -> dict[str, str | timedelta]: ...

@dataclasses.dataclass(order=True)
class ToEventWindowBounds:
    left_inclusive: bool
    end_event: str
    right_inclusive: bool
    offset: timedelta | None = None

    def __post_init__(self) -> None: ...
    def __iter__(self) -> Iterator[Any]: ...
    def __getitem__(self, key: int) -> Any: ...
    @property
    def boolean_expr_bound_sum_kwargs(self) -> dict[str, str | timedelta | pl.Expr]: ...
