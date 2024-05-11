"""This module contains the functions for summarizing windows that make up a task."""

from datetime import timedelta
from typing import Any

import polars as pl
from loguru import logger

from .types import TemporalWindowBounds, ToEventWindowBounds

PRED_CNT_TYPE = pl.UInt16


def aggregate_temporal_window(
    predicates_df: pl.LazyFrame | pl.DataFrame,
    endpoint_expr: TemporalWindowBounds | tuple[bool, timedelta, bool, timedelta | None],
) -> pl.LazyFrame | pl.DataFrame:
    """Aggregates the predicates dataframe into the specified temporal buckets.

    TODO: Use https://hypothesis.readthedocs.io/en/latest/quickstart.html to add extra tests.

    Args:
        predicates_df: The dataframe containing the predicates. The input must be sorted in ascending order by
            timestamp within each subject group. It must contain the following columns:
              - A column ``subject_id`` which contains the subject ID.
              - A column ``timestamp`` which contains the timestamp at which the event contained in any given
                row occurred.
              - A set of "predicate" columns that contain counts of the number of times a given predicate
                is satisfied in the event contained in any given row.
        endpoint_expr: The expression defining the temporal window endpoints. Can be specified as a tuple or a
            TemporalWindowBounds object, which is just a named tuple of the expected form. Said expected form
            is as follows:
              - The first element is a boolean indicating whether the start of the window is inclusive.
              - The second element is a timedelta object indicating the size of the window.
              - The third element is a boolean indicating whether the end of the window is inclusive.
              - The fourth element is a timedelta object indicating the offset from the timestamp of the row
                to the start of the window.

    Returns:
        The dataframe that has been aggregated temporally according to the specified ``endpoint_expr``. This
        dataframe will contain the same number of rows and be in the same order (in terms of ``subject_id``
        and ``timestamp``) as the input dataframe. The column names will also be the same as the input
        dataframe, but the values in the predicate columns of the output dataframe will be the sum of the
        values in the predicate columns of the input dataframe from the timestamp of the event in the row of
        the output dataframe spanning the specified temporal window.
    Returns:
        The dataframe that has been aggregated in a temporal manner according to the specified
        ``endpoint_expr``. This aggregation means the following:
          - The output dataframe will contain the same number of rows and be in the same order (in terms of
            ``subject_id`` and ``timestamp``) as the input dataframe.
          - The column names will also be the same as the input dataframe.
          - The values in the predicate columns of the output dataframe will contain the sum of the values in
            the predicate columns of the input dataframe from the timestamp of the event in any given row plus
            the specified offset (``endpoint_expr[3]``) to the timestamp of the given row plus the offset plus
            the window size (``endpoint_expr[1]``), less either the start or end of the window pending the
            ``left_inclusive`` (``endpoint_expr[0]``) and ``right_inclusive`` (``endpoint_expr[2]``) values.
            The sum of the predicate columns over the rows in the original dataframe spanning each row's
            ``timestamp + offset`` to each row's ``timestamp + offset + window_size`` should be exactly equal
            to the values in the predicate columns of the output dataframe (again, less the left or right
            inclusive values as specified).

    Examples:
        >>> import polars as pl
        >>> _ = pl.Config.set_tbl_width_chars(100)
        >>> from datetime import datetime, timedelta
        >>> df = pl.DataFrame({
        ...     "subject_id": [1, 1, 1, 1, 2, 2],
        ...     "timestamp": [
        ...         # Subject 1
        ...         datetime(year=1989, month=12, day=1, hour=12, minute=3),
        ...         datetime(year=1989, month=12, day=2, hour=5,  minute=17),
        ...         datetime(year=1989, month=12, day=2, hour=12, minute=3),
        ...         datetime(year=1989, month=12, day=6, hour=11, minute=0),
        ...         # Subject 2
        ...         datetime(year=1989, month=12, day=1, hour=13, minute=14),
        ...         datetime(year=1989, month=12, day=3, hour=15, minute=17),
        ...     ],
        ...     "is_A": [1, 0, 1, 0, 0, 0],
        ...     "is_B": [0, 1, 0, 1, 1, 0],
        ...     "is_C": [1, 1, 0, 0, 1, 0],
        ... })
        >>> aggregate_temporal_window(df, TemporalWindowBounds(True, timedelta(days=7), True, None))
        shape: (6, 5)
        ┌────────────┬─────────────────────┬──────┬──────┬──────┐
        │ subject_id ┆ timestamp           ┆ is_A ┆ is_B ┆ is_C │
        │ ---        ┆ ---                 ┆ ---  ┆ ---  ┆ ---  │
        │ i64        ┆ datetime[μs]        ┆ u16  ┆ u16  ┆ u16  │
        ╞════════════╪═════════════════════╪══════╪══════╪══════╡
        │ 1          ┆ 1989-12-01 12:03:00 ┆ 2    ┆ 2    ┆ 2    │
        │ 1          ┆ 1989-12-02 05:17:00 ┆ 1    ┆ 2    ┆ 1    │
        │ 1          ┆ 1989-12-02 12:03:00 ┆ 1    ┆ 1    ┆ 0    │
        │ 1          ┆ 1989-12-06 11:00:00 ┆ 0    ┆ 1    ┆ 0    │
        │ 2          ┆ 1989-12-01 13:14:00 ┆ 0    ┆ 1    ┆ 1    │
        │ 2          ┆ 1989-12-03 15:17:00 ┆ 0    ┆ 0    ┆ 0    │
        └────────────┴─────────────────────┴──────┴──────┴──────┘
        >>> aggregate_temporal_window(df, (True, timedelta(days=1), True, timedelta(days=0)))
        shape: (6, 5)
        ┌────────────┬─────────────────────┬──────┬──────┬──────┐
        │ subject_id ┆ timestamp           ┆ is_A ┆ is_B ┆ is_C │
        │ ---        ┆ ---                 ┆ ---  ┆ ---  ┆ ---  │
        │ i64        ┆ datetime[μs]        ┆ u16  ┆ u16  ┆ u16  │
        ╞════════════╪═════════════════════╪══════╪══════╪══════╡
        │ 1          ┆ 1989-12-01 12:03:00 ┆ 2    ┆ 1    ┆ 2    │
        │ 1          ┆ 1989-12-02 05:17:00 ┆ 1    ┆ 1    ┆ 1    │
        │ 1          ┆ 1989-12-02 12:03:00 ┆ 1    ┆ 0    ┆ 0    │
        │ 1          ┆ 1989-12-06 11:00:00 ┆ 0    ┆ 1    ┆ 0    │
        │ 2          ┆ 1989-12-01 13:14:00 ┆ 0    ┆ 1    ┆ 1    │
        │ 2          ┆ 1989-12-03 15:17:00 ┆ 0    ┆ 0    ┆ 0    │
        └────────────┴─────────────────────┴──────┴──────┴──────┘
        >>> aggregate_temporal_window(df, (True, timedelta(days=1), False, timedelta(days=0)))
        shape: (6, 5)
        ┌────────────┬─────────────────────┬──────┬──────┬──────┐
        │ subject_id ┆ timestamp           ┆ is_A ┆ is_B ┆ is_C │
        │ ---        ┆ ---                 ┆ ---  ┆ ---  ┆ ---  │
        │ i64        ┆ datetime[μs]        ┆ u16  ┆ u16  ┆ u16  │
        ╞════════════╪═════════════════════╪══════╪══════╪══════╡
        │ 1          ┆ 1989-12-01 12:03:00 ┆ 1    ┆ 1    ┆ 2    │
        │ 1          ┆ 1989-12-02 05:17:00 ┆ 1    ┆ 1    ┆ 1    │
        │ 1          ┆ 1989-12-02 12:03:00 ┆ 1    ┆ 0    ┆ 0    │
        │ 1          ┆ 1989-12-06 11:00:00 ┆ 0    ┆ 1    ┆ 0    │
        │ 2          ┆ 1989-12-01 13:14:00 ┆ 0    ┆ 1    ┆ 1    │
        │ 2          ┆ 1989-12-03 15:17:00 ┆ 0    ┆ 0    ┆ 0    │
        └────────────┴─────────────────────┴──────┴──────┴──────┘
        >>> aggregate_temporal_window(df, (False, timedelta(days=1), False, timedelta(days=0)))
        shape: (6, 5)
        ┌────────────┬─────────────────────┬──────┬──────┬──────┐
        │ subject_id ┆ timestamp           ┆ is_A ┆ is_B ┆ is_C │
        │ ---        ┆ ---                 ┆ ---  ┆ ---  ┆ ---  │
        │ i64        ┆ datetime[μs]        ┆ u16  ┆ u16  ┆ u16  │
        ╞════════════╪═════════════════════╪══════╪══════╪══════╡
        │ 1          ┆ 1989-12-01 12:03:00 ┆ 0    ┆ 1    ┆ 1    │
        │ 1          ┆ 1989-12-02 05:17:00 ┆ 1    ┆ 0    ┆ 0    │
        │ 1          ┆ 1989-12-02 12:03:00 ┆ 0    ┆ 0    ┆ 0    │
        │ 1          ┆ 1989-12-06 11:00:00 ┆ 0    ┆ 0    ┆ 0    │
        │ 2          ┆ 1989-12-01 13:14:00 ┆ 0    ┆ 0    ┆ 0    │
        │ 2          ┆ 1989-12-03 15:17:00 ┆ 0    ┆ 0    ┆ 0    │
        └────────────┴─────────────────────┴──────┴──────┴──────┘
        >>> aggregate_temporal_window(df, (False, timedelta(days=-1), False, timedelta(days=0)))
        shape: (6, 5)
        ┌────────────┬─────────────────────┬──────┬──────┬──────┐
        │ subject_id ┆ timestamp           ┆ is_A ┆ is_B ┆ is_C │
        │ ---        ┆ ---                 ┆ ---  ┆ ---  ┆ ---  │
        │ i64        ┆ datetime[μs]        ┆ u16  ┆ u16  ┆ u16  │
        ╞════════════╪═════════════════════╪══════╪══════╪══════╡
        │ 1          ┆ 1989-12-01 12:03:00 ┆ 0    ┆ 0    ┆ 0    │
        │ 1          ┆ 1989-12-02 05:17:00 ┆ 1    ┆ 0    ┆ 1    │
        │ 1          ┆ 1989-12-02 12:03:00 ┆ 0    ┆ 1    ┆ 1    │
        │ 1          ┆ 1989-12-06 11:00:00 ┆ 0    ┆ 0    ┆ 0    │
        │ 2          ┆ 1989-12-01 13:14:00 ┆ 0    ┆ 0    ┆ 0    │
        │ 2          ┆ 1989-12-03 15:17:00 ┆ 0    ┆ 0    ┆ 0    │
        └────────────┴─────────────────────┴──────┴──────┴──────┘
        >>> aggregate_temporal_window(df, (False, timedelta(hours=12), False, timedelta(hours=12)))
        shape: (6, 5)
        ┌────────────┬─────────────────────┬──────┬──────┬──────┐
        │ subject_id ┆ timestamp           ┆ is_A ┆ is_B ┆ is_C │
        │ ---        ┆ ---                 ┆ ---  ┆ ---  ┆ ---  │
        │ i64        ┆ datetime[μs]        ┆ u16  ┆ u16  ┆ u16  │
        ╞════════════╪═════════════════════╪══════╪══════╪══════╡
        │ 1          ┆ 1989-12-01 12:03:00 ┆ 0    ┆ 1    ┆ 1    │
        │ 1          ┆ 1989-12-02 05:17:00 ┆ 0    ┆ 0    ┆ 0    │
        │ 1          ┆ 1989-12-02 12:03:00 ┆ 0    ┆ 0    ┆ 0    │
        │ 1          ┆ 1989-12-06 11:00:00 ┆ 0    ┆ 0    ┆ 0    │
        │ 2          ┆ 1989-12-01 13:14:00 ┆ 0    ┆ 0    ┆ 0    │
        │ 2          ┆ 1989-12-03 15:17:00 ┆ 0    ┆ 0    ┆ 0    │
        └────────────┴─────────────────────┴──────┴──────┴──────┘
        >>> # Note that left_inclusive and right_inclusive are relative to the temporal ordering of the window
        >>> # and not the timestamp of the row. E.g., if left_inclusive is False, the window will not include
        >>> # the earliest event in the aggregation window, regardless of whether that is earlier than the
        >>> # timestamp of the row.
        >>> aggregate_temporal_window(df, (False, timedelta(days=-1), True, timedelta(days=1)))
        shape: (6, 5)
        ┌────────────┬─────────────────────┬──────┬──────┬──────┐
        │ subject_id ┆ timestamp           ┆ is_A ┆ is_B ┆ is_C │
        │ ---        ┆ ---                 ┆ ---  ┆ ---  ┆ ---  │
        │ i64        ┆ datetime[μs]        ┆ u16  ┆ u16  ┆ u16  │
        ╞════════════╪═════════════════════╪══════╪══════╪══════╡
        │ 1          ┆ 1989-12-01 12:03:00 ┆ 1    ┆ 1    ┆ 1    │
        │ 1          ┆ 1989-12-02 05:17:00 ┆ 1    ┆ 0    ┆ 0    │
        │ 1          ┆ 1989-12-02 12:03:00 ┆ 0    ┆ 0    ┆ 0    │
        │ 1          ┆ 1989-12-06 11:00:00 ┆ 0    ┆ 0    ┆ 0    │
        │ 2          ┆ 1989-12-01 13:14:00 ┆ 0    ┆ 0    ┆ 0    │
        │ 2          ┆ 1989-12-03 15:17:00 ┆ 0    ┆ 0    ┆ 0    │
        └────────────┴─────────────────────┴──────┴──────┴──────┘
        >>> aggregate_temporal_window(df, (True, timedelta(days=-1), False, timedelta(days=1)))
        shape: (6, 5)
        ┌────────────┬─────────────────────┬──────┬──────┬──────┐
        │ subject_id ┆ timestamp           ┆ is_A ┆ is_B ┆ is_C │
        │ ---        ┆ ---                 ┆ ---  ┆ ---  ┆ ---  │
        │ i64        ┆ datetime[μs]        ┆ u16  ┆ u16  ┆ u16  │
        ╞════════════╪═════════════════════╪══════╪══════╪══════╡
        │ 1          ┆ 1989-12-01 12:03:00 ┆ 1    ┆ 1    ┆ 2    │
        │ 1          ┆ 1989-12-02 05:17:00 ┆ 1    ┆ 1    ┆ 1    │
        │ 1          ┆ 1989-12-02 12:03:00 ┆ 1    ┆ 0    ┆ 0    │
        │ 1          ┆ 1989-12-06 11:00:00 ┆ 0    ┆ 1    ┆ 0    │
        │ 2          ┆ 1989-12-01 13:14:00 ┆ 0    ┆ 1    ┆ 1    │
        │ 2          ┆ 1989-12-03 15:17:00 ┆ 0    ┆ 0    ┆ 0    │
        └────────────┴─────────────────────┴──────┴──────┴──────┘
    """
    if not isinstance(endpoint_expr, TemporalWindowBounds):
        endpoint_expr = TemporalWindowBounds(*endpoint_expr)

    predicate_cols = [c for c in predicates_df.columns if c not in {"subject_id", "timestamp"}]

    return (
        predicates_df.rolling(
            index_column="timestamp",
            group_by="subject_id",
            **endpoint_expr.polars_gp_rolling_kwargs,
        )
        .agg(*[pl.col(c).sum().cast(PRED_CNT_TYPE).alias(c) for c in predicate_cols])
        .sort(by=["subject_id", "timestamp"])
    )


def aggregate_event_bound_window(
    predicates_df: pl.LazyFrame | pl.DataFrame,
    endpoint_expr: ToEventWindowBounds | tuple[bool, str, bool, timedelta | None],
) -> pl.LazyFrame | pl.DataFrame:
    """Aggregates ``predicates_df`` between each successive row and the next per-subject matching event.

    TODO: Use https://hypothesis.readthedocs.io/en/latest/quickstart.html to test this function.

    Args:
        predicates_df: The dataframe containing the predicates. The input must be sorted in ascending order by
            timestamp within each subject group. It must contain the following columns:
              - A column ``subject_id`` which contains the subject ID.
              - A column ``timestamp`` which contains the timestamp at which the event contained in any given
                row occurred.
              - A set of "predicate" columns that contain counts of the number of times a given predicate
                is satisfied in the event contained in any given row.
        endpoint_expr: The expression defining the event bound window endpoints. Can be specified as a tuple
            or a ToEventWindowBounds object, which is just a named tuple of the expected form. Said expected
            form is as follows:
              - The first element is a boolean indicating whether the start of the window is inclusive.
              - The second element is a string indicating the name of the column in which non-zero counts
                indicate the event is a valid "end event" of the window.
              - The third element is a boolean indicating whether the end of the window is inclusive.
              - The fourth element is a timedelta object indicating the offset from the timestamp of the row
                to the start of the window. The offset here can _only_ be positive.

    Returns:
        The dataframe that has been aggregated in an event bound manner according to the specified
        ``endpoint_expr``. This aggregation means the following:
          - The output dataframe will contain the same number of rows and be in the same order (in terms of
            ``subject_id`` and ``timestamp``) as the input dataframe.
          - The column names will also be the same as the input dataframe, plus there will be one additional
            column, ``timestamp_at_end`` that contains the timestamp of the end of the aggregation window for
            each row (or null if no such aggregation window exists).
          - The values in the predicate columns of the output dataframe will contain the sum of the values in
            the predicate columns of the input dataframe from the timestamp of the event in any given row plus
            the specified offset (``endpoint_expr[3]``) to the timestamp of the next row for that subject in
            the input dataframe such that the specified event predicate (``endpoint_expr[1]``) has a value
            greater than zero for that patient, less either the start or end of the window pending the
            ``left_inclusive`` (``endpoint_expr[0]``) and ``right_inclusive`` (``endpoint_expr[2]``) values.
            If there is no valid "next row" for a given event, the values in the predicate columns of the
            output dataframe will be 0, as the sum of an empty set is 0.
            The sum of the predicate columns over the rows in the original dataframe spanning each row's
            ``timestamp + offset`` to each row's ``timestamp_at_end`` should be exactly equal to the values in
            the predicate columns of the output dataframe.

    Raises:
        ValueError: If the offset is negative.

    Examples:
        >>> import polars as pl
        >>> _ = pl.Config.set_tbl_width_chars(100)
        >>> from datetime import datetime
        >>> df = pl.DataFrame({
        ...     "subject_id": [1, 1, 1, 2, 2, 2, 2, 2],
        ...     "timestamp": [
        ...         # Subject 1
        ...         datetime(year=1989, month=12, day=1,  hour=12, minute=3),
        ...         datetime(year=1989, month=12, day=3,  hour=13, minute=14),
        ...         datetime(year=1989, month=12, day=5,  hour=15, minute=17),
        ...         # Subject 2
        ...         datetime(year=1989, month=12, day=2,  hour=12, minute=3),
        ...         datetime(year=1989, month=12, day=4,  hour=13, minute=14),
        ...         datetime(year=1989, month=12, day=6,  hour=15, minute=17),
        ...         datetime(year=1989, month=12, day=8,  hour=16, minute=22),
        ...         datetime(year=1989, month=12, day=10, hour=3,  minute=7),
        ...     ],
        ...     "is_A": [1, 0, 1, 1, 1, 1, 0, 0],
        ...     "is_B": [0, 1, 0, 1, 0, 1, 1, 1],
        ...     "is_C": [0, 1, 0, 0, 0, 1, 0, 1],
        ... })
        >>> aggregate_event_bound_window(df, ToEventWindowBounds(True, "is_C", True, timedelta(days=-1)))
        Traceback (most recent call last):
            ...
        ValueError: offset must be non-negative. Got -1 day, 0:00:00
        >>> aggregate_event_bound_window(df, ToEventWindowBounds(True, "is_C", True, None))
        shape: (8, 6)
        ┌────────────┬─────────────────────┬─────────────────────┬──────┬──────┬──────┐
        │ subject_id ┆ timestamp           ┆ timestamp_at_end    ┆ is_A ┆ is_B ┆ is_C │
        │ ---        ┆ ---                 ┆ ---                 ┆ ---  ┆ ---  ┆ ---  │
        │ i64        ┆ datetime[μs]        ┆ datetime[μs]        ┆ i64  ┆ i64  ┆ i64  │
        ╞════════════╪═════════════════════╪═════════════════════╪══════╪══════╪══════╡
        │ 1          ┆ 1989-12-01 12:03:00 ┆ 1989-12-03 13:14:00 ┆ 1    ┆ 1    ┆ 1    │
        │ 1          ┆ 1989-12-03 13:14:00 ┆ null                ┆ 0    ┆ 0    ┆ 0    │
        │ 1          ┆ 1989-12-05 15:17:00 ┆ null                ┆ 0    ┆ 0    ┆ 0    │
        │ 2          ┆ 1989-12-02 12:03:00 ┆ 1989-12-06 15:17:00 ┆ 3    ┆ 2    ┆ 1    │
        │ 2          ┆ 1989-12-04 13:14:00 ┆ 1989-12-06 15:17:00 ┆ 2    ┆ 1    ┆ 1    │
        │ 2          ┆ 1989-12-06 15:17:00 ┆ 1989-12-10 03:07:00 ┆ 1    ┆ 3    ┆ 2    │
        │ 2          ┆ 1989-12-08 16:22:00 ┆ 1989-12-10 03:07:00 ┆ 0    ┆ 2    ┆ 1    │
        │ 2          ┆ 1989-12-10 03:07:00 ┆ null                ┆ 0    ┆ 0    ┆ 0    │
        └────────────┴─────────────────────┴─────────────────────┴──────┴──────┴──────┘
        >>> aggregate_event_bound_window(df, ToEventWindowBounds(True, "is_C", False, None))
        shape: (8, 6)
        ┌────────────┬─────────────────────┬─────────────────────┬──────┬──────┬──────┐
        │ subject_id ┆ timestamp           ┆ timestamp_at_end    ┆ is_A ┆ is_B ┆ is_C │
        │ ---        ┆ ---                 ┆ ---                 ┆ ---  ┆ ---  ┆ ---  │
        │ i64        ┆ datetime[μs]        ┆ datetime[μs]        ┆ i64  ┆ i64  ┆ i64  │
        ╞════════════╪═════════════════════╪═════════════════════╪══════╪══════╪══════╡
        │ 1          ┆ 1989-12-01 12:03:00 ┆ 1989-12-03 13:14:00 ┆ 1    ┆ 0    ┆ 0    │
        │ 1          ┆ 1989-12-03 13:14:00 ┆ null                ┆ 0    ┆ 0    ┆ 0    │
        │ 1          ┆ 1989-12-05 15:17:00 ┆ null                ┆ 0    ┆ 0    ┆ 0    │
        │ 2          ┆ 1989-12-02 12:03:00 ┆ 1989-12-06 15:17:00 ┆ 2    ┆ 1    ┆ 0    │
        │ 2          ┆ 1989-12-04 13:14:00 ┆ 1989-12-06 15:17:00 ┆ 1    ┆ 0    ┆ 0    │
        │ 2          ┆ 1989-12-06 15:17:00 ┆ 1989-12-10 03:07:00 ┆ 1    ┆ 2    ┆ 1    │
        │ 2          ┆ 1989-12-08 16:22:00 ┆ 1989-12-10 03:07:00 ┆ 0    ┆ 1    ┆ 0    │
        │ 2          ┆ 1989-12-10 03:07:00 ┆ null                ┆ 0    ┆ 0    ┆ 0    │
        └────────────┴─────────────────────┴─────────────────────┴──────┴──────┴──────┘
        >>> aggregate_event_bound_window(df, ToEventWindowBounds(False, "is_C", True, None))
        shape: (8, 6)
        ┌────────────┬─────────────────────┬─────────────────────┬──────┬──────┬──────┐
        │ subject_id ┆ timestamp           ┆ timestamp_at_end    ┆ is_A ┆ is_B ┆ is_C │
        │ ---        ┆ ---                 ┆ ---                 ┆ ---  ┆ ---  ┆ ---  │
        │ i64        ┆ datetime[μs]        ┆ datetime[μs]        ┆ i64  ┆ i64  ┆ i64  │
        ╞════════════╪═════════════════════╪═════════════════════╪══════╪══════╪══════╡
        │ 1          ┆ 1989-12-01 12:03:00 ┆ 1989-12-03 13:14:00 ┆ 0    ┆ 1    ┆ 1    │
        │ 1          ┆ 1989-12-03 13:14:00 ┆ null                ┆ 0    ┆ 0    ┆ 0    │
        │ 1          ┆ 1989-12-05 15:17:00 ┆ null                ┆ 0    ┆ 0    ┆ 0    │
        │ 2          ┆ 1989-12-02 12:03:00 ┆ 1989-12-06 15:17:00 ┆ 2    ┆ 1    ┆ 1    │
        │ 2          ┆ 1989-12-04 13:14:00 ┆ 1989-12-06 15:17:00 ┆ 1    ┆ 1    ┆ 1    │
        │ 2          ┆ 1989-12-06 15:17:00 ┆ 1989-12-10 03:07:00 ┆ 0    ┆ 2    ┆ 1    │
        │ 2          ┆ 1989-12-08 16:22:00 ┆ 1989-12-10 03:07:00 ┆ 0    ┆ 1    ┆ 1    │
        │ 2          ┆ 1989-12-10 03:07:00 ┆ null                ┆ 0    ┆ 0    ┆ 0    │
        └────────────┴─────────────────────┴─────────────────────┴──────┴──────┴──────┘
        >>> aggregate_event_bound_window(df, ToEventWindowBounds(True, "is_C", True, timedelta(days=3)))
        shape: (8, 6)
        ┌────────────┬─────────────────────┬─────────────────────┬──────┬──────┬──────┐
        │ subject_id ┆ timestamp           ┆ timestamp_at_end    ┆ is_A ┆ is_B ┆ is_C │
        │ ---        ┆ ---                 ┆ ---                 ┆ ---  ┆ ---  ┆ ---  │
        │ i64        ┆ datetime[μs]        ┆ datetime[μs]        ┆ i64  ┆ i64  ┆ i64  │
        ╞════════════╪═════════════════════╪═════════════════════╪══════╪══════╪══════╡
        │ 1          ┆ 1989-12-01 12:03:00 ┆ null                ┆ 0    ┆ 0    ┆ 0    │
        │ 1          ┆ 1989-12-03 13:14:00 ┆ null                ┆ 0    ┆ 0    ┆ 0    │
        │ 1          ┆ 1989-12-05 15:17:00 ┆ null                ┆ 0    ┆ 0    ┆ 0    │
        │ 2          ┆ 1989-12-02 12:03:00 ┆ 1989-12-06 15:17:00 ┆ 1    ┆ 1    ┆ 1    │
        │ 2          ┆ 1989-12-04 13:14:00 ┆ 1989-12-10 03:07:00 ┆ 0    ┆ 2    ┆ 1    │
        │ 2          ┆ 1989-12-06 15:17:00 ┆ 1989-12-10 03:07:00 ┆ 0    ┆ 1    ┆ 1    │
        │ 2          ┆ 1989-12-08 16:22:00 ┆ null                ┆ 0    ┆ 0    ┆ 0    │
        │ 2          ┆ 1989-12-10 03:07:00 ┆ null                ┆ 0    ┆ 0    ┆ 0    │
        └────────────┴─────────────────────┴─────────────────────┴──────┴──────┴──────┘
    """
    if not isinstance(endpoint_expr, ToEventWindowBounds):
        endpoint_expr = ToEventWindowBounds(*endpoint_expr)

    predicate_cols = [c for c in predicates_df.columns if c not in {"subject_id", "timestamp"}]

    is_end_event = pl.col(endpoint_expr.end_event) > 0
    timestamp_at_next_end = (
        pl.when(is_end_event)
        .then(pl.col("timestamp"))
        .shift(-1)
        .fill_null(strategy="backward")
        .alias("timestamp_at_next_end")
    )

    cumsum_cols = {c: pl.col(c).cum_sum().alias(f"{c}_cumsum") for c in predicate_cols}

    if endpoint_expr.right_inclusive:
        cumsum_at_next_end = {
            c: (
                pl.when(is_end_event)
                .then(expr)
                .shift(-1)
                .fill_null(strategy="backward")
                .alias(f"{c}_cumsum_at_next_end")
            )
            for c, expr in cumsum_cols.items()
        }
    else:
        cumsum_at_next_end = {
            c: (
                pl.when(is_end_event)
                .then(expr - pl.col(c))
                .shift(-1)
                .fill_null(strategy="backward")
                .alias(f"{c}_cumsum_at_next_end")
            )
            for c, expr in cumsum_cols.items()
        }

    def summarize_predicate(predicate: str) -> pl.Expr:
        cumsum_col = pl.col(f"{predicate}_cumsum")
        cumsum_at_next_end_col = pl.col(f"{predicate}_cumsum_at_next_end")
        raw_val_col = pl.col(predicate)

        if endpoint_expr.offset == timedelta(0):
            if endpoint_expr.left_inclusive:
                out_val = cumsum_at_next_end_col - cumsum_col + raw_val_col
            else:
                out_val = cumsum_at_next_end_col - cumsum_col
        else:
            raise NotImplementedError

        return out_val.fill_null(0).alias(predicate)

    aggd_df_no_offset = (
        predicates_df.group_by("subject_id", maintain_order=True)
        .agg(
            "timestamp",
            timestamp_at_next_end,
            *cumsum_cols.values(),
            *cumsum_at_next_end.values(),
            *predicate_cols,
        )
        .explode(
            "timestamp",
            "timestamp_at_next_end",
            *(f"{c}_cumsum" for c in predicate_cols),
            *(f"{c}_cumsum_at_next_end" for c in predicate_cols),
            *predicate_cols,
        )
        .select(
            "subject_id",
            "timestamp",
            pl.col("timestamp_at_next_end").alias("timestamp_at_end"),
            *(summarize_predicate(c) for c in predicate_cols),
        )
    )

    if endpoint_expr.offset == timedelta(0):
        return aggd_df_no_offset
    else:
        raise NotImplementedError


def summarize_temporal_window(
    predicates_df: pl.LazyFrame | pl.DataFrame,
    predicate_cols: list[str],
    endpoint_expr: (tuple[bool, timedelta, bool, timedelta] | tuple[bool, str, bool, timedelta]),
    anchor_to_subtree_root_by_subtree_anchor: pl.LazyFrame | pl.DataFrame,
) -> pl.LazyFrame | pl.DataFrame:
    """Summarizes the temporal window based on the given predicates and anchor-to-subtree-root mapping.

    TODO: Significantly clarify docstring!!!

    Args:
        predicates_df: The dataframe containing the predicates. The input must be sorted in ascending order by
            timestamp within each subject group. It must contain the following columns:
              - A column ``subject_id`` which contains the subject ID.
              - A column ``timestamp`` which contains the timestamp at which the event contained in any given
                row occurred.
              - A set of "predicate" columns that contain counts of the number of times a given predicate
                is satisfied in the event contained in any given row.
        predicate_cols: The list of predicate columns.
        endpoint_expr: The expression defining the temporal window endpoints. Can be specified as a tuple or a
            TemporalWindowBounds object, which is just a named tuple of the expected form. Said expected form
            is as follows:
              - The first element is a boolean indicating whether the start of the window is inclusive.
              - The second element is a timedelta object indicating the size of the window.
              - The third element is a boolean indicating whether the end of the window is inclusive.
              - The fourth element is a timedelta object indicating the offset from the timestamp of the row
                to the start of the window.
        anchor_to_subtree_root_by_subtree_anchor: A dataframe with a row for each possible realization of an
            anchor event for the subtree, with the corresponding row timestamp of the associated subtree
            root event. This is used to define the starting points of possible valid window starts for the
            eventual child anchor events.

    Returns:
        The summarized dataframe.

    Examples:
        >>> import polars as pl
        >>> _ = pl.Config.set_tbl_width_chars(100)
        >>> from datetime import datetime, timedelta
        >>> predicates_df = pl.DataFrame(
        ...     {
        ...         "subject_id": [1, 1, 1],
        ...         "timestamp": [
        ...             datetime(year=1989, month=12, day=1, hour=12, minute=3),
        ...             datetime(year=1989, month=12, day=1, hour=13, minute=14),
        ...             datetime(year=1989, month=12, day=1, hour=15, minute=17),
        ...         ],
        ...         "is_A": [1, 0, 1],
        ...         "is_B": [0, 1, 0],
        ...     }
        ... )
        >>> anchor_to_subtree_root_by_subtree_anchor = pl.DataFrame(
        ...     {
        ...         "subject_id": [1, 1, 1],
        ...         "timestamp": [
        ...             datetime(year=1989, month=12, day=1, hour=12, minute=3),
        ...             datetime(year=1989, month=12, day=1, hour=13, minute=14),
        ...             datetime(year=1989, month=12, day=1, hour=15, minute=17),
        ...         ],
        ...         "timestamp_at_anchor": [
        ...             datetime(year=1989, month=12, day=1, hour=12, minute=3),
        ...             datetime(year=1989, month=12, day=1, hour=13, minute=14),
        ...             datetime(year=1989, month=12, day=1, hour=15, minute=17),
        ...         ],
        ...         "is_A": [0, 0, 0],
        ...         "is_B": [0, 0, 0],
        ...     }
        ... )
        >>> predicate_cols = ["is_A", "is_B"]
        >>> endpoint_expr = (True, timedelta(days=1), True, timedelta(days=0))
        >>> summarize_temporal_window(
        ...     predicates_df,
        ...     predicate_cols,
        ...     endpoint_expr,
        ...     anchor_to_subtree_root_by_subtree_anchor,
        ... )
        shape: (3, 5)
        ┌────────────┬─────────────────────┬─────────────────────┬──────┬──────┐
        │ subject_id ┆ timestamp           ┆ timestamp_at_anchor ┆ is_A ┆ is_B │
        │ ---        ┆ ---                 ┆ ---                 ┆ ---  ┆ ---  │
        │ i64        ┆ datetime[μs]        ┆ datetime[μs]        ┆ i64  ┆ i64  │
        ╞════════════╪═════════════════════╪═════════════════════╪══════╪══════╡
        │ 1          ┆ 1989-12-01 12:03:00 ┆ 1989-12-01 12:03:00 ┆ 2    ┆ 1    │
        │ 1          ┆ 1989-12-01 13:14:00 ┆ 1989-12-01 13:14:00 ┆ 1    ┆ 1    │
        │ 1          ┆ 1989-12-01 15:17:00 ┆ 1989-12-01 15:17:00 ┆ 1    ┆ 0    │
        └────────────┴─────────────────────┴─────────────────────┴──────┴──────┘
    """

    return (
        aggregate_temporal_window(predicates_df, endpoint_expr)
        .join(
            anchor_to_subtree_root_by_subtree_anchor,
            on=["subject_id", "timestamp"],
            suffix="_summary",
        )
        .with_columns(
            "subject_id",
            "timestamp",
            pl.col("timestamp").alias("timestamp_at_anchor"),
            *[pl.col(c).fill_null(strategy="zero") for c in predicate_cols],
        )
        .select(
            "subject_id",
            "timestamp",
            "timestamp_at_anchor",
            *[(pl.col(c) - pl.col(f"{c}_summary")).alias(c) for c in predicate_cols],
        )
    )


def summarize_event_bound_window(
    predicates_df: pl.LazyFrame | pl.DataFrame,
    predicate_cols: list[str],
    endpoint_expr: (tuple[bool, timedelta, bool, timedelta] | tuple[bool, str, bool, timedelta]),
    anchor_to_subtree_root_by_subtree_anchor: pl.LazyFrame | pl.DataFrame,
) -> pl.LazyFrame | pl.DataFrame:
    """Summarizes the event bound window based on the given predicates and anchor-to-subtree-root mapping.

    Given an input dataframe of event-level predicate counts and a dataframe keyed by subtree anchor events,
    this function computes the counts of predicates that have occurred between successive pairs of subtree
    anchor and prospective child anchor events (as dictated by `endpoint_expr`), keyed by the child anchor.
    This function will use any given child anchor node only once, as only the nearest possible subtree anchor
    event will be considered for each child anchor event. However, the same subtree anchor event can be used
    multiple times for different child anchor events, resulting in potentially overlapping windows that have
    the same start time and different end times. In the event, however, of multiple possible subtree anchors
    that could be used for a given child anchor, only the one nearest to the subtree anchor will be used.

    Args:
        predicates_df: The dataframe containing the predicates. The input must be sorted in ascending order by
            timestamp within each subject group. It must contain the following columns:
              - A column ``subject_id`` which contains the subject ID.
              - A column ``timestamp`` which contains the timestamp at which the event contained in any given
                row occurred.
              - A set of "predicate" columns that contain counts of the number of times a given predicate
                is satisfied in the event contained in any given row.
        predicate_cols: The list of predicate columns.
        endpoint_expr: The expression defining the event bound window endpoints. Can be specified as a tuple
            or a ToEventWindowBounds object, which is just a named tuple of the expected form. Said expected
            form is as follows:
              - The first element is a boolean indicating whether the start of the window is inclusive.
              - The second element is a string indicating the name of the column in which non-zero counts
                indicate the event is a valid "end event" of the window.
              - The third element is a boolean indicating whether the end of the window is inclusive.
              - The fourth element is a timedelta object indicating the offset from the timestamp of the row
                to the start of the window.
        anchor_to_subtree_root_by_subtree_anchor: A dataframe with a row for each possible realization of an
            anchor event for the subtree, with the corresponding row timestamp of the associated subtree
            root event. This is used to define the starting points of possible valid window starts for the
            eventual child anchor events.

    Returns:
        The summarized dataframe.

    Examples:
        >>> import polars as pl
        >>> _ = pl.Config.set_tbl_width_chars(100)
        >>> from datetime import datetime
        >>> predicates_df = pl.DataFrame(
        ...     {
        ...         "subject_id": [1, 1, 1, 2, 2, 2, 2, 2],
        ...         "timestamp": [
        ...             # Subject 1
        ...             datetime(year=1989, month=12, day=1, hour=12, minute=3),
        ...             datetime(year=1989, month=12, day=3, hour=13, minute=14),
        ...             datetime(year=1989, month=12, day=5, hour=15, minute=17),
        ...             # Subject 2
        ...             datetime(year=1989, month=12, day=2, hour=12, minute=3),
        ...             datetime(year=1989, month=12, day=4, hour=13, minute=14),
        ...             datetime(year=1989, month=12, day=6, hour=15, minute=17),
        ...             datetime(year=1989, month=12, day=8, hour=15, minute=17),
        ...             datetime(year=1989, month=12, day=10, hour=15, minute=17),
        ...         ],
        ...         "is_A": [1, 0, 1, 1, 1, 1, 0, 0],
        ...         "is_B": [0, 1, 0, 1, 0, 1, 1, 1],
        ...         "is_C": [0, 1, 0, 0, 0, 1, 0, 1],
        ...     }
        ... )
        >>> anchor_to_subtree_root_by_subtree_anchor = pl.DataFrame(
        ...     {
        ...         "subject_id": [1, 1, 1, 2, 2, 2, 2, 2],
        ...         "timestamp": [
        ...             # Subject 1
        ...             datetime(year=1989, month=12, day=1, hour=12, minute=3),
        ...             datetime(year=1989, month=12, day=3, hour=13, minute=14),
        ...             datetime(year=1989, month=12, day=5, hour=15, minute=17),
        ...             # Subject 2
        ...             datetime(year=1989, month=12, day=2, hour=12, minute=3),
        ...             datetime(year=1989, month=12, day=4, hour=13, minute=14),
        ...             datetime(year=1989, month=12, day=6, hour=15, minute=17),
        ...             datetime(year=1989, month=12, day=8, hour=15, minute=17),
        ...             datetime(year=1989, month=12, day=10, hour=15, minute=17),
        ...         ],
        ...         "timestamp_at_anchor": [
        ...             # Subject 1
        ...             datetime(year=1989, month=12, day=1, hour=12, minute=3),
        ...             datetime(year=1989, month=12, day=3, hour=13, minute=14),
        ...             datetime(year=1989, month=12, day=5, hour=15, minute=17),
        ...             # Subject 2
        ...             datetime(year=1989, month=12, day=2, hour=12, minute=3),
        ...             datetime(year=1989, month=12, day=4, hour=13, minute=14),
        ...             datetime(year=1989, month=12, day=6, hour=15, minute=17),
        ...             datetime(year=1989, month=12, day=8, hour=15, minute=17),
        ...             datetime(year=1989, month=12, day=10, hour=15, minute=17),
        ...         ],
        ...         "is_A": [0, 0, 0, 0, 0, 0, 0, 0],
        ...         "is_B": [0, 0, 0, 0, 0, 0, 0, 0],
        ...         "is_C": [0, 0, 0, 0, 0, 0, 0, 0],
        ...     }
        ... )
        >>> predicate_cols = ["is_A", "is_B", "is_C"]
        >>> endpoint_expr = (True, "is_C", True, timedelta(days=0))
        >>> summarize_event_bound_window(
        ...     predicates_df,
        ...     predicate_cols,
        ...     endpoint_expr,
        ...     anchor_to_subtree_root_by_subtree_anchor,
        ... )
        shape: (3, 6)
        ┌────────────┬─────────────────────┬─────────────────────┬──────┬──────┬──────┐
        │ subject_id ┆ timestamp           ┆ timestamp_at_anchor ┆ is_A ┆ is_B ┆ is_C │
        │ ---        ┆ ---                 ┆ ---                 ┆ ---  ┆ ---  ┆ ---  │
        │ i64        ┆ datetime[μs]        ┆ datetime[μs]        ┆ i64  ┆ i64  ┆ i64  │
        ╞════════════╪═════════════════════╪═════════════════════╪══════╪══════╪══════╡
        │ 1          ┆ 1989-12-03 13:14:00 ┆ 1989-12-03 13:14:00 ┆ 0    ┆ 1    ┆ 1    │
        │ 2          ┆ 1989-12-06 15:17:00 ┆ 1989-12-06 15:17:00 ┆ 1    ┆ 1    ┆ 1    │
        │ 2          ┆ 1989-12-10 15:17:00 ┆ 1989-12-10 15:17:00 ┆ 0    ┆ 1    ┆ 1    │
        └────────────┴─────────────────────┴─────────────────────┴──────┴──────┴──────┘
        >>> anchor_to_subtree_root_by_subtree_anchor = pl.DataFrame(
        ...     {
        ...         "subject_id": [1, 2],
        ...         "timestamp": [
        ...             # Subject 1
        ...             datetime(year=1989, month=12, day=1, hour=12, minute=3),
        ...             # Subject 2
        ...             datetime(year=1989, month=12, day=2, hour=12, minute=3),
        ...         ],
        ...         "timestamp_at_anchor": [
        ...             # Subject 1
        ...             datetime(year=1989, month=12, day=1, hour=12, minute=3),
        ...             # Subject 2
        ...             datetime(year=1989, month=12, day=2, hour=12, minute=3),
        ...         ],
        ...         "is_A": [0, 0],
        ...         "is_B": [0, 0],
        ...         "is_C": [0, 0],
        ...     }
        ... )
        >>> predicate_cols = ["is_A", "is_B", "is_C"]
        >>> endpoint_expr = (True, "is_C", True, timedelta(days=0))
        >>> summarize_event_bound_window(
        ...     predicates_df,
        ...     predicate_cols,
        ...     endpoint_expr,
        ...     anchor_to_subtree_root_by_subtree_anchor,
        ... )
        shape: (3, 6)
        ┌────────────┬─────────────────────┬─────────────────────┬──────┬──────┬──────┐
        │ subject_id ┆ timestamp           ┆ timestamp_at_anchor ┆ is_A ┆ is_B ┆ is_C │
        │ ---        ┆ ---                 ┆ ---                 ┆ ---  ┆ ---  ┆ ---  │
        │ i64        ┆ datetime[μs]        ┆ datetime[μs]        ┆ i64  ┆ i64  ┆ i64  │
        ╞════════════╪═════════════════════╪═════════════════════╪══════╪══════╪══════╡
        │ 1          ┆ 1989-12-03 13:14:00 ┆ 1989-12-01 12:03:00 ┆ 1    ┆ 1    ┆ 1    │
        │ 2          ┆ 1989-12-06 15:17:00 ┆ 1989-12-02 12:03:00 ┆ 3    ┆ 2    ┆ 1    │
        │ 2          ┆ 1989-12-10 15:17:00 ┆ 1989-12-02 12:03:00 ┆ 3    ┆ 4    ┆ 2    │
        └────────────┴─────────────────────┴─────────────────────┴──────┴──────┴──────┘
    """
    left_inclusive, end_event, right_inclusive, offset = endpoint_expr

    if not offset:
        offset = timedelta(days=0)

    # overall cumulative sum of predicates
    cumsum_predicates_df = predicates_df.with_columns(
        *[pl.col(c).cum_sum().over(pl.col("subject_id")).alias(f"{c}_cumsum") for c in predicate_cols],
    )
    cumsum_predicates_df = cumsum_predicates_df.select(
        "subject_id",
        "timestamp",
        *[pl.col(f"{c}") for c in predicate_cols],
        *[pl.col(f"{c}_cumsum") for c in predicate_cols],
    )

    # get the counts at the anchor
    cnts_at_anchor = (
        anchor_to_subtree_root_by_subtree_anchor.select("subject_id", "timestamp")
        .join(cumsum_predicates_df, on=["subject_id", "timestamp"], how="left")
        .select(
            "subject_id",
            "timestamp",
            pl.col("timestamp").alias("timestamp_at_anchor"),
            *[pl.col(c).alias(f"{c}_at_anchor") for c in predicate_cols],
            *[pl.col(f"{c}_cumsum").alias(f"{c}_cumsum_at_anchor") for c in predicate_cols],
        )
    )

    # get the counts at the child anchor
    cumsum_predicates_df = cumsum_predicates_df.join(
        cnts_at_anchor, on=["subject_id", "timestamp"], how="left"
    ).with_columns(
        pl.col("timestamp_at_anchor").forward_fill().over("subject_id"),
        *[pl.col(f"{c}_at_anchor").forward_fill().over("subject_id") for c in predicate_cols],
        *[pl.col(f"{c}_cumsum_at_anchor").forward_fill().over("subject_id") for c in predicate_cols],
    )
    cumsum_anchor_child = cumsum_predicates_df.with_columns(
        "subject_id",
        "timestamp",
        *[
            (pl.col(f"{c}_cumsum") - pl.col(f"{c}_cumsum_at_anchor")).alias(f"{c}_final")
            for c in predicate_cols
        ],
    )

    # left_inclusive and right_inclusive handling
    if left_inclusive:
        cumsum_anchor_child = cumsum_anchor_child.with_columns(
            "subject_id",
            "timestamp",
            *[(pl.col(f"{c}_final") + pl.col(f"{c}_at_anchor")) for c in predicate_cols],
        )
    if not right_inclusive:
        cumsum_anchor_child = cumsum_anchor_child.with_columns(
            "subject_id",
            "timestamp",
            *[(pl.col(f"{c}_final") - pl.col(f"{c}")) for c in predicate_cols],
        )

    # filter and clean up
    at_child_anchor = cumsum_anchor_child.select(
        "subject_id",
        "timestamp",
        "timestamp_at_anchor",
        *[pl.col(f"{c}_final").alias(c) for c in predicate_cols],
    )

    at_child_anchor = at_child_anchor.with_columns(
        *[pl.when(pl.col(c) < 0).then(0).otherwise(pl.col(c)).alias(c) for c in predicate_cols]
    )

    filtered_by_end_event_at_child_anchor = (
        predicates_df.filter(pl.col(end_event) >= 1)
        .join(at_child_anchor, on=["subject_id", "timestamp"], how="inner")
        .select(
            "subject_id",
            "timestamp",
            "timestamp_at_anchor",
            *[pl.col(f"{c}_right").alias(c) for c in predicate_cols],
        )
    )

    return filtered_by_end_event_at_child_anchor.join(
        anchor_to_subtree_root_by_subtree_anchor,
        left_on=["subject_id", "timestamp_at_anchor"],
        right_on=["subject_id", "timestamp"],
        suffix="_summary",
    ).select(
        "subject_id",
        "timestamp",
        "timestamp_at_anchor",
        *[(pl.col(c) - pl.col(f"{c}_summary")).alias(c) for c in predicate_cols],
    )


def summarize_window(
    child: Any,
    anchor_to_subtree_root_by_subtree_anchor: pl.DataFrame,
    predicates_df: pl.DataFrame | pl.LazyFrame,
    predicate_cols: list[str],
) -> pl.DataFrame | pl.LazyFrame:
    """Summarizes the window based on the given child, anchor-to-subtree-root mapping, predicates, and
    predicate columns. At end of this process, we have rows corresponding to possible end events (anchors for
    child) with counts of predicates that have occurred since the subtree root, already having handled
    subtracting the anchor to subtree root component.

    Args:
        child: The child node.
        anchor_to_subtree_root_by_subtree_anchor: The mapping of anchor to subtree root.
        predicates_df: The dataframe containing the predicates.
        predicate_cols: The list of predicate columns.

    Returns:
        subtree_root_to_child_root_by_child_anchor: Dataframe with a row for every possible realization
            of the anchor of the subtree rooted by _child_ (not the input subtree)
            with the counts occurring between subtree_root and the child
    """
    match child.endpoint_expr[1]:
        case timedelta():
            fn = summarize_temporal_window
        case str():
            fn = summarize_event_bound_window
        case _:
            raise ValueError(f"Invalid endpoint expression: {child.endpoint_expr}")

    return fn(
        predicates_df,
        predicate_cols,
        child.endpoint_expr,
        anchor_to_subtree_root_by_subtree_anchor,
    )


def check_constraints(window_constraints, summary_df):
    """Checks the constraints on the counts of predicates in the summary dataframe.

    Args:
        window_constraints: constraints on counts of predicates that must
            be satsified.
        summary_df: A dataframe containing a row for every possible realization

    Return: A filtered dataframe containing only the rows that satisfy the constraints.

    Raises:
        ValueError: If the constraint for a column is empty.
    """
    valid_exprs = []
    for col, (cnt_ge, cnt_le) in window_constraints.items():
        if cnt_ge is None and cnt_le is None:
            raise ValueError(f"Empty constraint for {col}!")

        if col == "*":
            col = "__ALL_EVENTS"

        if cnt_ge is not None:
            valid_exprs.append(pl.col(col) >= cnt_ge)
        if cnt_le is not None:
            valid_exprs.append(pl.col(col) <= cnt_le)

    if not valid_exprs:
        valid_exprs.append(pl.lit(True))

    # log filtered subjects
    summary_df_shape = summary_df.shape[0]
    for condition in valid_exprs:
        dropped = summary_df.filter(~condition)
        summary_df = summary_df.filter(condition)
        if summary_df.shape[0] < summary_df_shape:
            logger.debug(
                f"{dropped['subject_id'].unique().shape[0]} subjects ({dropped.shape[0]} rows) "
                f"were excluded due to constraint: {condition}."
            )
            summary_df_shape = summary_df.shape[0]

    return summary_df


def summarize_subtree(
    subtree,
    anchor_to_subtree_root_by_subtree_anchor: pl.DataFrame | None,
    predicates_df: pl.DataFrame,
    anchor_offset: float,
):
    """

    Args:
        subtree: The subtree object.
        anchor_to_subtree_root_by_subtree_anchor: The mapping of anchor to subtree root.
        predicates_df: The dataframe containing the predicates.
        anchor_offset: The anchor offset.

    Returns:
        The queried dataframe.

    Raises:
        None.
    """
    """Queries the subtree based on the given subtree, anchor-to-subtree-root mapping, predicates, and anchor
    offset.

    Args:
        subtree:
            Subtree object.

        anchor_to_subtree_root_by_subtree_anchor: A dataframe with a row for each possible
            realization of the anchoring node for this subtree containing the
            counts of predicates that have occurred from the anchoring node to
            the subtree root for that realization of `subtree.root`.

            # First iteration:
            subj_id, ts,  is_admission, is_discharge, pred_C
            1,       1,   0,            0,            0
            1,       10,  0,            0,            0
            1,       26,  0,            0,            0
            1,       33,  0,            0,            0
            1,       81,  0,            0,            0
            1,       88,  0,            0,            0
            1,       89,  0,            0,            0
            1,       122, 0,            0,            0

            # Example:
            (admission_event)
            |
            24h
            |
            (node_A)
            |
            to_discharge
            |
            (node_B)
            |
            36h
            |
            (node_C)

            predicates_df: A dataframe containing a row for every event for every
            subject with the following columns:

            - A column ``subject_id`` which contains the subject ID.
            - A column ``timestamp`` which contains the timestamp at which the
                event contained in any given row occurred.
            - A set of "predicate" columns that contain counts of the
                number of times a given predicate is satisfied in the
                event contained in any given row.

        anchor_offset: The sum of all timedelta edges between subtree_root and
            the anchor node for this subtree. It is 0 for the first iteration.

        Returns: A dataframe with a row corresponding to the anchor event for each
            possible valid realization of this subtree (and all its children)
            containing the timestamp values realizing the nodes in this subtree in
            that realization.
    """
    predicate_cols = [col for col in predicates_df.columns if col.startswith("is_")]

    recursive_results = []

    for child in subtree.children:
        logger.debug(f"Summarizing subtree rooted at '{child.name}'...")

        # Added to reset anchor_offset and anchor_to_subtree_root_by_subtree_anchor for diverging subtrees
        # if len(child.parent.children) > 1:
        #     anchor_offset = timedelta(hours=0)
        #     anchor_to_subtree_root_by_subtree_anchor = (
        #         predicates_df.filter(predicates_df[child.parent.endpoint_expr[1]] == 1)
        #         .select("subject_id", "timestamp", *[pl.col(c) for c in predicate_cols])
        #         .with_columns("subject_id", "timestamp", *[pl.lit(0).alias(c) for c in predicate_cols])
        #     )

        # Step 1: Summarize the window from the subtree.root to child
        subtree_root_to_child_root_by_child_anchor = summarize_window(
            child,
            anchor_to_subtree_root_by_subtree_anchor,
            predicates_df,
            predicate_cols,
        )

        # subtree_root_to_child_root_by_child_anchor has a row for every possible realization of the anchor
        # of the subtree rooted by _child_ (not the input subtree) with the counts occurring between
        # subtree_root and the child

        # Step 2: Filter to where constraints are valid
        subtree_root_to_child_root_by_child_anchor = check_constraints(
            child.constraints, subtree_root_to_child_root_by_child_anchor
        )

        # Step 3: Update parameters for recursive step
        match child.endpoint_expr[1]:
            # if time-bounded window
            case timedelta():
                # account for offset
                anchor_offset_branch = anchor_offset + child.endpoint_expr[1] + child.endpoint_expr[3]
                joined = anchor_to_subtree_root_by_subtree_anchor.join(
                    subtree_root_to_child_root_by_child_anchor,
                    on=["subject_id", "timestamp"],
                    suffix="_summary",
                )
                anchor_to_subtree_root_by_subtree_anchor_branch = joined.select(
                    "subject_id",
                    "timestamp",
                    *[pl.col(c) + pl.col(f"{c}_summary") for c in predicate_cols],
                )
            # if event-bounded window
            case str():
                # reset offset
                anchor_offset_branch = timedelta(days=0) + child.endpoint_expr[3]
                joined = anchor_to_subtree_root_by_subtree_anchor.join(
                    subtree_root_to_child_root_by_child_anchor,
                    left_on=["subject_id", "timestamp"],
                    right_on=["subject_id", "timestamp_at_anchor"],
                    suffix="_summary",
                )
                anchor_to_subtree_root_by_subtree_anchor_branch = joined.select(
                    "subject_id",
                    "timestamp_summary",
                    *[pl.col(c) + pl.col(f"{c}_summary") for c in predicate_cols],
                ).rename({"timestamp_summary": "timestamp"})

                anchor_to_subtree_root_by_subtree_anchor_branch = (
                    anchor_to_subtree_root_by_subtree_anchor_branch.with_columns(
                        "subject_id",
                        "timestamp",
                        *[pl.lit(0).alias(c) for c in predicate_cols],
                    )
                )

        # Can try to move some joins before recursive call to reduce memory for Inovalon but stable for MIMIC
        # for some reason

        # Step 4: Recurse
        recursive_result = summarize_subtree(
            child,
            anchor_to_subtree_root_by_subtree_anchor_branch,
            predicates_df,
            anchor_offset_branch,
        )

        # Update timestamp for recursive result
        match child.endpoint_expr[1]:
            case timedelta():
                recursive_result = recursive_result.with_columns(
                    (pl.col("timestamp") + anchor_offset_branch).alias(f"{child.name}/timestamp")
                )
            case str():
                recursive_result = recursive_result.with_columns(
                    pl.col("timestamp").alias(f"{child.name}/timestamp")
                )

        # Step 5: Push results back to subtree anchor
        subtree_root_to_child_root_by_child_anchor = subtree_root_to_child_root_by_child_anchor.with_columns(
            pl.struct([pl.col(c).alias(c) for c in predicate_cols]).alias(f"{child.name}/window_summary")
        )

        match child.endpoint_expr[1]:
            case timedelta():
                final_recursive_result = recursive_result.join(
                    subtree_root_to_child_root_by_child_anchor.select(
                        "subject_id", "timestamp", f"{child.name}/window_summary"
                    ),
                    on=["subject_id", "timestamp"],
                )
            case str():
                # Need a dataframe with one col with a "True" in the possible realizations of subtree anchor
                # and another col with a "True" in the possible valid corresponding realizations of the child
                # node.
                # Make this with anchor_to_subtree_root_by_subtree_anchor_branch (contains rows corresponding
                # to possible start events) and recursive_result (contains rows corresponding to possible end
                # events).
                final_recursive_result = (
                    recursive_result.join(
                        subtree_root_to_child_root_by_child_anchor.select(
                            "subject_id",
                            "timestamp",
                            "timestamp_at_anchor",
                            f"{child.name}/window_summary",
                        ),
                        on=["subject_id", "timestamp"],
                    )
                    .drop("timestamp")
                    .rename({"timestamp_at_anchor": "timestamp"})
                )

        recursive_results.append(final_recursive_result)

    # Step 6: Join children recursive results where all children find a valid realization
    if not recursive_results:
        all_children = anchor_to_subtree_root_by_subtree_anchor.select("subject_id", "timestamp")
    else:
        all_children = recursive_results[0]
        for df in recursive_results[1:]:
            all_children = all_children.join(df, on=["subject_id", "timestamp"], how="inner")

    return all_children
