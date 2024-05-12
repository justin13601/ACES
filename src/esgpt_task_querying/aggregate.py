"""This module contains the functions for aggregating windows over conditional row bounds."""

from datetime import timedelta

import polars as pl

from .types import PRED_CNT_TYPE, TemporalWindowBounds, ToEventWindowBounds


def aggregate_temporal_window(
    predicates_df: pl.DataFrame,
    endpoint_expr: TemporalWindowBounds | tuple[bool, timedelta, bool, timedelta | None],
) -> pl.DataFrame:
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
        ┌────────────┬─────────────────────┬─────────────────────┬──────┬──────┬──────┐
        │ subject_id ┆ timestamp           ┆ timestamp_at_end    ┆ is_A ┆ is_B ┆ is_C │
        │ ---        ┆ ---                 ┆ ---                 ┆ ---  ┆ ---  ┆ ---  │
        │ i64        ┆ datetime[μs]        ┆ datetime[μs]        ┆ u16  ┆ u16  ┆ u16  │
        ╞════════════╪═════════════════════╪═════════════════════╪══════╪══════╪══════╡
        │ 1          ┆ 1989-12-01 12:03:00 ┆ 1989-12-08 12:03:00 ┆ 2    ┆ 2    ┆ 2    │
        │ 1          ┆ 1989-12-02 05:17:00 ┆ 1989-12-09 05:17:00 ┆ 1    ┆ 2    ┆ 1    │
        │ 1          ┆ 1989-12-02 12:03:00 ┆ 1989-12-09 12:03:00 ┆ 1    ┆ 1    ┆ 0    │
        │ 1          ┆ 1989-12-06 11:00:00 ┆ 1989-12-13 11:00:00 ┆ 0    ┆ 1    ┆ 0    │
        │ 2          ┆ 1989-12-01 13:14:00 ┆ 1989-12-08 13:14:00 ┆ 0    ┆ 1    ┆ 1    │
        │ 2          ┆ 1989-12-03 15:17:00 ┆ 1989-12-10 15:17:00 ┆ 0    ┆ 0    ┆ 0    │
        └────────────┴─────────────────────┴─────────────────────┴──────┴──────┴──────┘
        >>> aggregate_temporal_window(df, (True, timedelta(days=1), True, timedelta(days=0)))
        shape: (6, 5)
        ┌────────────┬─────────────────────┬─────────────────────┬──────┬──────┬──────┐
        │ subject_id ┆ timestamp           ┆ timestamp_at_end    ┆ is_A ┆ is_B ┆ is_C │
        │ ---        ┆ ---                 ┆ ---                 ┆ ---  ┆ ---  ┆ ---  │
        │ i64        ┆ datetime[μs]        ┆ datetime[μs]        ┆ u16  ┆ u16  ┆ u16  │
        ╞════════════╪═════════════════════╪═════════════════════╪══════╪══════╪══════╡
        │ 1          ┆ 1989-12-01 12:03:00 ┆ 1989-12-02 12:03:00 ┆ 2    ┆ 1    ┆ 2    │
        │ 1          ┆ 1989-12-02 05:17:00 ┆ 1989-12-03 05:17:00 ┆ 1    ┆ 1    ┆ 1    │
        │ 1          ┆ 1989-12-02 12:03:00 ┆ 1989-12-03 12:03:00 ┆ 1    ┆ 0    ┆ 0    │
        │ 1          ┆ 1989-12-06 11:00:00 ┆ 1989-12-07 11:00:00 ┆ 0    ┆ 1    ┆ 0    │
        │ 2          ┆ 1989-12-01 13:14:00 ┆ 1989-12-02 13:14:00 ┆ 0    ┆ 1    ┆ 1    │
        │ 2          ┆ 1989-12-03 15:17:00 ┆ 1989-12-04 15:17:00 ┆ 0    ┆ 0    ┆ 0    │
        └────────────┴─────────────────────┴─────────────────────┴──────┴──────┴──────┘
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
        .select(
            "subject_id",
            "timestamp",
            (pl.col("timestamp") + endpoint_expr.offset + endpoint_expr.window_size).alias(
                "timestamp_at_end"
            ),
            *predicate_cols,
        )
    )


def aggregate_event_bound_window(
    predicates_df: pl.DataFrame,
    endpoint_expr: ToEventWindowBounds | tuple[bool, str, bool, timedelta | None],
) -> pl.DataFrame:
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
        ...         datetime(year=1989, month=12, day=3,  hour=13, minute=14), # HAS EVENT BOUND
        ...         datetime(year=1989, month=12, day=5,  hour=15, minute=17),
        ...         # Subject 2
        ...         datetime(year=1989, month=12, day=2,  hour=12, minute=3),
        ...         datetime(year=1989, month=12, day=4,  hour=13, minute=14),
        ...         datetime(year=1989, month=12, day=6,  hour=15, minute=17), # HAS EVENT BOUND
        ...         datetime(year=1989, month=12, day=8,  hour=16, minute=22),
        ...         datetime(year=1989, month=12, day=10, hour=3,  minute=7),  # HAS EVENT BOUND
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
        │ i64        ┆ datetime[μs]        ┆ datetime[μs]        ┆ u16  ┆ u16  ┆ u16  │
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
        │ i64        ┆ datetime[μs]        ┆ datetime[μs]        ┆ u16  ┆ u16  ┆ u16  │
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
        │ i64        ┆ datetime[μs]        ┆ datetime[μs]        ┆ u16  ┆ u16  ┆ u16  │
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
        │ i64        ┆ datetime[μs]        ┆ datetime[μs]        ┆ u16  ┆ u16  ┆ u16  │
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

    if endpoint_expr.offset == timedelta(0):
        return _aggregate_event_bound_window_no_offset(predicates_df, endpoint_expr)
    else:
        return _aggregate_event_bound_window_with_offset(predicates_df, endpoint_expr)


def _aggregate_event_bound_window_with_offset(
    predicates_df: pl.DataFrame,
    endpoint_expr: ToEventWindowBounds,
) -> pl.DataFrame:
    """See aggregate_event_bound_window -- this case specifically has a positive offset.

    Note this function assumes that no time gaps in the original data are less than or equal to one
    microsecond.
    """

    time_aggd_to_subtract = aggregate_temporal_window(
        predicates_df,
        TemporalWindowBounds(
            left_inclusive=False,
            window_size=endpoint_expr.offset,
            right_inclusive=(not endpoint_expr.left_inclusive),
            offset=None,
        ),
    )

    predicate_cols = [c for c in predicates_df.columns if c not in {"subject_id", "timestamp"}]

    predicates_df = (
        predicates_df.lazy()
        .collect()
        .with_columns(*(pl.col(c).cum_sum().over("subject_id").alias(f"{c}_cumsum") for c in predicate_cols))
    )

    is_end_event = pl.col(endpoint_expr.end_event) > 0

    if endpoint_expr.right_inclusive:
        end_vals_for_sub = {c: pl.col(f"{c}_cumsum") for c in predicate_cols}
    else:
        end_vals_for_sub = {c: pl.col(f"{c}_cumsum") - pl.col(c) for c in predicate_cols}

    at_end_df = predicates_df.filter(is_end_event).select(
        "subject_id",
        (pl.col("timestamp") - endpoint_expr.offset - timedelta(seconds=1e-6)).alias("timestamp"),
        pl.col("timestamp").alias("timestamp_at_end"),
        *(expr.alias(f"{c}_at_end") for c, expr in end_vals_for_sub.items()),
    )

    with_offset_period = (
        pl.concat(
            [
                predicates_df.with_columns(
                    pl.lit(True).alias("is_real"),
                    pl.lit(None, dtype=at_end_df.schema["timestamp_at_end"]).alias("timestamp_at_end"),
                    *(
                        pl.lit(None, dtype=at_end_df.schema[f"{c}_at_end"]).alias(f"{c}_at_end")
                        for c in predicate_cols
                    ),
                ),
                at_end_df.with_columns(
                    pl.lit(False).alias("is_real"),
                    *(
                        pl.lit(None, dtype=predicates_df.schema[f"{c}_cumsum"]).alias(f"{c}_cumsum")
                        for c in predicate_cols
                    ),
                ),
            ],
            how="diagonal",
        )
        .sort(by=["subject_id", "timestamp"])
        .select(
            "subject_id",
            "timestamp",
            pl.col("timestamp_at_end").fill_null(strategy="backward").over("subject_id"),
            *(
                (
                    pl.col(f"{c}_at_end").fill_null(strategy="backward").over("subject_id")
                    - pl.col(f"{c}_cumsum")
                ).alias(c)
                for c in predicate_cols
            ),
            "is_real",
        )
        .filter("is_real")
        .drop("is_real")
    )

    return with_offset_period.join(
        time_aggd_to_subtract,
        on=["subject_id", "timestamp"],
        how="left",
        suffix="_in_offset_period",
    ).select(
        "subject_id",
        "timestamp",
        "timestamp_at_end",
        *(
            (pl.col(c) - pl.col(f"{c}_in_offset_period")).fill_null(0).cast(PRED_CNT_TYPE).alias(c)
            for c in predicate_cols
        ),
    )


def _aggregate_event_bound_window_no_offset(
    predicates_df: pl.DataFrame,
    endpoint_expr: ToEventWindowBounds,
) -> pl.DataFrame:
    """See aggregate_event_bound_window -- this case specifically has no offset."""

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

        if endpoint_expr.left_inclusive:
            out_val = cumsum_at_next_end_col - cumsum_col + pl.col(predicate)
        else:
            out_val = cumsum_at_next_end_col - cumsum_col

        return out_val.fill_null(0).cast(PRED_CNT_TYPE).alias(predicate)

    return (
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
