"""Contains utilities for validating that windows satisfy a set of constraints."""

import polars as pl
from loguru import logger

from .types import ANY_EVENT_COLUMN


def check_constraints(
    window_constraints: dict[str, tuple[int | None, int | None]], summary_df: pl.DataFrame
) -> pl.DataFrame:
    """Checks the constraints on the counts of predicates in the summary dataframe.

    Args:
        window_constraints: constraints on counts of predicates that must
            be satsified, organized as a dictionary from predicate column name to the lowerbound and upper
            bound range required for that constraint to be satisfied.
        summary_df: A dataframe containing a row for every possible prospective window to be analyzed. The
            only columns expected are predicate columns within the ``window_constraints`` dictionary.

    Returns: A filtered dataframe containing only the rows that satisfy the constraints.

    Raises:
        ValueError: If the constraint for a column is empty.

    Examples:
        >>> from datetime import datetime
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
        ...     "is_A": [1, 4, 1, 3, 3,  3],
        ...     "is_B": [0, 2, 0, 2, 10, 2],
        ...     "is_C": [1, 1, 1, 0, 1,  1],
        ... })
        >>> check_constraints({"is_A": (None, None), "is_B": (2, 6), "is_C": (1, 1)}, df)
        Traceback (most recent call last):
            ...
        ValueError: Invalid constraint for 'is_A': None - None
        >>> check_constraints({"is_A": (2, 1), "is_B": (2, 6), "is_C": (1, 1)}, df)
        Traceback (most recent call last):
            ...
        ValueError: Invalid constraint for 'is_A': 2 - 1
        >>> check_constraints({"is_A": (3, 4), "is_B": (2, 6), "is_C": (1, 1)}, df)
        shape: (2, 5)
        ┌────────────┬─────────────────────┬──────┬──────┬──────┐
        │ subject_id ┆ timestamp           ┆ is_A ┆ is_B ┆ is_C │
        │ ---        ┆ ---                 ┆ ---  ┆ ---  ┆ ---  │
        │ i64        ┆ datetime[μs]        ┆ i64  ┆ i64  ┆ i64  │
        ╞════════════╪═════════════════════╪══════╪══════╪══════╡
        │ 1          ┆ 1989-12-02 05:17:00 ┆ 4    ┆ 2    ┆ 1    │
        │ 2          ┆ 1989-12-03 15:17:00 ┆ 3    ┆ 2    ┆ 1    │
        └────────────┴─────────────────────┴──────┴──────┴──────┘
        >>> check_constraints({"is_A": (3, 4), "is_B": (2, None), "is_C": (None, 1)}, df)
        shape: (4, 5)
        ┌────────────┬─────────────────────┬──────┬──────┬──────┐
        │ subject_id ┆ timestamp           ┆ is_A ┆ is_B ┆ is_C │
        │ ---        ┆ ---                 ┆ ---  ┆ ---  ┆ ---  │
        │ i64        ┆ datetime[μs]        ┆ i64  ┆ i64  ┆ i64  │
        ╞════════════╪═════════════════════╪══════╪══════╪══════╡
        │ 1          ┆ 1989-12-02 05:17:00 ┆ 4    ┆ 2    ┆ 1    │
        │ 1          ┆ 1989-12-06 11:00:00 ┆ 3    ┆ 2    ┆ 0    │
        │ 2          ┆ 1989-12-01 13:14:00 ┆ 3    ┆ 10   ┆ 1    │
        │ 2          ┆ 1989-12-03 15:17:00 ┆ 3    ┆ 2    ┆ 1    │
        └────────────┴─────────────────────┴──────┴──────┴──────┘
    """

    should_drop = pl.lit(False)

    for col, (valid_min_inc, valid_max_inc) in window_constraints.items():
        if valid_min_inc is None and valid_max_inc is None:
            raise ValueError(f"Invalid constraint for '{col}': {valid_min_inc} - {valid_max_inc}")
        elif valid_min_inc is not None and valid_max_inc is not None and valid_max_inc < valid_min_inc:
            raise ValueError(f"Invalid constraint for '{col}': {valid_min_inc} - {valid_max_inc}")

        if col == "*":
            col = ANY_EVENT_COLUMN

        drop_expr = pl.lit(False)
        if valid_min_inc is not None:
            drop_expr = drop_expr | (pl.col(col) < valid_min_inc)
        if valid_max_inc is not None:
            drop_expr = drop_expr | (pl.col(col) > valid_max_inc)

        logger.info(
            f"Excluding {summary_df.select(drop_expr.sum()).item():,} rows "
            f"as they failed to satisfy '{valid_min_inc} <= {col} <= {valid_max_inc}'."
        )

        should_drop = should_drop | drop_expr

    return summary_df.filter(~should_drop)
