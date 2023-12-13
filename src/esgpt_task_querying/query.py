"""This module contains the functions for querying the ESGPT task."""

from datetime import timedelta
from typing import Any

import polars as pl


def summarize_temporal_window(
    predicates_df: pl.LazyFrame | pl.DataFrame,
    predicate_cols: list[str],
    endpoint_expr: tuple[bool, timedelta, bool, timedelta] | tuple[bool, str, bool, timedelta],
    anchor_to_subtree_root_by_subtree_anchor: pl.LazyFrame | pl.DataFrame,
) -> pl.LazyFrame | pl.DataFrame:
    """Summarizes the temporal window based on the given predicates and anchor-to-subtree-root mapping.

    Args:
        predicates_df: The dataframe containing the predicates.
        predicate_cols: The list of predicate columns.
        endpoint_expr: The expression defining the temporal window endpoints.
        anchor_to_subtree_root_by_subtree_anchor: The mapping of anchor to subtree root.

    Returns:
        The summarized dataframe.
    """
    st_inclusive, window_size, end_inclusive, offset = endpoint_expr
    if not offset:
        offset = timedelta(days=0)

    if st_inclusive and end_inclusive:
        closed = "both"
    elif st_inclusive:
        closed = "left"
    elif end_inclusive:
        closed = "right"
    else:
        closed = "none"

    if window_size < timedelta(days=0):
        period = -window_size
        offset = -period + offset
    else:
        period = window_size
        offset = timedelta(days=0) + offset

    result = (
        predicates_df.rolling(
            index_column="timestamp",
            by="subject_id",
            closed=closed,
            period=period,
            offset=offset,
        )
        .agg([pl.col(c).sum().alias(c) for c in predicate_cols])
        .sort(by=["subject_id", "timestamp"])
    )

    filtered_result = result.join(
        anchor_to_subtree_root_by_subtree_anchor,
        on=["subject_id", "timestamp"],
        suffix="_summary",
    ).with_columns(pl.col("timestamp").alias("timestamp_at_anchor"))

    filtered_result = filtered_result.select(
        "subject_id",
        "timestamp",
        "timestamp_at_anchor",
        *[
            pl.col(c)
            for c in filtered_result.columns
            if c not in ["subject_id", "timestamp", "timestamp_at_anchor"]
        ],
    )

    return filtered_result


def summarize_event_bound_window(
    predicates_df: pl.LazyFrame | pl.DataFrame,
    predicate_cols: list[str],
    endpoint_expr: tuple[bool, timedelta, bool, timedelta] | tuple[bool, str, bool, timedelta],
    anchor_to_subtree_root_by_subtree_anchor: pl.LazyFrame | pl.DataFrame,
) -> pl.LazyFrame | pl.DataFrame:
    """Summarizes the event-bound window based on the given predicates and anchor-to-subtree-root mapping.

    Args:
        predicates_df: The dataframe containing the predicates.
        predicate_cols: The list of predicate columns.
        endpoint_expr: The expression defining the event-bound window endpoints.
        anchor_to_subtree_root_by_subtree_anchor: The mapping of anchor to subtree root.

    Returns:
        The summarized dataframe.
    """
    st_inclusive, end_event, end_inclusive, offset = endpoint_expr
    if not offset:
        offset = timedelta(days=0)

    cumsum_predicates_df = predicates_df.with_columns(
        *[pl.col(c).cum_sum().over(pl.col("subject_id")).alias(f"{c}_cumsum") for c in predicate_cols],
    )

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

    cumsum_predicates_df = cumsum_predicates_df.select(
        "subject_id",
        "timestamp",
        *[pl.col(f"{c}") for c in predicate_cols],
        *[pl.col(f"{c}_cumsum") for c in predicate_cols],
    )

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

    if not st_inclusive:
        cumsum_anchor_child = cumsum_anchor_child.with_columns(
            "subject_id",
            "timestamp",
            *[(pl.col(f"{c}_final") - pl.col(f"{c}_at_anchor")) for c in predicate_cols],
        )
    if not end_inclusive:
        cumsum_anchor_child = cumsum_anchor_child.with_columns(
            "subject_id",
            "timestamp",
            *[(pl.col(f"{c}_final") - pl.col(f"{c}")) for c in predicate_cols],
        )

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

    filtered_result = filtered_by_end_event_at_child_anchor.join(
        anchor_to_subtree_root_by_subtree_anchor,
        left_on=["subject_id", "timestamp_at_anchor"],
        right_on=["subject_id", "timestamp"],
        suffix="_summary",
    )

    return filtered_result


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
            subtree_anchor_to_child_root_by_child_anchor = summarize_temporal_window(
                predicates_df,
                predicate_cols,
                child.endpoint_expr,
                anchor_to_subtree_root_by_subtree_anchor,
            )

        case str():
            subtree_anchor_to_child_root_by_child_anchor = summarize_event_bound_window(
                predicates_df,
                predicate_cols,
                child.endpoint_expr,
                anchor_to_subtree_root_by_subtree_anchor,
            )

    subtree_root_to_child_root_by_child_anchor = subtree_anchor_to_child_root_by_child_anchor.select(
        "subject_id",
        "timestamp",
        "timestamp_at_anchor",
        *[pl.col(c) - pl.col(f"{c}_summary") for c in predicate_cols],
    )

    return subtree_root_to_child_root_by_child_anchor


def check_constraints(window_constraints, summary_df):
    """Checks the constraints on the counts of predicates in the summary dataframe.

    Args:
        window_constraints: constraints on counts of predicates that must
            be satisfied.
        summary_df: contains counts of times predicates are satisfied in windows
            anchored at the rows in question of the dataframe.

    Return: A column or expression that evaluates to True or False for each row
        depending on whether or not the constraints therein are met.

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

    return pl.all_horizontal(valid_exprs)


def query_subtree(
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
        print(f"Querying subtree rooted at {child.name}...")

        if isinstance(child.parent.endpoint_expr[1], str):
            anchor_offset = timedelta(hours=0)
        print(anchor_offset)

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
        valid_windows = check_constraints(child.constraints, subtree_root_to_child_root_by_child_anchor)

        # Step 3: Update parameters for recursive step
        match child.endpoint_expr[1]:
            case timedelta():
                anchor_offset += child.endpoint_expr[1] + child.endpoint_expr[3]
                joined = anchor_to_subtree_root_by_subtree_anchor.join(
                    subtree_root_to_child_root_by_child_anchor,
                    on=["subject_id", "timestamp"],
                    suffix="_summary",
                )
                anchor_to_subtree_root_by_subtree_anchor = joined.select(
                    "subject_id",
                    "timestamp",
                    *[pl.col(c) + pl.col(f"{c}_summary") for c in predicate_cols],
                )
                anchor_to_subtree_root_by_subtree_anchor = anchor_to_subtree_root_by_subtree_anchor.filter(
                    valid_windows
                )
            case str():
                anchor_offset = timedelta(days=0) + child.endpoint_expr[3]
                joined = anchor_to_subtree_root_by_subtree_anchor.join(
                    subtree_root_to_child_root_by_child_anchor,
                    left_on=["subject_id", "timestamp"],
                    right_on=["subject_id", "timestamp_at_anchor"],
                    suffix="_summary",
                )
                anchor_to_subtree_root_by_subtree_anchor = joined.select(
                    "subject_id",
                    "timestamp_summary",
                    *[pl.col(c) + pl.col(f"{c}_summary") for c in predicate_cols],
                ).rename({"timestamp_summary": "timestamp"})
                anchor_to_subtree_root_by_subtree_anchor = anchor_to_subtree_root_by_subtree_anchor.filter(
                    valid_windows
                )
                anchor_to_subtree_root_by_subtree_anchor = (
                    anchor_to_subtree_root_by_subtree_anchor.with_columns(
                        "subject_id",
                        "timestamp",
                        *[pl.lit(0).alias(c) for c in predicate_cols],
                    )
                )

        # Step 4: Recurse
        recursive_result = query_subtree(
            child,
            anchor_to_subtree_root_by_subtree_anchor,
            predicates_df,
            anchor_offset,
        )

        match child.endpoint_expr[1]:
            case timedelta():
                recursive_result = recursive_result.with_columns(
                    (pl.col("timestamp") + anchor_offset).alias(f"{child.name}/timestamp")
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
                # Make this with anchor_to_subtree_root_by_subtree_anchor (contains rows corresponding to
                # possible start events) and recursive_result (contains rows corresponding to possible end
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
