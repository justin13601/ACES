"""This module contains the main function for querying a task.

It accepts the configuration file and predicate columns, builds the tree, and recursively queries the tree.
"""

import logging

import polars as pl
from bigtree import preorder_iter

from .config import EventConfig, TaskExtractorConfig
from .constraints import check_constraints, check_static_variables
from .extract_subtree import extract_subtree
from .types import PRED_CNT_TYPE
from .utils import log_tree, parse_timedelta

logger = logging.getLogger(__name__)

TRIGGER_GRID_COLUMN = "_TRIGGER_GRID"


def add_trigger_grid(trigger: EventConfig, predicates_df: pl.DataFrame) -> pl.DataFrame:
    """Marks a row every ``trigger.every`` from each trigger event in a new ``TRIGGER_GRID_COLUMN`` column.

    The grid of each ``trigger.predicate`` event runs up to (but excluding) the next ``trigger.until`` event,
    or the end of the subject's record if there is none. Grid times that do not fall on an existing event
    are added as new rows with all predicate counts zero, so they do not change any window's counts.

    Examples:
        >>> predicates_df = pl.DataFrame({
        ...     "subject_id": [1, 1, 1, 1],
        ...     "timestamp": [datetime(2020, 1, 1, 0), datetime(2020, 1, 1, 2), datetime(2020, 1, 1, 5),
        ...                   datetime(2020, 1, 2, 0)],
        ...     "icu_admission": [1, 0, 0, 0],
        ...     "icu_discharge": [0, 0, 1, 0],
        ... })
        >>> trigger = EventConfig("icu_admission", every="2h", until="icu_discharge")
        >>> add_trigger_grid(trigger, predicates_df)
        shape: (5, 5)
        ┌────────────┬─────────────────────┬───────────────┬───────────────┬───────────────┐
        │ subject_id ┆ timestamp           ┆ icu_admission ┆ icu_discharge ┆ _TRIGGER_GRID │
        │ ---        ┆ ---                 ┆ ---           ┆ ---           ┆ ---           │
        │ i64        ┆ datetime[μs]        ┆ i64           ┆ i64           ┆ i64           │
        ╞════════════╪═════════════════════╪═══════════════╪═══════════════╪═══════════════╡
        │ 1          ┆ 2020-01-01 00:00:00 ┆ 1             ┆ 0             ┆ 1             │
        │ 1          ┆ 2020-01-01 02:00:00 ┆ 0             ┆ 0             ┆ 1             │
        │ 1          ┆ 2020-01-01 04:00:00 ┆ 0             ┆ 0             ┆ 1             │
        │ 1          ┆ 2020-01-01 05:00:00 ┆ 0             ┆ 1             ┆ 0             │
        │ 1          ┆ 2020-01-02 00:00:00 ┆ 0             ┆ 0             ┆ 0             │
        └────────────┴─────────────────────┴───────────────┴───────────────┴───────────────┘
    """
    grid_end = pl.col("timestamp").max().over("subject_id")
    if trigger.until:
        # The first `until` event strictly after each row, as rows are sorted with unique timestamps.
        next_stop = pl.when(pl.col(trigger.until) > 0).then(pl.col("timestamp")).shift(-1).backward_fill()
        grid_end = pl.coalesce(next_stop.over("subject_id"), grid_end)

    grid = (
        predicates_df.sort("subject_id", "timestamp")
        .with_columns(grid_end.alias("grid_end"))
        .filter(pl.col(trigger.predicate) > 0)
        .select(
            "subject_id",
            pl.datetime_ranges("timestamp", "grid_end", parse_timedelta(trigger.every), closed="left").alias(
                "timestamp"
            ),
        )
        .explode("timestamp")
        .drop_nulls()
        .unique()
        .with_columns(pl.lit(1, dtype=PRED_CNT_TYPE).alias(TRIGGER_GRID_COLUMN))
    )

    return (
        pl.concat([predicates_df, grid], how="diagonal_relaxed")
        .group_by("subject_id", "timestamp")
        .agg(pl.all().sum())
        .sort("subject_id", "timestamp")
    )


def query(cfg: TaskExtractorConfig, predicates_df: pl.DataFrame) -> pl.DataFrame:
    """Query a task using the provided configuration file and predicates dataframe.

    Args:
        cfg: TaskExtractorConfig object of the configuration file.
        predicates_df: Polars predicates dataframe.

    Returns:
        polars.DataFrame: The result of the task query, containing subjects who satisfy the conditions
            defined in cfg. Timestamps for the start/end boundaries of each window specified in the task
            configuration, as well as predicate counts for each window, are provided.

    Raises:
        TypeError: If predicates_df is not a polars.DataFrame.
        ValueError: If the (subject_id, timestamp) columns are not unique.

    Examples:
        >>> from .config import PlainPredicateConfig, WindowConfig, EventConfig

        >>> cfg = None # This is obviously invalid, but we're just testing the error case.
        >>> predicates_df = {"subject_id": [1, 1], "timestamp": [1, 1]}
        >>> query(cfg, predicates_df)
        Traceback (most recent call last):
            ...
        TypeError: Predicates dataframe type must be a polars.DataFrame. Got: <class 'dict'>.
        >>> query(cfg, pl.DataFrame(predicates_df))
        Traceback (most recent call last):
            ...
        ValueError: The (subject_id, timestamp) columns must be unique.
        >>> cfg = TaskExtractorConfig(
        ...     predicates={"A": PlainPredicateConfig("A")},
        ...     trigger=EventConfig("_ANY_EVENT"),
        ...     windows={
        ...         "pre": WindowConfig(None, "trigger", True, False, index_timestamp="start"),
        ...         "post": WindowConfig("pre.end", None, True, True, label="A"),
        ...     },
        ...     index_timestamp_window="pre",
        ...     label_window="post",
        ... )
        >>> predicates_df = pl.DataFrame({
        ...     "subject_id": [1, 1, 3],
        ...     "timestamp": [datetime(1980, 12, 28), datetime(2010, 6, 20), datetime(2010, 5, 11)],
        ...     "A": [False, False, False],
        ...     "_ANY_EVENT": [True, True, True],
        ... })
        >>> with caplog.at_level(logging.INFO):
        ...     result = query(cfg, predicates_df)
        >>> result.select("subject_id", "trigger")
        shape: (3, 2)
        ┌────────────┬─────────────────────┐
        │ subject_id ┆ trigger             │
        │ ---        ┆ ---                 │
        │ i64        ┆ datetime[μs]        │
        ╞════════════╪═════════════════════╡
        │ 1          ┆ 1980-12-28 00:00:00 │
        │ 1          ┆ 2010-06-20 00:00:00 │
        │ 3          ┆ 2010-05-11 00:00:00 │
        └────────────┴─────────────────────┘
        >>> "index_timestamp" in result.columns
        True
        >>> "label" in result.columns
        True

        A window may be defined backwards from its end (``start: end - 1 day, end: trigger``). The
        ``index_timestamp`` must still resolve to the boundary the config names -- ``end`` is the trigger,
        not the point a day earlier. See issue #196:

        >>> backward_df = pl.DataFrame({
        ...     "subject_id": [1],
        ...     "timestamp": [datetime(2010, 6, 20)],
        ...     "_ANY_EVENT": [True],
        ... })
        >>> def index_timestamp_for(index_timestamp: str) -> datetime:
        ...     '''Run a backward-defined input window and return its resolved index_timestamp.'''
        ...     cfg = TaskExtractorConfig(
        ...         predicates={},
        ...         trigger=EventConfig("_ANY_EVENT"),
        ...         windows={
        ...             "input": WindowConfig(
        ...                 "end - 1 day", "trigger", True, True, index_timestamp=index_timestamp
        ...             ),
        ...         },
        ...     )
        ...     with caplog.at_level(logging.INFO):
        ...         return query(cfg, backward_df)["index_timestamp"].item()
        >>> index_timestamp_for("end")
        datetime.datetime(2010, 6, 20, 0, 0)
        >>> index_timestamp_for("start")
        datetime.datetime(2010, 6, 19, 0, 0)

        >>> cfg = TaskExtractorConfig(
        ...     predicates={"A": PlainPredicateConfig("A", static=True)},
        ...     trigger=EventConfig("_ANY_EVENT"),
        ...     windows={},
        ... )
        >>> with caplog.at_level(logging.INFO):
        ...     query(cfg, predicates_df)
        shape: (0, 0)
        ┌┐
        ╞╡
        └┘
        >>> "Static variable criteria specified, filtering patient demographics..." in caplog.text
        True
        >>> "No static variable criteria specified, removing all rows with null timestamps..." in caplog.text
        True
        >>> predicates_df = pl.DataFrame({
        ...     "subject_id": [1, 1, 3],
        ...     "timestamp": [None, datetime(2010, 6, 20), datetime(2010, 5, 11)],
        ...     "A": [True, False, False],
        ...     "_ANY_EVENT": [False, False, False],
        ... })
        >>> with caplog.at_level(logging.INFO):
        ...     result = query(cfg, predicates_df)
        >>> "No valid rows found for the trigger event" in caplog.text
        True
    """
    if not isinstance(predicates_df, pl.DataFrame):
        raise TypeError(f"Predicates dataframe type must be a polars.DataFrame. Got: {type(predicates_df)}.")

    logger.info("Checking if '(subject_id, timestamp)' columns are unique...")

    is_unique = predicates_df.n_unique(subset=["subject_id", "timestamp"]) == predicates_df.shape[0]

    if not is_unique:
        raise ValueError("The (subject_id, timestamp) columns must be unique.")

    log_tree(cfg.window_tree)

    logger.info("Beginning query...")

    static_variables = [pred for pred in cfg.predicates if cfg.predicates[pred].static]
    if static_variables:
        logger.info("Static variable criteria specified, filtering patient demographics...")
        predicates_df = check_static_variables(static_variables, predicates_df)
    else:
        logger.info("No static variable criteria specified, removing all rows with null timestamps...")
        predicates_df = predicates_df.drop_nulls(subset=["subject_id", "timestamp"])

    if predicates_df.is_empty():
        logger.warning("No valid rows found after filtering patient demographics. Exiting.")
        return pl.DataFrame()

    trigger_column = cfg.trigger.predicate
    if cfg.trigger.every:
        logger.info(f"Adding a trigger time every {cfg.trigger.every} from each '{trigger_column}' event...")
        predicates_df = add_trigger_grid(cfg.trigger, predicates_df)
        trigger_column = TRIGGER_GRID_COLUMN

    logger.info("Identifying possible trigger nodes based on the specified trigger event...")
    prospective_root_anchors = check_constraints({trigger_column: (1, None)}, predicates_df).select(
        "subject_id", pl.col("timestamp").alias("subtree_anchor_timestamp")
    )

    if prospective_root_anchors.is_empty():
        logger.warning(f"No valid rows found for the trigger event '{cfg.trigger.predicate}'. Exiting.")
        return pl.DataFrame()

    result = extract_subtree(cfg.window_tree, prospective_root_anchors, predicates_df)
    if result.is_empty():  # pragma: no cover
        logger.warning("No valid rows found.")
        return pl.DataFrame()
    else:
        # number of patients
        logger.info(
            f"Done. {result.shape[0]:,} valid rows returned corresponding to "
            f"{result['subject_id'].n_unique():,} subjects."
        )

    result = result.rename({"subtree_anchor_timestamp": "trigger"})

    to_return_cols = [
        "subject_id",
        "trigger",
        *[f"{node.node_name}_summary" for node in preorder_iter(cfg.window_tree)][1:],
    ]

    # add label column if specified
    if cfg.label_window:
        logger.info(  # pragma: no cover
            f"Extracting label '{cfg.windows[cfg.label_window].label}' from window '{cfg.label_window}'..."
        )
        label_col = "end" if cfg.windows[cfg.label_window].root_node == "start" else "start"
        result = result.with_columns(
            pl.col(f"{cfg.label_window}.{label_col}_summary")
            .struct.field(cfg.windows[cfg.label_window].label)
            .alias("label")
        )
        to_return_cols.insert(1, "label")

        if result["label"].n_unique() == 1:  # pragma: no cover
            logger.warning(
                f"All labels in the extracted cohort are the same: '{result['label'][0]}'. "
                "This may indicate an issue with the task logic. "
                "Please double-check your configuration file if this is not expected."
            )

    # add index_timestamp column if specified
    if cfg.index_timestamp_window:
        logger.info(  # pragma: no cover
            f"Setting index timestamp as '{cfg.windows[cfg.index_timestamp_window].index_timestamp}' "
            f"of window '{cfg.index_timestamp_window}'..."
        )
        index_timestamp_col = (
            "end" if cfg.windows[cfg.index_timestamp_window].root_node == "start" else "start"
        )
        result = result.with_columns(
            pl.col(f"{cfg.index_timestamp_window}.{index_timestamp_col}_summary")
            .struct.field(f"timestamp_at_{cfg.windows[cfg.index_timestamp_window].index_timestamp}")
            .alias("index_timestamp")
        )
        to_return_cols.insert(1, "index_timestamp")

    return result.select(to_return_cols)
