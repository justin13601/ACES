"""This module contains the main function for querying a task.

It accepts the configuration file and predicate columns, builds the tree, and recursively queries the tree.
"""


import polars as pl
from bigtree import preorder_iter
from loguru import logger

from .config import TaskExtractorConfig
from .constraints import check_constraints
from .extract_subtree import extract_subtree
from .utils import log_tree


def query(cfg: TaskExtractorConfig, predicates_df: pl.DataFrame) -> pl.DataFrame:
    """Query a task using the provided configuration file and predicates dataframe.

    Args:
        cfg: TaskExtractorConfig object of the configuration file.
        predicates_df: Polars predicates dataframe.

    Returns:
        polars.DataFrame: The result of the task query, containing subjects who satisfy the conditions
        defined in cfg. Timestamps for the start/end boundaries of each window specified in the task
        configuration, as well as predicate counts for each window, are provided.
    """
    if not isinstance(predicates_df, pl.DataFrame):
        raise TypeError(f"Predicates dataframe type must be a polars.DataFrame. Got: {type(predicates_df)}.")

    logger.info("Checking if '(subject_id, timestamp)' columns are unique...")
    try:
        assert (
            predicates_df.n_unique(subset=["subject_id", "timestamp"]) == predicates_df.shape[0]
        ), "The (subject_id, timestamp) columns must be unique."
    except AssertionError as e:
        logger.error(str(e))
        return pl.DataFrame()

    log_tree(cfg.window_tree)

    logger.info("Beginning query...")
    logger.info("Identifying possible trigger nodes based on the specified trigger event...")
    prospective_root_anchors = check_constraints({cfg.trigger.predicate: (1, None)}, predicates_df).select(
        "subject_id", pl.col("timestamp").alias("subtree_anchor_timestamp")
    )
    try:
        assert (
            not prospective_root_anchors.is_empty()
        ), f"No valid rows found for the trigger event '{cfg.trigger.predicate}'. Exiting."
    except AssertionError as e:
        logger.error(str(e))
        return pl.DataFrame()

    result = extract_subtree(cfg.window_tree, prospective_root_anchors, predicates_df)
    if result.is_empty():
        logger.info("No valid rows found.")
    else:
        logger.info(f"Done. {result.shape[0]:,} valid rows returned.")

    result = result.rename({"subtree_anchor_timestamp": "trigger"})

    to_return_cols = [
        "subject_id",
        "trigger",
        *[f"{node.node_name}_summary" for node in preorder_iter(cfg.window_tree)][1:],
    ]

    # add label column if specified
    if cfg.label_window:
        logger.info(
            f"Extracting label '{cfg.windows[cfg.label_window].label}' from window "
            f"'{cfg.label_window}'..."
        )
        label_col = "end" if cfg.windows[cfg.label_window].root_node == "start" else "start"
        result = result.with_columns(
            pl.col(f"{cfg.label_window}.{label_col}_summary")
            .struct.field(cfg.windows[cfg.label_window].label)
            .alias("label")
        )
        to_return_cols.insert(1, "label")

    # add index_timestamp column if specified
    if cfg.index_timestamp_window:
        logger.info(
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
