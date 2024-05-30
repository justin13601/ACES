"""This module contains the main function for querying a task.

It accepts the configuration file and predicate columns, builds the tree, and recursively queries the tree.
"""


import polars as pl
from loguru import logger

from .config import TaskExtractorConfig
from .constraints import check_constraints
from .extract_subtree import extract_subtree
from .utils import log_tree


def query(cfg: TaskExtractorConfig, predicates_df: pl.DataFrame) -> pl.DataFrame:
    """Query a task using the provided configuration file and predicates dataframe.

    Args:
        cfg: dictionary representation of the configuration file.
        df_predicates: predicates dataframe.

    Returns:
        polars.DataFrame: The result of the task query.
    """
    if not isinstance(predicates_df, pl.DataFrame):
        raise TypeError(f"Predicates dataframe type must be a polars.DataFrame. Got {type(predicates_df)}.")

    logger.info("Checking if (subject_id, timestamp) columns are unique...")
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

    # add label column if specified
    if cfg.label_window:
        label_col = "end" if cfg.windows[cfg.label_window].root_node == "start" else "start"
        result = result.with_columns(
            pl.col(f"{cfg.label_window}.{label_col}_summary")
            .struct.field(cfg.windows[cfg.label_window].label)
            .alias("label")
        )

    # add index_timestamp column if specified
    if cfg.index_timestamp_window:
        index_timestamp_col = (
            "end" if cfg.windows[cfg.index_timestamp_window].root_node == "start" else "start"
        )
        result = result.with_columns(
            pl.col(f"{cfg.index_timestamp_window}.{index_timestamp_col}_summary")
            .struct.field("timestamp_at_end")
            .alias("index_timestamp")
        )

    return result.select(
        "subject_id",
        "index_timestamp",
        "label",
        "trigger",
        *[col for col in result.columns if col not in ["subject_id", "index_timestamp", "label", "trigger"]],
    )
