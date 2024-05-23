"""This module contains the main function for querying a task.

It accepts the configuration file and predicate columns, builds the tree, and recursively queries the tree.
"""


import polars as pl
from bigtree import print_tree
from loguru import logger

from .config import TaskExtractorConfig
from .constraints import check_constraints
from .extract_subtree import extract_subtree


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

    logger.info(print_tree(cfg.window_tree, style="const_bold"))

    logger.info("Beginning query...")
    prospective_root_anchors = check_constraints(
        {cfg.trigger_event.predicate: (1, None)}, predicates_df
    ).select("subject_id", pl.col("timestamp").alias("subtree_anchor_timestamp"))

    result = extract_subtree(cfg.window_tree, prospective_root_anchors, predicates_df)
    logger.info("Done.")

    # add label column if specified
    label_window = None
    for name, window in cfg.windows.items():
        if window.label:
            label = window.label
            label_window = name
            break
    if label_window:
        result = result.with_columns(pl.col(f"{label_window}.end_summary").struct.field(label).alias("label"))
    return result
