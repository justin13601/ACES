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

    log_tree(cfg.window_tree)

    logger.info("Beginning query...")
    logger.info("Identifying possible trigger nodes based on the specified trigger event...")
    prospective_root_anchors = check_constraints(
        {cfg.trigger_event.predicate: (1, None)}, predicates_df
    ).select("subject_id", pl.col("timestamp").alias("subtree_anchor_timestamp"))

    result = extract_subtree(cfg.window_tree, prospective_root_anchors, predicates_df)
    logger.info(f"Done. {result.shape[0]} rows returned.")

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
