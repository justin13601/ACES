"""This module contains the main function for querying a task.

It accepts the configuration file and predicate columns, builds the tree, and recursively queries the tree.
"""


import polars as pl
from bigtree import preorder_iter, print_tree
from loguru import logger

from .config import build_tree_from_config, get_config
from .constraints import check_constraints
from .extract_subtree import extract_subtree


def query(cfg: dict, predicates_df: pl.DataFrame) -> pl.DataFrame:
    """Query a task using the provided configuration file and predicates dataframe.

    Args:
        cfg: dictionary representation of the configuration file.
        df_predicates: predicates dataframe.

    Returns:
        polars.DataFrame: The result of the task query.
    """
    if not isinstance(predicates_df, pl.DataFrame):
        raise TypeError(f"Predicates dataframe type must be a polars.DataFrame. Got {type(predicates_df)}.")

    logger.debug("Building tree...")
    tree = build_tree_from_config(cfg)
    print_tree(tree, style="const_bold")

    logger.debug("Beginning query...")
    prospective_root_anchors = check_constraints(tree.constraints, predicates_df).select(
        "subject_id", pl.col("timestamp").alias("subtree_anchor_timestamp")
    )

    result = extract_subtree(tree, prospective_root_anchors, predicates_df)
    logger.debug("Done.")

    # reorder columns in output dataframe
    output_order = [node for node in preorder_iter(tree)]
    result = result.select("subject_id", "timestamp", *(f"{c.name}_summary" for c in output_order[1:]))

    # add label column if specified
    label_window = None
    for window, window_info in cfg["windows"].items():
        if get_config(window_info, "label", None):
            label_window = window
            break
    if label_window:
        label = cfg["windows"][label_window]["label"]
        result = result.with_columns(
            pl.col(f"{label_window}_summary").struct.field(f"is_{label}").alias("label")
        )
    return result
