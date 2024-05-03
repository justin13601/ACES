"""This module contains the main function for querying a task.

It generates the predicate columns, builds the tree, and recursively queries the tree.
"""

from datetime import timedelta

import polars as pl
from bigtree import preorder_iter, print_tree
from loguru import logger

from .config import build_tree_from_config, get_max_duration, get_config
from .summarize import summarize_subtree


def query(cfg: dict, df_predicates: pl.DataFrame) -> pl.DataFrame:
    """Query a task using the provided configuration file and predicates dataframe.

    Args:
        cfg: dictionary representation of the configuration file.
        df_predicates: predicates dataframe.

    Returns:
        polars.DataFrame: The result of the task query.
    """
    if type(cfg) != dict:
        raise TypeError("Config type is not dict.")
    if type(df_predicates) != pl.DataFrame:
        raise TypeError("Predicates dataframe type is not polars.DataFrame.")

    # checking for "Beginning of record" in the configuration file
    null_starts = {window: get_config(cfg['windows'][window], "start", "") for window in cfg['windows'] if get_config(cfg['windows'][window], "start", "") == "None"}
    if null_starts:
        max_duration = -get_max_duration(df_predicates)
        for each_window in null_starts:
            logger.debug(
                f"Setting start of the '{each_window}' window to the beginning of the record."
            )
            cfg['windows'][each_window].pop("start")
            cfg['windows'][each_window]["duration"] = max_duration

    logger.debug("Building tree...")
    tree = build_tree_from_config(cfg)
    print_tree(tree, style="const_bold")

    logger.debug("Beginning query...")
    predicate_cols = [col for col in df_predicates.columns if col.startswith("is_")]

    trigger = cfg["windows"]["trigger"]
    # filter out subjects that do not have the trigger event if specified in inclusion criteria
    if get_config(trigger, "includes", []):
        valid_trigger_exprs = [
            (df_predicates[f"is_{x['predicate']}"] == 1) for x in trigger["includes"]
        ]
    # filter out subjects that do not have the trigger event if specified as the start
    else:
        valid_trigger_exprs = [(df_predicates[f"is_{trigger['start']}"] == 1)]
    anchor_to_subtree_root_by_subtree_anchor = df_predicates.clone()
    anchor_to_subtree_root_by_subtree_anchor_shape = (
        anchor_to_subtree_root_by_subtree_anchor.shape[0]
    )
    # log filtered subjects
    for i, condition in enumerate(valid_trigger_exprs):
        dropped = anchor_to_subtree_root_by_subtree_anchor.filter(~condition)
        anchor_to_subtree_root_by_subtree_anchor = (
            anchor_to_subtree_root_by_subtree_anchor.filter(condition)
        )
        if (
            anchor_to_subtree_root_by_subtree_anchor.shape[0]
            < anchor_to_subtree_root_by_subtree_anchor_shape
        ):
            if get_config(trigger, "includes", []):
                logger.debug(
                    f"{dropped['subject_id'].unique().shape[0]} subjects ({dropped.shape[0]} rows) were excluded due to trigger condition: {cfg['windows']['trigger']['includes'][i]}."
                )
            else:
                logger.debug(
                    f"{dropped['subject_id'].unique().shape[0]} subjects ({dropped.shape[0]} rows) were excluded due to trigger event: {cfg['windows']['trigger']['start']}."
                )
            anchor_to_subtree_root_by_subtree_anchor_shape = (
                anchor_to_subtree_root_by_subtree_anchor.shape[0]
            )

    # prepare anchor_to_subtree_root_by_subtree_anchor for summarize_subtree
    anchor_to_subtree_root_by_subtree_anchor = (
        anchor_to_subtree_root_by_subtree_anchor.select(
            "subject_id", "timestamp", *[pl.col(c) for c in predicate_cols]
        ).with_columns(
            "subject_id", "timestamp", *[pl.lit(0).alias(c) for c in predicate_cols]
        )
    )

    # recursively summarize the windows of the task tree
    result = summarize_subtree(
        subtree=tree,
        anchor_to_subtree_root_by_subtree_anchor=anchor_to_subtree_root_by_subtree_anchor,
        predicates_df=df_predicates,
        anchor_offset=timedelta(hours=0),
    )
    logger.debug("Done.")

    # reorder columns in output dataframe
    output_order = [node for node in preorder_iter(tree)]
    result = result.select(
        "subject_id",
        "timestamp",
        *[f"{c.name}/timestamp" for c in output_order[1:]],
        *[f"{c.name}/window_summary" for c in output_order[1:]],
    ).rename({"timestamp": f"{tree.name}/timestamp"})

    # replace timestamps for windows that start at the beginning of the record
    if null_starts:
        record_starts = df_predicates.groupby("subject_id").agg(
            [
                pl.col("timestamp").min().alias("start_of_record"),
            ]
        )
        result = (
            result.join(
                record_starts,
                on="subject_id",
                how="left",
            )
            .with_columns(
                *[
                    pl.col("start_of_record").alias(f"{each_window}/timestamp")
                    for each_window in null_starts
                ]
            )
            .drop(["start_of_record"])
        )

    # add label column if specified
    label_window = None
    for window, window_info in cfg["windows"].items():
        if get_config(window_info, "label", None):
            label_window = window
            break
    if label_window:
        label = cfg["windows"][label_window]["label"]
        result = result.with_columns(
            pl.col(f"{label_window}/window_summary")
            .struct.field(f"is_{label}")
            .alias("label")
        )
    return result
