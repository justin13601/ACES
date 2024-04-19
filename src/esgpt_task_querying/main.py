"""This module contains the main function for querying a task.

It generates the predicate columns, builds the tree, and recursively queries the tree.
"""

from datetime import timedelta
from pathlib import Path

import polars as pl
from bigtree import preorder_iter, print_tree
from EventStream.data.dataset_polars import Dataset
from loguru import logger

from .config import build_tree_from_config, get_max_duration, load_config
from .event_predicates import generate_predicate_columns
from .query import query_subtree


def query_task(cfg_path: str, data: str | pl.DataFrame) -> pl.DataFrame:
    """Query a task using the provided configuration file and event stream data.

    Args:
        cfg_path: The path to the configuration file.
        ESD: The event stream data.

    Returns:
        polars.DataFrame: The result of the task query.
    """
    # load configuration
    logger.debug("Loading config...")
    cfg = load_config(cfg_path)

    # load data if path is provided and compute predicate columns, else compute predicate columns on provided data
    match data:
        case str():
            logger.debug("Data path provided, loading using ESGPT...")
            DATA_DIR = Path(data)
            try:
                ESD = Dataset.load(DATA_DIR)
            except Exception as e:
                raise ValueError(
                    "Error loading data using ESGPT! Please ensure the path provided is a valid for ESGPT."
                ) from e

            events_df = ESD.events_df
            dynamic_measurements_df = ESD.dynamic_measurements_df

            logger.debug("Generating predicate columns...")
            try:
                ESD_data = generate_predicate_columns(cfg, [events_df, dynamic_measurements_df])
            except Exception as e:
                raise ValueError(
                    "Error generating predicate columns from configuration file! Check to make sure the format of the configuration file is valid."
                ) from e
        case pl.DataFrame():            
            # check if data is in correct format
            if data.shape[0] == 0:
                raise ValueError("Provided dataset is empty!")
            if "subject_id" not in data.columns:
                raise ValueError("Provided dataset does not have subject_id column!")
            if "timestamp" not in data.columns:
                raise ValueError("Provided dataset does not have timestamp column!")

            # check if timestamp is in the correct format
            if data["timestamp"].dtype != pl.Datetime:
                data = data.with_columns(
                    pl.col("timestamp").str.strptime(pl.Datetime, format="%m/%d/%Y %H:%M").cast(pl.Datetime)
                )
            
            logger.debug("Generating predicate columns...")
            try:
                ESD_data = generate_predicate_columns(cfg, data)
            except Exception as e:
                raise ValueError(
                    "Error generating predicate columns from configuration file! Check to make sure the format of the configuration file is valid."
                ) from e

    # checking for "Beginning of record" in the configuration file
    # TODO(mmd): This doesn't look right to me.
    starts = [window['start'] for window in cfg['windows'].values()]
    if None in starts:
        max_duration = -get_max_duration(ESD_data)
        for each_window, window_info in cfg["windows"].items():
            if window_info["start"] is None:
                logger.debug(f"Setting start of the {each_window} window to the beginning of the record.")
                window_info["start"] = None
                window_info["duration"] = max_duration

    logger.debug("Building tree...")
    tree = build_tree_from_config(cfg)
    print_tree(tree, style="const_bold")

    logger.debug("Beginning query...")
    predicate_cols = [col for col in ESD_data.columns if col.startswith("is_")]

    trigger = cfg["windows"]["trigger"]
    # filter out subjects that do not have the trigger event if specified in inclusion criteria
    if trigger["includes"]:
        valid_trigger_exprs = [(ESD_data[f"is_{x['predicate']}"] == 1) for x in trigger["includes"]]
    # filter out subjects that do not have the trigger event if specified as the start
    else:
        valid_trigger_exprs = [(ESD_data[f"is_{trigger['start']}"] == 1)]
    anchor_to_subtree_root_by_subtree_anchor = ESD_data.clone()
    anchor_to_subtree_root_by_subtree_anchor_shape = anchor_to_subtree_root_by_subtree_anchor.shape[0]
    # log filtered subjects
    for i, condition in enumerate(valid_trigger_exprs):
        dropped = anchor_to_subtree_root_by_subtree_anchor.filter(~condition)
        anchor_to_subtree_root_by_subtree_anchor = anchor_to_subtree_root_by_subtree_anchor.filter(condition)
        if anchor_to_subtree_root_by_subtree_anchor.shape[0] < anchor_to_subtree_root_by_subtree_anchor_shape:
            if trigger["includes"]:
                logger.debug(
                    f"{dropped['subject_id'].unique().shape[0]} subjects ({dropped.shape[0]} rows) were excluded due to trigger condition: {cfg['windows']['trigger']['includes'][i]}."
                )
            else:
                logger.debug(
                    f"{dropped['subject_id'].unique().shape[0]} subjects ({dropped.shape[0]} rows) were excluded due to trigger event: {cfg['windows']['trigger']['start']}."
                )
            anchor_to_subtree_root_by_subtree_anchor_shape = anchor_to_subtree_root_by_subtree_anchor.shape[0]

    # prepare anchor_to_subtree_root_by_subtree_anchor for query_subtree
    anchor_to_subtree_root_by_subtree_anchor = anchor_to_subtree_root_by_subtree_anchor.select(
        "subject_id", "timestamp", *[pl.col(c) for c in predicate_cols]
    ).with_columns("subject_id", "timestamp", *[pl.lit(0).alias(c) for c in predicate_cols])

    # recursively query the tree
    result = query_subtree(
        subtree=tree,
        anchor_to_subtree_root_by_subtree_anchor=anchor_to_subtree_root_by_subtree_anchor,
        predicates_df=ESD_data,
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

    # TODO: need to check if struct in window_summary has any null values and set them to 0

    # replace timestamps for windows that start at the beginning of the record
    if None in starts:
        record_starts = ESD_data.groupby("subject_id").agg(
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
                    for each_window, window_info in cfg["windows"].items()
                    if window_info["start"] is None
                ]
            )
            .drop(["start_of_record"])
        )

    # add label column if specified
    label_window = None
    for window, window_info in cfg["windows"].items():
        if window_info.get("label", None):
            label_window = window
            break
    if label_window:
        label = cfg["windows"][label_window]["label"]
        result = result.with_columns(
            pl.col(f"{label_window}/window_summary").struct.field(f"is_{label}").alias("label")
        )
    return result
