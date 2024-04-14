"""This module contains the main function for querying a task.

It generates the predicate columns, builds the tree, and recursively queries the tree.
"""

from pathlib import Path
from datetime import timedelta

import polars as pl
from bigtree import preorder_iter, print_tree

from EventStream.data.dataset_polars import Dataset

from .config import build_tree_from_config, load_config, get_max_duration
from .event_predicates import generate_predicate_columns
from .query import query_subtree

from loguru import logger


def query_task(cfg_path: str, data: str | pl.DataFrame) -> pl.DataFrame:
    """Query a task using the provided configuration file and event stream data.

    Args:
        cfg_path: The path to the configuration file.
        ESD: The event stream data.

    Returns:
        polars.DataFrame: The result of the task query.
    """
    match data:
        case str():
            DATA_DIR = Path(data)
            ESD = Dataset.load(DATA_DIR)

            events_df = ESD.events_df.filter(~pl.all_horizontal(pl.all().is_null()))
            dynamic_measurements_df = ESD.dynamic_measurements_df.filter(
                ~pl.all_horizontal(pl.all().is_null())
            )

            ESD_data = (
                events_df.join(dynamic_measurements_df, on="event_id", how="left")
                .drop(["event_id"])
                .sort(by=["subject_id", "timestamp", "event_type"])
            )
        case pl.DataFrame():
            ESD_data = data

    if ESD_data["timestamp"].dtype != pl.Datetime:
        ESD_data = ESD_data.with_columns(
            pl.col("timestamp")
            .str.strptime(pl.Datetime, format="%m/%d/%Y %H:%M")
            .cast(pl.Datetime)
        )

    if ESD_data.shape[0] == 0:
        raise ValueError("Empty ESD!")
    if "timestamp" not in ESD_data.columns:
        raise ValueError("ESD does not have timestamp column!")
    if "subject_id" not in ESD_data.columns:
        raise ValueError("ESD does not have subject_id column!")

    logger.debug("Loading config...")
    cfg = load_config(cfg_path)

    logger.debug("Generating predicate columns...")
    try:
        ESD_data = generate_predicate_columns(cfg, ESD_data)
    except Exception as e:
        raise ValueError(
            "Error generating predicate columns from configuration file! Check to make sure the format of the configuration file is valid."
        ) from e

    starts = [window.start for window in cfg.windows.values()]
    if None in starts:
        max_duration = -get_max_duration(ESD_data)
        for each_window in cfg.windows:
            if cfg.windows[each_window].start is None:
                logger.debug(f"Setting start of the {each_window} window to the beginning of the record.")
                cfg.windows[each_window].start = None
                cfg.windows[each_window].duration = max_duration

    logger.debug("Building tree...")
    tree = build_tree_from_config(cfg)
    print_tree(tree, style="const_bold")

    logger.debug("Beginning query...")
    predicate_cols = [col for col in ESD_data.columns if col.startswith("is_")]

    if cfg.windows.trigger.includes:
        valid_trigger_exprs = [
            (ESD_data[f"is_{x['predicate']}"] == 1)
            for x in cfg.windows.trigger.includes
        ]
    else:
        valid_trigger_exprs = [(ESD_data[f"is_{cfg.windows.trigger.start}"] == 1)]
    anchor_to_subtree_root_by_subtree_anchor = ESD_data.clone()
    anchor_to_subtree_root_by_subtree_anchor_shape = (
        anchor_to_subtree_root_by_subtree_anchor.shape[0]
    )
    for i, condition in enumerate(valid_trigger_exprs):
        dropped = anchor_to_subtree_root_by_subtree_anchor.filter(~condition)
        anchor_to_subtree_root_by_subtree_anchor = (
            anchor_to_subtree_root_by_subtree_anchor.filter(condition)
        )
        if (
            anchor_to_subtree_root_by_subtree_anchor.shape[0]
            < anchor_to_subtree_root_by_subtree_anchor_shape
        ):
            if cfg.windows.trigger.includes:
                logger.debug(
                    f"{dropped['subject_id'].unique().shape[0]} subjects ({dropped.shape[0]} rows) were excluded due to trigger condition: {cfg.windows.trigger.includes[i]}."
                )
            else:
                logger.debug(
                    f"{dropped['subject_id'].unique().shape[0]} subjects ({dropped.shape[0]} rows) were excluded due to trigger event: {cfg.windows.trigger.start}."
                )
            anchor_to_subtree_root_by_subtree_anchor_shape = (
                anchor_to_subtree_root_by_subtree_anchor.shape[0]
            )

    anchor_to_subtree_root_by_subtree_anchor = (
        anchor_to_subtree_root_by_subtree_anchor.select(
            "subject_id", "timestamp", *[pl.col(c) for c in predicate_cols]
        ).with_columns(
            "subject_id", "timestamp", *[pl.lit(0).alias(c) for c in predicate_cols]
        )
    )

    result = query_subtree(
        subtree=tree,
        anchor_to_subtree_root_by_subtree_anchor=anchor_to_subtree_root_by_subtree_anchor,
        predicates_df=ESD_data,
        anchor_offset=timedelta(hours=0),
    )
    logger.debug("Done.")

    output_order = [node for node in preorder_iter(tree)]

    result = result.select(
        "subject_id",
        "timestamp",
        *[f"{c.name}/timestamp" for c in output_order[1:]],
        *[f"{c.name}/window_summary" for c in output_order[1:]],
    ).rename({"timestamp": f"{tree.name}/timestamp"})

    if None in starts:
        record_starts = ESD_data.groupby("subject_id").agg(
            [
                pl.col("timestamp").min().alias("start_of_record"),
            ]
        )
        result = result.join(
            record_starts,
            on="subject_id",
            how="left",
        ).with_columns(
            *[
                pl.col("start_of_record").alias(f"{each_window}/timestamp")
                for each_window in cfg.windows if cfg.windows[each_window].start is None
            ]
        ).drop(["start_of_record"])

    label_window = None
    for window in cfg.windows:
        if "label" in cfg.windows[window]:
            if cfg.windows[window].label:
                label_window = window
                break

    if label_window:
        label = cfg.windows[label_window].label
        result = result.with_columns(
            pl.col(f"{label_window}/window_summary")
            .struct.field(f"is_{label}")
            .alias("label")
        )
    return result
