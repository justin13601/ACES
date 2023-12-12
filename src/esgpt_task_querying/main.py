"""TODO(justin): Add a module docstring."""

from datetime import timedelta

import polars as pl
from bigtree import preorder_iter, print_tree

from .config import load_config
from .event_predicates import generate_predicate_columns
from .query import build_tree_from_config, query_subtree


def query_task(cfg_path, ESD):
    if ESD["timestamp"].dtype != pl.Datetime:
        ESD = ESD.with_columns(
            pl.col("timestamp").str.strptime(pl.Datetime, format="%m/%d/%Y %H:%M").cast(pl.Datetime)
        )

    if ESD.shape[0] == 0:
        raise ValueError("Empty ESD!")
    if "timestamp" not in ESD.columns:
        raise ValueError("ESD does not have timestamp column!")
    if "subject_id" not in ESD.columns:
        raise ValueError("ESD does not have subject_id column!")

    print("Loading config...\n")
    cfg = load_config(cfg_path)

    print("Generating predicate columns...\n")
    try:
        ESD = generate_predicate_columns(cfg, ESD)
    except Exception as e:
        raise ValueError(
            "Error generating predicate columns from configuration file! Check to make sure the format of "
            "the configuration file is valid."
        ) from e

    print("\nBuilding tree...")
    tree = build_tree_from_config(cfg)
    print_tree(tree, style="const_bold")
    print("\n")

    predicate_cols = [col for col in ESD.columns if col.startswith("is_")]

    valid_trigger_exprs = [(ESD[f"is_{x['predicate']}"] == 1) for x in cfg.windows.trigger.includes]
    anchor_to_subtree_root_by_subtree_anchor = (
        ESD.filter(pl.all_horizontal(valid_trigger_exprs))
        .select("subject_id", "timestamp", *[pl.col(c) for c in predicate_cols])
        .with_columns("subject_id", "timestamp", *[pl.lit(0).alias(c) for c in predicate_cols])
    )

    print("Querying...")
    result = query_subtree(
        subtree=tree,
        anchor_to_subtree_root_by_subtree_anchor=anchor_to_subtree_root_by_subtree_anchor,
        predicates_df=ESD,
        anchor_offset=timedelta(hours=0),
    )
    print("Done.\n")

    output_order = [node for node in preorder_iter(tree)]

    result = result.select(
        "subject_id",
        "timestamp",
        *[f"{c.name}/timestamp" for c in output_order[1:]],
        *[f"{c.name}/window_summary" for c in output_order[1:]],
    ).rename({"timestamp": f"{tree.name}/timestamp"})

    label_window = None
    for window in cfg.windows:
        if "label" in cfg.windows[window]:
            label_window = window
            break

    if label_window:
        label = cfg.windows[label_window].label
        result = result.with_columns(
            pl.col(f"{label_window}/window_summary").struct.field(f"is_{label}").alias("label")
        )
    return result
