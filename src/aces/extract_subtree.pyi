from datetime import timedelta

import polars as pl
from bigtree import Node

def extract_subtree(
    subtree: Node,
    subtree_anchor_realizations: pl.DataFrame,
    predicates_df: pl.DataFrame,
    subtree_root_offset: timedelta = timedelta(0),
) -> pl.DataFrame: ...
