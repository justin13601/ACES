"""Tests that expose the extract_subtree offset mutation bug.

Bug: extract_subtree mutates child.endpoint_expr.offset in-place (line 292 of
extract_subtree.py). When called with a non-zero subtree_root_offset, the offset
accumulates on every call, producing silently wrong results on subsequent invocations.
"""

from datetime import datetime, timedelta

import polars as pl
import pytest
from bigtree import Node

from aces.config import (
    EventConfig,
    PlainPredicateConfig,
    TaskExtractorConfig,
    WindowConfig,
)
from aces.extract_subtree import extract_subtree
from aces.query import query
from aces.types import TemporalWindowBounds


class TestExtractSubtreeIdempotency:
    """Calling extract_subtree twice on the same tree must produce identical results."""

    @pytest.fixture()
    def simple_tree(self):
        """A minimal tree: trigger -> gap (2-day temporal window)."""
        root = Node("trigger")
        child = Node("gap")
        child.endpoint_expr = TemporalWindowBounds(True, timedelta(days=2), True)
        child.constraints = {}
        child.parent = root
        return root

    @pytest.fixture()
    def predicates_df(self):
        return pl.DataFrame(
            {
                "subject_id": [1, 1, 1],
                "timestamp": [
                    datetime(2020, 1, 1),
                    datetime(2020, 1, 2),
                    datetime(2020, 1, 3),
                ],
                "is_A": [1, 0, 1],
            }
        )

    @pytest.fixture()
    def anchors(self):
        return pl.DataFrame(
            {
                "subject_id": [1],
                "subtree_anchor_timestamp": [datetime(2020, 1, 1)],
            }
        )

    def test_repeated_calls_produce_same_result(self, simple_tree, predicates_df, anchors):
        """Two identical calls to extract_subtree must return the same result.

        With a 1-day offset and 2-day window anchored at 2020-01-01, the window should span [2020-01-02,
        2020-01-04]. Both calls must agree on this.
        """
        offset = timedelta(days=1)

        result1 = extract_subtree(simple_tree, anchors, predicates_df, subtree_root_offset=offset)
        result2 = extract_subtree(simple_tree, anchors, predicates_df, subtree_root_offset=offset)

        assert result1.equals(result2), (
            "extract_subtree returned different results on second call.\n"
            f"Call 1:\n{result1}\nCall 2:\n{result2}"
        )


class TestQueryIdempotency:
    """Calling query() twice on the same TaskExtractorConfig must produce identical results."""

    @pytest.fixture()
    def task_config(self):
        predicates = {
            "admission": PlainPredicateConfig("admission"),
            "discharge": PlainPredicateConfig("discharge"),
        }
        trigger = EventConfig("admission")
        windows = {
            "input": WindowConfig(
                start=None,
                end="trigger + 24h",
                start_inclusive=True,
                end_inclusive=True,
                has={},
                index_timestamp="end",
            ),
            "target": WindowConfig(
                start="input.end",
                end="start + 48h",
                start_inclusive=False,
                end_inclusive=True,
                has={},
                label="discharge",
            ),
        }
        return TaskExtractorConfig(predicates=predicates, trigger=trigger, windows=windows)

    @pytest.fixture()
    def predicates_df(self):
        return pl.DataFrame(
            {
                "subject_id": [1, 1, 1, 1],
                "timestamp": [
                    datetime(2020, 1, 1),
                    datetime(2020, 1, 2),
                    datetime(2020, 1, 3),
                    datetime(2020, 1, 5),
                ],
                "admission": [1, 0, 0, 0],
                "discharge": [0, 0, 1, 1],
                "_ANY_EVENT": [1, 1, 1, 1],
            }
        )

    def test_query_idempotent(self, task_config, predicates_df):
        """Query() must be idempotent -- calling it twice yields the same result.

        Setup: admission on Jan 1, discharges on Jan 3 and Jan 5.
          - input window: [record_start, trigger + 24h] = [record_start, Jan 2]
          - index_timestamp = end of input = Jan 2
          - target window: (Jan 2, Jan 2 + 48h] = (Jan 2, Jan 4]
        Both calls must produce the same index_timestamp and label.
        """
        result1 = query(task_config, predicates_df)
        result2 = query(task_config, predicates_df)

        assert result1.equals(result2), (
            f"query() returned different results on second call.\nCall 1:\n{result1}\nCall 2:\n{result2}"
        )
