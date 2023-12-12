import rootutils

root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)

import unittest
from datetime import timedelta
from unittest.mock import patch

import polars as pl
from polars.testing import assert_frame_equal

from esgpt_task_querying.query import (  # query_subtree,; summarize_event_bound_window,
    summarize_temporal_window,
    summarize_window,
)


class TestQueryFunctions(unittest.TestCase):
    def setUp(self):
        self.addTypeEqualityFunc(
            pl.DataFrame, lambda a, b, msg: assert_frame_equal(a, b, check_column_order=False)
        )

    def test_summarize_temporal_window(self):
        empty_dataframe = pl.DataFrame(
            {"subject_id": [], "timestamp": []},
            schema={
                "subject_id": pl.UInt32,
                "timestamp": pl.Datetime,
            },
        )
        cases = [
            {
                "msg": "When passed an empty dataframe should return an empty dataframe",
                "predicates_df": empty_dataframe,
                "predicate_cols": [],
                "endpoint_expr": (True, timedelta(seconds=10), False, timedelta(seconds=20)),
                "anchor_to_subtree_root_by_subtree_anchor": empty_dataframe,
                "want": empty_dataframe.with_columns(pl.col("timestamp").alias("timestamp_at_anchor")),
            },
        ]

        for c in cases:
            with self.subTest(msg=c.pop("msg")):
                want = c.pop("want")
                got = summarize_temporal_window(**c)
                self.assertEqual(got, want)

    def test_summarize_event_bound_window(self):
        raise NotImplementedError

    def test_summarize_window(self):
        with patch("esgpt_task_querying.query.summarize_temporal_window") as mock_summarize_temporal_window:
            with patch(
                "esgpt_task_querying.query.summarize_event_bound_window"
            ) as mock_summarize_event_bound_window:
                summarize_window(
                    child=None,  # TODO(justin): Fix
                    anchor_to_subtree_root_by_subtree_anchor=pl.DataFrame(
                        {"subject_id": [], "timestamp": []}
                    ),
                    predicates_df=pl.DataFrame({"subject_id": [], "timestamp": []}),
                    predicate_cols=[],
                )
                mock_summarize_temporal_window.assert_called_once()
                mock_summarize_event_bound_window.assert_not_called()

    def test_query_subtree(self):
        raise NotImplementedError
