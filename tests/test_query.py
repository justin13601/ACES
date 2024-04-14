import rootutils

root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)

import unittest
from datetime import timedelta
from unittest.mock import patch

import polars as pl
from bigtree import Node
from polars.testing import assert_frame_equal

from esgpt_task_querying.query import (
    check_constraints,
    query_subtree,
    summarize_event_bound_window,
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

        test_simple_predicates_df = pl.DataFrame(
            {
                "subject_id": [1, 1, 1],
                "timestamp": ["12/1/1900 12:00", "12/1/1900 13:00", "12/1/1900 14:00"],
                "event_type": ["ADMISSION", "LAB", "DISCHARGE"],
                "dx": ["", "", ""],
                "lab_test": ["", "SpO2", ""],
                "lab_value": ["", "99", ""],
                "is_admission": [1, 0, 0],
                "is_lab": [0, 1, 0],
                "is_discharge": [0, 0, 1],
                "is_any": [1, 1, 1],
            }
        ).with_columns(
            pl.col("timestamp").str.strptime(pl.Datetime, format="%m/%d/%Y %H:%M").cast(pl.Datetime)
        )
        test_1_endpoint_expr = (False, timedelta(hours=2), True, None)
        test_2_endpoint_expr = (False, timedelta(hours=-2), True, None)
        test_1_anchor_to_subtree_root_by_subtree_anchor = pl.DataFrame(
            {
                "subject_id": [1],
                "timestamp": ["12/1/1900 12:00"],
                "is_admission": [1],
                "is_lab": [0],
                "is_discharge": [0],
                "is_any": [1],
            }
        ).with_columns(
            pl.col("timestamp").str.strptime(pl.Datetime, format="%m/%d/%Y %H:%M").cast(pl.Datetime)
        )
        test_2_anchor_to_subtree_root_by_subtree_anchor = pl.DataFrame(
            {
                "subject_id": [1],
                "timestamp": ["12/1/1900 14:00"],
                "is_admission": [1],
                "is_lab": [0],
                "is_discharge": [0],
                "is_any": [1],
            }
        ).with_columns(
            pl.col("timestamp").str.strptime(pl.Datetime, format="%m/%d/%Y %H:%M").cast(pl.Datetime)
        )
        test_1_result = pl.DataFrame(
            {
                "subject_id": [1],
                "timestamp": ["12/1/1900 12:00"],
                "timestamp_at_anchor": ["12/1/1900 12:00"],
                "is_admission": [0],
                "is_lab": [1],
                "is_discharge": [1],
                "is_any": [2],
                "is_admission_summary": [1],
                "is_lab_summary": [0],
                "is_discharge_summary": [0],
                "is_any_summary": [1],
            }
        ).with_columns(
            pl.col("timestamp").str.strptime(pl.Datetime, format="%m/%d/%Y %H:%M").cast(pl.Datetime),
            pl.col("timestamp_at_anchor")
            .str.strptime(pl.Datetime, format="%m/%d/%Y %H:%M")
            .cast(pl.Datetime),
        )
        test_2_result = pl.DataFrame(
            {
                "subject_id": [1],
                "timestamp": ["12/1/1900 14:00"],
                "timestamp_at_anchor": ["12/1/1900 14:00"],
                "is_admission": [0],
                "is_lab": [1],
                "is_discharge": [1],
                "is_any": [2],
                "is_admission_summary": [1],
                "is_lab_summary": [0],
                "is_discharge_summary": [0],
                "is_any_summary": [1],
            }
        ).with_columns(
            pl.col("timestamp").str.strptime(pl.Datetime, format="%m/%d/%Y %H:%M").cast(pl.Datetime),
            pl.col("timestamp_at_anchor")
            .str.strptime(pl.Datetime, format="%m/%d/%Y %H:%M")
            .cast(pl.Datetime),
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
            {
                "msg": "Testing summarize_temporal_window with simple dataframe and None as offset, should be equal",
                "predicates_df": test_simple_predicates_df,
                "predicate_cols": [col for col in test_simple_predicates_df.columns if col.startswith("is_")],
                "endpoint_expr": test_1_endpoint_expr,
                "anchor_to_subtree_root_by_subtree_anchor": test_1_anchor_to_subtree_root_by_subtree_anchor,
                "want": test_1_result,
            },
            {
                "msg": "Testing summarize_temporal_window with simple dataframe and None as offset, should be equal",
                "predicates_df": test_simple_predicates_df,
                "predicate_cols": [col for col in test_simple_predicates_df.columns if col.startswith("is_")],
                "endpoint_expr": test_2_endpoint_expr,
                "anchor_to_subtree_root_by_subtree_anchor": test_2_anchor_to_subtree_root_by_subtree_anchor,
                "want": test_2_result,
            },
        ]

        for c in cases:
            with self.subTest(msg=c.pop("msg")):
                want = c.pop("want")
                got = summarize_temporal_window(**c)
                self.assertEqual(got, want)

    def test_summarize_event_bound_window(self):
        test_simple_predicates_df = pl.DataFrame(
            {
                "subject_id": [1, 1, 1],
                "timestamp": ["12/1/1900 12:00", "12/1/1900 13:00", "12/1/1900 14:00"],
                "event_type": ["ADMISSION", "LAB", "DISCHARGE"],
                "dx": ["", "", ""],
                "lab_test": ["", "SpO2", ""],
                "lab_value": ["", "99", ""],
                "is_admission": [1, 0, 0],
                "is_lab": [0, 1, 0],
                "is_discharge": [0, 0, 1],
                "is_any": [1, 1, 1],
            }
        ).with_columns(
            pl.col("timestamp").str.strptime(pl.Datetime, format="%m/%d/%Y %H:%M").cast(pl.Datetime)
        )
        test_1_endpoint_expr = (False, "is_discharge", True, None)
        test_1_anchor_to_subtree_root_by_subtree_anchor = pl.DataFrame(
            {
                "subject_id": [1],
                "timestamp": ["12/1/1900 12:00"],
                "is_admission": [0],
                "is_lab": [0],
                "is_discharge": [0],
                "is_any": [0],
            }
        ).with_columns(
            pl.col("timestamp").str.strptime(pl.Datetime, format="%m/%d/%Y %H:%M").cast(pl.Datetime)
        )
        test_1_result = pl.DataFrame(
            {
                "subject_id": [1],
                "timestamp": ["12/1/1900 14:00"],
                "timestamp_at_anchor": ["12/1/1900 12:00"],
                "is_admission": [0],
                "is_lab": [1],
                "is_discharge": [1],
                "is_any": [2],
                "is_admission_summary": [0],
                "is_lab_summary": [0],
                "is_discharge_summary": [0],
                "is_any_summary": [0],
            }
        ).with_columns(
            pl.col("timestamp").str.strptime(pl.Datetime, format="%m/%d/%Y %H:%M").cast(pl.Datetime),
            pl.col("timestamp_at_anchor")
            .str.strptime(pl.Datetime, format="%m/%d/%Y %H:%M")
            .cast(pl.Datetime),
        )

        cases = [
            {
                "msg": "Testing summarize_event_bound_window with simple dataframe and None as offset, should be equal",
                "predicates_df": test_simple_predicates_df,
                "predicate_cols": [col for col in test_simple_predicates_df.columns if col.startswith("is_")],
                "endpoint_expr": test_1_endpoint_expr,
                "anchor_to_subtree_root_by_subtree_anchor": test_1_anchor_to_subtree_root_by_subtree_anchor,
                "want": test_1_result,
            }
        ]

        for c in cases:
            with self.subTest(msg=c.pop("msg")):
                want = c.pop("want")
                got = summarize_event_bound_window(**c)
                self.assertEqual(got, want)

    def test_summarize_window(self):
        raise NotImplementedError
        print("---Temporally-Bound Case---")
        test_temporal_node = Node(
            "gap",
            endpoint_expr=(False, timedelta(hours=48), True),
            constraints={"is_admission": (None, 0), "is_death": (None, 0), "is_discharge": (None, 0)},
        )

        with patch("esgpt_task_querying.query.summarize_temporal_window") as mock_summarize_temporal_window:
            with patch(
                "esgpt_task_querying.query.summarize_event_bound_window"
            ) as mock_summarize_event_bound_window:
                mock_summarize_temporal_window.return_value = pl.DataFrame(
                    {
                        "subject_id": [1],
                        "timestamp": ["12/1/1900 12:00"],
                        "timestamp_at_anchor": ["12/1/1900 12:00"],
                        "is_admission": [0],
                        "is_lab": [1],
                        "is_discharge": [1],
                        "is_any": [2],
                        "is_admission_summary": [1],
                        "is_lab_summary": [0],
                        "is_discharge_summary": [0],
                        "is_any_summary": [1],
                    }
                ).with_columns(
                    pl.col("timestamp").str.strptime(pl.Datetime, format="%m/%d/%Y %H:%M").cast(pl.Datetime),
                    pl.col("timestamp_at_anchor")
                    .str.strptime(pl.Datetime, format="%m/%d/%Y %H:%M")
                    .cast(pl.Datetime),
                )

                test_temporal_anchor_to_subtree_root_by_subtree_anchor = pl.DataFrame(
                    {
                        "subject_id": [1],
                        "timestamp": ["12/1/1900 12:00"],
                        "is_admission": [1],
                        "is_lab": [0],
                        "is_discharge": [0],
                        "is_any": [1],
                    }
                ).with_columns(
                    pl.col("timestamp").str.strptime(pl.Datetime, format="%m/%d/%Y %H:%M").cast(pl.Datetime)
                )

                test_temporal_result = pl.DataFrame(
                    {
                        "subject_id": [1],
                        "timestamp": ["12/1/1900 14:00"],
                        "timestamp_at_anchor": ["12/1/1900 12:00"],
                        "is_admission": [0],
                        "is_lab": [1],
                        "is_discharge": [1],
                        "is_any": [2],
                    }
                ).with_columns(
                    pl.col("timestamp").str.strptime(pl.Datetime, format="%m/%d/%Y %H:%M").cast(pl.Datetime),
                    pl.col("timestamp_at_anchor")
                    .str.strptime(pl.Datetime, format="%m/%d/%Y %H:%M")
                    .cast(pl.Datetime),
                )

                test_simple_predicates_df = pl.DataFrame(
                    {
                        "subject_id": [1, 1, 1],
                        "timestamp": ["12/1/1900 12:00", "12/1/1900 13:00", "12/1/1900 14:00"],
                        "event_type": ["ADMISSION", "LAB", "DISCHARGE"],
                        "dx": ["", "", ""],
                        "lab_test": ["", "SpO2", ""],
                        "lab_value": ["", "99", ""],
                        "is_admission": [1, 0, 0],
                        "is_lab": [0, 1, 0],
                        "is_discharge": [0, 0, 1],
                        "is_any": [1, 1, 1],
                    }
                ).with_columns(
                    pl.col("timestamp").str.strptime(pl.Datetime, format="%m/%d/%Y %H:%M").cast(pl.Datetime)
                )
                test_simple_predicate_cols = (
                    [col for col in test_simple_predicates_df.columns if col.startswith("is_")],
                )

                got = summarize_window(
                    child=test_temporal_node,
                    anchor_to_subtree_root_by_subtree_anchor=test_temporal_anchor_to_subtree_root_by_subtree_anchor,
                    predicates_df=test_simple_predicates_df,
                    predicate_cols=test_simple_predicate_cols,
                )

                mock_summarize_temporal_window.assert_called_once()
                mock_summarize_event_bound_window.assert_not_called()
                want = test_temporal_result
                self.assertEqual(got, want)

        print("---Event-Bound Case---")
        test_event_bound_node = Node(
            "target",
            endpoint_expr=(True, None, True),
            constraints={"is_death_or_is_discharge": (1, None), "is_admission": (1, None)},
        )

        with patch("esgpt_task_querying.query.summarize_temporal_window") as mock_summarize_temporal_window:
            with patch(
                "esgpt_task_querying.query.summarize_event_bound_window"
            ) as mock_summarize_event_bound_window:
                mock_summarize_event_bound_window.return_value = pl.DataFrame(
                    {
                        "subject_id": [1],
                        "timestamp": ["12/1/1900 14:00"],
                        "timestamp_at_anchor": ["12/1/1900 12:00"],
                        "is_admission": [0],
                        "is_lab": [1],
                        "is_discharge": [1],
                        "is_any": [2],
                        "is_admission_summary": [1],
                        "is_lab_summary": [0],
                        "is_discharge_summary": [0],
                        "is_any_summary": [1],
                    }
                ).with_columns(
                    pl.col("timestamp").str.strptime(pl.Datetime, format="%m/%d/%Y %H:%M").cast(pl.Datetime),
                    pl.col("timestamp_at_anchor")
                    .str.strptime(pl.Datetime, format="%m/%d/%Y %H:%M")
                    .cast(pl.Datetime),
                )

                test_event_bound_anchor_to_subtree_root_by_subtree_anchor = pl.DataFrame(
                    {
                        "subject_id": [1],
                        "timestamp": ["12/1/1900 12:00"],
                        "is_admission": [1],
                        "is_lab": [0],
                        "is_discharge": [0],
                        "is_any": [1],
                    }
                ).with_columns(
                    pl.col("timestamp").str.strptime(pl.Datetime, format="%m/%d/%Y %H:%M").cast(pl.Datetime)
                )

                test_event_bound_result = pl.DataFrame(
                    {
                        "subject_id": [1],
                        "timestamp": ["12/1/1900 14:00"],
                        "timestamp_at_anchor": ["12/1/1900 12:00"],
                        "is_admission": [0],
                        "is_lab": [1],
                        "is_discharge": [1],
                        "is_any": [2],
                    }
                ).with_columns(
                    pl.col("timestamp").str.strptime(pl.Datetime, format="%m/%d/%Y %H:%M").cast(pl.Datetime),
                    pl.col("timestamp_at_anchor")
                    .str.strptime(pl.Datetime, format="%m/%d/%Y %H:%M")
                    .cast(pl.Datetime),
                )

                test_simple_predicates_df = pl.DataFrame(
                    {
                        "subject_id": [1, 1, 1],
                        "timestamp": ["12/1/1900 12:00", "12/1/1900 13:00", "12/1/1900 14:00"],
                        "event_type": ["ADMISSION", "LAB", "DISCHARGE"],
                        "dx": ["", "", ""],
                        "lab_test": ["", "SpO2", ""],
                        "lab_value": ["", "99", ""],
                        "is_admission": [1, 0, 0],
                        "is_lab": [0, 1, 0],
                        "is_discharge": [0, 0, 1],
                        "is_any": [1, 1, 1],
                    }
                ).with_columns(
                    pl.col("timestamp").str.strptime(pl.Datetime, format="%m/%d/%Y %H:%M").cast(pl.Datetime)
                )

                test_simple_predicate_cols = (
                    [col for col in test_simple_predicates_df.columns if col.startswith("is_")],
                )

                summarize_window(
                    child=test_event_bound_node,
                    anchor_to_subtree_root_by_subtree_anchor=test_event_bound_anchor_to_subtree_root_by_subtree_anchor,
                    predicates_df=test_simple_predicates_df,
                    predicate_cols=test_simple_predicate_cols,
                )

                mock_summarize_temporal_window.assert_not_called()
                mock_summarize_event_bound_window.assert_called_once()
                want = test_event_bound_result
                self.assertEqual(got, want)

    def test_query_subtree(self):
        raise NotImplementedError

    def test_check_constraints(self):
        test_summary_df = pl.DataFrame(
            {
                "subject_id": [1, 1, 1],
                "timestamp": ["12/1/1900 12:00", "12/1/1900 13:00", "12/1/1900 14:00"],
                "is_admission": [1, 0, 0],
                "is_lab": [0, 1, 0],
                "is_discharge": [0, 0, 1],
                "is_any": [1, 1, 1],
            }
        ).with_columns(
            pl.col("timestamp").str.strptime(pl.Datetime, format="%m/%d/%Y %H:%M").cast(pl.Datetime)
        )

        test_empty_constraints = {}
        test_empty_constraints_result = pl.all_horizontal([pl.lit(True)])

        test_non_empty_constraints = {
            "is_admission": (None, 0),
            "is_death": (None, 0),
            "is_discharge": (None, 0),
        }
        test_non_empty_constraints_result = pl.all_horizontal(
            [pl.col("is_admission") <= 0, pl.col("is_death") <= 0, pl.col("is_discharge") <= 0]
        )

        cases = [
            {
                "msg": "Testing check_constraints with empty constraints, should keep all rows",
                "window_constraints": test_empty_constraints,
                "summary_df": test_summary_df,
                "want": test_empty_constraints_result,
            },
            {
                "msg": "Testing check_constraints with exclude constraints, should filter accordingly",
                "window_constraints": test_non_empty_constraints,
                "summary_df": test_summary_df,
                "want": test_non_empty_constraints_result,
            },
        ]

        for c in cases:
            with self.subTest(msg=c.pop("msg")):
                want = c.pop("want")
                got = check_constraints(**c)
                assert str(got) == str(want)
