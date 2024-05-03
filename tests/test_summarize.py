import unittest
from datetime import timedelta
from unittest.mock import patch
import polars as pl
from bigtree import Node
from polars.testing import assert_frame_equal
from esgpt_task_querying.summarize import (
    summarize_temporal_window,
    summarize_event_bound_window,
    summarize_window,
    summarize_subtree,
    check_constraints,
)

import rootutils
root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)


class TestQueryFunctions(unittest.TestCase):
    def setUp(self):

        self.addTypeEqualityFunc(
            pl.DataFrame, lambda a, b, msg: assert_frame_equal(a, b, check_column_order=False)
        )

    def test_summarize_temporal_window(self):
        predicates_df = pl.DataFrame(
            {
                "subject_id": [1, 1, 1, 1, 1, 1, 1, 1],
                "timestamp": ["12/1/1989 12:03", "12/1/1989 13:14", "12/1/1989 15:17", "12/1/1989 16:17", "12/2/1989 3:00", "1/27/1991 23:32", "1/27/1991 23:46", "1/28/1991 3:18"],
                "is_A": [1, 0, 0, 1, 1, 0, 0, 1],
                "is_B": [0, 1, 0, 1, 0, 1, 0, 1],
                "is_C": [0, 0, 1, 0, 0, 0, 1, 1],
            }
        ).with_columns(pl.col("timestamp").str.strptime(pl.Datetime, format="%m/%d/%Y %H:%M").cast(pl.Datetime))

        anchor_to_subtree_root_by_subtree_anchor = pl.DataFrame(
            {
                "subject_id": [1, 1, 1, 1, 1, 1, 1, 1],
                "timestamp": ["12/1/1989 12:03", "12/1/1989 13:14", "12/1/1989 15:17", "12/1/1989 16:17", "12/2/1989 3:00", "1/27/1991 23:32", "1/27/1991 23:46", "1/28/1991 3:18"],
                "timestamp_at_anchor": ["12/1/1989 12:03", "12/1/1989 13:14", "12/1/1989 15:17", "12/1/1989 16:17", "12/2/1989 3:00", "1/27/1991 23:32", "1/27/1991 23:46", "1/28/1991 3:18"],
                "is_A": [0, 0, 0, 0, 0, 0, 0, 0],
                "is_B": [0, 0, 0, 0, 0, 0, 0, 0],
                "is_C": [0, 0, 0, 0, 0, 0, 0, 0],
            }
        ).with_columns(
            pl.col("timestamp").str.strptime(pl.Datetime, format="%m/%d/%Y %H:%M").cast(pl.Datetime),
            pl.col("timestamp_at_anchor").str.strptime(pl.Datetime, format="%m/%d/%Y %H:%M").cast(pl.Datetime)
        )

        predicate_cols = ["is_A", "is_B", "is_C"]

        endpoint_expr_case_both_true = (True, timedelta(days=1), True, timedelta(days=0))
        
        endpoint_expr_case_end_true = (False, timedelta(days=1), True, timedelta(days=0))
        
        endpoint_expr_case_end_true_offset = (False, timedelta(days=1), True, timedelta(hours=4))
        
        endpoint_expr_case_both_false_neg = (False, timedelta(days=-1), False, timedelta(days=0))

        empty_dataframe = pl.DataFrame(
            {"subject_id": [], "timestamp": []},
            schema={
                "subject_id": pl.UInt32,
                "timestamp": pl.Datetime,
            },
        )

        cases = [
            {
                "msg": "When passed an empty dataframe, should return an empty dataframe",
                "predicates_df": empty_dataframe,
                "predicate_cols": [],
                "endpoint_expr": endpoint_expr_case_both_true,
                "anchor_to_subtree_root_by_subtree_anchor": empty_dataframe,
                "want": empty_dataframe.with_columns(pl.col("timestamp").alias("timestamp_at_anchor")),
            },
            {
                "msg": "Testing summarize_temporal_window with both st_inclusive and end_inclusive as True, should be equal",
                "predicates_df": predicates_df,
                "predicate_cols": predicate_cols,
                "endpoint_expr": endpoint_expr_case_both_true,
                "anchor_to_subtree_root_by_subtree_anchor": anchor_to_subtree_root_by_subtree_anchor,
                "want": pl.DataFrame(
                    {
                        "subject_id": [1, 1, 1, 1, 1, 1, 1, 1],
                        "timestamp": ["12/1/1989 12:03", "12/1/1989 13:14", "12/1/1989 15:17", "12/1/1989 16:17", "12/2/1989 3:00", "1/27/1991 23:32", "1/27/1991 23:46", "1/28/1991 3:18"],
                        "is_A": [3, 2, 2, 2, 1, 1, 1, 1],
                        "is_B": [2, 2, 1, 1, 0, 2, 1, 1],
                        "is_C": [1, 1, 1, 0, 0, 2, 2, 1],
                        "timestamp_at_anchor": ["12/1/1989 12:03", "12/1/1989 13:14", "12/1/1989 15:17", "12/1/1989 16:17", "12/2/1989 3:00", "1/27/1991 23:32", "1/27/1991 23:46", "1/28/1991 3:18"],
                        "is_A_summary": [0, 0, 0, 0, 0, 0, 0, 0],
                        "is_B_summary": [0, 0, 0, 0, 0, 0, 0, 0],
                        "is_C_summary": [0, 0, 0, 0, 0, 0, 0, 0],
                    }
                ).with_columns(
                    pl.col("timestamp").str.strptime(pl.Datetime, format="%m/%d/%Y %H:%M").cast(pl.Datetime),
                    pl.col("timestamp_at_anchor").str.strptime(pl.Datetime, format="%m/%d/%Y %H:%M").cast(pl.Datetime)
                )
            },
            {
                "msg": "Testing summarize_temporal_window with st_inclusive as False and end_inclusive as True, should be equal",
                "predicates_df": predicates_df,
                "predicate_cols": predicate_cols,
                "endpoint_expr": endpoint_expr_case_end_true,
                "anchor_to_subtree_root_by_subtree_anchor": anchor_to_subtree_root_by_subtree_anchor,
                "want": pl.DataFrame(
                    {
                        "subject_id": [1, 1, 1, 1, 1, 1, 1, 1],
                        "timestamp": ["12/1/1989 12:03", "12/1/1989 13:14", "12/1/1989 15:17", "12/1/1989 16:17", "12/2/1989 3:00", "1/27/1991 23:32", "1/27/1991 23:46", "1/28/1991 3:18"],
                        "is_A": [2, 2, 2, 1, 0, 1, 1, 0],
                        "is_B": [2, 1, 1, 0, 0, 1, 1, 0],
                        "is_C": [1, 1, 0, 0, 0, 2, 1, 0],
                        "timestamp_at_anchor": ["12/1/1989 12:03", "12/1/1989 13:14", "12/1/1989 15:17", "12/1/1989 16:17", "12/2/1989 3:00", "1/27/1991 23:32", "1/27/1991 23:46", "1/28/1991 3:18"],
                        "is_A_summary": [0, 0, 0, 0, 0, 0, 0, 0],
                        "is_B_summary": [0, 0, 0, 0, 0, 0, 0, 0],
                        "is_C_summary": [0, 0, 0, 0, 0, 0, 0, 0],
                    }
                ).with_columns(
                    pl.col("timestamp").str.strptime(pl.Datetime, format="%m/%d/%Y %H:%M").cast(pl.Datetime),
                    pl.col("timestamp_at_anchor").str.strptime(pl.Datetime, format="%m/%d/%Y %H:%M").cast(pl.Datetime)
                )
            },
            {
                "msg": "Testing summarize_temporal_window with st_inclusive as False, end_inclusive as True, and offset of 4 hours, should be equal",
                "predicates_df": predicates_df,
                "predicate_cols": predicate_cols,
                "endpoint_expr": endpoint_expr_case_end_true_offset,
                "anchor_to_subtree_root_by_subtree_anchor": anchor_to_subtree_root_by_subtree_anchor,
                "want": pl.DataFrame(
                    {
                        "subject_id": [1, 1, 1, 1, 1, 1, 1, 1],
                        "timestamp": ["12/1/1989 12:03", "12/1/1989 13:14", "12/1/1989 15:17", "12/1/1989 16:17", "12/2/1989 3:00", "1/27/1991 23:32", "1/27/1991 23:46", "1/28/1991 3:18"],
                        "is_A": [2, 1, 1, 1, 0, 0, 0, 0],
                        "is_B": [1, 0, 0, 0, 0, 0, 0, 0],
                        "is_C": [0, 0, 0, 0, 0, 0, 0, 0],
                        "timestamp_at_anchor": ["12/1/1989 12:03", "12/1/1989 13:14", "12/1/1989 15:17", "12/1/1989 16:17", "12/2/1989 3:00", "1/27/1991 23:32", "1/27/1991 23:46", "1/28/1991 3:18"],
                        "is_A_summary": [0, 0, 0, 0, 0, 0, 0, 0],
                        "is_B_summary": [0, 0, 0, 0, 0, 0, 0, 0],
                        "is_C_summary": [0, 0, 0, 0, 0, 0, 0, 0],
                    }
                ).with_columns(
                    pl.col("timestamp").str.strptime(pl.Datetime, format="%m/%d/%Y %H:%M").cast(pl.Datetime),
                    pl.col("timestamp_at_anchor").str.strptime(pl.Datetime, format="%m/%d/%Y %H:%M").cast(pl.Datetime)
                )
            },
            {
                "msg": "Testing summarize_temporal_window with both st_inclusive and end_inclusive as False, and negative duration of 1 day, should be equal",
                "predicates_df": predicates_df,
                "predicate_cols": predicate_cols,
                "endpoint_expr": endpoint_expr_case_both_false_neg,
                "anchor_to_subtree_root_by_subtree_anchor": anchor_to_subtree_root_by_subtree_anchor,
                "want": pl.DataFrame(
                    {
                        "subject_id": [1, 1, 1, 1, 1, 1, 1, 1],
                        "timestamp": ["12/1/1989 12:03", "12/1/1989 13:14", "12/1/1989 15:17", "12/1/1989 16:17", "12/2/1989 3:00", "1/27/1991 23:32", "1/27/1991 23:46", "1/28/1991 3:18"],
                        "is_A": [0, 1, 1, 1, 2, 0, 0, 0],
                        "is_B": [0, 0, 1, 1, 2, 0, 1, 1],
                        "is_C": [0, 0, 0, 1, 1, 0, 0, 1],
                        "timestamp_at_anchor": ["12/1/1989 12:03", "12/1/1989 13:14", "12/1/1989 15:17", "12/1/1989 16:17", "12/2/1989 3:00", "1/27/1991 23:32", "1/27/1991 23:46", "1/28/1991 3:18"],
                        "is_A_summary": [0, 0, 0, 0, 0, 0, 0, 0],
                        "is_B_summary": [0, 0, 0, 0, 0, 0, 0, 0],
                        "is_C_summary": [0, 0, 0, 0, 0, 0, 0, 0],
                    }
                ).with_columns(
                    pl.col("timestamp").str.strptime(pl.Datetime, format="%m/%d/%Y %H:%M").cast(pl.Datetime),
                    pl.col("timestamp_at_anchor").str.strptime(pl.Datetime, format="%m/%d/%Y %H:%M").cast(pl.Datetime)
                )
            },
        ]

        for c in cases:
            with self.subTest(msg=c.pop("msg")):
                want = c.pop("want")
                got = summarize_temporal_window(**c)
                self.assertEqual(got, want)

    def test_summarize_event_bound_window(self):
        raise NotImplementedError
        predicates_df = pl.DataFrame(
            {
                "subject_id": [1, 1, 1, 1, 1, 1, 1, 1],
                "timestamp": ["12/1/1989 12:03", "12/1/1989 13:14", "12/1/1989 15:17", "12/1/1989 16:17", "12/2/1989 3:00", "1/27/1991 23:32", "1/27/1991 23:46", "1/28/1991 3:18"],
                "is_A": [1, 0, 0, 1, 1, 0, 0, 1],
                "is_B": [0, 1, 0, 1, 0, 1, 0, 1],
                "is_C": [0, 0, 1, 0, 0, 0, 1, 1],
            }
        ).with_columns(pl.col("timestamp").str.strptime(pl.Datetime, format="%m/%d/%Y %H:%M").cast(pl.Datetime))

        anchor_to_subtree_root_by_subtree_anchor = pl.DataFrame(
            {
                "subject_id": [1, 1, 1, 1, 1, 1, 1, 1],
                "timestamp": ["12/1/1989 12:03", "12/1/1989 13:14", "12/1/1989 15:17", "12/1/1989 16:17", "12/2/1989 3:00", "1/27/1991 23:32", "1/27/1991 23:46", "1/28/1991 3:18"],
                "timestamp_at_anchor": ["12/1/1989 12:03", "12/1/1989 13:14", "12/1/1989 15:17", "12/1/1989 16:17", "12/2/1989 3:00", "1/27/1991 23:32", "1/27/1991 23:46", "1/28/1991 3:18"],
                "is_A": [0, 0, 0, 0, 0, 0, 0, 0],
                "is_B": [0, 0, 0, 0, 0, 0, 0, 0],
                "is_C": [0, 0, 0, 0, 0, 0, 0, 0],
            }
        ).with_columns(
            pl.col("timestamp").str.strptime(pl.Datetime, format="%m/%d/%Y %H:%M").cast(pl.Datetime),
            pl.col("timestamp_at_anchor").str.strptime(pl.Datetime, format="%m/%d/%Y %H:%M").cast(pl.Datetime)
        )

        predicate_cols = ["is_A", "is_B", "is_C"]

        endpoint_expr_case_both_true = (True, timedelta(days=1), True, timedelta(days=0))
        
        endpoint_expr_case_end_true = (False, timedelta(days=1), True, timedelta(days=0))
        
        endpoint_expr_case_end_true_offset = (False, timedelta(days=1), True, timedelta(hours=4))
        
        endpoint_expr_case_both_false_neg = (False, timedelta(days=-1), False, timedelta(days=0))

        empty_dataframe = pl.DataFrame(
            {"subject_id": [], "timestamp": []},
            schema={
                "subject_id": pl.UInt32,
                "timestamp": pl.Datetime,
            },
        )

        cases = [
            {
                "msg": "When passed an empty dataframe, should return an empty dataframe",
                "predicates_df": empty_dataframe,
                "predicate_cols": [],
                "endpoint_expr": endpoint_expr_case_both_true,
                "anchor_to_subtree_root_by_subtree_anchor": empty_dataframe,
                "want": empty_dataframe.with_columns(pl.col("timestamp").alias("timestamp_at_anchor")),
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

    def test_summarize_subtree(self):
        raise NotImplementedError

    def test_check_constraints(self):
        predicates_df = pl.DataFrame(
            {
                "subject_id": [1, 1, 1, 1, 1, 1, 1, 1],
                "timestamp": ["12/1/1989 12:03", "12/1/1989 13:14", "12/1/1989 15:17", "12/1/1989 16:17", "12/2/1989 3:00", "1/27/1991 23:32", "1/27/1991 23:46", "1/28/1991 3:18"],
                "is_A": [1, 0, 0, 1, 1, 0, 0, 1],
                "is_B": [0, 1, 0, 1, 0, 1, 0, 1],
                "is_C": [0, 0, 1, 0, 0, 0, 1, 1],
            }
        ).with_columns(pl.col("timestamp").str.strptime(pl.Datetime, format="%m/%d/%Y %H:%M").cast(pl.Datetime))

        empty_constraints = {}

        constraints_case_max_zero = {
            "is_A": (None, 0),
            "is_B": (None, 0),
            "is_C": (None, 0),
        }
        
        constraints_case_min_zero = {
            "is_A": (0, None),
            "is_B": (0, None),
            "is_C": (0, None),
        }

        constraints_case_exactly_zero = {
            "is_A": (0, 0),
            "is_B": (0, 0),
            "is_C": (0, 0),
        }

        constraints_case_exactly_one = {
            "is_A": (1, 1),
            "is_B": (1, 1),
            "is_C": (1, 1),
        }

        constraints_case_between_zero_one = {
            "is_A": (0, 1),
            "is_B": (0, 1),
            "is_C": (0, 1),
        }

        constraints_case_between_one_zero = {
            "is_A": (1, 0),
            "is_B": (1, 0),
            "is_C": (1, 0),
        }

        cases = [
            {
                "msg": "Testing check_constraints with empty constraints, should keep all rows",
                "window_constraints": empty_constraints,
                "summary_df": predicates_df,
                "want": predicates_df.filter(pl.all_horizontal([pl.lit(True)])),
            },
            {
                "msg": "Testing check_constraints with exclude constraints, should filter accordingly",
                "window_constraints": constraints_case_max_zero,
                "summary_df": predicates_df,
                "want": predicates_df.filter(
                    pl.all_horizontal(
                        [pl.col("is_A") <= 0, pl.col("is_B") <= 0, pl.col("is_C") <= 0]
                    )
                ),
            },
            {
                "msg": "Testing check_constraints with include constraints, should filter accordingly",
                "window_constraints": constraints_case_min_zero,
                "summary_df": predicates_df,
                "want": predicates_df.filter(
                    pl.all_horizontal(
                        [pl.col("is_A") >= 0, pl.col("is_B") >= 0, pl.col("is_C") >= 0]
                    )
                ),
            },
            {
                "msg": "Testing check_constraints with exactly constraints, should filter accordingly",
                "window_constraints": constraints_case_exactly_zero,
                "summary_df": predicates_df,
                "want": predicates_df.filter(
                    pl.all_horizontal(
                        [pl.col("is_A") == 0, pl.col("is_B") == 0, pl.col("is_C") == 0]
                    )
                ),
            },
            {
                "msg": "Testing check_constraints with exactly constraints, should filter accordingly",
                "window_constraints": constraints_case_exactly_one,
                "summary_df": predicates_df,
                "want": predicates_df.filter(
                    pl.all_horizontal(
                        [pl.col("is_A") == 1, pl.col("is_B") == 1, pl.col("is_C") == 1]
                    )
                ),
            },
            {
                "msg": "Testing check_constraints with between constraints, should filter accordingly",
                "window_constraints": constraints_case_between_zero_one,
                "summary_df": predicates_df,
                "want": predicates_df.filter(
                    pl.all_horizontal(
                        [pl.col("is_A") >= 0, pl.col("is_A") <= 1, pl.col("is_B") >= 0, pl.col("is_B") <= 1, pl.col("is_C") >= 0, pl.col("is_C") <= 1]
                    )
                ),
            },
            {
                "msg": "Testing check_constraints with between constraints, should filter accordingly",
                "window_constraints": constraints_case_between_one_zero,
                "summary_df": predicates_df,
                "want": predicates_df.filter(
                    pl.all_horizontal(
                        [pl.col("is_A") <= 0, pl.col("is_A") >= 1, pl.col("is_B") <= 0, pl.col("is_B") >= 1, pl.col("is_C") <= 0, pl.col("is_C") >= 1]
                    )
                ),
            },
        ]

        for c in cases:
            with self.subTest(msg=c.pop("msg")):
                want = c.pop("want")
                got = check_constraints(**c)
                assert str(got) == str(want)
