import rootutils

root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)

import unittest
from datetime import datetime, timedelta

import polars as pl
from polars.testing import assert_frame_equal

from esgpt_task_querying.summarize import check_constraints, summarize_temporal_window


class TestQueryFunctions(unittest.TestCase):
    def setUp(self):
        self.addTypeEqualityFunc(
            pl.DataFrame,
            lambda a, b, msg: assert_frame_equal(a, b, check_column_order=False),
        )

    def test_summarize_temporal_window(self):
        """This tests the ``summarize_temporal_window`` function.

        We need this to test several cases:
          [ ] The function should not aggregate predicates across subject boundaries.
          [x] The function should aggregate the correct time windows.
          [ ] The function should respect the passes st_inclusive and end_inclusive values, or error if
              necessary.
          [ ] The function should appropriately subtract the anchor summary values.
        """
        predicates_df = pl.DataFrame(
            {
                "subject_id": [1, 1, 1, 1, 1, 1, 1, 1],
                "timestamp": [
                    datetime(year=1989, month=12, day=1, hour=12, minute=3),
                    datetime(year=1989, month=12, day=1, hour=13, minute=14),
                    datetime(year=1989, month=12, day=1, hour=15, minute=17),
                    datetime(year=1989, month=12, day=1, hour=16, minute=17),
                    datetime(year=1989, month=12, day=2, hour=3, minute=0),
                    datetime(year=1991, month=1, day=27, hour=23, minute=32),
                    datetime(year=1991, month=1, day=27, hour=23, minute=46),
                    datetime(year=1991, month=1, day=28, hour=3, minute=18),
                ],
                "is_A": [1, 0, 0, 1, 1, 0, 0, 1],
                "is_B": [0, 1, 0, 1, 0, 1, 0, 1],
                "is_C": [0, 0, 1, 0, 0, 0, 1, 1],
            }
        )

        anchor_to_subtree_root_by_subtree_anchor = pl.DataFrame(
            {
                "subject_id": [1, 1, 1, 1, 1, 1, 1, 1],
                "timestamp": [
                    datetime(year=1989, month=12, day=1, hour=12, minute=3),
                    datetime(year=1989, month=12, day=1, hour=13, minute=14),
                    datetime(year=1989, month=12, day=1, hour=15, minute=17),
                    datetime(year=1989, month=12, day=1, hour=16, minute=17),
                    datetime(year=1989, month=12, day=2, hour=3, minute=0),
                    datetime(year=1991, month=1, day=27, hour=23, minute=32),
                    datetime(year=1991, month=1, day=27, hour=23, minute=46),
                    datetime(year=1991, month=1, day=28, hour=3, minute=18),
                ],
                "timestamp_at_anchor": [
                    datetime(year=1989, month=12, day=1, hour=12, minute=3),
                    datetime(year=1989, month=12, day=1, hour=13, minute=14),
                    datetime(year=1989, month=12, day=1, hour=15, minute=17),
                    datetime(year=1989, month=12, day=1, hour=16, minute=17),
                    datetime(year=1989, month=12, day=2, hour=3, minute=0),
                    datetime(year=1991, month=1, day=27, hour=23, minute=32),
                    datetime(year=1991, month=1, day=27, hour=23, minute=46),
                    datetime(year=1991, month=1, day=28, hour=3, minute=18),
                ],
                "is_A": [0, 0, 0, 0, 0, 0, 0, 0],
                "is_B": [0, 0, 0, 0, 0, 0, 0, 0],
                "is_C": [0, 0, 0, 0, 0, 0, 0, 0],
            }
        )

        predicate_cols = ["is_A", "is_B", "is_C"]

        cases = [
            {
                "msg": (
                    "Testing summarize_temporal_window with both st_inclusive and end_inclusive as True, "
                    "should be equal"
                ),
                "predicates_df": predicates_df,
                "predicate_cols": predicate_cols,
                "endpoint_expr": (True, timedelta(days=1), True, timedelta(days=0)),
                "anchor_to_subtree_root_by_subtree_anchor": anchor_to_subtree_root_by_subtree_anchor,
                "want": pl.DataFrame(
                    {
                        "subject_id": [1, 1, 1, 1, 1, 1, 1, 1],
                        "timestamp": [
                            datetime(year=1989, month=12, day=1, hour=12, minute=3),
                            datetime(year=1989, month=12, day=1, hour=13, minute=14),
                            datetime(year=1989, month=12, day=1, hour=15, minute=17),
                            datetime(year=1989, month=12, day=1, hour=16, minute=17),
                            datetime(year=1989, month=12, day=2, hour=3, minute=0),
                            datetime(year=1991, month=1, day=27, hour=23, minute=32),
                            datetime(year=1991, month=1, day=27, hour=23, minute=46),
                            datetime(year=1991, month=1, day=28, hour=3, minute=18),
                        ],
                        "timestamp_at_anchor": [
                            datetime(year=1989, month=12, day=1, hour=12, minute=3),
                            datetime(year=1989, month=12, day=1, hour=13, minute=14),
                            datetime(year=1989, month=12, day=1, hour=15, minute=17),
                            datetime(year=1989, month=12, day=1, hour=16, minute=17),
                            datetime(year=1989, month=12, day=2, hour=3, minute=0),
                            datetime(year=1991, month=1, day=27, hour=23, minute=32),
                            datetime(year=1991, month=1, day=27, hour=23, minute=46),
                            datetime(year=1991, month=1, day=28, hour=3, minute=18),
                        ],
                        "is_A": [3, 2, 2, 2, 1, 1, 1, 1],
                        "is_B": [2, 2, 1, 1, 0, 2, 1, 1],
                        "is_C": [1, 1, 1, 0, 0, 2, 2, 1],
                    }
                ),
            },
            {
                "msg": (
                    "Testing summarize_temporal_window with st_inclusive as False and end_inclusive as True, "
                    "should be equal"
                ),
                "predicates_df": predicates_df,
                "predicate_cols": predicate_cols,
                "endpoint_expr": (False, timedelta(days=1), True, timedelta(days=0)),
                "anchor_to_subtree_root_by_subtree_anchor": anchor_to_subtree_root_by_subtree_anchor,
                "want": pl.DataFrame(
                    {
                        "subject_id": [1, 1, 1, 1, 1, 1, 1, 1],
                        "timestamp": [
                            datetime(year=1989, month=12, day=1, hour=12, minute=3),
                            datetime(year=1989, month=12, day=1, hour=13, minute=14),
                            datetime(year=1989, month=12, day=1, hour=15, minute=17),
                            datetime(year=1989, month=12, day=1, hour=16, minute=17),
                            datetime(year=1989, month=12, day=2, hour=3, minute=0),
                            datetime(year=1991, month=1, day=27, hour=23, minute=32),
                            datetime(year=1991, month=1, day=27, hour=23, minute=46),
                            datetime(year=1991, month=1, day=28, hour=3, minute=18),
                        ],
                        "timestamp_at_anchor": [
                            datetime(year=1989, month=12, day=1, hour=12, minute=3),
                            datetime(year=1989, month=12, day=1, hour=13, minute=14),
                            datetime(year=1989, month=12, day=1, hour=15, minute=17),
                            datetime(year=1989, month=12, day=1, hour=16, minute=17),
                            datetime(year=1989, month=12, day=2, hour=3, minute=0),
                            datetime(year=1991, month=1, day=27, hour=23, minute=32),
                            datetime(year=1991, month=1, day=27, hour=23, minute=46),
                            datetime(year=1991, month=1, day=28, hour=3, minute=18),
                        ],
                        "is_A": [2, 2, 2, 1, 0, 1, 1, 0],
                        "is_B": [2, 1, 1, 0, 0, 1, 1, 0],
                        "is_C": [1, 1, 0, 0, 0, 2, 1, 0],
                    }
                ),
            },
            {
                "msg": (
                    "Testing summarize_temporal_window with st_inclusive as False, end_inclusive as True, "
                    "and offset of 4 hours, should be equal"
                ),
                "predicates_df": predicates_df,
                "predicate_cols": predicate_cols,
                "endpoint_expr": (False, timedelta(days=1), True, timedelta(hours=4)),
                "anchor_to_subtree_root_by_subtree_anchor": anchor_to_subtree_root_by_subtree_anchor,
                "want": pl.DataFrame(
                    {
                        "subject_id": [1, 1, 1, 1, 1, 1, 1, 1],
                        "timestamp": [
                            datetime(year=1989, month=12, day=1, hour=12, minute=3),
                            datetime(year=1989, month=12, day=1, hour=13, minute=14),
                            datetime(year=1989, month=12, day=1, hour=15, minute=17),
                            datetime(year=1989, month=12, day=1, hour=16, minute=17),
                            datetime(year=1989, month=12, day=2, hour=3, minute=0),
                            datetime(year=1991, month=1, day=27, hour=23, minute=32),
                            datetime(year=1991, month=1, day=27, hour=23, minute=46),
                            datetime(year=1991, month=1, day=28, hour=3, minute=18),
                        ],
                        "timestamp_at_anchor": [
                            datetime(year=1989, month=12, day=1, hour=12, minute=3),
                            datetime(year=1989, month=12, day=1, hour=13, minute=14),
                            datetime(year=1989, month=12, day=1, hour=15, minute=17),
                            datetime(year=1989, month=12, day=1, hour=16, minute=17),
                            datetime(year=1989, month=12, day=2, hour=3, minute=0),
                            datetime(year=1991, month=1, day=27, hour=23, minute=32),
                            datetime(year=1991, month=1, day=27, hour=23, minute=46),
                            datetime(year=1991, month=1, day=28, hour=3, minute=18),
                        ],
                        "is_A": [2, 1, 1, 1, 0, 0, 0, 0],
                        "is_B": [1, 0, 0, 0, 0, 0, 0, 0],
                        "is_C": [0, 0, 0, 0, 0, 0, 0, 0],
                    }
                ),
            },
            {
                "msg": (
                    "Testing summarize_temporal_window with both st_inclusive and end_inclusive as False, "
                    "and negative duration of 1 day, should be equal"
                ),
                "predicates_df": predicates_df,
                "predicate_cols": predicate_cols,
                "endpoint_expr": (False, timedelta(days=-1), False, timedelta(days=0)),
                "anchor_to_subtree_root_by_subtree_anchor": anchor_to_subtree_root_by_subtree_anchor,
                "want": pl.DataFrame(
                    {
                        "subject_id": [1, 1, 1, 1, 1, 1, 1, 1],
                        "timestamp": [
                            datetime(year=1989, month=12, day=1, hour=12, minute=3),
                            datetime(year=1989, month=12, day=1, hour=13, minute=14),
                            datetime(year=1989, month=12, day=1, hour=15, minute=17),
                            datetime(year=1989, month=12, day=1, hour=16, minute=17),
                            datetime(year=1989, month=12, day=2, hour=3, minute=0),
                            datetime(year=1991, month=1, day=27, hour=23, minute=32),
                            datetime(year=1991, month=1, day=27, hour=23, minute=46),
                            datetime(year=1991, month=1, day=28, hour=3, minute=18),
                        ],
                        "timestamp_at_anchor": [
                            datetime(year=1989, month=12, day=1, hour=12, minute=3),
                            datetime(year=1989, month=12, day=1, hour=13, minute=14),
                            datetime(year=1989, month=12, day=1, hour=15, minute=17),
                            datetime(year=1989, month=12, day=1, hour=16, minute=17),
                            datetime(year=1989, month=12, day=2, hour=3, minute=0),
                            datetime(year=1991, month=1, day=27, hour=23, minute=32),
                            datetime(year=1991, month=1, day=27, hour=23, minute=46),
                            datetime(year=1991, month=1, day=28, hour=3, minute=18),
                        ],
                        "is_A": [0, 1, 1, 1, 2, 0, 0, 0],
                        "is_B": [0, 0, 1, 1, 2, 0, 1, 1],
                        "is_C": [0, 0, 0, 1, 1, 0, 0, 1],
                    }
                ),
            },
        ]

        for c in cases:
            with self.subTest(msg=c.pop("msg")):
                want = c.pop("want")
                got = summarize_temporal_window(**c)
                self.assertEqual(got, want)

    @unittest.skip("Not yet implemented")
    def test_summarize_event_bound_window(self):
        """This tests the ``summarize_event_bound_window`` function.

        We need this to test several cases:
          [ ] The function should not aggregate predicates across subject boundaries.
          [ ] The function should aggregate between the correct events.
          [ ] The function should respect the passes st_inclusive and end_inclusive values, or error if
              necessary.
          [ ] The function should appropriately subtract the anchor summary values.
        """
        raise NotImplementedError

    @unittest.skip("Not yet implemented")
    def test_summarize_subtree(self):
        raise NotImplementedError

    def test_check_constraints(self):
        predicates_df = pl.DataFrame(
            {
                "subject_id": [1, 1, 1, 1, 1, 1, 1, 1],
                "timestamp": [
                    "12/1/1989 12:03",
                    "12/1/1989 13:14",
                    "12/1/1989 15:17",
                    "12/1/1989 16:17",
                    "12/2/1989 3:00",
                    "1/27/1991 23:32",
                    "1/27/1991 23:46",
                    "1/28/1991 3:18",
                ],
                "is_A": [1, 0, 0, 1, 1, 0, 0, 1],
                "is_B": [0, 1, 0, 1, 0, 1, 0, 1],
                "is_C": [0, 0, 1, 0, 0, 0, 1, 1],
            }
        ).with_columns(
            pl.col("timestamp").str.strptime(pl.Datetime, format="%m/%d/%Y %H:%M").cast(pl.Datetime)
        )

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
                    pl.all_horizontal([pl.col("is_A") <= 0, pl.col("is_B") <= 0, pl.col("is_C") <= 0])
                ),
            },
            {
                "msg": "Testing check_constraints with include constraints, should filter accordingly",
                "window_constraints": constraints_case_min_zero,
                "summary_df": predicates_df,
                "want": predicates_df.filter(
                    pl.all_horizontal([pl.col("is_A") >= 0, pl.col("is_B") >= 0, pl.col("is_C") >= 0])
                ),
            },
            {
                "msg": "Testing check_constraints with exactly constraints, should filter accordingly",
                "window_constraints": constraints_case_exactly_zero,
                "summary_df": predicates_df,
                "want": predicates_df.filter(
                    pl.all_horizontal([pl.col("is_A") == 0, pl.col("is_B") == 0, pl.col("is_C") == 0])
                ),
            },
            {
                "msg": "Testing check_constraints with exactly constraints, should filter accordingly",
                "window_constraints": constraints_case_exactly_one,
                "summary_df": predicates_df,
                "want": predicates_df.filter(
                    pl.all_horizontal([pl.col("is_A") == 1, pl.col("is_B") == 1, pl.col("is_C") == 1])
                ),
            },
            {
                "msg": "Testing check_constraints with between constraints, should filter accordingly",
                "window_constraints": constraints_case_between_zero_one,
                "summary_df": predicates_df,
                "want": predicates_df.filter(
                    pl.all_horizontal(
                        [
                            pl.col("is_A") >= 0,
                            pl.col("is_A") <= 1,
                            pl.col("is_B") >= 0,
                            pl.col("is_B") <= 1,
                            pl.col("is_C") >= 0,
                            pl.col("is_C") <= 1,
                        ]
                    )
                ),
            },
            {
                "msg": "Testing check_constraints with between constraints, should filter accordingly",
                "window_constraints": constraints_case_between_one_zero,
                "summary_df": predicates_df,
                "want": predicates_df.filter(
                    pl.all_horizontal(
                        [
                            pl.col("is_A") <= 0,
                            pl.col("is_A") >= 1,
                            pl.col("is_B") <= 0,
                            pl.col("is_B") >= 1,
                            pl.col("is_C") <= 0,
                            pl.col("is_C") >= 1,
                        ]
                    )
                ),
            },
        ]

        for c in cases:
            with self.subTest(msg=c.pop("msg")):
                want = c.pop("want")
                got = check_constraints(**c)
                assert str(got) == str(want)
