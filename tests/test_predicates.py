import rootutils

root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)

import unittest

import polars as pl
from polars.testing import assert_frame_equal

from esgpt_task_querying.predicates import (
    get_config,
    has_event_type,
    generate_simple_predicates,
    generate_predicate_columns,
)


class TestQueryFunctions(unittest.TestCase):
    def setUp(self):

        self.addTypeEqualityFunc(
            pl.DataFrame,
            lambda a, b, msg: assert_frame_equal(a, b, check_column_order=False),
        )

    def test_get_config(self):
        cfg = {"key1": "value1", "key2": None}

        cases = [
            {
                "msg": "Should return default value",
                "cfg": cfg,
                "key": "key3",
                "default": "default",
                "want": "default",
            },
            {
                "msg": "Should return value",
                "cfg": cfg,
                "key": "key1",
                "default": "default",
                "want": "value1",
            },
            {
                "msg": "Should return None",
                "cfg": cfg,
                "key": "key2",
                "default": "default",
                "want": "default",
            },
        ]

        for c in cases:
            with self.subTest(msg=c.pop("msg")):
                want = c.pop("want")
                got = get_config(**c)
                self.assertEqual(got, want)

    def test_has_event_type(self):
        data = pl.DataFrame({"event_type": ["A&B&C", "A&B", "C", "", "A", "B", "C&A"]})
        cases = [
            {
                "msg": "",
                "type_str": "A",
                "want": pl.DataFrame(
                    {"event_type": [True, True, False, False, True, False, True]}
                ),
            },
        ]

        for c in cases:
            with self.subTest(msg=c.pop("msg")):
                want = c.pop("want")
                got = data.with_columns(has_event_type(**c))
                self.assertEqual(got, want)

    def test_generate_simple_predicates(self):
        predicate_name = "A"
        predicate_info = {"column": "event_type", "value": "A", "system": "boolean"}
        data = pl.DataFrame(
            {
                "subject_id": [1, 1, 1, 2, 2, 3, 3],
                "timestamp": [1, 1, 3, 1, 2, 1, 3],
                "event_type": ["A", "B", "C", "A", "A&C&D", "B", "C"],
            }
        )
        cases = [
            {
                "msg": "Should return simple boolean predicates",
                "predicate_name": predicate_name,
                "predicate_info": predicate_info,
                "df": data,
                "want": pl.DataFrame(
                    {
                        "subject_id": [1, 1, 1, 2, 2, 3, 3],
                        "timestamp": [1, 1, 3, 1, 2, 1, 3],
                        "event_type": ["A", "B", "C", "A", "A&C&D", "B", "C"],
                        "is_A": [1, 0, 0, 1, 1, 0, 0],
                    }
                ),
            },
        ]

        for c in cases:
            with self.subTest(msg=c.pop("msg")):
                want = c.pop("want")
                got = generate_simple_predicates(**c)
                self.assertEqual(got, want)

    def test_generate_predicate_columns(self):
        cfg = {
            "predicates": {
                "A": {"column": "event_type", "value": "A", "system": "boolean"},
                "B": {"column": "event_type", "value": "B", "system": "boolean"},
                "C": {"column": "event_type", "value": "C", "system": "boolean"},
                "D": {"column": "event_type", "value": "D", "system": "boolean"},
                "A_or_B": {
                    "type": "ANY",
                    "predicates": ["A", "B"],
                    "system": "boolean",
                },
                "A_and_C_and_D": {
                    "type": "ALL",
                    "predicates": ["A", "C", "D"],
                    "system": "boolean",
                },
                "any": {"type": "special"},
            }
        }

        data = pl.DataFrame(
            {
                "subject_id": [1, 1, 1, 2, 2, 3, 3],
                "timestamp": [1, 1, 3, 1, 2, 1, 3],
                "event_type": ["A", "B", "C", "A", "A&C&D", "B", "C"],
            }
        )

        cases = [
            {
                "msg": "Should return full predicates dataframe",
                "cfg": cfg,
                "data": data,
                "want": pl.DataFrame(
                    {
                        "subject_id": [1, 1, 2, 2, 3, 3],
                        "timestamp": [1, 3, 1, 2, 1, 3],
                        "is_A": [1, 0, 1, 1, 0, 0],
                        "is_B": [1, 0, 0, 0, 1, 0],
                        "is_C": [0, 1, 0, 1, 0, 1],
                        "is_D": [0, 0, 0, 1, 0, 0],
                        "is_A_or_B": [1, 0, 1, 1, 1, 0],
                        "is_A_and_C_and_D": [0, 0, 0, 1, 0, 0],
                        "is_any": [1, 1, 1, 1, 1, 1],
                    }
                ),
            },
        ]

        for c in cases:
            with self.subTest(msg=c.pop("msg")):
                want = c.pop("want")
                got = generate_predicate_columns(**c)
                self.assertEqual(got, want)
