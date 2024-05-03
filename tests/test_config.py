import rootutils

root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)

import unittest

import polars as pl
from polars.testing import assert_frame_equal

from esgpt_task_querying.predicates import (
    get_config,
    has_event_type,
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
