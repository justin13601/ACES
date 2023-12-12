import rootutils

root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)

import unittest

from esgpt_task_querying import (
    MLTypeEqualityCheckableMixin,
    count_or_proportion,
    lt_count_or_proportion,
    num_initial_spaces,
)


class TestUtilFunctions(MLTypeEqualityCheckableMixin, unittest.TestCase):
    """Tests `EventStreamData.utils`."""

    def test_count_or_proportion(self):
        self.assertEqual(4, count_or_proportion(10, 4))
        self.assertEqual(3, count_or_proportion(10, 1 / 3))
        with self.assertRaises(TypeError):
            count_or_proportion(10, "foo")
        with self.assertRaises(ValueError):
            count_or_proportion(10, 1.1)
        with self.assertRaises(ValueError):
            count_or_proportion(10, -0.1)
        with self.assertRaises(ValueError):
            count_or_proportion(10, 0)
        with self.assertRaises(TypeError):
            count_or_proportion("foo", 1 / 3)

    def test_lt_count_or_proportion(self):
        self.assertFalse(lt_count_or_proportion(10, None, 100))
        self.assertTrue(lt_count_or_proportion(10, 11, 100))
        self.assertFalse(lt_count_or_proportion(12, 11, 100))

    def test_num_initial_spaces(self):
        self.assertEqual(0, num_initial_spaces("foo"))
        self.assertEqual(3, num_initial_spaces("   \tfoo"))
