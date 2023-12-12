import rootutils

root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)

import unittest

from esgpt_task_querying.query import (
    query_subtree,
    summarize_event_bound_window,
    summarize_temporal_window,
    summarize_window,
)


class TestQueryFunctions(unittest.TestCase):
    def test_summarize_temporal_window(self):
        raise NotImplementedError

    def test_summarize_event_bound_window(self):
        raise NotImplementedError

    def test_summarize_window(self):
        raise NotImplementedError

    def test_query_subtree(self):
        raise NotImplementedError
