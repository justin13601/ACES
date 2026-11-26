"""Temporal windows of length zero (#212).

A window like ``end: start + 0d`` (or ``end: start``) spans a single instant. ``[t, t]`` holds the events at
exactly ``t``, and a window with an exclusive end holds nothing. Such a window used to lose its node when the
tree was built, so its ``has`` criteria were ignored and, as a label window, the query failed with
``ColumnNotFoundError: unable to find column "target.end_summary"``.
"""

from datetime import datetime, timedelta

import polars as pl
import pytest

from aces.aggregate import aggregate_temporal_window
from aces.config import EventConfig, PlainPredicateConfig, TaskExtractorConfig, WindowConfig
from aces.query import query
from aces.types import TemporalWindowBounds

PREDICATES_DF = pl.DataFrame(
    {
        "subject_id": [1, 1, 1],
        "timestamp": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
        "A": [1, 0, 0],
        "B": [0, 1, 0],
        "_ANY_EVENT": [1, 1, 1],
    }
)
PREDICATES = {"A": PlainPredicateConfig("A"), "B": PlainPredicateConfig("B")}
INPUT_WINDOW = WindowConfig("trigger", "start + 1d", True, True, index_timestamp="end")


def config(**windows: WindowConfig) -> TaskExtractorConfig:
    return TaskExtractorConfig(
        predicates=PREDICATES, trigger=EventConfig("A"), windows={"input": INPUT_WINDOW, **windows}
    )


@pytest.mark.parametrize("end", ["start + 0d", "start"])
def test_criteria_on_an_empty_zero_length_window_are_checked(end):
    """``(t, t]`` holds no event, so requiring one in it excludes every trigger."""
    cfg = config(
        instant=WindowConfig("input.end", end, False, True, has={"_ANY_EVENT": "(1, None)"}),
        target=WindowConfig("input.end", "start + 1d", False, True, label="B"),
    )
    assert query(cfg, PREDICATES_DF).is_empty()


def test_criteria_on_a_closed_zero_length_window_count_events_at_that_instant():
    """``[t, t]`` at the end of the input window holds the B event at Jan 2."""
    cfg = config(
        instant=WindowConfig("input.end", "start + 0d", True, True, has={"B": "(1, None)"}),
        target=WindowConfig("input.end", "start + 1d", False, True, label="B"),
    )
    assert query(cfg, PREDICATES_DF)["subject_id"].to_list() == [1]

    cfg = config(
        instant=WindowConfig("input.end", "start + 0d", True, True, has={"B": "(None, 0)"}),
        target=WindowConfig("input.end", "start + 1d", False, True, label="B"),
    )
    assert query(cfg, PREDICATES_DF).is_empty()


@pytest.mark.parametrize(("start_inclusive", "end_inclusive", "label"), [(True, True, 1), (True, False, 0)])
def test_zero_length_label_window(start_inclusive, end_inclusive, label):
    cfg = config(target=WindowConfig("input.end", "start + 0d", start_inclusive, end_inclusive, label="B"))
    out = query(cfg, PREDICATES_DF)
    assert out["label"].to_list() == [label]
    assert out["index_timestamp"].to_list() == [datetime(2020, 1, 2)]


@pytest.mark.parametrize(
    ("left_inclusive", "right_inclusive", "offset", "expected_b"),
    [
        (True, True, timedelta(0), [0, 1, 0]),
        (False, True, timedelta(0), [0, 0, 0]),
        (True, False, timedelta(0), [0, 0, 0]),
        (True, True, timedelta(days=1), [1, 0, 0]),
        (True, True, timedelta(hours=12), [0, 0, 0]),
    ],
)
def test_aggregate_zero_length_temporal_window(left_inclusive, right_inclusive, offset, expected_b):
    bounds = TemporalWindowBounds(left_inclusive, timedelta(0), right_inclusive, offset)
    for df in (PREDICATES_DF, PREDICATES_DF[1:2]):
        out = aggregate_temporal_window(df, bounds)
        assert out["timestamp"].to_list() == df["timestamp"].to_list()
        assert out["timestamp_at_start"].to_list() == out["timestamp_at_end"].to_list()
        assert out["timestamp_at_start"].to_list() == [t + offset for t in df["timestamp"]]
        if len(df) == len(PREDICATES_DF):
            assert out["B"].to_list() == expected_b
