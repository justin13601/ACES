"""Event-bound windows must not depend on the time scale of the data.

``boolean_expr_bound_sum`` places each boundary event 1 microsecond beside its (offset) timestamp, so that it
falls between rows. When rows are only 1 microsecond apart, a shifted boundary ties with a real row, and the
tie must be broken the way the shift intends. These tests run the same events 1 second apart and 1
microsecond apart and require identical windows.
"""

from datetime import datetime, timedelta

import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from aces.aggregate import boolean_expr_bound_sum
from aces.config import TaskExtractorConfig
from aces.query import query

T0 = datetime(2020, 1, 1)
SECOND, MICROSECOND = timedelta(seconds=1), timedelta(microseconds=1)


def run(is_boundary: list[bool], counts: list[int], mode: str, closed: str, k: int, unit: timedelta):
    """Runs one subject with rows ``unit`` apart and an offset of ``k`` units; times are returned in units."""
    df = pl.DataFrame(
        {
            "subject_id": [1] * len(counts),
            "timestamp": [T0 + i * unit for i in range(len(counts))],
            "is_boundary": [int(b) for b in is_boundary],
            "count": counts,
        }
    )
    out = boolean_expr_bound_sum(df, pl.col("is_boundary") > 0, mode, closed, k * unit)

    def in_units(col: str) -> pl.Expr:
        return ((pl.col(col) - T0) / unit).round().cast(pl.Int64)

    return out.select(
        in_units("timestamp"), in_units("timestamp_at_start"), in_units("timestamp_at_end"), "count"
    ).rows()


MODES = ["bound_to_row", "row_to_bound"]
CLOSED = ["both", "left", "right", "none"]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("closed", CLOSED)
@pytest.mark.parametrize("k", [-2, -1, 0, 1, 2])
def test_docstring_pattern_is_scale_invariant(mode, closed, k):
    is_boundary = [False, True, False, True, True, False, False]
    counts = [1, 2, 4, 8, 16, 32, 64]  # distinct powers of two, so each sum names its rows
    assert run(is_boundary, counts, mode, closed, k, MICROSECOND) == run(
        is_boundary, counts, mode, closed, k, SECOND
    )


@settings(max_examples=300, deadline=None)
@given(
    rows=st.lists(st.booleans(), min_size=1, max_size=8),
    mode=st.sampled_from(MODES),
    closed=st.sampled_from(CLOSED),
    k=st.integers(-3, 3),
)
def test_event_bound_sums_are_scale_invariant(rows, mode, closed, k):
    counts = [2**i for i in range(len(rows))]
    assert run(rows, counts, mode, closed, k, MICROSECOND) == run(rows, counts, mode, closed, k, SECOND)


def test_event_bound_window_ignores_event_1us_before_trigger(tmp_path):
    """A discharge 1 microsecond before an admission must not end that admission's stay."""
    config_path = tmp_path / "task.yaml"
    config_path.write_text(
        "predicates:\n"
        "  adm: {code: ADM}\n"
        "  dis: {code: DIS}\n"
        "trigger: adm\n"
        "windows:\n"
        "  stay: {start: trigger, end: start -> dis, start_inclusive: True, end_inclusive: True}\n"
    )
    cfg = TaskExtractorConfig.load(config_path)
    hour = timedelta(hours=1)
    predicates_df = pl.DataFrame(
        {
            "subject_id": [1, 1, 1, 1],
            "timestamp": [T0, T0 + hour - MICROSECOND, T0 + hour, T0 + 2 * hour],
            "adm": [1, 0, 1, 0],
            "dis": [0, 1, 0, 1],
        }
    )

    result = query(cfg, predicates_df).sort("trigger")

    assert result["trigger"].to_list() == [T0, T0 + hour]
    assert result["stay.end_summary"].struct.field("timestamp_at_end").to_list() == [
        T0 + hour - MICROSECOND,
        T0 + 2 * hour,
    ]
