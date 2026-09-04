"""Regression test for issue #168: `_ANY_EVENT` must not include MEDS_BIRTH rows."""

import tempfile
from datetime import datetime
from pathlib import Path

import meds
import polars as pl

from aces.predicates import generate_plain_predicates_from_meds


def test_meds_birth_rows_do_not_contribute_to_any_event():
    """A MEDS birth row should not inflate downstream `_ANY_EVENT` counts.

    Historically `_ANY_EVENT` was defined as "every (subject_id, timestamp) with a non-null
    timestamp", which incorrectly swept in `meds.birth_code` rows (the birth row is a quasi-
    static demographic marker anchored at the patient's birth time, not a clinical event).
    See https://github.com/justin13601/ACES/issues/168.
    """
    data = pl.DataFrame(
        {
            "subject_id": [1, 1, 1, 2],
            "time": [
                datetime(1980, 1, 1),  # birth
                datetime(2020, 1, 1),  # clinical event
                datetime(2020, 1, 2),  # clinical event
                datetime(1990, 6, 1),  # birth-only patient
            ],
            "code": [meds.birth_code, "LAB//HR", "LAB//HR", meds.birth_code],
            "numeric_value": [None, 72.0, 80.0, None],
        }
    )

    with tempfile.TemporaryDirectory() as d:
        data_path = Path(d) / "data.parquet"
        data.write_parquet(data_path)
        got = generate_plain_predicates_from_meds(data_path, predicates={})

    # Subject 1 should have 2 rows (the two clinical events), not 3 (which would include birth).
    # Subject 2 had only a birth row, so it should drop out entirely.
    got = got.sort("subject_id", "timestamp")
    assert got.height == 2, f"Expected 2 rows post-birth-filter; got {got.height}:\n{got}"
    assert got["subject_id"].to_list() == [1, 1]
    assert got["timestamp"].to_list() == [datetime(2020, 1, 1), datetime(2020, 1, 2)]
