"""Tests for handling incomplete follow-up in the label window via ``censoring``."""

from datetime import datetime

import polars as pl
import pytest

from aces.config import TaskExtractorConfig
from aces.query import query

TASK_YAML = """
predicates:
  admission:
    code: admission
  discharge:
    code: discharge
trigger: discharge
windows:
  input:
    start: null
    end: trigger
    start_inclusive: True
    end_inclusive: True
    index_timestamp: end
  target:
    start: input.end
    end: start + 30 days
    start_inclusive: False
    end_inclusive: True
    label: admission
"""


def load_config(tmp_path, censoring: str | None) -> TaskExtractorConfig:
    config_path = tmp_path / "readmission.yaml"
    config_path.write_text(TASK_YAML + (f"    censoring: {censoring}\n" if censoring else ""))
    return TaskExtractorConfig.load(config_path)


@pytest.fixture()
def predicates_df():
    """Every subject is discharged on Jan 3; the target window is (Jan 3, Feb 2].

    - Subject 1: not readmitted, but the record ends on Jan 8, so the outcome is unobserved (censored).
    - Subject 2: not readmitted, followed until March (negative).
    - Subject 3: not readmitted, record ends exactly at the end of the window (negative).
    - Subject 4: readmitted on Jan 8, after which the record ends (positive, not censored).
    """
    rows = [
        (1, datetime(2020, 1, 1), "admission"),
        (1, datetime(2020, 1, 3), "discharge"),
        (1, datetime(2020, 1, 8), "other"),
        (2, datetime(2020, 1, 1), "admission"),
        (2, datetime(2020, 1, 3), "discharge"),
        (2, datetime(2020, 3, 1), "other"),
        (3, datetime(2020, 1, 1), "admission"),
        (3, datetime(2020, 1, 3), "discharge"),
        (3, datetime(2020, 2, 2), "other"),
        (4, datetime(2020, 1, 1), "admission"),
        (4, datetime(2020, 1, 3), "discharge"),
        (4, datetime(2020, 1, 8), "admission"),
    ]
    return pl.DataFrame(
        {
            "subject_id": [s for s, _, _ in rows],
            "timestamp": [t for _, t, _ in rows],
            "admission": [int(c == "admission") for _, _, c in rows],
            "discharge": [int(c == "discharge") for _, _, c in rows],
        }
    )


def test_default_keeps_censored_rows_as_negative(tmp_path, predicates_df):
    result = query(load_config(tmp_path, None), predicates_df).sort("subject_id")
    assert result["subject_id"].to_list() == [1, 2, 3, 4]
    assert result["label"].to_list() == [0, 0, 0, 1]
    assert "censored" not in result.columns


def test_drop_removes_only_censored_negatives(tmp_path, predicates_df):
    result = query(load_config(tmp_path, "drop"), predicates_df).sort("subject_id")
    assert result["subject_id"].to_list() == [2, 3, 4]
    assert result["label"].to_list() == [0, 0, 1]


def test_flag_adds_censored_column(tmp_path, predicates_df):
    result = query(load_config(tmp_path, "flag"), predicates_df).sort("subject_id")
    assert result["subject_id"].to_list() == [1, 2, 3, 4]
    assert result["censored"].to_list() == [True, False, False, False]


UNTIL_DISCHARGE_YAML = """
predicates:
  admission:
    code: admission
  discharge:
    code: discharge
  death:
    code: death
  discharge_or_death:
    expr: or(discharge, death)
trigger: admission
windows:
  target:
    start: trigger
    end: start -> discharge_or_death
    start_inclusive: False
    end_inclusive: True
    label: death
    censoring: {censoring}
"""


@pytest.mark.parametrize(("censoring", "expected"), [("drop", [1, 2]), ("flag", [1, 2, 3])])
def test_event_bound_window_that_never_ends_is_censored(tmp_path, censoring, expected):
    """Subject 1 is discharged and 2 dies, but 3's record ends before any discharge or death."""
    config_path = tmp_path / "inhospital_mortality.yaml"
    config_path.write_text(UNTIL_DISCHARGE_YAML.format(censoring=censoring))
    rows = [(1, 1, "admission"), (1, 5, "discharge"), (2, 1, "admission"), (2, 3, "death")]
    rows += [(3, 1, "admission"), (3, 4, "lab")]
    predicates_df = pl.DataFrame(
        {
            "subject_id": [s for s, _, _ in rows],
            "timestamp": [datetime(2020, 1, d) for _, d, _ in rows],
            **{c: [int(code == c) for _, _, code in rows] for c in ["admission", "discharge", "death"]},
        }
    ).with_columns((pl.col("discharge") + pl.col("death")).alias("discharge_or_death"))

    result = query(TaskExtractorConfig.load(config_path), predicates_df).sort("subject_id")

    assert result["subject_id"].to_list() == expected
    assert result["label"].to_list()[:2] == [0, 1]
    if censoring == "flag":
        assert result["censored"].to_list() == [False, False, True]
