"""Tests for repeated prediction times via a trigger with ``every`` (and ``until``)."""

from datetime import datetime

import polars as pl
import pytest

from aces.config import TaskExtractorConfig
from aces.query import query

TASK_YAML = """
predicates:
  icu_admission:
    code: icu_admission
  icu_discharge:
    code: icu_discharge
  death:
    code: death
trigger:
  predicate: icu_admission
  every: 1h
  until: icu_discharge
windows:
  input:
    start: null
    end: trigger
    start_inclusive: True
    end_inclusive: True
    index_timestamp: end
  target:
    start: input.end
    end: start + 2h
    start_inclusive: False
    end_inclusive: True
    label: death
"""


def at(hour: int, minute: int = 0) -> datetime:
    return datetime(2020, 1, 1, hour, minute)


@pytest.fixture()
def predicates_df():
    """Subject 1 has two ICU stays; subject 2 dies in the ICU, which ends their record (no discharge)."""
    rows = [
        (1, at(0), "icu_admission"),
        (1, at(2, 30), "lab"),
        (1, at(3, 30), "icu_discharge"),
        (1, at(10), "icu_admission"),
        (1, at(11), "icu_discharge"),
        (1, at(20), "lab"),
        (2, at(0), "icu_admission"),
        (2, at(2, 30), "death"),
    ]
    codes = ["icu_admission", "icu_discharge", "death"]
    return pl.DataFrame(
        {
            "subject_id": [s for s, _, _ in rows],
            "timestamp": [t for _, t, _ in rows],
            **{c: [int(code == c) for _, _, code in rows] for c in codes},
        }
    )


def test_trigger_every_hour_until_discharge(tmp_path, predicates_df):
    config_path = tmp_path / "decompensation.yaml"
    config_path.write_text(TASK_YAML)
    cfg = TaskExtractorConfig.load(config_path)

    result = query(cfg, predicates_df).sort("subject_id", "trigger")

    # Subject 1: hourly from each admission up to (excluding) the following discharge.
    # Subject 2: hourly from admission up to (excluding) the end of their record at death.
    assert result.select("subject_id", "trigger").rows() == [
        (1, at(0)),
        (1, at(1)),
        (1, at(2)),
        (1, at(3)),
        (1, at(10)),
        (2, at(0)),
        (2, at(1)),
        (2, at(2)),
    ]
    assert result["index_timestamp"].to_list() == result["trigger"].to_list()
    # Death at 02:30 is within 2h of 01:00 and 02:00 only; the added grid rows count as no events.
    assert result["label"].to_list() == [0, 0, 0, 0, 0, 0, 1, 1]


def test_trigger_every_without_until_runs_to_record_end(tmp_path, predicates_df):
    config_path = tmp_path / "every_6h.yaml"
    config_path.write_text(TASK_YAML.replace("  every: 1h\n  until: icu_discharge\n", "  every: 6h\n"))
    cfg = TaskExtractorConfig.load(config_path)

    result = query(cfg, predicates_df).sort("subject_id", "trigger")

    # Subject 1's grids from 00:00 and 10:00 both run to their last event at 20:00.
    assert result.select("subject_id", "trigger").rows() == [
        (1, at(0)),
        (1, at(6)),
        (1, at(10)),
        (1, at(12)),
        (1, at(16)),
        (1, at(18)),
        (2, at(0)),
    ]
