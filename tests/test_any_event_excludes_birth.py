"""MEDS_BIRTH rows are part of the record but are not events.

- ``_ANY_EVENT`` does not count them (issue #168).
- They stay in the predicates dataframe, so ``_RECORD_START`` is the birth date and ``code: MEDS_BIRTH`` can
  be matched, e.g. to exclude children (age criteria broke in 0.8.0, when the rows were dropped).
"""

from datetime import datetime

import meds
import polars as pl
import pytest
from omegaconf import DictConfig

from aces.config import TaskExtractorConfig
from aces.predicates import get_predicates_df
from aces.query import query

ADULT_DAYS = 6575  # 18 years


@pytest.fixture()
def data_path(tmp_path):
    """Admissions on 2020-01-01, each after a lab on 2019-06-01.

    - Subject 1 is a child (born 2010).
    - Subject 2 is an adult (born 1980) with a long record.
    - Subject 3 is an adult (born 1980) whose first event other than birth is in 2019.
    - Subject 4 has a static code and a lab at the same timestamp as their birth (born 1980).
    - Subject 5 has only a birth row.
    """
    rows = [
        (1, datetime(2010, 1, 1), meds.birth_code),
        (1, datetime(2019, 6, 1), "LAB"),
        (1, datetime(2020, 1, 1), "ADMISSION"),
        (2, datetime(1980, 1, 1), meds.birth_code),
        (2, datetime(2000, 1, 1), "LAB"),
        (2, datetime(2019, 6, 1), "LAB"),
        (2, datetime(2020, 1, 1), "ADMISSION"),
        (3, datetime(1980, 1, 1), meds.birth_code),
        (3, datetime(2019, 6, 1), "LAB"),
        (3, datetime(2020, 1, 1), "ADMISSION"),
        (4, None, "GENDER//F"),
        (4, datetime(1980, 1, 1), meds.birth_code),
        (4, datetime(1980, 1, 1), "LAB"),
        (4, datetime(2020, 1, 1), "ADMISSION"),
        (5, datetime(1990, 6, 1), meds.birth_code),
    ]
    path = tmp_path / "data.parquet"
    pl.DataFrame(
        {
            "subject_id": [s for s, _, _ in rows],
            "time": pl.Series([t for _, t, _ in rows], dtype=pl.Datetime("us")),
            "code": [c for _, _, c in rows],
            "numeric_value": pl.Series([None] * len(rows), dtype=pl.Float32),
        }
    ).write_parquet(path)
    return path


def run(tmp_path, data_path, task_yaml: str, prune: bool = True) -> pl.DataFrame:
    config_path = tmp_path / "task.yaml"
    config_path.write_text(task_yaml)
    cfg = TaskExtractorConfig.load(config_path)
    data_config = DictConfig({"path": str(data_path), "standard": "meds", "prune": prune})
    return query(cfg, get_predicates_df(cfg, data_config)).sort("subject_id")


def test_any_event_does_not_count_birth_rows(tmp_path, data_path):
    """Birth plus two events must not pass a gate of three events; a lab at the birth timestamp counts."""
    task = """
predicates:
  admission: {code: ADMISSION}
trigger: admission
windows:
  input:
    start: null
    end: trigger
    start_inclusive: True
    end_inclusive: True
    has: {_ANY_EVENT: "(3, None)"}
"""
    result = run(tmp_path, data_path, task)

    assert result["subject_id"].to_list() == [2]
    counts = run(tmp_path, data_path, task.replace("(3, None)", "(1, None)"))
    any_events = counts.select("subject_id", pl.col("input.start_summary").struct.field("_ANY_EVENT"))
    assert any_events.rows() == [(1, 2), (2, 3), (3, 2), (4, 2)]


def test_any_event_trigger_skips_birth_only_timestamps(tmp_path, data_path):
    task = """
predicates:
  admission: {code: ADMISSION}
trigger: _ANY_EVENT
windows:
  input: {start: null, end: trigger, start_inclusive: True, end_inclusive: True}
"""
    result = run(tmp_path, data_path, task)

    assert datetime(2010, 1, 1) not in result["trigger"].to_list()
    assert 5 not in result["subject_id"].to_list()
    assert (4, datetime(1980, 1, 1)) in result.select("subject_id", "trigger").rows()


@pytest.mark.parametrize("prune", [True, False])
def test_meds_birth_predicate_excludes_children(tmp_path, data_path, prune):
    task = f"""
predicates:
  admission: {{code: ADMISSION}}
  birth: {{code: {meds.birth_code}}}
trigger: admission
windows:
  age:
    start: end - {ADULT_DAYS}d
    end: trigger
    start_inclusive: True
    end_inclusive: True
    has: {{birth: "(None, 0)"}}
"""
    assert run(tmp_path, data_path, task, prune=prune)["subject_id"].to_list() == [2, 3, 4]


@pytest.mark.parametrize("prune", [True, False])
def test_record_start_is_the_birth_date(tmp_path, data_path, prune):
    task = f"""
predicates:
  admission: {{code: ADMISSION}}
trigger: admission
windows:
  age:
    start: end - {ADULT_DAYS}d
    end: trigger
    start_inclusive: True
    end_inclusive: True
    has: {{_RECORD_START: "(None, 0)"}}
  history:
    start: null
    end: trigger
    start_inclusive: True
    end_inclusive: True
"""
    result = run(tmp_path, data_path, task, prune=prune)

    assert result["subject_id"].to_list() == [2, 3, 4]
    assert (
        result["history.start_summary"].struct.field("timestamp_at_start").to_list()
        == [datetime(1980, 1, 1)] * 3
    )
