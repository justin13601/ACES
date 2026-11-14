"""Dropping MEDS events that match no predicate must not change any query result.

``get_predicates_df`` prunes MEDS events that match no predicate (keeping static events and each subject's
first and last timestamps) unless the task counts every event. These tests run each task with pruning on
and off and require identical output, on data whose record bounds, static rows and in-between events all
match no predicate.
"""

from datetime import datetime, timedelta

import polars as pl
import pytest
from omegaconf import DictConfig

from aces.config import TaskExtractorConfig
from aces.predicates import get_predicates_df
from aces.query import query

T0 = datetime(2020, 1, 1)


def _events(subject_id: int, rows: list[tuple[float | None, str, float | None]]) -> list[dict]:
    return [
        {
            "subject_id": subject_id,
            "time": None if days is None else T0 + timedelta(days=days),
            "code": code,
            "numeric_value": value,
        }
        for days, code, value in rows
    ]


EVENTS = [
    # Starts and ends with events that match no predicate, so the record bounds are prunable rows.
    *_events(
        1,
        [
            (None, "GENDER//M", None),
            (0, "LAB//other", 1.0),
            (1, "ADMISSION//ER", None),
            (1.5, "LAB//other", 2.0),
            (1.6, "LAB//spo2", 95.0),
            (2, "LAB//spo2", 85.0),
            (3, "PROCEDURE//START//vent", None),
            (3.5, "PROCEDURE//END//vent", None),
            (4, "DISCHARGE//HOME", None),
            (20, "ADMISSION//ER", None),
            (20.5, "LAB//spo2", 92.0),
            (21, "LAB//spo2", 130.0),
            (23, "DISCHARGE//HOME", None),
            (400, "DIAGNOSIS//MI", None),
            (500, "LAB//other", 3.0),
        ],
    ),
    *_events(
        2,
        [
            (None, "GENDER//F", None),
            (10, "LAB//other", 1.0),
            (11, "ADMISSION//ER", None),
            (11.2, "LAB//spo2", 97.0),
            (12, "LAB//other", 1.0),
            (14, "MEDS_DEATH", None),
        ],
    ),
    # Only events that match no predicate.
    *_events(3, [(None, "GENDER//M", None), (5, "LAB//other", 1.0), (6, "LAB//other", 2.0)]),
    # A single timestamp, which is both record bounds.
    *_events(4, [(30, "ADMISSION//ER", None), (30, "LAB//other", 1.0)]),
]

ADM = 'admission: {code: {regex: "^ADMISSION//"}}'
DIS = 'discharge: {code: {regex: "^DISCHARGE//"}}'

CONFIGS = {
    "inhospital_mortality_demographics": f"""
predicates:
  {ADM}
  {DIS}
  death: {{code: MEDS_DEATH}}
  discharge_or_death: {{expr: "or(discharge, death)"}}
patient_demographics:
  male: {{code: GENDER//M}}
trigger: admission
windows:
  input: {{start: NULL, end: trigger + 6h, start_inclusive: True, end_inclusive: True, index_timestamp: end}}
  gap: {{start: input.end, end: start + 12h, start_inclusive: False, end_inclusive: True,
        has: {{discharge: "(None, 0)", death: "(None, 0)"}}}}
  target: {{start: gap.end, end: start -> discharge_or_death, start_inclusive: False, end_inclusive: True,
           label: death}}
""",
    "readmission_event_bound": f"""
predicates:
  {ADM}
  {DIS}
trigger: admission
windows:
  input: {{start: trigger, end: start -> discharge, start_inclusive: False, end_inclusive: True,
          index_timestamp: end}}
  target: {{start: input.end, end: start + 30 days, start_inclusive: False, end_inclusive: True,
           label: admission}}
""",
    "backward_event_bound": f"""
predicates:
  {ADM}
  {DIS}
  mi: {{code: DIAGNOSIS//MI}}
trigger: discharge
windows:
  input: {{start: end <- admission, end: trigger, start_inclusive: False, end_inclusive: True,
          index_timestamp: end}}
  target: {{start: trigger, end: start + 1000 days, start_inclusive: False, end_inclusive: True, label: mi}}
""",
    "record_bounds": f"""
predicates:
  {ADM}
  death: {{code: MEDS_DEATH}}
trigger: admission
windows:
  input: {{start: NULL, end: trigger, start_inclusive: True, end_inclusive: True, index_timestamp: start}}
  target: {{start: trigger, end: start + 30 days, start_inclusive: False, end_inclusive: True,
           has: {{_RECORD_END: "(None, 0)"}}, label: death}}
""",
    "value_ranges": """
predicates:
  spo2: {code: LAB//spo2}
  normal_spo2: {code: LAB//spo2, value_min: 90, value_max: 120, value_min_inclusive: True,
                value_max_inclusive: True}
  low_spo2: {code: LAB//spo2, value_max: 90, value_max_inclusive: False}
  high_spo2: {code: LAB//spo2, value_min: 120, value_min_inclusive: False}
  abnormal_spo2: {expr: "or(low_spo2, high_spo2)"}
trigger: normal_spo2
windows:
  input: {start: NULL, end: trigger, start_inclusive: True, end_inclusive: True, index_timestamp: end}
  target: {start: trigger, end: start + 2 days, start_inclusive: False, end_inclusive: True,
           has: {spo2: "(1, None)"}, label: abnormal_spo2}
""",
    "derived_and": """
predicates:
  procedure_start: {code: {regex: "^PROCEDURE//START//"}}
  procedure_end: {code: {regex: "^PROCEDURE//END//"}}
  vent: {code: {regex: "//vent$"}}
  vent_start: {expr: "and(procedure_start, vent)"}
  vent_end: {expr: "and(procedure_end, vent)"}
trigger: vent_start
windows:
  input: {start: NULL, end: trigger, start_inclusive: True, end_inclusive: True, index_timestamp: end}
  target: {start: trigger, end: start -> vent_end, start_inclusive: False, end_inclusive: True}
""",
}

ANY_EVENT_CONFIG = f"""
predicates:
  {ADM}
trigger: admission
windows:
  input: {{start: NULL, end: trigger, start_inclusive: True, end_inclusive: True,
          has: {{_ANY_EVENT: "(3, None)"}}, index_timestamp: end}}
"""


@pytest.fixture()
def shard(tmp_path):
    path = tmp_path / "0.parquet"
    pl.DataFrame(
        EVENTS,
        schema={
            "subject_id": pl.Int64,
            "time": pl.Datetime("us"),
            "code": pl.String,
            "numeric_value": pl.Float32,
        },
    ).write_parquet(path)
    return path


def _run(config_text: str, tmp_path, shard, prune: bool) -> tuple[pl.DataFrame, pl.DataFrame]:
    config_path = tmp_path / "task.yaml"
    config_path.write_text(config_text)
    cfg = TaskExtractorConfig.load(config_path=config_path)
    predicates_df = get_predicates_df(
        cfg, DictConfig({"path": str(shard), "standard": "meds", "prune": prune})
    )
    result = query(TaskExtractorConfig.load(config_path=config_path), predicates_df)
    return predicates_df, result.sort(["subject_id", "trigger"])


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_pruning_does_not_change_results(name, tmp_path, shard):
    full_predicates, expected = _run(CONFIGS[name], tmp_path, shard, prune=False)
    pruned_predicates, got = _run(CONFIGS[name], tmp_path, shard, prune=True)

    assert pruned_predicates.height < full_predicates.height, "the test data should have prunable rows"
    assert expected.height > 0, "the task should select something"
    assert got.equals(expected)


def test_tasks_counting_every_event_are_not_pruned(tmp_path, shard):
    full_predicates, expected = _run(ANY_EVENT_CONFIG, tmp_path, shard, prune=False)
    pruned_predicates, got = _run(ANY_EVENT_CONFIG, tmp_path, shard, prune=True)

    assert pruned_predicates.height == full_predicates.height
    assert got.equals(expected)
