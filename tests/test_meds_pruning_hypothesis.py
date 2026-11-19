"""Property test: on random MEDS data, every task gives the same result with and without pruning, and every
returned row is complete (no null subject, trigger, index timestamp or label).

The tasks use windows of hours and few constraints so that random data usually selects some subjects, and
between them they depend on everything pruning must preserve: the record start (``start: NULL`` with
``index_timestamp: start``) and end (``_RECORD_END``), forward and backward event bounds, value ranges,
``and()`` predicates, repeated triggers, static demographics (also inside ``and()`` / ``or()``), MEDS_BIRTH
rows and ``_ANY_EVENT``.
"""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
from hypothesis import event, given, settings
from hypothesis import strategies as st
from omegaconf import DictConfig

from aces.config import TaskExtractorConfig
from aces.predicates import get_predicates_df
from aces.query import query

T0 = datetime(2020, 1, 1)

ADM = 'admission: {code: {regex: "^ADMISSION//"}}'
DIS = 'discharge: {code: {regex: "^DISCHARGE//"}}'
DEATH = "death: {code: MEDS_DEATH}"

TASKS = {
    "record_start_index": f"""
predicates:
  {ADM}
  {DEATH}
trigger: admission
windows:
  input: {{start: NULL, end: trigger, start_inclusive: True, end_inclusive: True, index_timestamp: start}}
  target: {{start: trigger, end: start + 24h, start_inclusive: False, end_inclusive: True, label: death}}
""",
    "record_end_constraint": f"""
predicates:
  {ADM}
  {DEATH}
trigger: admission
windows:
  target: {{start: trigger, end: start + 12h, start_inclusive: False, end_inclusive: True,
           has: {{_RECORD_END: "(None, 0)"}}, label: death, index_timestamp: start}}
""",
    "forward_event_bound": f"""
predicates:
  {ADM}
  {DIS}
trigger: admission
windows:
  input: {{start: trigger, end: start -> discharge, start_inclusive: False, end_inclusive: True,
          index_timestamp: end}}
  target: {{start: input.end, end: start + 48h, start_inclusive: False, end_inclusive: True,
           label: admission}}
""",
    "backward_event_bound": f"""
predicates:
  {ADM}
  {DIS}
  {DEATH}
trigger: discharge
windows:
  input: {{start: end <- admission, end: trigger, start_inclusive: False, end_inclusive: True,
          index_timestamp: end}}
  target: {{start: trigger, end: start + 48h, start_inclusive: False, end_inclusive: True, label: death}}
""",
    # The event-bound window is the last of a chain with one child per level (no sibling to inner-join
    # with), so a stay with no later discharge or death leaves it unclosed (#232).
    "event_bound_label_window": f"""
predicates:
  {ADM}
  {DIS}
  {DEATH}
  discharge_or_death: {{expr: "or(discharge, death)"}}
trigger: admission
windows:
  input: {{start: trigger, end: start + 6h, start_inclusive: True, end_inclusive: True,
          index_timestamp: end}}
  target: {{start: input.end, end: start -> discharge_or_death, start_inclusive: False,
           end_inclusive: True, label: death}}
""",
    "value_ranges": """
predicates:
  normal_spo2: {code: LAB//spo2, value_min: 90, value_max: 120, value_min_inclusive: True,
                value_max_inclusive: True}
  low_spo2: {code: LAB//spo2, value_max: 90, value_max_inclusive: False}
  high_spo2: {code: LAB//spo2, value_min: 120, value_min_inclusive: False}
  abnormal_spo2: {expr: "or(low_spo2, high_spo2)"}
trigger: normal_spo2
windows:
  input: {start: NULL, end: trigger, start_inclusive: True, end_inclusive: True, index_timestamp: end}
  target: {start: trigger, end: start + 24h, start_inclusive: False, end_inclusive: True,
           label: abnormal_spo2}
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
  input: {start: NULL, end: trigger, start_inclusive: True, end_inclusive: True, index_timestamp: start}
  target: {start: trigger, end: start -> vent_end, start_inclusive: False, end_inclusive: True}
""",
    "repeated_trigger": f"""
predicates:
  {ADM}
  {DIS}
  {DEATH}
trigger:
  predicate: admission
  every: 6h
  until: discharge
windows:
  input: {{start: NULL, end: trigger, start_inclusive: True, end_inclusive: True, index_timestamp: end}}
  target: {{start: trigger, end: start + 24h, start_inclusive: False, end_inclusive: True, label: death}}
""",
    "demographics": f"""
predicates:
  {ADM}
  {DEATH}
patient_demographics:
  male: {{code: GENDER//M}}
trigger: admission
windows:
  input: {{start: NULL, end: trigger, start_inclusive: True, end_inclusive: True, index_timestamp: start}}
  target: {{start: trigger, end: start + 24h, start_inclusive: False, end_inclusive: True, label: death}}
""",
    "static_and_derived": f"""
predicates:
  {ADM}
  {DEATH}
  male_admission: {{expr: "and(male, admission)"}}
patient_demographics:
  male: {{code: GENDER//M}}
trigger: male_admission
windows:
  target: {{start: trigger, end: start + 24h, start_inclusive: False, end_inclusive: True, label: death,
           index_timestamp: start}}
""",
    "static_or_derived": f"""
predicates:
  {ADM}
  {DIS}
  male_or_discharge: {{expr: "or(male, discharge)"}}
patient_demographics:
  male: {{code: GENDER//M}}
trigger: admission
windows:
  input: {{start: trigger, end: start + 12h, start_inclusive: True, end_inclusive: True,
          has: {{male_or_discharge: "(3, None)"}}, index_timestamp: end}}
""",
    "any_event": f"""
predicates:
  {ADM}
trigger: admission
windows:
  input: {{start: NULL, end: trigger, start_inclusive: True, end_inclusive: True,
          has: {{_ANY_EVENT: "(2, None)"}}, index_timestamp: end}}
""",
}

NOISE_CODES = [
    "LAB//other",
    "LAB//spo2",
    "PROCEDURE//START//vent",
    "PROCEDURE//END//vent",
    "PROCEDURE//START//other",
    "ADMISSION//ER",
    "DISCHARGE//HOME",
    "MEDS_DEATH",
]
VALUES = st.one_of(st.none(), st.sampled_from([50.0, 89.9, 90.0, 95.0, 120.0, 130.0]))


@st.composite
def meds_events(draw) -> list[dict]:
    """Subjects with hospital stays (admission, in-stay labs and procedures, discharge or death), static and
    MEDS_BIRTH rows, plus random events anywhere, on an hourly grid so that events often share a timestamp."""
    events = []

    def add(subject_id: int, hour: int | None, code: str, value: float | None = None):
        time = None if hour is None else T0 + timedelta(hours=hour)
        events.append({"subject_id": subject_id, "time": time, "code": code, "numeric_value": value})

    for subject_id in range(1, draw(st.integers(1, 4)) + 1):
        if draw(st.booleans()):
            add(subject_id, None, draw(st.sampled_from(["GENDER//M", "GENDER//F"])))
        if draw(st.booleans()):
            add(subject_id, -draw(st.integers(1, 1000)), "MEDS_BIRTH")
        hour = draw(st.integers(0, 12))
        for _ in range(draw(st.integers(0, 3))):
            length = draw(st.integers(0, 48))
            add(subject_id, hour, "ADMISSION//ER")
            for _ in range(draw(st.integers(0, 4))):
                add(
                    subject_id,
                    hour + draw(st.integers(0, length)),
                    draw(st.sampled_from(NOISE_CODES[:5])),
                    draw(VALUES),
                )
            end = draw(st.sampled_from(["DISCHARGE//HOME", "DISCHARGE//HOME", "MEDS_DEATH", None]))
            if end is not None:
                add(subject_id, hour + length, end)
            hour += length + draw(st.integers(0, 48))
        for _ in range(draw(st.integers(0, 5))):
            add(subject_id, draw(st.integers(0, hour + 24)), draw(st.sampled_from(NOISE_CODES)), draw(VALUES))
    return events


def _run(config_path: Path, shard: Path, prune: bool) -> pl.DataFrame | type[Exception]:
    try:
        cfg = TaskExtractorConfig.load(config_path=config_path)
        predicates_df = get_predicates_df(
            cfg, DictConfig({"path": str(shard), "standard": "meds", "prune": prune})
        )
        result = query(TaskExtractorConfig.load(config_path=config_path), predicates_df)
    except Exception as e:  # The same task must fail the same way either way.
        return type(e)
    sort_by = [c for c in ("subject_id", "trigger") if c in result.columns]
    return result.sort(sort_by) if sort_by else result


def _assert_complete_rows(result: pl.DataFrame | type[Exception], task: str, which: str) -> None:
    """Every returned row is a realized trigger: subject, trigger time, and (when the task defines them) the
    index timestamp and label are set.

    The pruned / unpruned comparison alone cannot see a bug both paths share, such as rows null but for
    ``subject_id`` for windows that never close (#232).
    """
    if not isinstance(result, pl.DataFrame) or result.is_empty():
        return
    for column in ("subject_id", "trigger", "index_timestamp", "label"):
        if column in result.columns:
            nulls = result[column].null_count()
            assert nulls == 0, f"{task} ({which}): {nulls} row(s) with a null {column}\n{result}"


@given(events=meds_events(), task=st.sampled_from(sorted(TASKS)))
@settings(max_examples=300, deadline=None)
def test_pruning_never_changes_results(events: list[dict], task: str):
    with tempfile.TemporaryDirectory() as root:
        shard = Path(root) / "0.parquet"
        pl.DataFrame(
            events,
            schema={
                "subject_id": pl.Int64,
                "time": pl.Datetime("us"),
                "code": pl.String,
                "numeric_value": pl.Float32,
            },
        ).write_parquet(shard)
        config_path = Path(root) / "task.yaml"
        config_path.write_text(TASKS[task])

        expected = _run(config_path, shard, prune=False)
        got = _run(config_path, shard, prune=True)

    if isinstance(expected, pl.DataFrame):
        event(f"{task}: {'some' if expected.height else 'no'} rows")
    else:
        event(f"{task}: {expected.__name__}")

    _assert_complete_rows(expected, task, "without pruning")
    _assert_complete_rows(got, task, "with pruning")

    if isinstance(expected, pl.DataFrame) and isinstance(got, pl.DataFrame):
        assert got.equals(expected), f"{task}: pruned result differs\nwant:\n{expected}\ngot:\n{got}"
    else:
        assert got == expected, f"{task}: without pruning -> {expected}, with pruning -> {got}"
