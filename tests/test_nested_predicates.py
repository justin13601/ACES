"""Tests for nested predicates, which embed a ``trigger`` and ``windows`` in a predicate (issue #43)."""

from datetime import datetime

import polars as pl
import pytest
from omegaconf import DictConfig

from aces.config import TaskExtractorConfig
from aces.predicates import get_predicates_df
from aces.query import query

PREDICATES_YAML = """
predicates:
  admission:
    code: ADMISSION
  discharge:
    code: DISCHARGE
  death:
    code: DEATH
  heart_failure:
    code: {regex: "^ICD10//I50"}
  hf_hospitalization:
    trigger: admission
    windows:
      stay:
        start: trigger
        end: start -> discharge
        start_inclusive: True
        end_inclusive: True
        index_timestamp: start
        has:
          heart_failure: (1, None)
"""

HF_READMISSION_YAML = (
    PREDICATES_YAML
    + """
trigger: discharge
windows:
  target:
    start: trigger
    end: start + 30d
    start_inclusive: False
    end_inclusive: True
    index_timestamp: start
    label: hf_hospitalization
"""
)


def day(d: int, hour: int = 0) -> datetime:
    return datetime(2020, 1, d, hour)


@pytest.fixture()
def meds_data_config(tmp_path) -> DictConfig:
    """Subject 1 is readmitted for heart failure after a heart failure stay; subject 2 is readmitted for
    pneumonia; subject 3 is diagnosed with heart failure after discharge, but outside of any
    hospitalization."""
    rows = [
        (1, day(1), "ADMISSION"),
        (1, day(2), "ICD10//I50.9"),
        (1, day(4), "DISCHARGE"),
        (1, day(20), "ADMISSION"),
        (1, day(21, 12), "ICD10//I50.1"),
        (1, day(23), "DISCHARGE"),
        (2, day(1), "ADMISSION"),
        (2, day(3), "DISCHARGE"),
        (2, day(10), "ADMISSION"),
        (2, day(11), "ICD10//J18.9"),
        (2, day(12), "DISCHARGE"),
        (3, day(1), "ADMISSION"),
        (3, day(2), "DISCHARGE"),
        (3, day(10), "ICD10//I50.9"),
        (3, day(30), "DEATH"),
    ]
    data_path = tmp_path / "data.parquet"
    pl.DataFrame(
        {
            "subject_id": [s for s, _, _ in rows],
            "time": [t for _, t, _ in rows],
            "code": [c for _, _, c in rows],
            "numeric_value": pl.Series([None] * len(rows), dtype=pl.Float32),
        }
    ).write_parquet(data_path)
    return DictConfig({"path": str(data_path), "standard": "meds"})


def load(tmp_path, task_yaml: str) -> TaskExtractorConfig:
    config_path = tmp_path / "task.yaml"
    config_path.write_text(task_yaml)
    return TaskExtractorConfig.load(config_path)


def test_hf_readmission(tmp_path, meds_data_config):
    cfg = load(tmp_path, HF_READMISSION_YAML)
    predicates_df = get_predicates_df(cfg, meds_data_config)

    # The nested predicate occurs at the admission of each stay in which heart failure is diagnosed, even if
    # (as in the second stay of subject 1) the diagnosis comes after the admission.
    hf_hospitalizations = predicates_df.filter(pl.col("hf_hospitalization") > 0)
    assert hf_hospitalizations.select("subject_id", "timestamp").rows() == [(1, day(1)), (1, day(20))]

    result = query(cfg, predicates_df).sort("subject_id", "trigger")
    assert result.select("subject_id", "trigger", "label").rows() == [
        (1, day(4), 1),
        (1, day(23), 0),
        (2, day(3), 0),
        (2, day(12), 0),
        (3, day(2), 0),
    ]


def test_nested_predicates_can_use_and_be_used_by_other_predicates(tmp_path, meds_data_config):
    """A nested predicate can use another nested predicate and be used by a derived predicate."""
    task_yaml = (
        PREDICATES_YAML
        + """
  discharge_before_hf_readmission:
    trigger: discharge
    windows:
      target:
        start: trigger
        end: start + 30d
        start_inclusive: False
        end_inclusive: True
        has:
          hf_hospitalization: (1, None)
  hf_readmission_or_death:
    expr: or(discharge_before_hf_readmission, death)

trigger: hf_readmission_or_death
"""
    )
    cfg = load(tmp_path, task_yaml)
    result = query(cfg, get_predicates_df(cfg, meds_data_config))

    assert result.select("subject_id", "trigger").sort("subject_id").rows() == [(1, day(4)), (3, day(30))]


def test_nested_predicate_new_events_are_empty(tmp_path, meds_data_config):
    """A nested predicate indexed between events adds a new event, which does not count as any other event."""
    task_yaml = """
predicates:
  admission:
    code: ADMISSION
  half_day_after_admission:
    trigger: admission
    windows:
      first_half_day:
        start: trigger
        end: start + 12h
        start_inclusive: True
        end_inclusive: True
        index_timestamp: end

trigger: half_day_after_admission
windows:
  input:
    start: null
    end: trigger
    start_inclusive: True
    end_inclusive: True
    index_timestamp: end
    has:
      _ANY_EVENT: (1, None)
"""
    cfg = load(tmp_path, task_yaml)
    result = query(cfg, get_predicates_df(cfg, meds_data_config)).sort("subject_id", "trigger")

    assert result.select("subject_id", "trigger").rows() == [
        (1, day(1, 12)),
        (1, day(20, 12)),
        (2, day(1, 12)),
        (2, day(10, 12)),
        (3, day(1, 12)),
    ]
    # E.g., by noon on day 20, subject 1 has 4 real events; the new event at noon on day 1 is not counted.
    assert result["input.start_summary"].struct.field("_ANY_EVENT").to_list() == [1, 4, 1, 3, 1]


@pytest.mark.parametrize(
    "nested_predicate_yaml, error, match",
    [
        pytest.param(
            "hf_hospitalization:\n    trigger: hf_hospitalization\n",
            ValueError,
            "Predicate graph is not a directed acyclic graph",
            id="cycle",
        ),
        pytest.param(
            "hf_hospitalization:\n    trigger: transfer\n",
            KeyError,
            "Predicate 'transfer' referenced in 'hf_hospitalization' is not defined",
            id="undefined_predicate",
        ),
        pytest.param(
            "hf_hospitalization:\n    trigger: admission\n    code: ADMISSION\n",
            ValueError,
            "Nested predicate 'hf_hospitalization' can only have 'trigger' and 'windows' fields",
            id="mixed_with_plain_predicate",
        ),
    ],
)
def test_invalid_nested_predicates(tmp_path, nested_predicate_yaml, error, match):
    task_yaml = PREDICATES_YAML.split("  hf_hospitalization:")[0] + "  " + nested_predicate_yaml
    task_yaml += "trigger: hf_hospitalization\n"

    with pytest.raises(error, match=match):
        load(tmp_path, task_yaml)
