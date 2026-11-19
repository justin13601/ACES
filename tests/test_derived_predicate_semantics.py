"""Pins the derived predicate behaviour documented in docs/source/configuration.md (issues #148, #191)."""

from datetime import datetime

import polars as pl
from omegaconf import DictConfig

from aces.config import (
    DerivedPredicateConfig,
    EventConfig,
    PlainPredicateConfig,
    TaskExtractorConfig,
    WindowConfig,
)
from aces.predicates import get_predicates_df


def test_derived_predicates_are_evaluated_per_timestamp(tmp_path):
    """At Jan 1, creatinine and two high potassium values are separate rows (measurements) at one time."""
    t1, t2 = datetime(2020, 1, 1), datetime(2020, 1, 2)
    data_path = tmp_path / "data.parquet"
    pl.DataFrame(
        {
            "subject_id": [1] * 6,
            "time": [None, t1, t1, t1, t2, t2],
            "code": [
                "SEX//male",
                "LAB//creatinine",
                "LAB//potassium",
                "LAB//potassium",
                "LAB//creatinine",
                "LAB//potassium",
            ],
            "numeric_value": pl.Series([None, 1.0, 6.0, 6.5, 1.0, 4.0], dtype=pl.Float32),
        }
    ).write_parquet(data_path)

    predicates = {
        "creatinine": PlainPredicateConfig("LAB//creatinine"),
        "high_potassium": PlainPredicateConfig("LAB//potassium", value_min=5.5),
        "male": PlainPredicateConfig("SEX//male", static=True),
        "creatinine_and_high_potassium": DerivedPredicateConfig("and(creatinine, high_potassium)"),
        "male_high_potassium": DerivedPredicateConfig("and(male, high_potassium)"),
        "nested": DerivedPredicateConfig("or(creatinine_and_high_potassium, male_high_potassium)"),
    }
    derived = ["creatinine_and_high_potassium", "male_high_potassium", "nested"]
    cfg = TaskExtractorConfig(
        predicates=predicates,
        trigger=EventConfig("creatinine"),
        windows={
            "input": WindowConfig(None, "trigger", True, True, has=dict.fromkeys(derived, "(None, None)"))
        },
    )

    out = get_predicates_df(cfg, DictConfig({"path": str(data_path), "standard": "meds"}))
    out = out.drop_nulls("timestamp").sort("timestamp")

    # Plain predicates count every matching row at a timestamp...
    assert out["high_potassium"].to_list() == [2, 0]
    # ...while `and` only needs each input to match some row at that timestamp, and counts at most once.
    assert out["creatinine_and_high_potassium"].to_list() == [1, 0]
    # Static predicates apply to all of the subject's events, and derived predicates can be nested.
    assert out["male_high_potassium"].to_list() == [1, 0]
    assert out["nested"].to_list() == [1, 0]
