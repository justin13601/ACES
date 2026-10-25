"""Tests for keeping one valid prediction time per subject via ``index_selection``."""

from datetime import datetime

import polars as pl
import pytest

from aces.config import (
    EventConfig,
    IndexSelectionConfig,
    PlainPredicateConfig,
    TaskExtractorConfig,
    WindowConfig,
)
from aces.query import query


def make_config(index_selection: IndexSelectionConfig | None) -> TaskExtractorConfig:
    """Predicts at each discharge that has a prior diagnosis."""
    return TaskExtractorConfig(
        predicates={"discharge": PlainPredicateConfig("discharge"), "dx": PlainPredicateConfig("dx")},
        trigger=EventConfig("discharge"),
        windows={
            "input": WindowConfig(
                start=None,
                end="trigger",
                start_inclusive=True,
                end_inclusive=True,
                has={"dx": "(1, None)"},
                index_timestamp="end",
            ),
        },
        index_selection=index_selection,
    )


@pytest.fixture()
def predicates_df():
    """Subject 1 has three discharges; the first has no prior diagnosis, so only the later two are valid.

    Subject 2 has a single valid discharge.
    """
    return pl.DataFrame(
        {
            "subject_id": [1, 1, 1, 1, 2, 2],
            "timestamp": [
                datetime(2020, 1, 1),
                datetime(2020, 2, 1),
                datetime(2020, 3, 1),
                datetime(2020, 4, 1),
                datetime(2020, 1, 1),
                datetime(2020, 1, 2),
            ],
            "discharge": [1, 0, 1, 1, 0, 1],
            "dx": [0, 1, 0, 0, 1, 0],
        }
    )


def triggers(result: pl.DataFrame) -> dict[int, list[datetime]]:
    return {s: g["trigger"].to_list() for (s,), g in result.sort("trigger").group_by("subject_id")}


def test_default_keeps_all_valid_rows(predicates_df):
    assert triggers(query(make_config(None), predicates_df)) == {
        1: [datetime(2020, 3, 1), datetime(2020, 4, 1)],
        2: [datetime(2020, 1, 2)],
    }


def test_first_keeps_first_valid_row(predicates_df):
    """The first discharge of subject 1 is not valid, so the second one is kept rather than none."""
    assert triggers(query(make_config(IndexSelectionConfig("first")), predicates_df)) == {
        1: [datetime(2020, 3, 1)],
        2: [datetime(2020, 1, 2)],
    }


def test_last_keeps_last_valid_row(predicates_df):
    assert triggers(query(make_config(IndexSelectionConfig("last")), predicates_df)) == {
        1: [datetime(2020, 4, 1)],
        2: [datetime(2020, 1, 2)],
    }


def test_random_keeps_one_valid_row_independent_of_other_subjects(predicates_df):
    cfg = make_config(IndexSelectionConfig("random", seed=3))
    picked = triggers(query(cfg, predicates_df))

    assert picked[2] == [datetime(2020, 1, 2)]
    assert len(picked[1]) == 1 and picked[1][0] in {datetime(2020, 3, 1), datetime(2020, 4, 1)}

    # Same pick when subject 1 is queried on its own (e.g. in a different shard) and when queried again.
    assert triggers(query(cfg, predicates_df.filter(pl.col("subject_id") == 1)))[1] == picked[1]
    assert triggers(query(cfg, predicates_df)) == picked


def test_random_depends_on_seed(predicates_df):
    picks = {
        triggers(query(make_config(IndexSelectionConfig("random", seed=seed)), predicates_df))[1][0]
        for seed in range(20)
    }
    assert picks == {datetime(2020, 3, 1), datetime(2020, 4, 1)}


@pytest.mark.parametrize(
    ("yaml_value", "expected"),
    [
        ("first", IndexSelectionConfig("first")),
        ("{strategy: random, seed: 7}", IndexSelectionConfig("random", seed=7)),
    ],
)
def test_load_from_yaml(tmp_path, yaml_value, expected):
    config_path = tmp_path / "task.yaml"
    config_path.write_text(
        "predicates:\n"
        "  discharge:\n"
        "    code: discharge\n"
        "trigger: discharge\n"
        "windows:\n"
        "  input:\n"
        "    start: null\n"
        "    end: trigger\n"
        "    start_inclusive: True\n"
        "    end_inclusive: True\n"
        f"index_selection: {yaml_value}\n"
    )
    assert TaskExtractorConfig.load(config_path).index_selection == expected
