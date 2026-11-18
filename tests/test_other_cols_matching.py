"""Tests that ``other_cols`` accept the same ``regex`` / ``any`` forms as ``code`` (issues #188, #153)."""

from datetime import datetime

import polars as pl

from aces.config import TaskExtractorConfig
from aces.predicates import generate_plain_predicates_from_meds

TASK_YAML = """
predicates:
  high_systolic_bp:
    code: BP
    text_value: {regex: "^1[4-9][0-9]/"}
  bp_in_mmhg:
    code: BP
    unit: {any: [mmHg, "mm[Hg]"]}
trigger: high_systolic_bp
windows:
  input:
    start: null
    end: trigger
    start_inclusive: True
    end_inclusive: True
    has:
      bp_in_mmhg: (1, None)
"""


def test_regex_and_any_on_other_columns(tmp_path):
    config_path = tmp_path / "task.yaml"
    config_path.write_text(TASK_YAML)
    cfg = TaskExtractorConfig.load(config_path)

    data_path = tmp_path / "data.parquet"
    pl.DataFrame(
        {
            "subject_id": [1, 1, 1],
            "time": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
            "code": ["BP", "BP", "BP"],
            "numeric_value": pl.Series([None, None, None], dtype=pl.Float32),
            "text_value": ["150/90", "120/80", "145/95"],
            "unit": ["mmHg", "mm[Hg]", "kPa"],
        }
    ).write_parquet(data_path)

    predicates_df = generate_plain_predicates_from_meds(data_path, cfg.plain_predicates)

    assert predicates_df["high_systolic_bp"].to_list() == [1, 0, 1]
    assert predicates_df["bp_in_mmhg"].to_list() == [1, 1, 0]
