"""Tests for ``code: {descendants_of: [...]}`` predicates, expanded through MEDS ``parent_codes``."""

from datetime import datetime

import polars as pl
import pytest
from omegaconf import DictConfig

from aces.config import TaskExtractorConfig
from aces.predicates import get_predicates_df
from aces.query import query

TASK_YAML = """
predicates:
  admission:
    code: ADMISSION
  mi:
    code: {descendants_of: ["ICD10//I21"]}
trigger: admission
windows:
  input:
    start: null
    end: trigger
    start_inclusive: True
    end_inclusive: True
    has:
      mi: (1, None)
"""


@pytest.fixture()
def meds_root(tmp_path):
    """A MEDS dataset where each subject has one diagnosis before an admission.

    In ``codes.parquet``, ``ICD10//I21.01`` is a grandchild of ``ICD10//I21`` and ``ICD9//410`` is linked to
    it as a child (a cross-vocabulary link); ``ICD10//I50`` is unrelated.
    """
    diagnoses = {1: "ICD10//I21.01", 2: "ICD9//410", 3: "ICD10//I50", 4: "ICD10//I21"}
    data = pl.DataFrame(
        {
            "subject_id": [s for s in diagnoses for _ in range(2)],
            "time": [datetime(2020, 1, 1), datetime(2020, 2, 1)] * len(diagnoses),
            "code": [c for dx in diagnoses.values() for c in (dx, "ADMISSION")],
            "numeric_value": pl.Series([None] * 2 * len(diagnoses), dtype=pl.Float32),
        }
    )
    codes = pl.DataFrame(
        {
            "code": ["ICD10//I21", "ICD10//I21.0", "ICD10//I21.01", "ICD9//410", "ICD10//I50", "ADMISSION"],
            "description": [None] * 6,
            "parent_codes": [None, ["ICD10//I21"], ["ICD10//I21.0"], ["ICD10//I21"], None, None],
        },
        schema_overrides={"description": pl.String, "parent_codes": pl.List(pl.String)},
    )
    (tmp_path / "data" / "train").mkdir(parents=True)
    (tmp_path / "metadata").mkdir()
    data.write_parquet(tmp_path / "data" / "train" / "0.parquet")
    codes.write_parquet(tmp_path / "metadata" / "codes.parquet")
    return tmp_path


def test_descendants_of_matches_codes_and_their_descendants(tmp_path, meds_root):
    config_path = tmp_path / "task.yaml"
    config_path.write_text(TASK_YAML)
    cfg = TaskExtractorConfig.load(config_path)

    data_config = DictConfig({"path": str(meds_root / "data" / "train" / "0.parquet"), "standard": "meds"})
    result = query(cfg, get_predicates_df(cfg, data_config))

    assert sorted(result["subject_id"].to_list()) == [1, 2, 4]


def test_descendants_of_needs_code_metadata(tmp_path, meds_root):
    config_path = tmp_path / "task.yaml"
    config_path.write_text(TASK_YAML)
    cfg = TaskExtractorConfig.load(config_path)

    (meds_root / "metadata" / "codes.parquet").unlink()
    data_config = DictConfig({"path": str(meds_root / "data" / "train" / "0.parquet"), "standard": "meds"})
    with pytest.raises(FileNotFoundError, match="descendants_of"):
        get_predicates_df(cfg, data_config)


def test_descendants_of_with_relative_data_path(tmp_path, meds_root, monkeypatch):
    config_path = tmp_path / "task.yaml"
    config_path.write_text(TASK_YAML)
    cfg = TaskExtractorConfig.load(config_path)

    monkeypatch.chdir(meds_root / "data" / "train")
    result = query(cfg, get_predicates_df(cfg, DictConfig({"path": "0.parquet", "standard": "meds"})))

    assert sorted(result["subject_id"].to_list()) == [1, 2, 4]
