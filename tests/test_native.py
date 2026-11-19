"""PROTOTYPE (not ready to merge): aces.native.plain_predicates returns exactly what
aces.predicates.generate_plain_predicates_from_meds returns."""

import math
from pathlib import Path

import polars as pl
import pytest
from aces.config import PlainPredicateConfig

from aces import config, native, predicates

ROOT = Path(__file__).resolve().parents[1]
SHARDS = sorted((ROOT / "sample_data" / "meds_sample").rglob("*.parquet"))
CONFIGS = sorted((ROOT / "sample_configs").glob("*.yaml"))

pytestmark = pytest.mark.skipif(not native.available(), reason="aces._native is not built")


def same(path, plain, prune):
    want = predicates.generate_plain_predicates_from_meds(path, plain, prune=prune)
    got = native.plain_predicates(path, plain, prune=prune)
    assert got.equals(want), (path, prune, got, want)


@pytest.mark.parametrize("prune", [False, True])
@pytest.mark.parametrize("cfg_path", CONFIGS, ids=lambda p: p.stem)
def test_sample_configs_match_aces(cfg_path, prune):
    plain = config.TaskExtractorConfig.load(config_path=cfg_path).plain_predicates
    for shard in SHARDS:
        same(shard, plain, prune)


def edge_case_shard(tmp_path) -> Path:
    """Unsorted and duplicated rows, null and MEDS_BIRTH codes, static rows, null/NaN/inf values, and a value
    whose float32 rounding decides a bound (5.1 as float32 is just below 5.1)."""
    rows = [
        (2, "2020-01-02 00:00", "LAB//A", 5.1),
        (1, None, "GENDER//M", None),
        (1, "2020-01-01 00:00", "MEDS_BIRTH", None),
        (1, "2020-01-03 00:00", "LAB//A", math.nan),
        (1, "2020-01-02 00:00", "LAB//A", 7.0),
        (2, "2020-01-01 00:00", None, 1.0),
        (1, "2020-01-02 00:00", "LAB//B", math.inf),
        (2, "2020-01-02 00:00", "ADMIT//X", None),
        (1, "2020-01-03 00:00", "LAB//A", 3.0),
        (3, "2020-01-05 00:00", "NOTHING", None),
        (2, None, "GENDER//F", None),
        (1, "2020-01-02 00:00", "LAB//A", None),
        (3, "2020-01-04 00:00", "ADMIT//Y", 2.0),
        (2, "2020-01-03 00:00", "LAB//A", 5.0),
    ]
    frame = pl.DataFrame(
        rows,
        schema={"subject_id": pl.Int64, "time": pl.String, "code": pl.String, "numeric_value": pl.Float32},
        orient="row",
    ).with_columns(pl.col("time").str.to_datetime("%Y-%m-%d %H:%M", time_unit="us"))
    path = tmp_path / "0.parquet"
    frame.write_parquet(path, row_group_size=4)  # Several record batches, with different dictionaries.
    return path


EDGE_PREDICATES = {
    "lab": PlainPredicateConfig("LAB//A"),
    "lab_ge_5_1": PlainPredicateConfig("LAB//A", value_min=5.1, value_min_inclusive=True),
    "lab_gt_5": PlainPredicateConfig("LAB//A", value_min=5, value_min_inclusive=False),
    "lab_lt_6": PlainPredicateConfig("LAB//A", value_max=6, value_max_inclusive=False),
    "lab_any_le_inf": PlainPredicateConfig(
        code={"regex": "^LAB//"}, value_max=math.inf, value_max_inclusive=True
    ),
    "admit": PlainPredicateConfig(code={"regex": "^ADMIT//"}),
    "male_or_b": PlainPredicateConfig(code={"any": ["GENDER//M", "LAB//B"]}),
}


@pytest.mark.parametrize("prune", [False, True])
def test_edge_cases_match_aces(tmp_path, prune):
    same(edge_case_shard(tmp_path), EDGE_PREDICATES, prune)


def test_other_cols_falls_back_to_aces(tmp_path):
    path = tmp_path / "0.parquet"
    pl.DataFrame(
        {
            "subject_id": [1, 1],
            "time": [None, None],
            "code": ["A", "A"],
            "numeric_value": [1.0, 2.0],
            "unit": ["x", "y"],
        },
        schema_overrides={"time": pl.Datetime("us")},
    ).write_parquet(path)
    plain = {"a_x": PlainPredicateConfig("A", other_cols={"unit": "x"})}
    assert not native.supported(plain)
    same(path, plain, False)


def test_shards_on_threads_match_aces():
    plain = config.TaskExtractorConfig.load(config_path=CONFIGS[0]).plain_predicates
    got = native.plain_predicates_many(SHARDS, plain, prune=True, workers=4)
    for shard, frame in zip(SHARDS, got, strict=True):
        assert frame.equals(predicates.generate_plain_predicates_from_meds(shard, plain, prune=True))
