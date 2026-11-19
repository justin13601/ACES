"""Event-bound windows whose bounding event never occurs (#232).

A window like ``end: start -> discharge_or_death`` cannot close for a stay with no later discharge or death
(the record ends during the stay, or the outcome is dated before the trigger). Such a window has no
realization, so its trigger must yield no row. It used to yield a row null in every column but
``subject_id``: the null child anchor went through the recursion and the left joins of ``extract_subtree``.
"""

from datetime import datetime, timedelta

import polars as pl
import pytest
from bigtree import Node
from omegaconf import DictConfig

from aces.config import TaskExtractorConfig
from aces.extract_subtree import extract_subtree
from aces.predicates import get_predicates_df
from aces.query import query
from aces.types import ToEventWindowBounds

T0 = datetime(2020, 1, 1)


def at(hours: float) -> datetime:
    return T0 + timedelta(hours=hours)


# ---------------------------------------------------------------------------------------------------------
# extract_subtree on a hand-built tree


def tree(end_event: str) -> Node:
    """Trigger -> one event-bound window ending at the next (``is_B``) or previous (``-is_B``) B event."""
    root = Node("trigger")
    child = Node("window")
    child.endpoint_expr = ToEventWindowBounds(
        left_inclusive=False, end_event=end_event, right_inclusive=True, offset=None
    )
    child.constraints = {}
    child.parent = root
    return root


def predicates_df(rows: list[tuple[int, float, int, int]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "subject_id": [r[0] for r in rows],
            "timestamp": [at(r[1]) for r in rows],
            "is_A": [r[2] for r in rows],
            "is_B": [r[3] for r in rows],
        }
    )


def anchors(subjects: list[int]) -> pl.DataFrame:
    return pl.DataFrame({"subject_id": subjects, "subtree_anchor_timestamp": [at(0)] * len(subjects)})


@pytest.mark.parametrize(
    "end_event, closing, never",
    [
        # forward: subject 1 has a B after the anchor; subject 2's only B is before it.
        ("is_B", (1, 5.0, 0, 1), (2, -5.0, 0, 1)),
        # backward: subject 1 has a B before the anchor; subject 2's only B is after it.
        ("-is_B", (1, -5.0, 0, 1), (2, 5.0, 0, 1)),
    ],
    ids=["forward", "backward"],
)
def test_a_window_whose_event_never_occurs_has_no_realization(end_event, closing, never):
    df = predicates_df([(1, 0.0, 1, 0), closing, (2, 0.0, 1, 0), never]).sort("subject_id", "timestamp")
    got = extract_subtree(tree(end_event), anchors([1, 2]), df)

    assert got["subject_id"].to_list() == [1]
    assert got["subtree_anchor_timestamp"].null_count() == 0
    assert got["window_summary"].struct.field("timestamp_at_end").null_count() == 0


def test_no_subject_with_an_unclosed_window_gives_an_empty_result():
    df = predicates_df([(1, 0.0, 1, 0), (1, 3.0, 0, 0)])  # never a B event
    got = extract_subtree(tree("is_B"), anchors([1]), df)
    assert got.is_empty()


# ---------------------------------------------------------------------------------------------------------
# query, end to end on MEDS data

ICU_MORTALITY = """
predicates:
  icu_admission:
    code: {regex: "^ICU_ADMISSION//"}
  icu_discharge:
    code: {regex: "^ICU_DISCHARGE//"}
  death:
    code: MEDS_DEATH
  icu_discharge_or_death:
    expr: or(icu_discharge, death)
trigger: icu_admission
windows:
  input:
    start: trigger
    end: start + 24h
    start_inclusive: True
    end_inclusive: True
    has:
      icu_discharge: (None, 0)
      death: (None, 0)
    index_timestamp: end
  target:
    start: input.end
    end: start -> icu_discharge_or_death
    start_inclusive: False
    end_inclusive: True
    label: death
"""

STAYS = {
    1: [(0, "ICU_ADMISSION//X"), (72, "ICU_DISCHARGE//X")],  # discharged: label 0
    2: [(0, "ICU_ADMISSION//X"), (48, "MEDS_DEATH")],  # dies: label 1
    3: [(0, "ICU_ADMISSION//X"), (30, "LAB//1")],  # still open when the record ends
    4: [(-8, "ICU_DISCHARGE//X"), (-8, "MEDS_DEATH"), (0, "ICU_ADMISSION//X")],  # outcome dated before
}


@pytest.mark.parametrize("with_static_rows", [False, True])
def test_query_leaves_out_triggers_whose_window_never_closes(tmp_path, with_static_rows):
    rows = [(s, at(h), code) for s, events in STAYS.items() for h, code in events]
    if with_static_rows:
        rows += [(s, None, "GENDER//F") for s in STAYS]
    data = pl.DataFrame(
        rows, schema={"subject_id": pl.Int64, "time": pl.Datetime("us"), "code": pl.String}, orient="row"
    ).with_columns(pl.lit(None, pl.Float32).alias("numeric_value"))
    data.write_parquet(tmp_path / "0.parquet")
    (tmp_path / "task.yaml").write_text(ICU_MORTALITY)

    cfg = TaskExtractorConfig.load(config_path=tmp_path / "task.yaml")
    data_config = DictConfig({"path": str(tmp_path / "0.parquet"), "standard": "meds"})
    result = query(cfg, get_predicates_df(cfg, data_config))

    assert result.select("subject_id", "label").sort("subject_id").rows() == [(1, 0), (2, 1)]
    for column in ("subject_id", "trigger", "index_timestamp"):
        assert result[column].null_count() == 0, column
