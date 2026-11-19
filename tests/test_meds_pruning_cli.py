"""End to end: ``aces-cli`` on MEDS data writes the same cohort with pruning (the default) and without it.

The existing MEDS CLI tests all count ``_ANY_EVENT``, which turns pruning off, so these tasks do not.
"""

import tempfile
from pathlib import Path

import polars as pl
import pytest

from .test_other_meds import MEDS_SHARDS
from .utils import assert_df_equal, needs_working_cli, run_command

TASKS = {
    "inhospital_mortality": """
        predicates:
          admission:
            code: {regex: ADMISSION//.*}
          discharge:
            code: {regex: DISCHARGE//.*}
          death:
            code: DEATH
          discharge_or_death:
            expr: or(discharge, death)
        trigger: admission
        windows:
          input:
            start: NULL
            end: trigger + 24h
            start_inclusive: True
            end_inclusive: True
            index_timestamp: end
          gap:
            start: trigger
            end: start + 48h
            start_inclusive: False
            end_inclusive: True
            has:
              admission: (None, 0)
              discharge_or_death: (None, 0)
          target:
            start: gap.end
            end: start -> discharge_or_death
            start_inclusive: False
            end_inclusive: True
            label: death
        """,
    "readmission_30d": """
        predicates:
          admission:
            code: {regex: ADMISSION//.*}
          discharge:
            code: {regex: DISCHARGE//.*}
        trigger: discharge
        windows:
          stay:
            start: end <- admission
            end: trigger
            start_inclusive: True
            end_inclusive: True
            index_timestamp: end
          target:
            start: stay.end
            end: start + 30d
            start_inclusive: False
            end_inclusive: True
            label: admission
        """,
    "record_bounds": """
        predicates:
          admission:
            code: {regex: ADMISSION//.*}
          death:
            code: DEATH
        trigger: admission
        windows:
          history:
            start: NULL
            end: trigger
            start_inclusive: True
            end_inclusive: True
            index_timestamp: start
          target:
            start: trigger
            end: start + 365d
            start_inclusive: False
            end_inclusive: True
            has:
              _RECORD_END: (None, 0)
            label: death
        """,
}


def _labels(rows: list[tuple[int, str, bool]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={"subject_id": pl.Int64, "prediction_time": pl.String, "boolean_value": pl.Boolean},
        orient="row",
    ).with_columns(pl.col("prediction_time").str.strptime(pl.Datetime("us"), "%Y-%m-%d %H:%M"))


# The expected cohorts, checked by hand against MEDS_SHARDS. In record_bounds, the prediction time is the
# record start, which is the MEDS_BIRTH row (subject 1 born 1960-12-18 11:03, subject 3 born 1982-02-28).
WANT_LABELS = {
    "inhospital_mortality": _labels(
        [
            (1, "2020-01-15 15:14", False),
            (1, "2022-01-19 04:46", False),
            (1, "2022-01-25 08:11", True),
            (3, "2020-01-15 15:14", False),
            (3, "2024-03-19 17:11", False),
            (3, "2024-03-30 11:00", False),
        ]
    ),
    "readmission_30d": _labels(
        [
            (1, "2020-01-17 10:00", False),
            (1, "2022-01-20 08:00", True),
            (3, "2020-01-17 10:00", True),
            (3, "2020-01-20 15:18", False),
            (3, "2024-03-28 10:00", True),
            (3, "2024-04-19 13:32", False),
        ]
    ),
    "record_bounds": _labels(
        [
            (1, "1960-12-18 11:03", False),
            (3, "1982-02-28 00:00", False),
            (3, "1982-02-28 00:00", False),
        ]
    ),
}


PRUNING_LOG = "Dropping events that match no predicate"


def _run_cli(root: Path, data_fp: Path, task: str, prune: bool) -> pl.DataFrame:
    cohort_dir = root / f"cohort_prune_{prune}"
    cohort_dir.mkdir()
    (cohort_dir / f"{task}.yaml").write_text(TASKS[task])
    stderr, stdout = run_command(
        "aces-cli",
        {
            "cohort_dir": str(cohort_dir.resolve()),
            "cohort_name": task,
            "data.standard": "meds",
            "data.path": str(data_fp.resolve()),
            "data.prune": str(prune).lower(),
        },
        f"CLI should run for {task} with prune={prune}",
    )
    # The CLI must actually honour data.prune, or both runs would take the same path.
    assert (PRUNING_LOG in stderr + stdout) == prune, f"{task}: pruning should be {'on' if prune else 'off'}"
    return pl.read_parquet(cohort_dir / f"{task}.parquet")


@needs_working_cli
@pytest.mark.parametrize("task", sorted(TASKS))
def test_cli_cohort_is_the_same_with_and_without_pruning(task):
    with tempfile.TemporaryDirectory() as root:
        root = Path(root)
        data_fp = root / "data" / "data.parquet"
        data_fp.parent.mkdir()
        pl.concat(MEDS_SHARDS.values()).write_parquet(data_fp)

        want = _run_cli(root, data_fp, task, prune=False)
        got = _run_cli(root, data_fp, task, prune=True)

    assert_df_equal(want, got, f"{task}: cohort differs with pruning")
    assert_df_equal(
        WANT_LABELS[task],
        got.select("subject_id", "prediction_time", "boolean_value"),
        f"{task}: unexpected cohort",
    )
